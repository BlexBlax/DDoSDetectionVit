import os
import threading
import time
from time import time as now_time

import requests
from flask import Flask, Response, jsonify, request

from detector import evaluate_request, block_ip, redis_client

app = Flask(__name__)

GATEWAY_ID = os.environ.get("GATEWAY_ID", "1")
PORT = int(os.environ.get("PORT", 5002))
BACKEND_URL = os.environ.get("BACKEND_URL", "http://netguard:5001").rstrip("/")

# Local gateway counters
gw_stats = {
    "requests": 0,
    "tarpit_hits": 0,
    "block_hits": 0
}
stats_lock = threading.Lock()
last_reset_seen = 0.0

# ---------------------------------------------------------------------------
# Tarpit limits. A tarpitted request parks a worker thread in time.sleep(), so
# without caps a big attack could fill every thread and starve legitimate
# traffic (the defence itself becoming the DoS).
#   MAX_CONCURRENT_TARPITS         - sleeping requests allowed across this gateway
#   MAX_CONCURRENT_TARPITS_PER_IP  - sleeping requests allowed for one IP; an IP
#                                    that exceeds it is clearly ignoring the
#                                    tarpit, so it gets a hard block instead
# ---------------------------------------------------------------------------
MAX_CONCURRENT_TARPITS = int(os.environ.get("MAX_CONCURRENT_TARPITS", 50))
MAX_CONCURRENT_TARPITS_PER_IP = int(os.environ.get("MAX_CONCURRENT_TARPITS_PER_IP", 10))
SHED_RETRY_SECONDS = 2
BLOCK_SECONDS_FALLBACK = 60   # matches detector.BLOCK_SECONDS
_tarpit_slots = threading.BoundedSemaphore(MAX_CONCURRENT_TARPITS)
_tarpits_by_ip = {}   # ip -> sleeping requests right now (guarded by stats_lock)
gw_stats["shed_hits"] = 0


def _try_enter_tarpit(ip):
    """Reserve a tarpit slot. Returns 'ok', 'ip_limit' or 'full'."""
    with stats_lock:
        if _tarpits_by_ip.get(ip, 0) >= MAX_CONCURRENT_TARPITS_PER_IP:
            return "ip_limit"
        _tarpits_by_ip[ip] = _tarpits_by_ip.get(ip, 0) + 1
    if not _tarpit_slots.acquire(blocking=False):
        _leave_ip(ip)
        return "full"
    return "ok"


def _leave_ip(ip):
    with stats_lock:
        n = _tarpits_by_ip.get(ip, 0) - 1
        if n > 0:
            _tarpits_by_ip[ip] = n
        else:
            _tarpits_by_ip.pop(ip, None)


def _leave_tarpit(ip):
    _tarpit_slots.release()
    _leave_ip(ip)


def _blocked_response(seconds):
    resp = jsonify({
        "error": "Access Denied: Temporarily blocked due to excessive traffic.",
        "gateway": f"Gateway-{GATEWAY_ID}",
        "retry_after_seconds": int(seconds)
    })
    resp.status_code = 429
    resp.headers["Retry-After"] = str(int(seconds))
    resp.headers["X-Defense-Action"] = "Blocked"
    resp.headers["X-Handled-By"] = f"Gateway-{GATEWAY_ID}"
    return resp


def _check_reset_signal():
    """Reset gateway counters if cluster broadcasted a reset signal."""
    global last_reset_seen
    if redis_client:
        try:
            sig = redis_client.get("signal:reset")
            if sig and float(sig) > last_reset_seen:
                last_reset_seen = float(sig)
                with stats_lock:
                    gw_stats["requests"] = 0
                    gw_stats["tarpit_hits"] = 0
                    gw_stats["block_hits"] = 0
                    gw_stats["shed_hits"] = 0
        except Exception:
            pass


def _heartbeat_worker():
    """Periodically publish gateway health and telemetry to Redis."""
    while True:
        try:
            _check_reset_signal()
            if redis_client:
                with stats_lock:
                    r = gw_stats["requests"]
                    t = gw_stats["tarpit_hits"]
                    b = gw_stats["block_hits"]
                    sh = gw_stats["shed_hits"]
                redis_client.hset(
                    f"gateway:{GATEWAY_ID}",
                    mapping={
                        "id": f"Gateway-{GATEWAY_ID}",
                        "port": str(PORT),
                        "requests": str(r),
                        "tarpit_hits": str(t),
                        "block_hits": str(b),
                        "shed_hits": str(sh),
                        "timestamp": str(now_time())
                    }
                )
                redis_client.expire(f"gateway:{GATEWAY_ID}", 30)
        except Exception as e:
            pass
        time.sleep(1.5)


# Start background heartbeat
hb_thread = threading.Thread(target=_heartbeat_worker, daemon=True)
hb_thread.start()


def get_real_client_ip():
    """Extract real client IP forwarded from Nginx / Edge Proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.headers.get("X-Real-IP") or request.remote_addr or "127.0.0.1"


@app.route("/_health")
def health():
    redis_ok = False
    if redis_client:
        try:
            redis_ok = bool(redis_client.ping())
        except Exception:
            pass
    return jsonify({
        "status": "healthy",
        "redis_reachable": redis_ok,
        "active_tarpits": sum(_tarpits_by_ip.values()),
        "max_tarpits": MAX_CONCURRENT_TARPITS,
        "gateway": f"Gateway-{GATEWAY_ID}",
        "port": PORT,
        "backend": BACKEND_URL
    })


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
def proxy_handler(path):
    _check_reset_signal()
    client_ip = get_real_client_ip()
    user_agent = request.headers.get("User-Agent", "")
    header_size = sum(len(k) + len(v) + 4 for k, v in request.headers.items()) or 350
    payload_size = request.content_length or len(request.get_data()) or 0
    total_pkt_size = header_size + payload_size

    # Run rate limiter & ML flow check
    action, value = evaluate_request(
        client_ip,
        request.path,
        request.method,
        user_agent=user_agent,
        gateway_id=f"Gateway-{GATEWAY_ID}",
        size=total_pkt_size,
        header_len=header_size
    )

    with stats_lock:
        gw_stats["requests"] += 1

    # 1. Blocked requests get an immediate 429
    if action == "BLOCK":
        with stats_lock:
            gw_stats["block_hits"] += 1
        return _blocked_response(value)

    # 2. Tarpit: pause response to slow down attacker threads (capped)
    tarpitted = False
    if action == "TARPIT":
        with stats_lock:
            gw_stats["tarpit_hits"] += 1
        slot = _try_enter_tarpit(client_ip)

        if slot == "ip_limit":
            # One IP holding too many sleeping connections is ignoring the tarpit.
            block_ip(client_ip, MAX_CONCURRENT_TARPITS_PER_IP,
                     reason=f"Tarpit abuse (>{MAX_CONCURRENT_TARPITS_PER_IP} concurrent throttled connections)")
            with stats_lock:
                gw_stats["block_hits"] += 1
            return _blocked_response(BLOCK_SECONDS_FALLBACK)

        if slot == "full":
            # Gateway is saturated with tarpits: shed load instead of sleeping.
            with stats_lock:
                gw_stats["shed_hits"] += 1
            resp = jsonify({
                "error": "Service busy. Please retry shortly.",
                "gateway": f"Gateway-{GATEWAY_ID}",
                "retry_after_seconds": SHED_RETRY_SECONDS
            })
            resp.status_code = 429
            resp.headers["Retry-After"] = str(SHED_RETRY_SECONDS)
            resp.headers["X-Defense-Action"] = "Shed"
            resp.headers["X-Handled-By"] = f"Gateway-{GATEWAY_ID}"
            return resp

        tarpitted = True
        try:
            time.sleep(value)
        finally:
            _leave_tarpit(client_ip)

    # 3. Forward clean traffic to the main backend server
    target_url = f"{BACKEND_URL}/{path}"
    if request.query_string:
        target_url += f"?{request.query_string.decode('utf-8')}"

    # Filter incoming headers before forwarding
    headers = {key: value for key, value in request.headers if key.lower() not in ("host", "content-length")}
    headers["X-Forwarded-For"] = client_ip
    headers["X-Gateway-ID"] = f"Gateway-{GATEWAY_ID}"
    if tarpitted:
        headers["X-Defense-Tarpit"] = "true"

    try:
        backend_response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=10.0
        )

        # Build Flask response from backend response
        excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        res_headers = [
            (k, v) for k, v in backend_response.headers.items()
            if k.lower() not in excluded_headers
        ]
        res_headers.append(("X-Handled-By", f"Gateway-{GATEWAY_ID}"))
        if tarpitted:
            res_headers.append(("X-Defense-Status", "Tarpitted"))

        return Response(
            backend_response.content,
            status=backend_response.status_code,
            headers=res_headers
        )

    except requests.RequestException as err:
        return jsonify({
            "error": "Gateway Error: Unable to reach core Netguard application server.",
            "details": str(err),
            "gateway": f"Gateway-{GATEWAY_ID}"
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)