import math
import os
from collections import deque
from threading import Lock
from time import sleep as time_sleep, time
from uuid import uuid4

from ml_model import classify_live_flow

try:
    import redis
except ImportError:
    redis = None

# ---------------------------------------------------------------------------
# Local (per-process) state. Used for the dashboard on the core server and as
# a fallback for each gateway if Redis is unreachable.
# ---------------------------------------------------------------------------
request_log = deque()
ip_requests_history = {}  # ip -> deque of {"timestamp", "size", "header_len"}
window_counts = {}        # ip -> requests currently inside the window
blocked = {}              # ip -> {"until", "hits", "trigger", "reason"}
tarpitted = {}            # ip -> {"until", "hits"}
local_strikes = {}        # ip -> {"count", "reset_at"}   (tarpit escalation fallback)
ml_checked = {}           # ip -> last ML inspection timestamp (fallback throttle)
blocked_total = 0
ml_blocked_total = 0
tarpit_total = 0
lock = Lock()
_last_gc = 0.0

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 10
RATE_THRESHOLD = 100          # cluster-wide requests per window -> dashboard "alert"
ELEVATED_RATIO = 0.6
MAX_RECENT = 50
BLOCK_SECONDS = 60
ALLOWLIST = {ip.strip() for ip in os.environ.get("ALLOWLIST_IPS", "127.0.0.1,localhost,::1").split(",") if ip.strip()}

# Leaky bucket (shared across all gateways through Redis)
BUCKET_RATE = 5.0             # requests/sec that leak out of the bucket
BUCKET_CAPACITY = 20          # burst allowance; the 21st rapid request overflows
BUCKET_TARPIT_LEVEL = 10      # above this fill level, requests are tarpitted

# Tarpit
TARPIT_DELAY = 2.0
TARPIT_ESCALATION_HITS = 15   # this many tarpitted requests ...
TARPIT_ESCALATION_WINDOW = 30  # ... within this many seconds -> hard block

# ML gray zone
ML_GRAY_ZONE_THRESHOLD = 12   # run the model once an IP has >= 12 requests in its flow
ML_CONFIDENCE_THRESHOLD = 0.70
ML_THROTTLE_MS = 300          # run the model at most once per IP per 300 ms (cluster-wide)
FLOW_HISTORY_LEN = 60

# Fallback limits (only used when Redis is down and the bucket is unavailable)
TARPIT_THRESHOLD = 25
IP_THRESHOLD = 50

# ---------------------------------------------------------------------------
# Redis connection (shared blocklist, buckets, flow history, telemetry)
# ---------------------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_CONNECT_RETRIES = 20    # x 1s: Redis may still be booting when we start
redis_client = None
redis_connected = False       # was Redis reachable at startup? (informational)

if REDIS_HOST and redis:
    # IMPORTANT: keep the client even if the first ping fails. redis-py connects
    # lazily and reconnects on its own, and every call below is wrapped in
    # try/except. Setting redis_client = None on a failed ping (as before) made
    # a Redis that was a second late at boot disable the shared blocklist for
    # the whole life of the process.
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_timeout=1.0,
        socket_connect_timeout=1.0
    )
    for _attempt in range(REDIS_CONNECT_RETRIES):
        try:
            redis_client.ping()
            redis_connected = True
            break
        except Exception:
            time_sleep(1.0)
    print(f"[detector] Redis {REDIS_HOST}:{REDIS_PORT} "
          f"{'connected' if redis_connected else 'NOT reachable yet (will keep retrying per request)'}",
          flush=True)
else:
    print("[detector] REDIS_HOST not set: running with per-process state only", flush=True)

# Atomic leaky bucket. Runs inside Redis so all gateways share one bucket per
# IP with no race conditions, and uses Redis' own clock (no clock skew).
LEAKY_BUCKET_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local rate = tonumber(ARGV[1])
local cap = tonumber(ARGV[2])
local d = redis.call('HMGET', KEYS[1], 'level', 'ts')
local level = tonumber(d[1]) or 0
local ts = tonumber(d[2]) or now
level = math.max(0, level - (now - ts) * rate)
local allowed = 1
if level + 1 > cap then
  allowed = 0
else
  level = level + 1
end
redis.call('HSET', KEYS[1], 'level', string.format('%.4f', level), 'ts', string.format('%.6f', now))
redis.call('EXPIRE', KEYS[1], math.ceil(cap / rate) + 5)
return {allowed, string.format('%.4f', level)}
"""

_bucket_script = None
if redis_client:
    try:
        _bucket_script = redis_client.register_script(LEAKY_BUCKET_LUA)
    except Exception:
        _bucket_script = None


# ---------------------------------------------------------------------------
# Local state housekeeping
# ---------------------------------------------------------------------------
def _trim(now):
    """Drop requests older than the window. Caller must hold the lock."""
    global _last_gc
    cutoff = now - WINDOW_SECONDS
    while request_log and request_log[0]["timestamp"] < cutoff:
        old = request_log.popleft()
        ip = old["ip"]
        window_counts[ip] -= 1
        if window_counts[ip] <= 0:
            del window_counts[ip]

    # Per-IP housekeeping is O(number of IPs), so do it at most once a second
    # instead of on every request.
    if now - _last_gc >= 1.0:
        _last_gc = now
        for ip in list(ip_requests_history.keys()):
            h = ip_requests_history[ip]
            while h and h[0]["timestamp"] < cutoff:
                h.popleft()
            if not h:
                del ip_requests_history[ip]
        for ip in [ip for ip, t in ml_checked.items() if now - t > WINDOW_SECONDS]:
            del ml_checked[ip]
        for ip in [ip for ip, s in local_strikes.items() if s["reset_at"] <= now]:
            del local_strikes[ip]


def _expire_local_states(now):
    """Lift local blocks and tarpits whose time is up. Caller must hold the lock."""
    for ip in [ip for ip, b in blocked.items() if b["until"] <= now]:
        del blocked[ip]
    for ip in [ip for ip, t in tarpitted.items() if t["until"] <= now]:
        del tarpitted[ip]


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------
def check_is_blocked(ip, count_hits=True):
    """
    Check if an IP is blocked in Redis or locally. Returns seconds_left or 0.
    Pass count_hits=False from components that only re-check an already
    inspected request (the core server), so rejections are not counted twice.
    """
    global blocked_total
    now = time()

    if redis_client:
        try:
            ttl = redis_client.ttl(f"block:{ip}")
            if ttl and ttl > 0:
                if count_hits:
                    pipe = redis_client.pipeline()
                    pipe.incr("stats:blocked_hits")
                    pipe.hincrby(f"block_hits:{ip}", "hits", 1)
                    pipe.execute()
                return ttl
        except Exception:
            pass

    with lock:
        _expire_local_states(now)
        entry = blocked.get(ip)
        if not entry:
            return 0
        if count_hits:
            entry["hits"] += 1
            blocked_total += 1
        return max(1, math.ceil(entry["until"] - now))


def block_ip(ip, trigger_count, reason="Rate Threshold Exceeded"):
    """Marks an IP as blocked across the cluster (Redis) and locally."""
    now = time()
    print(f"[BLOCK] {ip} for {BLOCK_SECONDS}s - {reason}", flush=True)

    if redis_client:
        try:
            # nx=True: if another gateway already blocked this IP a moment ago,
            # don't extend its ban or overwrite the reason.
            created = redis_client.set(
                f"block:{ip}", int(trigger_count), ex=BLOCK_SECONDS, nx=True
            )
            if created:
                redis_client.hset(f"block_meta:{ip}", mapping={
                    "ip": ip,
                    "trigger": int(trigger_count),
                    "reason": reason,
                    "start": now
                })
                redis_client.expire(f"block_meta:{ip}", BLOCK_SECONDS)
        except Exception:
            pass

    with lock:
        blocked[ip] = {
            "until": now + BLOCK_SECONDS,
            "hits": 0,
            "trigger": int(trigger_count),
            "reason": reason
        }


# ---------------------------------------------------------------------------
# Request recording (local log + shared flow history)
# ---------------------------------------------------------------------------
def record_request(ip, path, method, gateway_id=None, size=120, header_len=40):
    """Records a request into the local sliding window and local flow history."""
    now = time()
    request = {
        "id": str(uuid4()),
        "timestamp": now,
        "ip": ip,
        "path": path,
        "method": method,
        "gateway": gateway_id or "gateway-1"
    }
    with lock:
        request_log.append(request)
        window_counts[ip] = window_counts.get(ip, 0) + 1

        if ip not in ip_requests_history:
            ip_requests_history[ip] = deque(maxlen=FLOW_HISTORY_LEN)
        ip_requests_history[ip].append({
            "timestamp": now,
            "size": size,
            "header_len": header_len
        })

        _trim(now)
    return request


def _push_and_get_flow(ip, now, size, header_len):
    """
    Append this request to the cluster-wide flow history for the IP and return
    the recent history (all gateways combined) for the ML model.
    Returns None if Redis is unavailable.
    """
    if not redis_client:
        return None
    try:
        key = f"flow:{ip}"
        pipe = redis_client.pipeline()
        pipe.rpush(key, f"{now}|{size}|{header_len}")
        pipe.ltrim(key, -FLOW_HISTORY_LEN, -1)
        pipe.expire(key, WINDOW_SECONDS * 2)
        pipe.lrange(key, 0, -1)
        raw = pipe.execute()[-1]

        cutoff = now - WINDOW_SECONDS
        history = []
        for item in raw:
            ts, sz, hl = item.split("|")
            ts = float(ts)
            if ts >= cutoff:
                history.append({"timestamp": ts, "size": float(sz), "header_len": float(hl)})
        history.sort(key=lambda r: r["timestamp"])
        return history
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Defense primitives
# ---------------------------------------------------------------------------
def bucket_hit(ip):
    """
    Atomic shared leaky bucket.
    Returns (allowed, level), or None if Redis is unavailable.
    """
    if not _bucket_script:
        return None
    try:
        allowed, level = _bucket_script(
            keys=[f"bucket:{ip}"], args=[BUCKET_RATE, BUCKET_CAPACITY]
        )
        return bool(int(allowed)), float(level)
    except Exception:
        return None


def mark_tarpit(ip):
    """Tracks that this IP is in a tarpitted state."""
    global tarpit_total
    now = time()

    if redis_client:
        try:
            pipe = redis_client.pipeline()
            pipe.setex(f"tarpit:{ip}", WINDOW_SECONDS, 1)
            pipe.incr("stats:tarpit_hits")
            pipe.execute()
        except Exception:
            pass

    with lock:
        tarpit_total += 1
        tarpitted[ip] = {
            "until": now + WINDOW_SECONDS,
            "hits": tarpitted.get(ip, {}).get("hits", 0) + 1
        }


def _tarpit_strikes(ip, now):
    """Count tarpit hits for an IP in a rolling window (shared via Redis)."""
    if redis_client:
        try:
            key = f"strikes:{ip}"
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            strikes, ttl = pipe.execute()
            if ttl is None or ttl < 0:
                redis_client.expire(key, TARPIT_ESCALATION_WINDOW)
            return int(strikes)
        except Exception:
            pass

    with lock:
        entry = local_strikes.get(ip)
        if not entry or entry["reset_at"] <= now:
            entry = {"count": 0, "reset_at": now + TARPIT_ESCALATION_WINDOW}
            local_strikes[ip] = entry
        entry["count"] += 1
        return entry["count"]


def _ml_due(ip, now):
    """Throttle ML inspection to once per ML_THROTTLE_MS per IP, cluster-wide."""
    if redis_client:
        try:
            return bool(redis_client.set(f"mlcheck:{ip}", 1, nx=True, px=ML_THROTTLE_MS))
        except Exception:
            pass
    with lock:
        if now - ml_checked.get(ip, 0) >= ML_THROTTLE_MS / 1000.0:
            ml_checked[ip] = now
            return True
    return False


# ---------------------------------------------------------------------------
# Main inspection pipeline (called by each gateway)
# ---------------------------------------------------------------------------
def evaluate_request(ip, path, method, user_agent=None, gateway_id=None, size=120, header_len=40):
    """
    Evaluates an incoming request:
      1. Global blocklist (Redis)
      2. Shared leaky bucket (Redis)
      3. ML flow classification in the gray zone
      4. Tarpit, escalating to a block if the client keeps hammering
    Returns (action, value):
      ("BLOCK", seconds_left) | ("TARPIT", delay_seconds) | ("ALLOW", 0)
    """
    global ml_blocked_total
    now = time()

    if ip in ALLOWLIST:
        record_request(ip, path, method, gateway_id, size, header_len)
        return "ALLOW", 0

    # 1. Already banned anywhere in the cluster?
    sec_left = check_is_blocked(ip)
    if sec_left > 0:
        return "BLOCK", sec_left

    # Record locally (dashboard / fallback) and in the shared flow history (ML)
    record_request(ip, path, method, gateway_id, size, header_len)
    history = _push_and_get_flow(ip, now, size, header_len)

    with lock:
        local_count = window_counts.get(ip, 0)
        if history is None:
            history = list(ip_requests_history.get(ip, []))

    # 2. Leaky bucket (shared). Falls back to local counters if Redis is down.
    level = None
    res = bucket_hit(ip)
    if res is not None:
        allowed, level = res
        if not allowed:
            block_ip(
                ip, int(level),
                reason=f"Leaky bucket overflow ({BUCKET_RATE:g} req/s leak, burst {BUCKET_CAPACITY})"
            )
            return "BLOCK", BLOCK_SECONDS
    elif local_count >= IP_THRESHOLD:
        block_ip(ip, local_count, reason=f"Rate limit exceeded ({IP_THRESHOLD} reqs/window)")
        return "BLOCK", BLOCK_SECONDS

    # 3. ML gray zone: enough flow data to classify, throttled per IP
    if len(history) >= ML_GRAY_ZONE_THRESHOLD and _ml_due(ip, now):
        ml_res = classify_live_flow(history)
        if ml_res.get("is_attack") and ml_res.get("confidence", 0) >= ML_CONFIDENCE_THRESHOLD:
            conf_pct = int(ml_res["confidence"] * 100)
            block_ip(ip, len(history), reason=f"ML: Random Forest ({conf_pct}% confidence)")
            with lock:
                ml_blocked_total += 1
            if redis_client:
                try:
                    redis_client.incr("stats:ml_blocked_hits")
                except Exception:
                    pass
            return "BLOCK", BLOCK_SECONDS

    # 4. Tarpit suspicious clients; block the ones that ignore it
    should_tarpit = (
        (level is not None and level >= BUCKET_TARPIT_LEVEL)
        or (level is None and local_count >= TARPIT_THRESHOLD)
    )
    if should_tarpit:
        mark_tarpit(ip)
        if _tarpit_strikes(ip, now) >= TARPIT_ESCALATION_HITS:
            block_ip(
                ip, local_count,
                reason=f"Tarpit escalation ({TARPIT_ESCALATION_HITS} throttled requests in {TARPIT_ESCALATION_WINDOW}s)"
            )
            return "BLOCK", BLOCK_SECONDS
        return "TARPIT", TARPIT_DELAY

    return "ALLOW", 0


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
def classify_traffic(total, top_count, has_tarpit):
    """Returns 'normal', 'elevated', 'tarpitting', or 'alert'."""
    if total >= RATE_THRESHOLD or top_count >= IP_THRESHOLD:
        return "alert"
    if has_tarpit or top_count >= TARPIT_THRESHOLD:
        return "tarpitting"
    if (total >= RATE_THRESHOLD * ELEVATED_RATIO
            or top_count >= TARPIT_THRESHOLD * ELEVATED_RATIO):
        return "elevated"
    return "normal"


def get_stats():
    """Aggregates telemetry from the sliding window, Redis, ML engine, and gateways."""
    now = time()

    with lock:
        _trim(now)
        _expire_local_states(now)
        total = len(request_log)
        ip_counts = dict(window_counts)
        recent_requests = list(request_log)[-MAX_RECENT:]

        local_blocked = [
            {
                "ip": ip,
                "seconds_left": max(1, math.ceil(b["until"] - now)),
                "requests_blocked": b["hits"],
                "trigger_count": b["trigger"],
                "reason": b.get("reason", "Rate Threshold Exceeded")
            }
            for ip, b in blocked.items()
        ]
        rejected_total = blocked_total
        t_total = tarpit_total
        ml_total = ml_blocked_total
        active_tarpits = [ip for ip, t in tarpitted.items() if t["until"] > now]

    merged_blocked = {b["ip"]: b for b in local_blocked}
    active_buckets = 0

    if redis_client:
        try:
            for k in redis_client.scan_iter(match="block:*", count=200):
                ip = k.replace("block:", "", 1)
                ttl = redis_client.ttl(k)
                if ttl and ttl > 0:
                    hits = int(redis_client.hget(f"block_hits:{ip}", "hits") or 0)
                    meta = redis_client.hgetall(f"block_meta:{ip}") or {}
                    merged_blocked[ip] = {
                        "ip": ip,
                        "seconds_left": ttl,
                        "requests_blocked": hits,
                        "trigger_count": int(meta.get("trigger", IP_THRESHOLD)),
                        "reason": meta.get("reason", "Cluster Rate Threshold")
                    }

            v = redis_client.get("stats:blocked_hits")
            if v:
                rejected_total = max(rejected_total, int(v))
            v = redis_client.get("stats:tarpit_hits")
            if v:
                t_total = max(t_total, int(v))
            v = redis_client.get("stats:ml_blocked_hits")
            if v:
                ml_total = max(ml_total, int(v))

            for tk in redis_client.scan_iter(match="tarpit:*", count=200):
                tip = tk.replace("tarpit:", "", 1)
                if tip not in active_tarpits:
                    active_tarpits.append(tip)

            active_buckets = sum(1 for _ in redis_client.scan_iter(match="bucket:*", count=200))
        except Exception:
            pass

    blocked_list = list(merged_blocked.values())

    if ip_counts:
        top_ip, top_count = max(ip_counts.items(), key=lambda item: item[1])
    else:
        top_ip, top_count = None, 0

    has_tarpit = len(active_tarpits) > 0
    if blocked_list:
        status = "mitigating"
    else:
        status = classify_traffic(total, top_count, has_tarpit)

    gateways_info = []
    if redis_client:
        try:
            gw_keys = sorted(redis_client.scan_iter(match="gateway:*", count=50))
            for gk in gw_keys:
                gdata = redis_client.hgetall(gk)
                if gdata:
                    last_seen = float(gdata.get("timestamp", 0))
                    is_active = (now - last_seen) < 15
                    gateways_info.append({
                        "id": gdata.get("id", gk.replace("gateway:", "")),
                        "status": "Online" if is_active else "Degraded",
                        "port": gdata.get("port", "?"),
                        "requests": int(gdata.get("requests", 0)),
                        "tarpit_hits": int(gdata.get("tarpit_hits", 0)),
                        "block_hits": int(gdata.get("block_hits", 0))
                    })
        except Exception:
            pass

    if not gateways_info:
        gateways_info = [
            {"id": "Gateway-1", "status": "Online", "port": "5002", "requests": total // 3, "tarpit_hits": t_total, "block_hits": rejected_total},
            {"id": "Gateway-2", "status": "Online", "port": "5003", "requests": total // 3, "tarpit_hits": 0, "block_hits": 0},
            {"id": "Gateway-3", "status": "Online", "port": "5004", "requests": total // 3, "tarpit_hits": 0, "block_hits": 0}
        ]

    return {
        "window_seconds": WINDOW_SECONDS,
        "requests_in_window": total,
        "requests_per_second": round(total / WINDOW_SECONDS, 2),
        "requests_by_ip": ip_counts,
        "recent_requests": recent_requests,
        "status": status,
        "threshold": RATE_THRESHOLD,
        "ml_gray_zone_threshold": ML_GRAY_ZONE_THRESHOLD,
        "tarpit_threshold": TARPIT_THRESHOLD,
        "ip_threshold": IP_THRESHOLD,
        "tarpit_delay": TARPIT_DELAY,
        "bucket_rate": BUCKET_RATE,
        "bucket_capacity": BUCKET_CAPACITY,
        "active_buckets": active_buckets,
        "top_ip": top_ip,
        "top_ip_count": top_count,
        "blocked": blocked_list,
        "blocked_total": rejected_total,
        "ml_blocked_total": ml_total,
        "tarpit_total": t_total,
        "tarpitted_ips": active_tarpits,
        "block_seconds": BLOCK_SECONDS,
        "gateways": gateways_info
    }


def reset_all_stats():
    """Reset all local and cluster-wide telemetry, blocklists, buckets and counters."""
    global blocked_total, ml_blocked_total, tarpit_total
    with lock:
        request_log.clear()
        ip_requests_history.clear()
        window_counts.clear()
        blocked.clear()
        tarpitted.clear()
        local_strikes.clear()
        ml_checked.clear()
        blocked_total = 0
        ml_blocked_total = 0
        tarpit_total = 0

    if redis_client:
        try:
            now_ts = str(time())
            redis_client.set("signal:reset", now_ts, ex=15)

            patterns = (
                "block:*", "block_hits:*", "block_meta:*", "tarpit:*",
                "stats:*", "flow:*", "bucket:*", "strikes:*", "mlcheck:*"
            )
            for pattern in patterns:
                keys = list(redis_client.scan_iter(match=pattern, count=500))
                if keys:
                    redis_client.delete(*keys)

            for gk in list(redis_client.scan_iter(match="gateway:*", count=50)):
                redis_client.hset(gk, mapping={
                    "requests": "0",
                    "tarpit_hits": "0",
                    "block_hits": "0",
                    "timestamp": now_ts
                })
        except Exception:
            pass