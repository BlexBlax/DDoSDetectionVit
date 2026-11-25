import os
from time import time

from flask import Flask, jsonify, request, render_template

from detector import record_request, get_stats, check_is_blocked, reset_all_stats
from ml_model import analyze_sample
from llm import explain_alert

app = Flask(__name__)

# The dashboard polls /stats every second. Don't count its own calls
# (or static files) as traffic, or the numbers are inflated.
IGNORED_PREFIXES = ("/static", "/stats", "/analyze", "/explain", "/reset", "/favicon", "/_health")

LLM_COOLDOWN_SECONDS = 5
last_ml_result = None
last_explanation = {"time": 0.0, "text": None}


def get_client_ip():
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.headers.get("X-Real-IP") or request.remote_addr or "127.0.0.1"


@app.before_request
def monitor_request():
    if request.path.startswith(IGNORED_PREFIXES):
        return

    ip = get_client_ip()
    gateway_id = request.headers.get("X-Gateway-ID", "Direct")

    # Check if this IP is already blocked in Redis
    seconds_left = check_is_blocked(ip)
    if seconds_left > 0:
        response = jsonify({
            "error": "Too many requests. You are temporarily blocked by the security system.",
            "retry_after_seconds": seconds_left
        })
        response.status_code = 429
        response.headers["Retry-After"] = str(seconds_left)
        return response

    record_request(ip, request.path, request.method, gateway_id=gateway_id)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/stats")
def stats():
    return jsonify(get_stats())


@app.route("/reset", methods=["POST", "GET"])
def reset():
    """Reset all telemetry metrics, Redis state, and active blocks to zero."""
    reset_all_stats()
    return jsonify({"status": "ok", "message": "All metrics and active blocks reset to 0"})


@app.route("/analyze")
def analyze():
    """Run the Random Forest on a random sample of dataset flow records."""
    global last_ml_result
    n = request.args.get("n", default=200, type=int)
    n = max(10, min(n, 2000))

    try:
        last_ml_result = analyze_sample(n)
        return jsonify(last_ml_result)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500
    except Exception as error:
        app.logger.exception("ML analysis failed")
        return jsonify({"error": f"{type(error).__name__}: {error}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """Ask the LLM to explain the current traffic snapshot."""
    now = time()
    if (last_explanation["text"]
            and now - last_explanation["time"] < LLM_COOLDOWN_SECONDS):
        return jsonify({"explanation": last_explanation["text"]})

    snapshot = get_stats()

    ml_summary = None
    if last_ml_result:
        pct = round(
            100 * last_ml_result["predicted_attack"]
            / last_ml_result["rows_analyzed"]
        )
        ml_summary = (
            f"A separate offline test of the classifier on "
            f"{last_ml_result['rows_analyzed']} randomly chosen records from "
            f"a labelled dataset (not live traffic) classified {pct}% as "
            f"attack. This says nothing about the current live traffic."
        )

    try:
        text = explain_alert(
            request_count=snapshot["requests_in_window"],
            threshold=snapshot["threshold"],
            window_seconds=snapshot["window_seconds"],
            top_ip=snapshot["top_ip"],
            top_ip_count=snapshot["top_ip_count"],
            ml_summary=ml_summary,
            blocked=snapshot["blocked"],
            blocked_total=snapshot["blocked_total"],
            block_seconds=snapshot["block_seconds"],
            tarpit_total=snapshot.get("tarpit_total", 0),
            tarpitted_ips=snapshot.get("tarpitted_ips", []),
            gateways=snapshot.get("gateways", []),
            ml_blocked_total=snapshot.get("ml_blocked_total", 0)
        )
    except ValueError as error:           # missing API key
        return jsonify({"error": str(error)}), 500
    except Exception:
        app.logger.exception("LLM request failed")
        return jsonify({"error": "The AI explanation failed. Check the server log."}), 502

    last_explanation.update(time=now, text=text)
    return jsonify({"explanation": text})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "0.0.0.0"),
        port=5001,
        debug=False
    )
