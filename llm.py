import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env")

_client = None

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant for an advanced multi-tier DDoS mitigation system. "
    "Explain DDoS monitoring alerts and defensive actions in simple, executive language. "
    "Write plain text only, with no markdown, asterisks or bullet symbols. "
    "Do not claim an attack is confirmed from request counts alone. "
    "Live request counts and offline dataset tests are separate things; "
    "never treat the dataset test as evidence about live traffic. "
    "Always answer in exactly these three labelled parts, one short paragraph each, about 140 words in total:\n"
    "Why this may be an attack: describe what the traffic volume and gateway distribution show and why "
    "that pattern is suspicious, using hedged wording such as 'may' or 'is consistent with'.\n"
    "Action already taken: detail the multi-tiered defense response: mention whether progressive tarpitting "
    "(artificial response throttling) was engaged to exhaust attacker sockets, and if automatic blocking "
    "was applied across the gateway cluster, naming blocked IPs and rejection numbers. "
    "If no defensive action was needed, state that traffic remains within normal operational limits.\n"
    "What to check next: one or two sensible follow-up steps, such as inspecting gateway load distribution, "
    "verifying whether blocked IPs resume attacks after the cooldown, or examining user-agent signatures."
)


def _get_client():
    """Create the Groq client on first use so the app still starts
    (and the dashboard still works) if the key is missing."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found. Check your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def explain_alert(request_count, threshold, window_seconds=10,
                  top_ip=None, top_ip_count=None, ml_summary=None,
                  blocked=None, blocked_total=0, block_seconds=60,
                  tarpit_total=0, tarpitted_ips=None, gateways=None,
                  ml_blocked_total=0):
    lines = [
        f"Requests across cluster in last {window_seconds} seconds: {request_count}.",
        f"Alert threshold: {threshold} total requests per window.",
    ]
    if top_ip:
        lines.append(
            f"Busiest source IP in the window: {top_ip} with {top_ip_count} requests."
        )

    if gateways:
        gw_summary = ", ".join(f"{g['id']}: {g['requests']} reqs" for g in gateways[:3])
        lines.append(f"Gateway distribution: {gw_summary}.")

    if ml_blocked_total:
        lines.append(
            f"Pre-emptive Machine Learning Detection: {ml_blocked_total} attack source(s) were "
            f"classified as malicious by the Random Forest model in the gray zone and blocked early."
        )

    if tarpitted_ips:
        lines.append(
            f"Progressive Tarpitting is active: {len(tarpitted_ips)} IP(s) currently being slowed down "
            f"with artificial latency to exhaust attacker connection pools. Total tarpitted hits: {tarpit_total}."
        )

    if blocked:
        details = "; ".join(
            f"{b['ip']} (blocked after {b.get('trigger_count', 50)} requests in "
            f"{window_seconds}s, {b['requests_blocked']} requests rejected, "
            f"{b['seconds_left']}s remaining)"
            for b in blocked[:5]
        )
        lines.append(
            f"Cluster-wide blocking is active. {len(blocked)} IP(s) are "
            f"currently blocked: {details}. Each block lasts "
            f"{block_seconds} seconds and blocked clients receive HTTP 429. "
            f"Total requests rejected so far: {blocked_total}."
        )
    else:
        lines.append("Automatic blocking: no IPs are currently blocked.")

    if ml_summary:
        lines.append(f"ML classifier note: {ml_summary}")

    response = _get_client().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": " ".join(lines)}
        ],
        temperature=0.2,
        max_completion_tokens=800,
        reasoning_effort="low"
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("\nLLM Response:")
    print(explain_alert(
        request_count=12,
        threshold=100,
        top_ip="172.18.0.3",
        top_ip_count=3,
        blocked=[{"ip": "172.18.0.4", "trigger_count": 60,
                  "requests_blocked": 2400, "seconds_left": 41}],
        blocked_total=4100,
        tarpit_total=150,
        tarpitted_ips=["172.18.0.4"]
    ))
