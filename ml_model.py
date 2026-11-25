from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from joblib import load

BASE_DIR = Path(__file__).resolve().parent / "ml"
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
DATA_PATH = BASE_DIR / "data" / "test.csv"

POOL_SIZE = 5000  
BENIGN_NAMES = {"BENIGN", "NORMAL", "0", "SAFE"}

_lock = Lock()
_bundle = None
_pool = None


def _to_label(value):
    """Collapse any class name into BENIGN or ATTACK."""
    return "BENIGN" if str(value).strip().upper() in BENIGN_NAMES else "ATTACK"


def _to_binary(labels):
    return labels.apply(_to_label)


def _load_model():
    global _bundle
    with _lock:
        if _bundle is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "Model not found. Run ml/train_ml.py first."
                )
            _bundle = load(MODEL_PATH)
        return _bundle


def _load_pool():
    """Load the dataset once and keep a random sample of it in memory."""
    global _pool
    with _lock:
        if _pool is None:
            if not DATA_PATH.exists():
                raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
            data = pd.read_csv(DATA_PATH, low_memory=False)
            data.columns = data.columns.str.strip()
            data = data.dropna(subset=["Label"])
            _pool = data.sample(
                min(POOL_SIZE, len(data)), random_state=42
            ).reset_index(drop=True)
        return _pool


def _prepare(df, features):
    """Mirror the preprocessing in train_ml.py, in the same column order."""
    X = df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median()).fillna(0)


def classify_live_flow(requests_history):
    """
    Extracts the 14 CIC-DDoS features from live request history of an IP
    and runs the trained Random Forest classifier.
    Returns: dict with is_attack, confidence, label, and key metrics.
    """
    if not requests_history or len(requests_history) < 2:
        return {"is_attack": False, "confidence": 0.0, "label": "SAFE"}

    try:
        bundle = _load_model()
        model = bundle["model"]
        features = bundle["features"]

        timestamps = [r["timestamp"] for r in requests_history]
        sizes = [r.get("size", 120.0) for r in requests_history]
        headers_len = [r.get("header_len", 40.0) for r in requests_history]

        duration_sec = max(timestamps[-1] - timestamps[0], 0.000001)
        duration_us = duration_sec * 1e6

        iats = [(t2 - t1) * 1e6 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        iat_mean = float(np.mean(iats)) if iats else 0.0

        count = len(requests_history)
        packets_per_sec = float(count / duration_sec)

        payload_sizes = [float(s) for s in sizes]
        pkt_len_mean = float(np.mean(payload_sizes))
        pkt_len_std = float(np.std(payload_sizes))
        pkt_len_var = float(np.var(payload_sizes))

        # Build feature map for the 14 trained model columns
        live_dict = {
            "Flow Duration": float(duration_us),
            "Fwd Packet Length Std": float(pkt_len_std),
            "Flow Packets/s": float(packets_per_sec),
            "Flow IAT Mean": float(iat_mean),
            "Fwd Header Length": float(sum(headers_len)),
            "Packet Length Mean": float(pkt_len_mean),
            "Packet Length Std": float(pkt_len_std),
            "Packet Length Variance": float(pkt_len_var),
            "ACK Flag Count": float(count),
            "Subflow Fwd Packets": float(count),
            "Subflow Fwd Bytes": float(sum(payload_sizes)),
            "Init_Win_bytes_forward": 252.0 if count < 10 else -1.0,
            "act_data_pkt_fwd": float(count),
            "min_seg_size_forward": 20.0
        }

        df = pd.DataFrame([live_dict])
        df = df.reindex(columns=features).fillna(0)

        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0]

        # Locate the UNSAFE / ATTACK class probability
        unsafe_idx = None
        for i, c in enumerate(model.classes_):
            if str(c).upper() in ("UNSAFE", "ATTACK", "1"):
                unsafe_idx = i
                break

        if unsafe_idx is not None:
            unsafe_prob = float(probs[unsafe_idx])
        else:
            unsafe_prob = 1.0 if str(pred).upper() in ("UNSAFE", "ATTACK") else 0.0

        is_attack = (str(pred).upper() in ("UNSAFE", "ATTACK")) or (unsafe_prob >= 0.70)

        return {
            "is_attack": bool(is_attack),
            "label": str(pred),
            "confidence": round(unsafe_prob, 3),
            "packets_per_sec": round(packets_per_sec, 1),
            "iat_mean_us": round(iat_mean, 1)
        }
    except Exception as e:
        return {"is_attack": False, "confidence": 0.0, "label": "ERROR", "error": str(e)}


def analyze_sample(n=200):
    """Classify n random flow records from the dataset and summarise."""
    bundle = _load_model()
    model = bundle["model"]
    features = bundle["features"]

    pool = _load_pool()
    sample = pool.sample(min(n, len(pool)))

    y_true = _to_binary(sample["Label"]).values
    X = _prepare(sample, features)

    predictions = np.array([_to_label(p) for p in model.predict(X)])
    probabilities = model.predict_proba(X)
    attack_cols = [
        i for i, c in enumerate(model.classes_) if _to_label(c) == "ATTACK"
    ]
    attack_prob = probabilities[:, attack_cols].sum(axis=1) if len(attack_cols) > 0 else np.zeros(len(X))

    rows = len(sample)
    predicted_attack = int((predictions == "ATTACK").sum())
    attack_ratio = predicted_attack / rows if rows else 0

    if attack_ratio >= 0.5:
        verdict = "Most sampled flows look like attack traffic."
    elif attack_ratio > 0.1:
        verdict = "A notable share of sampled flows look like attack traffic."
    else:
        verdict = "Sampled flows look mostly benign."

    return {
        "rows_analyzed": rows,
        "predicted_attack": predicted_attack,
        "predicted_benign": rows - predicted_attack,
        "actual_attack": int((y_true == "ATTACK").sum()),
        "accuracy": round(float((predictions == y_true).mean()), 4),
        "avg_attack_probability": round(float(attack_prob.mean()), 3),
        "verdict": verdict
    }
