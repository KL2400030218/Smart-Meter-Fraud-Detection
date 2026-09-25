"""
predict.py
==========
Loads the trained Random Forest model (selected by train.py based on
actual F1-score) and scores a single incoming meter reading, using the
SAME feature-engineering transform as training (preprocess.py) to avoid
training-serving skew.

Produces:
  - fraud_probability  (actual model output, 0.0-1.0)
  - risk_score          (0-100, derived from fraud_probability)
  - risk_level          (LOW / MEDIUM / HIGH, configurable thresholds)
  - prediction           (NORMAL / SUSPICIOUS)
  - explanation          (rule-based, generated from the ACTUAL feature
                          values of this reading — explicitly labeled as
                          rule-based, not a SHAP/model-native explanation)

Risk thresholds are configurable via RISK_THRESHOLDS below (also meant
to be exposed as backend config, per spec section 11).
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from preprocess import build_features_single, FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

RISK_THRESHOLDS = {"LOW_MAX": 30, "MEDIUM_MAX": 70}  # HIGH is anything above MEDIUM_MAX
ENTROPY_THRESHOLD = 0.75  # uncertainty threshold tau_H for the adaptive blockchain decision


def prediction_entropy(p: float) -> float:
    """Binary predictive entropy H(p), in bits, per
    docs/architecture/mathematical_formulation.md section 5.
    H(p)=0 when the model is fully confident (p near 0 or 1),
    H(p)=1 when the model is maximally unsure (p near 0.5)."""
    eps = 1e-12  # avoid log(0)
    p = min(max(p, eps), 1 - eps)
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def should_record_on_chain(risk_score: int, fraud_probability: float) -> dict:
    """Adaptive Blockchain Decision stage (pipeline section between
    Explainability and Hash Generation). Returns whether this prediction
    is worth the on-chain recording cost, and why — always record
    HIGH-risk predictions, and additionally record MEDIUM-risk
    predictions the model itself is uncertain about (high entropy),
    since those are the cases most likely to need a later audit trail.
    LOW-risk, confident predictions are skipped to avoid unnecessary
    blockchain writes."""
    entropy = prediction_entropy(fraud_probability)
    high_risk = risk_score > RISK_THRESHOLDS["MEDIUM_MAX"]
    uncertain_medium_risk = (
        entropy > ENTROPY_THRESHOLD and risk_score > RISK_THRESHOLDS["LOW_MAX"]
    )
    decision = high_risk or uncertain_medium_risk
    if high_risk:
        reason = "high risk score — always recorded for audit trail"
    elif uncertain_medium_risk:
        reason = f"medium risk with high model uncertainty (entropy={entropy:.3f} > {ENTROPY_THRESHOLD})"
    else:
        reason = "low risk and/or confident prediction — recording skipped"
    return {"record_on_chain": decision, "entropy": round(entropy, 4), "reason": reason}


def load_artifacts(model_dir=MODEL_DIR):
    production_path = os.path.join(model_dir, "production_model.joblib")
    fallback_path = os.path.join(model_dir, "random_forest.joblib")
    model_path = production_path if os.path.exists(production_path) else fallback_path
    model = joblib.load(model_path)
    baselines = joblib.load(os.path.join(model_dir, "meter_baselines.joblib"))
    with open(os.path.join(model_dir, "metrics.json")) as f:
        metadata = json.load(f)
    return model, baselines, metadata


def risk_level_from_score(score: int) -> str:
    if score <= RISK_THRESHOLDS["LOW_MAX"]:
        return "LOW"
    elif score <= RISK_THRESHOLDS["MEDIUM_MAX"]:
        return "MEDIUM"
    return "HIGH"


def generate_explanation(feature_values: dict, is_suspicious: bool) -> dict:
    """Transparent RULE-BASED explanation layer generated from the actual
    feature values of this reading. Explicitly NOT presented as a direct
    SHAP/model-native explanation — labeled as rule-based per project spec
    section 12."""
    reasons = []

    hist_dev = feature_values.get("historical_deviation", 0)
    if hist_dev <= -50:
        reasons.append(f"Consumption is {abs(hist_dev):.0f}% below this meter's historical average.")
    elif hist_dev >= 80:
        reasons.append(f"Consumption is {hist_dev:.0f}% above this meter's historical average.")

    change_pct = feature_values.get("consumption_change_percentage", 0)
    if abs(change_pct) >= 60:
        direction = "dropped" if change_pct < 0 else "increased"
        reasons.append(f"Consumption {direction} by {abs(change_pct):.0f}% from the previous reading.")

    current_dev = feature_values.get("current_deviation", 0)
    if abs(current_dev) >= 1.5:
        reasons.append("Current draw is significantly different from the meter's normal range, "
                        "inconsistent with the reported consumption.")

    voltage_dev = feature_values.get("voltage_deviation", 0)
    if abs(voltage_dev) >= 15:
        reasons.append("Voltage reading deviates abnormally from the expected supply range.")

    night_consumption = feature_values.get("night_consumption", 0)
    daily_consumption = feature_values.get("daily_consumption", 1e-6)
    if daily_consumption > 0 and (night_consumption / daily_consumption) >= 0.55:
        reasons.append("Night-time consumption pattern is unusually high relative to the rest of the day.")

    rolling_std = feature_values.get("rolling_std", 0)
    rolling_avg = feature_values.get("rolling_average", 1e-6)
    if rolling_avg > 0 and (rolling_std / rolling_avg) >= 0.9:
        reasons.append("Recent readings show erratic, unstable fluctuation rather than a normal usage pattern.")

    if not reasons and is_suspicious:
        reasons.append("Model flagged this reading based on a combination of feature values "
                        "that did not individually cross a single rule threshold.")
    if not reasons:
        reasons.append("No individual factors exceeded configured deviation thresholds.")

    return {
        "type": "rule_based",  # explicitly labeled, per academic-honesty requirement
        "factors": reasons,
    }


def predict_reading(reading: dict, history_df: pd.DataFrame, meter_id: str,
                     model_dir=MODEL_DIR) -> dict:
    """
    reading: dict with keys meter_id, timestamp, consumption_kwh, voltage,
             current, power_factor
    history_df: recent readings for this meter (for rolling-window features)
    """
    model, baselines, metadata = load_artifacts(model_dir)
    baseline = baselines.get(meter_id)
    if baseline is None:
        raise ValueError(f"No historical baseline found for meter {meter_id}. "
                          f"Meter must have prior readings before prediction.")

    feature_values = build_features_single(reading, history_df, baseline)
    X = pd.DataFrame([feature_values])[FEATURE_COLUMNS].fillna(0)

    fraud_probability = float(model.predict_proba(X)[0][1])
    risk_score = int(round(fraud_probability * 100))
    risk_level = risk_level_from_score(risk_score)
    prediction = "SUSPICIOUS" if risk_level == "HIGH" or fraud_probability >= 0.5 else "NORMAL"

    explanation = generate_explanation(feature_values, is_suspicious=(prediction == "SUSPICIOUS"))
    blockchain_decision = should_record_on_chain(risk_score, fraud_probability)

    return {
        "meter_id": meter_id,
        "fraud_probability": round(fraud_probability, 5),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": prediction,
        "model_version": metadata["model_version"],
        "explanation": explanation,
        "blockchain_decision": blockchain_decision,
        "feature_values": feature_values,
    }


if __name__ == "__main__":
    # Quick smoke test using real generated data
    readings = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "readings.csv"),
                            parse_dates=["timestamp"])
    meter_id = readings["meter_id"].unique()[0]
    meter_hist = readings[readings["meter_id"] == meter_id].sort_values("timestamp")

    # Take a normal-looking early reading and re-score it as "new"
    history = meter_hist.iloc[:-1]
    last_row = meter_hist.iloc[-1]
    new_reading = {
        "meter_id": meter_id,
        "timestamp": last_row["timestamp"],
        "consumption_kwh": last_row["consumption_kwh"],
        "voltage": last_row["voltage"],
        "current": last_row["current"],
        "power_factor": last_row["power_factor"],
    }
    result = predict_reading(new_reading, history, meter_id)
    print(json.dumps({k: v for k, v in result.items() if k != "feature_values"}, indent=2))
