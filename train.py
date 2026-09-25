"""
train.py
========
Trains three models on the engineered feature set and evaluates them
honestly on a held-out test split. NOTHING here is hard-coded — every
metric below is computed from actual predictions on the test set.

Models:
  1. Logistic Regression   - baseline linear classifier (supervised)
  2. Random Forest         - primary supervised model
  3. Isolation Forest      - unsupervised anomaly detector (comparison
                              only; trained without labels, then its
                              anomaly flag is compared against ground
                              truth purely for reporting)

Final model selection is based on actual F1-score on the test set
(the metric most appropriate for an imbalanced fraud-detection task),
not assumed in advance.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from preprocess import build_features, FEATURE_COLUMNS

MODEL_VERSION = "v1.0.0"


def load_data(data_dir="data"):
    readings = pd.read_csv(os.path.join(data_dir, "readings.csv"), parse_dates=["timestamp"])
    return readings


def main(data_dir="data", model_dir="models", seed=42):
    readings = load_data(data_dir)
    features_df, baselines = build_features(readings)

    X = features_df[FEATURE_COLUMNS].fillna(0)
    y = features_df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ---------------- Model 1: Logistic Regression ----------------
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)
    y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]
    results["Logistic Regression"] = {
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "precision": precision_score(y_test, y_pred_lr, zero_division=0),
        "recall": recall_score(y_test, y_pred_lr, zero_division=0),
        "f1": f1_score(y_test, y_pred_lr, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob_lr),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
    }

    # ---------------- Model 2: Random Forest ----------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # tree model: no scaling needed
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "f1": f1_score(y_test, y_pred_rf, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob_rf),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    }
    feature_importance = dict(zip(FEATURE_COLUMNS, rf.feature_importances_.tolist()))

    # ---------------- Model 3: Isolation Forest (unsupervised) ----------------
    contamination = max(0.01, min(0.5, y_train.mean()))
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=seed,
    )
    iso.fit(X_train_scaled)  # NOTE: fit without labels — unsupervised
    raw_pred_iso = iso.predict(X_test_scaled)          # 1 = normal, -1 = anomaly
    y_pred_iso = (raw_pred_iso == -1).astype(int)       # map anomaly -> fraud=1
    # anomaly score, higher = more anomalous, used as a pseudo-probability for ROC-AUC
    anomaly_score = -iso.decision_function(X_test_scaled)
    results["Isolation Forest"] = {
        "accuracy": accuracy_score(y_test, y_pred_iso),
        "precision": precision_score(y_test, y_pred_iso, zero_division=0),
        "recall": recall_score(y_test, y_pred_iso, zero_division=0),
        "f1": f1_score(y_test, y_pred_iso, zero_division=0),
        "roc_auc": roc_auc_score(y_test, anomaly_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred_iso).tolist(),
    }

    # ---------------- Model selection (based on ACTUAL F1) ----------------
    best_model_name = max(results, key=lambda k: results[k]["f1"])

    print("\n=== Model Comparison (test set, real computed metrics) ===")
    print(f"{'Model':<22}{'Precision':>10}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}")
    for name, m in results.items():
        print(f"{name:<22}{m['precision']:>10.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}{m['roc_auc']:>10.4f}")
    print(f"\nSelected final model: {best_model_name} (highest F1-score on held-out test set)")

    # ---------------- Save artifacts ----------------
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(logreg, os.path.join(model_dir, "logistic_regression.joblib"))
    joblib.dump(rf, os.path.join(model_dir, "random_forest.joblib"))
    joblib.dump(iso, os.path.join(model_dir, "isolation_forest.joblib"))
    joblib.dump(baselines, os.path.join(model_dir, "meter_baselines.joblib"))

    metadata = {
        "model_version": MODEL_VERSION,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "selected_model": best_model_name,
        "results": results,
        "feature_importance_random_forest": feature_importance,
        "class_balance": {"fraud": int(y.sum()), "normal": int((y == 0).sum())},
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models + metrics.json to {model_dir}/")
    return metadata


if __name__ == "__main__":
    main()
