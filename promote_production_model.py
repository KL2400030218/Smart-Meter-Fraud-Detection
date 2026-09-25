"""
promote_production_model.py
============================
train.py originally selected Random Forest as the production model
(models/random_forest.joblib, loaded by predict.py) before the expanded
8-model comparison in train_v2.py existed. That comparison found the
Proposed Model (soft-voting ensemble of Random Forest + XGBoost +
LightGBM) has the best F1 and Recall of all 8 candidates
(model_comparison.md), which matters most for fraud detection since a
missed fraud case is costlier than a slightly slower prediction.

This script re-trains that Proposed Model on the SAME train/test split
train.py used (same seed, same 75/25 split) so its test-set metrics are
directly comparable to the original single-model numbers, saves it as
models/production_model.joblib, and updates metrics.json so the
ML Model page and backend both reflect the new production model.

predict.py is updated separately to prefer production_model.joblib when
present, falling back to random_forest.joblib for backward
compatibility.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocess import build_features, FEATURE_COLUMNS

PRODUCTION_MODEL_VERSION = "v2.0.0-proposed-ensemble"


def main(data_dir="data", model_dir="models", seed=42):
    readings = pd.read_csv(os.path.join(data_dir, "readings.csv"), parse_dates=["timestamp"])
    features_df, baselines = build_features(readings)

    X = features_df[FEATURE_COLUMNS].fillna(0)
    y = features_df["is_fraud"].astype(int)

    # Same split as train.py, so this test score is directly comparable
    # to the original single-model Random Forest number in PROGRESS.md.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                 random_state=seed, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=seed, n_jobs=-1, verbose=-1)

    proposed = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)], voting="soft", n_jobs=-1,
    )
    proposed.fit(X_train, y_train)

    y_pred = proposed.predict(X_test)
    y_prob = proposed.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print("Proposed Model test-set performance (same 75/25 split as train.py):")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix: {test_metrics['confusion_matrix']}")

    # Refit on ALL data for the artifact that actually gets deployed
    # (standard practice: cross-validation/held-out test already proved
    # generalization; the shipped model uses every available row).
    proposed_full = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)], voting="soft", n_jobs=-1,
    )
    proposed_full.fit(X, y)

    joblib.dump(proposed_full, os.path.join(model_dir, "production_model.joblib"))
    joblib.dump(baselines, os.path.join(model_dir, "meter_baselines.joblib"))  # keep in sync

    # Feature importance isn't natively defined for a VotingClassifier;
    # report the underlying Random Forest's, labeled honestly as such.
    rf_importance = dict(zip(FEATURE_COLUMNS, proposed_full.named_estimators_["rf"].feature_importances_.tolist()))

    metadata = {
        "model_version": PRODUCTION_MODEL_VERSION,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "selected_model": "Proposed Model (RF+XGB+LGBM soft-voting)",
        "selection_rationale": (
            "Selected from an 8-model, 5-fold cross-validated comparison "
            "(see docs/architecture/model_comparison.md) for having the "
            "best F1 (0.9965) and Recall (0.9957) of all candidates. "
            "Latency (~18ms/prediction) and storage (~4.3MB) are higher "
            "than lighter alternatives (XGBoost/LightGBM alone) but both "
            "remain well within acceptable bounds for this system's "
            "single-reading-at-a-time API pattern."
        ),
        "results": {"Proposed Model (RF+XGB+LGBM soft-voting)": test_metrics},
        "feature_importance_random_forest": rf_importance,
        "class_balance": {"fraud": int(y.sum()), "normal": int((y == 0).sum())},
        "full_8_model_comparison": "see models/model_comparison_v2.json (5-fold CV)",
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models/production_model.joblib and updated models/metrics.json")
    print(f"model_version: {PRODUCTION_MODEL_VERSION}")
    return metadata


if __name__ == "__main__":
    main()
