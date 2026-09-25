"""
train_v2.py
===========
Expanded model comparison, built in response to faculty review feedback:

  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. SVM (RBF kernel)
  5. XGBoost
  6. LightGBM
  7. Neural Network (MLP)
  8. Proposed Model — a soft-voting ensemble of Random Forest + XGBoost +
     LightGBM (the three strongest individual tree-based learners),
     averaging their class-1 probabilities. This is "proposed" in the
     sense that it is this project's own combination, not a named
     off-the-shelf single algorithm.

Evaluation protocol: **5-fold stratified cross-validation** (preferred
over a single 70/15/15 split per the feedback) over the full engineered
feature set. For every model and every fold we record:

  - F1-score, Recall, Precision, Accuracy, ROC-AUC   (mean +/- std across folds)
  - Per-sample inference latency in milliseconds       (mean +/- std)
  - Fitted model's on-disk size in KB                  (storage cost)

This does NOT replace train.py — train.py still produces the single
selected production model (models/random_forest.joblib) that predict.py
and the backend load. train_v2.py produces the comparison table used in
the report and viva (models/model_comparison_v2.json).
"""
import json
import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocess import build_features, FEATURE_COLUMNS

warnings.filterwarnings("ignore")

N_FOLDS = 5
RANDOM_STATE = 42


def load_data(data_dir="data"):
    readings = pd.read_csv(os.path.join(data_dir, "readings.csv"), parse_dates=["timestamp"])
    return readings


def build_models(random_state=RANDOM_STATE):
    """Returns dict[name] -> (estimator, needs_scaling: bool)."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                 random_state=random_state, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         eval_metric="logloss", random_state=random_state, n_jobs=-1,
                         verbosity=0)
    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=random_state, n_jobs=-1, verbose=-1)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state), True),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=random_state), False),
        "Random Forest": (rf, False),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state), True),
        "XGBoost": (xgb, False),
        "LightGBM": (lgbm, False),
        "Neural Network (MLP)": (
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=random_state), True),
        "Proposed Model (RF+XGB+LGBM soft-voting)": (
            VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
                voting="soft", n_jobs=-1,
            ), False),
    }
    return models


def measure_inference_latency(model, X_sample, n_repeats=200):
    """Average per-sample inference latency in milliseconds, measured by
    timing single-row predict_proba calls (matches how predict.py scores
    one incoming reading at a time in production)."""
    row = X_sample.iloc[[0]]
    # warm-up
    for _ in range(5):
        model.predict_proba(row)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def model_storage_kb(model) -> float:
    tmp_path = "/tmp/_model_size_check.joblib"
    joblib.dump(model, tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)
    return round(size_kb, 1)


def main(data_dir="data", out_path="models/model_comparison_v2.json"):
    readings = load_data(data_dir)
    features_df, _ = build_features(readings)
    X = features_df[FEATURE_COLUMNS].fillna(0)
    y = features_df["is_fraud"].astype(int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    models = build_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, (estimator, needs_scaling) in models.items():
        print(f"\n=== {name} ===")
        data = X_scaled if needs_scaling else X
        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
        fold_latency = []

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(data, y), 1):
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            import sklearn.base
            model = sklearn.base.clone(estimator)
            t_fit0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t_fit0

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))

            lat_mean, _ = measure_inference_latency(model, X_test, n_repeats=50)
            fold_latency.append(lat_mean)

            print(f"  fold {fold_i}/{N_FOLDS}: F1={fold_metrics['f1'][-1]:.4f} "
                  f"Recall={fold_metrics['recall'][-1]:.4f} fit_time={fit_time:.2f}s "
                  f"latency={lat_mean:.3f}ms")

        final_model = sklearn.base.clone(estimator)
        final_model.fit(data, y)
        storage_kb = model_storage_kb(final_model)

        results[name] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in fold_metrics.items()
        }
        results[name]["latency_ms"] = {
            "mean": float(np.mean(fold_latency)), "std": float(np.std(fold_latency))
        }
        results[name]["storage_kb"] = storage_kb

    summary = {
        "evaluation_protocol": f"{N_FOLDS}-fold stratified cross-validation",
        "n_samples": int(len(X)),
        "n_features": len(FEATURE_COLUMNS),
        "class_balance": {"fraud": int(y.sum()), "normal": int((y == 0).sum())},
        "results": results,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n=== SUMMARY (mean across 5 folds) ===")
    print(f"{'Model':<42}{'F1':>10}{'Recall':>10}{'ROC-AUC':>10}{'Latency(ms)':>13}{'Size(KB)':>10}")
    for name, r in results.items():
        print(f"{name:<42}{r['f1']['mean']:>10.4f}{r['recall']['mean']:>10.4f}"
              f"{r['roc_auc']['mean']:>10.4f}{r['latency_ms']['mean']:>13.3f}{r['storage_kb']:>10.1f}")

    print(f"\nSaved to {out_path}")
    return summary


if __name__ == "__main__":
    main()
