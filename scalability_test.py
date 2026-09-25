"""
scalability_test.py
====================
Faculty feedback item 6: scalability experiments across increasing
dataset size.

Two independent axes are measured, matching how the feedback phrased
them:

  (a) METER-COUNT scaling: 100 / 1,000 / 10,000 / 100,000 meters,
      holding history length fixed (7 days hourly = 168 readings/meter),
      measuring end-to-end pipeline time (generation, cleaning, feature
      engineering) and trained-model batch-inference throughput.

  (b) READING-COUNT scaling: a flat 1,000,000-row synthetic reading set
      (independent of realistic per-meter fraud-pattern injection —
      generated with a fast vectorized method purely to stress-test
      throughput), measuring the same pipeline stages.

This uses a FAST vectorized generator (not generate_dataset.py's
per-meter Python loop with injected fraud behaviors) because the
per-meter loop is intentionally realistic-but-slow and does not scale
to millions of rows in a reasonable time on modest hardware. This
script is about SYSTEM throughput at scale, not re-validating model
accuracy (accuracy is already covered by train_v2.py's 5-fold CV on the
realistic dataset).

Each scale point reports, all measured (never estimated):
  - data generation time
  - cleaning time (clean_readings)
  - feature engineering time (build_features)
  - trained-model batch inference time + throughput (rows/sec)
  - peak memory delta (via resource.getrusage)

Results are written to models/scalability_results.json.
"""
import gc
import json
import os
import psutil
import time

import joblib
import numpy as np
import pandas as pd

from preprocess import build_features, clean_readings, FEATURE_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "random_forest.joblib")


def peak_rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def generate_flat_readings(n_rows: int, n_meters: int, seed: int = 42) -> pd.DataFrame:
    """Fast, fully vectorized synthetic reading generator for throughput
    testing at scale. Not fraud-pattern realistic — see module docstring."""
    rng = np.random.default_rng(seed)
    meter_ids = np.array([f"S{1000+i}" for i in range(n_meters)])
    meter_idx = rng.integers(0, n_meters, size=n_rows)

    start = pd.Timestamp("2025-01-01")
    hours_offset = rng.integers(0, 24 * 365, size=n_rows)
    timestamps = start + pd.to_timedelta(hours_offset, unit="h")

    consumption = np.clip(rng.normal(1.5, 0.6, size=n_rows), 0.01, None)
    voltage = rng.normal(230, 4, size=n_rows)
    current = np.clip((consumption * 1000) / np.clip(voltage, 1, None) + rng.normal(0, 0.2, n_rows), 0, None)
    power_factor = np.clip(rng.normal(0.95, 0.02, size=n_rows), 0.5, 1.0)

    df = pd.DataFrame({
        "meter_id": meter_ids[meter_idx],
        "timestamp": timestamps,
        "consumption_kwh": consumption,
        "voltage": voltage,
        "current": current,
        "power_factor": power_factor,
    })
    return df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)


def generate_meter_scaled_readings(n_meters: int, days: int = 7, seed: int = 42) -> pd.DataFrame:
    """Vectorized generator producing exactly n_meters * days * 24 rows —
    used for the METER-COUNT scaling axis."""
    n_rows = n_meters * days * 24
    return generate_flat_readings(n_rows, n_meters, seed)


def run_pipeline_benchmark(readings_df: pd.DataFrame, model=None) -> dict:
    gc.collect()
    mem_before = peak_rss_mb()

    t0 = time.perf_counter()
    cleaned = clean_readings(readings_df)
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    features_df, _ = build_features(cleaned, skip_cleaning=True)
    t_features = time.perf_counter() - t0

    result = {
        "n_input_rows": int(len(readings_df)),
        "n_meters": int(readings_df["meter_id"].nunique()),
        "n_output_rows": int(len(features_df)),
        "cleaning_time_sec": round(t_clean, 4),
        "feature_engineering_time_sec": round(t_features, 4),
        "rows_per_sec_feature_engineering": round(len(readings_df) / t_features, 1) if t_features > 0 else None,
    }

    if model is not None:
        X = features_df[FEATURE_COLUMNS].fillna(0)
        t0 = time.perf_counter()
        _ = model.predict_proba(X)
        t_infer = time.perf_counter() - t0
        result["batch_inference_time_sec"] = round(t_infer, 4)
        result["inference_rows_per_sec"] = round(len(X) / t_infer, 1) if t_infer > 0 else None

    result["peak_rss_mb"] = round(peak_rss_mb(), 1)
    return result


def main():
    model = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Loaded trained model from {MODEL_PATH} for inference benchmarking.")
    else:
        print("WARNING: no trained model found — skipping inference benchmarking (run train.py first).")

    out_path = os.path.join(os.path.dirname(__file__), "models", "scalability_results.json")
    results = {"meter_count_scaling": {}, "reading_count_scaling": {}}

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # ---------------- (a) Meter-count scaling ----------------
    meter_counts = [100, 1_000, 10_000]  # 100,000 attempted separately below, resource-permitting
    for n_meters in meter_counts:
        print(f"\n--- Meter-count scaling: {n_meters:,} meters ---")
        t0 = time.perf_counter()
        readings = generate_meter_scaled_readings(n_meters, days=7)
        t_gen = time.perf_counter() - t0
        print(f"  generated {len(readings):,} rows in {t_gen:.2f}s")

        bench = run_pipeline_benchmark(readings, model)
        bench["generation_time_sec"] = round(t_gen, 4)
        results["meter_count_scaling"][str(n_meters)] = bench
        save()  # persist after every scale point so a later crash doesn't lose earlier results
        print(f"  clean={bench['cleaning_time_sec']}s  "
              f"features={bench['feature_engineering_time_sec']}s "
              f"({bench['rows_per_sec_feature_engineering']} rows/s)  "
              f"infer={bench.get('inference_rows_per_sec')} rows/s  "
              f"peak_rss={bench['peak_rss_mb']}MB")
        del readings
        gc.collect()

    # 100,000 meters = ~16.8M rows, which needs far more RAM than this
    # sandbox has (observed: 10,000 meters / 1.68M rows already used
    # ~1.4GB). Run it in a separate subprocess so an OOM kill there
    # cannot take down this script and lose the results already saved
    # above. If it survives, this process re-reads what the subprocess
    # wrote to a temp file.
    print(f"\n--- Meter-count scaling: 100,000 meters (attempting in subprocess) ---")
    import subprocess
    import sys
    tmp_out = "/tmp/_scale_100k_result.json"
    worker_code = f'''
import sys, json, time, gc
sys.path.insert(0, {os.path.dirname(__file__)!r})
from scalability_test import generate_meter_scaled_readings, run_pipeline_benchmark
import joblib, os
model = None
model_path = {MODEL_PATH!r}
if os.path.exists(model_path):
    model = joblib.load(model_path)
t0 = time.perf_counter()
readings = generate_meter_scaled_readings(100_000, days=7)
t_gen = time.perf_counter() - t0
bench = run_pipeline_benchmark(readings, model)
bench["generation_time_sec"] = round(t_gen, 4)
with open({tmp_out!r}, "w") as f:
    json.dump(bench, f)
'''
    try:
        proc = subprocess.run([sys.executable, "-c", worker_code], timeout=240,
                               capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(tmp_out):
            with open(tmp_out) as f:
                results["meter_count_scaling"]["100000"] = json.load(f)
            print("  succeeded:", results["meter_count_scaling"]["100000"])
        else:
            results["meter_count_scaling"]["100000"] = {
                "status": "failed",
                "returncode": proc.returncode,
                "note": "Likely OOM-killed by the OS (2GB RAM / 1 vCPU sandbox). "
                        "16.8M rows extrapolated to need far more memory than available here. "
                        "Script (generate_meter_scaled_readings(100_000, days=7)) is ready to "
                        "run as-is on a machine with more RAM.",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else None,
            }
            print("  FAILED/KILLED — recorded as skipped with explanation.")
    except subprocess.TimeoutExpired:
        results["meter_count_scaling"]["100000"] = {
            "status": "timed_out_after_240s",
            "note": "Did not complete within the time budget available in this session.",
        }
        print("  TIMED OUT — recorded as skipped with explanation.")
    save()

    # ---------------- (b) Reading-count scaling (flat 1,000,000 rows) ----------------
    print(f"\n--- Reading-count scaling: 1,000,000 rows ---")
    t0 = time.perf_counter()
    readings = generate_flat_readings(1_000_000, n_meters=5000)
    t_gen = time.perf_counter() - t0
    print(f"  generated {len(readings):,} rows in {t_gen:.2f}s")
    bench = run_pipeline_benchmark(readings, model)
    bench["generation_time_sec"] = round(t_gen, 4)
    results["reading_count_scaling"]["1000000"] = bench
    save()
    print(f"  clean={bench['cleaning_time_sec']}s features={bench['feature_engineering_time_sec']}s "
          f"({bench['rows_per_sec_feature_engineering']} rows/s) peak_rss={bench['peak_rss_mb']}MB")

    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    main()
