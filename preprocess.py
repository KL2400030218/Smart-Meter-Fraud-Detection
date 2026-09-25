"""
preprocess.py
=============
Reusable feature-engineering pipeline used identically at TRAINING time
(train.py) and SERVING time (predict.py) to avoid training-serving skew.

Given a per-meter time-ordered readings dataframe, produces the feature
set described in the spec:

    consumption_change_percentage
    rolling_average
    rolling_std
    daily_consumption
    night_consumption
    peak_consumption
    off_peak_consumption
    historical_deviation
    voltage_deviation
    current_deviation
    power_factor_deviation

`historical_deviation`, `voltage_deviation`, `current_deviation`, and
`power_factor_deviation` are computed relative to a per-meter historical
baseline (mean of the first ROLL_WINDOW*2 readings, treated as the
"trusted" early history) so the same transform can be applied to a
single new reading at prediction time by supplying that meter's stored
baseline.
"""
import numpy as np
import pandas as pd

ROLL_WINDOW = 24          # 1 day of hourly readings
NIGHT_HOURS = {0, 1, 2, 3, 4}
PEAK_HOURS = {18, 19, 20, 21}

# Physically plausible bounds for a low-voltage residential smart meter.
# Used by clean_readings() to catch sensor glitches/corrupt rows before
# they reach feature engineering (the "Data Cleaning" pipeline stage).
VALID_RANGES = {
    "consumption_kwh": (0.0, 50.0),
    "voltage": (150.0, 300.0),
    "current": (0.0, 120.0),
    "power_factor": (0.0, 1.0),
}


def clean_readings(readings_df: pd.DataFrame) -> pd.DataFrame:
    """Data Cleaning stage: drops duplicate (meter_id, timestamp) rows,
    coerces reading columns to numeric, clips/rejects physically
    implausible sensor values, and fills isolated missing values via
    per-meter forward-fill (so a single dropped reading doesn't break
    the rolling-window features). Returns a cleaned copy; never mutates
    the input in place.

    This is a deterministic, rule-based step — no model is involved —
    which is why it runs before feature engineering rather than as part
    of it.
    """
    df = readings_df.copy()

    # 1. Coerce to numeric; anything unparseable becomes NaN
    numeric_cols = ["consumption_kwh", "voltage", "current", "power_factor"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Drop exact duplicate (meter_id, timestamp) rows — keep first
    if {"meter_id", "timestamp"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["meter_id", "timestamp"], keep="first")

    # 3. Reject physically implausible values (set to NaN, then fill)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            out_of_range = (df[col] < lo) | (df[col] > hi)
            df.loc[out_of_range, col] = np.nan

    # 4. Per-meter forward-fill then backward-fill for any remaining NaNs
    #    (isolated sensor dropouts) — never fill across different meters.
    if "meter_id" in df.columns:
        df[numeric_cols] = (
            df.groupby("meter_id")[numeric_cols]
            .apply(lambda g: g.ffill().bfill())
            .reset_index(level=0, drop=True)
        )
    else:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # 5. Any row still NaN after fill (e.g. a meter with zero valid
    #    readings) can't be used — drop it and report how many.
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    dropped = before - len(df)
    if dropped:
        df.attrs["rows_dropped_by_cleaning"] = dropped

    return df.reset_index(drop=True)


FEATURE_COLUMNS = [
    "consumption_change_percentage",
    "rolling_average",
    "rolling_std",
    "daily_consumption",
    "night_consumption",
    "peak_consumption",
    "off_peak_consumption",
    "historical_deviation",
    "voltage_deviation",
    "current_deviation",
    "power_factor_deviation",
]


def compute_meter_baseline(meter_df: pd.DataFrame) -> dict:
    """Baseline statistics from a meter's earliest trusted history.
    Used both to build training features and to be stored/looked-up
    at serving time for single-reading predictions."""
    early = meter_df.sort_values("timestamp").iloc[: ROLL_WINDOW * 2]
    return {
        "avg_consumption": float(early["consumption_kwh"].mean()),
        "avg_voltage": float(early["voltage"].mean()),
        "avg_current": float(early["current"].mean()),
        "avg_power_factor": float(early["power_factor"].mean()),
    }


def build_features_for_meter(meter_df: pd.DataFrame, baseline: dict = None) -> pd.DataFrame:
    """Compute the full feature set for one meter's time-ordered readings."""
    df = meter_df.sort_values("timestamp").reset_index(drop=True).copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    if baseline is None:
        baseline = compute_meter_baseline(df)

    # consumption_change_percentage: vs previous reading
    prev = df["consumption_kwh"].shift(1)
    df["consumption_change_percentage"] = (
        (df["consumption_kwh"] - prev) / prev.replace(0, np.nan)
    ).fillna(0) * 100

    # rolling stats
    df["rolling_average"] = df["consumption_kwh"].rolling(ROLL_WINDOW, min_periods=1).mean()
    df["rolling_std"] = df["consumption_kwh"].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)

    # daily / night / peak / off-peak consumption (same-day aggregate, broadcast to each row)
    daily = df.groupby("date")["consumption_kwh"].transform("sum")
    df["daily_consumption"] = daily

    night_mask = df["hour"].isin(NIGHT_HOURS)
    df["night_consumption"] = np.where(night_mask, df["consumption_kwh"], 0.0)
    night_daily = df.groupby("date")["night_consumption"].transform("sum")
    df["night_consumption"] = night_daily

    peak_mask = df["hour"].isin(PEAK_HOURS)
    df["peak_consumption"] = np.where(peak_mask, df["consumption_kwh"], 0.0)
    peak_daily = df.groupby("date")["peak_consumption"].transform("sum")
    df["peak_consumption"] = peak_daily

    off_peak_mask = ~df["hour"].isin(PEAK_HOURS) & ~df["hour"].isin(NIGHT_HOURS)
    df["off_peak_consumption"] = np.where(off_peak_mask, df["consumption_kwh"], 0.0)
    off_peak_daily = df.groupby("date")["off_peak_consumption"].transform("sum")
    df["off_peak_consumption"] = off_peak_daily

    # deviations from meter's own historical baseline
    avg_c = baseline["avg_consumption"] or 1e-6
    df["historical_deviation"] = (df["consumption_kwh"] - avg_c) / avg_c * 100
    df["voltage_deviation"] = df["voltage"] - baseline["avg_voltage"]
    df["current_deviation"] = df["current"] - baseline["avg_current"]
    df["power_factor_deviation"] = df["power_factor"] - baseline["avg_power_factor"]

    return df


def build_features(readings_df: pd.DataFrame, baselines: dict = None, skip_cleaning: bool = False) -> tuple[pd.DataFrame, dict]:
    """Build features for ALL meters. Returns (feature_df, baselines_used).
    Runs clean_readings() first unless skip_cleaning=True (predict.py's
    single-reading path does its own targeted cleaning upstream)."""
    if not skip_cleaning:
        readings_df = clean_readings(readings_df)
    baselines = baselines or {}
    out_frames = []
    for meter_id, g in readings_df.groupby("meter_id"):
        baseline = baselines.get(meter_id) or compute_meter_baseline(g)
        baselines[meter_id] = baseline
        out_frames.append(build_features_for_meter(g, baseline))
    result = pd.concat(out_frames, ignore_index=True)
    return result, baselines


def build_features_single(reading: dict, history_df: pd.DataFrame, baseline: dict) -> dict:
    """Build the same feature set for ONE new incoming reading, given that
    meter's recent history (for rolling stats) and stored baseline
    (for deviation features). Used by predict.py at serving time so the
    transform is provably identical to training."""
    combined = pd.concat([
        history_df,
        pd.DataFrame([reading]),
    ], ignore_index=True)
    features_df = build_features_for_meter(combined, baseline)
    return features_df.iloc[-1][FEATURE_COLUMNS].to_dict()
