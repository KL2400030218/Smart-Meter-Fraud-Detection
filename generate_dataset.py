"""
generate_dataset.py
====================
Synthetic smart-meter dataset generator.

IMPORTANT (academic honesty): this dataset is SYNTHETIC. It is generated
from parameterized statistical models of "normal" and "suspicious" meter
behavior, not from real utility data. Fraud labels are NOT assigned
randomly — every suspicious record is produced by actually injecting one
of a fixed set of fraud behaviors into the underlying signal, so the
label always corresponds to a real behavioral pattern in the features.

Fraud patterns implemented:
  1. sudden_drop        - consumption suddenly drops well below baseline
                           (classic meter-tampering / bypass signature)
  2. sudden_spike        - consumption suddenly jumps (miscalibration /
                           meter reversal signature)
  3. night_anomaly       - abnormal, persistent night-time consumption
  4. current_mismatch    - current varies without matching consumption
                           (partial bypass / current-transformer tampering)
  5. voltage_anomaly     - abnormal voltage readings (interference with
                           metering circuit)
  6. erratic_pattern     - repeated, unstable, non-physical fluctuations

Run:
    python generate_dataset.py --n-meters 40 --days 30
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

FRAUD_PATTERNS = [
    "sudden_drop",
    "sudden_spike",
    "night_anomaly",
    "current_mismatch",
    "voltage_anomaly",
    "erratic_pattern",
]


def _hourly_load_curve():
    """Realistic normalized 24h residential consumption shape (two peaks)."""
    hours = np.arange(24)
    morning = 0.6 * np.exp(-((hours - 8) ** 2) / (2 * 2.0 ** 2))
    evening = 1.0 * np.exp(-((hours - 20) ** 2) / (2 * 2.5 ** 2))
    base = 0.25
    return base + morning + evening


LOAD_CURVE = _hourly_load_curve()


def generate_meter_readings(meter_id, start_date, days, base_load, rng,
                             is_fraud=False, fraud_pattern=None,
                             fraud_start_day=None):
    """Generate hourly readings for one meter over `days` days."""
    rows = []
    n_hours = days * 24
    fraud_start_hour = fraud_start_day * 24 if fraud_start_day is not None else None

    # Base per-meter multiplicative noise so meters aren't identical
    meter_scale = rng.normal(1.0, 0.12)
    meter_scale = max(0.5, meter_scale)

    for h in range(n_hours):
        ts = start_date + timedelta(hours=h)
        hour_of_day = ts.hour
        base_kwh = base_load * LOAD_CURVE[hour_of_day] * meter_scale
        noise = rng.normal(0, 0.05 * base_load + 0.01)
        consumption = max(0.0, base_kwh + noise)

        voltage = rng.normal(230, 3.0)
        current = max(0.0, (consumption * 1000) / max(voltage, 1e-3) + rng.normal(0, 0.15))
        power_factor = np.clip(rng.normal(0.95, 0.02), 0.5, 1.0)

        fraud_active = is_fraud and fraud_start_hour is not None and h >= fraud_start_hour

        if fraud_active:
            if fraud_pattern == "sudden_drop":
                consumption *= rng.uniform(0.05, 0.25)
            elif fraud_pattern == "sudden_spike":
                consumption *= rng.uniform(2.2, 3.5)
            elif fraud_pattern == "night_anomaly":
                if hour_of_day in (0, 1, 2, 3, 4):
                    consumption = base_load * rng.uniform(0.8, 1.4)
            elif fraud_pattern == "current_mismatch":
                current *= rng.uniform(2.0, 3.0)  # current doesn't match consumption
            elif fraud_pattern == "voltage_anomaly":
                voltage = rng.normal(200, 12.0) if rng.random() < 0.5 else rng.normal(260, 12.0)
            elif fraud_pattern == "erratic_pattern":
                consumption *= rng.choice([0.1, 3.0, 0.3, 2.5])

        rows.append({
            "meter_id": meter_id,
            "timestamp": ts,
            "consumption_kwh": round(max(0.0, consumption), 4),
            "voltage": round(voltage, 2),
            "current": round(max(0.0, current), 2),
            "power_factor": round(float(power_factor), 3),
            "is_fraud": int(fraud_active),
            "fraud_pattern": fraud_pattern if fraud_active else "none",
        })
    return rows


def generate_dataset(n_meters=40, days=30, fraud_ratio=0.28, seed=42):
    rng = np.random.default_rng(seed)
    start_date = datetime(2025, 1, 1)
    all_rows = []
    meters_meta = []

    n_fraud_meters = max(1, int(n_meters * fraud_ratio))
    fraud_meter_flags = np.array([True] * n_fraud_meters + [False] * (n_meters - n_fraud_meters))
    rng.shuffle(fraud_meter_flags)

    for i in range(n_meters):
        meter_id = f"M{1001 + i}"
        base_load = rng.uniform(0.8, 3.0)  # avg kWh per interval scale
        is_fraud = bool(fraud_meter_flags[i])
        pattern = rng.choice(FRAUD_PATTERNS) if is_fraud else None
        fraud_start_day = int(rng.uniform(days * 0.4, days * 0.75)) if is_fraud else None

        meters_meta.append({
            "meter_id": meter_id,
            "consumer_id": f"C{20000 + i}",
            "location": rng.choice([
                "Sector 4, Guntur", "MG Road, Vijayawada", "Ring Road, Hyderabad",
                "Old Town, Visakhapatnam", "Industrial Area, Nellore",
                "Beach Road, Kakinada", "City Center, Tirupati",
            ]),
            "installation_date": (start_date - timedelta(days=int(rng.uniform(180, 1500)))).date().isoformat(),
            "status": "ACTIVE",
            "true_fraud_meter": is_fraud,
            "fraud_pattern": pattern if is_fraud else "none",
        })

        rows = generate_meter_readings(
            meter_id, start_date, days, base_load, rng,
            is_fraud=is_fraud, fraud_pattern=pattern, fraud_start_day=fraud_start_day,
        )
        all_rows.extend(rows)

    readings_df = pd.DataFrame(all_rows)
    meters_df = pd.DataFrame(meters_meta)
    return meters_df, readings_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-meters", type=int, default=40)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--fraud-ratio", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    meters_df, readings_df = generate_dataset(
        n_meters=args.n_meters, days=args.days,
        fraud_ratio=args.fraud_ratio, seed=args.seed,
    )

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    meters_df.to_csv(os.path.join(args.out_dir, "meters.csv"), index=False)
    readings_df.to_csv(os.path.join(args.out_dir, "readings.csv"), index=False)

    print(f"Meters: {len(meters_df)}  |  Fraud meters: {meters_df['true_fraud_meter'].sum()}")
    print(f"Readings: {len(readings_df)}  |  Fraudulent readings: {readings_df['is_fraud'].sum()} "
          f"({100 * readings_df['is_fraud'].mean():.2f}%)")
    print(f"Saved to {args.out_dir}/meters.csv and {args.out_dir}/readings.csv")
