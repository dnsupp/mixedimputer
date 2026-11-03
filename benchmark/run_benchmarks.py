"""Headless benchmark runner for the tables in BENCHMARKS.md.

Run from the repository root:

    python benchmark/run_benchmarks.py

This script intentionally does not import Plotly/Jupyter.  The notebooks remain
useful for interactive exploration; this script is the reproducible,
CI-friendly path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from mixedimputer import DataCorrupter, MixedImputer, TimeSeriesImputer


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _rmse(true_values, predicted_values) -> float:
    return float(
        np.sqrt(mean_squared_error(true_values, predicted_values))
    )


def run_titanic(max_iter: int = 10, seed: int = 42):
    df = pd.read_csv(DATA / "titanic.csv")
    df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"]).copy()
    numeric = ["Age", "Fare"]
    nominal = ["Sex", "Embarked"]
    columns = numeric + nominal

    rows = []
    for fraction in (0.01, 0.10, 0.30):
        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=fraction,
            numeric_columns=numeric,
            nominal_columns=nominal,
            random_state=seed,
        )
        corrupted, mask, original = corrupter.corrupt(df)
        imputed = MixedImputer(
            max_iter=max_iter, random_state=seed
        ).fit_transform(corrupted)

        row = {"fraction": fraction}
        for col in numeric:
            m = mask[col]
            if m.sum() == 0:
                continue
            truth = original.loc[m, col].to_numpy(dtype=float)
            pred = imputed.loc[m, col].to_numpy(dtype=float)
            observed_mean = corrupted[col].mean()
            row[col] = {
                "n": int(m.sum()),
                "rmse": _rmse(truth, pred),
                "mean_rmse": _rmse(
                    truth, np.full_like(truth, observed_mean)
                ),
            }
        for col in nominal:
            m = mask[col]
            if m.sum() == 0:
                continue
            truth = original.loc[m, col].astype(str)
            pred = imputed.loc[m, col].astype(str)
            mode = corrupted[col].mode().iloc[0]
            row[col] = {
                "n": int(m.sum()),
                "accuracy": float(accuracy_score(truth, pred)),
                "mode_accuracy": float(
                    accuracy_score(truth, np.full(len(truth), mode))
                ),
            }
        rows.append(row)
    return rows


def run_gpu(max_iter: int = 10, seed: int = 42):
    raw = pd.read_csv(
        DATA / "gpu_sensor_log.txt", skipinitialspace=True, encoding="latin1"
    )
    raw.columns = raw.columns.str.strip()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["time_sec"] = (
        raw["Date"] - raw["Date"].iloc[0]
    ).dt.total_seconds().astype(float)

    targets = ["GPU Temperature [C]", "Fan Speed (RPM) [RPM]"]
    df = raw[["time_sec"] + targets].copy()

    rows = []
    for fraction in (0.01, 0.10, 0.30):
        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=fraction,
            numeric_columns=targets,
            random_state=seed,
        )
        corrupted, mask, original = corrupter.corrupt(df)
        imputed = TimeSeriesImputer(
            time_column="time_sec",
            lags=[1, 2, 3, 5],
            rolling_windows=[10, 30],
            initial_strategy="forward_fill",
            max_iter=max_iter,
            random_state=seed,
        ).fit_transform(corrupted)

        ffill = corrupted.copy()
        mean_fill = corrupted.copy()
        for col in targets:
            ffill[col] = ffill[col].ffill().bfill()
            mean_fill[col] = mean_fill[col].fillna(corrupted[col].mean())

        row = {"fraction": fraction}
        for col in targets:
            m = mask[col]
            if m.sum() == 0:
                continue
            truth = original.loc[m, col].to_numpy(dtype=float)
            row[col] = {
                "n": int(m.sum()),
                "rmse": _rmse(
                    truth, imputed.loc[m, col].to_numpy(dtype=float)
                ),
                "ffill_rmse": _rmse(
                    truth, ffill.loc[m, col].to_numpy(dtype=float)
                ),
                "mean_rmse": _rmse(
                    truth, mean_fill.loc[m, col].to_numpy(dtype=float)
                ),
            }
        rows.append(row)
    return rows


def _print_rows(title, rows):
    print(f"\n## {title}")
    for row in rows:
        print(f"\n{row['fraction']:.0%} corruption")
        for key, value in row.items():
            if key == "fraction":
                continue
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _print_rows(
        "Titanic (MixedImputer)",
        run_titanic(max_iter=args.max_iter, seed=args.seed),
    )
    _print_rows(
        "GPU sensor log (TimeSeriesImputer)",
        run_gpu(max_iter=args.max_iter, seed=args.seed),
    )


if __name__ == "__main__":
    main()
