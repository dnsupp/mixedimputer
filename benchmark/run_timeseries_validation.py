"""Time-series imputation validation harness (Phase 1).

This is deliberately benchmark-only.  It does not change the package API.

What it compares
----------------
- ``ffill`` / ``bfill``
- linear interpolation
- observed mean
- a one-shot HistGradientBoosting model on lag features (no iterative wrapper)
- ``TimeSeriesImputer`` with no temporal features
- ``TimeSeriesImputer`` with lags only
- ``TimeSeriesImputer`` with lags + rolling features

Missingness
-----------
- point MCAR at 10% and 30%
- contiguous block gaps at 10% (block length 10)

The script prints a Markdown report and writes JSON with the raw records.  Use
``--quick`` for a fast smoke run; use ``--seeds 20`` for the full validation.

Run from the repository root:

    python benchmark/run_timeseries_validation.py --quick
    python benchmark/run_timeseries_validation.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression

from mixedimputer import MixedImputer, TimeSeriesImputer


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def synthetic_white_noise(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


def synthetic_ar1(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"time": np.arange(n, dtype=float)}
    for name, phi in (("x", 0.95), ("y", 0.8), ("z", 0.6)):
        values = np.empty(n, dtype=float)
        values[0] = rng.normal()
        noise = rng.normal(scale=0.5, size=n)
        for t in range(1, n):
            values[t] = phi * values[t - 1] + noise[t]
        data[name] = values
    return pd.DataFrame(data)


def synthetic_trend_seasonal(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    data = {
        "time": t,
        "x": 0.02 * t + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.4, size=n),
        "y": 0.01 * t + 3.0 * np.sin(2 * np.pi * t / 168) + rng.normal(scale=0.6, size=n),
        "z": 5.0 + 0.5 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3, size=n),
    }
    return pd.DataFrame(data)


def synthetic_random_walk(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "x": np.cumsum(rng.normal(scale=0.4, size=n)),
            "y": np.cumsum(rng.normal(scale=0.8, size=n)),
        }
    )


def synthetic_regime_shift(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    level = np.where(t < n / 3, 0.0, np.where(t < 2 * n / 3, 3.0, 1.0))
    return pd.DataFrame(
        {
            "time": t,
            "x": level + rng.normal(scale=0.5, size=n),
            "y": -level + 1.5 * np.sin(2 * np.pi * t / 48) + rng.normal(scale=0.5, size=n),
        }
    )


def synthetic_multivariate(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f1 = np.empty(n, dtype=float)
    f2 = np.empty(n, dtype=float)
    f1[0] = rng.normal()
    f2[0] = rng.normal()
    e1 = rng.normal(scale=0.5, size=n)
    e2 = rng.normal(scale=0.5, size=n)
    for t in range(1, n):
        f1[t] = 0.85 * f1[t - 1] + e1[t]
        f2[t] = 0.7 * f2[t - 1] + e2[t]
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "x": 0.8 * f1 + rng.normal(scale=0.2, size=n),
            "y": 0.5 * f1 + 0.5 * f2 + rng.normal(scale=0.2, size=n),
            "z": f2 + rng.normal(scale=0.2, size=n),
        }
    )


SYNTHETIC: Dict[str, Callable[[int, int], pd.DataFrame]] = {
    "white_noise": synthetic_white_noise,
    "ar1": synthetic_ar1,
    "trend_seasonal": synthetic_trend_seasonal,
    "random_walk": synthetic_random_walk,
    "regime_shift": synthetic_regime_shift,
    "multivariate": synthetic_multivariate,
}


def load_gpu_log() -> Optional[pd.DataFrame]:
    path = DATA / "gpu_sensor_log.txt"
    if not path.exists():
        return None
    raw = pd.read_csv(path, skipinitialspace=True, encoding="latin1")
    raw.columns = raw.columns.str.strip()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["time"] = (
        raw["Date"] - raw["Date"].iloc[0]
    ).dt.total_seconds().astype(float)
    targets = [
        "GPU Temperature [C]",
        "Fan Speed (RPM) [RPM]",
    ]
    if not all(c in raw.columns for c in targets):
        return None
    return raw[["time"] + targets].copy()


def datasets_for(names: Sequence[str], n: int) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for name in names:
        if name == "gpu":
            gpu = load_gpu_log()
            if gpu is not None:
                result[name] = gpu
        elif name in SYNTHETIC:
            result[name] = SYNTHETIC[name](n, seed=0)
        else:
            raise ValueError(f"Unknown dataset {name!r}")
    return result


# ---------------------------------------------------------------------------
# Missingness patterns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissingPattern:
    name: str
    kind: str
    rate: float
    block_length: int = 0


def parse_patterns(spec: str) -> List[MissingPattern]:
    patterns = []
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name in ("point10", "point_10"):
            patterns.append(MissingPattern(name="point10", kind="point", rate=0.10))
        elif name in ("point30", "point_30"):
            patterns.append(MissingPattern(name="point30", kind="point", rate=0.30))
        elif name in ("block10", "block_10"):
            patterns.append(
                MissingPattern(
                    name="block10", kind="block", rate=0.10, block_length=10
                )
            )
        elif name in ("block30", "block_30"):
            patterns.append(
                MissingPattern(
                    name="block30", kind="block", rate=0.30, block_length=10
                )
            )
        else:
            raise ValueError(
                f"Unknown missingness pattern {raw!r}. "
                "Choose from point10, point30, block10, block30."
            )
    if not patterns:
        raise ValueError("At least one missingness pattern is required.")
    return patterns


def _block_positions(n: int, k: int, length: int, rng) -> np.ndarray:
    """Pick non-overlapping blocks totalling about k positions."""
    if k <= 0 or length <= 0:
        return np.array([], dtype=int)
    length = min(length, n)
    selected = set()
    starts = np.arange(0, n - length + 1, dtype=int)
    rng.shuffle(starts)
    for start in starts:
        block = range(int(start), int(start) + length)
        if any(pos in selected for pos in block):
            continue
        selected.update(block)
        if len(selected) >= k:
            break
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        rng.shuffle(remaining)
        selected.update(remaining[: k - len(selected)])
    return np.array(sorted(selected)[:k], dtype=int)


def make_mask(
    df: pd.DataFrame,
    targets: Sequence[str],
    pattern: MissingPattern,
    seed: int,
) -> pd.DataFrame:
    """Build a boolean mask without touching already-missing cells."""
    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=df.index, columns=targets)
    for col in targets:
        available = np.flatnonzero(df[col].notna().to_numpy())
        if len(available) == 0:
            continue
        k = int(round(pattern.rate * len(available)))
        if k <= 0:
            continue
        if pattern.kind == "point":
            chosen = rng.choice(
                available, size=min(k, len(available)), replace=False
            )
        else:
            chosen = _block_positions(
                len(df), k, pattern.block_length, rng
            )
            chosen = chosen[df[col].iloc[chosen].notna().to_numpy()]
        mask.iloc[chosen, mask.columns.get_loc(col)] = True
    return mask


def corrupt(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    corrupted = df.copy()
    for col in mask.columns:
        corrupted.loc[mask[col], col] = np.nan
    return corrupted


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def fill_ffill(df: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in targets:
        out[col] = out[col].ffill().bfill()
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mean())
    return out


def fill_linear(df: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in targets:
        out[col] = out[col].interpolate(
            method="linear", limit_direction="both"
        )
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mean())
    return out


def fill_mean(df: pd.DataFrame, targets: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in targets:
        out[col] = out[col].fillna(out[col].mean())
    return out


def fill_hgb_lag(
    df: pd.DataFrame,
    targets: Sequence[str],
    seed: int,
    lags: Sequence[int] = (1, 2, 3),
    rolling_windows: Sequence[int] = (5,),
) -> pd.DataFrame:
    """One-shot HGB on lag/rolling features, with no iterative wrapper."""
    out = df.copy()
    time_col = "time"
    feature_cols = [c for c in df.columns if c != time_col]

    features = pd.DataFrame(index=df.index)
    features["time"] = df[time_col].to_numpy(dtype=float)
    for col in feature_cols:
        for lag in lags:
            features[f"{col}_lag{lag}"] = df[col].shift(lag)
        for window in rolling_windows:
            features[f"{col}_roll{window}"] = (
                df[col].shift(1).rolling(window=window, min_periods=1).mean()
            )

    for target in targets:
        observed = df[target].notna().to_numpy()
        missing = ~observed
        if observed.sum() < 10:
            out[target] = out[target].fillna(df[target].mean())
            continue
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=seed,
        )
        model.fit(features.loc[observed], df.loc[observed, target])
        prediction = model.predict(features.loc[missing])
        observed_values = df.loc[observed, target]
        prediction = np.clip(
            prediction,
            observed_values.min(),
            observed_values.max(),
        )
        out.loc[missing, target] = prediction
    return out


def fill_ts(
    df: pd.DataFrame,
    seed: int,
    lags,
    rolling_windows,
    max_iter: int,
    estimator: str = "fast",
) -> pd.DataFrame:
    if estimator == "fast":
        internal = MixedImputer(
            regressor=BayesianRidge(),
            classifier=LogisticRegression(max_iter=500),
            max_iter=max_iter,
            random_state=seed,
        )
        imputer = TimeSeriesImputer(
            time_column="time",
            lags=lags,
            rolling_windows=rolling_windows,
            initial_strategy="forward_fill",
            imputer=internal,
            random_state=seed,
        )
    elif estimator == "default":
        imputer = TimeSeriesImputer(
            time_column="time",
            lags=lags,
            rolling_windows=rolling_windows,
            initial_strategy="forward_fill",
            max_iter=max_iter,
            random_state=seed,
        )
    else:
        raise ValueError(
            f"Unknown estimator mode {estimator!r}; choose 'fast' or 'default'."
        )
    return imputer.fit_transform(df)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def _mae(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(truth - pred)))


def evaluate(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    mask: pd.DataFrame,
    targets: Sequence[str],
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    per_target: Dict[str, Dict[str, float]] = {}
    rmse_values = []
    mae_values = []
    for col in targets:
        selected = mask[col].to_numpy(dtype=bool)
        if not selected.any():
            continue
        true_values = truth.loc[selected, col].to_numpy(dtype=float)
        pred_values = prediction.loc[selected, col].to_numpy(dtype=float)
        if np.isnan(pred_values).any():
            rmse = float("nan")
            mae = float("nan")
        else:
            rmse = _rmse(true_values, pred_values)
            mae = _mae(true_values, pred_values)
        per_target[col] = {
            "n": int(selected.sum()),
            "rmse": rmse,
            "mae": mae,
        }
        rmse_values.append(rmse)
        mae_values.append(mae)
    return (
        float(np.nanmean(rmse_values)) if rmse_values else float("nan"),
        float(np.nanmean(mae_values)) if mae_values else float("nan"),
        per_target,
    )


def method_names() -> List[str]:
    return [
        "ffill",
        "linear",
        "mean",
        "hgb_lag",
        "ts_no_temporal",
        "ts_lags",
        "ts_full",
    ]


def run_method(
    method: str,
    corrupted: pd.DataFrame,
    targets: Sequence[str],
    seed: int,
    max_iter: int,
    estimator: str = "fast",
) -> pd.DataFrame:
    if method == "ffill":
        return fill_ffill(corrupted, targets)
    if method == "linear":
        return fill_linear(corrupted, targets)
    if method == "mean":
        return fill_mean(corrupted, targets)
    if method == "hgb_lag":
        return fill_hgb_lag(corrupted, targets, seed=seed)
    if method == "ts_no_temporal":
        return fill_ts(
            corrupted,
            seed=seed,
            lags=None,
            rolling_windows=None,
            max_iter=max_iter,
            estimator=estimator,
        )
    if method == "ts_lags":
        return fill_ts(
            corrupted,
            seed=seed,
            lags=(1, 2, 3),
            rolling_windows=None,
            max_iter=max_iter,
            estimator=estimator,
        )
    if method == "ts_full":
        return fill_ts(
            corrupted,
            seed=seed,
            lags=(1, 2, 3),
            rolling_windows=(5, 10),
            max_iter=max_iter,
            estimator=estimator,
        )
    raise ValueError(f"Unknown method {method!r}")


def run_validation(
    datasets: Dict[str, pd.DataFrame],
    patterns: Sequence[MissingPattern],
    methods: Sequence[str],
    seeds: int,
    max_iter: int,
    estimator: str = "fast",
) -> List[dict]:
    records: List[dict] = []
    for dataset_name, original in datasets.items():
        targets = [c for c in original.columns if c != "time"]
        for pattern in patterns:
            for seed in range(seeds):
                mask = make_mask(original, targets, pattern, seed)
                corrupted = corrupt(original, mask)
                for method in methods:
                    start = time.perf_counter()
                    error = ""
                    try:
                        prediction = run_method(
                            method,
                            corrupted,
                            targets,
                            seed,
                            max_iter,
                            estimator=estimator,
                        )
                        rmse, mae, per_target = evaluate(
                            prediction, original, mask, targets
                        )
                    except Exception as exc:  # keep the harness running
                        rmse = float("nan")
                        mae = float("nan")
                        per_target = {}
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - start
                    records.append(
                        {
                            "dataset": dataset_name,
                            "pattern": pattern.name,
                            "seed": seed,
                            "method": method,
                            "rmse": rmse,
                            "mae": mae,
                            "runtime": elapsed,
                            "error": error,
                            "per_target": per_target,
                        }
                    )
    return records


# ---------------------------------------------------------------------------
# Aggregation/reporting
# ---------------------------------------------------------------------------

def _ci95(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    sem = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - half, mean + half


def summarize(records: Sequence[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    groups = frame.groupby(["dataset", "pattern", "method"], dropna=False)
    rows = []
    for (dataset, pattern, method), group in groups:
        rmse_mean, rmse_low, rmse_high = _ci95(
            group["rmse"].to_numpy(dtype=float)
        )
        mae_mean, _, _ = _ci95(group["mae"].to_numpy(dtype=float))
        rows.append(
            {
                "dataset": dataset,
                "pattern": pattern,
                "method": method,
                "seeds": int(group["seed"].nunique()),
                "rmse": rmse_mean,
                "rmse_low": rmse_low,
                "rmse_high": rmse_high,
                "mae": mae_mean,
                "runtime": float(group["runtime"].mean()),
                "errors": int((group["error"] != "").sum()),
            }
        )
    summary = pd.DataFrame(rows)
    # Win rate against ts_full for each dataset/pattern/seed.
    reference = frame[frame["method"] == "ts_full"][
        ["dataset", "pattern", "seed", "rmse"]
    ].rename(columns={"rmse": "rmse_ts_full"})
    frame = frame.merge(reference, on=["dataset", "pattern", "seed"], how="left")
    win_rates = []
    for (dataset, pattern, method), group in frame.groupby(
        ["dataset", "pattern", "method"], dropna=False
    ):
        valid = group.dropna(subset=["rmse", "rmse_ts_full"])
        if method == "ts_full" or len(valid) == 0:
            win_rate = float("nan")
        else:
            win_rate = float(
                (valid["rmse"] < valid["rmse_ts_full"]).mean()
            )
        win_rates.append((dataset, pattern, method, win_rate))
    win_frame = pd.DataFrame(
        win_rates,
        columns=["dataset", "pattern", "method", "beats_ts_full"],
    )
    summary = summary.merge(
        win_frame, on=["dataset", "pattern", "method"], how="left"
    )
    return summary


def paired_tests(records: Sequence[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    reference = frame[frame["method"] == "ts_full"][
        ["dataset", "pattern", "seed", "rmse"]
    ].rename(columns={"rmse": "rmse_ts_full"})
    rows = []
    for (dataset, pattern), group in frame.groupby(["dataset", "pattern"]):
        pivot = group.pivot_table(
            index="seed", columns="method", values="rmse", aggfunc="mean"
        )
        if "ts_full" not in pivot.columns:
            continue
        ts = pivot["ts_full"]
        for method in pivot.columns:
            if method == "ts_full":
                continue
            paired = pd.concat(
                [ts.rename("ts"), pivot[method].rename("baseline")],
                axis=1,
            ).dropna()
            if len(paired) < 5:
                p_value = float("nan")
            else:
                differences = paired["ts"] - paired["baseline"]
                if np.allclose(differences, 0):
                    p_value = 1.0
                else:
                    try:
                        p_value = float(
                            stats.wilcoxon(
                                paired["ts"], paired["baseline"]
                            ).pvalue
                        )
                    except ValueError:
                        p_value = float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "pattern": pattern,
                    "method": method,
                    "n": len(paired),
                    "mean_ts": float(paired["ts"].mean())
                    if len(paired)
                    else float("nan"),
                    "mean_baseline": float(paired["baseline"].mean())
                    if len(paired)
                    else float("nan"),
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def render_markdown(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    datasets: Sequence[str],
    patterns: Sequence[str],
    seeds: int,
    max_iter: int,
    estimator: str = "fast",
) -> str:
    lines = [
        "# Time-series validation — Phase 1",
        "",
        "Generated by `benchmark/run_timeseries_validation.py`.",
        "",
        f"- Datasets: {', '.join(datasets)}",
        f"- Patterns: {', '.join(patterns)}",
        f"- Seeds per dataset/pattern: {seeds}",
        f"- `TimeSeriesImputer` max_iter: {max_iter}",
        f"- `TimeSeriesImputer` estimator mode: {estimator}",
        "",
        "`fast` mode uses `BayesianRidge`/`LogisticRegression` inside the "
        "same TimeSeries pipeline; `default` uses the package's "
        "HistGradientBoosting defaults and is much slower.",
        "",
        "RMSE is the mean across target columns; lower is better. "
        "`beats_ts_full` is the fraction of seeds where a baseline has lower "
        "RMSE than `ts_full`.",
        "",
        "## Aggregate results",
        "",
        "| Dataset | Pattern | Method | Seeds | RMSE | 95% CI | MAE | Runtime (s) | Beats ts_full |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(
        ["dataset", "pattern", "method"]
    ).iterrows():
        ci = (
            f"[{_fmt(row['rmse_low'])}, {_fmt(row['rmse_high'])}]"
            if not np.isnan(row["rmse_low"])
            else "—"
        )
        lines.append(
            "| {dataset} | {pattern} | {method} | {seeds} | {rmse} | {ci} | "
            "{mae} | {runtime} | {win} |".format(
                dataset=row["dataset"],
                pattern=row["pattern"],
                method=row["method"],
                seeds=int(row["seeds"]),
                rmse=_fmt(row["rmse"]),
                ci=ci,
                mae=_fmt(row["mae"]),
                runtime=_fmt(row["runtime"]),
                win=(
                    _fmt(row["beats_ts_full"], 2)
                    if not np.isnan(row["beats_ts_full"])
                    else "—"
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Paired test vs ts_full",
            "",
            "Wilcoxon signed-rank test on per-seed aggregate RMSE. "
            "A p-value below 0.05 means the difference is unlikely under "
            "the null of no difference; it does not establish practical "
            "importance, and the multiple comparisons are uncorrected.",
            "",
            "| Dataset | Pattern | Baseline | n | ts_full mean | Baseline mean | p |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in tests.sort_values(
        ["dataset", "pattern", "method"]
    ).iterrows():
        lines.append(
            "| {dataset} | {pattern} | {method} | {n} | {ts} | {base} | {p} |".format(
                dataset=row["dataset"],
                pattern=row["pattern"],
                method=row["method"],
                n=int(row["n"]),
                ts=_fmt(row["mean_ts"]),
                base=_fmt(row["mean_baseline"]),
                p=_fmt(row["p_value"], 4),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    records: Sequence[dict],
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    datasets: Sequence[str],
    patterns: Sequence[str],
    seeds: int,
    max_iter: int,
    markdown_path: Path,
    json_path: Path,
    estimator: str = "fast",
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_markdown(
            summary,
            tests,
            datasets,
            patterns,
            seeds,
            max_iter,
            estimator=estimator,
        ),
        encoding="utf-8",
    )
    payload = {
        "meta": {
            "datasets": list(datasets),
            "patterns": list(patterns),
            "seeds": seeds,
            "max_iter": max_iter,
            "estimator": estimator,
        },
        "records": list(records),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, default=float),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="white_noise,ar1,trend_seasonal,random_walk,regime_shift,multivariate,gpu",
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument(
        "--patterns",
        default="point10,point30,block10",
        help="Comma-separated missingness patterns.",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument(
        "--estimator",
        choices=("fast", "default"),
        default="fast",
        help="TimeSeriesImputer estimator: fast (BayesianRidge) or package default (HGB).",
    )
    parser.add_argument(
        "--methods",
        default=",".join(method_names()),
        help="Comma-separated methods.",
    )
    parser.add_argument(
        "--output",
        default="benchmark/timeseries_validation_results.md",
    )
    parser.add_argument(
        "--json-output",
        default="benchmark/timeseries_validation_results.json",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run: 2 seeds, three datasets, point10 only.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_datasets:
        print("Synthetic:", ", ".join(SYNTHETIC))
        print("Real: gpu")
        return

    if args.quick:
        datasets_arg = "white_noise,ar1,gpu"
        patterns_arg = "point10"
        seeds = 2
        max_iter = 3
    else:
        datasets_arg = args.datasets
        patterns_arg = args.patterns
        seeds = args.seeds
        max_iter = args.max_iter

    if datasets_arg == "all":
        dataset_names = list(SYNTHETIC) + ["gpu"]
    else:
        dataset_names = [
            name.strip() for name in datasets_arg.split(",") if name.strip()
        ]
    patterns = parse_patterns(patterns_arg)
    methods = [
        name.strip() for name in args.methods.split(",") if name.strip()
    ]
    unknown = set(methods) - set(method_names())
    if unknown:
        raise SystemExit(f"Unknown methods: {sorted(unknown)}")

    datasets = datasets_for(dataset_names, args.length)
    if not datasets:
        raise SystemExit("No datasets available.")

    print(
        f"Running {len(datasets)} datasets × {len(patterns)} patterns × "
        f"{seeds} seeds × {len(methods)} methods "
        f"(max_iter={max_iter}, estimator={args.estimator})"
    )
    records = run_validation(
        datasets,
        patterns,
        methods,
        seeds=seeds,
        max_iter=max_iter,
        estimator=args.estimator,
    )
    summary = summarize(records)
    tests = paired_tests(records)
    output = Path(args.output)
    json_output = Path(args.json_output)
    write_outputs(
        records,
        summary,
        tests,
        list(datasets),
        [p.name for p in patterns],
        seeds,
        max_iter,
        output,
        json_output,
        estimator=args.estimator,
    )
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")
    print(
        summary.sort_values(["dataset", "pattern", "rmse"])
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
