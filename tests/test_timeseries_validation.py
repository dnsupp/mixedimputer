"""Tests for the Phase 1 time-series validation harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parent.parent / "benchmark" / "run_timeseries_validation.py"
spec = importlib.util.spec_from_file_location("ts_validation", PATH)
validation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = validation
spec.loader.exec_module(validation)


def test_point_mask_targets_only_observed_cells():
    df = pd.DataFrame(
        {
            "time": np.arange(100, dtype=float),
            "x": np.arange(100, dtype=float),
            "y": np.arange(100, 200, dtype=float),
        }
    )
    df.loc[[3, 7, 11], "x"] = np.nan
    pattern = validation.MissingPattern(
        name="point10", kind="point", rate=0.10
    )
    mask = validation.make_mask(df, ["x", "y"], pattern, seed=42)
    assert mask.sum().sum() == 20
    assert not mask.loc[[3, 7, 11], "x"].any()
    assert mask["x"].sum() == 10
    assert mask["y"].sum() == 10


def test_block_mask_is_contiguous():
    df = pd.DataFrame(
        {
            "time": np.arange(100, dtype=float),
            "x": np.arange(100, dtype=float),
        }
    )
    pattern = validation.MissingPattern(
        name="block10", kind="block", rate=0.10, block_length=10
    )
    mask = validation.make_mask(df, ["x"], pattern, seed=0)
    selected = np.flatnonzero(mask["x"].to_numpy())
    assert len(selected) == 10
    assert selected.max() - selected.min() + 1 == 10


def test_invalid_pattern_raises():
    with pytest.raises(ValueError, match="Unknown missingness pattern"):
        validation.parse_patterns("not-a-pattern")


def test_evaluate_returns_metrics():
    truth = pd.DataFrame(
        {"time": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    prediction = truth.copy()
    prediction.loc[2, "x"] = 5.0
    mask = pd.DataFrame(
        {"x": [False, False, True, False]}, index=truth.index
    )
    rmse, mae, per_target = validation.evaluate(
        prediction, truth, mask, ["x"]
    )
    assert rmse == pytest.approx(2.0)
    assert mae == pytest.approx(2.0)
    assert per_target["x"]["n"] == 1
