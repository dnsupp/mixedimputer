"""Regression tests for the release-blocking fixes in REVIEW.md."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mixedimputer import (
    DataCorrupter,
    MixedImputer,
    TimeSeriesImputer,
    is_time_series_suitable,
)


# ---------------------------------------------------------------------------
# MixedImputer
# ---------------------------------------------------------------------------

class TestMixedImputerReleaseFixes:
    def test_boolean_column_stays_boolean(self):
        df = pd.DataFrame(
            {
                "flag": pd.array([True, False, None, True], dtype="boolean"),
                "x": [1.0, 2.0, np.nan, 4.0],
                "cat": ["a", "b", "a", np.nan],
            }
        )
        result = MixedImputer(max_iter=2, random_state=42).fit_transform(df)
        assert pd.api.types.is_bool_dtype(result["flag"].dtype)
        assert set(result["flag"].dropna().tolist()).issubset({True, False})

    def test_all_missing_categorical_column_is_dropped(self):
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0],
                "cat": ["a", None, "a", "b"],
                "empty": pd.Series([None] * 4, dtype=object),
            }
        )
        result = MixedImputer(max_iter=2, random_state=42).fit_transform(df)
        assert "empty" not in result.columns
        assert list(result.columns) == ["num", "cat"]
        assert not result.isna().any().any()

    def test_all_missing_categorical_kept_when_requested(self):
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0, np.nan, 4.0],
                "empty": pd.Series([None] * 4, dtype=object),
            }
        )
        result = MixedImputer(
            keep_empty_features=True, max_iter=2, random_state=42
        ).fit_transform(df)
        assert "empty" in result.columns
        assert result["empty"].isna().all()

    def test_datetime_column_raises_clear_error(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
                "x": [1.0, np.nan, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="cannot handle"):
            MixedImputer(max_iter=2, random_state=42).fit(df)

    def test_array_dropped_feature_is_nan_not_broadcast(self):
        X = np.array(
            [[1.0, np.nan], [2.0, np.nan], [3.0, np.nan], [4.0, np.nan]]
        )
        result = MixedImputer(
            categorical_features=[], max_iter=2, random_state=42
        ).fit_transform(X)
        assert result.shape == X.shape
        assert np.isnan(result[:, 1]).all()
        np.testing.assert_allclose(result[:, 0], X[:, 0])

    def test_array_string_categoricals(self):
        X = np.array(
            [["a", 1.0], ["b", np.nan], [np.nan, 3.0], ["a", 4.0]],
            dtype=object,
        )
        result = MixedImputer(
            categorical_features=[0], max_iter=2, random_state=42
        ).fit_transform(X)
        assert result.shape == X.shape
        assert set(result[:, 0]).issubset({"a", "b"})
        assert not np.isnan(result[:, 1].astype(float)).any()

    def test_add_indicator_columns(self):
        df = pd.DataFrame(
            {
                "age": [25, 30, np.nan, 40, 35],
                "city": ["paris", "london", np.nan, "paris", "london"],
                "income": [50000, np.nan, 70000, 60000, 55000],
                "gender": ["M", "F", "M", np.nan, "F"],
            }
        )
        imputer = MixedImputer(
            add_indicator=True, max_iter=2, random_state=42
        )
        result = imputer.fit_transform(df)
        expected = {
            f"missingindicator_{col}" for col in df.columns
        }
        assert expected.issubset(result.columns)
        assert all(result[col].dtype == bool for col in expected)
        assert imputer.n_features_out_ == len(result.columns)
        assert set(imputer.get_feature_names_out()).issuperset(expected)

    def test_refit_on_new_schema(self):
        imputer = MixedImputer(max_iter=2, random_state=42)
        imputer.fit(
            pd.DataFrame(
                {"age": [1.0, np.nan, 3.0], "city": ["a", "b", np.nan]}
            )
        )
        imputer.fit(
            pd.DataFrame(
                {"value": [1.0, np.nan, 3.0], "kind": ["x", "y", np.nan]}
            )
        )
        result = imputer.transform(
            pd.DataFrame(
                {"value": [1.0, 2.0, 3.0], "kind": ["x", "y", "x"]}
            )
        )
        assert list(result.columns) == ["value", "kind"]

    def test_transform_uses_fit_column_order(self):
        df = pd.DataFrame(
            {
                "age": [25, 30, np.nan, 40, 35],
                "city": ["paris", "london", np.nan, "paris", "london"],
            }
        )
        imputer = MixedImputer(max_iter=2, random_state=42).fit(df)
        result = imputer.transform(df[["city", "age"]])
        assert list(result.columns) == ["age", "city"]

    def test_integer_feature_positions_for_dataframe(self):
        df = pd.DataFrame(
            {
                "age": [25, 30, np.nan, 40, 35],
                "city": ["paris", "london", np.nan, "paris", "london"],
            }
        )
        result = MixedImputer(
            categorical_features=[1], max_iter=2, random_state=42
        ).fit_transform(df)
        assert not result["city"].isna().any()

    def test_fitted_attributes_and_names(self):
        df = pd.DataFrame(
            {
                "age": [25, 30, np.nan, 40, 35],
                "city": ["paris", "london", np.nan, "paris", "london"],
                "income": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        imputer = MixedImputer(max_iter=2, random_state=42).fit(df)
        assert imputer.n_features_in_ == 3
        assert list(imputer.feature_names_in_) == list(df.columns)
        assert imputer.categorical_features_ == ["city"]
        assert imputer.numeric_features_ == ["age", "income"]
        assert list(imputer.get_feature_names_out()) == list(df.columns)
        # Constructor parameters are not mutated by fit.
        assert imputer.categorical_features is None
        assert imputer.numeric_features is None

    def test_unseen_category_is_imputed(self):
        imputer = MixedImputer(max_iter=3, random_state=42)
        imputer.fit(
            pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, np.nan, 3.0]})
        )
        result = imputer.transform(
            pd.DataFrame({"cat": ["a", "c", "b"], "num": [4.0, np.nan, 6.0]})
        )
        assert not result["cat"].isna().any()
        assert set(result["cat"]).issubset({"a", "b"})


# ---------------------------------------------------------------------------
# TimeSeriesImputer
# ---------------------------------------------------------------------------

class TestTimeSeriesReleaseFixes:
    def test_datetime_time_column(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=12, freq="h"
                ),
                "temperature": [20.0, 21.0, np.nan, 23.0, 24.0]
                + [np.nan] * 7,
                "sensor_status": ["OK", "OK", np.nan, "WARN", "OK"]
                + [np.nan] * 7,
            }
        )
        imputer = TimeSeriesImputer(
            time_column="timestamp",
            lags=[1],
            rolling_windows=[2],
            max_iter=2,
            random_state=42,
        )
        result = imputer.fit_transform(df)
        assert list(result.columns) == list(df.columns)
        pd.testing.assert_series_equal(
            result["timestamp"].reset_index(drop=True),
            df["timestamp"].reset_index(drop=True),
        )
        assert not result["temperature"].isna().any()
        assert imputer.n_features_in_ == 3

    def test_string_time_column_with_categorical(self):
        df = pd.DataFrame(
            {
                "time": [f"2024-01-{i:02d}" for i in range(1, 9)],
                "value": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0],
                "status": ["a", "b", np.nan, "a", "b", "a", np.nan, "b"],
            }
        )
        result = TimeSeriesImputer(
            time_column="time", lags=[1], max_iter=2, random_state=42
        ).fit_transform(df)
        assert list(result.columns) == ["time", "value", "status"]
        assert result["time"].tolist() == df["time"].tolist()

    def test_unsorted_input_restores_order_and_index(self):
        df = pd.DataFrame(
            {
                "time": [5, 1, 3, 2, 4],
                "value": [10.0, np.nan, 30.0, np.nan, 50.0],
            },
            index=[100, 101, 102, 103, 104],
        )
        result = TimeSeriesImputer(
            time_column="time", lags=[1], max_iter=2, random_state=42
        ).fit_transform(df)
        assert list(result.index) == list(df.index)
        assert result["time"].tolist() == [5, 1, 3, 2, 4]
        assert result.loc[100, "value"] == pytest.approx(10.0)
        assert result.loc[102, "value"] == pytest.approx(30.0)
        assert result.loc[104, "value"] == pytest.approx(50.0)

    def test_lags_none_disables_lag_features(self):
        df = pd.DataFrame(
            {"time": [1, 2, 3, 4], "value": [1.0, np.nan, 3.0, 4.0]}
        )
        imputer = TimeSeriesImputer(
            time_column="time",
            lags=None,
            rolling_windows=[],
            max_iter=2,
            random_state=42,
        )
        result = imputer.fit_transform(df)
        assert imputer.lags is None
        assert not any("_lag" in name for name in imputer.engineered_feature_names_)
        assert not result["value"].isna().any()

    def test_linear_interpolation_with_categorical(self):
        df = pd.DataFrame(
            {
                "time": [1, 2, 3, 4, 5, 6],
                "value": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
                "status": ["a", "b", np.nan, "a", np.nan, "b"],
            }
        )
        result = TimeSeriesImputer(
            time_column="time",
            lags=[],
            rolling_windows=[],
            initial_strategy="linear",
            max_iter=2,
            random_state=42,
        ).fit_transform(df)
        assert not result["status"].isna().any()

    def test_rolling_feature_excludes_current_row(self):
        df = pd.DataFrame(
            {
                "time": [1, 2, 3, 4],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        imputer = TimeSeriesImputer(
            time_column="time",
            lags=None,
            rolling_windows=[2],
            max_iter=1,
            random_state=42,
        )
        imputer.fit(df)
        prepared, engineered, _, _ = imputer._prepare_time_features(df)
        name = engineered[0]
        # At t=3 the shifted window contains values at t=1 and t=2 (1.5),
        # not the current value 3.0.
        assert prepared[name].iloc[2] == pytest.approx(1.5)
        assert prepared[name].iloc[3] == pytest.approx(2.5)

    def test_is_time_series_suitable_irregular(self):
        report = is_time_series_suitable(
            pd.DataFrame(
                {
                    "time": [0.0, 1.0, 2.0, 10.0, 11.0],
                    "value": [1.0, 2.0, np.nan, 4.0, 5.0],
                }
            ),
            "time",
        )
        assert report["is_regular"] is False
        assert "irregular" in report["recommendation"].lower()


# ---------------------------------------------------------------------------
# DataCorrupter
# ---------------------------------------------------------------------------

class TestDataCorrupterReleaseFixes:
    def test_top_level_import(self):
        assert DataCorrupter is not None

    def test_mcar_targets_observed_fraction(self):
        df = pd.DataFrame({"value": np.arange(100, dtype=float)})
        df.loc[np.arange(20), "value"] = np.nan
        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.10,
            numeric_columns=["value"],
            random_state=42,
        )
        _, mask, _ = corrupter.corrupt(df)
        n_observed = df["value"].notna().sum()
        assert mask["value"].sum() == round(0.10 * n_observed)
        assert not (mask["value"] & df["value"].isna()).any()

    def test_mar_produces_nonzero_exact_fraction_and_is_reproducible(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "a": rng.normal(size=200),
                "b": rng.normal(size=200),
                "c": rng.normal(size=200),
            }
        )
        masks = []
        for _ in range(2):
            corrupter = DataCorrupter(
                mechanism="MAR",
                corruption_fraction=0.20,
                numeric_columns=["a"],
                random_state=7,
            )
            _, mask, _ = corrupter.corrupt(df)
            masks.append(mask)
        assert mask["a"].sum() == round(0.20 * len(df))
        assert masks[0].equals(masks[1])

    def test_mar_requires_another_column(self):
        df = pd.DataFrame({"a": np.arange(20, dtype=float)})
        corrupter = DataCorrupter(
            mechanism="MAR",
            corruption_fraction=0.20,
            numeric_columns=["a"],
            random_state=0,
        )
        with pytest.raises(ValueError, match="at least one other column"):
            corrupter.corrupt(df)

    def test_mnar_high_targets_high_values(self):
        df = pd.DataFrame({"value": np.arange(1000, dtype=float)})
        high_masks = []
        low_masks = []
        for extreme, store in (("high", high_masks), ("low", low_masks)):
            corrupter = DataCorrupter(
                mechanism="MNAR",
                corruption_fraction=0.10,
                numeric_columns=["value"],
                mnar_extreme=extreme,
                random_state=0,
            )
            _, mask, _ = corrupter.corrupt(df)
            store.append(mask["value"].to_numpy())
        high_mean = df.loc[high_masks[0], "value"].mean()
        low_mean = df.loc[low_masks[0], "value"].mean()
        assert high_mean > 800
        assert low_mean < 200
        assert not np.array_equal(high_masks[0], low_masks[0])

    def test_no_global_numpy_state_is_changed(self):
        df = pd.DataFrame(
            {
                "a": np.arange(100, dtype=float),
                "b": np.arange(100, 200, dtype=float),
            }
        )
        np.random.seed(123)
        expected = np.random.rand(3)
        np.random.seed(123)
        DataCorrupter(
            mechanism="MAR",
            corruption_fraction=0.2,
            numeric_columns=["a"],
            random_state=42,
        ).corrupt(df)
        actual = np.random.rand(3)
        np.testing.assert_allclose(expected, actual)

    def test_path_input(self):
        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.01,
            numeric_columns=["Age"],
            random_state=0,
        )
        corrupted, mask, original = corrupter.corrupt(
            Path("data/titanic.csv")
        )
        assert corrupted.shape == original.shape
        assert mask.shape == original.shape

    def test_invalid_dict_fraction_raises(self):
        df = pd.DataFrame({"a": np.arange(10, dtype=float)})
        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction={"a": 1.5},
            numeric_columns=["a"],
            random_state=0,
        )
        with pytest.raises(ValueError, match="must be in"):
            corrupter.corrupt(df)
