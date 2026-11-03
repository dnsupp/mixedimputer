"""
Time-series-aware iterative imputer for mixed-type DataFrames.

Provides ``TimeSeriesImputer``, which wraps ``MixedImputer`` with temporal
feature engineering (lag features, rolling statistics) and time-aware
initial imputation strategies, making iterative MICE-style imputation
suitable for time series data with mixed column types.

Core idea
---------
Plain ``IterativeImputer`` treats rows as i.i.d. — it cannot exploit
temporal autocorrelation.  ``TimeSeriesImputer`` addresses this by:

1. **Sorting** the data by a user-supplied time/ordering column.
2. **Temporary fill** — using time-aware strategies (forward fill,
   interpolation) instead of a global mean, preserving trends.
3. **Feature engineering** — creating lag and rolling-mean features for
   numeric columns so the iterative imputer can learn temporal
   dependencies.
4. **Delegating** to ``MixedImputer`` for the core MICE-style iterative
   imputation.
5. **Cleanup** — stripping engineered features, restoring the original row
   order and index, and re-inserting the time column.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._imputer import MixedImputer


# ---------------------------------------------------------------------------
# Public transformer
# ---------------------------------------------------------------------------

class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """
    Time-series-aware imputer for mixed-type DataFrames.

    The time column is treated as metadata: it is used for sorting, is not
    imputed, and is re-inserted at its original position in the output.  The
    original row order and index are restored after imputation.

    Parameters
    ----------
    time_column : str or int
        Column name or positional index identifying the time/ordering
        dimension.  Numeric, datetime, and string time columns are supported.

    lags : tuple/list of int or None, default=(1, 2, 3)
        Positive lag steps for which lag features are created for each
        numeric feature.  ``None`` disables lag features.

    rolling_windows : list of int or None, default=None
        Positive window sizes for shifted rolling-mean features.  ``None``
        disables rolling features.

    initial_strategy : str, default='forward_fill'
        Temporary fill used **only** to build lag/rolling features.  The
        final imputation is performed by ``MixedImputer``.  Supported values:

        - ``'forward_fill'`` / ``'ffill'``
        - ``'backward_fill'`` / ``'bfill'``
        - ``'linear'`` / ``'spline'`` (numeric columns only; categorical
          columns use forward/backward fill)
        - ``'mean'`` / ``'median'`` / ``'most_frequent'`` / ``'constant'``

    imputer : MixedImputer or None, default=None
        Pre-configured ``MixedImputer``.  If supplied, ``max_iter``,
        ``tol``, ``sample_posterior``, ``random_state``, and ``verbose``
        are not used to construct a new imputer.

    categorical_features : list of str/int or None
        Categorical columns, used to skip lag/rolling creation.  Also passed
        to a default ``MixedImputer``.

    numeric_features : list of str/int or None
        Numeric columns that receive lag/rolling features.  This parameter
        controls feature engineering only; engineered numeric features are
        auto-detected by the internal ``MixedImputer``.

    max_iter : int, default=10
    tol : float, default=1e-3
    sample_posterior : bool, default=False
    random_state : int, RandomState or None, default=None
    verbose : int, default=0

    Attributes
    ----------
    time_column_ : str
        Resolved time column label.

    original_column_order_ : list
        Input column order during fit.

    engineered_feature_names_ : list of str
        Lag/rolling names created during fit.

    imputer_ : MixedImputer or None
        Fitted internal imputer (``None`` when the frame has only a time
        column).

    n_features_in_ : int
        Number of input columns, including the time column.

    feature_names_in_ : ndarray of str
        Defined when all input column names are strings.

    n_features_out_ : int
        Number of output columns (the original columns).
    """

    def __init__(
        self,
        time_column: Union[str, int],
        lags: Optional[List[int]] = (1, 2, 3),
        rolling_windows: Optional[List[int]] = None,
        initial_strategy: str = "forward_fill",
        imputer: Optional[MixedImputer] = None,
        categorical_features=None,
        numeric_features=None,
        max_iter: int = 10,
        tol: float = 1e-3,
        sample_posterior: bool = False,
        random_state=None,
        verbose: int = 0,
    ):
        self.time_column = time_column
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.initial_strategy = initial_strategy
        self.imputer = imputer
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.max_iter = max_iter
        self.tol = tol
        self.sample_posterior = sample_posterior
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Feature resolution
    # ------------------------------------------------------------------
    def _resolve_time_column(self, X: pd.DataFrame) -> str:
        columns = list(X.columns)
        if isinstance(self.time_column, (int, np.integer)):
            # Prefer an exact integer label, then fall back to a position.
            if self.time_column in columns:
                return self.time_column
            pos = int(self.time_column)
            if 0 <= pos < len(columns):
                return columns[pos]
            raise ValueError(
                f"time_column index {pos} is out of bounds for a DataFrame "
                f"with {len(columns)} columns."
            )
        if self.time_column not in columns:
            raise ValueError(
                f"time_column {self.time_column!r} not found in DataFrame "
                f"columns: {columns!r}"
            )
        return self.time_column

    def _resolve_features(self, X: pd.DataFrame):
        feature_cols = [
            col for col in X.columns if col != self.time_column_
        ]
        cat_spec = MixedImputer._normalize_feature_spec(
            self.categorical_features,
            len(X.columns),
            list(X.columns),
            "categorical_features",
        )
        if cat_spec is None:
            cat = [
                col
                for col in feature_cols
                if (
                    pd.api.types.is_object_dtype(X[col].dtype)
                    or pd.api.types.is_string_dtype(X[col].dtype)
                    or isinstance(X[col].dtype, pd.CategoricalDtype)
                    or pd.api.types.is_bool_dtype(X[col].dtype)
                )
            ]
        else:
            cat = [c for c in cat_spec if c != self.time_column_]

        num_spec = MixedImputer._normalize_feature_spec(
            self.numeric_features,
            len(X.columns),
            list(X.columns),
            "numeric_features",
        )
        if num_spec is None:
            num = [
                col
                for col in feature_cols
                if col not in cat
                and pd.api.types.is_numeric_dtype(X[col].dtype)
                and not pd.api.types.is_bool_dtype(X[col].dtype)
            ]
        else:
            num = [c for c in num_spec if c != self.time_column_]

        if self.time_column_ in (cat_spec or []):
            warnings.warn(
                "time_column was present in categorical_features; it is "
                "used only for ordering and will not be imputed.",
                UserWarning,
                stacklevel=2,
            )
        if self.time_column_ in (num_spec or []):
            warnings.warn(
                "time_column was present in numeric_features; it is used "
                "only for ordering and will not be imputed.",
                UserWarning,
                stacklevel=2,
            )

        overlap = set(cat).intersection(num)
        if overlap:
            raise ValueError(
                f"Columns cannot be both categorical and numeric: "
                f"{sorted(overlap)!r}"
            )

        # Explicit lists are overrides; assign remaining columns by dtype.
        assigned = set(cat).union(num)
        for col in feature_cols:
            if col in assigned:
                continue
            dtype = X[col].dtype
            if (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(dtype)
            ):
                cat.append(col)
            elif (
                pd.api.types.is_numeric_dtype(dtype)
                and not pd.api.types.is_bool_dtype(dtype)
                and not pd.api.types.is_complex_dtype(dtype)
            ):
                num.append(col)
            assigned.add(col)

        for col in num:
            if not pd.api.types.is_numeric_dtype(X[col].dtype):
                raise ValueError(
                    f"numeric_features column {col!r} has non-numeric dtype "
                    f"{X[col].dtype!r}."
                )
            if pd.api.types.is_bool_dtype(X[col].dtype):
                raise ValueError(
                    f"Boolean column {col!r} must be treated as categorical."
                )

        unassigned = [
            col for col in feature_cols if col not in set(cat).union(num)
        ]
        if unassigned:
            details = ", ".join(
                f"{col!r} ({X[col].dtype!r})" for col in unassigned
            )
            raise ValueError(
                "Some non-time columns have unsupported dtypes "
                f"(datetime/timedelta/other): {details}."
            )
        return cat, num

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_int_list(value, name, allow_none=True):
        if value is None:
            if allow_none:
                return []
            raise TypeError(f"{name} must not be None.")
        if isinstance(value, (int, np.integer)):
            value = [value]
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError(
                f"{name} must be a list/tuple of positive integers or None; "
                f"got {type(value).__name__}."
            )
        result = []
        for item in value:
            if not isinstance(item, (int, np.integer)) or int(item) <= 0:
                raise ValueError(
                    f"{name} entries must be positive integers; got {item!r}."
                )
            result.append(int(item))
        return result

    def _fill_modes(self, df: pd.DataFrame, columns) -> None:
        for col in columns:
            if df[col].isna().any():
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])

    # ------------------------------------------------------------------
    # Temporary fill for feature engineering
    # ------------------------------------------------------------------
    def _apply_initial_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a temporary fill used only to build lag/rolling features."""
        strategy = str(self.initial_strategy).lower()
        df = df.copy()
        columns = list(df.columns)
        cat_cols = [c for c in columns if c in getattr(self, "_cat_features_", [])]
        num_cols = [c for c in columns if c not in cat_cols]

        if strategy in ("forward_fill", "ffill"):
            df[columns] = df[columns].ffill().bfill()

        elif strategy in ("backward_fill", "bfill"):
            df[columns] = df[columns].bfill().ffill()

        elif strategy in ("linear", "spline"):
            if cat_cols:
                df[cat_cols] = df[cat_cols].ffill().bfill()
                self._fill_modes(df, cat_cols)
            if num_cols:
                try:
                    if strategy == "spline":
                        df[num_cols] = df[num_cols].interpolate(
                            method="spline",
                            order=3,
                            limit_direction="both",
                        )
                    else:
                        df[num_cols] = df[num_cols].interpolate(
                            method="linear", limit_direction="both"
                        )
                except (TypeError, ValueError):
                    # Too few points, constant series, or an engine edge
                    # case: fall back to a robust linear/ffill fill.
                    try:
                        df[num_cols] = df[num_cols].interpolate(
                            method="linear", limit_direction="both"
                        )
                    except (TypeError, ValueError):
                        df[num_cols] = df[num_cols].ffill().bfill()

        elif strategy in ("mean", "median"):
            if num_cols:
                if strategy == "mean":
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                else:
                    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            self._fill_modes(df, cat_cols)

        elif strategy == "most_frequent":
            self._fill_modes(df, columns)

        elif strategy == "constant":
            if num_cols:
                df[num_cols] = df[num_cols].fillna(0.0)
            if cat_cols:
                df[cat_cols] = (
                    df[cat_cols].astype(object).fillna("missing")
                )

        else:
            raise ValueError(
                f"Unsupported initial_strategy {self.initial_strategy!r}. "
                "Choose from: forward_fill, backward_fill, linear, spline, "
                "mean, median, most_frequent, constant."
            )

        return df

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------
    def _prepare_time_features(self, X: pd.DataFrame):
        """Sort, fill temporarily, add lag/rolling features, restore NaN.

        Returns
        -------
        X_features : DataFrame
            Feature columns only (time column excluded), sorted by time.
        engineered_names : list of str
        sort_order : ndarray
            Positions that sort the original rows into time order.
        time_values : Series
            Time column in the same sorted order.
        """
        time_values_original = X[self.time_column_]
        order_frame = pd.DataFrame(
            {
                "_time": time_values_original.to_numpy(),
                "_pos": np.arange(len(X)),
            }
        )
        try:
            sort_order = order_frame.sort_values(
                "_time", kind="mergesort"
            )["_pos"].to_numpy(dtype=int)
        except TypeError:
            # Mixed object timestamps that pandas cannot sort: fall back to
            # string rendering so the ordering is at least deterministic.
            order_frame["_time"] = time_values_original.astype(str).to_numpy()
            sort_order = order_frame.sort_values(
                "_time", kind="mergesort"
            )["_pos"].to_numpy(dtype=int)

        X_sorted = X.iloc[sort_order]
        time_values = X_sorted[self.time_column_]
        feature_frame = X_sorted[self._feature_columns_].copy()

        if feature_frame.shape[1] == 0:
            return feature_frame, [], sort_order, time_values

        nan_mask = feature_frame.isna()
        feature_filled = self._apply_initial_fill(feature_frame)

        lags = self._validate_int_list(self.lags, "lags", allow_none=True)
        windows = self._validate_int_list(
            self.rolling_windows, "rolling_windows", allow_none=True
        )

        engineered = []
        existing = set(self._fit_columns_).union(feature_filled.columns)
        for col in self._num_features_:
            for lag in lags:
                name = f"{col}_lag{lag}"
                if name in existing:
                    raise ValueError(
                        f"Engineered feature name {name!r} collides with an "
                        "existing column."
                    )
                feature_filled[name] = feature_filled[col].shift(lag)
                engineered.append(name)
                existing.add(name)
            for window in windows:
                name = f"{col}_rollmean{window}"
                if name in existing:
                    raise ValueError(
                        f"Engineered feature name {name!r} collides with an "
                        "existing column."
                    )
                # Shift by one so the current target value is excluded from
                # its own rolling feature.
                feature_filled[name] = (
                    feature_filled[col]
                    .shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
                engineered.append(name)
                existing.add(name)

        # Restore the original missingness for the original feature columns;
        # only the engineered features retain the temporary fill.
        feature_filled[self._feature_columns_] = feature_filled[
            self._feature_columns_
        ].where(~nan_mask, other=np.nan)
        return feature_filled, engineered, sort_order, time_values

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the time-series imputer."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"TimeSeriesImputer requires a pandas DataFrame, got "
                f"{type(X).__name__}."
            )
        if len(X) == 0:
            raise ValueError("X must contain at least one sample.")
        if len(X.columns) == 0:
            raise ValueError("X must contain at least one column.")
        if len(set(X.columns)) != len(X.columns):
            raise ValueError("X has duplicate column names.")

        self.time_column_ = self._resolve_time_column(X)
        self._fit_columns_ = list(X.columns)
        self._time_position_ = self._fit_columns_.index(self.time_column_)
        self._feature_columns_ = [
            c for c in self._fit_columns_ if c != self.time_column_
        ]
        self.n_features_in_ = X.shape[1]
        if all(isinstance(c, str) for c in self._fit_columns_):
            self.feature_names_in_ = np.asarray(
                self._fit_columns_, dtype=object
            )
        else:
            if hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        self.original_column_order_ = list(self._fit_columns_)

        self._cat_features_, self._num_features_ = self._resolve_features(X)

        if len(self._feature_columns_) == 0:
            self.imputer_ = None
            self.engineered_feature_names_ = []
            self.n_features_out_ = len(self._fit_columns_)
            self._is_fitted_ = True
            return self

        if self.imputer is None:
            self.imputer_ = MixedImputer(
                categorical_features=self._cat_features_,
                max_iter=self.max_iter,
                tol=self.tol,
                sample_posterior=self.sample_posterior,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            self.imputer_ = self.imputer

        (
            X_prepared,
            self.engineered_feature_names_,
            _,
            _,
        ) = self._prepare_time_features(X)
        if X_prepared.shape[1] > 0 and np.all(pd.isna(X_prepared.to_numpy())):
            # Nothing observable anywhere: keep the schema and return NaN.
            self.imputer_ = None
            self.n_features_out_ = len(self._fit_columns_)
            self._is_fitted_ = True
            return self
        self.imputer_.fit(X_prepared)
        self.n_features_out_ = len(self._fit_columns_)
        self._is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values, restoring original row order and index."""
        check_is_fitted(self, "_is_fitted_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"TimeSeriesImputer requires a pandas DataFrame, got "
                f"{type(X).__name__}."
            )
        columns = list(X.columns)
        if len(set(columns)) != len(columns):
            raise ValueError("X has duplicate column names.")
        missing_columns = [
            c for c in self._fit_columns_ if c not in columns
        ]
        extra_columns = [
            c for c in columns if c not in self._fit_columns_
        ]
        if missing_columns or extra_columns:
            raise ValueError(
                "X columns do not match the fitted schema. "
                f"Missing: {missing_columns!r}; extra: {extra_columns!r}."
            )
        X = X.loc[:, self._fit_columns_].copy()
        original_index = X.index

        if len(self._feature_columns_) == 0 or self.imputer_ is None:
            return X.copy()

        (
            X_prepared,
            engineered,
            sort_order,
            time_values,
        ) = self._prepare_time_features(X)
        self.engineered_feature_names_ = engineered
        imputed_features = self.imputer_.transform(X_prepared)

        # Re-add features that the internal imputer dropped because they were
        # entirely missing during fit; keep the original schema.
        for col in self._feature_columns_:
            if col not in imputed_features.columns:
                imputed_features[col] = np.nan

        result_sorted = imputed_features.copy()
        result_sorted.insert(
            self._time_position_, self.time_column_, time_values
        )
        result_sorted = result_sorted.reindex(columns=self._fit_columns_)

        # Undo the stable sort and restore the caller's index.
        inverse_order = np.argsort(sort_order)
        result = result_sorted.iloc[inverse_order].copy()
        result.index = original_index
        return result

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        """Return the original column names, in original order."""
        check_is_fitted(self, "_is_fitted_")
        if input_features is not None:
            provided = list(input_features)
            if provided != list(self._fit_columns_):
                raise ValueError(
                    "input_features must match the columns used during fit."
                )
        return np.asarray([str(c) for c in self._fit_columns_], dtype=object)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def is_time_series_suitable(df: pd.DataFrame, time_column: str) -> dict:
    """Assess whether a DataFrame is suitable for time-aware imputation.

    Returns
    -------
    report : dict
        Keys: ``is_sorted``, ``is_regular``, ``has_gaps``,
        ``has_duplicates``, ``missing_rate``, ``recommendation``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if time_column not in df.columns:
        raise ValueError(f"time_column {time_column!r} not found in columns.")

    report = {}
    time_series = df[time_column]
    report["is_sorted"] = bool(time_series.is_monotonic_increasing)

    time_vals = time_series.dropna()
    report["has_duplicates"] = bool(time_vals.duplicated().any())
    report["is_regular"] = None
    report["has_gaps"] = False

    if len(time_vals) >= 3:
        if pd.api.types.is_datetime64_any_dtype(time_vals):
            diffs = time_vals.diff().dropna()
            if len(diffs) >= 2:
                mode_diff = (
                    diffs.mode().iloc[0] if not diffs.mode().empty else None
                )
                if mode_diff is not None and mode_diff != pd.Timedelta(0):
                    report["is_regular"] = bool((diffs == mode_diff).all())
                else:
                    report["is_regular"] = False
            else:
                report["is_regular"] = True
        elif pd.api.types.is_numeric_dtype(time_vals):
            diffs = time_vals.diff().dropna()
            if len(diffs) >= 2:
                mode_diff = (
                    diffs.mode().iloc[0] if not diffs.mode().empty else None
                )
                if mode_diff is not None and not np.isclose(mode_diff, 0):
                    report["is_regular"] = bool(np.allclose(diffs, mode_diff))
                else:
                    report["is_regular"] = False
            else:
                report["is_regular"] = True
        else:
            report["is_regular"] = None

    if report.get("is_regular") and len(time_vals) >= 3:
        if pd.api.types.is_numeric_dtype(time_vals):
            step = float(time_vals.diff().dropna().mode().iloc[0])
            if step > 0:
                expected = int(round((time_vals.max() - time_vals.min()) / step)) + 1
                report["has_gaps"] = expected > len(time_vals)
        elif pd.api.types.is_datetime64_any_dtype(time_vals):
            step = time_vals.diff().dropna().mode().iloc[0]
            if step > pd.Timedelta(0):
                expected = int(
                    round(
                        (time_vals.max() - time_vals.min()) / step
                    )
                ) + 1
                report["has_gaps"] = expected > len(time_vals)

    feature_cols = [c for c in df.columns if c != time_column]
    total_cells = df[feature_cols].size
    missing_cells = df[feature_cols].isna().sum().sum()
    report["missing_rate"] = (
        float(missing_cells / total_cells) if total_cells > 0 else 0.0
    )

    reasons = []
    if not report["is_sorted"]:
        reasons.append("data is not sorted by time (TimeSeriesImputer will sort it)")
    if report["is_regular"] is False:
        reasons.append("irregular time steps — resample to a regular frequency first")
    if report["has_gaps"]:
        reasons.append("missing time steps — consider resampling")
    if report["has_duplicates"]:
        reasons.append("duplicate timestamps — aggregate or de-duplicate first")
    if report["missing_rate"] > 0.3:
        reasons.append(
            "missing rate is high (>30%) — imputation uncertainty will be large"
        )

    if reasons:
        report["recommendation"] = (
            "TimeSeriesImputer can handle this data, but note: "
            + "; ".join(reasons)
        )
    else:
        report["recommendation"] = (
            "Data appears well-suited for TimeSeriesImputer."
        )
    return report
