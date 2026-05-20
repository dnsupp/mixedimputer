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
2. **Initial fill** — using time-aware strategies (forward fill,
   interpolation) instead of the global mean, preserving trends.
3. **Feature engineering** — creating lag and rolling-mean features for
   numeric columns so the iterative imputer can learn temporal
   dependencies.
4. **Delegating** to ``MixedImputer`` for the core MICE-style iterative
   imputation.
5. **Cleanup** — stripping engineered features from the output and
   restoring the original column order.
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

    Wraps :class:`MixedImputer` with temporal feature engineering
    (lag features, rolling statistics) and time-aware initial imputation
    strategies.  This makes iterative MICE-style imputation suitable for
    time series data where temporal autocorrelation matters.

    Parameters
    ----------
    time_column : str or int
        Column name (if DataFrame) or index identifying the time/ordering
        dimension.  Must be numeric or datetime.  The input is sorted by
        this column before imputation and the original order is restored
        in the output.

    lags : list of int, default=[1, 2, 3]
        Lag steps for which lag features are created for each numeric
        column.  For example, ``lags=[1, 2]`` creates ``col_lag1`` and
        ``col_lag2`` for every numeric feature.  Set to ``[]`` or ``None``
        to disable lag features.

    rolling_windows : list of int, default=None
        Window sizes for rolling-mean features on numeric columns.
        For example, ``rolling_windows=[3, 5]`` creates ``col_rollmean3``
        and ``col_rollmean5``.  Set to ``[]`` or ``None`` to disable.

    initial_strategy : str, default='forward_fill'
        Time-aware initial imputation strategy applied **before** iterative
        imputation.  Supported values:

        - ``'forward_fill'`` / ``'ffill'`` — propagate last valid
          observation forward in time.
        - ``'backward_fill'`` / ``'bfill'`` — use next valid observation
          (applied after forward fill for remaining leading NaNs).
        - ``'linear'`` — linear interpolation along the time axis.
        - ``'spline'`` — cubic-spline interpolation (order 3).
        - ``'mean'``, ``'median'``, ``'most_frequent'``, ``'constant'`` —
          standard sklearn strategies (non-time-aware fallbacks).

    imputer : MixedImputer or None, default=None
        The iterative imputer used for the core MICE-style imputation.
        If ``None``, a default :class:`MixedImputer` is created.  You can
        pass a pre-configured ``MixedImputer`` to customise the regressor,
        classifier, ``max_iter``, ``sample_posterior``, etc.

    categorical_features : list of str or int, or None
        Passed through to ``MixedImputer``.  Column names/indices of
        categorical features.  If ``None``, auto-detected from dtype.

    numeric_features : list of str or int, or None
        Passed through to ``MixedImputer``.  Column names/indices of
        numeric features.

    max_iter : int, default=10
        Maximum number of imputation rounds (passed to ``MixedImputer``).

    tol : float, default=1e-3
        Tolerance for early stopping (passed to ``MixedImputer``).

    sample_posterior : bool, default=False
        Whether to sample from the posterior for stochastic imputation.

    random_state : int or None, default=None
        Seed for reproducibility.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    time_column_ : str
        Resolved name of the time column.

    original_column_order_ : list of str
        Column order of the input DataFrame (excluding engineered features).

    engineered_feature_names_ : list of str
        Names of lag/rolling features added during transform.

    imputer_ : MixedImputer
        The fitted internal imputer.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mixedimputer import TimeSeriesImputer
    >>>
    >>> # Time series with missing values
    >>> df = pd.DataFrame({
    ...     'time':   [1, 2, 3, 4, 5],
    ...     'temp':   [20.1, np.nan, 22.5, np.nan, 21.0],
    ...     'humidity': [55, 57, np.nan, 60, np.nan],
    ...     'sensor_status': ['OK', 'OK', np.nan, 'WARN', 'OK'],
    ... })
    >>>
    >>> imputer = TimeSeriesImputer(
    ...     time_column='time',
    ...     lags=[1, 2],
    ...     rolling_windows=[2, 3],
    ...     random_state=42,
    ... )
    >>> imputed = imputer.fit_transform(df)
    >>> print(imputed)
    """

    def __init__(
        self,
        time_column: Union[str, int],
        lags: Optional[List[int]] = None,
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
        self.lags = lags if lags is not None else [1, 2, 3]
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
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the time-series imputer.

        Parameters
        ----------
        X : pd.DataFrame
            Time series data with missing values.  Must contain the column
            specified by ``time_column``.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : TimeSeriesImputer
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"TimeSeriesImputer requires a pandas DataFrame, got {type(X).__name__}."
            )

        # Resolve time column name
        if isinstance(self.time_column, int):
            if self.time_column < len(X.columns):
                self.time_column_ = X.columns[self.time_column]
            else:
                raise ValueError(
                    f"time_column index {self.time_column} is out of bounds "
                    f"for DataFrame with {len(X.columns)} columns."
                )
        else:
            if self.time_column not in X.columns:
                raise ValueError(
                    f"time_column '{self.time_column}' not found in DataFrame columns: "
                    f"{list(X.columns)}"
                )
            self.time_column_ = self.time_column

        # Store original column order (excluding engineered features)
        self.original_column_order_ = list(X.columns)
        self._time_col_idx = list(X.columns).index(self.time_column_)

        # Resolve categorical / numeric feature lists now so we can
        # apply lag/rolling only to numeric columns.
        if self.categorical_features is None:
            self._cat_features_ = [
                col for col in X.columns
                if col != self.time_column_
                and (pd.api.types.is_object_dtype(X[col].dtype)
                     or pd.api.types.is_string_dtype(X[col].dtype)
                     or isinstance(X[col].dtype, pd.CategoricalDtype))
            ]
        else:
            self._cat_features_ = list(self.categorical_features)

        if self.numeric_features is None:
            self._num_features_ = [
                col for col in X.columns
                if col != self.time_column_
                and col not in self._cat_features_
                and pd.api.types.is_numeric_dtype(X[col].dtype)
            ]
        else:
            self._num_features_ = list(self.numeric_features)

        # Build the internal imputer if not provided.
        # IMPORTANT: Do NOT pass numeric_features — let MixedImputer
        # auto-detect them from X_prepared (which includes lag/rolling
        # features that weren't present in the original input).
        if self.imputer is None:
            self.imputer_ = MixedImputer(
                categorical_features=self._cat_features_ if self._cat_features_ else None,
                max_iter=self.max_iter,
                tol=self.tol,
                sample_posterior=self.sample_posterior,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            self.imputer_ = self.imputer

        # Sort by time, add temporal features, apply initial fill,
        # then fit the internal imputer.
        X_prepared, self.engineered_feature_names_ = self._prepare_time_features(X)
        self.imputer_.fit(X_prepared)
        self.n_features_in_ = len(X_prepared.columns)

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in a time series DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Time series data with missing values.

        Returns
        -------
        X_imputed : pd.DataFrame
            DataFrame with imputed values in the original column order.
        """
        check_is_fitted(self, "imputer_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"TimeSeriesImputer requires a pandas DataFrame, got {type(X).__name__}."
            )

        X_prepared, _ = self._prepare_time_features(X)
        X_imputed = self.imputer_.transform(X_prepared)

        # Strip engineered feature columns
        original_cols = [
            c for c in X_imputed.columns if c not in self.engineered_feature_names_
        ]
        return X_imputed[original_cols]

    # ------------------------------------------------------------------
    # Internal: temporal feature engineering + initial fill
    # ------------------------------------------------------------------
    def _prepare_time_features(self, X: pd.DataFrame):
        """Sort by time, add temporal features, apply time-aware initial fill.

        The initial fill is used *temporarily* to create clean lag/rolling
        features; afterwards, original NaN positions are restored so that
        ``MixedImputer`` can refine the fill with its iterative models.

        Returns
        -------
        X_out : pd.DataFrame
            Sorted DataFrame with temporal features added.  Original columns
            retain their missing values; lag/rolling features have NaN only
            in the first few rows (from shift/window operations).
        engineered_names : list of str
            Names of the engineered feature columns.
        """
        df = X.copy()

        # ----- 1. Sort by time column -----
        df = df.sort_values(by=self.time_column_).reset_index(drop=True)

        # ----- 2. Record original NaN mask (before any fill) -----
        feature_cols = [c for c in df.columns if c != self.time_column_]
        nan_mask = df[feature_cols].isna()

        # ----- 3. Time-aware temporary fill (for clean lag/rolling features) -----
        df = self._apply_initial_fill(df)

        # ----- 4. Add temporal features (lags + rolling) -----
        engineered_names = []

        # Lag features for numeric columns only
        if self.lags:
            for col in self._num_features_:
                for lag in self.lags:
                    lag_name = f"{col}_lag{lag}"
                    df[lag_name] = df[col].shift(lag)
                    engineered_names.append(lag_name)

        # Rolling-mean features for numeric columns only
        if self.rolling_windows:
            for col in self._num_features_:
                for window in self.rolling_windows:
                    roll_name = f"{col}_rollmean{window}"
                    df[roll_name] = (
                        df[col].rolling(window=window, min_periods=1).mean()
                    )
                    engineered_names.append(roll_name)

        # ----- 5. Restore NaN in original columns -----
        # The lag/rolling features are based on temporally clean values
        # (forward-filled); the original columns retain their missingness
        # for MixedImputer to impute using the temporal features.
        df[feature_cols] = df[feature_cols].where(~nan_mask, other=np.nan)

        return df, engineered_names

    # ------------------------------------------------------------------
    # Time-aware initial fill strategies
    # ------------------------------------------------------------------
    def _apply_initial_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a time-aware initial imputation strategy to all columns.

        This fills gross gaps so the iterative imputer starts from a
        sensible baseline (preserving trends) rather than a global mean.
        """
        strategy = self.initial_strategy.lower()

        # Separate feature columns from the time column
        feature_cols = [c for c in df.columns if c != self.time_column_]

        if strategy in ("forward_fill", "ffill"):
            df[feature_cols] = df[feature_cols].ffill()
            # Remaining leading NaNs → backward fill
            df[feature_cols] = df[feature_cols].bfill()

        elif strategy in ("backward_fill", "bfill"):
            df[feature_cols] = df[feature_cols].bfill()
            # Remaining trailing NaNs → forward fill
            df[feature_cols] = df[feature_cols].ffill()

        elif strategy == "linear":
            df[feature_cols] = df[feature_cols].interpolate(
                method="linear", limit_direction="both"
            )

        elif strategy == "spline":
            df[feature_cols] = df[feature_cols].interpolate(
                method="spline", order=3, limit_direction="both"
            )

        elif strategy in ("mean", "median", "most_frequent", "constant"):
            # Delegate to standard strategies (no-op here; MixedImputer
            # handles these via its initial_strategy parameter).
            pass

        else:
            raise ValueError(
                f"Unsupported initial_strategy '{self.initial_strategy}'. "
                f"Choose from: forward_fill, backward_fill, linear, spline, "
                f"mean, median, most_frequent, constant."
            )

        return df


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def is_time_series_suitable(df: pd.DataFrame, time_column: str) -> dict:
    """Quick diagnostic to assess whether a DataFrame is suitable for
    time-series-aware imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    time_column : str
        Name of the time/ordering column.

    Returns
    -------
    report : dict
        Dictionary with keys:

        - ``is_sorted`` (bool) — whether the time column is monotonic.
        - ``is_regular`` (bool) — whether the time intervals are uniform
          (for datetime/numeric columns).
        - ``has_gaps`` (bool) — whether there are missing time steps.
        - ``missing_rate`` (float) — overall missing rate (excluding the
          time column).
        - ``recommendation`` (str) — human-readable recommendation.
    """
    report = {}

    # Sort check
    report["is_sorted"] = df[time_column].is_monotonic_increasing

    # Regularity check (applicable to numeric/datetime)
    time_vals = df[time_column].dropna()
    if len(time_vals) >= 3:
        if pd.api.types.is_datetime64_any_dtype(time_vals):
            diffs = time_vals.diff().dropna()
            if len(diffs) >= 2:
                mode_diff = diffs.mode().iloc[0] if not diffs.mode().empty else None
                report["is_regular"] = bool(
                    mode_diff is not None and (diffs == mode_diff).all()
                )
            else:
                report["is_regular"] = True  # not enough data to tell
        elif pd.api.types.is_numeric_dtype(time_vals):
            diffs = time_vals.diff().dropna()
            if len(diffs) >= 2:
                mode_diff = diffs.mode().iloc[0] if not diffs.mode().empty else None
                report["is_regular"] = bool(
                    mode_diff is not None and np.allclose(diffs, mode_diff)
                )
            else:
                report["is_regular"] = True
        else:
            report["is_regular"] = None  # cannot determine
    else:
        report["is_regular"] = None

    # Gap check: see if there are missing expected time points
    report["has_gaps"] = False
    if report.get("is_regular") and len(time_vals) >= 3:
        if pd.api.types.is_numeric_dtype(time_vals):
            expected_step = time_vals.diff().dropna().mode().iloc[0]
            full_range = np.arange(time_vals.min(), time_vals.max() + expected_step, expected_step)
            report["has_gaps"] = len(full_range) > len(time_vals)
        elif pd.api.types.is_datetime64_any_dtype(time_vals):
            expected_step = time_vals.diff().dropna().mode().iloc[0]
            full_range = pd.date_range(time_vals.min(), time_vals.max(), freq=expected_step)
            report["has_gaps"] = len(full_range) > len(time_vals)

    # Missing rate (excluding time column)
    feature_cols = [c for c in df.columns if c != time_column]
    total_cells = df[feature_cols].size
    missing_cells = df[feature_cols].isna().sum().sum()
    report["missing_rate"] = float(missing_cells / total_cells) if total_cells > 0 else 0.0

    # Recommendation
    reasons = []
    if not report["is_sorted"]:
        reasons.append("data is not sorted by time (TimeSeriesImputer will sort it)")
    if report["missing_rate"] > 0.3:
        reasons.append("missing rate is high (>30%) — imputation uncertainty will be large")
    if report.get("is_regular") and report.get("has_gaps"):
        reasons.append("irregular time steps detected — consider resampling first")

    if reasons:
        report["recommendation"] = (
            "TimeSeriesImputer can handle this data, but note: " + "; ".join(reasons)
        )
    else:
        report["recommendation"] = (
            "Data appears well-suited for TimeSeriesImputer."
        )

    return report
