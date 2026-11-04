"""
Data Corruption Module for MixedImputer
=======================================

Provides a :class:`DataCorrupter` to load datasets from various formats
(.csv, .xlsx, .xls, .arff), select columns for corruption, and introduce
missing values using MCAR, MAR, or MNAR mechanisms.

The mechanisms are implemented directly with NumPy so that:

- the requested fraction of **observed** cells is targeted exactly
  (up to integer rounding),
- already-missing cells are never overwritten,
- the same ``random_state`` produces the same mask,
- no global NumPy/PyTorch RNG state is modified.

Column-Type Classification
-------------------------
This module aligns with :class:`MixedImputer`'s type detection:

* **Numeric columns**: dtypes that pass
  ``pd.api.types.is_numeric_dtype`` and are not boolean.
* **Nominal (categorical) columns**: ``object``, ``string``, ``category``,
  or boolean dtype.

Usage::

    from mixedimputer import DataCorrupter

    corrupter = DataCorrupter(
        mechanism="MCAR",
        corruption_fraction=0.10,
        num_numeric=2,
        num_nominal=2,
        random_state=42,
    )
    corrupted_df, mask_df, original_df = corrupter.corrupt("data/titanic.csv")
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ARFF / file loading
# ---------------------------------------------------------------------------

def _read_arff(filepath: str) -> pd.DataFrame:
    """Load an ARFF file into a pandas DataFrame using ``liac-arff``."""
    try:
        import arff
    except ImportError as exc:
        raise ImportError(
            "Reading .arff files requires the optional dependency 'liac-arff'. "
            "Install it with: pip install 'mixedimputer[io]'"
        ) from exc

    with open(filepath, "r", encoding="utf-8") as fh:
        data_dict = arff.load(fh)

    df = pd.DataFrame(
        data_dict["data"],
        columns=[attr[0] for attr in data_dict["attributes"]],
    )

    for attr_name, attr_type in data_dict["attributes"]:
        if attr_name not in df.columns:
            continue
        if isinstance(attr_type, list) or str(attr_type).upper() == "STRING":
            df[attr_name] = df[attr_name].replace({"?": np.nan, "": np.nan})
        elif str(attr_type).upper() in ("REAL", "INTEGER", "NUMERIC"):
            df[attr_name] = pd.to_numeric(df[attr_name], errors="coerce")

    return df


def _load_file(filepath: Union[str, os.PathLike], **kwargs) -> pd.DataFrame:
    """Load a dataset from .csv, .xlsx, .xls, or .arff."""
    path = os.fspath(filepath)
    lower = path.lower()

    if lower.endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    if lower.endswith((".xls", ".xlsx")):
        try:
            return pd.read_excel(path, **kwargs)
        except ImportError as exc:
            package = "openpyxl" if lower.endswith(".xlsx") else "xlrd"
            raise ImportError(
                f"Reading {os.path.splitext(path)[1]} files requires the "
                f"optional dependency '{package}'. Install it with: "
                "pip install 'mixedimputer[io]'"
            ) from exc
    if lower.endswith(".arff"):
        return _read_arff(path)
    raise ValueError(
        f"Unsupported file extension: {path}. "
        "Supported: .csv, .xlsx, .xls, .arff"
    )


# ---------------------------------------------------------------------------
# Column-type detection (mirrors MixedImputer logic)
# ---------------------------------------------------------------------------

def _detect_nominal_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that MixedImputer would treat as categorical."""
    cats = []
    for col, dtype in df.dtypes.items():
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            cats.append(col)
    return cats


def _detect_numeric_columns(
    df: pd.DataFrame, nominal_cols: List[str]
) -> List[str]:
    """Return column names that MixedImputer would treat as numeric."""
    nums = []
    for col in df.columns:
        if (
            col not in nominal_cols
            and pd.api.types.is_numeric_dtype(df[col].dtype)
            and not pd.api.types.is_bool_dtype(df[col].dtype)
        ):
            nums.append(col)
    return nums


def _encode_nominal_to_numeric(series: pd.Series) -> np.ndarray:
    """Label-encode a nominal series to float codes for scoring."""
    codes = pd.Categorical(series).codes.astype(float)
    codes[codes < 0] = np.nan
    return codes


def _numeric_values(series: pd.Series) -> np.ndarray:
    """Return a numeric float array preserving NaN for missing values."""
    return series.to_numpy(dtype=float, na_value=np.nan)


def _make_rng(random_state):
    """Create a local Generator without mutating global RNG state."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        # A RandomState is not a documented default_rng seed in NumPy 1.x.
        seed = int(random_state.randint(0, 2**32 - 1))
        return np.random.default_rng(seed)
    return np.random.default_rng(random_state)


# ---------------------------------------------------------------------------
# Mechanism helpers
# ---------------------------------------------------------------------------

def _validate_fraction(fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"corruption_fraction must be in [0, 1], got {fraction}"
        )
    return fraction


def _n_to_corrupt(fraction: float, n_available: int) -> int:
    """Number of observed cells to mask for one column."""
    if n_available <= 0 or fraction <= 0:
        return 0
    k = int(round(fraction * n_available))
    if k < 1:
        k = 1
    return min(k, n_available)


def _as_label(value):
    """Convert a NumPy scalar to the corresponding Python object."""
    return value.item() if isinstance(value, np.generic) else value


def _standardize(values: np.ndarray) -> np.ndarray:
    """Mean-fill and standardize a 1-D array."""
    values = np.asarray(values, dtype=float)
    out = values.copy()
    missing = np.isnan(out)
    if missing.all():
        return np.zeros_like(out)
    mean = np.nanmean(out)
    out[missing] = mean
    std = np.nanstd(out)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(out)
    return (out - mean) / std


def _predictor_matrix(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: List[str],
    nominal_cols: List[str],
) -> Optional[np.ndarray]:
    """Build a standardized numeric matrix from all non-target columns."""
    columns = [c for c in df.columns if c != target_col]
    if not columns:
        return None

    feature_columns = []
    for col in columns:
        if col in numeric_cols:
            feature_columns.append(_numeric_values(df[col]))
        else:
            feature_columns.append(_encode_nominal_to_numeric(df[col]))

    matrix = np.column_stack(feature_columns) if feature_columns else None
    if matrix is None or matrix.size == 0:
        return None
    standardized = [
        _standardize(matrix[:, j]) for j in range(matrix.shape[1])
    ]
    return np.column_stack(standardized)


def _apply_mcar(available: np.ndarray, fraction: float, rng) -> np.ndarray:
    """MCAR: choose a uniform random subset of observed cells."""
    k = _n_to_corrupt(fraction, len(available))
    if k == 0:
        return available[:0]
    return rng.choice(available, size=k, replace=False)


def _apply_mar(
    predictors: np.ndarray,
    available: np.ndarray,
    fraction: float,
    rng,
) -> np.ndarray:
    """MAR: mask cells with high scores from a model of other columns."""
    k = _n_to_corrupt(fraction, len(available))
    if k == 0:
        return available[:0]
    n_features = predictors.shape[1]
    coefficients = rng.normal(size=n_features)
    scores = predictors @ coefficients
    scores = scores + rng.normal(scale=0.5, size=len(predictors))
    order = np.argsort(-scores[available], kind="mergesort")
    return available[order[:k]]


def _apply_mnar(
    values: np.ndarray,
    available: np.ndarray,
    fraction: float,
    rng,
    extreme: str,
) -> np.ndarray:
    """MNAR: mask cells with high/low values of the target itself."""
    k = _n_to_corrupt(fraction, len(available))
    if k == 0:
        return available[:0]
    standardized = _standardize(values)
    direction = 1.0 if extreme == "high" else -1.0
    scores = direction * standardized
    scores = scores + rng.normal(scale=0.5, size=len(values))
    order = np.argsort(-scores[available], kind="mergesort")
    return available[order[:k]]


# ---------------------------------------------------------------------------
# Main corrupter class
# ---------------------------------------------------------------------------

class DataCorrupter:
    """Introduce missing values into a dataset with fine-grained control.

    Parameters
    ----------
    mechanism : {"MCAR", "MAR", "MNAR"}, default="MCAR"
        - ``"MCAR"``: uniform random masking of observed cells.
        - ``"MAR"``: masking probability depends on the other columns
          through a random linear score.
        - ``"MNAR"``: masking probability depends on the target's own
          values; ``mnar_extreme="high"`` selects high values,
          ``"low"`` selects low values.

    corruption_fraction : float or Dict[str, float], default=0.10
        Fraction of **observed** values to corrupt per target column.
        Dictionary values override the fraction per column.

    num_numeric : int or None, default=None
        Number of numeric columns to corrupt.  ``None`` means "use
        numeric_columns".
    num_nominal : int or None, default=None
        Number of nominal columns to corrupt.  ``None`` means "use
        nominal_columns".
    numeric_columns : List[str] or None
        Explicit numeric target columns.
    nominal_columns : List[str] or None
        Explicit nominal target columns.
    num_random_columns : int or None
        Convenience shortcut: choose this many arbitrary columns.
    mnar_extreme : {"high", "low"}, default="high"
        Whether high or low target values are more likely to be masked.
    random_state : int, RandomState, or None
        Seed/state for all random choices.  No global RNG state is changed.

    Attributes
    ----------
    corrupted_columns_ : List[str]
    mask_ : pd.DataFrame
    original_df_ : pd.DataFrame
    """

    _VALID_MECHANISMS = {"MCAR", "MAR", "MNAR"}
    _VALID_EXTREMES = {"high", "low"}

    def __init__(
        self,
        mechanism: str = "MCAR",
        corruption_fraction: Union[float, Dict[str, float]] = 0.10,
        num_numeric: Optional[int] = None,
        num_nominal: Optional[int] = None,
        numeric_columns: Optional[List[str]] = None,
        nominal_columns: Optional[List[str]] = None,
        num_random_columns: Optional[int] = None,
        mnar_extreme: str = "high",
        random_state=None,
    ):
        if mechanism not in self._VALID_MECHANISMS:
            raise ValueError(
                f"mechanism must be one of {sorted(self._VALID_MECHANISMS)}, "
                f"got {mechanism!r}"
            )
        if mnar_extreme not in self._VALID_EXTREMES:
            raise ValueError(
                f"mnar_extreme must be one of {sorted(self._VALID_EXTREMES)}, "
                f"got {mnar_extreme!r}"
            )
        for name, value in (
            ("num_numeric", num_numeric),
            ("num_nominal", num_nominal),
            ("num_random_columns", num_random_columns),
        ):
            if value is not None and (
                not isinstance(value, (int, np.integer)) or int(value) < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer or None; "
                    f"got {value!r}"
                )

        self.mechanism = mechanism
        self.corruption_fraction = corruption_fraction
        self.num_numeric = num_numeric
        self.num_nominal = num_nominal
        self.numeric_columns = numeric_columns
        self.nominal_columns = nominal_columns
        self.num_random_columns = num_random_columns
        self.mnar_extreme = mnar_extreme
        self.random_state = random_state

        self.corrupted_columns_: List[str] = []
        self.mask_: Optional[pd.DataFrame] = None
        self.original_df_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def corrupt(
        self,
        filepath_or_df: Union[str, os.PathLike, pd.DataFrame],
        **load_kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load (if needed), copy, and corrupt a dataset.

        ``corruption_fraction`` is applied to the observed (non-NaN) cells of
        each target column.  Already-missing cells are never changed.
        """
        if isinstance(filepath_or_df, pd.DataFrame):
            df_original = filepath_or_df.copy()
        else:
            df_original = _load_file(filepath_or_df, **load_kwargs)

        if len(df_original.columns) == 0:
            raise ValueError("Cannot corrupt a DataFrame with no columns.")
        if len(set(df_original.columns)) != len(df_original.columns):
            raise ValueError("DataFrame has duplicate column names.")

        self.original_df_ = df_original.copy()
        all_nominal = _detect_nominal_columns(df_original)
        all_numeric = _detect_numeric_columns(df_original, all_nominal)

        rng = _make_rng(self.random_state)
        target_cols = self._select_target_columns(
            all_numeric=all_numeric,
            all_nominal=all_nominal,
            df=df_original,
            rng=rng,
        )

        if not target_cols:
            warnings.warn(
                "No columns selected for corruption. Returning uncorrupted data."
            )
            mask = pd.DataFrame(
                False, index=df_original.index, columns=df_original.columns
            )
            self.mask_ = mask
            self.corrupted_columns_ = []
            return df_original.copy(), mask, df_original.copy()

        self.corrupted_columns_ = list(target_cols)
        frac_map = self._resolve_fractions(target_cols)

        df_corrupted = df_original.copy()
        mask = pd.DataFrame(
            False, index=df_original.index, columns=df_original.columns
        )

        for col in target_cols:
            frac = frac_map[col]
            if frac <= 0:
                continue

            already_missing = df_original[col].isna().to_numpy()
            available = np.flatnonzero(~already_missing)
            if len(available) == 0:
                warnings.warn(
                    f"Column {col!r} has no observed values; skipping."
                )
                continue

            is_numeric = col in all_numeric
            if is_numeric:
                values = _numeric_values(df_original[col])
            else:
                values = _encode_nominal_to_numeric(df_original[col])

            if self.mechanism == "MCAR":
                corrupted_indices = _apply_mcar(available, frac, rng)
            elif self.mechanism == "MAR":
                predictors = _predictor_matrix(
                    df_original, col, all_numeric, all_nominal
                )
                if predictors is None:
                    raise ValueError(
                        "MAR requires at least one other column to use as a "
                        "predictor; this DataFrame has no other columns."
                    )
                corrupted_indices = _apply_mar(
                    predictors, available, frac, rng
                )
            else:  # MNAR
                corrupted_indices = _apply_mnar(
                    values,
                    available,
                    frac,
                    rng,
                    extreme=self.mnar_extreme,
                )

            corrupted_indices = np.asarray(corrupted_indices, dtype=int)
            if len(corrupted_indices) == 0:
                continue
            col_mask = np.zeros(len(df_original), dtype=bool)
            col_mask[corrupted_indices] = True
            col_mask &= ~already_missing

            df_corrupted.loc[col_mask, col] = np.nan
            mask.loc[col_mask, col] = True

        self.mask_ = mask
        return df_corrupted, mask, df_original.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_target_columns(
        self,
        all_numeric: List[str],
        all_nominal: List[str],
        df: pd.DataFrame,
        rng,
    ) -> List[str]:
        if self.num_random_columns is not None:
            n = min(int(self.num_random_columns), len(df.columns))
            if n == 0:
                return []
            return [
                _as_label(c) for c in rng.choice(
                    df.columns.to_numpy(dtype=object), size=n, replace=False
                )
            ]

        targets: List[str] = []

        if self.numeric_columns is not None:
            for col in self.numeric_columns:
                if col not in all_numeric:
                    raise ValueError(
                        f"Column {col!r} is not a numeric column. "
                        f"Available numeric columns: {all_numeric}"
                    )
            targets.extend(self.numeric_columns)
        elif self.num_numeric is not None and self.num_numeric > 0:
            if self.num_numeric > len(all_numeric):
                raise ValueError(
                    f"Requested {self.num_numeric} numeric columns but "
                    f"only {len(all_numeric)} available: {all_numeric}"
                )
            chosen = rng.choice(
                all_numeric, size=int(self.num_numeric), replace=False
            )
            targets.extend(_as_label(c) for c in chosen)

        if self.nominal_columns is not None:
            for col in self.nominal_columns:
                if col not in all_nominal:
                    raise ValueError(
                        f"Column {col!r} is not a nominal column. "
                        f"Available nominal columns: {all_nominal}"
                    )
            targets.extend(self.nominal_columns)
        elif self.num_nominal is not None and self.num_nominal > 0:
            if self.num_nominal > len(all_nominal):
                raise ValueError(
                    f"Requested {self.num_nominal} nominal columns but "
                    f"only {len(all_nominal)} available: {all_nominal}"
                )
            chosen = rng.choice(
                all_nominal, size=int(self.num_nominal), replace=False
            )
            targets.extend(_as_label(c) for c in chosen)

        if (
            self.num_random_columns is None
            and self.num_numeric is None
            and self.num_nominal is None
            and self.numeric_columns is None
            and self.nominal_columns is None
        ):
            targets = list(df.columns)

        # Preserve order and remove duplicates.
        seen = set()
        unique_targets = []
        for col in targets:
            if col not in seen:
                seen.add(col)
                unique_targets.append(col)
        return unique_targets

    def _resolve_fractions(self, target_cols: List[str]) -> Dict[str, float]:
        if isinstance(self.corruption_fraction, dict):
            fractions = {}
            for col, value in self.corruption_fraction.items():
                if col not in target_cols:
                    warnings.warn(
                        f"Column {col!r} in corruption_fraction dict is not "
                        f"in the target columns {target_cols}. Ignoring."
                    )
                fractions[col] = _validate_fraction(value)
            return {
                col: fractions.get(col, 0.10) for col in target_cols
            }
        fraction = _validate_fraction(self.corruption_fraction)
        return {col: fraction for col in target_cols}

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def describe_columns(
        self, filepath_or_df: Union[str, os.PathLike, pd.DataFrame]
    ) -> Dict:
        """Return a summary of column types in the dataset."""
        if isinstance(filepath_or_df, pd.DataFrame):
            df = filepath_or_df.copy()
        else:
            df = _load_file(filepath_or_df)

        nominal = _detect_nominal_columns(df)
        numeric = _detect_numeric_columns(df, nominal)
        return {
            "numeric": numeric,
            "nominal": nominal,
            "all": df.columns.tolist(),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        }
