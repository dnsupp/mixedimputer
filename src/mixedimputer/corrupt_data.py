"""
Data Corruption Module for MixedImputer
=======================================

Provides a :class:`DataCorrupter` to load datasets from various formats
(.csv, .xlsx, .arff), select columns for corruption, and introduce
missing values using MCAR, MAR, or MNAR mechanisms.

Uses the `pygrinder <https://github.com/WenjieDu/PyGrinder>`_ library
for generating realistic missing-data patterns.

Column-Type Classification
--------------------------
This module aligns with :class:`MixedImputer`'s type detection:

* **Numeric columns**: Columns whose pandas dtype passes
  ``pd.api.types.is_numeric_dtype`` (int, float, etc.). These are imputed
  via regression.
* **Nominal (categorical) columns**: Columns with ``object``, ``string``,
  or ``category`` dtype. These are imputed via classification.

.. important::
   **String columns** — YES, they fall under the nominal/categorical
   category for the MixedImputer.  They are encoded with
   ``OrdinalEncoder`` and imputed by a classifier.

   **Ordinal columns** — If stored as *strings* (e.g. ``"low"``,
   ``"medium"``, ``"high"``), they are treated as nominal (the
   MixedImputer does **not** preserve order information — it uses
   ``OrdinalEncoder`` internally, which assigns arbitrary integer codes).
   If stored as *integers* (e.g. 1, 2, 3), they are treated as
   **numeric**, so the order IS preserved via regression.

Usage::

    from mixedimputer.corrupt_data import DataCorrupter

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

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pygrinder import mcar as _pygrinder_mcar
from pygrinder import mar_logistic as _pygrinder_mar_logistic
from pygrinder import mnar_nonuniform as _pygrinder_mnar_nonuniform
from pygrinder import calc_missing_rate


# ---------------------------------------------------------------------------
# ARFF loader (via liac-arff)
# ---------------------------------------------------------------------------

def _read_arff(filepath: str) -> pd.DataFrame:
    """Load an ARFF file into a pandas DataFrame using ``liac-arff``.

    Parameters
    ----------
    filepath : str
        Path to the ``.arff`` file.

    Returns
    -------
    pd.DataFrame
    """
    import arff

    with open(filepath, "r", encoding="utf-8") as fh:
        data_dict = arff.load(fh)

    # data_dict["data"] is a list of rows; data_dict["attributes"] is a
    # list of (name, type) tuples.
    df = pd.DataFrame(
        data_dict["data"],
        columns=[attr[0] for attr in data_dict["attributes"]],
    )

    # Handle column types: liac-arff returns strings for nominal values.
    # Convert numeric columns to proper numeric types.
    for attr_name, attr_type in data_dict["attributes"]:
        if attr_name not in df.columns:
            continue
        if isinstance(attr_type, list) or attr_type.upper() == "STRING":
            # Nominal/string column — keep as string, replace '?' with NaN
            df[attr_name] = df[attr_name].replace({"?": np.nan, "": np.nan})
        elif attr_type.upper() in ("REAL", "INTEGER", "NUMERIC"):
            try:
                df[attr_name] = pd.to_numeric(df[attr_name], errors="coerce")
            except (ValueError, TypeError):
                pass

    return df


# ---------------------------------------------------------------------------
# File dispatcher
# ---------------------------------------------------------------------------

def _load_file(filepath: str, **kwargs) -> pd.DataFrame:
    """Load a dataset from .csv, .xlsx, or .arff."""
    fp = filepath.lower()
    if fp.endswith(".csv"):
        return pd.read_csv(filepath, **kwargs)
    elif fp.endswith((".xls", ".xlsx")):
        return pd.read_excel(filepath, **kwargs)
    elif fp.endswith(".arff"):
        return _read_arff(filepath)
    else:
        raise ValueError(
            f"Unsupported file extension: {filepath}. "
            f"Supported: .csv, .xlsx, .xls, .arff"
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
        ):
            cats.append(col)
    return cats


def _detect_numeric_columns(df: pd.DataFrame, nominal_cols: List[str]) -> List[str]:
    """Return column names that MixedImputer would treat as numeric."""
    nums = []
    for col in df.columns:
        if col not in nominal_cols and pd.api.types.is_numeric_dtype(df[col].dtype):
            nums.append(col)
    return nums


# ---------------------------------------------------------------------------
# Helpers for pygrinder integration
# ---------------------------------------------------------------------------

def _encode_nominal_to_numeric(series: pd.Series) -> np.ndarray:
    """Label-encode a nominal series to float for pygrinder consumption."""
    codes = pd.Categorical(series).codes.astype(float)
    codes[codes < 0] = np.nan
    return codes


def _compute_mask_from_corrupted(
    original_arr: np.ndarray,
    corrupted_arr: np.ndarray,
    already_missing: np.ndarray,
) -> np.ndarray:
    """Extract a boolean mask: True where pygrinder *introduced* NaN."""
    new_nan = np.isnan(corrupted_arr)
    return new_nan & ~already_missing


def _build_mar_feature_matrix(
    df: pd.DataFrame,
    numeric_cols_all: List[str],
    target_col: str,
    encoded_target: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Build a 2D numeric feature matrix for pygrinder.mar_logistic.

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
        Numeric feature matrix with imputed NaNs.
    target_idx : int
        Column index of the target in X.
    """
    if target_col in numeric_cols_all:
        X = df[numeric_cols_all].to_numpy(dtype=float)
        target_idx = numeric_cols_all.index(target_col)
    else:
        X_num = df[numeric_cols_all].to_numpy(dtype=float) if numeric_cols_all else np.empty((len(df), 0))
        X = np.column_stack([X_num, encoded_target.reshape(-1, 1)])
        target_idx = X.shape[1] - 1

    # Impute NaN in feature matrix with column means
    for c in range(X.shape[1]):
        col_mask = np.isnan(X[:, c])
        if col_mask.any():
            X[col_mask, c] = np.nanmean(X[:, c])

    return X, target_idx


# ---------------------------------------------------------------------------
# Missing-data mechanisms (powered by pygrinder)
# ---------------------------------------------------------------------------

def _apply_mcar(
    column_values: np.ndarray,
    fraction: float,
    already_missing: np.ndarray,
) -> np.ndarray:
    """MCAR via pygrinder.mcar: uniform random missingness."""
    n_total = len(column_values)
    n_available = n_total - already_missing.sum()
    if n_available == 0:
        return np.zeros(n_total, dtype=bool)

    X = column_values.reshape(-1, 1).copy()
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.nanmean(X)

    p_effective = fraction * n_available / n_total
    if p_effective <= 0:
        return np.zeros(n_total, dtype=bool)

    corrupted = _pygrinder_mcar(X, p=p_effective)
    return _compute_mask_from_corrupted(column_values, corrupted.ravel(), already_missing)


def _apply_mar(
    column_values: np.ndarray,
    all_features: np.ndarray,
    target_idx: int,
    fraction: float,
    already_missing: np.ndarray,
) -> np.ndarray:
    """MAR via pygrinder.mar_logistic.

    Missingness in the target column is determined by a logistic model
    on *all* features.  Only the mask for the target column is returned.
    """
    n_total = len(column_values)
    n_available = n_total - already_missing.sum()
    if n_available == 0:
        return np.zeros(n_total, dtype=bool)

    X = all_features.copy()
    corrupted = _pygrinder_mar_logistic(X, obs_rate=0.9, missing_rate=fraction)
    target_corrupted = corrupted[:, target_idx]
    return _compute_mask_from_corrupted(column_values, target_corrupted, already_missing)


def _apply_mnar(
    column_values: np.ndarray,
    fraction: float,
    already_missing: np.ndarray,
    extreme: str = "high",
) -> np.ndarray:
    """MNAR via pygrinder.mnar_nonuniform.

    Missingness depends on the column's own values.  Higher values are
    more likely to be missing by default.
    """
    n_total = len(column_values)
    n_available = n_total - already_missing.sum()
    if n_available == 0 or np.all(already_missing):
        return np.zeros(n_total, dtype=bool)

    vals = column_values.copy()
    if extreme == "low":
        vals = -vals

    X = vals.reshape(-1, 1, 1).copy()
    nan_mask_3d = np.isnan(X)
    if nan_mask_3d.any():
        X[nan_mask_3d] = np.nanmean(X)

    corrupted, _ = _pygrinder_mnar_nonuniform(X, p=fraction, increase_factor=0.5)
    corrupted_1d = corrupted.ravel()
    return _compute_mask_from_corrupted(column_values, corrupted_1d, already_missing)


# ---------------------------------------------------------------------------
# Main corrupter class
# ---------------------------------------------------------------------------

class DataCorrupter:
    """Introduce missing values into a dataset with fine-grained control.

    Uses the ``pygrinder`` library under the hood for generating
    realistic MCAR, MAR, and MNAR patterns.

    Parameters
    ----------
    mechanism : {"MCAR", "MAR", "MNAR"}, default="MCAR"
        The missing-data mechanism:

        * ``"MCAR"`` — Missing Completely At Random.  Every cell in the
          targeted columns has an equal chance of being set to NaN
          (via ``pygrinder.mcar``).
        * ``"MAR"`` — Missing At Random.  Missingness in a column depends
          on the *observed* values of other columns through a logistic
          model (via ``pygrinder.mar_logistic``).
        * ``"MNAR"`` — Missing Not At Random.  Missingness in a column
          depends on the column's *own* (soon-to-be-unobserved) values.
          Higher values are more likely to go missing by default
          (via ``pygrinder.mnar_nonuniform``); use
          ``mnar_extreme="low"`` for the opposite.

    corruption_fraction : float or Dict[str, float], default=0.10
        Fraction of values to corrupt (0–1).  If a single float, applied
        uniformly to all targeted columns.  If a dictionary mapping column
        names to fractions, each column gets its specified fraction.

    num_numeric : int or None, default=None
        Number of **numeric** columns to corrupt.  If ``None``, use
        ``numeric_columns`` (see below).  Set to 0 to skip numeric columns.

    num_nominal : int or None, default=None
        Number of **nominal** (string/categorical) columns to corrupt.
        If ``None``, use ``nominal_columns`` (see below).  Set to 0 to
        skip nominal columns.

    numeric_columns : List[str] or None, default=None
        Explicit list of numeric column names to corrupt.  Overrides
        ``num_numeric``.

    nominal_columns : List[str] or None, default=None
        Explicit list of nominal column names to corrupt.  Overrides
        ``num_nominal``.

    num_random_columns : int or None, default=None
        Number of **random** columns (any type) to corrupt.  Convenience
        shortcut — internally selects that many columns at random ignoring
        ``num_numeric`` / ``num_nominal``.

    mnar_extreme : {"high", "low"}, default="high"
        For MNAR: whether high or low values are more likely to go missing.

    random_state : int or None, default=None
        Seed for reproducibility (sets NumPy's global seed before each
        pygrinder call).

    Attributes
    ----------
    corrupted_columns_ : List[str]
        Column names that were actually corrupted (after ``corrupt()``).
    mask_ : pd.DataFrame
        Boolean mask of same shape as input — ``True`` where a value was
        set to NaN.
    original_df_ : pd.DataFrame
        The original, uncorrupted DataFrame (cached copy).
    """

    _VALID_MECHANISMS = {"MCAR", "MAR", "MNAR"}

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
        random_state: Optional[int] = None,
    ):
        if mechanism not in self._VALID_MECHANISMS:
            raise ValueError(
                f"mechanism must be one of {self._VALID_MECHANISMS}, "
                f"got '{mechanism}'"
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

        # Set after corruption
        self.corrupted_columns_: List[str] = []
        self.mask_: Optional[pd.DataFrame] = None
        self.original_df_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def corrupt(
        self,
        filepath_or_df: Union[str, pd.DataFrame],
        **load_kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load (if needed), copy, and corrupt a dataset.

        Parameters
        ----------
        filepath_or_df : str or pd.DataFrame
            Either a file path (``.csv``, ``.xlsx``, ``.arff``) or an
            already-loaded DataFrame.
        **load_kwargs
            Extra arguments forwarded to ``pd.read_csv`` or ``pd.read_excel``
            (ignored when a DataFrame is passed or for ``.arff`` files).

        Returns
        -------
        corrupted : pd.DataFrame
            The corrupted DataFrame.
        mask : pd.DataFrame
            Boolean mask (same shape) with ``True`` at corrupted positions.
        original : pd.DataFrame
            The original (uncorrupted) DataFrame.
        """
        # --- Load ---
        if isinstance(filepath_or_df, pd.DataFrame):
            df_original = filepath_or_df.copy()
        else:
            df_original = _load_file(filepath_or_df, **load_kwargs)

        self.original_df_ = df_original.copy()

        # --- Determine column types ---
        all_nominal = _detect_nominal_columns(df_original)
        all_numeric = _detect_numeric_columns(df_original, all_nominal)

        # --- Select columns to corrupt ---
        target_cols = self._select_target_columns(
            all_numeric=all_numeric,
            all_nominal=all_nominal,
            df=df_original,
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

        self.corrupted_columns_ = target_cols

        # --- Resolve corruption fractions ---
        frac_map = self._resolve_fractions(target_cols)

        # --- Set seed for pygrinder reproducibility ---
        if self.random_state is not None:
            np.random.seed(self.random_state)

        df_corrupted = df_original.copy()
        mask = pd.DataFrame(
            False, index=df_original.index, columns=df_original.columns
        )

        for col in target_cols:
            frac = frac_map.get(col, 0.10)
            if frac <= 0:
                continue

            is_numeric = col in all_numeric
            already_missing = df_original[col].isna().to_numpy()

            if is_numeric:
                col_values = df_original[col].to_numpy(dtype=float)
                col_values_for_grinder = col_values.copy()
            else:
                col_values_for_grinder = _encode_nominal_to_numeric(
                    df_original[col]
                )

            if self.mechanism == "MCAR":
                col_mask = _apply_mcar(col_values_for_grinder, frac, already_missing)
            elif self.mechanism == "MAR":
                X_mar, target_idx = _build_mar_feature_matrix(
                    df_original, all_numeric, col, col_values_for_grinder
                )
                col_mask = _apply_mar(
                    col_values_for_grinder, X_mar, target_idx, frac, already_missing
                )
            else:  # MNAR
                col_mask = _apply_mnar(
                    col_values_for_grinder, frac, already_missing,
                    extreme=self.mnar_extreme,
                )

            # Safety: never corrupt already-missing positions
            col_mask = col_mask & ~already_missing

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
    ) -> List[str]:
        """Determine which columns to corrupt based on the configuration."""
        rng = np.random.default_rng(self.random_state)

        # -- "num_random_columns" shortcut --
        if self.num_random_columns is not None:
            n = min(self.num_random_columns, len(df.columns))
            return list(rng.choice(df.columns.tolist(), size=n, replace=False))

        targets: List[str] = []

        # Numeric selection
        if self.numeric_columns is not None:
            for c in self.numeric_columns:
                if c not in all_numeric:
                    raise ValueError(
                        f"Column '{c}' is not a numeric column. "
                        f"Available numeric columns: {all_numeric}"
                    )
            targets.extend(self.numeric_columns)
        elif self.num_numeric is not None:
            if self.num_numeric > 0:
                if self.num_numeric > len(all_numeric):
                    raise ValueError(
                        f"Requested {self.num_numeric} numeric columns but "
                        f"only {len(all_numeric)} available: {all_numeric}"
                    )
                chosen = rng.choice(
                    all_numeric, size=self.num_numeric, replace=False
                )
                targets.extend(chosen.tolist())

        # Nominal selection
        if self.nominal_columns is not None:
            for c in self.nominal_columns:
                if c not in all_nominal:
                    raise ValueError(
                        f"Column '{c}' is not a nominal column. "
                        f"Available nominal columns: {all_nominal}"
                    )
            targets.extend(self.nominal_columns)
        elif self.num_nominal is not None:
            if self.num_nominal > 0:
                if self.num_nominal > len(all_nominal):
                    raise ValueError(
                        f"Requested {self.num_nominal} nominal columns but "
                        f"only {len(all_nominal)} available: {all_nominal}"
                    )
                chosen = rng.choice(
                    all_nominal, size=self.num_nominal, replace=False
                )
                targets.extend(chosen.tolist())

        # If nothing specified at all (all None), corrupt ALL columns.
        # If any was specified (even 0), use the explicit targets list.
        if (
            self.num_random_columns is None
            and self.num_numeric is None
            and self.num_nominal is None
            and self.numeric_columns is None
            and self.nominal_columns is None
        ):
            targets = df.columns.tolist()

        return targets

    def _resolve_fractions(self, target_cols: List[str]) -> Dict[str, float]:
        """Build a per-column fraction dictionary."""
        if isinstance(self.corruption_fraction, dict):
            for col in self.corruption_fraction:
                if col not in target_cols:
                    warnings.warn(
                        f"Column '{col}' in corruption_fraction dict is not "
                        f"in the target columns {target_cols}. Ignoring."
                    )
            return {
                col: self.corruption_fraction.get(col, 0.10)
                for col in target_cols
            }
        else:
            frac = float(self.corruption_fraction)
            if not 0.0 <= frac <= 1.0:
                raise ValueError(
                    f"corruption_fraction must be in [0, 1], got {frac}"
                )
            return {col: frac for col in target_cols}

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def describe_columns(self, filepath_or_df: Union[str, pd.DataFrame]) -> Dict:
        """Return a summary of column types in the dataset.

        Useful for deciding which columns to corrupt.

        Returns
        -------
        dict with keys:
            ``"numeric"`` — list of numeric column names,
            ``"nominal"`` — list of nominal column names,
            ``"all"`` — list of all column names,
            ``"dtypes"`` — dict of ``{column_name: dtype_string}``.
        """
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
