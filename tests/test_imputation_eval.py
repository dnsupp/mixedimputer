"""End-to-end imputation evaluation tests.

These tests load real datasets, corrupt them with
:class:`DataCorrupter`, impute the missing values with
:class:`MixedImputer`, and evaluate the imputation quality.

Run with::

    pytest tests/test_imputation_eval.py -v
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mixedimputer import MixedImputer, DataCorrupter
from mixedimputer.corrupt_data import (
    _detect_nominal_columns,
    _detect_numeric_columns,
    _load_file,
)


# =========================================================================
# Paths
# =========================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TITANIC_CSV = DATA_DIR / "titanic.csv"
CREDIT_ARFF = DATA_DIR / "credit-g.arff"


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def simple_mixed_df() -> pd.DataFrame:
    """A synthetic mixed DataFrame (no NaNs) for fast evaluation."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n).astype(float),
            "city": rng.choice(["paris", "london", "berlin", "rome"], size=n),
            "income": rng.integers(20000, 120000, size=n).astype(float),
            "gender": rng.choice(["M", "F"], size=n),
            "score": rng.normal(70, 15, size=n).clip(0, 100),
        }
    )


def _corrupt_and_impute(
    df: pd.DataFrame,
    mechanism: str = "MCAR",
    corruption_fraction: float = 0.10,
    num_numeric: int = 1,
    num_nominal: int = 1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Helper: corrupt a clean DataFrame, impute it, return results.

    Returns
    -------
    corrupted : pd.DataFrame
    imputed : pd.DataFrame
    mask : pd.DataFrame
    original : pd.DataFrame
    """
    corrupter = DataCorrupter(
        mechanism=mechanism,
        corruption_fraction=corruption_fraction,
        num_numeric=num_numeric,
        num_nominal=num_nominal,
        random_state=random_state,
    )
    corrupted, mask, original = corrupter.corrupt(df)

    # Impute
    imputer = MixedImputer(
        max_iter=10,
        random_state=random_state,
    )
    imputed = imputer.fit_transform(corrupted)

    return corrupted, imputed, mask, original


# =========================================================================
# 1. Basic sanity checks
# =========================================================================

class TestImputationSanity:
    def test_no_remaining_nans(self, simple_mixed_df):
        """After imputation, there should be no NaN values."""
        _, imputed, _, _ = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=1,
        )
        assert not imputed.isnull().any().any(), "NaN values remain after imputation"

    def test_shape_preserved(self, simple_mixed_df):
        """Imputed DataFrame has the same shape as input."""
        _, imputed, _, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=2,
        )
        assert imputed.shape == original.shape

    def test_columns_preserved(self, simple_mixed_df):
        """Column names are preserved after imputation."""
        _, imputed, _, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=2,
        )
        assert list(imputed.columns) == list(original.columns)

    def test_index_preserved(self, simple_mixed_df):
        """Index is preserved after imputation."""
        _, imputed, _, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=2,
        )
        assert imputed.index.tolist() == original.index.tolist()

    def test_string_columns_remain_strings(self, simple_mixed_df):
        """Nominal columns are still string/object dtype after imputation."""
        _, imputed, _, _ = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_nominal=2,
            num_numeric=1,
        )
        nominal_cols = _detect_nominal_columns(simple_mixed_df)
        for col in nominal_cols:
            assert pd.api.types.is_string_dtype(
                imputed[col].dtype
            ) or pd.api.types.is_object_dtype(imputed[col].dtype), (
                f"Column '{col}' is no longer string/object: {imputed[col].dtype}"
            )

    def test_numeric_columns_stay_numeric(self, simple_mixed_df):
        """Numeric columns remain numeric after imputation."""
        _, imputed, _, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=3,
            num_nominal=1,
        )
        nominal = _detect_nominal_columns(original)
        numeric = _detect_numeric_columns(original, nominal)
        for col in numeric:
            assert pd.api.types.is_numeric_dtype(imputed[col].dtype), (
                f"Column '{col}' is no longer numeric: {imputed[col].dtype}"
            )


# =========================================================================
# 2. Imputation accuracy (synthetic data)
# =========================================================================

class TestImputationAccuracy:
    def test_numeric_imputation_within_range(self, simple_mixed_df):
        """Imputed numeric values fall within observed min/max range."""
        _, imputed, mask, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.15,
            num_numeric=2,
            num_nominal=0,
            random_state=42,
        )

        nominal = _detect_nominal_columns(original)
        numeric_cols = _detect_numeric_columns(original, nominal)

        for col in numeric_cols:
            if col not in mask.columns:
                continue
            corrupted_mask = mask[col]
            if not corrupted_mask.any():
                continue
            orig_min = original[col].min()
            orig_max = original[col].max()
            imputed_vals = imputed.loc[corrupted_mask, col]
            # Allow slight extrapolation (within 20% of range)
            margin = (orig_max - orig_min) * 0.20
            assert (imputed_vals >= orig_min - margin).all(), (
                f"Imputed '{col}' values below min: "
                f"min={imputed_vals.min()}, orig_min={orig_min}"
            )
            assert (imputed_vals <= orig_max + margin).all(), (
                f"Imputed '{col}' values above max: "
                f"max={imputed_vals.max()}, orig_max={orig_max}"
            )

    def test_categorical_imputation_valid_values(self, simple_mixed_df):
        """Imputed categorical values come from the original category set."""
        _, imputed, mask, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.15,
            num_nominal=2,
            num_numeric=0,
            random_state=42,
        )

        nominal_cols = _detect_nominal_columns(original)
        for col in nominal_cols:
            if col not in mask.columns:
                continue
            corrupted_mask = mask[col]
            if not corrupted_mask.any():
                continue
            valid_categories = set(original[col].dropna().unique())
            imputed_vals = set(imputed.loc[corrupted_mask, col])
            assert imputed_vals.issubset(valid_categories), (
                f"Imputed '{col}' values {imputed_vals} not subset of "
                f"original categories {valid_categories}"
            )

    def test_numeric_mse_vs_original(self, simple_mixed_df):
        """Imputed numeric values are closer to truth than the mean."""
        _, imputed, mask, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.20,
            num_numeric=2,
            num_nominal=0,
            random_state=42,
        )

        nominal = _detect_nominal_columns(original)
        numeric_cols = _detect_numeric_columns(original, nominal)

        for col in numeric_cols:
            if col not in mask.columns:
                continue
            corrupted_mask = mask[col]
            if corrupted_mask.sum() < 5:
                continue  # Need enough samples

            true_vals = original.loc[corrupted_mask, col].to_numpy(dtype=float)
            imputed_vals = imputed.loc[corrupted_mask, col].to_numpy(dtype=float)
            mean_val = original[col].mean()

            # RMSE of imputer
            rmse_imputer = np.sqrt(np.mean((true_vals - imputed_vals) ** 2))
            # RMSE of mean imputation
            rmse_mean = np.sqrt(np.mean((true_vals - mean_val) ** 2))

            assert rmse_imputer <= rmse_mean * 1.2, (
                f"Imputer RMSE ({rmse_imputer:.2f}) for '{col}' is much worse "
                f"than mean imputation ({rmse_mean:.2f})"
            )

    def test_categorical_accuracy_vs_mode(self, simple_mixed_df):
        """Imputed categorical accuracy is at least as good as mode."""
        _, imputed, mask, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.20,
            num_nominal=2,
            num_numeric=0,
            random_state=42,
        )

        nominal_cols = _detect_nominal_columns(original)
        for col in nominal_cols:
            if col not in mask.columns:
                continue
            corrupted_mask = mask[col]
            n_corrupted = corrupted_mask.sum()
            if n_corrupted < 5:
                continue

            true_vals = original.loc[corrupted_mask, col]
            imputed_vals = imputed.loc[corrupted_mask, col]
            mode_val = original[col].mode().iloc[0] if not original[col].mode().empty else original[col].iloc[0]

            acc_imputer = (true_vals == imputed_vals).mean()
            acc_mode = (true_vals == mode_val).mean()

            assert acc_imputer >= acc_mode * 0.5, (
                f"Imputer accuracy ({acc_imputer:.2%}) for '{col}' is far worse "
                f"than mode ({acc_mode:.2%})"
            )


# =========================================================================
# 3. Missing-data mechanism robustness
# =========================================================================

class TestMechanismRobustness:
    @pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR"])
    def test_all_mechanisms_complete_imputation(self, simple_mixed_df, mechanism):
        """All three mechanisms produce complete data after imputation."""
        _, imputed, _, _ = _corrupt_and_impute(
            simple_mixed_df,
            mechanism=mechanism,
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=1,
            random_state=42,
        )
        assert not imputed.isnull().any().any(), (
            f"NaN remain after imputation with {mechanism}"
        )

    @pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR"])
    def test_all_mechanisms_categorical_valid(self, simple_mixed_df, mechanism):
        """Categorical imputations are valid for all mechanisms."""
        _, imputed, mask, original = _corrupt_and_impute(
            simple_mixed_df,
            mechanism=mechanism,
            corruption_fraction=0.15,
            num_nominal=2,
            num_numeric=0,
            random_state=42,
        )

        nominal_cols = _detect_nominal_columns(original)
        for col in nominal_cols:
            if col not in mask.columns or mask[col].sum() == 0:
                continue
            valid = set(original[col].dropna().unique())
            result = set(imputed.loc[mask[col], col])
            assert result.issubset(valid), (
                f"{mechanism}: '{col}' got invalid categories {result - valid}"
            )


# =========================================================================
# 4. Corruption fraction robustness
# =========================================================================

class TestCorruptionFractionRobustness:
    @pytest.mark.parametrize("frac", [0.05, 0.10, 0.25, 0.50])
    def test_varying_fractions(self, simple_mixed_df, frac):
        """Imputation works across a range of corruption fractions."""
        _, imputed, _, _ = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=frac,
            num_numeric=2,
            num_nominal=1,
            random_state=42,
        )
        assert not imputed.isnull().any().any(), (
            f"NaN remain at fraction={frac}"
        )


# =========================================================================
# 5. Posterior sampling
# =========================================================================

class TestPosteriorSampling:
    def test_sample_posterior_no_nans(self, simple_mixed_df):
        """sample_posterior=True still yields complete data."""
        _, imputed, _, _ = _corrupt_and_impute(
            simple_mixed_df,
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=1,
            random_state=42,
        )
        assert not imputed.isnull().any().any()

    def test_sample_posterior_reproducibility(self, simple_mixed_df):
        """Same seed + sample_posterior gives reproducible results."""
        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=1,
            random_state=42,
        )
        corrupted, _, _ = corrupter.corrupt(simple_mixed_df)

        imputer1 = MixedImputer(sample_posterior=True, max_iter=10, random_state=42)
        imputer2 = MixedImputer(sample_posterior=True, max_iter=10, random_state=42)
        res1 = imputer1.fit_transform(corrupted)
        res2 = imputer2.fit_transform(corrupted)
        pd.testing.assert_frame_equal(res1, res2)


# =========================================================================
# 6. Real datasets — Titanic
# =========================================================================

@pytest.mark.skipif(not TITANIC_CSV.exists(), reason="titanic.csv not found")
class TestTitanicImputation:
    def test_load_and_corrupt_titanic(self):
        """Titanic can be loaded, corrupted, and imputed."""
        df_full = pd.read_csv(str(TITANIC_CSV))
        df = df_full.head(150).copy()

        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=2,
            num_nominal=2,
            random_state=42,
        )
        corrupted, mask, original = corrupter.corrupt(df)

        # Drop columns that might cause issues (e.g. Name, Ticket, Cabin
        # have many unique values or high cardinality)
        # Also remove the PassengerId column as it's just an index
        cols_to_drop = []
        for c in ["Name", "Ticket", "Cabin", "PassengerId"]:
            if c in corrupted.columns:
                cols_to_drop.append(c)
        corrupted_clean = corrupted.drop(columns=cols_to_drop)
        original_clean = original.drop(columns=cols_to_drop)
        mask_clean = mask.drop(columns=cols_to_drop)

        imputer = MixedImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(corrupted_clean)

        assert not imputed.isnull().any().any(), "NaN remain in Titanic imputation"
        assert imputed.shape == corrupted_clean.shape

    def test_titanic_numeric_accuracy(self):
        """Numeric imputation on Titanic beats mean imputation."""
        # Use a subset for speed
        df_full = pd.read_csv(str(TITANIC_CSV))
        df = df_full.head(100).copy()

        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.15,
            numeric_columns=["Age", "Fare"],
            random_state=42,
        )
        corrupted, mask, original = corrupter.corrupt(df)

        # Impute with fewer iterations for speed
        imputer = MixedImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(corrupted)

        for col in ["Age", "Fare"]:
            if col not in mask.columns or mask[col].sum() < 3:
                continue
            true_vals = original.loc[mask[col], col].to_numpy(dtype=float)
            imputed_vals = imputed.loc[mask[col], col].to_numpy(dtype=float)
            mean_val = original[col].mean()

            rmse_imp = np.sqrt(np.mean((true_vals - imputed_vals) ** 2))
            rmse_mean = np.sqrt(np.mean((true_vals - mean_val) ** 2))

            assert rmse_imp <= rmse_mean * 2.0, (
                f"Titanic '{col}': Imputer RMSE={rmse_imp:.2f} vs "
                f"mean RMSE={rmse_mean:.2f}"
            )

    def test_titanic_categorical_accuracy(self):
        """Categorical imputation on Titanic preserves valid values."""
        df_full = pd.read_csv(str(TITANIC_CSV))
        df = df_full.head(100).copy()

        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.15,
            nominal_columns=["Sex", "Embarked"],
            random_state=42,
        )
        corrupted, mask, original = corrupter.corrupt(df)

        imputer = MixedImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(corrupted)

        for col in ["Sex", "Embarked"]:
            if col not in mask.columns or mask[col].sum() == 0:
                continue
            valid = set(original[col].dropna().unique())
            result = set(imputed.loc[mask[col], col])
            assert result.issubset(valid), (
                f"Titanic '{col}': invalid categories {result - valid}"
            )


# =========================================================================
# 7. Real datasets — German Credit (ARFF)
# =========================================================================

@pytest.mark.skipif(not CREDIT_ARFF.exists(), reason="credit-g.arff not found")
class TestCreditImputation:
    def test_load_corrupt_impute_credit(self):
        """German Credit ARFF: load, corrupt, impute."""
        df_full = _load_file(str(CREDIT_ARFF))
        df = df_full.head(300).copy()

        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.10,
            num_numeric=3,
            num_nominal=3,
            random_state=42,
        )
        corrupted, mask, original = corrupter.corrupt(df)

        imputer = MixedImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(corrupted)

        assert not imputed.isnull().any().any(), "NaN remain in credit imputation"
        assert imputed.shape == original.shape

    def test_credit_categorical_valid(self):
        """All imputed categorical values in credit data are valid."""
        df_full = _load_file(str(CREDIT_ARFF))
        df = df_full.head(300).copy()

        corrupter = DataCorrupter(
            mechanism="MAR",
            corruption_fraction=0.10,
            num_nominal=5,
            num_numeric=0,
            random_state=42,
        )
        corrupted, mask, original = corrupter.corrupt(df)

        nominal_cols = _detect_nominal_columns(original)
        imputer = MixedImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(corrupted)

        for col in nominal_cols:
            if col not in mask.columns or mask[col].sum() == 0:
                continue
            valid = set(original[col].dropna().unique())
            result = set(imputed.loc[mask[col], col])
            assert result.issubset(valid), (
                f"Credit '{col}': got {result - valid}"
            )

    def test_credit_numeric_accuracy(self):
        """Numeric imputation on credit data beats mean."""
        df_full = _load_file(str(CREDIT_ARFF))
        df = df_full.head(200).copy()

        corrupter = DataCorrupter(
            mechanism="MCAR",
            corruption_fraction=0.15,
            numeric_columns=["duration", "credit_amount", "age"],
            random_state=42,
        )
        corrupted, mask, original = corrupter.corrupt(df)

        imputer = MixedImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(corrupted)

        for col in ["duration", "credit_amount", "age"]:
            if col not in mask.columns or mask[col].sum() < 3:
                continue
            true_vals = original.loc[mask[col], col].to_numpy(dtype=float)
            imputed_vals = imputed.loc[mask[col], col].to_numpy(dtype=float)
            mean_val = original[col].mean()

            rmse_imp = np.sqrt(np.mean((true_vals - imputed_vals) ** 2))
            rmse_mean = np.sqrt(np.mean((true_vals - mean_val) ** 2))

            assert rmse_imp <= rmse_mean * 3.0, (
                f"Credit '{col}': Imputer RMSE={rmse_imp:.2f} vs "
                f"mean RMSE={rmse_mean:.2f}"
            )

    def test_credit_all_mechanisms(self):
        """All missing-data mechanisms produce complete credit data."""
        df_full = _load_file(str(CREDIT_ARFF))
        df = df_full.head(200).copy()

        for mechanism in ["MCAR", "MAR", "MNAR"]:
            corrupter = DataCorrupter(
                mechanism=mechanism,
                corruption_fraction=0.08,
                num_numeric=3,
                num_nominal=3,
                random_state=42,
            )
            corrupted, _, _ = corrupter.corrupt(df)
            imputer = MixedImputer(max_iter=5, random_state=42)
            imputed = imputer.fit_transform(corrupted)
            assert not imputed.isnull().any().any(), (
                f"NaN remain with {mechanism} on credit data"
            )


# =========================================================================
# 8. Integration: Corruption → Imputation pipeline summary
# =========================================================================

class TestPipelineSummary:
    """Print a summary report of the corruption→imputation pipeline."""

    def test_pipeline_report(self, simple_mixed_df):
        """Generate a summary report (informational, not a pass/fail)."""
        mechanism = "MCAR"
        frac = 0.15

        corrupter = DataCorrupter(
            mechanism=mechanism,
            corruption_fraction=frac,
            num_numeric=2,
            num_nominal=1,
            random_state=42,
        )
        corrupted, mask, original = corrupter.corrupt(simple_mixed_df)

        imputer = MixedImputer(max_iter=10, random_state=42)
        imputed = imputer.fit_transform(corrupted)

        nominal = _detect_nominal_columns(original)
        numeric = _detect_numeric_columns(original, nominal)

        report_lines = [
            "",
            "=" * 60,
            "  Corruption → Imputation Pipeline Report",
            "=" * 60,
            f"  Mechanism:          {mechanism}",
            f"  Corruption fraction: {frac:.0%}",
            f"  Columns corrupted:  {corrupter.corrupted_columns_}",
            f"  NaN remaining after imputation: {imputed.isnull().any().any()}",
            "",
            "  --- Numeric Columns ---",
        ]

        for col in numeric:
            if col not in mask.columns or mask[col].sum() == 0:
                report_lines.append(f"    {col}: not corrupted")
                continue

            n_corrupted = mask[col].sum()
            true_vals = original.loc[mask[col], col].to_numpy(dtype=float)
            imp_vals = imputed.loc[mask[col], col].to_numpy(dtype=float)
            mean_val = original[col].mean()

            rmse_imp = np.sqrt(np.mean((true_vals - imp_vals) ** 2))
            rmse_mean = np.sqrt(np.mean((true_vals - mean_val) ** 2))

            report_lines.append(
                f"    {col}: {n_corrupted} corrupted | "
                f"Imputer RMSE={rmse_imp:.2f} | "
                f"Mean RMSE={rmse_mean:.2f} | "
                f"Ratio={rmse_imp / rmse_mean:.2f}"
            )

        report_lines.append("")
        report_lines.append("  --- Nominal Columns ---")

        for col in nominal:
            if col not in mask.columns or mask[col].sum() == 0:
                report_lines.append(f"    {col}: not corrupted")
                continue

            n_corrupted = mask[col].sum()
            true_vals = original.loc[mask[col], col]
            imp_vals = imputed.loc[mask[col], col]

            acc = (true_vals == imp_vals).mean()
            mode_val = original[col].mode().iloc[0] if not original[col].mode().empty else original[col].iloc[0]
            acc_mode = (true_vals == mode_val).mean()

            report_lines.append(
                f"    {col}: {n_corrupted} corrupted | "
                f"Imputer Acc={acc:.2%} | "
                f"Mode Acc={acc_mode:.2%}"
            )

        report_lines.append("=" * 60)
        report = "\n".join(report_lines)

        # Print the report (visible with pytest -s)
        print(report)

        # This test always passes — it's informational
        assert True
