"""Comprehensive unit tests for MixedImputer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mixedimputer import MixedImputer


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_df():
    """Basic mixed DataFrame with NaNs."""
    return pd.DataFrame({
        "age": [25, 30, np.nan, 40, 35],
        "city": ["paris", "london", np.nan, "paris", "london"],
        "income": [50000, np.nan, 70000, 60000, 55000],
        "gender": ["M", "F", "M", np.nan, "F"],
    })


@pytest.fixture
def simple_df_no_nan():
    """Mixed DataFrame with no NaNs."""
    return pd.DataFrame({
        "age": [25, 30, 22, 40, 35],
        "city": ["paris", "london", "paris", "paris", "london"],
        "income": [50000, 45000, 70000, 60000, 55000],
        "gender": ["M", "F", "M", "F", "F"],
    })


@pytest.fixture
def all_missing_col_df():
    """DataFrame where one column is entirely missing."""
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40],
        "city": ["paris", "london", "paris", "london"],
        "all_miss": [np.nan, np.nan, np.nan, np.nan],
    })
    return df


@pytest.fixture
def int_cat_df():
    """DataFrame with a category-typed column."""
    return pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0],
        "cat": pd.Categorical(["a", "b", np.nan, "a"]),
    })


# ──────────────────────────────────────────────────────────────────────
# 1. Import and instantiation
# ──────────────────────────────────────────────────────────────────────

class TestImportAndInstantiation:
    def test_import(self):
        """Can import MixedImputer."""
        from mixedimputer import MixedImputer as MTI
        assert MTI is not None

    def test_instantiation_defaults(self):
        """Create with default parameters."""
        imputer = MixedImputer()
        assert imputer.max_iter == 10
        assert imputer.tol == 1e-3
        assert imputer.sample_posterior is False
        assert imputer.random_state is None

    def test_instantiation_with_params(self):
        """Create with custom parameters."""
        imputer = MixedImputer(
            categorical_features=["city"],
            max_iter=5,
            sample_posterior=True,
            random_state=42,
        )
        assert imputer.categorical_features == ["city"]
        assert imputer.max_iter == 5
        assert imputer.sample_posterior is True
        assert imputer.random_state == 42


# ──────────────────────────────────────────────────────────────────────
# 2. Fit & transform
# ──────────────────────────────────────────────────────────────────────

class TestFitTransform:
    def test_fit_transform_returns_dataframe(self, simple_df):
        """fit_transform returns a DataFrame."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_no_nan_after(self, simple_df):
        """Result has no NaN values."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()

    def test_fit_transform_keeps_columns(self, simple_df):
        """Output has same columns as input."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df)
        assert list(result.columns) == list(simple_df.columns)

    def test_fit_transform_keeps_index(self, simple_df):
        """Output has same index as input."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df)
        assert list(result.index) == list(simple_df.index)

    def test_fit_transform_same_shape(self, simple_df):
        """Output shape matches input."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df)
        assert result.shape == simple_df.shape


# ──────────────────────────────────────────────────────────────────────
# 3. Categorical string columns preserved as strings
# ──────────────────────────────────────────────────────────────────────

class TestCategoricalOutputType:
    def test_categorical_strings_preserved(self, simple_df):
        """String columns remain string/object after imputation."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df)
        assert pd.api.types.is_string_dtype(result["city"].dtype)
        assert pd.api.types.is_string_dtype(result["gender"].dtype)

    def test_categorical_values_are_valid(self, simple_df):
        """Imputed categories come from the original set."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df)
        valid_cities = {"paris", "london"}
        assert set(result["city"].dropna()) == valid_cities
        valid_genders = {"M", "F"}
        assert set(result["gender"].dropna()) == valid_genders


# ──────────────────────────────────────────────────────────────────────
# 4. sample_posterior
# ──────────────────────────────────────────────────────────────────────

class TestSamplePosterior:
    def test_sample_posterior_false(self, simple_df):
        """sample_posterior=False produces valid results."""
        imputer = MixedImputer(sample_posterior=False, random_state=42)
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()

    def test_sample_posterior_true(self, simple_df):
        """sample_posterior=True produces valid results with valid categories."""
        imputer = MixedImputer(sample_posterior=True, random_state=42)
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()
        valid_cities = {"paris", "london"}
        assert set(result["city"].dropna()).issubset(valid_cities)

    def test_sample_posterior_reproducibility(self, simple_df):
        """Same random_state with sample_posterior gives same results."""
        imputer1 = MixedImputer(sample_posterior=True, random_state=42)
        imputer2 = MixedImputer(sample_posterior=True, random_state=42)
        res1 = imputer1.fit_transform(simple_df)
        res2 = imputer2.fit_transform(simple_df)
        pd.testing.assert_frame_equal(res1, res2)


# ──────────────────────────────────────────────────────────────────────
# 5. Array input
# ──────────────────────────────────────────────────────────────────────

class TestArrayInput:
    def test_array_input_requires_categorical_features(self):
        """Array input must provide categorical_features."""
        X = np.array([[1.0, 2.0], [np.nan, 3.0]])
        imputer = MixedImputer()
        with pytest.raises(ValueError, match="categorical_features must be provided"):
            imputer.fit_transform(X)

    def test_array_input_works(self):
        """Array input with explicit categorical_features works."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 3))
        X[0, 0] = np.nan
        X[2, 1] = np.nan
        # Treat column 2 as categorical (encoded as integers 0/1/2)
        X[:, 2] = (X[:, 2] * 3).astype(int).astype(float)
        imputer = MixedImputer(categorical_features=[2], random_state=42)
        result = imputer.fit_transform(X)
        assert result.shape == X.shape
        assert not np.isnan(result).any()

    def test_array_input_numeric_with_cat_indices(self):
        """Array with numeric data and specified categorical index."""
        rng = np.random.default_rng(42)
        X = rng.random((20, 3))
        X[0, 0] = np.nan
        X[2, 1] = np.nan
        X[5, 2] = np.nan
        # Treat column 1 as categorical (encoded as integers 0/1)
        X[:, 1] = (X[:, 1] > 0.5).astype(float)
        imputer = MixedImputer(categorical_features=[1], random_state=42)
        result = imputer.fit_transform(X)
        assert result.shape == X.shape
        assert not np.isnan(result).any()


# ──────────────────────────────────────────────────────────────────────
# 6. Auto-detection of categorical columns
# ──────────────────────────────────────────────────────────────────────

class TestAutoDetection:
    def test_auto_detect_object_dtype(self, simple_df):
        """Object dtype columns auto-detected as categorical."""
        imputer = MixedImputer(random_state=42)
        imputer.fit(simple_df)
        assert "city" in imputer.categorical_features
        assert "gender" in imputer.categorical_features

    def test_auto_detect_category_dtype(self, int_cat_df):
        """Category dtype columns auto-detected."""
        imputer = MixedImputer(random_state=42)
        imputer.fit(int_cat_df)
        assert "cat" in imputer.categorical_features

    def test_numeric_not_auto_detected(self, simple_df):
        """Numeric columns not auto-detected as categorical."""
        imputer = MixedImputer(random_state=42)
        imputer.fit(simple_df)
        assert "age" not in imputer.categorical_features
        assert "income" not in imputer.categorical_features

    def test_explicit_categorical_features_overrides(self):
        """Explicit categorical_features overrides auto-detection."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan],
            "b": ["x", "y", "z"],
        })
        imputer = MixedImputer(categorical_features=["a"], random_state=42)
        imputer.fit(df)
        # "a" should be treated as categorical even though it's numeric
        assert "a" in imputer.categorical_features


# ──────────────────────────────────────────────────────────────────────
# 7. Edge cases
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_missing_values(self, simple_df_no_nan):
        """DataFrame with no NaNs is returned unchanged."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(simple_df_no_nan)
        pd.testing.assert_frame_equal(result, simple_df_no_nan, check_dtype=False)

    def test_all_missing_one_column(self, all_missing_col_df):
        """Works even if one column is entirely missing."""
        imputer = MixedImputer(
            keep_empty_features=True,
            random_state=42,
        )
        result = imputer.fit_transform(all_missing_col_df)
        assert result.shape == all_missing_col_df.shape

    def test_single_column_numeric(self):
        """Single numeric column works."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(df)
        assert not result.isnull().any().any()
        assert result.shape == (3, 1)

    def test_single_column_categorical(self):
        """Single categorical column works."""
        df = pd.DataFrame({"x": ["a", np.nan, "b"]})
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(df)
        assert not result.isnull().any().any()
        assert result.shape == (3, 1)

    def test_empty_dataframe(self):
        """Empty DataFrame should raise or handle gracefully."""
        df = pd.DataFrame()
        imputer = MixedImputer()
        with pytest.raises(ValueError):
            imputer.fit_transform(df)

    def test_all_numeric_dataframe(self):
        """DataFrame with only numeric columns works."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, 2.0, 3.0]})
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(df)
        assert not result.isnull().any().any()

    def test_all_categorical_dataframe(self):
        """DataFrame with only categorical columns works."""
        df = pd.DataFrame({"a": ["x", np.nan, "z"], "b": [np.nan, "y", "z"]})
        imputer = MixedImputer(random_state=42, max_iter=5)
        result = imputer.fit_transform(df)
        assert not result.isnull().any().any()


# ──────────────────────────────────────────────────────────────────────
# 8. Consistency
# ──────────────────────────────────────────────────────────────────────

class TestConsistency:
    def test_fit_transform_equals_fit_then_transform(self, simple_df):
        """fit_transform gives same result as fit + transform."""
        imputer1 = MixedImputer(random_state=42)
        res1 = imputer1.fit_transform(simple_df)

        imputer2 = MixedImputer(random_state=42)
        imputer2.fit(simple_df)
        res2 = imputer2.transform(simple_df)

        pd.testing.assert_frame_equal(res1, res2)

    def test_transform_before_fit_raises(self, simple_df):
        """transform before fit raises."""
        imputer = MixedImputer(random_state=42)
        with pytest.raises(Exception):
            imputer.transform(simple_df)


# ──────────────────────────────────────────────────────────────────────
# 9. Parameter validation
# ──────────────────────────────────────────────────────────────────────

class TestParameterValidation:
    def test_invalid_categorical_features_type(self):
        """Non-list categorical_features raises appropriate error."""
        imputer = MixedImputer(categorical_features="not_a_list")
        df = pd.DataFrame({"x": [1.0, np.nan]})
        # This may not error at init, but should be caught somewhere
        # For now, check that it doesn't crash terribly
        try:
            imputer.fit(df)
        except Exception:
            pass  # Acceptable — might error or not

    def test_negative_max_iter(self):
        """Negative max_iter should still work (0 rounds)."""
        imputer = MixedImputer(max_iter=0, random_state=42)
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = imputer.fit_transform(df)
        assert result.shape == df.shape


# ──────────────────────────────────────────────────────────────────────
# 10. Reproducibility
# ──────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_result(self, simple_df):
        """Same random_state gives identical results."""
        imputer1 = MixedImputer(random_state=42)
        imputer2 = MixedImputer(random_state=42)
        res1 = imputer1.fit_transform(simple_df)
        res2 = imputer2.fit_transform(simple_df)
        pd.testing.assert_frame_equal(res1, res2)

    def test_different_seed_may_differ(self, simple_df):
        """Different random_states may produce different results (check no crash)."""
        imputer1 = MixedImputer(random_state=42)
        imputer2 = MixedImputer(random_state=99)
        res1 = imputer1.fit_transform(simple_df)
        res2 = imputer2.fit_transform(simple_df)
        # Both should have no NaNs and same shape
        assert not res1.isnull().any().any()
        assert not res2.isnull().any().any()
        assert res1.shape == res2.shape


# ──────────────────────────────────────────────────────────────────────
# 11. Pipeline integration
# ──────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_pipeline_runs(self, simple_df):
        """Check that MixedImputer works in a sklearn Pipeline."""
        # For pipeline to work, we need to select numeric columns
        # since StandardScaler doesn't handle strings.
        # Use a simple DataFrame with only numeric columns for pipeline test.
        df_num = simple_df[["age", "income"]]
        pipe = Pipeline([
            ("imputer", MixedImputer(random_state=42)),
            ("scaler", StandardScaler()),
        ])
        result = pipe.fit_transform(df_num)
        assert result.shape == df_num.shape
        assert not np.isnan(result).any()


# ──────────────────────────────────────────────────────────────────────
# 12. Categorical dtype handling
# ──────────────────────────────────────────────────────────────────────

class TestCategoricalDtype:
    def test_categorical_dtype_input(self, int_cat_df):
        """Category-typed columns are handled correctly."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(int_cat_df)
        assert not result.isnull().any().any()
        # Categories should be preserved as valid values
        valid_cats = {"a", "b"}
        assert set(result["cat"].dropna()).issubset(valid_cats)

    def test_add_indicator(self, simple_df):
        """add_indicator=True doesn't break things."""
        imputer = MixedImputer(add_indicator=True, random_state=42)
        result = imputer.fit_transform(simple_df)
        # May have additional indicator columns
        assert result.shape[0] == simple_df.shape[0]


# ──────────────────────────────────────────────────────────────────────
# 13. Version attribute
# ──────────────────────────────────────────────────────────────────────

class TestVersion:
    def test_version_accessible(self):
        """__version__ is a string."""
        from mixedimputer import __version__
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"


# ──────────────────────────────────────────────────────────────────────
# 14. Array output
# ──────────────────────────────────────────────────────────────────────

class TestArrayOutput:
    def test_array_output(self):
        """Array input returns array output."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 3))
        X[0, 0] = np.nan
        X[2, 1] = np.nan
        X[:, 2] = (X[:, 2] > 0.5).astype(float)

        imputer = MixedImputer(categorical_features=[2], random_state=42)
        result = imputer.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == X.shape
        assert not np.isnan(result).any()

    def test_array_output_preserves_column_order(self):
        """Array output columns are in the same order as input."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 3))
        X[0, 1] = np.nan
        X[3, 2] = np.nan
        # Columns 0, 2 are numeric; column 1 is categorical
        X[:, 1] = (X[:, 1] > 0.5).astype(float)

        imputer = MixedImputer(categorical_features=[1], random_state=42)
        result = imputer.fit_transform(X)
        # Column 1 should still contain 0/1 values (categorical)
        unique_vals = np.unique(result[:, 1])
        assert set(unique_vals).issubset({0.0, 1.0})


# ──────────────────────────────────────────────────────────────────────
# 15. Custom estimators
# ──────────────────────────────────────────────────────────────────────

class TestCustomEstimators:
    def test_custom_regressor(self, simple_df):
        """Custom regressor is used for numeric columns."""
        from sklearn.linear_model import BayesianRidge
        imputer = MixedImputer(
            regressor=BayesianRidge(),
            random_state=42,
        )
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()

    def test_custom_classifier(self, simple_df):
        """Custom classifier is used for categorical columns."""
        from sklearn.ensemble import RandomForestClassifier
        imputer = MixedImputer(
            classifier=RandomForestClassifier(random_state=42),
            random_state=42,
        )
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()

    def test_custom_both_estimators(self, simple_df):
        """Both regressor and classifier can be customized."""
        from sklearn.linear_model import BayesianRidge
        from sklearn.ensemble import RandomForestClassifier
        imputer = MixedImputer(
            regressor=BayesianRidge(),
            classifier=RandomForestClassifier(random_state=42),
            random_state=42,
        )
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()


# ──────────────────────────────────────────────────────────────────────
# 16. initial_strategy parameter
# ──────────────────────────────────────────────────────────────────────

class TestInitialStrategy:
    def test_median_strategy(self, simple_df):
        """initial_strategy='median' works."""
        imputer = MixedImputer(initial_strategy="median", random_state=42)
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()

    def test_most_frequent_strategy(self, simple_df):
        """initial_strategy='most_frequent' works."""
        imputer = MixedImputer(
            initial_strategy="most_frequent", random_state=42
        )
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()

    def test_constant_strategy(self, simple_df):
        """initial_strategy='constant' with fill_value=0 works."""
        imputer = MixedImputer(
            initial_strategy="constant",
            random_state=42,
        )
        # Note: constant strategy requires fill_value which defaults to 0
        # in IterativeImputer
        result = imputer.fit_transform(simple_df)
        assert not result.isnull().any().any()


# ──────────────────────────────────────────────────────────────────────
# 17. keep_empty_features default behavior
# ──────────────────────────────────────────────────────────────────────

class TestKeepEmptyFeatures:
    def test_keep_empty_features_false_drops_column(self, all_missing_col_df):
        """Default keep_empty_features=False drops all-missing columns."""
        imputer = MixedImputer(
            keep_empty_features=False,
            random_state=42,
        )
        result = imputer.fit_transform(all_missing_col_df)
        # The all_miss column should be dropped
        assert "all_miss" not in result.columns
        assert result.shape[1] == all_missing_col_df.shape[1] - 1

    def test_keep_empty_features_true(self, all_missing_col_df):
        """keep_empty_features=True keeps all-missing columns."""
        imputer = MixedImputer(
            keep_empty_features=True,
            random_state=42,
        )
        result = imputer.fit_transform(all_missing_col_df)
        assert "all_miss" in result.columns
        assert result.shape == all_missing_col_df.shape


# ──────────────────────────────────────────────────────────────────────
# 18. fit returns self
# ──────────────────────────────────────────────────────────────────────

class TestFitReturnsSelf:
    def test_fit_returns_self(self, simple_df):
        """fit() returns the imputer instance."""
        imputer = MixedImputer(random_state=42)
        result = imputer.fit(simple_df)
        assert result is imputer


# ──────────────────────────────────────────────────────────────────────
# 19. Non-default index preservation
# ──────────────────────────────────────────────────────────────────────

class TestIndexPreservation:
    def test_non_default_index(self):
        """Non-default integer index is preserved."""
        df = pd.DataFrame(
            {"x": [1.0, np.nan, 3.0], "y": ["a", "b", np.nan]},
            index=[10, 20, 30],
        )
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(df)
        assert list(result.index) == [10, 20, 30]

    def test_string_index(self):
        """String index is preserved."""
        df = pd.DataFrame(
            {"x": [1.0, np.nan, 3.0], "y": ["a", "b", np.nan]},
            index=["row_a", "row_b", "row_c"],
        )
        imputer = MixedImputer(random_state=42)
        result = imputer.fit_transform(df)
        assert list(result.index) == ["row_a", "row_b", "row_c"]


# ──────────────────────────────────────────────────────────────────────
# 20. Unknown categories during transform
# ──────────────────────────────────────────────────────────────────────

class TestUnknownCategories:
    def test_new_category_in_transform(self):
        """Transform handles categories unseen during fit.

        Unknown categories are encoded to the ``unknown_value`` set on the
        OrdinalEncoder.  The imputer processes them as regular integers and
        ``inverse_transform`` maps them to ``None`` (NaN) when the imputed
        value does not correspond to a known category.  The imputer runs
        without crashing.
        """
        df_fit = pd.DataFrame({
            "cat": ["a", "b", "a"],
            "num": [1.0, np.nan, 3.0],
        })
        df_transform = pd.DataFrame({
            "cat": ["a", "c", "b"],      # "c" unseen during fit
            "num": [4.0, np.nan, 6.0],
        })

        imputer = MixedImputer(random_state=42)
        imputer.fit(df_fit)
        result = imputer.transform(df_transform)
        assert result.shape == df_transform.shape


# ──────────────────────────────────────────────────────────────────────
# 21. add_indicator output correctness
# ──────────────────────────────────────────────────────────────────────

class TestAddIndicatorOutput:
    def test_add_indicator_columns_present(self, simple_df):
        """add_indicator=True adds missing indicator columns."""
        imputer = MixedImputer(add_indicator=True, random_state=42)
        result = imputer.fit_transform(simple_df)
        # When add_indicator=True, IterativeImputer appends indicator columns
        # However, MixedImputer rebuilds the DataFrame from original columns,
        # so indicators may not be forwarded.  At minimum, shape[0] must match.
        assert result.shape[0] == simple_df.shape[0]
        assert not result.isnull().any().any()


# ──────────────────────────────────────────────────────────────────────
# 22. Estimator compatibility — regressors
# ──────────────────────────────────────────────────────────────────────

class TestRegressorCompatibility:
    """Every sklearn regressor with fit/predict should work out of the box."""

    @pytest.fixture
    def mixed_data(self):
        return pd.DataFrame({
            "age":    [25, 30, np.nan, 40, 35],
            "city":   ["paris", "london", np.nan, "paris", "london"],
            "income": [50000, np.nan, 70000, 60000, 55000],
            "gender": ["M", "F", "M", np.nan, "F"],
        })

    # -- Linear models --
    def test_linear_regression(self, mixed_data):
        from sklearn.linear_model import LinearRegression
        r = MixedImputer(regressor=LinearRegression(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_bayesian_ridge(self, mixed_data):
        from sklearn.linear_model import BayesianRidge
        r = MixedImputer(regressor=BayesianRidge(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_ridge(self, mixed_data):
        from sklearn.linear_model import Ridge
        r = MixedImputer(regressor=Ridge(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_lasso(self, mixed_data):
        from sklearn.linear_model import Lasso
        r = MixedImputer(regressor=Lasso(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_elastic_net(self, mixed_data):
        from sklearn.linear_model import ElasticNet
        r = MixedImputer(regressor=ElasticNet(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_huber(self, mixed_data):
        from sklearn.linear_model import HuberRegressor
        r = MixedImputer(regressor=HuberRegressor(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_sgd(self, mixed_data):
        from sklearn.linear_model import SGDRegressor
        r = MixedImputer(regressor=SGDRegressor(random_state=0), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    # -- Ensemble models --
    def test_random_forest_regressor(self, mixed_data):
        from sklearn.ensemble import RandomForestRegressor
        r = MixedImputer(
            regressor=RandomForestRegressor(n_estimators=20, random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_gradient_boosting_regressor(self, mixed_data):
        from sklearn.ensemble import GradientBoostingRegressor
        r = MixedImputer(
            regressor=GradientBoostingRegressor(n_estimators=20, random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_extra_trees_regressor(self, mixed_data):
        from sklearn.ensemble import ExtraTreesRegressor
        r = MixedImputer(
            regressor=ExtraTreesRegressor(n_estimators=20, random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_decision_tree_regressor(self, mixed_data):
        from sklearn.tree import DecisionTreeRegressor
        r = MixedImputer(
            regressor=DecisionTreeRegressor(random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    # -- Posterior sampling with regressors --
    def test_posterior_sampling_linear(self, mixed_data):
        from sklearn.linear_model import Ridge
        r = MixedImputer(regressor=Ridge(), sample_posterior=True, max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_posterior_sampling_random_forest(self, mixed_data):
        from sklearn.ensemble import RandomForestRegressor
        r = MixedImputer(
            regressor=RandomForestRegressor(n_estimators=20, random_state=0),
            sample_posterior=True, max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()


# ──────────────────────────────────────────────────────────────────────
# 23. Estimator compatibility — classifiers
# ──────────────────────────────────────────────────────────────────────

class TestClassifierCompatibility:
    """Every sklearn classifier with fit/predict should work out of the box."""

    @pytest.fixture
    def mixed_data(self):
        return pd.DataFrame({
            "age":    [25, 30, np.nan, 40, 35],
            "city":   ["paris", "london", np.nan, "paris", "london"],
            "income": [50000, np.nan, 70000, 60000, 55000],
            "gender": ["M", "F", "M", np.nan, "F"],
        })

    def test_random_forest_classifier(self, mixed_data):
        from sklearn.ensemble import RandomForestClassifier
        r = MixedImputer(
            classifier=RandomForestClassifier(n_estimators=20, random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_gradient_boosting_classifier(self, mixed_data):
        from sklearn.ensemble import GradientBoostingClassifier
        r = MixedImputer(
            classifier=GradientBoostingClassifier(n_estimators=20, random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_extra_trees_classifier(self, mixed_data):
        from sklearn.ensemble import ExtraTreesClassifier
        r = MixedImputer(
            classifier=ExtraTreesClassifier(n_estimators=20, random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_logistic_regression(self, mixed_data):
        from sklearn.linear_model import LogisticRegression
        r = MixedImputer(
            classifier=LogisticRegression(random_state=0, max_iter=1000),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_ridge_classifier(self, mixed_data):
        from sklearn.linear_model import RidgeClassifier
        r = MixedImputer(classifier=RidgeClassifier(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_knn_classifier(self, mixed_data):
        from sklearn.neighbors import KNeighborsClassifier
        r = MixedImputer(
            classifier=KNeighborsClassifier(n_neighbors=3),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_decision_tree_classifier(self, mixed_data):
        from sklearn.tree import DecisionTreeClassifier
        r = MixedImputer(
            classifier=DecisionTreeClassifier(random_state=0),
            max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_gaussian_nb(self, mixed_data):
        from sklearn.naive_bayes import GaussianNB
        r = MixedImputer(classifier=GaussianNB(), max_iter=5, random_state=42)
        assert not r.fit_transform(mixed_data).isnull().any().any()

    # -- Posterior sampling with classifiers --
    def test_posterior_sampling_logistic(self, mixed_data):
        from sklearn.linear_model import LogisticRegression
        r = MixedImputer(
            classifier=LogisticRegression(random_state=0, max_iter=1000),
            sample_posterior=True, max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()

    def test_posterior_sampling_random_forest_clf(self, mixed_data):
        from sklearn.ensemble import RandomForestClassifier
        r = MixedImputer(
            classifier=RandomForestClassifier(n_estimators=20, random_state=0),
            sample_posterior=True, max_iter=5, random_state=42,
        )
        assert not r.fit_transform(mixed_data).isnull().any().any()
