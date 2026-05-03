"""
Mixed-type MICE imputer for scikit-learn.

Provides a single ``MixedTypeImputer`` transformer that handles DataFrames
containing both numerical and categorical (string) columns.  Internally it
encodes categoricals with ``OrdinalEncoder``, runs an iterative MICE-style
imputer that automatically chooses a regressor or classifier per column, and
decodes categoricals back to their original string values.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

# Unlock experimental feature (must be imported BEFORE IterativeImputer)
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import IterativeImputer
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted, check_array


# ----------------------------------------------------------------------
# _safe_assign was removed from sklearn.utils in recent versions;
# implement a minimal local replacement.
# ----------------------------------------------------------------------
def _safe_assign(X, values, *, row_indexer=None, column_indexer=None):
    """Assign values into X with numpy-compatible row/column indexers."""
    if row_indexer is None:
        row_indexer = slice(None)
    if column_indexer is None:
        column_indexer = slice(None)
    X[row_indexer, column_indexer] = values


# ----------------------------------------------------------------------
# 1. Internal mixed-type iterative imputer
# ----------------------------------------------------------------------
class _MixedTypeIterativeImputer(IterativeImputer):
    """
    IterativeImputer that uses a ``_ColumnTypeEstimator`` to switch between
    a regressor and a classifier depending on the target column type.

    This is a drop-in for scikit-learn's ``IterativeImputer`` that overrides
    ``_impute_one_feature`` to:

    - Use ``predict_proba`` for categorical target columns when posterior
      sampling is enabled.
    - Round categorical predictions back to integers.
    - Clip numeric predictions to the observed min/max range.
    - Support ``return_std`` from the regressor for posterior sampling on
      numeric columns.
    """

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        target_feature_index,
        predictor_feature_indices,
        estimator=None,
        fit_mode=True,
        params=None,
    ):
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, an already-fitted estimator must be provided."
            )
        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, target_feature_index]

        if fit_mode:
            X_train = _safe_indexing(
                _safe_indexing(X_filled, predictor_feature_indices, axis=1),
                ~missing_row_mask,
                axis=0,
            )
            y_train = _safe_indexing(
                _safe_indexing(X_filled, target_feature_index, axis=1),
                ~missing_row_mask,
                axis=0,
            )
            estimator.set_params(
                target_feature_index=target_feature_index,
                predictor_feature_indices=predictor_feature_indices,
            )
            if params is None:
                estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train, **params)

        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        X_test = _safe_indexing(
            _safe_indexing(X_filled, predictor_feature_indices, axis=1),
            missing_row_mask,
            axis=0,
        )

        target_is_categorical = estimator.is_categorical(target_feature_index)

        if target_is_categorical:
            if self.sample_posterior and hasattr(estimator.model_, "predict_proba"):
                proba = estimator.model_.predict_proba(X_test)
                imputed_values = _sample_from_probabilities(
                    proba, random_state=self.random_state_
                )
            else:
                imputed_values = estimator.predict(X_test)

            # Ensure integer output (same dtype as original)
            non_missing_vals = X_filled[~missing_row_mask, target_feature_index]
            target_dtype = non_missing_vals.dtype
            imputed_values = np.round(imputed_values).astype(target_dtype)

        else:
            if self.sample_posterior:
                mus, sigmas = estimator.predict(X_test, return_std=True)
                imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
                positive_sigmas = sigmas > 0
                imputed_values[~positive_sigmas] = mus[~positive_sigmas]
                mus_too_low = mus < self._min_value[target_feature_index]
                imputed_values[mus_too_low] = self._min_value[target_feature_index]
                mus_too_high = mus > self._max_value[target_feature_index]
                imputed_values[mus_too_high] = self._max_value[target_feature_index]
                inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
                mus = mus[inrange_mask]
                sigmas = sigmas[inrange_mask]
                a = (self._min_value[target_feature_index] - mus) / sigmas
                b = (self._max_value[target_feature_index] - mus) / sigmas
                truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
                imputed_values[inrange_mask] = truncated_normal.rvs(
                    random_state=self.random_state_
                )
            else:
                imputed_values = estimator.predict(X_test)
                imputed_values = np.clip(
                    imputed_values,
                    self._min_value[target_feature_index],
                    self._max_value[target_feature_index],
                )

        _safe_assign(
            X_filled,
            imputed_values,
            row_indexer=missing_row_mask,
            column_indexer=target_feature_index,
        )
        return X_filled, estimator


# ----------------------------------------------------------------------
# 2. Estimator that switches between regressor / classifier
# ----------------------------------------------------------------------
class _ColumnTypeEstimator(BaseEstimator):
    """
    Meta-estimator that delegates to either a regressor or a classifier
    depending on whether the target column is categorical.

    Parameters
    ----------
    regressor : estimator
        Regressor used for numerical target columns.
    classifier : estimator
        Classifier used for categorical target columns.
    categorical_indices : list of int
        Column indices in the concatenated array that are treated as categorical.
    target_feature_index : int or None
        Index of the column currently being imputed (set at fit time).
    predictor_feature_indices : list of int or None
        Indices of columns used as predictors (set at fit time).
    """

    def __init__(
        self,
        regressor,
        classifier,
        categorical_indices,
        target_feature_index=None,
        predictor_feature_indices=None,
        **params,
    ):
        self.regressor = regressor
        self.classifier = classifier
        self.categorical_indices = categorical_indices
        self.target_feature_index = target_feature_index
        self.predictor_feature_indices = predictor_feature_indices
        self.model_ = None

    def is_categorical(self, feature_index):
        """Return True if the given feature index is categorical."""
        return feature_index in self.categorical_indices

    def fit(self, X, y):
        """Fit the appropriate model (regressor or classifier) on the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target values.
        """
        X = np.asarray(X)
        if self.is_categorical(self.target_feature_index):
            self.model_ = clone(self.classifier)
        else:
            self.model_ = clone(self.regressor)
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        """Predict using the fitted model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying model's
            ``predict`` method.  ``return_std=True`` is handled specially for
            models that do not natively support it.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.  If ``return_std=True``, returns a tuple
            ``(y_pred, std)``.
        """
        check_is_fitted(self, "model_")
        # If return_std is requested but the model doesn't support it,
        # fall back to a plain prediction with unit standard deviation.
        return_std = kwargs.pop("return_std", False)
        try:
            if return_std:
                preds, std = self.model_.predict(X, return_std=True)
                return preds, std
            else:
                return self.model_.predict(X, **kwargs)
        except TypeError:
            if return_std:
                preds = self.model_.predict(X, **kwargs)
                return preds, np.ones_like(preds)
            raise


def _sample_from_probabilities(proba, random_state=None):
    """Draw a class for each row from the predicted probability matrix."""
    rng = np.random.default_rng(random_state)
    n_samples = proba.shape[0]
    cumulative = proba.cumsum(axis=1)
    random_values = rng.random(n_samples).reshape(-1, 1)
    sampled = (random_values < cumulative).argmax(axis=1)
    return sampled


# ----------------------------------------------------------------------
# 3. Public, user-friendly wrapper
# ----------------------------------------------------------------------
class MixedTypeImputer(BaseEstimator, TransformerMixin):
    """
    Imputer for DataFrames containing both numerical and categorical (string)
    columns.

    Internally encodes categoricals with OrdinalEncoder, runs an iterative
    MICE-style imputer that automatically chooses a regressor or classifier
    per column, and decodes categoricals back to their original strings.

    Parameters
    ----------
    categorical_features : list of int or str, or None
        Column indices or names (if DataFrame) that are categorical. If None
        and the input is a DataFrame, columns with object, string, or category
        dtype are automatically treated as categorical.

    numeric_features : list of int or str, or None
        Numeric columns. If None, all remaining columns (or float/int types)
        are treated as numeric. Only used as a check.

    regressor : estimator, default=HistGradientBoostingRegressor()
        Regressor used for numerical target columns.

    classifier : estimator, default=HistGradientBoostingClassifier()
        Classifier used for categorical target columns.

    max_iter : int, default=10
        Maximum number of imputation rounds.

    tol : float, default=1e-3
        Tolerance of the stopping condition.

    initial_strategy : str, default='mean'
        Initial imputation strategy passed to IterativeImputer.

    sample_posterior : bool, default=False
        Whether to sample from the predictive posterior for multiple imputation.
        Categorical sampling uses predicted class probabilities.

    random_state : int, RandomState instance or None, default=None
        Seed for reproducibility.

    verbose : int, default=0
        Verbosity of the underlying IterativeImputer.

    add_indicator : bool, default=False
        If True, add missing indicators. (They will appear in the output
        DataFrame as boolean columns.)

    keep_empty_features : bool, default=False
        Keep features that are all missing at fit time.

    Attributes
    ----------
    imputation_model_ : _MixedTypeIterativeImputer
        The fitted internal imputer.

    encoder_ : OrdinalEncoder
        Fitted encoder for categorical columns.

    categorical_features_ : list of int
        Column indices in the encoded array that are categorical.

    output_feature_order_ : list of str
        Final column order of the output DataFrame.
    """

    def __init__(
        self,
        categorical_features=None,
        numeric_features=None,
        regressor=None,
        classifier=None,
        max_iter=10,
        tol=1e-3,
        initial_strategy="mean",
        sample_posterior=False,
        random_state=None,
        verbose=0,
        add_indicator=False,
        keep_empty_features=False,
    ):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.regressor = regressor
        self.classifier = classifier
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.sample_posterior = sample_posterior
        self.random_state = random_state
        self.verbose = verbose
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _validate_input(self, X, reset=True):
        """Validate input and store feature metadata.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data.
        reset : bool
            If True, recompute categorical/numeric feature lists from scratch.
            Set to False during ``transform`` to reuse the lists from ``fit``.

        Returns
        -------
        X : DataFrame or ndarray
            The (possibly validated) input.
        """
        if isinstance(X, pd.DataFrame):
            self._input_is_df = True
            self._input_columns = X.columns.tolist()
            self._input_index = X.index
            if self.categorical_features is None:
                # auto-detect object, string, and category columns
                cats = []
                for col, dtype in X.dtypes.items():
                    if (pd.api.types.is_object_dtype(dtype)
                            or pd.api.types.is_string_dtype(dtype)
                            or isinstance(dtype, pd.CategoricalDtype)):
                        cats.append(col)
                self.categorical_features = cats
            if self.numeric_features is None:
                # Only include columns that are actually numeric dtype
                self.numeric_features = [
                    col
                    for col in X.columns
                    if col not in self.categorical_features
                    and pd.api.types.is_numeric_dtype(X[col].dtype)
                ]
            else:
                # ensure numeric_features not in categorical
                pass
            # map names to indices for internal use
            self._cat_idx_names = self.categorical_features
            self._num_idx_names = self.numeric_features
        else:
            # array input
            self._input_is_df = False
            X = check_array(X, ensure_all_finite=False, dtype=None)
            n_cols = X.shape[1]
            if self.categorical_features is None:
                raise ValueError(
                    "For array input, categorical_features must be provided "
                    "as a list of column indices."
                )
            self._cat_idx_names = self.categorical_features
            if self.numeric_features is None:
                self._num_idx_names = [
                    i for i in range(n_cols) if i not in self._cat_idx_names
                ]
            else:
                self._num_idx_names = self.numeric_features
        return X

    def fit(self, X, y=None):
        """Fit the encoder and the iterative imputer.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Input data with missing values.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : MixedTypeImputer
            The fitted instance.
        """
        X = self._validate_input(X, reset=True)

        # Separate numeric and categorical parts
        if self._input_is_df:
            X_cat = X[self._cat_idx_names].copy()
            X_num = X[self._num_idx_names].copy()
        else:
            X_cat = X[:, self._cat_idx_names]
            X_num = X[:, self._num_idx_names]

        # Replace categorical NaNs with a sentinel string so OrdinalEncoder works.
        # Convert all categorical columns to string and replace NaN with sentinel.
        sentinel = "__MISSING__"
        if hasattr(X_cat, "fillna"):
            # DataFrame path: convert each column to string, replacing NaN with sentinel
            X_cat_filled = X_cat.copy()
            for col in (X_cat_filled.columns if hasattr(X_cat_filled, "columns") else range(X_cat_filled.shape[1])):
                if hasattr(X_cat_filled, "iloc"):
                    na_mask = X_cat_filled[col].isna()
                    as_str = X_cat_filled[col].astype(str)
                    X_cat_filled[col] = as_str.where(~na_mask, other=sentinel)
                else:
                    # NumPy column access
                    col_data = X_cat_filled[:, col]
                    na_mask = pd.isnull(col_data)
                    X_cat_filled[:, col] = np.where(na_mask, sentinel, col_data.astype(str))
        else:
            # NumPy array path
            na_mask = pd.isnull(X_cat)
            X_cat_filled = X_cat.astype(str)
            X_cat_filled[na_mask] = sentinel

        # Fit OrdinalEncoder on the filled categorical part (skip if no cat cols)
        if X_cat_filled.shape[1] > 0:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int64,
            )
            self.encoder_.fit(X_cat_filled)
        else:
            self.encoder_ = None

        # Transform to integers and find the encoded value for sentinel
        if self.encoder_ is not None:
            X_cat_encoded = self.encoder_.transform(X_cat_filled)  # int array
            # For each categorical column, find the index of sentinel
            self._missing_encoded_values_ = []
            for i, categories in enumerate(self.encoder_.categories_):
                if sentinel in categories:
                    enc_val = np.where(categories == sentinel)[0][0]
                else:
                    # If sentinel not present (no missing values in that column),
                    # choose an impossible integer to later replace with NaN
                    enc_val = -9999
                self._missing_encoded_values_.append(enc_val)

            # Replace the encoded sentinel value with np.nan so the imputer sees it as missing
            X_cat_processed = X_cat_encoded.astype(np.float64)
            for col_idx, miss_val in enumerate(self._missing_encoded_values_):
                mask = X_cat_processed[:, col_idx] == miss_val
                X_cat_processed[mask, col_idx] = np.nan
        else:
            X_cat_processed = np.empty((X.shape[0], 0), dtype=np.float64)

        # Numeric part: ensure float and keep NaN
        if isinstance(X_num, pd.DataFrame) and X_num.shape[1] > 0:
            X_num_processed = X_num.to_numpy(dtype=np.float64, na_value=np.nan)
        elif isinstance(X_num, pd.DataFrame):
            X_num_processed = np.empty((X_num.shape[0], 0), dtype=np.float64)
        elif isinstance(X_num, np.ndarray) and X_num.shape[1] > 0:
            X_num_processed = X_num.astype(np.float64)
        else:
            X_num_processed = np.empty((X_num.shape[0], 0), dtype=np.float64)

        # Concatenate: first categorical encoded columns, then numeric
        X_all = np.hstack([X_cat_processed, X_num_processed])

        # Categorical indices in the concatenated array are 0..n_cat-1
        n_cat = X_cat_processed.shape[1]
        self.categorical_features_ = list(range(n_cat))

        # Build the internal imputer
        if self.regressor is None:
            from sklearn.ensemble import HistGradientBoostingRegressor
            reg = HistGradientBoostingRegressor(random_state=self.random_state)
        else:
            reg = self.regressor
        if self.classifier is None:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(random_state=self.random_state)
        else:
            clf = self.classifier

        self.imputation_model_ = _MixedTypeIterativeImputer(
            estimator=_ColumnTypeEstimator(
                regressor=reg,
                classifier=clf,
                categorical_indices=self.categorical_features_,
            ),
            max_iter=self.max_iter,
            tol=self.tol,
            initial_strategy=self.initial_strategy,
            sample_posterior=self.sample_posterior,
            random_state=self.random_state,
            verbose=self.verbose,
            add_indicator=self.add_indicator,
            keep_empty_features=self.keep_empty_features,
        )
        self.imputation_model_.fit(X_all)

        # Track which features were dropped by the internal imputer
        # (relevant when keep_empty_features=False and columns are all-NaN).
        if (not self.keep_empty_features
                and hasattr(self.imputation_model_, "_is_empty_feature")):
            empty_mask = self.imputation_model_._is_empty_feature
            self._all_empty_features_ = list(np.flatnonzero(empty_mask))
            self._all_valid_features_ = list(np.flatnonzero(~empty_mask))
        else:
            self._all_empty_features_ = []
            self._all_valid_features_ = list(range(X_all.shape[1]))

        return self

    def transform(self, X):
        """Impute missing values and return a DataFrame like the input.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_imputed : DataFrame or ndarray
        """
        check_is_fitted(self, "imputation_model_")
        X = self._validate_input(X, reset=False)

        # Split and encode just like in fit
        if self._input_is_df:
            X_cat = X[self._cat_idx_names].copy()
            X_num = X[self._num_idx_names].copy()
        else:
            X_cat = X[:, self._cat_idx_names]
            X_num = X[:, self._num_idx_names]

        sentinel = "__MISSING__"
        if hasattr(X_cat, "fillna"):
            # DataFrame path: convert each column to string, replacing NaN with sentinel
            X_cat_filled = X_cat.copy()
            for col in (X_cat_filled.columns if hasattr(X_cat_filled, "columns") else range(X_cat_filled.shape[1])):
                if hasattr(X_cat_filled, "iloc"):
                    na_mask = X_cat_filled[col].isna()
                    as_str = X_cat_filled[col].astype(str)
                    X_cat_filled[col] = as_str.where(~na_mask, other=sentinel)
                else:
                    col_data = X_cat_filled[:, col]
                    na_mask = pd.isnull(col_data)
                    X_cat_filled[:, col] = np.where(na_mask, sentinel, col_data.astype(str))
        else:
            # NumPy array path
            na_mask = pd.isnull(X_cat)
            X_cat_filled = X_cat.astype(str)
            X_cat_filled[na_mask] = sentinel

        if self.encoder_ is not None:
            X_cat_encoded = self.encoder_.transform(X_cat_filled)
            X_cat_processed = X_cat_encoded.astype(np.float64)
            for col_idx, miss_val in enumerate(self._missing_encoded_values_):
                mask = X_cat_processed[:, col_idx] == miss_val
                X_cat_processed[mask, col_idx] = np.nan
        else:
            X_cat_processed = np.empty((X.shape[0], 0), dtype=np.float64)

        if isinstance(X_num, pd.DataFrame) and X_num.shape[1] > 0:
            X_num_processed = X_num.to_numpy(dtype=np.float64, na_value=np.nan)
        elif isinstance(X_num, pd.DataFrame):
            X_num_processed = np.empty((X_num.shape[0], 0), dtype=np.float64)
        elif isinstance(X_num, np.ndarray) and X_num.shape[1] > 0:
            X_num_processed = X_num.astype(np.float64)
        else:
            X_num_processed = np.empty((X_num.shape[0], 0), dtype=np.float64)
        X_all = np.hstack([X_cat_processed, X_num_processed])

        # Impute
        X_imputed_all_reduced = self.imputation_model_.transform(X_all)

        # When keep_empty_features=False, the internal imputer drops all-NaN
        # columns.  We need to map the reduced output back to original columns.
        n_features_full = X_all.shape[1]
        if len(self._all_empty_features_) > 0:
            # Build a map from output column index → original (full) column index
            full_idx_from_reduced = {
                out_idx: full_idx
                for out_idx, full_idx in enumerate(self._all_valid_features_)
            }
            # Re-expand to full size with NaN in dropped positions
            X_imputed_all = np.full(
                (X_imputed_all_reduced.shape[0], n_features_full), np.nan
            )
            for out_idx, full_idx in full_idx_from_reduced.items():
                X_imputed_all[:, full_idx] = X_imputed_all_reduced[:, out_idx]

            # Drop dropped columns from the column name lists for reconstruction
            empty_set = set(self._all_empty_features_)
            n_cat = X_cat_processed.shape[1]
            # Categorical columns occupy indices 0..n_cat-1
            cat_idx_names_kept = [
                name for i, name in enumerate(self._cat_idx_names)
                if i not in empty_set
            ]
            # Numeric columns occupy indices n_cat..n_features_full-1
            num_idx_names_kept = [
                name for i, name in enumerate(self._num_idx_names)
                if (n_cat + i) not in empty_set
            ]
            # Also update input_columns for the final reorder (DataFrame only)
            if self._input_is_df:
                dropped_col_names = set(
                    list(self._cat_idx_names) + list(self._num_idx_names)
                ) - set(cat_idx_names_kept + num_idx_names_kept)
                input_columns_kept = [
                    c for c in self._input_columns if c not in dropped_col_names
                ]
            else:
                input_columns_kept = None
        else:
            X_imputed_all = X_imputed_all_reduced
            cat_idx_names_kept = list(self._cat_idx_names)
            num_idx_names_kept = list(self._num_idx_names)
            input_columns_kept = (
                list(self._input_columns) if self._input_is_df else None
            )

        # Separate imputed parts
        n_cat = X_cat_processed.shape[1]
        X_cat_imputed = X_imputed_all[:, :n_cat]
        X_num_imputed = X_imputed_all[:, n_cat:]

        # If features were dropped, keep only the surviving ones
        if len(self._all_empty_features_) > 0:
            empty_set = set(self._all_empty_features_)
            cat_kept_indices = [
                i for i in range(n_cat) if i not in empty_set
            ]
            num_kept_indices = [
                i for i in range(X_num_imputed.shape[1])
                if (n_cat + i) not in empty_set
            ]
            X_cat_imputed = X_cat_imputed[:, cat_kept_indices]
            X_num_imputed = X_num_imputed[:, num_kept_indices]

        # Decode categorical part if any categorical columns exist
        if n_cat > 0:
            # Round categorical imputations to integers (the classifier outputs discrete values)
            X_cat_imputed = np.round(X_cat_imputed).astype(int)

            # Replace any value that equals the missing-encoded integer with that integer
            # (they shouldn't persist, but just in case)
            # Then decode using inverse transform
            X_cat_decoded = self.encoder_.inverse_transform(X_cat_imputed)

            # If any decoded value is sentinel, replace with np.nan (shouldn't happen after imputation)
            if hasattr(X_cat_decoded, "astype"):
                X_cat_decoded = X_cat_decoded.astype(object)
            mask_missing = X_cat_decoded == sentinel
            if np.any(mask_missing):
                X_cat_decoded[mask_missing] = np.nan
        else:
            # No categorical columns: create empty object array
            X_cat_decoded = np.empty((X_all.shape[0], 0), dtype=object)

        # Rebuild DataFrame if input was DataFrame
        if self._input_is_df:
            # Build a DataFrame with correct column order
            result = pd.DataFrame(index=self._input_index)
            # categorical columns first (original order)
            for i, col_name in enumerate(cat_idx_names_kept):
                result[col_name] = X_cat_decoded[:, i]
            for i, col_name in enumerate(num_idx_names_kept):
                result[col_name] = X_num_imputed[:, i]
            # Ensure original column order (minus dropped columns)
            result = result[input_columns_kept]
            return result

        # else return ndarray in the same column order as input
        # Note: the original X was (cat then num)? No, the input order may differ.
        # We need to reconstruct the original column order.
        # For simplicity, we'll just return a concatenation of cat followed by num,
        # but that might not match input order. To preserve, we must reorder.
        # For array input, we assume user provided categorical_features as indices,
        # and we'll arrange output in the original order (0..n_features-1).
        # We'll build a full array with original column positions.
        n_features = len(self._cat_idx_names) + len(self._num_idx_names)
        X_out = np.empty((X_all.shape[0], n_features))
        X_out[:, self._cat_idx_names] = X_cat_decoded
        X_out[:, self._num_idx_names] = X_num_imputed
        return X_out
