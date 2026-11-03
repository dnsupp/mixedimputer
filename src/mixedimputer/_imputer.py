"""
Mixed-type MICE-style iterative imputer for scikit-learn.

Provides a single ``MixedImputer`` transformer that handles DataFrames
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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)


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
class _MixedIterativeImputer(IterativeImputer):
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
            y_train = np.asarray(y_train).ravel()
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
            model = estimator.model_
            if self.sample_posterior and hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X_test), dtype=float)
                sampled_indices = _sample_from_probabilities(
                    proba, random_state=self.random_state_
                )
                imputed_values = np.asarray(model.classes_)[sampled_indices]
            else:
                imputed_values = estimator.predict(X_test)

            imputed_values = np.rint(np.asarray(imputed_values).ravel()).astype(float)
        else:
            if self.sample_posterior:
                mus, sigmas = estimator.predict(X_test, return_std=True)
                mus = np.asarray(mus, dtype=float).ravel()
                sigmas = np.asarray(sigmas, dtype=float).ravel()
                imputed_values = np.zeros(mus.shape, dtype=float)
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
                truncated_normal = stats.truncnorm(
                    a=a, b=b, loc=mus, scale=sigmas
                )
                imputed_values[inrange_mask] = truncated_normal.rvs(
                    random_state=self.random_state_
                )
            else:
                imputed_values = np.asarray(
                    estimator.predict(X_test), dtype=float
                ).ravel()
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
    ):
        self.regressor = regressor
        self.classifier = classifier
        self.categorical_indices = categorical_indices
        self.target_feature_index = target_feature_index
        self.predictor_feature_indices = predictor_feature_indices

    def is_categorical(self, feature_index):
        """Return True if the given feature index is categorical."""
        return feature_index in self.categorical_indices

    def fit(self, X, y, **fit_params):
        """Fit the appropriate model (regressor or classifier) on the data."""
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if self.is_categorical(self.target_feature_index):
            self.model_ = clone(self.classifier)
        else:
            self.model_ = clone(self.regressor)
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X, **kwargs):
        """Predict using the fitted model.

        ``return_std=True`` is handled specially for models that do not
        natively support it: those fall back to a unit standard deviation.
        """
        check_is_fitted(self, "model_")
        return_std = kwargs.pop("return_std", False)
        if return_std:
            try:
                preds, std = self.model_.predict(X, return_std=True)
            except TypeError:
                # The model does not support return_std.  Re-run a plain
                # prediction and fall back to unit variance.
                preds = self.model_.predict(X, **kwargs)
                return np.asarray(preds).ravel(), np.ones_like(
                    np.asarray(preds, dtype=float).ravel()
                )
            return np.asarray(preds).ravel(), np.asarray(std).ravel()
        return np.asarray(self.model_.predict(X, **kwargs)).ravel()


def _sample_from_probabilities(proba, random_state=None):
    """Draw a class for each row from the predicted probability matrix."""
    rng = check_random_state(random_state)
    proba = np.asarray(proba, dtype=float)
    n_samples = proba.shape[0]
    cumulative = proba.cumsum(axis=1)
    random_values = rng.random_sample(n_samples).reshape(-1, 1)
    sampled = (random_values < cumulative).argmax(axis=1)
    return sampled.astype(int)


# ----------------------------------------------------------------------
# 3. Public, user-friendly wrapper
# ----------------------------------------------------------------------
class MixedImputer(BaseEstimator, TransformerMixin):
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
        and the input is a DataFrame, columns with object, string, category,
        or boolean dtype are automatically treated as categorical.

    numeric_features : list of int or str, or None
        Numeric column indices or names.  If None, auto-detected as all
        columns whose dtype passes ``pd.api.types.is_numeric_dtype``, are not
        boolean, and are not listed in ``categorical_features``.

    regressor : estimator, default=HistGradientBoostingRegressor()
        Regressor used for numerical target columns.  Any sklearn regressor
        with ``fit`` and ``predict`` works (e.g. ``Ridge``,
        ``RandomForestRegressor``, ``BayesianRidge``).  For ``sample_posterior``
        uncertainty, models that support ``return_std`` (like ``BayesianRidge``)
        give per-prediction variance; others fall back to unit standard
        deviation.

    classifier : estimator, default=HistGradientBoostingClassifier()
        Classifier used for categorical target columns.  Any sklearn
        classifier with ``fit`` and ``predict`` works (e.g.
        ``LogisticRegression``, ``RandomForestClassifier``, ``GaussianNB``).
        For ``sample_posterior``, the classifier should also implement
        ``predict_proba`` and expose ``classes_`` (most do — notable
        exception: ``RidgeClassifier`` lacks ``predict_proba``).

    max_iter : int, default=10
        Maximum number of imputation rounds.

    tol : float, default=1e-3
        Tolerance of the stopping condition.

    initial_strategy : str, default='mean'
        Initial imputation strategy passed to IterativeImputer.

    sample_posterior : bool, default=False
        Whether to sample from the predictive posterior.  This is stochastic
        single imputation, not a full multiple-imputation procedure.

    random_state : int, RandomState instance or None, default=None
        Seed for reproducibility.

    verbose : int, default=0
        Verbosity of the underlying IterativeImputer.

    add_indicator : bool, default=False
        If True, append boolean missing-indicator columns to the output, one
        for each feature that had missing values at fit time.  DataFrame
        columns are named ``missingindicator_<column>``.

    keep_empty_features : bool, default=False
        Keep features that are all missing at fit time.

    Attributes
    ----------
    imputation_model_ : _MixedIterativeImputer
        The fitted internal imputer.

    encoder_ : OrdinalEncoder or None
        Fitted encoder for categorical columns.

    categorical_features_ : list
        Fitted categorical column labels (DataFrame) or indices (array).

    numeric_features_ : list
        Fitted numeric column labels (DataFrame) or indices (array).

    n_features_in_ : int
        Number of input features.

    feature_names_in_ : ndarray of str
        Defined only when a DataFrame with all-string column names is used.

    n_features_out_ : int
        Number of output columns, including missing indicators.
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

    # ------------------------------------------------------------------
    # Feature resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_feature_spec(spec, n_features, columns, kind):
        """Normalize a feature spec to a list of labels (DataFrame) or ints."""
        if spec is None:
            return None
        if isinstance(spec, (str, bytes, int, np.integer)):
            items = [spec]
        elif isinstance(spec, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            items = list(spec)
        else:
            raise TypeError(
                f"{kind} must be a list of column names/indices or None; "
                f"got {type(spec).__name__}."
            )

        resolved = []
        for item in items:
            if isinstance(item, (list, dict, set, np.ndarray)):
                raise TypeError(
                    f"{kind} entries must be scalar column names or indices; "
                    f"got {type(item).__name__}."
                )
            if columns is not None:
                # Prefer an exact label match; fall back to a positional
                # index when the integer is not itself a column label.
                try:
                    if item in columns:
                        resolved.append(item)
                        continue
                except TypeError:
                    pass
                if isinstance(item, (int, np.integer)):
                    pos = int(item)
                    if 0 <= pos < n_features:
                        resolved.append(columns[pos])
                        continue
                raise ValueError(
                    f"{kind} entry {item!r} is neither a column label nor a "
                    f"valid positional index for {n_features} columns."
                )
            else:
                if not isinstance(item, (int, np.integer)):
                    raise TypeError(
                        f"For array input, {kind} must contain integer indices; "
                        f"got {type(item).__name__}."
                    )
                pos = int(item)
                if not 0 <= pos < n_features:
                    raise ValueError(
                        f"{kind} index {pos} is out of bounds for "
                        f"{n_features} columns."
                    )
                resolved.append(pos)

        if len(set(resolved)) != len(resolved):
            raise ValueError(f"{kind} contains duplicate entries: {resolved!r}")
        return resolved

    def _resolve_dataframe_features(self, X):
        columns = list(X.columns)
        if len(set(columns)) != len(columns):
            duplicated = [
                c for c in columns if columns.count(c) > 1
            ]
            raise ValueError(
                f"DataFrame has duplicate column names: {sorted(set(duplicated))!r}. "
                "MixedImputer requires unique column names."
            )
        if len(columns) == 0:
            raise ValueError("X must contain at least one column.")
        if len(X) == 0:
            raise ValueError("X must contain at least one sample.")

        cat = self._normalize_feature_spec(
            self.categorical_features, len(columns), columns, "categorical_features"
        )
        if cat is None:
            cat = [
                col
                for col in columns
                if (
                    pd.api.types.is_object_dtype(X[col].dtype)
                    or pd.api.types.is_string_dtype(X[col].dtype)
                    or isinstance(X[col].dtype, pd.CategoricalDtype)
                    or pd.api.types.is_bool_dtype(X[col].dtype)
                )
            ]

        num = self._normalize_feature_spec(
            self.numeric_features, len(columns), columns, "numeric_features"
        )
        if num is None:
            num = [
                col
                for col in columns
                if col not in cat
                and pd.api.types.is_numeric_dtype(X[col].dtype)
                and not pd.api.types.is_bool_dtype(X[col].dtype)
            ]

        overlap = set(cat).intersection(num)
        if overlap:
            raise ValueError(
                f"Columns cannot be both categorical and numeric: {sorted(overlap)!r}"
            )

        # Explicit feature lists are overrides, not an exhaustive partition.
        # Assign any remaining columns by dtype so a partial list does not
        # make valid string/numeric columns unusable.
        assigned = set(cat).union(num)
        for col in columns:
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
                    f"Boolean column {col!r} must be treated as categorical. "
                    "Remove it from numeric_features or add it to "
                    "categorical_features."
                )

        assigned = set(cat).union(num)
        unassigned = [col for col in columns if col not in assigned]
        if unassigned:
            details = ", ".join(
                f"{col!r} ({X[col].dtype!r})" for col in unassigned
            )
            raise ValueError(
                "Some columns have dtypes that MixedImputer cannot handle "
                f"(datetime/timedelta/other): {details}. Convert them to a "
                "numeric or string dtype, or list them explicitly in "
                "categorical_features."
            )
        return cat, num

    def _resolve_array_features(self, X):
        n_features = X.shape[1]
        cat = self._normalize_feature_spec(
            self.categorical_features, n_features, None, "categorical_features"
        )
        if cat is None:
            raise ValueError(
                "For array input, categorical_features must be provided "
                "as a list of column indices."
            )
        num = self._normalize_feature_spec(
            self.numeric_features, n_features, None, "numeric_features"
        )
        if num is None:
            num = [i for i in range(n_features) if i not in cat]

        overlap = set(cat).intersection(num)
        if overlap:
            raise ValueError(
                f"Columns cannot be both categorical and numeric: {sorted(overlap)!r}"
            )
        assigned = set(cat).union(num)
        unassigned = [i for i in range(n_features) if i not in assigned]
        if unassigned:
            raise ValueError(
                f"Some array columns are not assigned to categorical_features "
                f"or numeric_features: {unassigned!r}"
            )
        return cat, num

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_series_list(X_cat):
        n_cat = X_cat.shape[1]
        if isinstance(X_cat, pd.DataFrame):
            return [X_cat.iloc[:, j] for j in range(n_cat)]
        return [pd.Series(X_cat[:, j]) for j in range(n_cat)]

    def _fit_encode_categorical(self, X_cat):
        n = len(X_cat)
        n_cat = X_cat.shape[1]
        series_list = self._as_series_list(X_cat)

        string_values = []
        missing_masks = []
        existing = set()
        for series in series_list:
            missing = series.isna().to_numpy()
            values = series.astype(str).to_numpy(dtype=object)
            existing.update(pd.unique(values[~missing]))
            string_values.append(values)
            missing_masks.append(missing)

        sentinel = "__mixedimputer_missing__"
        while sentinel in existing:
            sentinel += "_"

        filled = np.empty((n, n_cat), dtype=object)
        for j in range(n_cat):
            values = string_values[j].copy()
            values[missing_masks[j]] = sentinel
            filled[:, j] = values

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
        encoder.fit(filled)
        encoded = encoder.transform(filled).astype(np.float64)

        sentinel_codes = []
        for j, categories in enumerate(encoder.categories_):
            matches = np.flatnonzero(categories == sentinel)
            code = int(matches[0]) if len(matches) else -9999
            sentinel_codes.append(code)
            if code != -9999:
                encoded[encoded[:, j] == code, j] = np.nan

        return encoded, encoder, sentinel, sentinel_codes

    def _transform_encode_categorical(self, X_cat):
        n = len(X_cat)
        n_cat = X_cat.shape[1]
        if n_cat == 0:
            return np.empty((n, 0), dtype=np.float64)
        if self.encoder_ is None:
            raise ValueError("Categorical columns were not present during fit.")

        series_list = self._as_series_list(X_cat)
        filled = np.empty((n, n_cat), dtype=object)
        for j, series in enumerate(series_list):
            missing = series.isna().to_numpy()
            values = series.astype(str).to_numpy(dtype=object)
            values[missing] = self._sentinel_
            filled[:, j] = values

        encoded = self.encoder_.transform(filled).astype(np.float64)
        for j, code in enumerate(self._sentinel_codes_):
            if code != -9999:
                encoded[encoded[:, j] == code, j] = np.nan
        # Unknown categories are encoded as -1; treat them as missing.
        encoded[encoded == -1] = np.nan
        return encoded

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the encoder and the iterative imputer."""
        if isinstance(X, pd.DataFrame):
            self._input_is_df = True
            self._fit_columns_ = list(X.columns)
            cat, num = self._resolve_dataframe_features(X)
            self.categorical_features_ = list(cat)
            self.numeric_features_ = list(num)
            self.n_features_in_ = X.shape[1]
            if all(isinstance(col, str) for col in self._fit_columns_):
                self.feature_names_in_ = np.asarray(
                    self._fit_columns_, dtype=object
                )
            else:
                if hasattr(self, "feature_names_in_"):
                    del self.feature_names_in_
            self._bool_labels_ = [
                col
                for col in self.categorical_features_
                if pd.api.types.is_bool_dtype(X[col].dtype)
            ]
            X_cat = X[self.categorical_features_].copy()
            X_num = X[self.numeric_features_].copy()
        else:
            self._input_is_df = False
            X = check_array(X, ensure_all_finite=False, dtype=None)
            if X.ndim != 2:
                raise ValueError("X must be a 2-dimensional array.")
            n_features = X.shape[1]
            cat, num = self._resolve_array_features(X)
            self.categorical_features_ = list(cat)
            self.numeric_features_ = list(num)
            self.n_features_in_ = n_features
            if hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
            self._fit_columns_ = list(range(n_features))
            self._bool_labels_ = []
            X_cat = X[:, self.categorical_features_]
            X_num = X[:, self.numeric_features_]

        self._cat_labels_ = list(self.categorical_features_)
        self._num_labels_ = list(self.numeric_features_)
        self._internal_labels_ = self._cat_labels_ + self._num_labels_
        n_samples = X.shape[0]

        if len(self._cat_labels_) > 0:
            (
                X_cat_processed,
                self.encoder_,
                self._sentinel_,
                self._sentinel_codes_,
            ) = self._fit_encode_categorical(X_cat)
        else:
            X_cat_processed = np.empty((n_samples, 0), dtype=float)
            self.encoder_ = None
            self._sentinel_ = "__mixedimputer_missing__"
            self._sentinel_codes_ = []

        if len(self._num_labels_) > 0:
            if isinstance(X_num, pd.DataFrame):
                X_num_processed = X_num.to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
            else:
                try:
                    X_num_processed = np.asarray(X_num, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Numeric columns must be convertible to float64."
                    ) from exc
        else:
            X_num_processed = np.empty((n_samples, 0), dtype=float)

        if np.isinf(X_num_processed).any():
            raise ValueError("Numeric columns contain infinite values.")

        X_all = np.hstack([X_cat_processed, X_num_processed])
        self._n_cat_ = len(self._cat_labels_)
        self._n_features_full_ = X_all.shape[1]

        if self._n_features_full_ == 0:
            raise ValueError("X must contain at least one usable feature.")
        if not self.keep_empty_features and np.all(np.isnan(X_all)):
            raise ValueError(
                "All features are entirely missing; nothing can be imputed. "
                "Remove empty columns or set keep_empty_features=True."
            )

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

        self.imputation_model_ = _MixedIterativeImputer(
            estimator=_ColumnTypeEstimator(
                regressor=reg,
                classifier=clf,
                categorical_indices=list(range(self._n_cat_)),
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

        if not self.keep_empty_features and hasattr(
            self.imputation_model_, "_is_empty_feature"
        ):
            empty_mask = np.asarray(
                self.imputation_model_._is_empty_feature, dtype=bool
            )
            self._all_empty_features_ = list(np.flatnonzero(empty_mask))
            self._all_valid_features_ = list(
                np.flatnonzero(~empty_mask)
            )
        else:
            self._all_empty_features_ = []
            self._all_valid_features_ = list(range(self._n_features_full_))

        if len(self._all_valid_features_) == 0:
            raise ValueError(
                "All features are entirely missing; nothing can be imputed. "
                "Remove empty columns or set keep_empty_features=True."
            )

        # Missing indicators are appended after the imputed base features.
        indicator = getattr(self.imputation_model_, "indicator_", None)
        if self.add_indicator and indicator is not None and hasattr(
            indicator, "features_"
        ):
            all_indicator_features = [int(i) for i in indicator.features_]
        else:
            all_indicator_features = []
        self._indicator_features_all_ = all_indicator_features
        self._indicator_features_ = [
            i for i in all_indicator_features if i in self._all_valid_features_
        ]
        self._indicator_positions_ = [
            pos
            for pos, i in enumerate(all_indicator_features)
            if i in self._all_valid_features_
        ]
        # Keep indicators in the same order as the fitted columns.
        fit_order = {
            label: pos for pos, label in enumerate(self._fit_columns_)
        }
        indicator_pairs = list(
            zip(self._indicator_features_, self._indicator_positions_)
        )
        indicator_pairs.sort(
            key=lambda pair: fit_order[self._internal_labels_[pair[0]]]
        )
        self._indicator_features_ = [pair[0] for pair in indicator_pairs]
        self._indicator_positions_ = [pair[1] for pair in indicator_pairs]

        valid_set = set(self._all_valid_features_)
        internal_index = {
            label: i for i, label in enumerate(self._internal_labels_)
        }
        self._kept_columns_ = [
            label
            for label in self._fit_columns_
            if internal_index[label] in valid_set
        ]
        self._output_columns_ = list(self._kept_columns_)
        self._output_feature_names_ = [
            str(label) for label in self._kept_columns_
        ]
        for i in self._indicator_features_:
            indicator_name = (
                f"missingindicator_{self._internal_labels_[i]}"
            )
            self._output_columns_.append(indicator_name)
            self._output_feature_names_.append(indicator_name)
        self.n_features_out_ = len(self._output_columns_)

        return self

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------
    def _decode_categorical(self, cat_codes):
        n = cat_codes.shape[0]
        n_cat = cat_codes.shape[1]
        decoded = np.empty((n, n_cat), dtype=object)
        decoded[:] = np.nan
        if n_cat == 0:
            return decoded

        for j, categories in enumerate(self.encoder_.categories_):
            codes = cat_codes[:, j]
            valid = ~np.isnan(codes)
            if not valid.any():
                continue
            integer_codes = np.zeros(n, dtype=int)
            integer_codes[valid] = np.rint(codes[valid]).astype(int)
            valid &= (integer_codes >= 0) & (integer_codes < len(categories))
            values = np.empty(n, dtype=object)
            values[:] = np.nan
            if valid.any():
                values[valid] = categories[integer_codes[valid]]
            values[values == self._sentinel_] = np.nan
            decoded[:, j] = values
        return decoded

    @staticmethod
    def _decode_boolean(values):
        """Convert decoded string values into a pandas boolean array."""
        mask = pd.isna(values)
        result = np.empty(len(values), dtype=object)
        for i, value in enumerate(values):
            if mask[i]:
                result[i] = pd.NA
            else:
                result[i] = str(value).strip().lower() in (
                    "true",
                    "1",
                    "1.0",
                )
        array = pd.array(result, dtype="boolean")
        if not array.isna().any():
            return array.astype(bool)
        return array

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, X):
        """Impute missing values and return a DataFrame/array like the input."""
        check_is_fitted(self, "imputation_model_")

        if isinstance(X, pd.DataFrame):
            if not self._input_is_df:
                raise ValueError(
                    "This MixedImputer was fitted on an array; transform() "
                    "must receive an array."
                )
            columns = list(X.columns)
            if len(set(columns)) != len(columns):
                raise ValueError("X has duplicate column names.")
            missing_columns = [
                col for col in self._fit_columns_ if col not in columns
            ]
            extra_columns = [
                col for col in columns if col not in self._fit_columns_
            ]
            if missing_columns or extra_columns:
                raise ValueError(
                    "X columns do not match the fitted schema. "
                    f"Missing: {missing_columns!r}; extra: {extra_columns!r}."
                )
            # Always use the fitted column order, even if X is reordered.
            X = X.loc[:, self._fit_columns_].copy()
            index = X.index
            X_cat = X[self._cat_labels_].copy()
            X_num = X[self._num_labels_].copy()
            n_samples = len(X)
        else:
            if self._input_is_df:
                raise ValueError(
                    "This MixedImputer was fitted on a DataFrame; transform() "
                    "must receive a DataFrame."
                )
            X = check_array(X, ensure_all_finite=False, dtype=None)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but MixedImputer was fitted "
                    f"with {self.n_features_in_} features."
                )
            index = None
            X_cat = X[:, self._cat_labels_]
            X_num = X[:, self._num_labels_]
            n_samples = X.shape[0]

        if self._n_cat_ > 0:
            X_cat_processed = self._transform_encode_categorical(X_cat)
        else:
            X_cat_processed = np.empty((n_samples, 0), dtype=float)

        if len(self._num_labels_) > 0:
            if isinstance(X_num, pd.DataFrame):
                try:
                    X_num_processed = X_num.to_numpy(
                        dtype=np.float64, na_value=np.nan
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Numeric columns at transform time are not convertible "
                        "to float64."
                    ) from exc
            else:
                try:
                    X_num_processed = np.asarray(X_num, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Numeric columns at transform time are not convertible "
                        "to float64."
                    ) from exc
        else:
            X_num_processed = np.empty((n_samples, 0), dtype=float)

        if np.isinf(X_num_processed).any():
            raise ValueError("Numeric columns contain infinite values.")

        X_all = np.hstack([X_cat_processed, X_num_processed])
        X_imputed_reduced = self.imputation_model_.transform(X_all)

        n_indicators = len(self._indicator_features_all_)
        base_width = X_imputed_reduced.shape[1] - n_indicators
        if self.keep_empty_features:
            expected_base_width = self._n_features_full_
        else:
            expected_base_width = len(self._all_valid_features_)
        if base_width != expected_base_width:
            raise RuntimeError(
                "Unexpected internal imputer output width: "
                f"got {base_width}, expected {expected_base_width}."
            )

        X_base = X_imputed_reduced[:, :base_width]
        if self.keep_empty_features:
            X_base_full = X_base
        else:
            X_base_full = np.full(
                (n_samples, self._n_features_full_), np.nan, dtype=float
            )
            if self._all_empty_features_:
                # Warn if a column that was entirely missing during fit now
                # contains values; it will still be dropped because the fitted
                # internal imputer has no model for it.
                empty_values = X_all[:, self._all_empty_features_]
                if not np.all(np.isnan(empty_values)):
                    warnings.warn(
                        "A feature that was entirely missing during fit now "
                        "contains values; it is still dropped from the output.",
                        UserWarning,
                        stacklevel=2,
                    )
            X_base_full[:, self._all_valid_features_] = X_base

        cat_codes = X_base_full[:, : self._n_cat_]
        num_values = X_base_full[:, self._n_cat_ :]
        decoded_cat = self._decode_categorical(cat_codes)

        # Restore boolean columns that were encoded as categorical strings.
        cat_index_by_label = {
            label: j for j, label in enumerate(self._cat_labels_)
        }
        bool_data = {}
        for label in self._bool_labels_:
            j = cat_index_by_label[label]
            bool_data[label] = self._decode_boolean(decoded_cat[:, j])
            decoded_cat[:, j] = np.asarray(bool_data[label], dtype=object)

        indicators = X_imputed_reduced[:, base_width:]
        indicator_data = [
            np.asarray(indicators[:, pos]).astype(bool)
            for pos in self._indicator_positions_
        ]

        if self._input_is_df:
            result = pd.DataFrame(index=index)
            num_index_by_label = {
                label: j for j, label in enumerate(self._num_labels_)
            }
            valid_set = set(self._all_valid_features_)
            internal_index = {
                label: i for i, label in enumerate(self._internal_labels_)
            }
            for label in self._fit_columns_:
                i = internal_index[label]
                if i not in valid_set:
                    continue
                if label in bool_data:
                    result[label] = bool_data[label]
                elif label in cat_index_by_label:
                    result[label] = decoded_cat[
                        :, cat_index_by_label[label]
                    ]
                else:
                    result[label] = num_values[
                        :, num_index_by_label[label]
                    ]
            for i, data in zip(self._indicator_features_, indicator_data):
                result[f"missingindicator_{self._internal_labels_[i]}"] = data
            result = result.loc[:, self._output_columns_]
            return result

        # Array output: preserve the original column order and fill dropped
        # features with NaN, never broadcast.
        out_columns = []
        internal_index = {
            label: i for i, label in enumerate(self._internal_labels_)
        }
        valid_set = set(self._all_valid_features_)
        cat_index_by_label = {
            label: j for j, label in enumerate(self._cat_labels_)
        }
        num_index_by_label = {
            label: j for j, label in enumerate(self._num_labels_)
        }
        for label in self._fit_columns_:
            i = internal_index[label]
            if i not in valid_set:
                out_columns.append(np.full(n_samples, np.nan))
            elif label in cat_index_by_label:
                out_columns.append(
                    decoded_cat[:, cat_index_by_label[label]]
                )
            else:
                out_columns.append(num_values[:, num_index_by_label[label]])
        for data in indicator_data:
            out_columns.append(data)

        out = np.empty((n_samples, len(out_columns)), dtype=object)
        for j, values in enumerate(out_columns):
            out[:, j] = values
        try:
            return out.astype(np.float64)
        except (TypeError, ValueError):
            return out

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        """Get output feature names, including missing-indicator columns."""
        check_is_fitted(self, "imputation_model_")
        if input_features is not None:
            provided = list(input_features)
            if len(provided) != len(self._fit_columns_):
                raise ValueError(
                    f"input_features has {len(provided)} entries, expected "
                    f"{len(self._fit_columns_)}."
                )
            if self._input_is_df and provided != list(self._fit_columns_):
                raise ValueError(
                    "input_features must match the DataFrame columns used "
                    "during fit."
                )
        return np.asarray(self._output_feature_names_, dtype=object)
