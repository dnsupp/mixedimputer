<p align="center">
  <a href="https://github.com/dnsupp/mixedimputer">
    <img src="https://img.shields.io/badge/github-repo-blue?logo=github" alt="GitHub Repo">
  </a>
  <a href="https://github.com/dnsupp/mixedimputer/actions/workflows/tests.yml">
    <img src="https://github.com/dnsupp/mixedimputer/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://pypi.org/project/mixed-imputer/">
    <img src="https://img.shields.io/pypi/v/mixed-imputer" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/mixed-imputer/">
    <img src="https://img.shields.io/pypi/pyversions/mixed-imputer" alt="Python Versions">
  </a>
  <a href="https://github.com/dnsupp/mixedimputer/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/mixed-imputer" alt="License">
  </a>
</p>

# MixedImputer

> **MixedImputer is a wrapper around scikit-learn's [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html), which implements an iterative imputation strategy inspired by MICE (Multivariate Imputation by Chained Equations).** It extends it to seamlessly handle DataFrames containing both numerical and categorical (string) columns.

It automatically detects column types, encodes categoricals with `OrdinalEncoder`, runs the `MixedImputer` (powered by sklearn's `IterativeImputer`) with a regressor or classifier chosen per column, and decodes categoricals back to their original string values.

> **Note:** `IterativeImputer` is an experimental feature in scikit-learn. `MixedImputer` handles the required `enable_iterative_imputer` import internally — just import `MixedImputer` and use it. If you are also importing `IterativeImputer` directly in your own code, make sure to import `MixedImputer` **before** `IterativeImputer`, or explicitly run `from sklearn.experimental import enable_iterative_imputer` first.

## Features

- **Wrapper for IterativeImputer** — builds on sklearn's experimental `IterativeImputer` (inspired by MICE), inheriting its battle-tested convergence logic while adding mixed-type support
- **Mixed-type support** — handles numeric and string categorical columns in the same DataFrame
- **Binary & multiclass classification** — categorical columns with 2, 3, or more classes are supported; the classifier automatically adapts to any number of unique categories
- **Auto-detection** — automatically identifies categorical columns by dtype (object/category/string)
- **Iterative imputation (MICE-style)** — models each column as a function of all others
- **Posterior sampling** — supports stochastic imputation via `sample_posterior=True`
- **scikit-learn compatible** — `fit` / `transform` / `fit_transform` API, works in `Pipeline`
- **DataFrame-native** — input a DataFrame, get a DataFrame back
- **Time-series imputation (experimental)** — `TimeSeriesImputer` adds lag features, rolling statistics, and time-aware initial fill (forward fill, interpolation) to make iterative imputation temporal-aware
- **Data corruption utility** — includes a `DataCorrupter` class for benchmarking: introduce MCAR, MAR, or MNAR missing values into datasets (supports .csv, .xlsx, .arff)

## Installation

```bash
pip install mixed-imputer
```

Or for development:

```bash
git clone https://github.com/dnsupp/mixedimputer.git
cd mixedimputer
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
import numpy as np
from mixedimputer import MixedImputer

# Create sample data with missing values (binary & multiclass categoricals)
data = pd.DataFrame({
    'age':       [25, 30, np.nan, 40],
    'city':      ['paris', 'london', np.nan, 'paris'],
    'income':    [50000, np.nan, 70000, 60000],
    'gender':    ['M', 'F', 'M', np.nan],
    'education': ['bachelor', 'master', 'bachelor', np.nan],  # multiclass (3+ categories)
})

# Auto-detect categorical columns (str, object, or category dtype)
# or specify them manually via categorical_features.
imputer = MixedImputer(
    max_iter=5,
    random_state=42,
)

imputed = imputer.fit_transform(data)
print(imputed)
#      age    city   income gender education
# 0  25.000   paris  50000.0      M  bachelor
# 1  30.000  london  57500.0      F    master
# 2  32.500  london  70000.0      M  bachelor
# 3  40.000   paris  60000.0      F  bachelor
```

### Using a custom regressor / classifier

```python
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier

imputer = MixedImputer(
    regressor=BayesianRidge(),
    classifier=RandomForestClassifier(random_state=0),
    sample_posterior=True,
    random_state=42,
)
imputed = imputer.fit_transform(data)
```

## MixedImputer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categorical_features` | list of int/str or None | `None` | Column indices or names that are categorical. If `None` and input is a DataFrame, columns with `object`, `string`, or `category` dtype are auto-detected. |
| `numeric_features` | list of int/str or None | `None` | Numeric column indices or names. If `None`, auto-detected as all columns whose dtype passes `pd.api.types.is_numeric_dtype` and are not listed in `categorical_features`. |
| `regressor` | estimator or None | `HistGradientBoostingRegressor()` | Regressor used for numerical target columns. **Any sklearn regressor** implementing `.fit(X, y)` and `.predict(X)` works (e.g. `Ridge`, `RandomForestRegressor`, `BayesianRidge`). For posterior sampling, a `return_std`-capable model (e.g. `BayesianRidge`) gives better uncertainty estimates, but other models fall back to unit standard deviation. |
| `classifier` | estimator or None | `HistGradientBoostingClassifier()` | Classifier used for categorical target columns. **Any sklearn classifier** implementing `.fit(X, y)` and `.predict(X)` works (e.g. `LogisticRegression`, `RandomForestClassifier`, `GaussianNB`). For posterior sampling, the classifier must also implement `.predict_proba(X)` and expose `.classes_` (most do — `RidgeClassifier` is a notable exception). |
| `max_iter` | int | `10` | Maximum number of imputation rounds. |
| `tol` | float | `1e-3` | Tolerance for early stopping. |
| `initial_strategy` | str | `"mean"` | Initial imputation strategy (`"mean"`, `"median"`, `"most_frequent"`, `"constant"`). |
| `sample_posterior` | bool | `False` | If `True`, sample from predictive posterior for stochastic imputation. |
| `random_state` | int, RandomState or None | `None` | Seed for reproducibility. |
| `verbose` | int | `0` | Verbosity level. |
| `add_indicator` | bool | `False` | If `True`, add missing indicator columns. |
| `keep_empty_features` | bool | `False` | If `True`, keep features that are all-missing at fit time. |

## Time Series Imputation (Experimental)

> **Status: Experimental** — `TimeSeriesImputer` has been verified for functional
> correctness (22 unit tests pass) but has **not yet been benchmarked** against
> established time-series imputation methods (e.g., Kalman filters, state-space
> models, `pandas.interpolate`).  The imputation pipeline works correctly —
> sorting, lag/rolling feature engineering, initial fill, iterative refinement —
> but the statistical accuracy of the imputed values on real-world time series
> has not been quantified.  Use with caution in production and validate results
> against domain knowledge.

### Why not use `MixedImputer` directly on time series?

`IterativeImputer` treats every row as independent — it models `column_j = f(other columns in same row)`.  It has no concept of temporal ordering and cannot exploit autocorrelation (`value_t` depends on `value_{t-1}, value_{t-2}, ...`).  Using a global mean as the initial fill destroys trends and seasonality.

`TimeSeriesImputer` addresses this by:

1. **Sorting** the data by a user-supplied time column.
2. **Temporarily forward-filling** NaN values to create clean lag/rolling features.
3. **Restoring NaN** in the original columns — the imputer then uses the temporal features as predictors to refine the fill.
4. **Delegating** to `MixedImputer` for the core MICE-style iterative imputation.
5. **Stripping** engineered features from the output.

### Quick Diagnostic

```python
from mixedimputer import is_time_series_suitable

report = is_time_series_suitable(df, time_column="timestamp")
print(report["is_sorted"])       # True if time is monotonic
print(report["missing_rate"])     # fraction of missing values
print(report["recommendation"])   # human-readable advice
```

### Quick Start (Time Series)

```python
import pandas as pd
import numpy as np
from mixedimputer import TimeSeriesImputer

# Sensor readings with missing values
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    'temperature': [20.1, 20.3, np.nan, np.nan, 21.0] + [np.nan] * 95,
    'humidity':    [55, 57, np.nan, 60, np.nan] + [np.nan] * 95,
    'sensor_status': ['OK', 'OK', np.nan, 'WARN', 'OK'] + [np.nan] * 95,
})

imputer = TimeSeriesImputer(
    time_column='timestamp',
    lags=[1, 2, 3],               # create t-1, t-2, t-3 lag features
    rolling_windows=[6, 12],       # 6h and 12h rolling means
    initial_strategy='forward_fill',  # time-aware initial fill
    random_state=42,
)
imputed = imputer.fit_transform(df)
# Result contains only original columns — lag/rolling features are stripped
```

### TimeSeriesImputer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_column` | str or int | *(required)* | Column name or index identifying the time/ordering dimension. Must be numeric or datetime. |
| `lags` | list of int | `[1, 2, 3]` | Lag steps for creating lag features per numeric column. Set to `[]` to disable. |
| `rolling_windows` | list of int | `None` | Window sizes for rolling-mean features per numeric column. Set to `[]` to disable. |
| `initial_strategy` | str | `"forward_fill"` | Time-aware fill used **temporarily** to build clean lag/rolling features: `"forward_fill"` / `"ffill"`, `"backward_fill"` / `"bfill"`, `"linear"`, `"spline"`, or standard strategies (`"mean"`, `"median"`, `"most_frequent"`, `"constant"`). After lag features are constructed, NaN is restored in original columns and `MixedImputer` refines the fill via iterative regression. |
| `imputer` | MixedImputer or None | `None` | Pre-configured `MixedImputer` for custom regressor/classifier. If `None`, a default is created. Use this to set `regressor`, `classifier`, `add_indicator`, `keep_empty_features` or other MixedImputer-specific options. |
| `categorical_features` | list of str/int or None | `None` | Column names/indices of categorical features. Used internally to skip lag/rolling creation for non-numeric columns, then passed through to `MixedImputer`. Auto-detected from dtype if `None`. |
| `numeric_features` | list of str/int or None | `None` | Column names/indices of numeric features that receive lag/rolling features. If `None`, auto-detected as all numeric columns (excluding `time_column` and categoricals). |
| `max_iter` | int | `10` | Maximum imputation rounds (passed to `MixedImputer`). |
| `tol` | float | `1e-3` | Early-stopping tolerance (passed to `MixedImputer`). |
| `sample_posterior` | bool | `False` | Posterior sampling for stochastic imputation (passed to `MixedImputer`). |
| `random_state` | int or None | `None` | Seed for reproducibility. |
| `verbose` | int | `0` | Verbosity level. |

> **Note on `initial_strategy`:** Unlike `MixedImputer` where `initial_strategy`
> controls the first-pass fill before iterative refinement, `TimeSeriesImputer`
> uses its `initial_strategy` only for building clean lag/rolling features.
> After those features are created, the original NaN positions are restored and
> the actual imputation is performed by `MixedImputer` using its own
> `initial_strategy` (default: `"mean"`).  To control the final fill, pass a
> pre-configured `MixedImputer` via the `imputer=` parameter:
>
> ```python
> from mixedimputer import MixedImputer, TimeSeriesImputer
>
> imputer = TimeSeriesImputer(
>     time_column="timestamp",
>     imputer=MixedImputer(initial_strategy="median", max_iter=20),
> )
> ```

### Known Limitations (Time Series)

- **Not benchmarked** — functional correctness is verified, but imputation accuracy
  on real time series (e.g., financial, weather, sensor data) has not been
  quantified.  Validate against domain knowledge before production use.
- **Irregular time steps** — the diagnostic can detect gaps, but `TimeSeriesImputer`
  does not automatically resample.  Resample to a regular frequency first.
- **Lag features create NaN at the start** of each series (the first `max(lag)` rows
  have no history).  These are filled by `MixedImputer`'s initial strategy.
- **Only numeric lags** — lag and rolling features are created for numeric columns
  only.  Categorical columns are imputed using cross-sectional correlations.

## How It Works

`MixedImputer` is a thin wrapper that orchestrates sklearn's `IterativeImputer` for mixed-type DataFrames:

1. **Column detection** — object/string/category dtype columns are identified as categorical; the rest as numeric.
2. **Encoding** — categorical columns are encoded to integers using `OrdinalEncoder`, with NaN replaced by a sentinel value so the imputer sees them as truly missing.
3. **Iterative imputation (MICE-style)** — a customized `IterativeImputer` uses `HistGradientBoostingRegressor` for numeric columns and `HistGradientBoostingClassifier` for categorical columns (binary or multiclass). The classifier handles any number of unique categories automatically.
4. **Decoding** — imputed integer values are inverse-transformed back to their original string categories.

## Compatible Estimators

`MixedImputer` accepts **any scikit-learn regressor or classifier** — you are not
limited to the defaults.  Tested and known to work:

| Category | Estimators |
|----------|-----------|
| **Regressors** | `LinearRegression`, `BayesianRidge`, `Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor`, `SGDRegressor`, `HistGradientBoostingRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `ExtraTreesRegressor`, `DecisionTreeRegressor` |
| **Classifiers** | `HistGradientBoostingClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `ExtraTreesClassifier`, `LogisticRegression`, `RidgeClassifier`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `GaussianNB` |

### Posterior sampling notes

- **Regressors with `return_std`** (e.g. `BayesianRidge`) provide per-prediction
  uncertainty, yielding better posterior draws.  Regressors without it fall back
  to unit standard deviation — stochastic imputation still works but assumes
  constant variance.
- **Classifiers with `predict_proba`** (almost all sklearn classifiers) draw
  from the predicted class distribution.  `RidgeClassifier` lacks `predict_proba`
  and will use its plain `predict` output instead.

## Requirements

- Python ≥ 3.10
- numpy
- pandas
- scipy
- scikit-learn ≥ 1.0

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite includes:

- **Imputer unit tests** — verify core functionality, edge cases, reproducibility, and pipeline compatibility.
- **Time-series imputer tests** — 22 tests covering sorting, lag/rolling features, initial fill strategies, categorical handling, edge cases, and diagnostic function.
- **Corruption tests** — verify the `DataCorrupter` with all three missing-data mechanisms (MCAR, MAR, MNAR) on synthetic and real datasets.
- **End-to-end benchmarks** — corrupt real datasets (`titanic.csv`, `credit-g.arff`) and evaluate imputation accuracy against naive baselines (mean/mode).

All tests pass on the included `data/titanic.csv` and `data/credit-g.arff` datasets.

## Benchmarks

MixedImputer and TimeSeriesImputer have been evaluated on real datasets against common baselines (mean/mode imputation, forward-fill) at multiple corruption levels (1%, 10%, 30%).

Key findings:
- **MixedImputer** reduces numeric RMSE by **48–55%** vs. mean imputation on Titanic, and achieves up to **100% categorical accuracy** at low corruption.
- **TimeSeriesImputer** outperforms mean imputation by up to **12.8×** on GPU sensor data, and closely tracks the forward-fill baseline for highly autocorrelated signals.

**Full results, methodology, and reproducible notebooks:** [BENCHMARKS.md](BENCHMARKS.md)

### Quick Benchmarking with Corrupted Data

To evaluate imputation quality on your own data, use the bundled `DataCorrupter`:

```python
from mixedimputer import DataCorrupter, MixedImputer

corrupter = DataCorrupter(mechanism="MCAR", corruption_fraction=0.10,
                          numeric_columns=["Age", "Fare"], random_state=42)
corrupted, mask, original = corrupter.corrupt("data/titanic.csv")

imputer = MixedImputer(max_iter=10, random_state=42)
imputed = imputer.fit_transform(corrupted)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(original.loc[mask["Age"], "Age"],
                          imputed.loc[mask["Age"], "Age"]) ** 0.5
print(f"RMSE on corrupted Age values: {rmse:.2f}")
```

See [`examples/example.py`](examples/example.py) for a complete pipeline.

## Examples

See [`examples/examples.py`](examples/examples.py) for a runnable script that demonstrates all major
features including auto-detection, posterior sampling, custom estimators, array
input, pipeline integration, and edge cases.

## Known Limitations

- **`TimeSeriesImputer` is experimental** — see the [Time Series section](#time-series-imputation-experimental) for details on what is and isn't verified.
- **Unseen categories at transform time** are encoded to the `unknown_value` of
  the `OrdinalEncoder`.  If the imputer assigns a value that decodes to an
  unknown category it will appear as `NaN` in the output.  Ensure the training
  set covers the full vocabulary of each categorical column when possible.
- **`keep_empty_features=False`** (the default) drops columns that are entirely
  `NaN` during `fit`.  The dropped columns are removed from the output
  DataFrame.

## License

MIT — see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/dnsupp/mixedimputer)
