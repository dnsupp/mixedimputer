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

> **MixedImputer is a wrapper around scikit-learn's [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html), which implements an iterative imputation strategy inspired by MICE (Multivariate Imputation by Chained Equations).** It extends it to handle DataFrames containing both numerical and categorical (string/boolean) columns.

It automatically detects column types, encodes categoricals with `OrdinalEncoder`, runs an iterative MICE-style imputer with a regressor or classifier chosen per column, and decodes categoricals back to their original values.

> **Note:** `IterativeImputer` is an experimental feature in scikit-learn. `MixedImputer` handles the required `enable_iterative_imputer` import internally — just import `MixedImputer` and use it. If you are also importing `IterativeImputer` directly, import `MixedImputer` first or run `from sklearn.experimental import enable_iterative_imputer` explicitly.

## Features

- **Mixed-type support** — numeric columns plus string, category, and boolean columns in one DataFrame.
- **Binary & multiclass classification** — categorical columns with any number of observed classes are imputed with a classifier.
- **Auto-detection** — object/string/category/boolean columns are detected automatically; datetime columns raise a clear error instead of being silently dropped.
- **Iterative imputation (MICE-style)** — each column is modeled as a function of the others.
- **Posterior sampling** — stochastic single imputation via `sample_posterior=True`.
- **scikit-learn-compatible API** — `fit` / `transform` / `fit_transform`, fitted attributes, `get_feature_names_out`, and Pipeline use.
- **Missing indicators** — `add_indicator=True` appends boolean `missingindicator_<column>` columns.
- **DataFrame-native** — input a DataFrame, get a DataFrame back with the fitted column order.
- **Time-series imputation (experimental)** — `TimeSeriesImputer` treats the time column as metadata, restores row order/index, and adds lag/rolling features.
- **Data corruption utility** — `DataCorrupter` creates MCAR, MAR, and MNAR missingness for benchmarking.

## Installation

```bash
pip install mixed-imputer
```

For reading `.arff` or `.xlsx` files with `DataCorrupter`:

```bash
pip install "mixed-imputer[io]"
```

For development (tests, build tools, notebooks):

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

data = pd.DataFrame({
    'age':       [25, 30, np.nan, 40],
    'city':      ['paris', 'london', np.nan, 'paris'],
    'income':    [50000, np.nan, 70000, 60000],
    'gender':    ['M', 'F', 'M', np.nan],
    'education': ['bachelor', 'master', 'bachelor', np.nan],
})

imputer = MixedImputer(max_iter=5, random_state=42)
imputed = imputer.fit_transform(data)
print(imputed)
#         age    city   income gender education
# 0  25.000000   paris  50000.0      M  bachelor
# 1  30.000000  london  60000.0      F    master
# 2  31.666667   paris  70000.0      M  bachelor
# 3  40.000000   paris  60000.0      M  bachelor
```

Output is illustrative for the current supported stack; rerunning on a different
scikit-learn version may produce slightly different numeric values.

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
| `categorical_features` | list of int/str or None | `None` | Column names or positional indices treated as categorical. If `None`, object/string/category/boolean dtypes are auto-detected. |
| `numeric_features` | list of int/str or None | `None` | Numeric columns. If `None`, auto-detected as numeric, non-boolean columns not listed as categorical. |
| `regressor` | estimator or None | `HistGradientBoostingRegressor()` | Regressor for numeric target columns. Any sklearn regressor with `fit`/`predict`; `return_std` gives better posterior draws. |
| `classifier` | estimator or None | `HistGradientBoostingClassifier()` | Classifier for categorical target columns. Any sklearn classifier with `fit`/`predict`; `predict_proba` improves posterior sampling. |
| `max_iter` | int | `10` | Maximum imputation rounds. |
| `tol` | float | `1e-3` | Early-stopping tolerance. |
| `initial_strategy` | str | `"mean"` | `"mean"`, `"median"`, `"most_frequent"`, or `"constant"` (sklearn initial fill). |
| `sample_posterior` | bool | `False` | Draw stochastic single imputations from the predictive distribution. |
| `random_state` | int, RandomState or None | `None` | Seed for reproducibility. |
| `verbose` | int | `0` | Verbosity of the underlying imputer. |
| `add_indicator` | bool | `False` | Append boolean missing-indicator columns for features missing at fit. |
| `keep_empty_features` | bool | `False` | Keep all-missing features. With `False`, they are dropped. |

Explicit feature lists are additive: a column not listed in either
`categorical_features` or `numeric_features` is assigned by dtype. Columns with
unsupported dtypes (datetime/timedelta) raise a clear error.

### Fitted attributes

- `categorical_features_`, `numeric_features_` — resolved feature lists.
- `n_features_in_`, `feature_names_in_` — input schema.
- `encoder_` — fitted `OrdinalEncoder`.
- `imputation_model_` — fitted internal iterative imputer.
- `n_features_out_` — number of output columns, including indicators.
- `get_feature_names_out()` — output names including `missingindicator_*`.

Constructor parameters are **not** mutated by `fit`; `fit` can be called again on a new schema.

## Time Series Imputation (Experimental)

> **Status:** `TimeSeriesImputer` has functional tests and a small benchmark on
> one GPU sensor log, but has **not** been validated against state-space models,
> Kalman filters, or multiple real-world time-series datasets. Treat it as
> experimental and validate against domain knowledge.

The time column is used only for ordering. It is not imputed, it is re-inserted
at its original position, and the input row order/index are restored after
imputation.

### Quick Diagnostic

```python
from mixedimputer import is_time_series_suitable

report = is_time_series_suitable(df, time_column="timestamp")
print(report["is_sorted"])
print(report["is_regular"])
print(report["has_gaps"])
print(report["recommendation"])
```

### Quick Start (Time Series)

```python
import pandas as pd
import numpy as np
from mixedimputer import TimeSeriesImputer

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
    'temperature': [20.1, 20.3, np.nan, np.nan, 21.0] + [np.nan] * 95,
    'humidity':    [55, 57, np.nan, 60, np.nan] + [np.nan] * 95,
    'sensor_status': ['OK', 'OK', np.nan, 'WARN', 'OK'] + [np.nan] * 95,
})

imputer = TimeSeriesImputer(
    time_column='timestamp',
    lags=[1, 2, 3],
    rolling_windows=[6, 12],
    initial_strategy='forward_fill',
    random_state=42,
)
imputed = imputer.fit_transform(df)
# Same columns/index/order as df; lag/rolling features are stripped.
```

### TimeSeriesImputer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_column` | str or int | *(required)* | Column name or positional index used for ordering. Not imputed. |
| `lags` | list/tuple of int or None | `(1, 2, 3)` | Positive lag steps for numeric features. `None` disables lags. |
| `rolling_windows` | list of int or None | `None` | Positive rolling-mean windows. The window is shifted by one row so it excludes the current target value. |
| `initial_strategy` | str | `"forward_fill"` | Temporary fill used only to build lag/rolling features: `forward_fill`/`ffill`, `backward_fill`/`bfill`, `linear`, `spline` (numeric only), `mean`, `median`, `most_frequent`, `constant`. |
| `imputer` | MixedImputer or None | `None` | Custom internal imputer. If supplied, the TimeSeries `max_iter`/`tol`/`sample_posterior`/`random_state`/`verbose` arguments are not used. |
| `categorical_features` | list of str/int or None | `None` | Categorical feature columns. Used to avoid lags/rolling on non-numeric columns and passed to the default `MixedImputer`. |
| `numeric_features` | list of str/int or None | `None` | Numeric columns that receive lag/rolling features. This controls feature engineering only; engineered features are auto-detected by the internal imputer. |
| `max_iter` | int | `10` | Passed to the default `MixedImputer`. |
| `tol` | float | `1e-3` | Passed to the default `MixedImputer`. |
| `sample_posterior` | bool | `False` | Passed to the default `MixedImputer`. |
| `random_state` | int or None | `None` | Seed for reproducibility. |
| `verbose` | int | `0` | Verbosity. |

> **Note on `initial_strategy`:** it is used only to create clean lag/rolling
> features. After those features are built, the original NaN positions are
> restored and `MixedImputer` performs the final imputation. To control the
> final fill, pass a pre-configured `MixedImputer` via `imputer=`.

### Known Limitations (Time Series)

- **Not validated broadly** — one GPU sensor log is not enough to establish
  general time-series accuracy. The benchmark is reported with sample sizes,
  and forward-fill beats the imputer on some columns.
- **Single series** — lags and rolling windows are row-based. Multiple entities
  in one DataFrame will leak across entity boundaries; add a group/resample
  step outside the imputer.
- **Irregular sampling** — resample to a regular frequency first. `lag 1` means
  one row, not one time unit.
- **Temporary fill can look ahead** — `forward_fill`/`backward_fill` may use
  future observations at the leading edge. This is acceptable for retrospective
  imputation but not for causal/online forecasting.
- **All-missing feature columns** are kept as `NaN` in the output; the
  underlying `MixedImputer` has no information to impute them.

## How It Works

`MixedImputer` orchestrates sklearn's `IterativeImputer` for mixed-type DataFrames:

1. **Column detection** — object/string/category/boolean columns are categorical; numeric non-boolean columns are numeric. Datetime/timedelta columns raise a clear error.
2. **Encoding** — categoricals are stringified and encoded with `OrdinalEncoder` using a per-fit sentinel for missing values; unseen categories at transform time are treated as missing.
3. **Iterative imputation** — a customized `IterativeImputer` uses a regressor for numeric columns and a classifier for categorical columns.
4. **Decoding** — integer predictions are inverse-decoded to the original categories; boolean columns are cast back to boolean.
5. **Reconstruction** — output is rebuilt in the fitted column order, with optional missing indicators.

`TimeSeriesImputer` additionally sorts by time, creates lag/rolling features, delegates to `MixedImputer`, then restores the original order/index and strips engineered features.

## Compatible Estimators

`MixedImputer` accepts scikit-learn regressors/classifiers that implement `fit` and `predict`. Tested examples:

| Category | Estimators |
|----------|-----------|
| **Regressors** | `LinearRegression`, `BayesianRidge`, `Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor`, `SGDRegressor`, `HistGradientBoostingRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `ExtraTreesRegressor`, `DecisionTreeRegressor` |
| **Classifiers** | `HistGradientBoostingClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `ExtraTreesClassifier`, `LogisticRegression`, `RidgeClassifier`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `GaussianNB` |

- Regressors with `return_std` give better posterior draws; others fall back to unit standard deviation.
- Classifiers with `predict_proba` use the predicted class distribution; others use `predict`.
- Very small datasets and single-class categorical columns may need a simpler estimator.

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.23
- pandas ≥ 1.5
- scipy ≥ 1.8
- scikit-learn ≥ 1.6, < 1.9

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite includes unit tests, real-dataset integration tests, time-series tests,
and regression tests for the release-blocking fixes. The full suite takes a few
minutes because it trains many small gradient-boosting models.

## Benchmarks

Benchmarks are reproducible with the headless script
[`benchmark/run_benchmarks.py`](https://github.com/dnsupp/mixedimputer/blob/main/benchmark/run_benchmarks.py):

```bash
python benchmark/run_benchmarks.py
```

The notebooks in `benchmark/` are for interactive exploration. Full tables,
methodology, and caveats are in
[`BENCHMARKS.md`](https://github.com/dnsupp/mixedimputer/blob/main/BENCHMARKS.md).

A separate Phase 1 time-series validation harness compares `TimeSeriesImputer`
with ffill/bfill, linear interpolation, observed mean, an HGB-lag model, and
temporal-feature ablations across synthetic datasets and point/block missingness:

```bash
python benchmark/run_timeseries_validation.py --quick
python benchmark/run_timeseries_validation.py --seeds 20
```

It uses a fast `BayesianRidge`/`LogisticRegression` mode by default for
practical many-seed runs; pass `--estimator default` to test the package's
HistGradientBoosting defaults. The harness currently covers synthetic data and
the bundled GPU log; public datasets and Kalman/state-space baselines remain
Phase 2–3 work. A 10-seed preliminary run found linear interpolation best in
14/21 dataset/pattern cells, so the current evidence does **not** support a
general time-series superiority claim. See `BENCHMARKS.md` for the details.

### What the numbers actually show

On Titanic at 10% MCAR corruption, `MixedImputer` reduced Age RMSE from 13.64
(observed-mean baseline) to 12.93, and Fare RMSE from 48.68 to 31.10. At 30%,
Age improved only from 14.36 to 13.22, while Fare improved from 38.04 to 31.63.
At 1%, only 7–9 cells per column were corrupted, so those results are noisy.

For categorical columns, Sex accuracy at 10% was 82.0% vs 65.2% for mode;
Embarked was 78.7% vs 69.7%. At 30%, Embarked was 75.7% vs 79.8% for mode — the
imputer did **not** beat the baseline there. There is no claim of universal
superiority.

On the single GPU sensor log, at 10% corruption `TimeSeriesImputer` reduced
Fan Speed RMSE from 289.10 (mean) to 18.38 and from 27.98 (forward fill) to
18.38. GPU Temperature was 1.06 vs 4.05 (mean) but 0.94 for forward fill, so
forward-fill was better there. At 1% only four cells per column were corrupted.

These are single-seed results; `BENCHMARKS.md` documents sample sizes and the
remaining statistical limitations. Repeated-seed confidence intervals are a
planned improvement, not a claim made here.

## Data Corruption Utility

`DataCorrupter` is part of the public API and works on `.csv` files or
DataFrames without extra dependencies:

```python
from pathlib import Path
from mixedimputer import DataCorrupter, MixedImputer

corrupter = DataCorrupter(
    mechanism="MCAR",
    corruption_fraction=0.10,
    numeric_columns=["Age", "Fare"],
    random_state=42,
)
corrupted, mask, original = corrupter.corrupt(Path("data/titanic.csv"))

imputer = MixedImputer(max_iter=10, random_state=42)
imputed = imputer.fit_transform(corrupted)
```

- **MCAR** masks a uniform random subset of observed cells.
- **MAR** masks cells with high scores from a linear model of the other columns.
- **MNAR** masks high or low values of the target itself (`mnar_extreme="high"`/`"low"`).
- The requested fraction is applied to **observed** cells, not total cells.
- Already-missing cells are never overwritten.
- The same `random_state` gives the same mask; global NumPy/PyTorch RNG state is not modified.
- `.arff` and `.xlsx`/`.xls` require `pip install "mixed-imputer[io]"`.

## Examples

See [`examples/examples.py`](https://github.com/dnsupp/mixedimputer/blob/main/examples/examples.py)
for a runnable tour of auto-detection, posterior sampling, custom estimators,
array input, Pipeline integration, and edge cases. A minimal example is in
[`examples/example.py`](https://github.com/dnsupp/mixedimputer/blob/main/examples/example.py).

## Known Limitations

- **Datetime columns in `MixedImputer`** are rejected with a `ValueError`; encode
  them as numeric/string first, or let `TimeSeriesImputer` handle the time column.
- **All-missing categorical columns** are dropped by default; set
  `keep_empty_features=True` to retain them as `NaN`.
- **Array input** requires explicit `categorical_features`; string categorical
  arrays return an object array, numeric-coded arrays return floats.
- **Unseen categories at transform time** are treated as missing and imputed to
  a known category when possible.
- **`sample_posterior=True` is single imputation** — it returns one completed
  dataset, not `m` multiply-imputed datasets with pooled inference.
- **`TimeSeriesImputer`** does not handle multiple entities, irregular sampling,
  or causal/online forecasting out of the box.
- **High-cardinality categoricals** can make the default classifier slow; use a
  simpler classifier or drop ID-like columns.

## Citation

This package accompanies the author's bachelor's thesis. If you use it in
academic work, please cite the thesis and this repository. A machine-readable
entry is provided in [`CITATION.cff`](https://github.com/dnsupp/mixedimputer/blob/main/CITATION.cff);
add the final thesis title/DOI there when available.

## License

MIT — see [LICENSE](https://github.com/dnsupp/mixedimputer/blob/main/LICENSE).

## Links

- [GitHub Repository](https://github.com/dnsupp/mixedimputer)
- [Issue Tracker](https://github.com/dnsupp/mixedimputer/issues)
- [Benchmarks](https://github.com/dnsupp/mixedimputer/blob/main/BENCHMARKS.md)
