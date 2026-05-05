# MixedTypeImputer

**MixedTypeImputer** is a scikit-learn compatible transformer that performs iterative imputation (MICE) on DataFrames containing both numerical and categorical (string) columns.

It automatically detects column types, encodes categoricals with `OrdinalEncoder`, runs an iterative MICE‑style imputer that chooses a regressor or classifier per column, and decodes categoricals back to their original string values.

## Features

- **Mixed-type support** — handles numeric and string categorical columns in the same DataFrame
- **Binary & multiclass classification** — categorical columns with 2, 3, or more classes are supported; the classifier automatically adapts to any number of unique categories
- **Auto-detection** — automatically identifies categorical columns by dtype (object/category/string)
- **Iterative imputation (MICE)** — models each column as a function of all others
- **Posterior sampling** — supports stochastic imputation via `sample_posterior=True`
- **scikit-learn compatible** — `fit` / `transform` / `fit_transform` API, works in `Pipeline`
- **DataFrame-native** — input a DataFrame, get a DataFrame back

## Installation

```bash
pip install mixed-imputer
```

Or for development:

```bash
git clone https://github.com/dnsupp/mixed-imputer.git
cd mixed-imputer
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
import numpy as np
from mixed_imputer import MixedTypeImputer

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
imputer = MixedTypeImputer(
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

imputer = MixedTypeImputer(
    regressor=BayesianRidge(),
    classifier=RandomForestClassifier(random_state=0),
    sample_posterior=True,
    random_state=42,
)
imputed = imputer.fit_transform(data)
```

## Parameters

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

## How It Works

1. **Column detection** — object/string/category dtype columns are identified as categorical; the rest as numeric.
2. **Encoding** — categorical columns are encoded to integers using `OrdinalEncoder`, with NaN replaced by a sentinel(regression) and `HistGradientBoostingClassifier` for categorical columns (binary or multiclass classification). The classifier handles any number of unique categories automatically
3. **Iterative imputation** — a modified `IterativeImputer` uses `HistGradientBoostingRegressor` for numeric columns and `HistGradientBoostingClassifier` for categorical columns.
4. **Decoding** — imputed integer values are inverse-transformed back to their original string categories.

## Compatible Estimators

`MixedTypeImputer` accepts **any scikit-learn regressor or classifier** — you are not
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

- Python ≥ 3.9
- numpy
- pandas
- scipy
- scikit-learn ≥ 1.0

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Examples

See [`examples/examples.py`](examples/examples.py) for a runnable script that demonstrates all major
features including auto-detection, posterior sampling, custom estimators, array
input, pipeline integration, and edge cases.

## Known Limitations

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

- [GitHub Repository](https://github.com/dnsupp/mixed-imputer)
