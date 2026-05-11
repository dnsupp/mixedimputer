"""Examples of MixedImputer usage.

Run with:
    pip install -e .
    python examples/examples.py
"""

import numpy as np
import pandas as pd
from mixedimputer import MixedImputer


# =========================================================================
# Helper
# =========================================================================

def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =========================================================================
# 1. Basic usage with auto-detection of categorical columns
# =========================================================================
section("1. Basic usage (auto-detection)")

data = pd.DataFrame({
    "age":    [25, 30, np.nan, 40, 35],
    "city":   ["paris", "london", np.nan, "paris", "london"],
    "income": [50000, np.nan, 70000, 60000, 55000],
    "gender": ["M", "F", "M", np.nan, "F"],
})

imputer = MixedImputer(max_iter=5, random_state=42)
imputed = imputer.fit_transform(data)

print("Original (with NaNs):")
print(data.to_string())
print()
print("Imputed:")
print(imputed.to_string())
print()
print(f"Any remaining NaN: {imputed.isnull().any().any()}")
print(f"Auto-detected categoricals: {imputer.categorical_features}")


# =========================================================================
# 2. Explicit categorical features
# =========================================================================
section("2. Explicit categorical_features")

imputer2 = MixedImputer(
    categorical_features=["city", "gender"],
    max_iter=5,
    random_state=99,
)
imputed2 = imputer2.fit_transform(data)
print(imputed2.to_string())


# =========================================================================
# 3. Posterior sampling (stochastic imputation)
# =========================================================================
section("3. Posterior sampling (sample_posterior=True)")

imputer3 = MixedImputer(sample_posterior=True, max_iter=5, random_state=42)
imputed3a = imputer3.fit_transform(data)

# Running again with a different seed shows variation
imputer3b = MixedImputer(
    sample_posterior=True, max_iter=5, random_state=123
)
imputed3b = imputer3b.fit_transform(data)

print("Run 1 (seed=42):")
print(imputed3a.to_string())
print()
print("Run 2 (seed=123) — different draws:")
print(imputed3b.to_string())


# =========================================================================
# 4. Custom regressor and classifier
# =========================================================================
section("4. Custom regressor & classifier")

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier

imputer4 = MixedImputer(
    regressor=BayesianRidge(),
    classifier=RandomForestClassifier(random_state=0),
    max_iter=5,
    random_state=42,
)
imputed4 = imputer4.fit_transform(data)
print(imputed4.to_string())


# =========================================================================
# 5. Array input (NumPy ndarray)
# =========================================================================
section("5. Array input")

rng = np.random.default_rng(42)
X_arr = rng.random((10, 4))
X_arr[0, 0] = np.nan
X_arr[2, 1] = np.nan
X_arr[5, 3] = np.nan
# Make column 3 categorical (0 or 1)
X_arr[:, 3] = (X_arr[:, 3] > 0.5).astype(float)

imputer5 = MixedImputer(
    categorical_features=[3],
    max_iter=5,
    random_state=42,
)
result_arr = imputer5.fit_transform(X_arr)

print("Original (first 4 rows):")
print(X_arr[:4])
print()
print("Imputed (first 4 rows):")
print(result_arr[:4])
print()
print(f"Any remaining NaN: {np.isnan(result_arr).any()}")


# =========================================================================
# 6. Pipeline integration
# =========================================================================
section("6. Pipeline integration")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df_num = data[["age", "income"]]

pipe = Pipeline([
    ("imputer", MixedImputer(max_iter=5, random_state=42)),
    ("scaler", StandardScaler()),
])
pipe_result = pipe.fit_transform(df_num)
print("Piped (age, income after imputation + scaling):")
print(pipe_result[:3])


# =========================================================================
# 7. Different initial strategies
# =========================================================================
section("7. Different initial_strategy values")

for strat in ["mean", "median", "most_frequent"]:
    imputer_s = MixedImputer(
        initial_strategy=strat, max_iter=5, random_state=42
    )
    imputed_s = imputer_s.fit_transform(data)
    print(f"  {strat:>14s}  →  age[2]={imputed_s.loc[2, 'age']:.2f}"
          f"  city[2]={imputed_s.loc[2, 'city']}")


# =========================================================================
# 8. All-numeric DataFrame
# =========================================================================
section("8. All-numeric DataFrame")

df_all_num = pd.DataFrame({
    "x": [1.0, np.nan, 3.0, 4.0, np.nan],
    "y": [np.nan, 2.0, np.nan, 4.0, 5.0],
})
imputer8 = MixedImputer(max_iter=5, random_state=42)
imputed8 = imputer8.fit_transform(df_all_num)
print(imputed8.to_string())
print(f"\nCategorical features: {imputer8.categorical_features}")


# =========================================================================
# 9. All-categorical DataFrame
# =========================================================================
section("9. All-categorical DataFrame")

df_all_cat = pd.DataFrame({
    "color": ["red", "blue", np.nan, "red", "green"],
    "size":  ["S", np.nan, "M", "L", "M"],
})
imputer9 = MixedImputer(max_iter=5, random_state=42)
imputed9 = imputer9.fit_transform(df_all_cat)
print(imputed9.to_string())
print(f"\nCategorical features: {imputer9.categorical_features}")


# =========================================================================
# 10. All-missing column (keep_empty_features)
# =========================================================================
section("10. keep_empty_features")

df_miss = pd.DataFrame({
    "age":      [25, 30, np.nan, 40],
    "city":     ["paris", "london", "paris", "london"],
    "all_nan":  [np.nan, np.nan, np.nan, np.nan],
})

# keep_empty_features=False (default) — drops all-NaN columns
imputer10a = MixedImputer(keep_empty_features=False, random_state=42)
result10a = imputer10a.fit_transform(df_miss)
print("keep_empty_features=False  →  columns:", list(result10a.columns))

# keep_empty_features=True — keeps them
imputer10b = MixedImputer(keep_empty_features=True, random_state=42)
result10b = imputer10b.fit_transform(df_miss)
print("keep_empty_features=True   →  columns:", list(result10b.columns))


# =========================================================================
# 11. Category dtype input
# =========================================================================
section("11. CategoricalDtype input")

df_cat = pd.DataFrame({
    "value": [1.0, np.nan, 3.0, 4.0],
    "group": pd.Categorical(["A", "B", np.nan, "A"]),
})
imputer11 = MixedImputer(max_iter=3, random_state=42)
imputed11 = imputer11.fit_transform(df_cat)
print(imputed11.to_string())
print()
print("group dtype:", imputed11["group"].dtype)
print("group values:", imputed11["group"].tolist())


# =========================================================================
# 12. Reproducibility
# =========================================================================
section("12. Reproducibility with random_state")

imputer_a = MixedImputer(random_state=42)
imputer_b = MixedImputer(random_state=42)

result_a = imputer_a.fit_transform(data)
result_b = imputer_b.fit_transform(data)

identical = result_a.equals(result_b)
print(f"Same seed → identical results: {identical}")


print()
print("All examples completed successfully.")
