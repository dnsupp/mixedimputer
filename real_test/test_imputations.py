"""
Real-world test of mixedimputer installed from PyPI.

Run:
    pip install mixedimputer
    python real_test/test_imputations.py
"""

import numpy as np
import pandas as pd
from mixedimputer import MixedImputer, __version__

print(f"mixedimputer version: {__version__}")
print()


# =============================================================================
# Test 1 — Basic mixed data
# =============================================================================
print("=" * 60)
print("Test 1 — Basic mixed DataFrame")
print("=" * 60)

data = pd.DataFrame({
    "age":        [25, 30, np.nan, 40, 35],
    "city":       ["paris", "london", np.nan, "paris", "london"],
    "income":     [50000, np.nan, 70000, 60000, 55000],
    "gender":     ["M", "F", "M", np.nan, "F"],
    "education":  ["bachelor", "master", "bachelor", np.nan, "phd"],
})

imputer = MixedImputer(max_iter=5, random_state=42)
result = imputer.fit_transform(data)

print("Original:")
print(data.to_string())
print()
print("Imputed:")
print(result.to_string())
print()
print(f"  NaNs remaining: {result.isnull().any().any()}")
print(f"  Categoricals detected: {imputer.categorical_features}")
assert not result.isnull().any().any(), "FAIL: NaNs remain!"
print("  PASS")


# =============================================================================
# Test 2 — Posterior sampling
# =============================================================================
print()
print("=" * 60)
print("Test 2 — Posterior sampling (stochastic)")
print("=" * 60)

imputer_sp = MixedImputer(sample_posterior=True, max_iter=5, random_state=42)
result_sp = imputer_sp.fit_transform(data)

print(result_sp.to_string())
print(f"  NaNs remaining: {result_sp.isnull().any().any()}")
assert not result_sp.isnull().any().any(), "FAIL: NaNs remain!"
print("  PASS")


# =============================================================================
# Test 3 — Custom estimators
# =============================================================================
print()
print("=" * 60)
print("Test 3 — Custom regressor & classifier")
print("=" * 60)

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier

imputer_cust = MixedImputer(
    regressor=BayesianRidge(),
    classifier=RandomForestClassifier(random_state=0),
    max_iter=5,
    random_state=42,
)
result_cust = imputer_cust.fit_transform(data)

print(result_cust.to_string())
print(f"  NaNs remaining: {result_cust.isnull().any().any()}")
assert not result_cust.isnull().any().any(), "FAIL: NaNs remain!"
print("  PASS")


# =============================================================================
# Test 4 — All-numeric DataFrame
# =============================================================================
print()
print("=" * 60)
print("Test 4 — All-numeric DataFrame")
print("=" * 60)

num_data = pd.DataFrame({
    "x": [1.0, np.nan, 3.0, np.nan, 5.0],
    "y": [np.nan, 2.0, np.nan, 4.0, 5.0],
    "z": [10.0, 20.0, np.nan, 40.0, np.nan],
})
imputer_num = MixedImputer(max_iter=5, random_state=42)
result_num = imputer_num.fit_transform(num_data)

print(result_num.to_string())
print(f"  Categoricals detected: {imputer_num.categorical_features}")
print(f"  NaNs remaining: {result_num.isnull().any().any()}")
assert not result_num.isnull().any().any(), "FAIL: NaNs remain!"
assert imputer_num.categorical_features == [], "FAIL: Numeric cols detected as categorical!"
print("  PASS")


# =============================================================================
# Test 5 — Reproducibility
# =============================================================================
print()
print("=" * 60)
print("Test 5 — Reproducibility with random_state")
print("=" * 60)

a = MixedImputer(random_state=42)
b = MixedImputer(random_state=42)

res_a = a.fit_transform(data)
res_b = b.fit_transform(data)

identical = res_a.equals(res_b)
print(f"  Same seed → identical: {identical}")
assert identical, "FAIL: Results differ with same seed!"
print("  PASS")


# =============================================================================
# Test 6 — Pipeline integration
# =============================================================================
print()
print("=" * 60)
print("Test 6 — scikit-learn Pipeline")
print("=" * 60)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("imputer", MixedImputer(max_iter=5, random_state=42)),
    ("scaler", StandardScaler()),
])
num_only = data[["age", "income"]]
pipe_result = pipe.fit_transform(num_only)

print("  Pipeline output shape:", pipe_result.shape)
print("  First 3 rows scaled:")
print(f"  {pipe_result[:3].tolist()}")
assert pipe_result.shape == num_only.shape, "FAIL: Wrong shape!"
assert not np.isnan(pipe_result).any(), "FAIL: NaNs in pipeline output!"
print("  PASS")


# =============================================================================
# Test 7 — Single column edge cases
# =============================================================================
print()
print("=" * 60)
print("Test 7 — Single-column edge cases")
print("=" * 60)

df_single_num = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
r = MixedImputer(random_state=42).fit_transform(df_single_num)
assert not r.isnull().any().any()
print(f"  Single numeric col:  PASS  ({r.iloc[1,0]:.3f})")

df_single_cat = pd.DataFrame({"x": ["a", np.nan, "b"]})
r = MixedImputer(random_state=42).fit_transform(df_single_cat)
assert not r.isnull().any().any()
print(f"  Single categorical:  PASS  ({r.iloc[1,0]})")


# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)
print(f"  Package version: {__version__}")
print(f"  Python:          {pd.__version__=}")
print(f"  NumPy:           {np.__version__}")
print(f"  sklearn:         {__import__('sklearn').__version__}")
