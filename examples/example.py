"""Example usage of MixedTypeImputer."""

import numpy as np
import pandas as pd
from mixed_imputer import MixedTypeImputer

# Fake mixed data with strings and numbers
data = pd.DataFrame({
    'age': [25, 30, np.nan, 40],
    'city': ['paris', 'london', np.nan, 'paris'],
    'income': [50000, np.nan, 70000, 60000],
    'gender': ['M', 'F', 'M', np.nan],
})

# Auto-detect categorical columns (object dtype is picked up automatically)
imputer = MixedTypeImputer(
    max_iter=5,
    random_state=42,
)

imputed = imputer.fit_transform(data)
print("Original data with missing values:")
print(data)
print()
print("Imputed data:")
print(imputed)
print()
print(f"Any remaining NaN: {imputed.isnull().any().any()}")
print(f"Categorical columns: {imputer.categorical_features}")
print(f"Output dtypes:\n{imputed.dtypes}")
