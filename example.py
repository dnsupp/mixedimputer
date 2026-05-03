"""Example usage of MixedTypeImputer.

To run: pip install -e . && python example.py
"""
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

imputer = MixedTypeImputer(
    categorical_features=['city', 'gender'],  # you can also omit this – auto-detects object dtype
    max_iter=5,
    random_state=42,
)

imputed = imputer.fit_transform(data)
print(imputed)
