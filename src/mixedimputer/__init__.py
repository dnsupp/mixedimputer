"""Mixed-type MICE-style iterative imputation for scikit-learn.

Provides :class:`MixedImputer` for tabular mixed-type DataFrames,
:class:`TimeSeriesImputer` for time-series data with temporal feature
engineering, and :class:`DataCorrupter` for controlled missing-data
benchmarks.
"""

from ._imputer import MixedImputer
from ._timeseries import TimeSeriesImputer, is_time_series_suitable
from ._version import __version__
from .corrupt_data import DataCorrupter

__all__ = [
    "MixedImputer",
    "TimeSeriesImputer",
    "is_time_series_suitable",
    "DataCorrupter",
    "__version__",
]