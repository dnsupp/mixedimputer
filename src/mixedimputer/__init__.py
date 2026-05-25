"""Mixed-type MICE-style iterative imputation for scikit-learn.

Provides :class:`MixedImputer` for tabular mixed-type DataFrames and
:class:`TimeSeriesImputer` for time-series data with temporal feature
engineering.
"""

from ._imputer import MixedImputer
from ._timeseries import TimeSeriesImputer, is_time_series_suitable
from ._version import __version__

__all__ = [
    "MixedImputer",
    "TimeSeriesImputer",
    "is_time_series_suitable",
    "__version__",
]