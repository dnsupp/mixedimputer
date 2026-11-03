# Changelog

All notable changes to this project are documented here.

## [0.2.0] — 2026-11-03

This release fixes the release-blocking findings from the repository review.
Some behavior changes are intentional and breaking; they are listed first.

### Breaking changes

- `MixedImputer` no longer mutates `categorical_features`/`numeric_features`
  during `fit`. Use the new fitted attributes `categorical_features_` and
  `numeric_features_`.
- `MixedImputer` now returns output in the **fitted** column order, not the
  transform input's order, and validates the transform schema.
- Boolean columns are treated as categorical and returned as boolean dtype
  instead of being imputed as continuous numbers.
- Datetime/timedelta columns now raise a clear `ValueError` instead of failing
  later with a `KeyError`. `TimeSeriesImputer` handles its own time column.
- `TimeSeriesImputer` restores the original row order and index; it no longer
  returns a sorted `RangeIndex`.
- `TimeSeriesImputer.lags` now defaults to `(1, 2, 3)`; pass `None` to disable
  lags. Previously `None` also meant "use the default".
- `TimeSeriesImputer` linear/spline interpolation applies to numeric columns
  only; categorical columns use forward/backward fill.
- `TimeSeriesImputer` rolling features are shifted by one row, excluding the
  current target value.
- `DataCorrupter` no longer depends on `pygrinder`/PyTorch. Mechanisms are
  implemented directly with NumPy, target the requested fraction of observed
  cells, and use a local RNG (the global NumPy RNG is untouched).
- MAR/MNAR corruption behavior is now deterministic from `random_state` and
  produces non-zero corruption for a non-zero fraction.
- `DataCorrupter` is exported from the top-level `mixedimputer` namespace.
- `scikit-learn` minimum version is now `1.6` (array input uses
  `ensure_all_finite`), with an upper bound of `<1.9` while the private
  iterative-imputer hook is used.

### Added

- `MixedImputer.add_indicator=True` now emits `missingindicator_<column>`
  boolean columns.
- `MixedImputer.n_features_in_`, `feature_names_in_`,
  `n_features_out_`, and `get_feature_names_out()`.
- `TimeSeriesImputer.n_features_in_`, `feature_names_in_`,
  `get_feature_names_out()`, and `n_features_out_`.
- `is_time_series_suitable()` now reports `has_duplicates` and recommends
  resampling for irregular timestamps.
- `DataCorrupter` accepts `pathlib.Path` and validates fractions/counts.
- `[io]` extra for `.arff` (`liac-arff`) and `.xlsx`/`.xls` (`openpyxl`, `xlrd`).
- `benchmark/run_benchmarks.py`, a headless reproducible benchmark runner.
- Windows and Python 3.13 CI jobs, a minimum-dependency job, and a build/twine
  check.
- `CHANGELOG.md`.

### Fixed

- All-missing categorical columns no longer crash with
  `keep_empty_features=False`; they are dropped like all-missing numeric
  columns.
- Array input no longer broadcasts a dropped all-missing feature into another
  column; dropped positions are `NaN`.
- Array input with string categoricals is supported (object output).
- `DataCorrupter` MCAR now targets the requested fraction of **observed** cells
  even when the target already contains NaNs.
- `DataCorrupter` `.xlsx`/`.arff` imports produce actionable errors when the
  optional extra is missing.
- Test-suite Unicode output no longer breaks cp1252 Windows consoles.
- README quick-start output and benchmark claims now match the implementation.
- Benchmark notebook column names and observed-data baselines are corrected.

### Documentation

- Added fitted-attribute documentation and explicit limitations.
- Documented single-imputation semantics of `sample_posterior`.
- Documented the time-series order/index, multi-entity, irregular-sampling, and
  temporary-fill caveats.
- Replaced relative PyPI links with absolute GitHub links.

## [0.1.4.post1] — 2026-05-22

Last release before the review fixes. See the repository history for details.
