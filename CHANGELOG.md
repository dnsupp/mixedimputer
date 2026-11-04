# Changelog

All notable changes to this project are documented here.

## [0.2.2] — 2026-11-04

This is a distribution-name release. The import package remains
`mixedimputer`; there are no runtime code or API changes from `0.2.1`.

### Changed

- PyPI distribution renamed from `mixed-imputer` to `mixedimputer` (no hyphen),
  matching the import package and the renamed GitHub repository.
- Wheel/sdist filenames and project metadata now use `mixedimputer`.
- Installation docs, badges, and optional-extra error messages updated.

Existing `mixed-imputer` releases remain available under the old name for
reproducibility; `0.2.2` is the start of the non-hyphenated distribution line.

## [0.2.1] — 2026-11-03

This is a benchmark and documentation release; there are no runtime-library API
or behavior changes from `0.2.0`.

### Added

- Phase 1 time-series validation harness
  (`benchmark/run_timeseries_validation.py`): synthetic datasets, point and
  block missingness, interpolation/mean/HGB-lag baselines, temporal-feature
  ablations, multi-seed confidence intervals, and paired Wilcoxon comparisons.
- Fast `BayesianRidge`/`LogisticRegression` estimator mode for practical
  many-seed validation; `--estimator default` retains the package's
  HistGradientBoosting defaults.
- Harness unit tests and documented Phase 1 findings; the 10-seed preliminary
  run found linear interpolation best in 14/21 dataset/pattern cells, so no
  general time-series superiority is claimed.

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
