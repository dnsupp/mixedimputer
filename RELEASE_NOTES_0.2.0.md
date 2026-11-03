# MixedImputer 0.2.0 — Release Notes

**Release date:** 2026-11-03  
**Package:** `mixed-imputer==0.2.0`  
**Build artifacts:**
- `dist/mixed_imputer-0.2.0-py3-none-any.whl`
- `dist/mixed_imputer-0.2.0.tar.gz`

## Highlights

`0.2.0` is a correctness and release-readiness release. It resolves the
release-blocking findings from the pre-0.2.0 review and makes the advertised
features match the implementation.

### MixedImputer

- Proper sklearn transformer behavior: fitted feature lists, no constructor
  parameter mutation, fixed output column order, schema validation, refitting.
- `n_features_in_`, `feature_names_in_`, `n_features_out_`, and
  `get_feature_names_out()`.
- `add_indicator=True` now emits real `missingindicator_<column>` boolean
  columns.
- Boolean columns are categorical/boolean, not continuous numeric targets.
- All-missing categorical columns no longer crash with the default
  `keep_empty_features=False`.
- Datetime/timedelta columns raise a clear `ValueError`; use
  `TimeSeriesImputer` for a time column.
- Array input no longer broadcasts dropped features into other columns; string
  categorical arrays are supported.
- Unseen categories are treated as missing and imputed where possible.

### TimeSeriesImputer

- The time column is metadata: excluded from imputation and re-inserted at its
  original position.
- Datetime and string time columns work.
- Original row order and index are restored.
- `lags` defaults to `(1, 2, 3)`; `None` disables lags.
- `initial_strategy` is implemented for all advertised strategies; linear and
  spline interpolation apply to numeric columns only.
- Rolling means are shifted by one row, removing current-target leakage.
- `n_features_in_`, `feature_names_in_`, and `get_feature_names_out()`.
- `is_time_series_suitable()` now flags irregular timestamps and duplicates.

### DataCorrupter

- Exported from the top-level package:
  `from mixedimputer import DataCorrupter`.
- No `pygrinder`/PyTorch dependency. MCAR, MAR, and MNAR are implemented in
  NumPy.
- The requested fraction applies to observed cells; existing NaNs are preserved.
- MAR depends on the other columns; MNAR high/low actually select high/low
  target values.
- Reproducible from `random_state`; no global RNG mutation.
- `Path`/`PathLike` inputs, validation, and optional `.arff`/`.xlsx`/`.xls`
  support via `mixed-imputer[io]`.

### Packaging and CI

- Version `0.2.0`; `Development Status :: 4 - Beta`.
- Dependencies: `numpy>=1.23`, `pandas>=1.5`, `scipy>=1.8`,
  `scikit-learn>=1.6,<1.9`.
- `[io]` and expanded `[dev]` extras.
- sdist includes `data/`, `tests/`, `benchmark/`, examples, changelog, and
  citation.
- CI covers Ubuntu + Windows, Python 3.10–3.13, a minimum-dependency job, and a
  build + `twine check` job.
- Headless benchmark runner: `python benchmark/run_benchmarks.py`.
- README/BENCHMARKS corrected: old “48–55%” and “12.8×” headlines were removed;
  the new tables report sample sizes and show where baselines win.

## Breaking changes

- Use `categorical_features_` / `numeric_features_` after fitting; the
  constructor parameters are no longer populated by `fit`.
- `MixedImputer.transform` returns columns in fitted order.
- Boolean columns are categorical/boolean.
- Datetime columns in `MixedImputer` raise; `TimeSeriesImputer` handles time.
- `TimeSeriesImputer` restores original row order/index instead of returning
  sorted data.
- `lags=None` disables lags; default is `(1, 2, 3)`.
- `DataCorrupter` MAR/MNAR results changed; `pygrinder` is no longer used.
- Minimum scikit-learn is `1.6`.

## Verification

- Full test suite: **196 passed** on Python 3.13.7, NumPy 2.4.4, pandas 3.0.2,
  scikit-learn 1.8.0.
- `twine check`: **PASSED** for both artifacts.
- Sdist contents verified: 3 data files, 6 test files, benchmark/examples, and
  changelog/citation.
- Benchmarks re-run with the new `DataCorrupter` fractions and shifted rolling
  features; README/BENCHMARKS reflect the new numbers.

## Artifact hashes

```text
mixed_imputer-0.2.0-py3-none-any.whl
SHA-256 260ab2c66da634cedadc51c5610a62fa4da4c014bdd6a351424165b9457ebad6

mixed_imputer-0.2.0.tar.gz
SHA-256 33fc58f554e9d89c31a158f7e88ab2df3421843ecf579c90ece4bcd36f0e738e
```

## Upload (manual, with your PyPI token)

From the repository root in PowerShell:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<your PyPI API token>"

.\.venv\Scripts\python.exe -m twine upload --non-interactive --repository pypi `
  dist\mixed_imputer-0.2.0-py3-none-any.whl `
  dist\mixed_imputer-0.2.0.tar.gz
```

Do **not** upload `dist/mixed_imputer-0.1.4.post1*`; those are old artifacts and
PyPI will reject them as already existing. The explicit file list above uploads
only `0.2.0`.

If you use a `.pypirc` file instead, omit the environment variables and let
twine read the configured repository.

## Post-upload verification

```powershell
.\.venv\Scripts\python.exe -m pip index versions mixed-imputer
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall mixed-imputer==0.2.0
.\.venv\Scripts\python.exe -c "import mixedimputer; print(mixedimputer.__version__)"
```

Expected: `0.2.0`, and the PyPI page should show the new README/BENCHMARKS.

## Known follow-ups (non-blocking)

- Repeated-seed confidence intervals for benchmark tables.
- Broader time-series validation (multiple datasets, Kalman/interpolation
  baselines).
- Add the thesis title/DOI to `CITATION.cff`.
- A full multiple-imputation API (`m` imputations + pooling).
