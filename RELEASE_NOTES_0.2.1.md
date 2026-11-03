# MixedImputer 0.2.1 — Release Notes

**Release date:** 2026-11-03  
**Package:** `mixed-imputer==0.2.1`  
**Build artifacts:**
- `dist/mixed_imputer-0.2.1-py3-none-any.whl`
- `dist/mixed_imputer-0.2.1.tar.gz`

## What changed

`0.2.1` is a benchmark and documentation release. There are **no runtime-library
API or behavior changes** from `0.2.0`.

### Added

- `benchmark/run_timeseries_validation.py`, a Phase 1 validation harness with:
  - synthetic white noise, AR(1), trend+seasonality, random walk, regime shift,
    and multivariate correlated datasets
  - the bundled GPU sensor log
  - point MCAR at 10%/30% and contiguous block gaps at 10%
  - ffill/bfill, linear interpolation, observed mean, and one-shot HGB-on-lags
    baselines
  - `TimeSeriesImputer` ablations with no temporal features, lags only, and
    lags + rolling features
  - multi-seed mean RMSE/MAE, 95% CIs, win rates, and paired Wilcoxon tests
- `--quick` smoke mode and configurable `--estimator fast|default` mode.
- Unit tests for mask generation and metric evaluation.
- Documented Phase 1 findings and limitations.

### Preliminary validation finding

A 10-seed fast-estimator run across seven datasets and three missingness
patterns found linear interpolation best in 14/21 dataset/pattern cells.
`ts_full` was never the single best method and averaged rank 2.9. This is
preliminary evidence, not a claim of general time-series superiority.

The fast mode uses `BayesianRidge`/`LogisticRegression`; `--estimator default`
tests the package's HistGradientBoosting defaults but is much slower.

## Verification

- Full test suite: **200 passed**.
- `twine check`: **PASSED** for both artifacts.
- Headline Titanic/GPU metrics reproduce exactly on a quiet machine.
- Phase 1 ranks and best-method counts reproduce exactly; only runtime changed.

## Artifact hashes

```text
mixed_imputer-0.2.1-py3-none-any.whl
SHA-256 24fb0d376204c4d125c975144c6ddc651d6baf4307f04629d38f96dc24878eb3

mixed_imputer-0.2.1.tar.gz
SHA-256 c3026e438b58a437a4944faca7d7a13e865f123c4aaf5e438d1237a5da76ed49
```

## Upload (manual, with your PyPI token)

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<your PyPI API token>"

.\.venv\Scripts\python.exe -m twine upload --non-interactive --repository pypi `
  dist\mixed_imputer-0.2.1-py3-none-any.whl `
  dist\mixed_imputer-0.2.1.tar.gz
```

Do **not** upload the older `0.1.4.post1` or `0.2.0` artifacts in `dist/`.

## Post-upload verification

```powershell
.\.venv\Scripts\python.exe -m pip index versions mixed-imputer
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall mixed-imputer==0.2.1
.\.venv\Scripts\python.exe -c "import mixedimputer; print(mixedimputer.__version__)"
```

Expected: `0.2.1`.

## Remaining follow-ups

- Phase 2: 2–3 clearly licensed public time-series datasets.
- Phase 3: Kalman/state-space baselines with a fair tuning protocol.
- Phase 4: panel/group support and probabilistic/multiple-imputation metrics if
  the validation evidence requires them.
