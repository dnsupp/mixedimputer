# Benchmarks

**Version:** `0.2.0`

**Reproduce with:** `python benchmark/run_benchmarks.py`

**Runner:** [`benchmark/run_benchmarks.py`](benchmark/run_benchmarks.py)
**Interaction notebooks:** [`benchmark/benchmark.ipynb`](benchmark/benchmark.ipynb), [`benchmark/benchmark_timeseries.ipynb`](benchmark/benchmark_timeseries.ipynb)

These are single-seed results (`random_state=42`) on small datasets. They are
useful as regression checks and as a sanity check against naive baselines, not
as evidence of general superiority. The sample size for every column is shown.
Repeated-seed confidence intervals are a planned improvement.

## Datasets

### Titanic (MixedImputer)

- **Source:** `data/titanic.csv` — a public passenger-list dataset bundled for
  the examples. Add the original dataset citation/provenance when redistributing.
- **Size:** 891 passengers, 12 columns.
- **Dropped before imputation:** `Name`, `Ticket`, `Cabin`, `PassengerId`
  (high-cardinality / identifier columns).
- **Benchmark columns:** `Age`, `Fare` (numeric) and `Sex`, `Embarked`
  (categorical). The imputer is fitted on the full remaining DataFrame.

### GPU Sensor Log (TimeSeriesImputer)

- **Source:** `data/gpu_sensor_log.txt` — one real GPU sensor log recorded during
  a gaming session.
- **Size:** 356 rows at one-second intervals (~6 minutes).
- **Benchmark columns:** `GPU Temperature [C]` and `Fan Speed (RPM) [RPM]`.
- **Properties:** strong autocorrelation and sudden load changes.

## Methodology

- **Missingness:** MCAR via `DataCorrupter` at 1%, 10%, and 30% of **observed**
  cells per target column.
- **Imputation:** `MixedImputer(max_iter=10, random_state=42)`; for time series,
  `TimeSeriesImputer(lags=[1,2,3,5], rolling_windows=[10,30],
  initial_strategy="forward_fill", max_iter=10, random_state=42)`.
- **Evaluation:** ground-truth comparison only at corrupted positions.
- **Baselines:** observed-column mean for numeric, observed-column mode for
  categorical, and forward-fill (then backward-fill) for time series. The mean
  baseline now uses the corrupted/observed column mean, not the full original
  mean.
- **Random seed:** 42 for corruption and imputation.
- **Sample sizes:** reported per cell; interpret rows with fewer than ~30 cells
  cautiously.

## Results: MixedImputer

### Numeric RMSE (lower is better)

| Corruption | Age n | Age (MixedImputer) | Age (Mean) | Fare n | Fare (MixedImputer) | Fare (Mean) |
|:-----------|------:|-------------------:|-----------:|-------:|--------------------:|------------:|
| 1% | 7 | **11.58** | 13.10 | 9 | **21.48** | 27.01 |
| 10% | 71 | **12.93** | 13.64 | 89 | **31.10** | 48.68 |
| 30% | 214 | **13.22** | 14.36 | 267 | **31.63** | 38.04 |

### Categorical accuracy (higher is better)

| Corruption | Sex n | Sex (MixedImputer) | Sex (Mode) | Embarked n | Embarked (MixedImputer) | Embarked (Mode) |
|:-----------|------:|-------------------:|-----------:|-----------:|------------------------:|----------------:|
| 1% | 9 | **77.8%** | 55.6% | 9 | **100.0%** | 77.8% |
| 10% | 89 | **82.0%** | 65.2% | 89 | **78.7%** | 69.7% |
| 30% | 267 | **76.0%** | 63.7% | 267 | 75.7% | **79.8%** |

### What these numbers mean

- The imputer beats the observed-mean baseline for both numeric columns at all
  three corruption levels, but the margin is small at 1% (7–9 cells) and for Age
  at 30% (13.22 vs 14.36).
- Sex and Embarked improve at 10%, but Embarked is **worse than the mode
  baseline at 30%** (75.7% vs 79.8%). There is no universal categorical win.
- The 1% rows are dominated by sampling noise and should not be used for
  headline claims.

## Results: TimeSeriesImputer

### RMSE — GPU sensor columns (lower is better)

| Corruption | GPU Temp n | TS | Forward-fill | Mean | Fan n | TS | Forward-fill | Mean |
|:-----------|-----------:|----:|-------------:|-----:|------:|----:|-------------:|-----:|
| 1% | 4 | **0.48** | 0.50 | 3.28 | 4 | **24.32** | 31.78 | 188.21 |
| 10% | 36 | 1.06 | **0.94** | 4.05 | 36 | **18.38** | 27.98 | 289.10 |
| 30% | 107 | 1.13 | **1.13** | 4.59 | 107 | **21.75** | 33.76 | 284.03 |

### What these numbers mean

- The imputer substantially beats mean imputation for both GPU columns at
  10%/30% (roughly 4× for temperature and 13–16× for fan speed).
- Forward-fill is the strongest baseline for GPU Temperature; it beats the
  imputer at 10%/30% (0.94 vs 1.06, 1.13 vs 1.13). The earlier, much larger
  “12.8×” headline was based on four 1%-corrupted cells and is not repeated here.
- The imputer beats forward-fill for Fan Speed at 10%/30%, where the signal has
  more complex state changes.
- The rolling windows used for feature engineering are shifted by one row in
  `0.2.0`, so these numbers are **not** directly comparable to the older
  benchmark tables. The old tables had a train/serving mismatch in the rolling
  features.

## How to Reproduce

```bash
git clone https://github.com/dnsupp/mixedimputer.git
cd mixedimputer
pip install -e ".[dev]"
python benchmark/run_benchmarks.py
```

The runner prints the tables above. It deliberately avoids Jupyter/Plotly so it
can run in CI and on a headless machine.

The notebooks are for exploration:

```bash
jupyter notebook benchmark/benchmark.ipynb
jupyter notebook benchmark/benchmark_timeseries.ipynb
```

The notebooks are not a substitute for the runner: committed notebook outputs
go stale, and the runner is the canonical reproducible path.

## Limitations

- **Single seed.** The tables use one corruption mask per fraction. Repeating
  with multiple seeds and reporting mean ± 95% CI is the next step.
- **Small samples at 1%.** Seven to nine corrupted cells per column on Titanic;
  four per column on the GPU log. Do not read the 1% rows as precise estimates.
- **MCAR only.** MAR and MNAR mechanisms are implemented in `DataCorrupter` but
  are not benchmarked here.
- **One time-series dataset.** The GPU log is 356 seconds from one machine. It
  does not establish general time-series performance.
- **Baselines are simple.** Mean/mode/forward-fill are useful sanity checks, not
  state-of-the-art competitors. Kalman/state-space/interpolation baselines are
  not included.
- **No multiple imputation.** `sample_posterior=True` produces one stochastic
  completed dataset; the benchmark measures single imputation.
- **Data provenance.** The Titanic and German Credit files need their original
  source/license/attribution documented before redistribution outside this
  repository.
