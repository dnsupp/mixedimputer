# Benchmarks

## Datasets

### Titanic (MixedImputer)
- **Source:** `data/titanic.csv` — public domain passenger list
- **Size:** 891 passengers, 12 columns (7 numeric, 5 categorical/string)
- **High-cardinality columns dropped:** `Name`, `Ticket`, `Cabin`, `PassengerId`
- **Benchmark columns:** `Age`, `Fare` (numeric) + `Sex`, `Embarked` (categorical)

### GPU Sensor Log (TimeSeriesImputer)
- **Source:** `data/gpu_sensor_log.txt` — real-world GPU sensor readings logged during a gaming session
- **Size:** 356 rows (1-second intervals, ~6 minutes), 14 sensor columns
- **Benchmark columns:** `GPU Temperature [°C]`, `Fan Speed (RPM) [RPM]`
- **Properties:** Strong temporal autocorrelation, sudden state changes (gaming load on/off)

## Methodology

- **Missingness mechanism:** MCAR (Missing Completely At Random) via `pygrinder`
- **Corruption levels:** 1%, 10%, 30% of non-missing cells per target column
- **Evaluation:** Ground-truth comparison at corrupted positions only
- **Metrics:**
  - Numeric columns: RMSE (Root Mean Square Error), MAE (Mean Absolute Error)
  - Categorical columns: Accuracy, F1-score (weighted)
- **Random seed:** Fixed at 42 for reproducibility

## Results: MixedImputer

**Notebook:** [`benchmark/benchmark.ipynb`](benchmark/benchmark.ipynb)  
**Baselines:** Mean imputation (numeric), Mode imputation (categorical)

### RMSE — Numeric Columns (lower is better)

| Corruption | Age (MixedImputer) | Age (Mean baseline) | Fare (MixedImputer) | Fare (Mean baseline) |
|:-----------|:------------------:|:-------------------:|:-------------------:|:--------------------:|
| 1% | **4.98** | 9.50 | **10.12** | 22.56 |
| 10% | **10.22** | 14.22 | **48.17** | 70.25 |
| 30% | **12.89** | 14.13 | **33.51** | 49.75 |

### Accuracy — Categorical Columns (higher is better)

| Corruption | Sex (MixedImputer) | Sex (Mode baseline) | Embarked (MixedImputer) | Embarked (Mode baseline) |
|:-----------|:------------------:|:-------------------:|:----------------------:|:------------------------:|
| 1% | **90.0%** | 70.0% | **100.0%** | 72.7% |
| 10% | **74.7%** | 69.5% | **83.5%** | 73.2% |
| 30% | **75.0%** | 63.6% | **75.6%** | 73.1% |

### Key Findings

- **MixedImputer consistently beats naive baselines** across all corruption levels and column types.
- At 1% corruption, MixedImputer reduces numeric RMSE by **48–55%** compared to mean imputation.
- Categorical imputation achieves **100% accuracy** on `Embarked` at 1% corruption and remains robust through 30%.
- The `Fare` column shows non-monotonic RMSE (48.17 at 10% vs 33.51 at 30%) — this is a **data-distribution effect** caused by the £512.33 outlier, not an imputer artifact. The baseline follows the same pattern.
- **Zero preprocessing required** — MixedImputer handles the full DataFrame (numeric + string columns) in a single `fit_transform` call.

## Results: TimeSeriesImputer

> **Status: Experimental** — `TimeSeriesImputer` has verified functional correctness (22 unit tests pass) but has not yet been benchmarked against established time-series imputation methods (e.g., Kalman filters, `pandas.interpolate`).

**Notebook:** [`benchmark/benchmark_timeseries.ipynb`](benchmark/benchmark_timeseries.ipynb)  
**Baselines:** Forward-fill (time-aware), Mean imputation (naive)

### RMSE — GPU Sensor Columns (lower is better)

| Corruption | GPU Temperature (TS) | GPU Temperature (FF) | GPU Temperature (Mean) | Fan Speed (TS) | Fan Speed (FF) | Fan Speed (Mean) |
|:-----------|:--------------------:|:--------------------:|:----------------------:|:--------------:|:--------------:|:----------------:|
| 1% | **0.49** | 0.87 | 6.26 | —* | —* | —* |
| 10% | **1.20** | 1.33 | 5.83 | 213.20 | **182.09** | 378.89 |
| 30% | **1.21** | 1.32 | 5.42 | 118.41 | **114.85** | 324.05 |

\* At 1% corruption, only 4 cells were selected — all fell in GPU Temperature by MCAR chance.

### Key Findings

- **TimeSeriesImputer dramatically outperforms mean imputation** — up to **12.8× lower RMSE** for GPU Temperature at 1%.
- **Forward-fill is a strong baseline** for highly autocorrelated data (Fan Speed): it slightly edges out TimeSeriesImputer. This is expected — when a signal changes slowly, the previous value is often the best guess.
- **TimeSeriesImputer wins on GPU Temperature**, where the signal has more complex patterns that benefit from the MICE-style iterative refinement.
- The temporal features (lags, rolling means) allow the imputer to exploit autocorrelation, unlike the mean baseline which ignores time entirely.

## How to Reproduce

1. Install the package in development mode with dependencies:

   ```bash
   git clone https://github.com/dnsupp/mixed-imputer.git
   cd mixed-imputer
   pip install -e ".[dev]"
   ```

2. Run the benchmark notebooks:

   ```bash
   jupyter notebook benchmark/benchmark.ipynb
   jupyter notebook benchmark/benchmark_timeseries.ipynb
   ```

   Execute all cells (Cell → Run All). Results are deterministic with the fixed random seeds.

## Notes

- **Reproducibility:** All benchmarks use fixed random seeds (`random_state=42` for corruption and imputation). Re-running the notebooks will produce identical results.
- **MixedImputer** is benchmarked on the Titanic dataset (891 rows) — a small but well-known public dataset with mixed column types.
- **TimeSeriesImputer** is benchmarked on a single 356-second GPU sensor log from one machine. Results may vary with different time-series characteristics (sampling rate, signal complexity, autocorrelation strength).
- **TimeSeriesImputer is experimental** — while functional correctness is verified by 22 unit tests, the statistical accuracy has not been quantified on diverse time-series datasets. Validate results against domain knowledge before production use.
- **MCAR only:** Both notebooks use MCAR for corruption. MAR and MNAR mechanisms are supported by `DataCorrupter` but were not benchmarked here.
- The MixedImputer benchmark excludes high-cardinality columns (`Name`, `Ticket`, `Cabin`, `PassengerId`) which degrade iterative imputation performance.
