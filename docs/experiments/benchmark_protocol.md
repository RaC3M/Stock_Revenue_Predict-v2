# Forecast Benchmark Protocol

> Status updated 2026-08-05. The current code contract is stricter than several output folders
> generated earlier on the same date. Treat `docs/experiments/experiment_registry.md` as the source
> of truth for evidence status.

## Purpose

`forecast_benchmark/` compares the two independent forecast systems under one evaluation contract:

- `ensemble_forecast/`
- `rolling_predict_LSTM/`

Model internals stay in their own folders. The benchmark owns normalization, the comparable cohort,
shared metrics, failure diagnostics, and downstream EPS/dividend/yield evaluation.

Both UIs obtain availability-safe downstream calculations from `financial_forecast/`; benchmark
method selection remains here rather than inside that shared transformation module.

## Current evaluation contract

- Target year: `2025` for the existing development evidence.
- Primary metric: `WMAPE`.
- Secondary metrics: `MAE`, `MAPE`, `MedianAPE`, `SMAPE`, `DirectionAccuracy`.
- Standard accuracy files use the exact intersection of `(stock_id, target_year, target_month)`
  across every explicitly requested `(source_family, model)` pair.
- Missing requested models, duplicate model-observation rows, or conflicting actual values fail the
  comparable-cohort build.
- `all_attempted_*` files are diagnostics and may contain incomplete model coverage; do not compare
  their rows as though they used the standard exact cohort.
- Target-year actual values are attached only after predictions and are used for evaluation, not for
  feature construction or prediction-time regime logic.
- Rolling normalization preserves `sequence_backbone` and `xlstm_backbone`; architecture-specific
  evidence must not be pooled or relabeled after export.
- The frozen financial ablation selects downstream methods on 2022–2024 actual-revenue replay, then
  evaluates existing 2025 predictions without revenue-model retraining. Each downstream stage uses
  the exact observation intersection across all requested methods.

## Reproducible commands

Development rerun with the historical mLSTM-only D1.16 Rolling artifact:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.run_benchmark --rolling-output-dir rolling_predict_LSTM\outputs\xlstm_main_flow_basket_100_d1_16 --output-dir forecast_benchmark\outputs\development_revenue_2025 --evidence-tier C --report-ready false --selection-protocol target-year-hindsight --registry-notes "D1.16 policy was developed against 2025"
```

Template for a future frozen-protocol run:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.run_benchmark --rolling-output-dir rolling_predict_LSTM\outputs\FROZEN_UNSEEN_YEAR_RUN --target-year UNSEEN_YEAR --output-dir forecast_benchmark\outputs\frozen_unseen_year --evidence-tier A --report-ready true --selection-protocol fixed-before-target --registry-notes "Protocol frozen before target scoring"
```

The Rolling input under `outputs/` is ignored by Git and is not available in a fresh clone. Generate
it first or obtain the artifact separately, then pass its directory explicitly.

## Recorded 2025 development result

The last human-reviewed exact-cohort comparison is:

```text
forecast_benchmark/outputs/data_migration_revenue_20260730
```

It compared 82 stocks after 17 Ensemble failures. Headline values were:

| Source | Model | WMAPE | MAPE | MedianAPE | DirectionAccuracy |
|---|---|---:|---:|---:|---:|
| Rolling | Rolling xLSTM (historical mLSTM-only) | 16.248% | 84.054% | 11.373% | 61.382% |
| Rolling | Rolling xLSTM + Conditional Adjustment (historical mLSTM-only) | 16.287% | 70.550% | 10.756% | 59.248% |
| Ensemble | XGBoost | 19.361% | 232.011% | 17.057% | 60.163% |
| Ensemble | ensemble_revenue | 19.390% | 182.551% | 16.174% | 59.451% |
| Ensemble | LightGBM | 19.505% | 208.377% | 17.124% | 59.350% |

This supports Rolling/historical mLSTM-only xLSTM as the main research direction, but not the claim
that Rolling wins every stock. D1.15 selected the balanced cap after inspecting 2025 and D1.16 reused
the same year, so the result is Tier C development evidence rather than an independent holdout. It
does not establish accuracy for the current `mLSTM → sLSTM` Hybrid UI architecture.

## Superseded 2026-07-31 output flag

`forecast_benchmark/outputs/report_ready_20260731_revenue` was produced from clean commit `18c316a`
and its historical metadata says `Tier A` / `report_ready=true`. It predates the stricter exact-pair
cohort and selection-protocol enforcement committed in `aa6b50d`; its standard table also contains
different stock coverage across requested models. Under the current policy it is a reproducibility
rerun, not final Tier A evidence.

No full cross-system rerun has yet been recorded after `aa6b50d` under both the current exact-cohort
contract and a newly unseen target year.

## Defensible interpretation

> Under the shared 2025 development benchmark, Rolling/historical mLSTM-only xLSTM reduces aggregate
> monthly revenue forecast error versus the Ensemble baseline on the comparable cohort, while
> stock-level winners remain mixed. Hybrid D1.21 separately passes its fixed-parameter within-run
> baselines, and D1.22 supplies an exact historical-backbone comparison with mixed plain-model
> results. Neither is an exact cross-system unseen-year comparison. Independent confirmation still
> requires a frozen-protocol rerun on an unseen year.
