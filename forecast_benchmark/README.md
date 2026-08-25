# Forecast Benchmark

Independent comparison layer for `ensemble_forecast/` and `rolling_predict_LSTM/`.

This folder owns only benchmark protocol, adapters, shared metrics, and benchmark outputs. Model
logic stays in its original system folder.

It is not a third forecast model. It consumes Ensemble and Rolling evidence through adapters and
keeps cross-system normalization out of both model engines.

## Before running

- Create the Ensemble Python environment first; benchmark runners use that environment.
- Ensure root `data/` and the Git LFS price file are present.
- Revenue comparison needs an existing Rolling output directory containing `monthly_predictions.csv`.
- All `outputs/` directories are ignored by Git. A fresh clone does not contain the local D1.16 or
  report-ready artifacts referenced by historical commands.
- Prefer passing `--rolling-output-dir` and `--input-predictions` explicitly instead of relying on
  a machine-local default artifact.

Current conclusions and human-reviewed evidence status are tracked in:

- `docs/experiments/benchmark_protocol.md`
- `docs/experiments/experiment_registry.md`

## Revenue benchmark

Small smoke run, using an existing Rolling output:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.run_benchmark --rolling-output-dir rolling_predict_LSTM\outputs\xlstm_main_flow_basket_100_d1_16 --stock-limit 3 --output-dir forecast_benchmark\outputs\smoke_3 --report-ready false
```

Historical 2025 development rerun:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.run_benchmark --output-dir forecast_benchmark\outputs\data_migration_revenue_20260730 --evidence-tier C --report-ready false --selection-protocol target-year-hindsight --registry-notes "Rolling D1.16 source was developed against 2025"
```

The code default Rolling source is:

```text
rolling_predict_LSTM/outputs/xlstm_main_flow_basket_100_d1_16/monthly_predictions.csv
```

That source is intentionally ignored and was developed with 2025 replay evidence. The command can
reproduce the development comparison when the artifact exists, but it is not a new independent
holdout. Freeze model/default selection before target scoring to produce new Tier A evidence.

Generated files:

- `monthly_predictions.csv`
- `comparable_monthly_predictions.csv`
- `overall_accuracy.csv`
- `stock_accuracy.csv`
- `winner_summary.csv`
- `all_attempted_overall_accuracy.csv`
- `all_attempted_stock_accuracy.csv`
- `all_attempted_winner_summary.csv`
- `failed_runs.csv`
- `run_config.json`
- `experiment_registry_entry.json`

The standard accuracy files use the exact common observation cohort: the intersection of
`(stock_id, target_year, target_month)` across every requested source/model pair. The
`all_attempted_*` files are diagnostics and include incomplete observations.

## Experiment Registry

Revenue benchmark and downstream benchmark runs write registry metadata into both
`run_config.json` and `experiment_registry_entry.json`. The registry entry records:

- `experiment_id`
- `evidence_tier`
- `report_ready`
- git commit and dirty-worktree status
- source manifest plus actual per-CSV and combined dataset hashes
- selection protocol and report-ready eligibility
- command arguments and output directory

Useful flags:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.run_benchmark --output-dir forecast_benchmark\outputs\future_frozen_run --evidence-tier A --report-ready true --selection-protocol fixed-before-target --registry-notes "Protocol frozen before target scoring"
```

Only `fixed-before-target` and `historical-validation` are eligible for `--report-ready true` in the
current registry contract; hindsight or unspecified runs are downgraded. Older output metadata may
predate this enforcement, so a historical `report_ready=true` field is not sufficient on its own.
Output directories remain ignored by Git; summarize human-reviewed runs in
`docs/experiments/experiment_registry.md` when they are cited.
The same registry flags are available on downstream runners such as `yield_benchmark`,
`eps_benchmark`, `dividend_layer_benchmark`, and direct dividend diagnostics.

## Frozen Financial Ablation

Use the shared downstream module to select EPS and cash-dividend methods on 2022–2024, then apply
the winner to existing frozen 2025 revenue predictions:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.financial_ablation --input-predictions forecast_benchmark\outputs\data_migration_revenue_20260730\comparable_monthly_predictions.csv --output-dir forecast_benchmark\outputs\financial_ablation
```

This runner does not import either revenue engine and does not retrain LSTM, xLSTM, or Ensemble
models. It first compares EPS methods, then announcement-safe dividend methods, and finally ranks
end-to-end combinations by cash-dividend MAE followed by yield MAE. The winning combination is
frozen before target-year scoring. Each stage uses the exact intersection of observations with
valid evidence for every requested method, so methods are not rewarded for dropping hard cases.

Historical validation uses observed monthly revenue as `actual_revenue_replay`. It isolates the
downstream financial transformation and is not validation evidence for the upstream revenue model.
`as_of_price_yield` is a deployable cutoff estimate; `target_month_end_yield` uses observed
target-year prices and are evaluation-only. Outputs default to Tier B and non-report-ready until a
human confirms the frozen upstream model was not tuned on the target year.

## Yield Benchmark

After the monthly revenue benchmark is available, run the downstream EPS/dividend-yield benchmark:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.yield_benchmark --output-dir forecast_benchmark\outputs\yield_benchmark
```

Default input:

```text
forecast_benchmark/outputs/data_migration_revenue_20260730/comparable_monthly_predictions.csv
```

Default compared models:

- `Rolling xLSTM`
- `Rolling xLSTM + Conditional Adjustment`
- `ensemble_revenue`
- `LightGBM`

Generated files:

- `yield_predictions.csv`
- `yield_overall_accuracy.csv`
- `yield_stock_accuracy.csv`
- `yield_winner_summary.csv`
- `yield_error_decomposition.csv`
- `yield_failed_runs.csv`
- `run_config.json`

Monthly stock prices at or below `1.0` are treated as invalid for yield-error metrics by default.
They remain in `yield_predictions.csv`, but predicted/actual yield fields are set to blank for those
months.

## Yield EPS-Layer Benchmark

After the EPS-layer validation is available, test whether alternative EPS transforms improve
downstream cash dividend and yield estimates:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.yield_eps_layer_benchmark --output-dir forecast_benchmark\outputs\yield_eps_layer_benchmark
```

Default EPS methods:

- `current_ratio`
- `lasso_annual`
- `elastic_net_annual`

Generated files:

- `yield_eps_layer_predictions.csv`
- `yield_eps_layer_stock_accuracy.csv`
- `yield_eps_layer_overall_accuracy.csv`
- `yield_eps_layer_winner_summary.csv`
- `yield_eps_layer_improvement_vs_current.csv`
- `yield_eps_layer_error_decomposition.csv`
- `yield_eps_layer_failed_runs.csv`
- `run_config.json`

This benchmark uses the same payout-ratio and stock-price logic as the standard yield benchmark,
but swaps the EPS transform. It answers whether EPS-layer gains actually transmit to dividend-yield
accuracy.

## Dividend Layer Benchmark

After the yield EPS-layer benchmark is available, isolate the dividend layer itself:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.dividend_layer_benchmark --output-dir forecast_benchmark\outputs\dividend_layer_benchmark
```

Default EPS methods:

- `current_ratio`
- `elastic_net_annual`

Default dividend methods:

- `current_system_payout_ratio`: legacy hindsight diagnostic that reproduces the earlier
  fiscal-year payout-ratio cutoff. The current Ensemble Forecast System uses a time-safe historical
  payout path instead.
- `time_safe_payout_ratio`: excludes dividends whose ex-dividend year is the target year or later.
- `announcement_safe_payout_ratio`: excludes dividends whose announcement/availability date is after
  `--as-of-date`.
- `announcement_safe_last_cash_dividend`: last known cash dividend by announcement/availability date.
- `last_cash_dividend`
- `recent_cash_dividend_median`
- `smoothed_cash_dividend`
- `eps_sign_guard_last_cash_dividend`

To run with a stricter dividend availability date:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.dividend_layer_benchmark --dividend-methods current_system_payout_ratio,time_safe_payout_ratio,announcement_safe_payout_ratio,announcement_safe_last_cash_dividend --as-of-date 2025-01-10 --output-dir forecast_benchmark\outputs\announcement_safe_dividend_benchmark
```

Generated files:

- `dividend_layer_predictions.csv`
- `dividend_layer_stock_accuracy.csv`
- `dividend_layer_overall_accuracy.csv`
- `dividend_layer_winner_summary.csv`
- `dividend_layer_improvement_vs_baseline.csv`
- `dividend_layer_leakage_diagnostic.csv`
- `dividend_layer_failed_runs.csv`
- `run_config.json`

This benchmark is also a leakage audit. The legacy `current_system_payout_ratio` diagnostic can
include fiscal-year dividends whose ex-dividend date falls in the target year, and can also include
dividends announced after the prediction as-of date. The current Ensemble Forecast System no longer
uses that leaky path. `announcement_safe_*` methods remain the stricter deployable comparison when
announcement dates are available.

## Direct Dividend Model Benchmark

After the announcement-safe dividend baselines are available, test whether a direct cash-dividend
model can improve the downstream yield layer:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.direct_dividend_model_benchmark --output-dir forecast_benchmark\outputs\direct_dividend_model_benchmark
```

This benchmark uses a hurdle protocol:

- First classify whether the stock pays a target-year cash dividend.
- Then estimate cash-dividend amount only for predicted payers.
- Select direct methods on 2022/2023/2024 validation folds using
  `average_cash_dividend_abs_error`.
- Sweep default classification thresholds `0.25,0.30,0.35,0.40,0.45,0.50,0.60` for
  Ridge and ElasticNet hurdle models.
- Apply both the global selected method and the dividend-history bucket selected strategy to 2025
  test.
- Require bucket support before using a bucket-specific method. Defaults are
  `--min-bucket-folds 2` and `--min-bucket-stock-years 15`; unsupported buckets fall back to the
  global selected method.
- Compare against `announcement_safe_payout_ratio` and `announcement_safe_last_cash_dividend`.

Direct model features are filtered by `available_date <= as_of_date`. The default validation cutoffs
are `2022-01-10`, `2023-01-10`, and `2024-01-10`; the default test cutoff is `2025-01-10`.

Useful overrides:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.direct_dividend_model_benchmark --validation-years 2023,2024 --threshold-values 0.35,0.45,0.60 --min-bucket-folds 2 --min-bucket-stock-years 15 --output-dir forecast_benchmark\outputs\direct_dividend_model_benchmark
```

Generated files:

- `direct_dividend_validation_predictions.csv`
- `direct_dividend_validation_stock_accuracy.csv`
- `direct_dividend_validation_overall_accuracy.csv`
- `direct_dividend_validation_method_scores.csv`
- `direct_dividend_bucket_method_scores.csv`
- `direct_dividend_bucket_method_selection.csv`
- `direct_dividend_method_selection.csv`
- `direct_dividend_test_predictions.csv`
- `direct_dividend_test_stock_accuracy.csv`
- `direct_dividend_test_overall_accuracy.csv`
- `direct_dividend_test_winner_summary.csv`
- `direct_dividend_selected_test_predictions.csv`
- `direct_dividend_selected_test_stock_accuracy.csv`
- `direct_dividend_selected_test_overall_accuracy.csv`
- `direct_dividend_baseline_predictions.csv`
- `direct_dividend_baseline_stock_accuracy.csv`
- `direct_dividend_baseline_overall_accuracy.csv`
- `direct_dividend_comparison_vs_baselines.csv`
- `direct_dividend_failed_runs.csv`
- `run_config.json`

## Direct Dividend Error Diagnostics

After the direct dividend benchmark is available, diagnose where the selected direct strategy wins
or loses against the announcement-safe baseline:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.direct_dividend_error_diagnostics --direct-benchmark-dir forecast_benchmark\outputs\direct_dividend_model_benchmark --output-dir forecast_benchmark\outputs\direct_dividend_error_diagnostics
```

This diagnostic compares:

- `bucket_validation_best`
- the globally selected direct method, such as `direct_hurdle_ridge_t060`
- `LightGBM + current_ratio + announcement_safe_payout_ratio`

Generated files:

- `direct_dividend_stock_error_comparison.csv`
- `direct_dividend_improvement_leaders.csv`
- `direct_dividend_regression_hotspots.csv`
- `direct_dividend_bucket_error_summary.csv`
- `direct_dividend_classification_outcomes.csv`
- `direct_dividend_classification_summary.csv`
- `direct_dividend_classification_errors.csv`
- `direct_dividend_amount_error_hotspots.csv`
- `direct_dividend_diagnostic_summary.csv`
- `run_config.json`

## EPS Benchmark

To isolate whether downstream error comes from revenue forecasts or from the EPS conversion formula:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.eps_benchmark --output-dir forecast_benchmark\outputs\eps_benchmark
```

Default input:

```text
forecast_benchmark/outputs/data_migration_revenue_20260730/comparable_monthly_predictions.csv
```

Default compared revenue models:

- `Rolling xLSTM`
- `Rolling xLSTM + Conditional Adjustment`
- `ensemble_revenue`
- `LightGBM`

Default EPS methods:

- `current_ratio`: the current ensemble-style annual revenue x recent EPS/revenue median baseline.
- `seasonal_quarter_median`: quarterly revenue x same-quarter historical EPS/revenue medians.
- `ridge_annual`: a global Ridge model trained on historical annual revenue/EPS features.
- `lasso_annual`: a global Lasso model trained on historical annual revenue/EPS features.
- `elastic_net_annual`: a global ElasticNet model trained on historical annual revenue/EPS features.

Generated files:

- `eps_predictions.csv`
- `eps_overall_accuracy.csv`
- `eps_stock_accuracy.csv`
- `eps_method_winner_summary.csv`
- `eps_error_decomposition.csv`
- `eps_failed_runs.csv`
- `run_config.json`

The benchmark also writes `oracle_current_ratio` diagnostic rows by default. These use actual 2025
annual revenue with the current EPS ratio formula, so they are excluded from the winner summary and
should only be used to judge how much error remains after revenue forecasting is perfect.

EPS percent-error metrics exclude stocks where absolute actual EPS is below `0.01` by default,
because percentage error is not meaningful when the denominator is near zero. EPS absolute error is
still reported for those stocks.

## EPS Diagnostics

After the EPS benchmark is available, diagnose which stocks can safely use an EPS/revenue ratio and
which need a different EPS layer:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.eps_diagnostics --output-dir forecast_benchmark\outputs\eps_diagnostics
```

Default input:

```text
forecast_benchmark/outputs/eps_benchmark/eps_stock_accuracy.csv
```

Generated files:

- `eps_ratio_stability.csv`
- `eps_method_recommendations.csv`
- `eps_error_hotspots.csv`
- `eps_current_ratio_driver_by_model.csv`
- `eps_diagnostic_summary.csv`
- `run_config.json`

Recommendations in this diagnostic use 2025 actual EPS, so they are hindsight research labels. Use
them to design the next time-safe validation experiment, not as a production selection rule.

## EPS Layer Validation

Convert hindsight EPS diagnostics into a time-safe validation experiment:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.eps_layer_validation --output-dir forecast_benchmark\outputs\eps_layer_validation
```

The default protocol uses actual 2024 revenue as the EPS-layer validation input, selects EPS methods
from 2024 validation, and applies those fixed rules to 2025 revenue benchmark predictions.

Generated files:

- `validation_eps_stock_accuracy.csv`
- `validation_ratio_stability.csv`
- `validation_bucket_method_scores.csv`
- `validation_bucket_method_selection.csv`
- `validation_stock_method_selection.csv`
- `test_all_method_stock_accuracy.csv`
- `test_ratio_stability.csv`
- `test_selected_stock_accuracy.csv`
- `test_strategy_overall_accuracy.csv`
- `test_strategy_winner_summary.csv`
- `test_strategy_improvement_vs_current.csv`
- `run_config.json`

`oracle_2025_best_method` is included as a hindsight upper bound only. It is excluded from stock-win
summaries and should not be presented as a deployable strategy.

## Tests

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m compileall -q forecast_benchmark
.\ensemble_forecast\.venv\Scripts\python.exe -m unittest discover -s forecast_benchmark\tests -v
```

## Ownership

This layer owns:

- adapters that normalize Ensemble and Rolling outputs;
- exact comparable-cohort construction and shared metrics;
- EPS, dividend, yield, and diagnostic runners;
- experiment registry metadata and local benchmark outputs.

It does not own model training logic. Changes to model behavior stay in the corresponding forecast
system, and generated `forecast_benchmark/outputs/` remain ignored by Git.
