# Agent Guide

This file is for future AI/coding agents working on this repository. Read this before changing code.

## Project Goal

This repository is a Taiwan stock monthly revenue forecasting prototype.

There are three independent forecasting workflows, one neutral downstream financial module, and one isolated comparison layer:

1. Ensemble Forecast System
   - Folder: `ensemble_forecast/`
   - Files: `ensemble_forecast/app.py`, `ensemble_forecast/forecast_engine.py`
   - Purpose: multi-model revenue forecasting, model comparison, and a local adapter for the shared financial forecast module.

2. Rolling LSTM Forecast System
   - Folder: `rolling_predict_LSTM/`
   - Purpose: rolling sequence forecasting with KMeans clustering, conditional growth adjustment, dynamic regimes, GPU acceleration, isolated batch ablation experiments, and a local adapter for the shared financial forecast module.

3. Rolling SARIMA Forecast System
   - Folder: `sarima_forecast/`
   - Purpose: log-SARIMA monthly revenue forecasting, compact AIC model selection, and time-safe one-step rolling replay.

4. Shared Financial Forecast Module
   - Folder: `financial_forecast/`
   - Purpose: availability-safe financial evidence, EPS strategies, dividend strategies, complete-year enforcement, and the distinct deployable/evaluation yield calculations.

5. Forecast Benchmark Layer
   - Folder: `forecast_benchmark/`
   - Purpose: normalize both systems' outputs, build exact comparable cohorts, calculate shared metrics, record evidence metadata, and evaluate downstream EPS/dividend/yield methods.

The two forecasting systems are peers. They may read the same files under `data/` and use the neutral `financial_forecast/` package, but they must not import one another. The shared package must not contain revenue training, cross-system comparison, cohort construction, or method selection. Cross-system comparison belongs in `forecast_benchmark/` or another explicitly isolated analysis tool.

## Important Rule

When the user says "ensemble forecast", "multi-model forecast", "original ML version", or "machine-learning version", work only on:

- `ensemble_forecast/app.py`
- `ensemble_forecast/forecast_engine.py`
- related files under `ensemble_forecast/`
- root `data/` files only when the request explicitly changes source data

Do not touch `rolling_predict_LSTM/`.

When the user says "rolling LSTM", "Rolling LSTM + Cluster", "Trend + Cycle", "Growth Adjustment", "ablation", "quarterly target", or "GPU training", work only inside:

- `rolling_predict_LSTM/app.py`
- `rolling_predict_LSTM/rolling_lstm_engine.py`
- related files under `rolling_predict_LSTM/`

When the user says "SARIMA", "ARIMA", "traditional time series", or the Chinese equivalents, work only inside:

- `sarima_forecast/`
- related root documentation or validation tooling when needed

Do not import Ensemble or Rolling LSTM code into the SARIMA workflow.

When the user says "cross-system benchmark", "comparable cohort", "evidence tier", "report-ready", "EPS benchmark", "dividend layer", or "direct dividend model", work primarily inside:

- `forecast_benchmark/`
- related files under `docs/experiments/`
- root `data/` only when the request explicitly changes or audits source data

The benchmark may consume both systems through its adapters. Do not move benchmark logic back into either forecasting engine.

When changing EPS, cash-dividend, availability cutoff, or yield semantics shared by both UIs, put
the calculation in `financial_forecast/` and keep the system-specific files as thin adapters. Method
selection and ablation scoring stay in `forecast_benchmark/financial_ablation.py`.

## Data Files

Current shared source data files:

- `data/Stock_revenue_2019~2025.csv`
- `data/EPS2020~2025.csv`
- `data/Dividend2019~2025.csv`
- `data/day K2020~2025.csv`
- `data/stock_list_new.csv`
- `data/target_stocks_2025.csv`
- `data/manifest.json`

Notes:

- `data/` is the tracked canonical CSV output generated from ignored `free_taiwan_data/`
  through `data_preprocessing.free_taiwan_data_preprocessor`.
- Local `free_taiwan_data/` intentionally retains only the five raw datasets consumed by the
  current preprocessor: stock info, stock price, monthly revenue, financial statements, and dividends.
- Candidate and audit artifacts belong under ignored `data_preprocessing/outputs/`; do not mix
  generated outputs back into the raw source directory.
- Future data refreshes should regenerate `data/`, validate `data/manifest.json`, and run
  `data_preprocessing.preprocessing_audit` before committing.
- `Stock_revenue_2019~2025.csv` already has industry category merged from `stock_list_new.csv`.
- EPS, dividends, and stock prices are used by the shared downstream financial module, not as core revenue-training features.
- `outstanding_shares` should not be used as a training feature unless the user explicitly asks to re-test it.
- 2025 actual data is for evaluation/comparison only. Do not use target-month actual values as future information in model features.

## Ensemble Forecast System Summary

Files:

- `ensemble_forecast/forecast_engine.py`: data loading, feature engineering, model training, ensemble weighting, dividend-yield calculation.
- `ensemble_forecast/app.py`: Streamlit UI.

Main model path:

- XGBoost
- LightGBM
- CatBoost
- Seasonal/quantile-style fallback logic
- Weighted ensemble based on validation errors

Current behavior:

- Ensemble app calls `build_forecast(selected_stock)`.
- LSTM and TensorFlow are intentionally absent from this system; sequence forecasting belongs to Rolling LSTM.
- The forecast output includes revenue forecast, backtest metrics, model weights, recommendation notes, yield calculation, and actual-vs-predicted 2025 yield comparison.
- The formal Ensemble dividend path delegates to `financial_forecast/` through `ensemble_forecast/yield_forecast.py`. `current_system_payout_ratio` remains only in legacy benchmark code as a hindsight diagnostic.
- Ensemble EPS inputs must be available by the target year's January 10 cutoff and must form a complete four-quarter annual total.
- Missing payout or stock-price evidence must remain unavailable or use a documented historical fallback; never fabricate random financial inputs.

If editing this path, check:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m py_compile ensemble_forecast\forecast_engine.py ensemble_forecast\app.py
.\ensemble_forecast\.venv\Scripts\python.exe -m unittest discover -s ensemble_forecast\tests -v
```

## Rolling LSTM Forecast System Summary

Files:

- `rolling_predict_LSTM/app.py`
- `rolling_predict_LSTM/rolling_lstm_engine.py`
- `rolling_predict_LSTM/yield_forecast.py`
- `rolling_predict_LSTM/batch_ablation_study.py`
- `rolling_predict_LSTM/batch_quarterly_target_ablation.py`
- `rolling_predict_LSTM/batch_xlstm_main_flow_comparison.py`
- `rolling_predict_LSTM/batch_xlstm_confidence_calibration.py`
- `rolling_predict_LSTM/experiment_metrics.py`
- `rolling_predict_LSTM/experiment_metadata.py`

Main ideas:

- Convert revenue into monthly growth-direction signals.
- Build 12-month rolling windows.
- Use KMeans to cluster window patterns.
- Train Rolling LSTM with and without cluster one-hot features.
- Evaluate 2025 monthly predictions.
- Convert each complete 12-month revenue forecast into an availability-safe EPS and payout estimate, then evaluate dividend yield against target-year month-end prices.

Rolling yield rules:

- The Rolling yield module must not import Ensemble code.
- `rolling_predict_LSTM/yield_forecast.py` is a compatibility adapter over `financial_forecast/`; do not duplicate shared financial formulas there.
- EPS and payout selection use `statement_available_date` and `DividendAvailableDate` at or before `yield_as_of_date`.
- Target-year actual cash dividends and monthly closing prices are evaluation evidence only.
- Do not describe target-year price denominators as stock-price forecasts.
- Incomplete 12-month model outputs must be marked unavailable rather than silently annualized.

Current comparison models:

- `Rolling LSTM`
- `Rolling LSTM + Cluster`
- `Rolling LSTM + Cluster + Conditional Adjustment`
- optional `Rolling xLSTM`
- optional `Rolling xLSTM + Conditional Adjustment`

The xLSTM rows are no-cluster and architecture-explicit:

- `xlstm`: historical mLSTM-only D1 path.
- `xlstm_hybrid`: one mLSTM block followed by one native-backend sLSTM block; this is the current Streamlit default.

Do not relabel historical D1 evidence as Hybrid evidence. Do not confuse the native PyTorch sLSTM
backend with xlstm's optional custom CUDA kernel.

Important distinction:

- KMeans cluster IDs are unsupervised pattern IDs.
- They are not permanent stock-type labels.
- `regime` is computed per rolling window from past data only.

Current regime logic:

```text
growth_ratio > 0.65  -> growth
growth_ratio < 0.40  -> decline
otherwise            -> cycle
```

This means a stock can change regime month by month. Do not hard-code a stock code as growth/cycle/decline.

## Rolling LSTM Features Already Added

Growth stock handling:

- Growth Adjustment
- asymmetric loss
- conditional growth phase filter
- fixed positive latest-growth direction gate inside Growth Adjustment

Cyclical stock handling:

- Trend + Cycle separation
- `trend_component = rolling_mean_12m`
- `cycle_component = revenue - trend`
- volatility scaling
- normalized trend slope boost
- kept as a research path; not a current Streamlit main output

Removed from the main flow:

- dynamic guardrail
- direction filter toggle
- AutoTune UI/control path
- monthly Trend + Cycle Streamlit output

Batch research workflows:

- `experiment_metrics.py`: shared observation filtering, stock counts, metric records, and grouped summaries for standard Rolling batch runners.
- `batch_ablation_study.py`: fixed-parameter method/feature ablation.
- `batch_quarterly_target_ablation.py`: compares rolling monthly-sum forecasts against direct next-3-month revenue targets.
- `batch_sequence_backbone_ablation.py`: compares any requested pair of LSTM, historical mLSTM-only xLSTM, and Hybrid xLSTM while recording the selected backbone.
- `batch_xlstm_main_flow_comparison.py`: defaults to the current Hybrid UI architecture; pass `--xlstm-backbone xlstm` only when reproducing historical D1 mLSTM-only runs.
- Latest quarterly target finding as of 2026-07-31: direct 3M target did not beat monthly-sum for cycle-dominant stocks, so it is not a main-flow replacement.
- D1.15/D1.16 xLSTM defaults were developed with 2025 replay evidence. Predictions remain time-safe at inference, but the headline 2025 comparison is development evidence rather than an independent holdout.

Performance improvements:

- PyTorch-only backend with automatic CUDA/CPU selection
- larger GPU batch
- cached revenue windows
- cached KMeans artifacts
- cached training samples and LSTM arrays
- batch array conversion instead of repeated pandas transformations

## Rolling SARIMA Forecast System Summary

Files:

- `sarima_forecast/sarima_engine.py`
- `sarima_forecast/app.py`
- `sarima_forecast/start_app.py`

Main rules:

- Keep this workflow independent from Ensemble and Rolling LSTM; do not import either engine.
- Use `log1p` monthly revenue with a 12-month seasonal period.
- Select compact SARIMA orders only from history ending before the evaluation year.
- For target-year replay, each month may use only observations dated before that target month.
- Attach target-month actual revenue only after predictions are produced.
- Clearly label the prior-year seasonal fallback when history is short or no candidate converges.

Validation:

```powershell
.\sarima_forecast\.venv\Scripts\python.exe -m py_compile sarima_forecast\sarima_engine.py sarima_forecast\app.py sarima_forecast\start_app.py
.\sarima_forecast\.venv\Scripts\python.exe -m unittest sarima_forecast.tests.test_sarima_engine -v
```

If editing this path, check:

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m py_compile rolling_predict_LSTM\rolling_lstm_engine.py rolling_predict_LSTM\yield_forecast.py rolling_predict_LSTM\app.py rolling_predict_LSTM\batch_ablation_study.py rolling_predict_LSTM\batch_quarterly_target_ablation.py
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m unittest discover -s rolling_predict_LSTM\tests -v
```

For a quick smoke test:

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -c "from rolling_predict_LSTM.rolling_lstm_engine import GrowthAdjustmentConfig, RollingExperimentConfig, run_rolling_lstm_experiment; config=RollingExperimentConfig(k=4, epochs=5, max_train_samples=5000, growth=GrowthAdjustmentConfig(enabled=True)); r=run_rolling_lstm_experiment(1101, config=config); print(r.metrics.to_string(index=False))"
```

Use small `epochs` and small `max_train_samples` for smoke tests. Full runs can take longer.

## Forecast Benchmark Summary

Files:

- `forecast_benchmark/run_benchmark.py`: exact-cohort revenue comparison.
- `forecast_benchmark/metrics.py`: shared metrics.
- `forecast_benchmark/experiment_registry.py`: git/data/selection provenance.
- `forecast_benchmark/*_benchmark.py`: EPS, dividend, and yield layers.
- `forecast_benchmark/financial_ablation.py`: 2022–2024 downstream method selection followed by frozen target-year testing; it never retrains revenue models.
- `forecast_benchmark/tests/`: benchmark and evidence-contract tests.

Evidence rules:

- Do not infer evidence quality from an output folder name alone.
- `report_ready=true` must be paired with an eligible selection protocol and a human check of upstream target-year development.
- `current_system_payout_ratio` is legacy/hindsight only.
- Keep `all_attempted_*` diagnostics separate from the exact comparable cohort used for standard accuracy tables.
- Historical actual-revenue replay validates only EPS/dividend transformations, not upstream revenue forecasting accuracy.
- Generated `outputs/` stay ignored; tracked evidence status belongs in `docs/experiments/experiment_registry.md` and the relevant protocol/detail document.

If editing this path, check:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m compileall -q forecast_benchmark
.\ensemble_forecast\.venv\Scripts\python.exe -m unittest discover -s forecast_benchmark\tests -v
.\ensemble_forecast\.venv\Scripts\python.exe -m unittest discover -s financial_forecast\tests -v
```

## Common User Intent Mapping

If the user asks about:

- "dividend yield", "EPS", "dividend", "stock price", "actual yield vs predicted yield", or the Chinese equivalents:
  - Work in the forecasting system named by the user; if comparing both systems, use the Forecast Benchmark Layer.

- "Rolling LSTM", "KMeans", "cluster", "growth adjustment", "underestimate", "GPU", "Trend + Cycle", "ablation", "quarterly target":
  - Rolling LSTM Forecast System.

- "SARIMA", "ARIMA", "traditional time series", or the Chinese equivalents:
  - Rolling SARIMA Forecast System.

- "benchmark", "compare both systems", "EPS layer", "dividend layer", "direct dividend", "evidence tier", "report-ready":
  - Forecast Benchmark Layer.

- "do not touch rolling" or the Chinese equivalent:
  - absolutely avoid `rolling_predict_LSTM/`.

- "do not touch the ensemble system" or the Chinese equivalent:
  - implement inside `rolling_predict_LSTM/` or a new isolated folder.

## Coding Guidelines

- Keep changes scoped.
- Never import one forecasting system from the other.
- Keep `financial_forecast/` neutral: financial transformations only, with one public orchestration interface.
- Keep cross-system normalization and downstream comparison in `forecast_benchmark/` adapters and runners.
- Do not refactor unrelated sections.
- Preserve existing CSV formats.
- Prefer adding optional parameters for experimental logic.
- Keep experimental mechanisms switchable from Streamlit.
- Do not use future actual results to create training labels or regime labels.
- Do not permanently label a stock as growth/cycle/decline by stock code.
- Use `apply_patch` for manual edits.
- Avoid destructive git commands.

## UI Guidelines

The Streamlit UI is functional and research-oriented. When adding results:

- Add metrics only when they help compare model behavior.
- Add dataframe columns that explain why a post-processing step was applied.
- For rolling LSTM, expose switches/sliders for experimental mechanisms.
- Keep labels clear enough to compare:
  - base prediction
  - cluster prediction
  - adjusted prediction

## Validation Checklist

Before finishing a code change:

1. Syntax check the touched app.
2. Run a small smoke test if model logic changed.
3. Confirm the correct workflow was touched.
4. Mention if a full training run was not performed.
5. If charts are affected, confirm the plotted columns exist.

For a full repository validation, run:

```powershell
python tools\validate_project.py
```

The runner compiles all project Python sources, validates the tracked canonical `data/` and manifest,
runs `pip check` for both owned virtualenvs, selects the owning virtualenv for every test suite, and
promotes `FutureWarning` to a failure. Use repeated `--suite` options only for scoped iteration;
cross-cutting changes should run the full set.

Useful stocks for quick checks:

- `1101`: long-cycle / cyclicality behavior.
- `1231`: cycle-like case where adjustment may overlap with cluster prediction.
- `3017`: high-growth / underestimation behavior.

## Known Limitations

- This is a research prototype, not an investment system.
- Model outputs are sensitive to parameters and data revisions.
- Rolling LSTM can be slow on cold cache.
- Growth Adjustment is experimental and should always be compared against base models.
- Trend + Cycle and quarterly target variants are research workflows, not current main-flow replacements.
- KMeans clusters are not semantic labels by themselves.
- Historical mLSTM-only xLSTM headline results reuse the 2025 development year and need a frozen-protocol unseen-year rerun before Tier A promotion.
- Hybrid has registered basket-100 results in D1.21 and an exact-cohort historical-backbone comparison in D1.22. Both remain Tier C because they evaluate the 2025 development year; an unseen-year rerun is still missing.
- Full raw experiment outputs are intentionally ignored and are not available in a fresh clone unless regenerated or shared separately.

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five mattpocock/skills triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context domain docs: root `CONTEXT.md` plus `docs/adr/`. See `docs/agents/domain.md`.
