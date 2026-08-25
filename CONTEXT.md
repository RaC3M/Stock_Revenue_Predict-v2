# Taiwan Stock Forecasting

This project contains two independent revenue forecasting systems, a neutral downstream financial transformation module, and an isolated benchmark layer that compares their outputs.

## Language

**Ensemble Forecast System**:
The independent forecasting system that compares multiple non-LSTM revenue models, combines them using historical validation, and estimates dividend yield.
_Avoid_: Main ML, main app, original version

**Rolling LSTM Forecast System**:
The independent forecasting system that predicts monthly revenue from rolling sequences, KMeans pattern clusters, dynamic regimes, and optional post-processing strategies, then estimates annual EPS, cash dividend, and dividend yield through its own downstream module.
_Avoid_: LSTM add-on, secondary system

**Forecast Benchmark Layer**:
The isolated `forecast_benchmark/` package that normalizes predictions, builds an exact comparable observation cohort, calculates shared metrics, and evaluates downstream EPS, dividend, and yield methods. It may consume both systems through adapters, but it does not own either model's logic.
_Avoid_: Third forecast model, shared model engine, importing one forecast system from the other

**Shared Financial Forecast Module**:
The neutral `financial_forecast/` package that enforces complete annual revenue predictions and
applies availability-safe EPS, cash-dividend, and yield strategies. Ensemble and Rolling use it
through local adapters. It does not train revenue models, select methods, or compare systems.
_Avoid_: Third forecasting system, benchmark selector, importing either revenue engine

**Rolling LSTM Main Outputs**:
The Streamlit-facing comparison set: base Rolling LSTM, Rolling LSTM + Cluster, Rolling LSTM + Cluster + Conditional Adjustment, and optional architecture-explicit Rolling xLSTM comparison rows.
_Avoid_: Trend + Cycle main model, dynamic guardrail model, architecture-ambiguous xLSTM result

**Rolling xLSTM**:
The umbrella name for an optional no-cluster xLSTM comparison row. Every stored result must identify whether it used the historical mLSTM-only architecture or the Hybrid architecture.
_Avoid_: Omitting architecture provenance, Ensemble model

**Rolling xLSTM mLSTM-only**:
The historical one-block xLSTM architecture used by D1.5 through D1.20 and retained for reproducibility.
_Avoid_: Calling historical D1 evidence Hybrid evidence

**Rolling xLSTM Hybrid**:
The current Streamlit-default architecture that applies an mLSTM block followed by an sLSTM block. It is a distinct candidate whose evidence must be evaluated separately from historical D1 results.
_Avoid_: Inheriting mLSTM-only metrics, treating the architecture name as a backend claim

**xLSTM Conditional Adjustment**:
The optional post-processing row for Rolling xLSTM. It reuses the same time-safe growth/regime gates as the cluster adjustment, but has its own default alpha of 0.0 after D1.11 showed decline-cap-only beat growth boost; cluster adjustment keeps alpha 0.8.
The prediction-time gate itself does not read 2025 actuals, but the current balanced default was developed after 2025 replay analysis and therefore is not an independent holdout policy.
_Avoid_: Reusing cluster alpha by default, using target-year actuals for correction, calling the 2025 rerun independent validation

**Monthly Revenue**:
The reported revenue for one stock in one calendar month, expressed in thousands of New Taiwan dollars in the forecasting datasets.
_Avoid_: Return, sales signal

**Shared Revenue CSV Contract**:
The root `data/Stock_revenue_2019~2025.csv` preserves raw-NTD derived monetary columns
(`last_year_revenue`, `last_3m_revenue`, `last_12m_revenue`, `acc_revenue`) while
`revenue_thousand` is already thousand NTD. The Ensemble Forecast System converts those
feature columns to thousand NTD through `RevenueDataContract` before modeling.
_Avoid_: Guessing revenue feature units from value scale

**Canonical Data Directory**:
The forecasting systems read canonical CSVs from `data/` by default, or from
`PREDICT_DATA_DIR` when explicitly overridden. Root `data/` is the tracked generated
canonical output from ignored raw `free_taiwan_data/`, with `data/manifest.json`
recording the generator and data contract. Future data refreshes must pass the
`data_preprocessing` manifest and audit checks before commit.
The local raw directory intentionally retains only the five datasets consumed by current
preprocessing adapters; ignored candidates and audit artifacts belong under
`data_preprocessing/outputs/`, not inside the raw source.
_Avoid_: Deleting `data/` just because raw free data exists

**Pattern Cluster**:
An unsupervised KMeans identifier describing the recent rolling growth-direction pattern. It is not a permanent stock classification.
_Avoid_: Stock type, stock label

**Regime**:
A month-specific growth, cycle, or decline state derived only from information available before the prediction target.
_Avoid_: Permanent stock category

**Revenue Forecast**:
A system's estimated monthly revenue for a future or held-out evaluation month.
_Avoid_: Actual revenue

**Quarterly Target Ablation**:
A Rolling LSTM research workflow that compares rolling sums of monthly predictions against a direct next-3-month revenue target. It is used to test whether cycle-dominant stocks benefit from interval prediction.
_Avoid_: Main Streamlit output, quarterly replacement model

**Dividend Yield Forecast**:
A forecasting system's estimated dividend yield derived after revenue forecasting from EPS, dividend, and stock-price evidence. `as_of_price_yield` uses the latest observed close at the cutoff and is deployable; `target_month_end_yield` uses target-year observed closes and is evaluation-only. Each system owns its adapter and does not import the other.
_Avoid_: Revenue forecast, investment recommendation

**Evidence Tier**:
A reporting label that describes how safely an experiment may support a claim. Tier A requires a fixed-before-target or historical-validation selection protocol; Tier B is supporting evidence with caveats; Tier C is target-year development or post-hoc analysis; Tier D is legacy or superseded evidence.
_Avoid_: Treating a folder name or `report_ready=true` flag as sufficient proof by itself

**Report-Ready Run**:
A reproducible run with recorded command, git state, data provenance, and an eligible selection protocol. Human review must still confirm that upstream model defaults were not selected from the same target year.
_Avoid_: Any completed run, any clean-worktree run

**Availability-Safe Evidence**:
EPS, dividend, or price information whose availability date is on or before the forecast `as_of_date`. Announcement-safe dividend methods use `DividendAvailableDate`; the legacy `current_system_payout_ratio` is retained only as a hindsight diagnostic.
_Avoid_: Assuming fiscal year or ex-dividend year alone proves the value was known at prediction time
