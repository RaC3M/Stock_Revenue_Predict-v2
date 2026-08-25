# Frozen Financial Ablation — 2026-07-31

## Status

Tier B supporting evidence; `report_ready=false`.

The downstream code selects methods on 2022–2024 only and applies them to frozen 2025 revenue
predictions. However, the supplied Rolling/xLSTM predictions inherit target-year development
history, and this run was generated from a dirty worktree before the implementation was committed.
Do not promote it to independent holdout evidence.

Architecture note (2026-08-05): every Rolling xLSTM input in this artifact is historical
mLSTM-only, not the current Streamlit-default Hybrid architecture.

## Scope

- Frozen input: `forecast_benchmark/outputs/data_migration_revenue_20260730/comparable_monthly_predictions.csv`
- Target cohort: 82 stocks, 10 revenue models, 9,840 monthly prediction rows
- Validation years: 2022–2024
- Validation revenue source: observed monthly revenue replay
- Revenue model training during this run: none
- Output: `forecast_benchmark/outputs/financial_ablation_20260731`

Observed-revenue replay isolates the downstream EPS/dividend transformation. It does not validate
the upstream revenue models.

## Exact-cohort validation

Every method in a stage is scored on the same valid observation intersection.

### EPS stage

| EPS method | Observations | EPS MAE |
|---|---:|---:|
| `current_ratio` | 221 | 4.3485 |
| `seasonal_quarter_median` | 221 | 4.3632 |

The two EPS methods are effectively close on this replay cohort; `current_ratio` has the slightly
lower standalone EPS MAE.

### End-to-end selection

All eight EPS × dividend combinations use 144 cash-dividend observations and 1,728 monthly yield
observations. Selection uses cash-dividend MAE first, then yield MAE as the tie-breaker.

Selected combination:

- EPS: `seasonal_quarter_median`
- dividend: `announcement_safe_payout_ratio`
- validation cash-dividend MAE: `1.0766`
- validation yield MAE: `1.6948` percentage points

The standalone EPS winner and end-to-end winner differ. This is expected: lower EPS error does not
guarantee a better cash-dividend result after applying payout evidence.

## Frozen 2025 test

| Metric | Value |
|---|---:|
| Cash-dividend observations | 580 stock-model pairs |
| Cash-dividend MAE | 1.0632 per share |
| Yield observations | 6,960 stock-model-month rows |
| Yield MAE | 1.6219 percentage points |

All 82 stocks have complete frozen revenue rows and the runner recorded zero annual-revenue
failures. Target-year actual cash-dividend evidence supports yield-error scoring for 58 stocks
across all 10 models; missing evaluation evidence remains missing instead of being fabricated.

`as_of_price_yield` uses the latest observed close at the January 10 cutoff and is the deployable
view. `target_month_end_yield` uses observed 2025 month-end closes and is evaluation-only.

## Reproduction

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.financial_ablation --input-predictions forecast_benchmark\outputs\data_migration_revenue_20260730\comparable_monthly_predictions.csv --output-dir forecast_benchmark\outputs\financial_ablation_20260731 --report-ready false --evidence-tier B --registry-notes "All 82 comparable stocks; exact downstream method cohort; selection on 2022-2024 actual revenue replay; frozen 2025 predictions; upstream remains development evidence"
```
