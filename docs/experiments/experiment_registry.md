# Experiment Registry

> Human-reviewed status updated 2026-08-06. Generated output folders stay ignored by Git. This file
> records what may be cited after considering both run metadata and the full model-selection history.

## Evidence tiers

| Tier | Meaning | Reporting use |
|---|---|---|
| A | Protocol fixed before target scoring, or method selected only from historical validation | Main evidence |
| B | Time-safe supporting analysis with scope or upstream-policy caveats | Cite with caveat |
| C | Target-year development, post-hoc replay, or hindsight diagnostic | Exploratory only |
| D | Legacy or superseded result | Historical context only |

## Required registry fields

Current benchmark runners record the following in `run_config.json` and
`experiment_registry_entry.json`:

- experiment ID and family;
- evidence tier, report-ready flag, reason, and eligibility;
- selection protocol;
- git commit and dirty-worktree state;
- source manifest hash, per-file hashes, and dataset bundle hash;
- command arguments and output directory.

Older outputs may predate one or more fields. A historical `report_ready=true` or `Tier A` flag does
not override a target-year-developed upstream policy.

## Current citable run

| Experiment | Output | Human tier | Main use | Current takeaway |
|---|---|---:|---|---|
| Direct dividend model | `forecast_benchmark/outputs/report_ready_20260731_direct_dividend` | A | Time-safe dividend/yield method selection | Multi-year validation selects `bucket_validation_best`; it gives a small aggregate improvement versus announcement-safe baselines, but stock-level wins remain mixed. |

The direct-dividend result is citable because its hurdle/bucket method is selected on 2022–2024
validation and then applied to 2025. Do not generalize the small average gain into an every-stock win.

## Latest rerun map

These folders are the newest local reruns, but their names and generated flags are not the final
human evidence classification:

| Output | Generated flag | Human status | Reason |
|---|---:|---:|---|
| `financial_ablation_20260731` | B / false | B | Exact-cohort downstream selection on 2022–2024 actual-revenue replay, followed by frozen 2025 scoring. No revenue retraining; inherits upstream historical mLSTM-only xLSTM development history and was generated from a dirty worktree. |
| `report_ready_20260731_revenue` | A / true | C | Clean rerun at `18c316a`, but the historical mLSTM-only D1.16 path was developed with 2025 replay and the output predates the exact-pair cohort enforcement in `aa6b50d`. |
| `report_ready_20260731_eps` | A / true | B | Useful downstream rerun; EPS method comparison is supporting evidence and must retain revenue-source caveats. |
| `report_ready_20260731_eps_diagnostics` | A / true | C | Diagnostic labels inspect target-year errors and are not deployable selection rules. |
| `report_ready_20260731_eps_layer_validation` | A / true | B | Uses prior-year validation, but inherits upstream revenue-policy development history. |
| `report_ready_20260731_yield` | A / true | D | Retained as transmission diagnostic; the historical default path is not the final dividend-layer claim. |
| `report_ready_20260731_yield_eps_layer` | A / true | D | Shows that EPS gains need not improve yield, but should not be used as the final time-safe yield ranking. |
| `report_ready_20260731_dividend_layer` | A / true | B | Announcement/time-safe rows are useful; legacy `current_system_payout_ratio` rows remain hindsight-only. |
| `report_ready_20260731_direct_dividend` | A / true | A | Multi-year historical validation followed by 2025 test; current primary downstream result. |
| `report_ready_20260731_direct_dividend_diagnostics` | A / true | B | Explains 2025 wins/losses after scoring; useful diagnosis, not a method-selection protocol. |

`report_ready_20260730_ensemble_fix_revenue` was generated from a dirty worktree and is superseded by
the later revenue rerun and current exact-cohort code.

The frozen financial ablation detail is recorded in
[`financial_ablation_20260731.md`](financial_ablation_20260731.md). Its selected downstream
combination is `seasonal_quarter_median + announcement_safe_payout_ratio`; the 2025 frozen test has
cash-dividend MAE `1.0632` and yield MAE `1.6219` percentage points on available evaluation rows.

## Rolling development evidence

D1.5 through D1.20 use the historical mLSTM-only architecture. D1.21 is the first separately
registered fixed-parameter basket-100 result for the current Streamlit default,
`xlstm_hybrid` (`mLSTM → sLSTM`). It has its own Tier C result and does not inherit historical D1
metrics or evidence tiers. D1.22 then compares both backbones on exact matching observations.

| Output | Tier | Reason |
|---|---:|---|
| `rolling_predict_LSTM/outputs/xlstm_backbone_same_cohort_100_d1_22` | C | Pre-registered exact-cohort architecture run at clean commit `b2f5605`; both backbones completed 100/100 stocks with 1,175 exact paired observations per model row. Hybrid no-cluster WMAPE is `17.598%` versus historical mLSTM-only `17.166%` (+`0.433` pp), although Hybrid wins stock-level WMAPE for 63/100 stocks. Hybrid cluster rows improve substantially, but are secondary development evidence and are not a current UI output. See [`xlstm_backbone_same_cohort_d1_22_protocol.md`](xlstm_backbone_same_cohort_d1_22_protocol.md). |
| `rolling_predict_LSTM/outputs/xlstm_hybrid_main_flow_basket_100_d1_21` | C | Pre-registered fixed-parameter Hybrid run at clean commit `f1bbe2d`: 100/100 stocks succeeded. Hybrid adjusted WMAPE is `17.448%` versus current-run cluster adjusted `19.359%`; Hybrid plain is `17.598%` versus LSTM plain `21.259%`. The balanced policy still carries 2025 development history, so `report_ready=false`. See [`xlstm_hybrid_d1_21_protocol.md`](xlstm_hybrid_d1_21_protocol.md). |
| `rolling_predict_LSTM/outputs/xlstm_main_flow_basket_100_d1_16` | C | Historical mLSTM-only. Balanced decline-cap policy was selected from D1.15 2025 replay and rerun on the same target year. |
| `rolling_predict_LSTM/outputs/xlstm_confidence_calibration_d1_20` | B | Historical mLSTM-only. Threshold selection uses 2024 validation, but the upstream xLSTM/default policy carries 2025 development history. |
| `rolling_predict_LSTM/outputs/quarterly_target_full` | B | Fixed comparison supports only the narrow negative result that this direct-3M version did not beat monthly-sum. |
| `rolling_predict_LSTM/outputs/ablation_full` | B | Useful pre-xLSTM method/feature background, not the current main-flow comparison. |

## Legacy and superseded outputs

| Output | Tier | Reason |
|---|---:|---|
| `forecast_benchmark/outputs/basket_100` | D | Pre-canonical-data comparison. |
| `forecast_benchmark/outputs/data_migration_revenue_20260730` | C | Exact-cohort development comparison; useful headline context, not an independent holdout. |
| `forecast_benchmark/outputs/yield_benchmark` | D | Legacy payout-path diagnostic. |
| `forecast_benchmark/outputs/dividend_layer_benchmark` | B | Leakage audit; cite time-safe rows, not the legacy winner. |
| `forecast_benchmark/outputs/announcement_safe_dividend_benchmark` | A | Announcement-safe comparison, superseded for the primary downstream conclusion by validated direct dividend where applicable. |

## Current gap

No full cross-system run has yet been recorded using both the current exact source/model cohort
contract and a newly unseen target year. D1.21 supplies formal fixed-parameter Hybrid large-sample
development evidence, and D1.22 supplies the previously missing exact-cohort Hybrid-versus-mLSTM
architecture comparison. Both still evaluate 2025 after earlier target-year development. A frozen
unseen-year rerun remains required before promoting the Hybrid result or either architecture's
headline comparison to Tier A.
