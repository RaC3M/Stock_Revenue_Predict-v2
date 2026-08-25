# Rolling LSTM Ablation Index

## Purpose

This file is the research map for Rolling LSTM experiments. It does not move scripts out of
`rolling_predict_LSTM/`; it records what each batch runner tests, which outputs are worth citing, and
which results are only exploratory.

Use this together with:

- `rolling_predict_LSTM/README.md`
- `docs/experiments/benchmark_protocol.md`
- `docs/experiments/experiment_registry.md`

## Evidence Tiers

| Tier | Meaning | Use In Report |
|---|---|---|
| A | Time-safe validation or fixed-parameter benchmark; suitable as main evidence. | Yes |
| B | Useful supporting evidence, but narrower basket or older main flow. | Yes, with caveat |
| C | Post-hoc replay or threshold scan against 2025 results. | Exploratory only |
| D | Archived/legacy experiment kept for research history. | Usually no |

## Main Report Stack

Recommended evidence order:

1. Human-reviewed status and caveats: `docs/experiments/experiment_registry.md`
2. Cross-system development benchmark: `forecast_benchmark/outputs/data_migration_revenue_20260730`
3. Exact-cohort historical mLSTM-only versus Hybrid development comparison: `rolling_predict_LSTM/outputs/xlstm_backbone_same_cohort_100_d1_22`
4. Current Hybrid fixed-parameter main-flow comparison: `rolling_predict_LSTM/outputs/xlstm_hybrid_main_flow_basket_100_d1_21` (pre-registered settings, but still Tier C because the policy carries 2025 development history)
5. Historical mLSTM-only Rolling main-flow development comparison: `rolling_predict_LSTM/outputs/xlstm_main_flow_basket_100_d1_16` (target-year hindsight; not the current Hybrid UI architecture)
6. Historically calibrated confidence result: `rolling_predict_LSTM/outputs/xlstm_confidence_calibration_d1_20` (time-safe selection step, but upstream policy was target-year-developed)
7. Quarterly target negative result: `rolling_predict_LSTM/outputs/quarterly_target_full`
8. Earlier method/feature ablations as background only: `rolling_predict_LSTM/outputs/ablation_full`

Do not build the final claim from post-hoc scans alone.

## Experiment Index

| Script | Latest Useful Output | Tier | Research Question | Current Takeaway |
|---|---|---:|---|---|
| `batch_xlstm_main_flow_comparison.py --xlstm-backbone xlstm` | `xlstm_main_flow_basket_100_d1_16` | C | Did the historical mLSTM-only main flow improve when adding no-cluster xLSTM rows? | Development result only. Historical `Rolling xLSTM` has best WMAPE `16.256%`; the balanced adjusted default was chosen after D1.15 inspected 2025 and D1.16 reused that year. This output is not Hybrid evidence. |
| `batch_xlstm_main_flow_comparison.py` (default Hybrid) | `xlstm_hybrid_main_flow_basket_100_d1_21` | C | Does the current `mLSTM → sLSTM` Hybrid UI architecture improve the same five rows? | Pre-registered fixed-parameter run completed 100/100 stocks. Hybrid adjusted WMAPE `17.448%` beats current-run cluster adjusted `19.359%`; Hybrid plain `17.598%` beats LSTM plain `21.259%`. The adjusted row only slightly improves WMAPE over Hybrid plain and lowers direction accuracy. |
| `batch_xlstm_confidence_calibration.py` | `xlstm_confidence_calibration_d1_20` | B | Can a decline-cap confidence threshold be selected from 2024 validation and applied to 2025? | The threshold-selection step is time-safe and the gain is small, but the upstream xLSTM/default policy was developed after inspecting 2025. Treat as supporting, not independent final proof. |
| `batch_quarterly_target_ablation.py` | `quarterly_target_full` | B | Does a direct next-3-month target beat rolling monthly-sum for quarterly revenue? | No. Rolling monthly sum with cluster + growth has WMAPE `25.232%`; best direct 3M variant is worse at WMAPE `28.747%`. The safe claim is limited to this implementation and evaluation setup. |
| `batch_ablation_study.py` | `ablation_full` | B | Which pre-xLSTM method/features matter across all stocks? | Growth adjustment with cluster/asymmetric setup gives the best WMAPE around `28.218%`; removing `log_revenue` severely hurts WMAPE. Useful background, but it predates xLSTM main-flow work. |
| `batch_ten_scenarios.py` | `rolling_lstm_10_scenarios_2025_20260724` | B | Which older scenario combination performed best on all stocks? | Fixed + Growth variants ranked best by WMAPE around `28.243%`. Use as history of why dynamic guardrail/autotune/trend-cycle were not kept. |
| `batch_all_stocks_penalty.py` | `rolling_lstm_all_stocks_2025_penalty_v2` | B | Does asymmetric underestimation penalty help all-stock Rolling LSTM? | Under-weight `2` improves WMAPE versus Huber/off baseline, but this is pre-xLSTM and should not be the final comparison. |
| `batch_sequence_backbone_ablation.py` | `xlstm_backbone_same_cohort_100_d1_22` | C | On exact matching observations, how does Hybrid compare with historical mLSTM-only? | Hybrid no-cluster has slightly worse pooled WMAPE (`17.598%` vs `17.166%`) but wins stock-level WMAPE for 63/100 stocks and stays within the frozen regression gate. Hybrid cluster rows are substantially better, but remain target-year development evidence and are not a current UI output. |
| `batch_xlstm_validation_fallback.py` | `xlstm_validation_fallback_d1_18_stock_regime_wmape5` | B | Can prior-year validation choose plain vs adjusted by stock/regime? | Time-safe but tiny improvement: WMAPE `16.25579%` vs xLSTM plain `16.25628%`; supports conservative gating but not enough for default. |
| `batch_xlstm_adjustment_ablation.py` | `xlstm_adjustment_ablation_d1_11` | C | Which xLSTM adjustment component helps after predictions already exist? | Decline cap only is the best basket-30 replay. This informed balanced cap design, but do not cite as final validation because it is post-hoc. |
| `batch_xlstm_decline_cap_confidence.py` | `xlstm_decline_cap_confidence_d1_19_fine` | C | Which confidence threshold would work best if scanned on 2025? | Threshold `0.45` has best WMAPE `16.184%`, but the threshold is chosen from 2025 results. Use only as exploratory lead-in to D1.20 calibration. |

## Report-Ready Claims

Safe claims:

- Rolling/historical mLSTM-only xLSTM is the better main research path under the shared 2025 development benchmark.
- The historical mLSTM-only xLSTM backbone has value, especially by WMAPE and MAE.
- Under D1.21's fixed basket-100 development run, Hybrid plain improves WMAPE by `3.661` percentage points versus current-run LSTM plain, and Hybrid adjusted improves it by `1.911` points versus current-run cluster adjusted.
- D1.21 supports retaining Hybrid as the Streamlit default, with Tier C and target-year-development caveats.
- D1.22 provides an exact-cohort architecture comparison: Hybrid plain is mixed rather than uniformly better, with pooled WMAPE `0.433` points worse but stock-level WMAPE wins on 63/100 stocks.
- Conditional adjustment improves MAPE/MedianAPE/SMAPE but can slightly hurt WMAPE and direction accuracy.
- Direct 3M quarterly target did not beat rolling monthly-sum in the current setup.
- Decline-cap confidence gating is promising, but calibrated gains are small.

Claims to avoid:

- "Rolling wins every stock."
- "Post-hoc threshold `0.45` is the final best setting."
- "Direct quarterly target is worse in general." The current evidence only says it did not beat this monthly-sum setup.
- "xLSTM + Cluster is better." Current evidence says no-cluster xLSTM was more stable.
- "Hybrid inherits the D1 results." D1.5～D1.20 used the historical mLSTM-only architecture.
- "D1.21 proves Hybrid beats historical mLSTM-only D1.16." Their cohorts overlap in only 13 stocks, so the metrics are not an exact architecture comparison.
- "D1.22 proves Hybrid plain is more accurate on every aggregate metric." Its pooled WMAPE, MAE, and MAPE are worse even though MedianAPE, SMAPE, direction accuracy, and most stock-level comparisons favor Hybrid.
- "D1.22 is enough to add Hybrid + Cluster to the UI." That row was inspected on 2025 and needs historical selection plus unseen-year confirmation.

## Leakage And Validity Notes

- D1.15 used 2025 replay metrics to select the balanced decline-cap default.
- D1.16 generates predictions before merging 2025 actuals, but its default was selected by D1.15 on the same target year; it is not an independent holdout.
- D1.20 chooses its threshold with 2024 validation, then applies it to 2025; its selection step is time-safe, while its upstream base policy still carries target-year development history.
- D1.19 confidence scan chooses thresholds from 2025 metrics; it is post-hoc.
- D1.11 adjustment ablation replays 2025 predictions and ranks variants with 2025 actuals; it is post-hoc.
- D1.5～D1.20 used mLSTM-only. Hybrid must receive a separate experiment ID and cannot inherit those evidence tiers or metrics.
- D1.21 used a pre-registered command and clean commit, but its balanced adjustment policy carries earlier 2025 development history; it remains Tier C and `report_ready=false`.
- D1.22 uses exact paired observations and frozen settings, but directly compares candidates on 2025; it remains Tier C and cannot be used for post-result tuning.
- Quarterly monthly-sum experiments may use actual monthly updates inside the quarter; cite this caveat when comparing to direct 3M target.

## Key Output Files

Main-flow comparison:

- `overall_accuracy.csv`
- `stock_accuracy.csv`
- `model_effects.csv`
- `winner_summary.csv`
- `industry_accuracy.csv`
- `regime_accuracy.csv`
- `monthly_predictions.csv`

Ablation runners commonly output:

- `overall_accuracy.csv`
- `stock_accuracy.csv`
- `winner_summary.csv` or effect summaries
- `monthly_predictions.csv` or replayed monthly predictions
- `run_config.json`

## Next Cleanup Candidates

1. Freeze the protocol and rerun the selected Rolling stack on a newly unseen year before promoting xLSTM claims to Tier A.
2. If Hybrid + Cluster is investigated, select it from historical validation rather than the inspected 2025 result.
3. Rerun the current exact-cohort cross-system benchmark after the protocol and target year are frozen.
4. Keep post-hoc scans under `outputs/` and cite them only as design exploration; keep `docs/experiments/experiment_registry.md` synchronized.
