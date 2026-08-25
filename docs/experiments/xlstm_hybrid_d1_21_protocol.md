# D1.21 Hybrid xLSTM Basket-100 Protocol

## Status

Pre-registered before the large-sample Hybrid result was generated.

- Protocol date: 2026-08-06
- Frozen model-code commit: `8e6320a534adba662289d64e40621084cb67e0ff`
- Data manifest SHA-256: `a69e63f427c28670b0615f0b8aac70e09a85cdd0cefff22ca7cf5c22466f950e`
- Planned experiment ID: `rolling_main_flow:xlstm_hybrid_main_flow_basket_100_d1_21`
- Planned output: `rolling_predict_LSTM/outputs/xlstm_hybrid_main_flow_basket_100_d1_21`

This protocol freezes the Hybrid evaluation settings before inspecting its basket-100 metrics. It
does not create an unseen-year holdout: the target remains 2025, and the balanced decline-cap policy
was developed from earlier 2025 replay. The result must therefore remain Tier C development evidence
and `report_ready=false` even though the Hybrid architecture and parameters are fixed before this run.

## Research Question

Under the current Streamlit five-row main flow and the same deterministic basket-100 selection used
by historical D1.16, does the `mLSTM → sLSTM` Hybrid provide useful large-sample revenue-forecast
performance without mixing its evidence with the historical mLSTM-only architecture?

## Frozen Command

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_main_flow_comparison.py --output-dir rolling_predict_LSTM\outputs\xlstm_hybrid_main_flow_basket_100_d1_21 --xlstm-backbone xlstm_hybrid --k 6 --epochs 35 --max-train-samples 40000 --stock-limit 100 --min-2025-months 12 --sample-seed 42
```

No parameter, stock-selection, or model-row change may be made after inspecting D1.21 results. A
changed configuration requires a new experiment ID and must be reported as follow-up development.

## Frozen Cohort And Models

- Target year: 2025.
- Cohort: stocks with at least 12 target-year months, deterministic sample seed `42`, limit `100`.
- Window: 12 months.
- KMeans clusters: `6`.
- Epochs: `35`.
- Maximum training samples: `40,000`.
- Asymmetric loss: enabled, under-weight `2.0`.
- Cluster Growth Adjustment: enabled, alpha `0.8`.
- Hybrid adjustment: alpha `0.0`, balanced decline cap with growth-ratio maximum `0.35` and
  prediction/last-observed minimum `1.10`.
- xLSTM architecture: `xlstm_hybrid`, one mLSTM block followed by one native-backend sLSTM block.

The five stored model rows are:

1. `Rolling LSTM`
2. `Rolling LSTM + Cluster`
3. `Rolling LSTM + Cluster + Conditional Adjustment`
4. `Rolling xLSTM` with `sequence_backbone=xlstm_hybrid`
5. `Rolling xLSTM + Conditional Adjustment` with `sequence_backbone=xlstm_hybrid`

## Metrics And Comparisons

Primary metric: WMAPE.

Secondary metrics: MAE, MAPE, MedianAPE, SMAPE, DirectionAccuracy, stock-level wins, regime-level
accuracy, and failure count.

Predeclared comparisons:

1. Hybrid plain versus `Rolling LSTM` plain.
2. Hybrid adjusted versus `Rolling LSTM + Cluster + Conditional Adjustment`.
3. Hybrid adjusted versus Hybrid plain.
4. Context-only comparison against historical mLSTM-only D1.16 on the same deterministic cohort.

The D1.16 comparison is contextual rather than a pooled metric: D1.21 and D1.16 must retain separate
architecture provenance and experiment IDs.

## Merge Recommendation Rule

Recommend retaining Hybrid as the Streamlit default for merge only when:

- at least 95 of the 100 selected stocks produce complete model outputs;
- Hybrid plain does not materially regress primary WMAPE versus the current-run `Rolling LSTM`
  plain baseline;
- Hybrid adjusted does not materially regress primary WMAPE versus the current-run clustered
  adjusted baseline; and
- architecture provenance is present in all stored summaries, effects, and monthly predictions.

For this development run, a WMAPE increase greater than `1.0` percentage point is treated as a
material regression. Secondary metrics may explain trade-offs but must not be used to tune D1.21
after seeing the result.

If the rule fails, do not recommend merging with Hybrid as the default. Preserve the result as Tier C
evidence and open a separately identified follow-up rather than changing this protocol retroactively.

## Completed Result

D1.21 completed on 2026-08-06 without changing the frozen command or parameters.

- Result commit: `f1bbe2ddef21521e012f1565ab2140b677a2f98f`.
- Worktree at run time: clean.
- Runtime: `419.522` seconds.
- Successful stocks: `100/100`; failed runs: `0`.
- Evaluation rows: `1,175` per model (`5,875` stored monthly rows).
- Evidence classification: Tier C, `report_ready=false`, because the balanced decline-cap policy
  carries earlier 2025 development history.

| Model | MAE | MAPE | MedianAPE | WMAPE | SMAPE | DirectionAccuracy |
|---|---:|---:|---:|---:|---:|---:|
| Rolling LSTM | 315,395 | 45.566% | 11.958% | 21.259% | 19.220% | 60.766% |
| Rolling LSTM + Cluster | 288,754 | 39.883% | 11.340% | 19.464% | 19.007% | 62.298% |
| Rolling LSTM + Cluster + Conditional Adjustment | 287,196 | 39.580% | 11.298% | 19.359% | 18.896% | 57.957% |
| Rolling xLSTM (Hybrid) | 261,083 | 42.749% | 10.565% | 17.598% | 18.193% | 62.383% |
| Rolling xLSTM + Conditional Adjustment (Hybrid) | 258,845 | 42.687% | 10.565% | 17.448% | 18.259% | 60.170% |

Predeclared primary comparisons:

| Comparison | WMAPE delta | Stock-level WMAPE wins | Interpretation |
|---|---:|---:|---|
| Hybrid plain minus LSTM plain | -3.661 pp | 57/100 | Pass; no material regression. |
| Hybrid adjusted minus cluster adjusted | -1.911 pp | 53/100 | Pass; no material regression. |
| Hybrid adjusted minus Hybrid plain | -0.151 pp | 6 wins, 10 losses, 84 ties | Small aggregate gain; not a universal improvement. |

The adjusted Hybrid applied the decline cap to `35/1,175` rows (`2.98%`). It improved aggregate
WMAPE slightly versus Hybrid plain, but DirectionAccuracy fell by `2.213` percentage points. Regime
results also show a trade-off: Hybrid adjusted improved decline WMAPE from `21.841%` to `20.658%`,
while decline DirectionAccuracy fell from `56.701%` to `29.897%`. The adjusted row should therefore
remain a comparison option rather than be described as uniformly better than Hybrid plain.

All xLSTM summaries and monthly rows record `sequence_backbone=xlstm_hybrid` and
`xlstm_backbone=xlstm_hybrid`; no provenance rows are missing.

## Cohort Limitation And Decision

Although D1.21 reused the deterministic seed/limit policy, its selected cohort overlaps historical
D1.16 in only 13 stocks. Dataset and candidate-universe changes mean D1.21 must not be used as an
exact Hybrid-versus-historical-mLSTM comparison. D1.16 remains historical context only.

All pre-registered merge-recommendation gates passed. The evidence supports retaining Hybrid as the
Streamlit default and makes the branch eligible for merge review. It does not support Tier A
promotion. At D1.21 completion, both a newly unseen target year and an exact-cohort architecture
comparison were missing; D1.22 later supplied the exact-cohort comparison, while the unseen-year gap
remains.
