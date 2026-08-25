# D1.22 Same-Cohort xLSTM Backbone Protocol

## Status

Pre-registered before inspecting the same-cohort architecture result.

- Protocol date: 2026-08-06
- Frozen model-code commit: `0b620ffee4020ad9599f77ea785980a7b39e530f`
- Data manifest SHA-256: `a69e63f427c28670b0615f0b8aac70e09a85cdd0cefff22ca7cf5c22466f950e`
- Planned experiment ID: `rolling_sequence_backbone:xlstm_backbone_same_cohort_100_d1_22`
- Planned output: `rolling_predict_LSTM/outputs/xlstm_backbone_same_cohort_100_d1_22`

This experiment answers the architecture question left open by D1.21 and historical D1.16. Both
backbones are trained and evaluated in one invocation over the exact same selected stocks and target
months. The result remains Tier C and `report_ready=false`: both candidates are compared on the 2025
target-year evaluation set, and this is not a newly unseen-year test.

## Research Question

With the rolling data, sample order, KMeans artifacts, loss, adjustment settings, and evaluation path
held fixed, does the current `mLSTM → sLSTM` Hybrid improve over the historical mLSTM-only xLSTM
backbone on an exact shared cohort?

## Frozen Command

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_sequence_backbone_ablation.py --output-dir rolling_predict_LSTM\outputs\xlstm_backbone_same_cohort_100_d1_22 --backbones xlstm,xlstm_hybrid --k 6 --epochs 35 --max-train-samples 40000 --stock-limit 100 --min-2025-months 12 --sample-seed 42
```

No parameter, cohort, model-row, or comparison change may be made after inspecting D1.22. Any changed
configuration requires a new experiment ID.

## Frozen Cohort And Settings

- Target year: 2025.
- Cohort: stocks with at least 12 target-year months, deterministic sample seed `42`, limit `100`.
- Backbones, in baseline/challenger order: `xlstm`, `xlstm_hybrid`.
- Historical backbone: mLSTM-only.
- Hybrid backbone: one mLSTM block followed by one native-PyTorch sLSTM block.
- Window: 12 months.
- KMeans clusters: `6`.
- Epochs: `35`.
- Maximum training samples: `40,000`.
- Asymmetric loss: enabled, under-weight `2.0`.
- Conditional Growth Adjustment: enabled, alpha `0.8`.

Each backbone produces the same three model variants:

1. no-cluster plain;
2. cluster plain;
3. cluster plus Conditional Adjustment.

## Metrics And Comparisons

Primary comparison: Hybrid versus historical mLSTM-only WMAPE for the no-cluster plain row. This
isolates the backbone used by the current optional xLSTM UI row.

Secondary comparisons:

- MAE, MAPE, MedianAPE, SMAPE, DirectionAccuracy, and runtime for no-cluster plain;
- WMAPE and the same secondary metrics for cluster plain and cluster adjusted;
- paired stock-level wins, losses, and ties;
- regime- and industry-level metrics;
- failure count and architecture provenance completeness.

All aggregate comparisons must use only exact backbone/model observation pairs. Lower error-metric
deltas mean Hybrid improved; higher DirectionAccuracy deltas mean Hybrid improved.

## Decision Rule

The same-cohort result supports retaining Hybrid as the default architecture when:

- at least 95 stocks have complete paired outputs for both backbones;
- no xLSTM row is missing or mixing `sequence_backbone` / `xlstm_backbone` provenance; and
- Hybrid no-cluster plain does not regress WMAPE by more than `1.0` percentage point versus
  historical mLSTM-only no-cluster plain.

A failure should open a separately scoped architecture issue. It must not trigger post-result tuning
or an automatic revert because D1.22 is still target-year development evidence. Passing this rule also
does not promote Hybrid to Tier A; that requires a fully frozen run on a newly unseen year.

## Completed Result

D1.22 completed on 2026-08-06 without changing the frozen command or parameters.

- Run commit: `b2f560571a953a5d09821169b18b0b4bf15918bd`.
- Worktree at run time: clean.
- Runtime: `438.534` seconds.
- Successful runs: `100/100` stocks for each backbone; failed runs: `0`.
- Exact paired observations: `1,175` per backbone/model row; no left-only or right-only keys.
- Stored monthly rows: `7,050`.
- Evidence classification: Tier C, `report_ready=false`.

| Model variant | Historical mLSTM-only WMAPE | Hybrid WMAPE | Hybrid minus historical | Hybrid stock WMAPE wins |
|---|---:|---:|---:|---:|
| No-cluster plain | 17.166% | 17.598% | +0.433 pp | 63/100 |
| Cluster plain | 20.823% | 15.907% | -4.916 pp | 77/100 |
| Cluster + Conditional Adjustment | 20.625% | 16.083% | -4.542 pp | 74/100 |

For the primary no-cluster comparison, Hybrid has slightly worse aggregate WMAPE, MAE
(`261,083` versus `254,665`), and MAPE (`42.749%` versus `34.399%`). It has slightly better
MedianAPE (`10.565%` versus `10.875%`), SMAPE (`18.193%` versus `18.252%`), and
DirectionAccuracy (`62.383%` versus `62.128%`). The aggregate and stock-level results point in
different directions: Hybrid wins stock-level WMAPE for 63 stocks, while a small number of large
errors, led by stock `1326`, make its pooled WMAPE worse.

The two clustered variants favor Hybrid much more clearly on this same cohort. Those rows are
secondary architecture evidence only: xLSTM + Cluster is not a current Streamlit main output, and
D1.22 must not be used to add it to the UI after inspecting the 2025 result.

All monthly keys match exactly across backbones. Rows under `xlstm` record
`xlstm_backbone=xlstm`; rows under `xlstm_hybrid` record `xlstm_backbone=xlstm_hybrid`. No provenance
violations were found.

## Decision

All pre-registered decision-rule conditions passed. Hybrid no-cluster plain regressed WMAPE by
`0.433` percentage points, below the frozen `1.0`-point material-regression threshold. This supports
retaining Hybrid as the default architecture, but not claiming that Hybrid plain beats historical
mLSTM-only on aggregate error.

Do not tune D1.22 after the result. The next architecture-evidence step is a fully frozen unseen-year
run. Any investigation of the strong Hybrid + Cluster development result must choose settings from
historical validation and remain separate from the current UI default.
