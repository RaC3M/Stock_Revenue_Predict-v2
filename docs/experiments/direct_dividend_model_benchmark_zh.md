# Direct Dividend Model Benchmark

> 2026-07-31 狀態：最新本機 rerun 為
> `forecast_benchmark/outputs/report_ready_20260731_direct_dividend`。其方法使用 2022–2024
> validation 選擇後套用到 2025，是目前主要可引用的 time-safe downstream 結果；最新總結與
> caveat 請以 `experiment_registry.md` 為準。

## 目的

本實驗接在 announcement-safe dividend layer 後面，建立一個獨立的 time-safe direct dividend benchmark：

> 先預測是否配息，再在預測會配息的樣本上估計現金股利金額。

這個 benchmark 放在 `forecast_benchmark/`，不改 Ensemble 或 Rolling 主模型。

## Time-safe 規則

所有 direct model feature 都必須符合：

```text
available_date <= as_of_date
```

目前使用的可得日：

| 資料 | 可得日欄位 / 規則 |
|---|---|
| 現金股利 | `DividendAvailableDate`，優先等於 `AnnouncementDate` |
| EPS | `statement_available_date` |
| 月營收 | `revenue_available_date` |

預設切點：

| 用途 | as-of date |
|---|---|
| 2022 validation | `2022-01-10` |
| 2023 validation | `2023-01-10` |
| 2024 validation | `2024-01-10` |
| 2025 test | `2025-01-10` |

## 執行指令

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.direct_dividend_model_benchmark --output-dir forecast_benchmark\outputs\direct_dividend_model_benchmark
```

## 方法

Direct candidate methods：

| 類型 | 方法 |
|---|---|
| heuristic | `direct_hurdle_last_known` |
| heuristic | `direct_hurdle_recent_median` |
| heuristic | `direct_hurdle_smoothed` |
| Ridge hurdle | `direct_hurdle_ridge_t025` 到 `direct_hurdle_ridge_t060` |
| ElasticNet hurdle | `direct_hurdle_elastic_net_t025` 到 `direct_hurdle_elastic_net_t060` |

Threshold sweep：

```text
0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60
```

Validation protocol：

1. 用 `2022`、`2023`、`2024` 三個 validation fold 選方法。
2. 每個 fold 都只使用該 validation 年以前的 label，例如 2024 fold 只能用 2020-2023。
3. 以 `average_cash_dividend_abs_error` 作為 primary metric。
4. 同時產生 global method selection 與 dividend-history bucket method selection。
5. 將選出的 method 套到 `2025` test，test training label 年份為 `2020-2024`。

## Bucket Selection

Bucket selection 使用 as-of date 當下已知的配息歷史建立 bucket，不使用 target-year 除息結果。

目前欄位：

| 欄位 | 意義 |
|---|---|
| `paid_rate_bucket` | 近三次是否常配息 |
| `dividend_history_bucket` | 已知股利歷史是否足夠 |
| `latest_dividend_bucket` | 最近一次已知股利是否大於 0 |
| `dividend_selection_bucket` | 上述三者串接後的選模 bucket |

預設 support fallback：

```text
min_bucket_folds = 2
min_bucket_stock_years = 15
```

若某個 bucket 在 validation 中低於上述門檻，正式 `selected_dividend_method` 會回退到 global method，但仍保留 `bucket_winner_dividend_method` 作為診斷欄位。

## Label 規則

因為這次要做「是否配息」classification，所以不能只評估有 2025 除息紀錄的股票。

本 benchmark 採用：

```text
若股票在 dividend data 中有任何股利資料，但 target-year 沒有 ex-dividend record，則 actual cash dividend = 0。
若股票完全沒有 dividend data coverage，則 actual cash dividend = NaN，不納入 cash/yield metric。
```

因此 2025 test 的有效現金股利股票數為 `68`，比只看有除息紀錄的樣本更適合評估「不配息」判斷。

## Validation 結果

三年度 validation 選出的 global method：

| method | validation folds | stock-year count | average cash dividend AE | average yield MAE |
|---|---:|---:|---:|---:|
| `direct_hurdle_ridge_t060` | 3 | 246 | 1.1024 | 2.2820 pp |
| `direct_hurdle_ridge_t035` | 3 | 246 | 1.1139 | 2.3164 pp |
| `direct_hurdle_ridge_t040` | 3 | 246 | 1.1139 | 2.3164 pp |
| `direct_hurdle_ridge_t045` | 3 | 246 | 1.1219 | 2.3218 pp |
| `direct_hurdle_ridge_t050` | 3 | 246 | 1.1219 | 2.3218 pp |
| `direct_hurdle_elastic_net_t060` | 3 | 246 | 1.1261 | 2.3074 pp |

正式 global selection：

```text
selected_direct_method = direct_hurdle_ridge_t060
```

Bucket selection 共選出 `6` 個 bucket，其中 `3` 個 support 足夠、`3` 個回退到 global method：

| bucket | selected method | bucket winner | support status | validation cash AE | stock-year count |
|---|---|---|---|---:|---:|
| `paid_high|history_enough|latest_positive` | `direct_hurdle_ridge_t060` | `direct_hurdle_ridge_t060` | supported | 0.9983 | 138 |
| `paid_high|history_sparse|latest_positive` | `direct_hurdle_ridge_t060` | `direct_hurdle_ridge_t060` | supported | 1.0771 | 37 |
| `paid_mixed|history_enough|latest_zero` | `direct_hurdle_ridge_t060` | `direct_hurdle_elastic_net_t025` | fallback_to_global | 0.0000 | 1 |
| `paid_mixed|history_sparse|latest_positive` | `direct_hurdle_ridge_t060` | `direct_hurdle_recent_median` | fallback_to_global | 0.0000 | 1 |
| `paid_no_history|history_none|latest_missing` | `direct_hurdle_last_known` | `direct_hurdle_last_known` | supported | 0.6671 | 65 |
| `paid_none|history_sparse|latest_zero` | `direct_hurdle_ridge_t060` | `direct_hurdle_ridge_t025` | fallback_to_global | 0.6785 | 4 |

## 2025 Test 結果

正式 selected direct methods 對照 announcement-safe baselines：

| family | model | EPS method | dividend method | valid cash stocks | average cash dividend AE | average yield MAE |
|---|---|---|---|---:|---:|---:|
| direct selected | all compared revenue models | time_safe_features | `bucket_validation_best` | 68 | 0.9903 | 1.3749 pp |
| baseline | LightGBM | current_ratio | `announcement_safe_payout_ratio` | 68 | 0.9982 | 1.4110 pp |
| direct selected | all compared revenue models | time_safe_features | `direct_hurdle_ridge_t060` | 68 | 1.0377 | 1.4268 pp |
| baseline | ensemble_revenue | current_ratio | `announcement_safe_payout_ratio` | 68 | 1.0756 | 1.4998 pp |
| baseline | announcement-safe last cash dividend | varies | `announcement_safe_last_cash_dividend` | 64 | 1.2515 | 2.0693 pp |

解讀：

- 保守版 V2 的 `bucket_validation_best` 在現金股利 AE 與 yield MAE 都小幅打敗最佳 announcement-safe baseline。
- Support fallback 讓結果從未加 fallback 時的 `0.9732 / 1.3484 pp` 回落到 `0.9903 / 1.3749 pp`，但結論更容易 defend。
- Global-selected `direct_hurdle_ridge_t060` 沒有打敗最佳 baseline，代表改進主要來自 support 足夠的 bucket-level method selection。
- Direct model 不依賴 revenue model 的 2025 預測結果，因此四個 revenue model row 的現金股利與殖利率結果相同；保留 model 維度是為了和下游表格格式一致。

## Support Threshold Sensitivity

為了確認 bucket fallback 門檻不是剛好挑出好結果，額外重跑三種 support 設定：

| sensitivity | supported buckets | fallback buckets | average cash dividend AE | average yield MAE |
|---|---:|---:|---:|---:|
| loose `min1/stock1` | 6 | 0 | 0.9732 | 1.3484 pp |
| default `min2/stock15` | 3 | 3 | 0.9903 | 1.3749 pp |
| strict `min3/stock50` | 2 | 4 | 0.9903 | 1.3749 pp |
| best announcement-safe baseline | - | - | 0.9982 | 1.4110 pp |

解讀：

- 寬鬆門檻會得到最好的 2025 test 數字，但包含 stock-year count 只有 1 或 4 的小 bucket。
- 預設門檻與嚴格門檻結果相同，代表目前真正影響結果的是高支援度 bucket，而不是小 bucket 的偶然 winner。
- 即使使用更保守的 `min3/stock50`，direct bucket strategy 仍小幅優於最佳 announcement-safe baseline。

## Hindsight Diagnostic

2025 test 中，如果事後看所有 direct variants：

| method | average cash dividend AE | average yield MAE |
|---|---:|---:|
| `direct_hurdle_ridge_t045` | 0.9654 | 1.2996 pp |
| `direct_hurdle_elastic_net_t045` | 0.9767 | 1.3563 pp |
| `direct_hurdle_ridge_t035` | 0.9868 | 1.3670 pp |

但 `direct_hurdle_ridge_t045` 不是 validation 選出的正式方法，所以只能當 hindsight diagnostic，不能當正式 selected result。

## 結論

Direct dividend model 值得保留，而且 V2 已經比 V1 更接近正式可用的 time-safe 股利層。

最保守的結論：

> 使用 2022-2024 多年度 validation、time-safe dividend-history bucket selection，以及 minimum bucket support fallback 後，direct dividend model 在 2025 test 中仍能小幅改善現金股利 AE 與殖利率 MAE。下一步應做 support threshold sensitivity 或錯誤樣本分解，而不是只追更複雜的模型。

已完成錯誤樣本分解：

```text
docs/experiments/direct_dividend_error_diagnostics_zh.md
```

## 輸出檔案

```text
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_validation_predictions.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_validation_stock_accuracy.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_validation_overall_accuracy.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_validation_method_scores.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_bucket_method_scores.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_bucket_method_selection.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_method_selection.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_test_predictions.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_test_stock_accuracy.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_test_overall_accuracy.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_test_winner_summary.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_selected_test_predictions.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_selected_test_stock_accuracy.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_selected_test_overall_accuracy.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_baseline_overall_accuracy.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/direct_dividend_comparison_vs_baselines.csv
forecast_benchmark/outputs/direct_dividend_model_benchmark/run_config.json
```
