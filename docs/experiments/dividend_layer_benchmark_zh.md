# Dividend Layer Benchmark 結果

> 2026-07-31 狀態：最新本機 rerun 為
> `forecast_benchmark/outputs/report_ready_20260731_dividend_layer`。其中 time-safe／announcement-safe
> rows 可作 supporting evidence；`current_system_payout_ratio` rows 仍只屬 legacy hindsight diagnostic。

> 2026-08-05 架構註記：其中 `Rolling xLSTM` rows 使用歷史 mLSTM-only 輸入，不是 Hybrid 證據。

## 目的

本實驗接在 EPS layer 對殖利率傳導 benchmark 後面，進一步檢查：

> 殖利率誤差到底是 EPS 轉換層造成，還是現金股利 / payout ratio 估計層造成？

答案目前是：股利層才是更大的風險點，而且當時的 payout ratio benchmark 含有目標年除息資訊。後續已用公告日欄位補做一版 `as_of_date = 2025-01-10` 的公告日安全 benchmark。Root `data/` 目前已包含這些 time-safe 欄位，不需再透過 `PREDICT_DATA_DIR` 指向 processed subset。

2026-07-30 更新：`current_system_payout_ratio` 是此 benchmark 保留的 legacy / hindsight diagnostic 名稱，不再代表目前 Ensemble Forecast System 的正式殖利率路徑。目前正式 Ensemble path 已改成 time-safe historical payout，不使用 target-year 除息資訊。

## 執行設定

執行指令：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.dividend_layer_benchmark --output-dir forecast_benchmark\outputs\dividend_layer_benchmark
```

歷史輸入資料：

```text
forecast_benchmark/outputs/basket_100/comparable_monthly_predictions.csv
```

設定：

- 預測年份：`2025`
- 共同股票池：`82` 檔
- 有效 EPS 股票數：`81`
- 有效現金股利 / 殖利率股票數：`58`
- 有效殖利率月份：`686`
- 股價門檻：`stock_price > 1.0`
- 失敗執行數：`0`

比較 EPS 方法：

- `current_ratio`
- `elastic_net_annual`

比較股利方法：

- `current_system_payout_ratio`
- `time_safe_payout_ratio`
- `announcement_safe_payout_ratio`
- `announcement_safe_last_cash_dividend`
- `last_cash_dividend`
- `recent_cash_dividend_median`
- `smoothed_cash_dividend`
- `eps_sign_guard_last_cash_dividend`

## 核心發現

第一，legacy `current_system_payout_ratio` 會使用目標年除息資訊。

在 `82` 檔共同股票池中，有 `49` 檔的 legacy current-system payout ratio 使用到 `ex_dividend_year = 2025` 的股利資料，比例為 `59.7561%`。如果目標是預測 2025 年殖利率，這不是 time-safe 設定。

若用更嚴格的公告日判斷，問題更明確：在 `free_taiwan_data` processed 版本中，legacy `current_system_payout_ratio` 有 `50 / 82` 檔使用到 `2025-01-10` 之後才公告或才可得的股利資訊，比例為 `60.9756%`。

第二，先前最佳殖利率結果需要重新標記。

| 系統 | 營收模型 | EPS 方法 | 股利方法 | 目標年除息使用檔數 | 平均 EPS AE | 平均現金股利 AE | 平均殖利率 AE |
|---|---|---|---|---:|---:|---:|---:|
| Rolling | Rolling xLSTM | current_ratio | current_system_payout_ratio | 49 | 4.0419 | 0.6541 | 0.9279 pp |
| Rolling | Rolling xLSTM + Adjustment | current_ratio | current_system_payout_ratio | 49 | 3.9566 | 0.6585 | 0.9319 pp |
| Ensemble | ensemble_revenue | current_ratio | current_system_payout_ratio | 49 | 3.0105 | 0.7172 | 1.0144 pp |
| Ensemble | LightGBM | current_ratio | current_system_payout_ratio | 49 | 2.8363 | 0.7758 | 1.0627 pp |

這些 row 雖然數值最好，但不可直接當作正式可部署殖利率 benchmark，因為 payout ratio 層可能吃到 2025 實際除息資料。

第三，在 time-safe 股利方法裡，目前最佳不是 Rolling。

| 系統 | 營收模型 | EPS 方法 | 股利方法 | 目標年除息使用檔數 | 平均 EPS AE | 平均現金股利 AE | 平均殖利率 AE |
|---|---|---|---|---:|---:|---:|---:|
| Ensemble | LightGBM | current_ratio | time_safe_payout_ratio | 0 | 2.8363 | 0.9371 | 1.2767 pp |
| Rolling | Rolling xLSTM | elastic_net_annual | eps_sign_guard_last_cash_dividend | 0 | 2.8922 | 1.0197 | 1.3830 pp |
| Rolling | Rolling xLSTM + Adjustment | elastic_net_annual | eps_sign_guard_last_cash_dividend | 0 | 2.8923 | 1.0197 | 1.3830 pp |
| Ensemble | ensemble_revenue | current_ratio | time_safe_payout_ratio | 0 | 3.0105 | 1.0274 | 1.3869 pp |
| Rolling | Rolling xLSTM | current_ratio | eps_sign_guard_last_cash_dividend | 0 | 4.0419 | 0.9947 | 1.3891 pp |

解讀時要注意：`eps_sign_guard_last_cash_dividend` 只有 `56` 檔有效現金股利股票，而 `time_safe_payout_ratio` 有 `58` 檔，因此最保守比較應優先看有效股票池一致的 `time_safe_payout_ratio`。

## Announcement-safe rerun

歷史重跑當時使用 `free_taiwan_data/processed_benchmark_82`。現在 root `data/` 已完成 migration，正式重跑可直接使用：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.dividend_layer_benchmark --dividend-methods current_system_payout_ratio,time_safe_payout_ratio,announcement_safe_payout_ratio,announcement_safe_last_cash_dividend --as-of-date 2025-01-10 --output-dir forecast_benchmark\outputs\announcement_safe_dividend_benchmark
```

輸出位置：

```text
forecast_benchmark/outputs/announcement_safe_dividend_benchmark
```

最佳列摘要：

| 系統 | 營收模型 | EPS 方法 | 股利方法 | 有效殖利率股票 | post-as-of 使用檔數 | 平均 EPS AE | 平均現金股利 AE | 平均殖利率 AE |
|---|---|---|---|---:|---:|---:|---:|---:|
| Rolling | Rolling xLSTM | current_ratio | current_system_payout_ratio | 52 | 50 | 3.5361 | 0.6245 | 0.8879 pp |
| Rolling | Rolling xLSTM + Adjustment | current_ratio | current_system_payout_ratio | 52 | 50 | 3.4478 | 0.6300 | 0.8942 pp |
| Ensemble | ensemble_revenue | current_ratio | current_system_payout_ratio | 52 | 50 | 2.5460 | 0.6787 | 0.9547 pp |
| Ensemble | LightGBM | current_ratio | current_system_payout_ratio | 52 | 50 | 2.3695 | 0.7432 | 1.0109 pp |
| Ensemble | LightGBM | current_ratio | time_safe_payout_ratio | 52 | 0 | 2.3695 | 0.9350 | 1.3031 pp |
| Ensemble | LightGBM | current_ratio | announcement_safe_payout_ratio | 52 | 0 | 2.3695 | 0.9356 | 1.3043 pp |
| Ensemble | ensemble_revenue | current_ratio | announcement_safe_payout_ratio | 52 | 0 | 2.5460 | 1.0355 | 1.4223 pp |
| Rolling | Rolling xLSTM | elastic_net_annual | announcement_safe_last_cash_dividend | 50 | 0 | 2.1917 | 1.0340 | 1.4279 pp |

重點解讀：

- legacy `current_system_payout_ratio` 仍是數值上最好的 hindsight baseline，但 `50 / 82 = 60.9756%` 股票使用到 `2025-01-10` 之後才可得的股利資訊。
- `announcement_safe_payout_ratio` 與 `time_safe_payout_ratio` 都能把 post-as-of 使用檔數降為 `0`，但平均殖利率 AE 會上升到約 `1.30 pp` 以上。
- 在公告日安全的 payout-ratio 方法中，目前最好是 `LightGBM + current_ratio + announcement_safe_payout_ratio`，平均殖利率 AE 為 `1.3043 pp`。
- `announcement_safe_last_cash_dividend` 不依賴 EPS 方法，因此不同營收 / EPS 組合的股利與殖利率誤差幾乎相同；它可作為簡單且可部署的現金股利 baseline。
- 這個結果表示殖利率預測的主要瓶頸不是單純月營收，而是「在預測當下能知道多少股利資訊」以及「如何 time-safe 地預測現金股利」。

## Leakage Diagnostic

| 系統 | 營收模型 | EPS 方法 | 股票數 | 使用目標年除息檔數 | 使用率 | 平均殖利率 AE | 平均現金股利 AE |
|---|---|---|---:|---:|---:|---:|---:|
| Ensemble | ensemble_revenue | current_ratio | 82 | 49 | 59.7561% | 1.0144 pp | 0.7172 |
| Ensemble | ensemble_revenue | elastic_net_annual | 82 | 49 | 59.7561% | 1.2328 pp | 0.6884 |
| Ensemble | LightGBM | current_ratio | 82 | 49 | 59.7561% | 1.0627 pp | 0.7758 |
| Ensemble | LightGBM | elastic_net_annual | 82 | 49 | 59.7561% | 1.2279 pp | 0.6877 |
| Rolling | Rolling xLSTM | current_ratio | 82 | 49 | 59.7561% | 0.9279 pp | 0.6541 |
| Rolling | Rolling xLSTM | elastic_net_annual | 82 | 49 | 59.7561% | 1.2336 pp | 0.6952 |
| Rolling | Rolling xLSTM + Adjustment | current_ratio | 82 | 49 | 59.7561% | 0.9319 pp | 0.6585 |
| Rolling | Rolling xLSTM + Adjustment | elastic_net_annual | 82 | 49 | 59.7561% | 1.2321 pp | 0.6946 |

## 對前面結論的修正

月營收 benchmark 的結論不受影響：

> Rolling/xLSTM 仍是目前月營收預測主研究方向。

但殖利率 benchmark 的結論需要改成：

> 在當時的 legacy payout ratio 設定下，Rolling xLSTM + current_ratio 的殖利率 AE 最低；但該 payout ratio 會使用目標年除息與公告日後才可得的股利資料，因此只能視為 hindsight-assisted baseline，不應視為正式 time-safe 結果。目前 Ensemble Forecast System 的正式路徑已改成 time-safe historical payout。

這也解釋了為什麼 EPS layer 變準時，殖利率反而變差：legacy current-system payout ratio 太接近實際 2025 配息狀況，導致 current_ratio yield 看起來特別好。

## 建議下一步

1. 將正式殖利率 benchmark 改成 `announcement_safe_payout_ratio`、`time_safe_payout_ratio` 或其他不使用公告日後資訊的方法。
2. 已建立 direct dividend model benchmark：先判斷是否配息，再預測配息金額。
3. 下一步改進 direct model 的 threshold selection 或 stock/bucket-level method selection。
4. 把 EPS layer 與 dividend layer 分開選模，不要只用 EPS AE 決定下游殖利率方法。
5. 在報告中保留 legacy current-system payout row，但標註為 hindsight-assisted diagnostic。

## 可放入報告的句子

保守版：

> Dividend layer benchmark 顯示，legacy payout ratio 估計會在多數股票中使用 2025 除息資料，因此先前殖利率結果應視為 hindsight-assisted baseline。若改用 time-safe dividend layer，殖利率誤差明顯上升，代表下游股利估計仍需獨立建模。

答辯版：

> 我們不是只停在 EPS layer，而是進一步拆解 dividend layer。結果發現 legacy payout-ratio baseline 對 2025 殖利率評估有目標年除息資訊污染，因此後續研究會將 EPS 預測與股利預測分開，並以 time-safe 現金股利誤差或殖利率誤差作為選模指標。

## 輸出檔案

```text
forecast_benchmark/outputs/dividend_layer_benchmark/dividend_layer_predictions.csv
forecast_benchmark/outputs/dividend_layer_benchmark/dividend_layer_stock_accuracy.csv
forecast_benchmark/outputs/dividend_layer_benchmark/dividend_layer_overall_accuracy.csv
forecast_benchmark/outputs/dividend_layer_benchmark/dividend_layer_winner_summary.csv
forecast_benchmark/outputs/dividend_layer_benchmark/dividend_layer_improvement_vs_baseline.csv
forecast_benchmark/outputs/dividend_layer_benchmark/dividend_layer_leakage_diagnostic.csv
forecast_benchmark/outputs/dividend_layer_benchmark/dividend_layer_failed_runs.csv
forecast_benchmark/outputs/dividend_layer_benchmark/run_config.json
```
