# EPS Layer 對殖利率傳導 Benchmark 結果

## 2026-07-31 狀態

本機已有 `forecast_benchmark/outputs/report_ready_20260731_yield_eps_layer` rerun。這份實驗仍定位為 EPS layer 對 yield 傳導的 diagnostic：它說明「EPS AE 變好不保證 yield AE 變好」，但其中搭配 legacy payout path 的低 yield AE 不能當作正式可部署結論。

2026-08-05 架構註記：本文件所有 `Rolling xLSTM` rows 都來自歷史 mLSTM-only 輸入，不是目前 Streamlit 預設的 Hybrid 證據。

正式殖利率結論應引用 dividend layer / direct dividend 的 time-safe benchmark；本文件的主要可引用結論是：下游選模不能只看 EPS AE，必須直接用 `cash_dividend_abs_error` 或 `yield_mae_percent_point` 驗證。

## 目的

本實驗檢查前一輪 EPS layer validation 的關鍵後續問題：

> 固定 ML EPS layer 讓 Rolling/xLSTM 的 EPS 誤差下降後，殖利率誤差是否也會下降？

答案目前是：沒有。

這是重要的負結果。它表示 EPS 變準不一定會讓現金股利或殖利率變準，因為下游還有配息率、現金股利截斷、股價分母與股利政策等因素。

## 執行設定

執行指令：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.yield_eps_layer_benchmark --output-dir forecast_benchmark\outputs\yield_eps_layer_benchmark
```

歷史輸入資料：

```text
forecast_benchmark/outputs/basket_100/comparable_monthly_predictions.csv
```

設定：

- 預測年份：`2025`
- 股票池：`82` 檔
- 有效 EPS 股票數：`81`
- 有效現金股利/殖利率股票數：`58`
- 有效殖利率月份：`686`
- 股價門檻：`stock_price > 1.0`
- 失敗執行數：`0`

比較營收模型：

- `Rolling xLSTM`
- `Rolling xLSTM + Conditional Adjustment`
- `ensemble_revenue`
- `LightGBM`

比較 EPS 方法：

- `current_ratio`
- `lasso_annual`
- `elastic_net_annual`

## Overall 結果

| 系統 | 營收模型 | EPS 方法 | 平均 EPS AE | 平均現金股利 AE | 平均殖利率 AE |
|---|---|---|---:|---:|---:|
| Rolling | Rolling xLSTM | current_ratio | 4.0419 | 0.6541 | 0.9279 pp |
| Rolling | Rolling xLSTM + Adjustment | current_ratio | 3.9566 | 0.6585 | 0.9319 pp |
| Ensemble | ensemble_revenue | current_ratio | 3.0105 | 0.7172 | 1.0144 pp |
| Ensemble | LightGBM | current_ratio | 2.8363 | 0.7758 | 1.0627 pp |
| Ensemble | LightGBM | lasso_annual | 2.9046 | 0.6874 | 1.2272 pp |
| Ensemble | LightGBM | elastic_net_annual | 2.9045 | 0.6877 | 1.2279 pp |
| Rolling | Rolling xLSTM + Adjustment | lasso_annual | 2.8924 | 0.6943 | 1.2315 pp |
| Ensemble | ensemble_revenue | lasso_annual | 2.9097 | 0.6881 | 1.2321 pp |
| Rolling | Rolling xLSTM + Adjustment | elastic_net_annual | 2.8923 | 0.6946 | 1.2321 pp |
| Rolling | Rolling xLSTM | lasso_annual | 2.8923 | 0.6949 | 1.2329 pp |
| Rolling | Rolling xLSTM | elastic_net_annual | 2.8922 | 0.6952 | 1.2336 pp |

## 重點解讀

第一，Rolling/xLSTM 接 ML EPS layer 後，EPS 確實改善。

- `Rolling xLSTM + current_ratio` EPS AE：`4.0419`
- `Rolling xLSTM + elastic_net_annual` EPS AE：`2.8922`
- EPS AE 改善：`1.1497`

第二，但殖利率誤差反而惡化。

- `Rolling xLSTM + current_ratio` yield AE：`0.9279 pp`
- `Rolling xLSTM + elastic_net_annual` yield AE：`1.2336 pp`
- yield AE 惡化：`0.3057 pp`
- 相對 current ratio 惡化：`32.9454%`

第三，這表示 EPS layer 不能只用 EPS AE 選模型。

現金股利估計公式是：

```text
estimated_cash_dividend = max(estimated_eps * payout_ratio, 0)
```

因此即使 EPS 更接近實際 EPS，也可能讓現金股利更偏離實際發放股利。例如：

- ML EPS 預測為負時，現金股利會被截斷成 `0`，但公司實際仍可能配息。
- 對低 EPS 公司，EPS 小誤差乘上配息率和股價分母後，yield 誤差可能被放大。
- current_ratio 有時雖然 EPS 較不準，但剛好更接近實際現金股利，導致 yield 較準。

## 相對 Current Ratio 的變化

| 系統 | 營收模型 | EPS 方法 | EPS AE 改善 | Yield AE 變化 |
|---|---|---|---:|---:|
| Rolling | Rolling xLSTM | elastic_net_annual | +1.1497 | -0.3057 pp |
| Rolling | Rolling xLSTM | lasso_annual | +1.1496 | -0.3050 pp |
| Rolling | Rolling xLSTM + Adjustment | elastic_net_annual | +1.0643 | -0.3002 pp |
| Rolling | Rolling xLSTM + Adjustment | lasso_annual | +1.0642 | -0.2996 pp |
| Ensemble | ensemble_revenue | lasso_annual | +0.1008 | -0.2177 pp |
| Ensemble | LightGBM | lasso_annual | -0.0683 | -0.1645 pp |

說明：

- `EPS AE 改善` 為正代表 EPS 變準。
- `Yield AE 變化` 為負代表殖利率變差。
- Rolling 的 EPS 改善最大，但 yield 惡化也最大。

## 股票層級 Winner

以 `yield_mae_percent_point` 作為主要指標：

| 系統 | 營收模型 | EPS 方法 | 勝出股票數 | 勝率 |
|---|---|---|---:|---:|
| Rolling | Rolling xLSTM | current_ratio | 17 / 58 | 29.3103% |
| Ensemble | LightGBM | current_ratio | 11 / 58 | 18.9655% |
| Ensemble | ensemble_revenue | current_ratio | 10 / 58 | 17.2414% |
| Ensemble | ensemble_revenue | lasso_annual | 5 / 58 | 8.6207% |
| Rolling | Rolling xLSTM | lasso_annual | 5 / 58 | 8.6207% |
| Ensemble | LightGBM | elastic_net_annual | 3 / 58 | 5.1724% |

解讀：

- 殖利率 winner 仍然集中在 `current_ratio`。
- ML EPS layer 在部分股票有幫助，但不是整體勝出。
- 因此不能把「EPS AE 最小」當成「殖利率最佳」的替代指標。

## 代表性案例

### 6804：ML EPS 較準，但現金股利被截斷為 0

在 Rolling xLSTM 上：

- EPS AE 從 `4.4705` 降到 `1.4576`
- 但 estimated EPS 變成 `-1.3724`
- estimated cash dividend 被截斷為 `0`
- 實際現金股利為 `1.00`
- yield AE 從 `0.1981 pp` 升到 `4.4581 pp`

這代表 EPS 變準，但股利估計變差。

### 8476：ML EPS 同時改善 EPS 與 yield

在 Rolling xLSTM 上：

- EPS AE 從 `2.6080` 降到 `1.6779`
- cash dividend AE 從 `0.4877` 降到 `0.0320`
- yield AE 從 `2.3267 pp` 降到 `0.1525 pp`

這表示 ML EPS layer 不是完全沒用，而是需要更針對「股利/殖利率目標」做選擇。

## 可放入報告的結論句

保守版：

> 固定 ML EPS layer 能降低 Rolling/xLSTM 的 EPS 誤差，但該改善沒有傳導到殖利率估計。以 legacy yield diagnostic 來看，Rolling xLSTM + current_ratio 表現最好；但正式殖利率結果應改引用 time-safe dividend benchmark。

答辯版：

> 我們進一步檢查 EPS layer 改善是否會傳導至殖利率。結果顯示，ElasticNet/Lasso 雖能降低 EPS AE，但因現金股利估計還受到配息率、負 EPS 截斷與實際股利政策影響，殖利率 AE 反而上升。因此，下游目標不應只以 EPS AE 選模型，而應直接以 cash dividend AE 或 yield AE 驗證。

研究方向版：

> 後續應把 EPS layer 與 dividend layer 分開建模。EPS 模型可以用 ElasticNet/Lasso 作為 baseline，但殖利率模型應直接針對現金股利或殖利率誤差調整，不能假設 EPS 變準就會自動改善 yield。

## 建議下一步

1. 不要把 ML EPS layer 直接接入主流程殖利率輸出。
2. 保留 `Rolling xLSTM + current_ratio` 作為 legacy yield diagnostic，不作為正式 time-safe baseline。
3. 正式路徑引用 dividend layer benchmark，比較：
   - historical payout ratio
   - fixed payout fallback
   - cash dividend smoothing
   - dividend classification：是否配息
4. 用 `cash_dividend_abs_error` 或 `yield_mae_percent_point` 作為下游選模指標，而不是只看 EPS AE。
5. 對負 EPS、低 EPS、仍配息公司建立獨立處理規則。

## 輸出檔案

```text
forecast_benchmark/outputs/yield_eps_layer_benchmark/yield_eps_layer_predictions.csv
forecast_benchmark/outputs/yield_eps_layer_benchmark/yield_eps_layer_stock_accuracy.csv
forecast_benchmark/outputs/yield_eps_layer_benchmark/yield_eps_layer_overall_accuracy.csv
forecast_benchmark/outputs/yield_eps_layer_benchmark/yield_eps_layer_winner_summary.csv
forecast_benchmark/outputs/yield_eps_layer_benchmark/yield_eps_layer_improvement_vs_current.csv
forecast_benchmark/outputs/yield_eps_layer_benchmark/yield_eps_layer_error_decomposition.csv
forecast_benchmark/outputs/yield_eps_layer_benchmark/yield_eps_layer_failed_runs.csv
forecast_benchmark/outputs/yield_eps_layer_benchmark/run_config.json
```
