# EPS / 殖利率傳導 Benchmark 結果

## 2026-07-31 狀態

本機已有 `forecast_benchmark/outputs/report_ready_20260731_yield` rerun，但這份 benchmark 仍保留為「月營收誤差是否會傳導到 EPS / 股利 / 殖利率」的 legacy diagnostic。它不是目前最終的 time-safe dividend-layer ranking，因此其中最低的歷史殖利率 AE 不應視為可部署正式結果。

2026-08-05 架構註記：本文件所有 `Rolling xLSTM` rows 都來自歷史 mLSTM-only 輸入，不代表 Hybrid 表現。

正式殖利率結論請優先引用：

- `docs/experiments/dividend_layer_benchmark_zh.md` 的 `time_safe_payout_ratio` / `announcement_safe_payout_ratio`
- `docs/experiments/direct_dividend_model_benchmark_zh.md`
- `docs/experiments/direct_dividend_error_diagnostics_zh.md`

目前 time-safe 版本的保守結論是：EPS layer 變準不保證 yield 變準；股利層才是下游瓶頸。Direct dividend V2 的 bucket validation selection 在 2025 test 中小幅改善現金股利與殖利率誤差，但改善幅度仍小，需繼續做 threshold calibration 與錯誤樣本分解。

## 目的

本實驗用來驗證核心假設：

> 較佳的月營收預測，是否真的能帶來較佳的 EPS、現金股利、殖利率或合理買入價估計？

歷史輸入資料來自資料遷移前跨系統 benchmark 的共同股票池：

```text
forecast_benchmark/outputs/basket_100/comparable_monthly_predictions.csv
```

執行指令：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.yield_benchmark --output-dir forecast_benchmark\outputs\yield_benchmark
```

## 比較設定

- 預測年份：`2025`
- 輸入股票數：`82`
- 比較模型：
  - `Rolling xLSTM`
  - `Rolling xLSTM + Conditional Adjustment`
  - `ensemble_revenue`
  - `LightGBM`
- 有效 EPS 股票數：`81`
- 有效現金股利/殖利率股票數：`58`
- 有效殖利率月份：`686`
- 股價門檻：`stock_price > 1.0`

說明：部分股票缺少有效現金股利或股價資料，因此殖利率比較不是 82 檔全部都有。股價小於等於 `1.0` 的月份視為異常或不可用，不納入殖利率誤差計算。

## Legacy Overall 結果

| 系統 | 模型 | 平均年營收 APE | 平均 EPS AE | 平均現金股利 AE | 平均殖利率 AE |
|---|---|---:|---:|---:|---:|
| Rolling | Rolling xLSTM | 28.3410% | 4.0419 | 0.6541 | 0.9279 pp |
| Rolling | Rolling xLSTM + Conditional Adjustment | 17.7022% | 3.9566 | 0.6585 | 0.9319 pp |
| Ensemble | ensemble_revenue | 69.1742% | 3.0105 | 0.7172 | 1.0144 pp |
| Ensemble | LightGBM | 69.2972% | 2.8363 | 0.7758 | 1.0627 pp |

重點解讀：

- 在這個 legacy diagnostic 中，Rolling/xLSTM 的年營收誤差明顯低於 ensemble。
- 在這個 legacy diagnostic 中，Rolling/xLSTM 的殖利率平均絕對誤差也低於 ensemble。
- 但 EPS 絕對誤差反而是 ensemble 較低，代表 EPS 轉換公式本身仍是下游瓶頸。
- Rolling xLSTM plain 的 legacy 殖利率平均誤差最低，為 `0.9279` 個百分點；此數字不可作為正式 time-safe 結論。
- Rolling xLSTM + Conditional Adjustment 的年營收 APE 最低，為 `17.7022%`，但殖利率誤差略高於 plain xLSTM。

## 股票層級 Winner

以 `yield_mae_percent_point` 作為主要指標：

| 系統 | 模型 | 勝出股票數 | 勝率 |
|---|---|---:|---:|
| Rolling | Rolling xLSTM | 26 / 58 | 44.8276% |
| Ensemble | LightGBM | 15 / 58 | 25.8621% |
| Ensemble | ensemble_revenue | 14 / 58 | 24.1379% |
| Rolling | Rolling xLSTM + Conditional Adjustment | 3 / 58 | 5.1724% |

解讀：

- Rolling xLSTM 是殖利率誤差的最多數 winner。
- 但 ensemble 仍在不少股票上勝出，尤其 LightGBM 和 ensemble_revenue 合計贏了 29 檔。
- 因此不能說 Rolling 在所有股票的殖利率預測都優於 ensemble。

## 可放入報告的診斷句

保守版：

> 在 legacy payout diagnostic 中，Rolling xLSTM 的平均殖利率絕對誤差為 `0.9279` 個百分點，低於 ensemble_revenue 的 `1.0144` 與 LightGBM 的 `1.0627`。但後續 dividend-layer audit 顯示舊 payout path 會使用目標年資訊，因此正式報告應改引用 time-safe dividend benchmark。

積極版：

> Rolling xLSTM 不只在月營收 benchmark 中取得較佳整體誤差，legacy 下游診斷也顯示營收優勢可部分傳導到 yield；不過正式殖利率路徑必須改用 time-safe dividend layer。

答辯版：

> 我們進一步驗證了月營收預測改善是否會傳導到殖利率。結果顯示 Rolling xLSTM 的殖利率平均絕對誤差最低，但 EPS 轉換誤差仍然明顯，因此未來的主要改良方向應該是 EPS 或配息率估計，而不只是繼續調整營收模型。

## 研究限制

- 殖利率結果只涵蓋 `58` 檔有有效現金股利與股價資料的股票。
- 股價資料中存在 `0.01` 這類異常值，已用 `stock_price > 1.0` 門檻排除。
- EPS 誤差比較使用目前資料中的 2025 EPS 欄位；若 EPS 資料未來修訂，結果會改變。
- 本實驗沿用舊 EPS/配息率估算公式；配息率路徑已被後續 dividend-layer audit 標記為 hindsight-assisted diagnostic。

## 下一步

如果要繼續提高殖利率或合理買入價準度，優先方向不應只是繼續調 Rolling LSTM 參數，而是：

1. 建立 EPS 估算子模型或季稅後淨利率模型。
2. 比較目前 `historical EPS/revenue ratio` 方法與 Lasso/Ridge/ElasticNet EPS 模型。
3. 將殖利率誤差拆成：
   - 營收預測誤差
   - EPS 轉換誤差
   - 配息率誤差
   - 股價基準誤差
4. 正式報告引用 time-safe / announcement-safe / direct dividend 結果，不再引用 legacy `current_system_payout_ratio` 作為可部署殖利率成績。
