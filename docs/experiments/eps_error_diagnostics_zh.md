# EPS 誤差診斷與股票分群結果

> 2026-07-31 狀態：最新本機 rerun 為
> `forecast_benchmark/outputs/report_ready_20260731_eps_diagnostics`。診斷會查看 target-year
> errors，因此只用來設計下一輪實驗，不可直接當作 deployable stock-level selector。

> 2026-08-05 架構註記：其中 `Rolling xLSTM` rows 使用歷史 mLSTM-only 輸入，不是 Hybrid 證據。

## 目的

本診斷接續 EPS benchmark，回答下一個問題：

> 哪些股票可以繼續使用 EPS/revenue ratio，哪些股票應該改用季節性 ratio 或 ML EPS layer？

這一步不是重新訓練月營收模型，而是拆解 EPS 誤差來源。原因是前一輪已經發現：Rolling/xLSTM 的營收預測較準，但 EPS 不一定較準，代表 EPS 轉換層是獨立瓶頸。

## 執行設定

輸入資料：

```text
forecast_benchmark/outputs/eps_benchmark/eps_stock_accuracy.csv
```

執行指令：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.eps_diagnostics --output-dir forecast_benchmark\outputs\eps_diagnostics
```

輸出資料夾：

```text
forecast_benchmark/outputs/eps_diagnostics
```

主要輸出：

- `eps_ratio_stability.csv`
- `eps_method_recommendations.csv`
- `eps_error_hotspots.csv`
- `eps_current_ratio_driver_by_model.csv`
- `eps_diagnostic_summary.csv`
- `run_config.json`

注意：`hindsight_recommended_eps_path` 使用 2025 實際 EPS 產生，是研究用 hindsight label，不是可直接部署的選股規則。下一步若要部署，必須改成 time-safe validation，例如用 2020-2024 驗證規則，再套到 2025。

## 核心發現

82 檔共同股票池中，歷史 EPS/revenue ratio 穩定度如下：

| Ratio 穩定度 | 股票數 | 佔比 |
|---|---:|---:|
| stable_ratio | 23 | 28.0488% |
| moderate_ratio | 20 | 24.3902% |
| unstable_ratio | 36 | 43.9024% |
| insufficient_history | 3 | 3.6585% |

最重要的結論是：`unstable_ratio` 有 `36 / 82` 檔，佔 `43.9024%`。這代表近半股票不適合單純使用「年營收 x 歷史 EPS/revenue ratio 中位數」作為 EPS 預測。

## 建議 EPS 路徑分布

以 2025 實際 EPS 回看，每檔股票的 hindsight 建議如下：

| 建議路徑 | 股票數 | 佔比 |
|---|---:|---:|
| test_ml_eps_layer | 23 | 28.0488% |
| test_seasonal_quarter_ratio | 16 | 19.5122% |
| stock_specific_or_ml_eps_layer | 15 | 18.2927% |
| keep_current_ratio | 14 | 17.0732% |
| stock_specific_validation | 11 | 13.4146% |
| manual_or_cross_sectional_ml | 2 | 2.4390% |
| manual_review | 1 | 1.2195% |

解讀：

- 只有 `14` 檔適合保留 `current_ratio`。
- `23` 檔顯示 ML EPS layer 值得優先測。
- `16` 檔顯示季節性 EPS/revenue ratio 值得測。
- 這支持下一步做 EPS layer，而不是只繼續調 Rolling LSTM。

## Current Ratio 誤差來源

針對所有非 oracle 的 current ratio 模型列，誤差來源拆解如下：

| 誤差來源 | 模型列數 | 佔比 |
|---|---:|---:|
| eps_ratio_formula_error | 225 | 68.5976% |
| forecast_error_offset_formula_error | 48 | 14.6341% |
| mixed_error | 27 | 8.2317% |
| revenue_forecast_error | 24 | 7.3171% |
| missing_current_ratio_result | 4 | 1.2195% |

解讀：

- 最大來源是 `eps_ratio_formula_error`，佔 `68.5976%`。
- 也就是說，多數 current ratio 錯誤不是因為月營收模型不夠準，而是 EPS/revenue ratio 公式本身不可靠。
- `forecast_error_offset_formula_error` 代表營收預測錯誤反而抵銷了 EPS ratio 公式錯誤，這會讓某些 ensemble EPS 結果看起來比 Rolling 好，但不代表 ensemble 的營收模型真的更合理。

## 最大 EPS 誤差股票

以下列出即使選用該股票 hindsight 最佳方法後，EPS 絕對誤差仍最大的股票：

| 股票 | 實際 EPS | Ratio 穩定度 | 最佳模型 | 最佳 EPS 方法 | 最佳 EPS AE | Current Ratio Driver | 建議路徑 |
|---:|---:|---|---|---|---:|---|---|
| 3708 | 35.57 | moderate_ratio | ensemble_revenue | seasonal_quarter_median | 27.9240 | eps_ratio_formula_error | stock_specific_validation |
| 2718 | 32.41 | unstable_ratio | Rolling xLSTM | seasonal_quarter_median | 17.5700 | revenue_forecast_error | test_seasonal_quarter_ratio |
| 3260 | 23.18 | stable_ratio | Rolling xLSTM | ridge_annual | 13.5107 | eps_ratio_formula_error | test_ml_eps_layer |
| 8499 | 6.34 | stable_ratio | ensemble_revenue | elastic_net_annual | 10.9932 | forecast_error_offset_formula_error | keep_current_ratio |
| 1432 | 6.03 | unstable_ratio | ensemble_revenue | seasonal_quarter_median | 5.7764 | eps_ratio_formula_error | stock_specific_or_ml_eps_layer |
| 6873 | 3.59 | insufficient_history | LightGBM | current_ratio | 4.1153 | forecast_error_offset_formula_error | manual_or_cross_sectional_ml |
| 8462 | 8.49 | unstable_ratio | Rolling xLSTM | ridge_annual | 3.3464 | eps_ratio_formula_error | test_ml_eps_layer |
| 1472 | 11.50 | unstable_ratio | Rolling xLSTM | current_ratio | 3.2548 | eps_ratio_formula_error | stock_specific_or_ml_eps_layer |
| 6438 | 5.24 | stable_ratio | LightGBM | elastic_net_annual | 2.5460 | forecast_error_offset_formula_error | test_ml_eps_layer |
| 2340 | -2.88 | unstable_ratio | ensemble_revenue | lasso_annual | 2.4379 | eps_ratio_formula_error | test_ml_eps_layer |

這張表的重點不是找單一最好模型，而是指出：即使用目前所有 EPS 方法挑最好的，一些股票仍然很難估，尤其 `3708`、`2718`、`3260`。

## 代表性異常案例

### 3708：公式層問題明顯

- 實際 EPS：`35.57`
- ratio 穩定度：`moderate_ratio`
- 最佳方法：`ensemble_revenue + seasonal_quarter_median`
- 最佳 EPS AE：`27.9240`
- oracle current ratio EPS AE：`30.1304`

解讀：即使用真實營收，current ratio 仍然錯很多，因此主要問題是 EPS/revenue ratio 公式沒有捕捉 2025 EPS 結構。

### 2718：營收預測與 ratio 都有問題

- 實際 EPS：`32.41`
- ratio 穩定度：`unstable_ratio`
- 最佳方法：`Rolling xLSTM + seasonal_quarter_median`
- 最佳 EPS AE：`17.5700`
- current ratio driver：`revenue_forecast_error`

解讀：這檔的歷史 ratio 很不穩，且 current ratio 下的 EPS 誤差受到營收預測誤差影響較大。季節性方法目前相對好，但仍不足。

### 5314：模型錯誤可能抵銷公式錯誤

- 實際 EPS：`2.67`
- ratio 穩定度：`unstable_ratio`
- Rolling xLSTM + current_ratio EPS AE：`60.3661`
- LightGBM + current_ratio EPS AE：`2.3685`
- oracle current ratio EPS AE：`73.4495`

解讀：LightGBM 在 EPS 上看起來比較準，並不是因為 current ratio 公式可靠，而是因為營收預測錯誤剛好抵銷公式錯誤。這種情況不能作為可部署規則。

### 4950：接近零 EPS 會讓比例法非常危險

- 實際 EPS：`-0.36`
- ratio 穩定度：`moderate_ratio`
- Rolling xLSTM + current_ratio EPS AE：`43.8635`
- ensemble_revenue + seasonal_quarter_median EPS AE：`0.1155`
- oracle current ratio EPS AE：`58.2869`

解讀：實際 EPS 接近 0 或轉虧損時，ratio 法非常容易爆。這類股票應避免只用 EPS/revenue ratio，並且 EPS 百分比誤差也不適合作為主要指標。

## 對研究方向的影響

這份診斷支持以下決策：

1. Rolling/xLSTM 仍適合作為月營收主模型，因為它在營收 benchmark 中勝出。
2. EPS/殖利率下游不應直接假設「營收越準，EPS 越準」。
3. current ratio 只能作為 baseline，不能作為唯一 EPS 估計方法。
4. 下一步應建立 time-safe EPS method selection，而不是繼續單純 grid search Rolling LSTM。

## 建議下一步

最合理的下一個實驗是：

> 建立 time-safe EPS layer validation。

更新：單年 `2024 validation -> 2025 test` 已完成，結果整理於 `docs/experiments/eps_layer_validation_zh.md`。該實驗顯示固定 ML EPS layer 對 Rolling/xLSTM 有幫助，但 stock-level method selection 尚未泛化。

做法：

1. 用 2020-2023 的歷史資料產生每檔股票的 ratio 穩定度與方法選擇規則。
2. 用 2024 作 validation，測試 `current_ratio`、`seasonal_quarter_median`、`ridge_annual`、`lasso_annual`、`elastic_net_annual`。
3. 固定規則後，再套到 2025。
4. 比較固定方法與 stock-dependent method selection 是否能降低 EPS AE。
5. 下一輪應改為 multi-year validation，降低只用 2024 過擬合的風險。

這樣才能把目前的 hindsight 診斷轉成可答辯、可驗證的研究方法。
