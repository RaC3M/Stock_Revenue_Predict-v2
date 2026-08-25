# Time-safe EPS Layer Validation 結果

## 2026-07-31 狀態

本機已完成 `forecast_benchmark/outputs/report_ready_20260731_eps_layer_validation` rerun。Validation protocol 使用 prior-year evidence，仍可作為 EPS layer supporting evidence；但下列表格保留早期 cohort 的詳細分析，引用精確最新數字前應查看 rerun output 與 `experiment_registry.md`。

2026-08-05 架構註記：本文件所有 `Rolling xLSTM` rows 都來自歷史 mLSTM-only 輸入，不代表 Hybrid 表現。

目前正式結論不變：固定 ML EPS layer 可作為 EPS baseline 候選，但 2024 validation 的 stock-level / bucket-level EPS method selection 尚未穩定泛化到 2025。

## 目的

本實驗把前一輪 hindsight EPS 診斷往前推一步，檢查：

> 用 2024 validation 選出的 EPS 方法，套到 2025 test 是否真的比固定 current ratio 更好？

這一步很重要，因為前一輪 `eps_error_diagnostics` 使用 2025 實際 EPS 回看，屬於 hindsight label。這次改用較嚴格的流程：

1. 用 2024 作為 EPS layer validation。
2. 根據 2024 validation 選方法。
3. 將固定後的選法套到 2025 revenue benchmark predictions。

## 執行設定

執行指令：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.eps_layer_validation --output-dir forecast_benchmark\outputs\eps_layer_validation
```

歷史輸入資料：

```text
forecast_benchmark/outputs/basket_100/comparable_monthly_predictions.csv
```

設定：

- Validation year：`2024`
- Test year：`2025`
- Test 股票池：`82` 檔
- Test 有效 EPS 股票數：`81`
- 營收模型：
  - `Rolling xLSTM`
  - `Rolling xLSTM + Conditional Adjustment`
  - `ensemble_revenue`
  - `LightGBM`
- EPS 方法：
  - `current_ratio`
  - `seasonal_quarter_median`
  - `ridge_annual`
  - `lasso_annual`
  - `elastic_net_annual`

Validation 階段使用 2024 真實營收作為 EPS layer 輸入，因此它評估的是 EPS 轉換層本身，不評估月營收預測。

## 2024 Validation 結果

先看固定 EPS 方法在 2024 validation 的表現：

| EPS 方法 | 有效股票數 | 平均 EPS AE | 中位 EPS AE |
|---|---:|---:|---:|
| lasso_annual | 81 | 2.7990 | 1.2680 |
| ridge_annual | 81 | 2.7991 | 1.2681 |
| elastic_net_annual | 81 | 2.7992 | 1.2696 |
| current_ratio | 81 | 4.5314 | 1.0973 |
| seasonal_quarter_median | 81 | 5.7087 | 1.2340 |

解讀：

- 2024 validation 中，平均 EPS AE 最好的固定方法是 `lasso_annual`。
- Ridge、Lasso、ElasticNet 三者非常接近。
- `current_ratio` 的中位數不差，但平均誤差較高，代表少數股票會爆得很嚴重。

## 2024 Bucket Rule

依照 2024 validation，各 ratio stability bucket 選出的 EPS 方法如下：

| Ratio bucket | 2024 選出方法 | 股票數 | 平均 validation EPS AE | 中位 validation EPS AE |
|---|---|---:|---:|---:|
| insufficient_history | elastic_net_annual | 9 | 2.3152 | 1.4846 |
| moderate_ratio | lasso_annual | 18 | 2.0154 | 1.2115 |
| stable_ratio | current_ratio | 28 | 2.0269 | 0.7091 |
| unstable_ratio | ridge_annual | 26 | 4.0953 | 1.2879 |

這個規則看起來合理：穩定 ratio 用 current ratio，不穩定或歷史不足則偏向 ML EPS layer。

但重點是它套到 2025 後是否有效。

## 2025 Test 結果

非 hindsight 策略中，平均 EPS AE 最低的前幾名如下：

| 系統 | 營收模型 | EPS selection strategy | 平均 EPS AE | 中位 EPS AE |
|---|---|---|---:|---:|
| Ensemble | LightGBM | fixed_current_ratio | 2.8363 | 1.1918 |
| Ensemble | ensemble_revenue | fixed_seasonal_quarter_median | 2.8870 | 1.1560 |
| Rolling | Rolling xLSTM | fixed_elastic_net_annual | 2.8922 | 1.4860 |
| Rolling | Rolling xLSTM + Adjustment | fixed_elastic_net_annual | 2.8923 | 1.4860 |
| Rolling | Rolling xLSTM | fixed_lasso_annual | 2.8923 | 1.4847 |
| Rolling | Rolling xLSTM + Adjustment | fixed_lasso_annual | 2.8924 | 1.4847 |
| Rolling | Rolling xLSTM | fixed_ridge_annual | 2.8928 | 1.4845 |
| Rolling | Rolling xLSTM + Adjustment | fixed_ridge_annual | 2.8929 | 1.4845 |

結果很清楚：

- 最好的非 hindsight 組合仍是 `LightGBM + fixed_current_ratio`，平均 EPS AE 為 `2.8363`。
- Rolling/xLSTM 接固定 ML EPS layer 後，EPS AE 大幅低於 Rolling 接 current ratio。
- 但 2024 選出的 stock-dependent 或 bucket-dependent selection 沒有打敗固定 ML 方法。

## Time-safe Selection 是否有效？

以 2025 test 平均 EPS AE 來看：

| 系統 | 營收模型 | Strategy | 平均 EPS AE |
|---|---|---|---:|
| Ensemble | ensemble_revenue | stock_validation_best | 3.0607 |
| Ensemble | ensemble_revenue | ratio_bucket_validation_best | 3.0944 |
| Ensemble | LightGBM | ratio_bucket_validation_best | 3.1014 |
| Rolling | Rolling xLSTM + Adjustment | ratio_bucket_validation_best | 3.1391 |
| Rolling | Rolling xLSTM | ratio_bucket_validation_best | 3.1397 |
| Ensemble | LightGBM | stock_validation_best | 3.1530 |
| Rolling | Rolling xLSTM + Adjustment | stock_validation_best | 3.3068 |
| Rolling | Rolling xLSTM | stock_validation_best | 3.4258 |

解讀：

- `stock_validation_best` 明顯過擬合 2024，套到 2025 後表現變差。
- `ratio_bucket_validation_best` 比 stock-level selection 穩定，但仍輸給固定 ML 方法。
- 因此目前不能宣稱「依股票選 EPS 方法」已經成功。

## 對 Rolling/xLSTM 的影響

如果只和 Rolling 自己的 current ratio baseline 比，固定 ML EPS layer 有明顯改善：

| 營收模型 | Strategy | 平均 EPS AE | 相對 current ratio 改善 |
|---|---|---:|---:|
| Rolling xLSTM | fixed_elastic_net_annual | 2.8922 | 28.4445% |
| Rolling xLSTM | fixed_lasso_annual | 2.8923 | 28.4421% |
| Rolling xLSTM | fixed_ridge_annual | 2.8928 | 28.4297% |
| Rolling xLSTM + Adjustment | fixed_elastic_net_annual | 2.8923 | 26.8994% |
| Rolling xLSTM + Adjustment | fixed_lasso_annual | 2.8924 | 26.8968% |
| Rolling xLSTM + Adjustment | fixed_ridge_annual | 2.8929 | 26.8842% |

這是本實驗最實用的發現：

> Rolling/xLSTM 不應再接 current ratio 作為主要 EPS baseline，而應優先接固定 ML EPS layer。

更新：後續 `yield_eps_layer_benchmark` 顯示，固定 ML EPS layer 雖然改善 EPS AE，但沒有改善殖利率 AE。因此這句話只適用於 EPS baseline，不適用於殖利率主流程。

## Oracle 上限

`oracle_2025_best_method` 使用 2025 實際 EPS 回看每檔股票最佳 EPS 方法，只能作為上限，不可部署：

| 系統 | 營收模型 | Oracle best EPS AE |
|---|---|---:|
| Rolling | Rolling xLSTM | 1.8924 |
| Rolling | Rolling xLSTM + Adjustment | 1.9158 |
| Ensemble | LightGBM | 1.9614 |
| Ensemble | ensemble_revenue | 2.0656 |

解讀：

- 如果能正確選到每檔股票的 EPS 方法，確實有很大改善空間。
- 但 2024 validation 的 stock-level selection 沒有成功泛化，代表 selection rule 需要更穩健，不能只看前一年哪個方法贏。

## 可放入報告的結論句

保守版：

> Time-safe EPS layer validation 顯示，固定 ML EPS layer 能顯著改善 Rolling/xLSTM 的 EPS 估計；但依 2024 validation 選出的 stock-dependent method selection 尚未在 2025 test 中泛化，因此目前不應將其作為主流程。

答辯版：

> 我們進一步將 hindsight EPS 診斷轉為 time-safe validation。結果顯示，2024 validation 最佳方法套到 2025 後並未穩定勝出，表示 stock-level EPS method selection 仍有過擬合風險。不過，固定 Ridge/Lasso/ElasticNet EPS layer 相較 current ratio 明顯降低 Rolling/xLSTM 的 EPS 誤差，支持後續將 EPS 轉換層獨立於月營收模型進行建模。

研究方向版：

> 後續不應直接做個股最佳方法查表，而應建立更穩健的 EPS layer。短期可先採固定 ElasticNet/Lasso 作為 Rolling/xLSTM 的 EPS baseline；中期再測試產業別模型、rolling validation、多年 validation average，以及使用毛利率或營益率資料的 EPS 模型。

## 建議下一步

1. 不要把 `stock_validation_best` 放進主流程，它目前是 2024 overfit。
2. 若目標是 EPS，優先把 downstream EPS baseline 從 `current_ratio` 改測固定 `elastic_net_annual` 或 `lasso_annual`。
3. 若目標是殖利率，不要直接套用固定 ML EPS layer；應先建立 dividend layer benchmark。
4. 下一輪可以做 `rolling multi-year EPS validation`，例如 2022、2023、2024 三個 validation fold，而不是只用 2024。
5. 若資料允許，補入毛利率、營益率、業外損益、股本等 EPS 直接特徵，因為 EPS/revenue ratio 本身不穩。

## 輸出檔案

```text
forecast_benchmark/outputs/eps_layer_validation/validation_eps_stock_accuracy.csv
forecast_benchmark/outputs/eps_layer_validation/validation_ratio_stability.csv
forecast_benchmark/outputs/eps_layer_validation/validation_bucket_method_scores.csv
forecast_benchmark/outputs/eps_layer_validation/validation_bucket_method_selection.csv
forecast_benchmark/outputs/eps_layer_validation/validation_stock_method_selection.csv
forecast_benchmark/outputs/eps_layer_validation/test_all_method_stock_accuracy.csv
forecast_benchmark/outputs/eps_layer_validation/test_ratio_stability.csv
forecast_benchmark/outputs/eps_layer_validation/test_selected_stock_accuracy.csv
forecast_benchmark/outputs/eps_layer_validation/test_strategy_overall_accuracy.csv
forecast_benchmark/outputs/eps_layer_validation/test_strategy_winner_summary.csv
forecast_benchmark/outputs/eps_layer_validation/test_strategy_improvement_vs_current.csv
forecast_benchmark/outputs/eps_layer_validation/run_config.json
```
