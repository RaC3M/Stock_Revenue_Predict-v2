# EPS 轉換層 Benchmark 結果

## 2026-07-31 狀態

本機已完成 `forecast_benchmark/outputs/report_ready_20260731_eps` rerun；但下列表格主要保留資料遷移前 `basket_100` 的詳細歷史分析，不能冒充目前 exact-cohort code 的全新結果。最新 evidence status 請引用 `experiment_registry.md`。

2026-08-05 架構註記：本文件所有 `Rolling xLSTM` rows 都來自歷史 mLSTM-only 輸入，不是目前 Streamlit 預設的 Hybrid 證據。

目前可保留的正式結論是方法論層級：月營收更準不保證 EPS 更準，EPS/revenue ratio 本身需要獨立驗證；`oracle_current_ratio` 只能作為 hindsight diagnostic。

## 目的

本實驗用來拆解一個關鍵問題：

> Rolling/xLSTM 的月營收預測比較準，是否代表轉成 EPS 後也一定比較準？

答案目前是：不一定。

Rolling/xLSTM 的年營收誤差明顯低於 ensemble，但接上不同 EPS 估算法後，EPS 誤差的 winner 會改變。這代表下游 EPS 轉換層本身是獨立瓶頸，不能只靠繼續調 Rolling LSTM 參數解決。

## 執行設定

歷史輸入資料：

```text
forecast_benchmark/outputs/basket_100/comparable_monthly_predictions.csv
```

執行指令：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.eps_benchmark --output-dir forecast_benchmark\outputs\eps_benchmark
```

比較設定：

- 預測年份：`2025`
- 共同股票池：`82` 檔
- 有效 EPS 股票數：`81`
- EPS 百分比誤差有效股票數：`80`
- EPS 百分比誤差門檻：`abs(actual EPS) >= 0.01`
- 失敗執行數：`0`

比較營收模型：

- `Rolling xLSTM`
- `Rolling xLSTM + Conditional Adjustment`
- `ensemble_revenue`
- `LightGBM`

比較 EPS 方法：

- `current_ratio`：目前系統使用的年營收 x 近三年 EPS/revenue ratio 中位數。
- `seasonal_quarter_median`：每季營收 x 歷史同季 EPS/revenue ratio 中位數。
- `ridge_annual`：用歷史年營收/EPS 特徵訓練 Ridge。
- `lasso_annual`：用歷史年營收/EPS 特徵訓練 Lasso。
- `elastic_net_annual`：用歷史年營收/EPS 特徵訓練 ElasticNet。
- `oracle_current_ratio`：用真實 2025 年營收接 current ratio，只作診斷，不列入 winner。

## Overall 結果

| 系統 | 營收模型 | EPS 方法 | 平均年營收 APE | 平均 EPS AE | 中位 EPS AE | EPS 低估率 |
|---|---|---|---:|---:|---:|---:|
| Ensemble | LightGBM | current_ratio | 69.2972% | 2.8363 | 1.1918 | 44.4444% |
| Ensemble | ensemble_revenue | seasonal_quarter_median | 69.1742% | 2.8870 | 1.1560 | 39.5062% |
| Rolling | Rolling xLSTM | elastic_net_annual | 28.3410% | 2.8922 | 1.4860 | 23.4568% |
| Rolling | Rolling xLSTM + Adjustment | elastic_net_annual | 17.7022% | 2.8923 | 1.4860 | 23.4568% |
| Rolling | Rolling xLSTM | lasso_annual | 28.3410% | 2.8923 | 1.4847 | 23.4568% |
| Rolling | Rolling xLSTM + Adjustment | lasso_annual | 17.7022% | 2.8924 | 1.4847 | 23.4568% |
| Rolling | Rolling xLSTM | ridge_annual | 28.3410% | 2.8928 | 1.4845 | 23.4568% |
| Rolling | Rolling xLSTM + Adjustment | ridge_annual | 17.7022% | 2.8929 | 1.4845 | 23.4568% |
| Ensemble | LightGBM | seasonal_quarter_median | 69.2972% | 2.8949 | 1.1469 | 41.9753% |
| Ensemble | LightGBM | elastic_net_annual | 69.2972% | 2.9045 | 1.4975 | 24.6914% |
| Ensemble | LightGBM | lasso_annual | 69.2972% | 2.9046 | 1.4952 | 24.6914% |
| Ensemble | ensemble_revenue | elastic_net_annual | 69.1742% | 2.9096 | 1.4979 | 24.6914% |
| Ensemble | ensemble_revenue | current_ratio | 69.1742% | 3.0105 | 1.1677 | 41.9753% |
| Rolling | Rolling xLSTM + Adjustment | seasonal_quarter_median | 17.7022% | 3.2342 | 1.0480 | 44.4444% |
| Rolling | Rolling xLSTM | seasonal_quarter_median | 28.3410% | 3.3428 | 1.0979 | 44.4444% |
| Rolling | Rolling xLSTM + Adjustment | current_ratio | 17.7022% | 3.9566 | 0.9840 | 38.2716% |
| Rolling | Rolling xLSTM | current_ratio | 28.3410% | 4.0419 | 1.0303 | 39.5062% |
| Oracle | actual_revenue | oracle_current_ratio | 0.0000% | 4.1861 | 0.9893 | 43.2099% |

## 重點解讀

第一，Rolling/xLSTM 的營收優勢沒有穩定傳導到 EPS。

- Rolling xLSTM + Adjustment 的平均年營收 APE 是 `17.7022%`，明顯低於 ensemble。
- 但若接目前的 `current_ratio`，平均 EPS AE 是 `3.9566`。
- LightGBM 雖然平均年營收 APE 高達 `69.2972%`，但接 `current_ratio` 的平均 EPS AE 反而最低，為 `2.8363`。

第二，`current_ratio` 對不同營收模型的反應很不穩定。

- 同樣是 `current_ratio`，LightGBM 的 EPS AE 是 `2.8363`。
- Rolling xLSTM 的 EPS AE 是 `4.0419`。
- 這表示「營收比較接近真實值」不保證乘上歷史 EPS/revenue ratio 後也比較接近真實 EPS。

第三，Ridge/Lasso/ElasticNet 讓不同營收模型之間的 EPS 誤差變得接近。

- Rolling xLSTM + ElasticNet：平均 EPS AE `2.8922`
- Rolling xLSTM + Lasso：平均 EPS AE `2.8923`
- Rolling xLSTM + Adjustment + Lasso：平均 EPS AE `2.8924`
- LightGBM + Lasso：平均 EPS AE `2.9046`
- ensemble_revenue + Lasso：平均 EPS AE `2.9097`

這代表 ML EPS 轉換層可能有用，但目前 Ridge/Lasso/ElasticNet 太簡單，彼此差距很小。

第四，Oracle 診斷顯示 EPS 公式本身有明顯瓶頸。

`oracle_current_ratio` 使用真實 2025 年營收，因此年營收誤差是 `0.0000%`。但它的平均 EPS AE 仍然是 `4.1861`，比 LightGBM + current_ratio 更差。

這表示問題不只是營收預測，還包括：

- EPS/revenue ratio 不穩定。
- 毛利率、營益率、業外損益、股本變動等因素沒有被建模。
- 只用歷史 EPS/revenue ratio 可能不足以描述 2025 EPS。

## 股票層級 Winner

以 `eps_abs_error` 作為主要指標，排除 oracle：

| 系統 | 營收模型 | EPS 方法 | 勝出股票數 | 勝率 |
|---|---|---|---:|---:|
| Ensemble | LightGBM | current_ratio | 14 / 81 | 17.2840% |
| Rolling | Rolling xLSTM | seasonal_quarter_median | 12 / 81 | 14.8148% |
| Rolling | Rolling xLSTM | elastic_net_annual | 8 / 81 | 9.8765% |
| Ensemble | ensemble_revenue | seasonal_quarter_median | 7 / 81 | 8.6420% |
| Ensemble | LightGBM | seasonal_quarter_median | 7 / 81 | 8.6420% |
| Rolling | Rolling xLSTM | ridge_annual | 5 / 81 | 6.1728% |
| Ensemble | LightGBM | lasso_annual | 5 / 81 | 6.1728% |

解讀：

- 沒有任何單一 EPS 方法壓倒性勝出。
- LightGBM + current_ratio 是目前最多股票勝出的組合。
- Rolling xLSTM + seasonal_quarter_median 和 Rolling xLSTM + ElasticNet 仍有競爭力。
- 未來比較適合走「stock-dependent EPS method selection」，而不是固定所有股票用同一個 EPS 方法。

## 可放入報告的結論句

保守版：

> 雖然 Rolling/xLSTM 在月營收與年營收誤差上優於 ensemble baseline，但 EPS benchmark 顯示此優勢不會自動傳導至 EPS 估計。這表示 EPS 轉換層本身需要獨立建模與驗證。

答辯版：

> 我們原本假設較準的營收預測會帶來較準的 EPS 與殖利率估計，但實驗結果顯示這個假設只部分成立。Rolling/xLSTM 改善了營收與殖利率平均誤差，但 EPS 估計受 EPS/revenue ratio 穩定性影響很大，因此下一步應建立專門的 EPS 估計 benchmark，而不是只繼續調整營收模型。

研究方向版：

> 後續研究應將模型拆為兩層：第一層預測月營收，第二層預測 EPS 或 EPS/revenue ratio。月營收模型仍以 Rolling/xLSTM 作為主線，但 EPS 層應比較 current ratio、seasonal ratio、Ridge/Lasso/ElasticNet，以及可能的產業分群或股票特定 fallback。

## 建議下一步

1. 不要再只用「營收預測更準」作為 EPS/殖利率會更準的論點。
2. 在報告中把 Rolling/xLSTM 定位為月營收主模型，把 ensemble 保留為 baseline。
3. 新增 EPS 轉換層小節，呈現 current ratio、seasonal quarter、Ridge/Lasso/ElasticNet 的比較。
4. 下一輪實驗優先做 EPS 層，而不是繼續盲目調 Rolling LSTM 參數。
5. 可嘗試 `ElasticNet`、產業別 Ridge、或依股票歷史穩定度選擇 EPS 方法。

## 輸出檔案

```text
forecast_benchmark/outputs/eps_benchmark/eps_predictions.csv
forecast_benchmark/outputs/eps_benchmark/eps_overall_accuracy.csv
forecast_benchmark/outputs/eps_benchmark/eps_stock_accuracy.csv
forecast_benchmark/outputs/eps_benchmark/eps_method_winner_summary.csv
forecast_benchmark/outputs/eps_benchmark/eps_error_decomposition.csv
forecast_benchmark/outputs/eps_benchmark/eps_failed_runs.csv
forecast_benchmark/outputs/eps_benchmark/run_config.json
```
