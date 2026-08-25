# Direct Dividend Error Diagnostics

> 2026-07-31 狀態：最新本機 rerun 為
> `forecast_benchmark/outputs/report_ready_20260731_direct_dividend_diagnostics`。本文件用於解釋
> target-year wins/losses，屬 supporting diagnosis，不是新的選模規則。

## 目的

本診斷接在 `direct_dividend_model_benchmark` 後面，回答：

```text
Direct dividend bucket strategy 為什麼贏？
又在哪些股票或 bucket 上輸？
錯誤主要來自「是否配息」classification，還是「配息金額」amount regression？
```

診斷腳本：

```text
forecast_benchmark/direct_dividend_error_diagnostics.py
```

輸出資料夾：

```text
forecast_benchmark/outputs/direct_dividend_error_diagnostics
```

## 執行指令

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.direct_dividend_error_diagnostics --direct-benchmark-dir forecast_benchmark\outputs\direct_dividend_model_benchmark --output-dir forecast_benchmark\outputs\direct_dividend_error_diagnostics
```

## 對照方法

本診斷比較三個方法：

| 類型 | 方法 |
|---|---|
| Direct bucket strategy | `bucket_validation_best` |
| Direct global method | `direct_hurdle_ridge_t060` |
| Baseline | `LightGBM + current_ratio + announcement_safe_payout_ratio` |

Direct bucket strategy 仍使用 time-safe features，不使用 target-year 除息資訊。

## 整體診斷摘要

股票層級比較：

| 結果 | 股票數 | 說明 |
|---|---:|---|
| improved vs baseline | 28 | Direct bucket cash AE 小於 baseline |
| worse vs baseline | 34 | Direct bucket cash AE 大於 baseline |
| tied | 6 | 兩者 cash AE 相同 |
| missing metric | 14 | 缺少可比較現金股利 metric |

在可比較的 `68` 檔中：

```text
28 improved
34 worse
6 tied
```

但平均效果仍小幅改善：

| 指標 | 平均改善 |
|---|---:|
| cash dividend AE improvement vs baseline | 0.0080 |
| yield MAE improvement vs baseline | 0.0361 pp |

解讀：

- Direct bucket strategy 的平均改善不大，屬於「整體小贏」。
- 它不是多數股票都贏，而是部分股票改善幅度較大，抵消了較多小幅變差的股票。
- 因此報告中不應說 direct dividend model 全面優於 baseline，而應說它在 time-safe 條件下提供小幅整體改善。

## Bucket-level 診斷

| bucket | stock count | valid cash stocks | fallback stocks | cash AE improvement vs baseline | improved | worse |
|---|---:|---:|---:|---:|---:|---:|
| `paid_high|history_sparse|latest_positive` | 6 | 6 | 0 | 0.5182 | 3 | 2 |
| `paid_no_history|history_none|latest_missing` | 18 | 4 | 0 | 0.2110 | 3 | 1 |
| `paid_mixed|history_enough|latest_zero` | 2 | 2 | 2 | 0.1708 | 2 | 0 |
| `paid_high|history_enough|latest_positive` | 54 | 54 | 0 | -0.0243 | 20 | 29 |
| `paid_mixed|history_enough|latest_positive` | 1 | 1 | 1 | -0.9519 | 0 | 1 |
| `paid_mixed|history_sparse|latest_positive` | 1 | 1 | 1 | -1.4875 | 0 | 1 |

解讀：

- 最大的穩定改善來自 `paid_high|history_sparse|latest_positive`，平均 cash AE 改善 `0.5182`。
- `paid_no_history|history_none|latest_missing` 也有改善，但只有 `4` 檔有可比較 cash metric，仍需保守解讀。
- 最大 bucket `paid_high|history_enough|latest_positive` 幾乎打平 baseline，平均 cash AE 反而小輸 `0.0243`；它是下一步 amount model 改良的主要戰場。

## 改善最多股票

| stock | bucket | direct cash AE | baseline cash AE | cash improvement |
|---|---|---:|---:|---:|
| 6629 泰金-KY | `paid_high|history_sparse|latest_positive` | 0.0000 | 4.0778 | 4.0778 |
| 5314 世紀* | `paid_no_history|history_none|latest_missing` | 0.2744 | 2.6422 | 2.3678 |
| 1102 亞泥 | `paid_high|history_enough|latest_positive` | 0.0000 | 2.2322 | 2.2322 |
| 1736 喬山 | `paid_high|history_enough|latest_positive` | 0.7883 | 2.7517 | 1.9634 |
| 3708 上緯投控 | `paid_high|history_enough|latest_positive` | 0.0361 | 1.5854 | 1.5493 |

解讀：

- Direct model 對「baseline 因 EPS/payout ratio 推出過高股利」的股票有幫助。
- 例如 1102、1593、2359 實際 cash dividend 為 0 或接近 0 時，direct hurdle 可以避免 payout ratio baseline 繼續估配息。

## 變差最多股票

| stock | bucket | direct cash AE | baseline cash AE | cash improvement |
|---|---|---:|---:|---:|
| 6757 台灣虎航-創 | `paid_no_history|history_none|latest_missing` | 6.0500 | 3.9111 | -2.1389 |
| 3260 威剛 | `paid_high|history_enough|latest_positive` | 5.6645 | 3.5768 | -2.0877 |
| 6870 騰雲 | `paid_high|history_enough|latest_positive` | 2.2196 | 0.1860 | -2.0336 |
| 2640 大車隊 | `paid_high|history_enough|latest_positive` | 2.5162 | 0.5370 | -1.9792 |
| 6756 威鋒電子 | `paid_high|history_enough|latest_positive` | 1.7996 | 0.0086 | -1.7910 |

解讀：

- 6757 是新/缺歷史型股票，direct model 因無歷史而預測不配，實際卻大額配息。
- 3260、6438 是 false positive：模型預測配息，但 2025 實際 cash dividend 為 0。
- 6870、2640、6756 屬於已知會配息附近的 amount 或 threshold 問題，baseline 反而更接近。

## Classification Error

Direct hurdle classification 結果：

| outcome | 股票數 | 比例 | 平均 cash AE | 平均 yield AE |
|---|---:|---:|---:|---:|
| correct_paid | 40 | 48.7805% | 1.0055 | 1.2312 pp |
| correct_no_dividend | 28 | 34.1463% | 0.0000 | 0.0000 pp |
| false_negative_missed_dividend | 12 | 14.6341% | 1.4525 | 2.5154 pp |
| false_positive_extra_dividend | 2 | 2.4390% | 4.8452 | 4.5768 pp |

錯誤重點：

| stock | error type | probability | estimated cash | actual cash | cash AE |
|---|---|---:|---:|---:|---:|
| 6757 台灣虎航-創 | false negative | 0.0000 | 0.0000 | 6.0500 | 6.0500 |
| 3260 威剛 | false positive | 0.8555 | 5.6645 | 0.0000 | 5.6645 |
| 6438 迅得 | false positive | 0.6999 | 4.0258 | 0.0000 | 4.0258 |
| 1472 三洋實業 | false negative | 0.3249 | 0.0000 | 3.0000 | 3.0000 |
| 6756 威鋒電子 | false negative | 0.5952 | 0.0000 | 1.7996 | 1.7996 |

解讀：

- False negative 有 `12` 檔，是目前最常見的 classification error。
- False positive 只有 `2` 檔，但平均錯誤很大，尤其 3260、6438。
- 下一步若只調 threshold，可能會減少 false negative，但也可能放大 false positive；應該用 validation 做 threshold calibration，而不是直接用 2025 test 調。

## Amount Error

在 classification 正確且實際有配息的股票中，金額錯誤最大的案例：

| stock | estimated cash | actual cash | cash error | 方向 |
|---|---:|---:|---:|---|
| 6187 萬潤 | 3.9120 | 10.1866 | -6.2746 | underestimated |
| 2228 劍麟 | 4.3607 | 9.0000 | -4.6393 | underestimated |
| 2718 晶悅 | 6.7498 | 3.0130 | 3.7368 | overestimated |
| 8499 鼎炫-KY | 4.5829 | 8.0622 | -3.4793 | underestimated |
| 2640 大車隊 | 4.4838 | 7.0000 | -2.5162 | underestimated |

解讀：

- 大額配息股票多數是 underestimated。
- 這代表 paid-only regression 對「高配息尾端」不足，可能需要加入歷史最大股利、近年股利成長、EPS/營收變化等 feature。
- 不過這一步要先用 validation 驗證，不能直接根據 2025 hindsight 改模型。

## 結論

最保守結論：

> Direct dividend bucket strategy 的改善來源不是全面性勝出，而是避免部分 baseline payout-ratio 過度估配息的錯誤。主要待改良處有兩個：第一，false negative 仍有 12 檔；第二，對高配息股票的 paid-only amount regression 容易低估。

下一步建議：

1. 優先做 classification threshold calibration，觀察 false negative / false positive tradeoff。
2. 針對 `paid_high|history_enough|latest_positive` 做 amount model 診斷，因為它是最大 bucket 且平均略輸 baseline。
3. 對新股或缺歷史股票不要硬依賴 dividend history；需要額外公告資料或跨股票特徵。

## 輸出檔案

```text
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_stock_error_comparison.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_improvement_leaders.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_regression_hotspots.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_bucket_error_summary.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_classification_outcomes.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_classification_summary.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_classification_errors.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_amount_error_hotspots.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/direct_dividend_diagnostic_summary.csv
forecast_benchmark/outputs/direct_dividend_error_diagnostics/run_config.json
```
