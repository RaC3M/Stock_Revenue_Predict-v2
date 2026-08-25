# 少量月營收資料的預測改善方法與實驗規劃

## 目標

目前單一股票約有 72～84 個月營收資料。這足以估計低參數的公式、ETS、ARIMA 與受限制的 SARIMA，但若每檔股票各自訓練 LSTM，扣除 12 個月輸入視窗後，可用樣本仍然太少。

本階段先建立可解釋的「營收調整公式」，再依序比較統計模型、模型組合與小型全域 LSTM 殘差修正。

## 改善方法優先順序

| 優先級 | 方法 | 目的 | 實作重點 |
|---|---|---|---|
| 1 | 上月與去年同月基準 | 建立最低門檻 | 所有模型都必須勝過簡單基準 |
| 2 | 營收調整公式 | 少參數、可解釋 | 去年同月、近期 YoY、目前水準與殘差修正 |
| 3 | ETS／Holt-Winters | 處理水準、阻尼趨勢與季節性 | 比較 additive、damped trend 與 log 轉換 |
| 4 | 低階 ARIMA | 處理去季節後的自相關 | 限制 p、q 不超過 2，使用 AICc 或 rolling validation |
| 5 | 受限制 SARIMA | 處理固定 12 個月季節性 | 收斂檢查、預測上限與基準 fallback |
| 6 | 統計模型組合 | 降低單一模型失效風險 | 比較簡單平均、中位數與驗證誤差權重 |
| 7 | 小型 global LSTM 殘差修正 | 學習公式未捕捉的非線性 | 不直接預測原始營收，只預測 log 殘差 |
| 8 | 模型分流 | 大型穩定股與中小型波動股使用不同模型 | 分流條件只能使用目標月份以前的資料 |

## 第一版營收調整公式

令 `y_t` 為本月營收，預測 `y_(t+1)`。

### 1. 穩健近期年增率

```text
g_t = clip(
    median(
        log((y_t + eps) / (y_(t-12) + eps)),
        log((y_(t-1) + eps) / (y_(t-13) + eps)),
        log((y_(t-2) + eps) / (y_(t-14) + eps))
    ),
    -growth_cap,
    growth_cap
)
```

使用中位數降低單月認列異常的影響，`growth_cap` 防止極端成長率直接外推。

### 2. 去年同月成長基準

```text
seasonal_growth_forecast =
    exp(log(y_(t-11) + eps) + g_t) - eps
```

`y_(t-11)` 是下個月在去年相同月份的營收。

### 3. 與本月水準做幾何加權

```text
log(base_(t+1) + eps) =
    seasonal_weight * log(seasonal_growth_forecast + eps)
    + (1 - seasonal_weight) * log(y_t + eps)
```

### 4. 指數平滑歷史殘差

```text
residual_t = log((y_t + eps) / (base_t + eps))

correction_t =
    residual_alpha * residual_t
    + (1 - residual_alpha) * correction_(t-1)
```

### 5. 最終預測

```text
forecast_(t+1) =
    exp(
        log(base_(t+1) + eps)
        + clip(residual_strength * correction_t,
               -correction_cap,
               correction_cap)
    ) - eps
```

如果連續歷史少於 15 個月，依序退回去年同月、上月營收或缺值，不假裝公式可以正常運作。

## 第一階段參數

只測少量可解釋組合：

- `seasonal_weight`：0.50、0.75、1.00
- `residual_alpha`：0.10、0.20、0.30
- `residual_strength`：0、0.50、1.00
- `growth_cap`：固定為 log(2)，約限制為 0.5～2 倍年增倍率
- `correction_cap`：固定為 0.50 log point

共 27 組。使用 2023–2024 的 one-step rolling validation 選出全市場共用參數，不使用 2025 實際值挑參數。

選擇分數：

```text
balanced_score = 0.5 * pooled_WMAPE + 0.5 * median_stock_WMAPE
```

這能避免只照顧最大型股票，也避免對每檔短序列分別調參造成過度擬合。

## 第二階段：ETS、ARIMA與SARIMA

固定使用與公式相同的 rolling-origin folds：

1. ETS：SES、Holt damped trend、Holt-Winters additive seasonality。
2. ARIMA：在 log 營收、去季節營收或公式殘差上測試低階模型。
3. SARIMA：限制候選參數，未收斂或碰到數值上限時改用公式預測。
4. 組合：先比較簡單平均與中位數，再測依 validation MASE 決定的權重。

## 第三階段：小型 LSTM 殘差修正

LSTM 不直接預測營收，目標改成：

```text
r_(t+1) = log(y_(t+1) + eps) - log(base_(t+1) + eps)
```

建議第一版：

- 所有股票共同訓練一個 global model。
- 1 層 LSTM，hidden size 8 或 16。
- 輸入最近 12 個月公式殘差、YoY、MoM 與波動度。
- 不先加入 stock ID embedding。
- 使用 Huber 或 MAE loss、weight decay 與 early stopping。
- 輸出的修正量必須 clipping，再乘回公式基準。

## 評估規範

### 資料切分

- 2019–2022：最初歷史與模型暖身。
- 2023–2024：rolling validation 與參數選擇。
- 2025：固定參數後的歷史回放。
- 2026：未來取得完整資料後，作為真正未參與開發的最終測試。

### 指標

- pooled WMAPE
- MedianAPE
- SMAPE
- MASE（以 seasonal naive 為尺度）
- Bias、低估率與方向準確率
- 每檔年度 WMAPE
- WMAPE 超過 15% 股票比例
- 模型失敗與 fallback 比例
- 依月份、產業、營收規模與波動度分組

## 驗收條件

第一版公式至少需要：

1. 在 2023–2024 validation 勝過去年同月基準。
2. 2025 WMAPE 不高於上月與去年同月基準。
3. 不出現負營收、無限值或數值飽和。
4. 至少一半股票的年度 WMAPE 優於 seasonal naive。
5. 報告需分開呈現公式正常預測、fallback 與缺值。

## 研究與實作資源

- Forecasting: Principles and Practice — Exponential smoothing  
  https://otexts.com/fpp3/expsmooth.html
- Forecasting: Principles and Practice — ARIMA  
  https://otexts.com/fpp3/arima.html
- Time series cross-validation  
  https://otexts.com/fpp3/tscv.html
- Forecast combinations  
  https://otexts.com/fpp3/combinations.html
- Hyndman & Khandakar, Automatic Time Series Forecasting  
  https://www.jstatsoft.org/index.php/jss/article/view/v027i03
- Smyl, Hybrid Exponential Smoothing and Recurrent Neural Networks  
  https://www.sciencedirect.com/science/article/pii/S0169207019301153
- M4 Competition  
  https://www.sciencedirect.com/science/article/pii/S0169207019301128
- statsmodels Exponential Smoothing  
  https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
- Hyndman & Koehler, Another Look at Measures of Forecast Accuracy  
  https://fpp.robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/

