# 程式預測邏輯快速筆記

> 文件狀態：更新至 2026-08-05。模型邏輯以本文件與
> [`../README.md`](../README.md) 為準；實驗是否可正式引用，請查看
> [`../../docs/experiments/experiment_registry.md`](../../docs/experiments/experiment_registry.md)。

## 1. 一句話介紹

用最近 12 個月月營收資料做 Rolling LSTM，加入 KMeans 型態特徵，依 growth／cycle／decline 判斷是否套用條件式 Growth Adjustment，最後用固定 Guardrail 防止預測失控。主流程也能加入 no-cluster Rolling xLSTM，並在歷史 mLSTM-only 與新的 `mLSTM → sLSTM` Hybrid 間切換；消融、xLSTM 參數掃描與季度 target 實驗放在 batch research runner。

## 2. 整體流程

```text
月營收資料
→ 建立特徵
→ 12 個月 rolling window
→ KMeans 分群
→ Rolling LSTM
→ optional Rolling xLSTM no-cluster comparison（mLSTM-only 或 mLSTM+sLSTM Hybrid）
→ regime 判斷
→ Growth Adjustment
→ optional xLSTM Conditional Adjustment
→ 固定 Guardrail
→ 預測下一月
→ 預測完成後合併 2025 actual 做 evaluation
→ 完整 12 個月預測加總成年營收
→ availability-safe EPS / payout → 預估現金股利
→ 以 2025 月末實際股價計算殖利率回測
```

## 3. 和 Ensemble 系統的關係

- Ensemble Forecast System 是另一個獨立系統，使用 XGBoost、LightGBM、CatBoost、SeasonalQuantile 等非 LSTM 模型。
- Rolling LSTM 不 import Ensemble，也不把 Ensemble 的資料流程當成自己的前置步驟。
- 兩個系統只共用 root `data/` source files。
- 跨系統比較放在已實作的第三個獨立分析工具 `forecast_benchmark/`。
- Rolling 殖利率由 `rolling_predict_LSTM/yield_forecast.py` 自己負責，不 import Ensemble。

## 4. Rolling window

- 視窗：12 個月。
- `M1～M12 → M13`，下一次是 `M2～M13 → M14`。
- 使用前月已公布營收，不使用目標月實際值。
- 數值特徵：對數營收、月成長率、3 月動能、6 月動能。
- 月份必須連續；只要 window 或 target 中間缺一個月，該 sample 就不建立。
- 月成長率、動能、trend/cycle rolling features 會在缺月後重設，不跨 gap 串接。

## 5. KMeans

- 上升記 1，下降記 0，形成 12 個月方向向量。
- K 可設 4～8，預設 6。
- `n_init=20`、`random_state=42`。
- Cluster 轉成 one-hot 後加入 LSTM。
- Cluster 是近期型態編號，不是永久股票分類。

## 6. Sequence backbone 設定

- hidden units：48。
- Dropout：0.15。
- Dense：24 → 1。
- Adam learning rate：0.0005。
- 介面 epochs：預設 35，可設 5～100。
- max train samples：預設 40,000。
- 一般 batch：128；CUDA 最低有效 batch：4096。
- 最後一個歷史 target year 保留作 forward validation；不再隨機拆分高度重疊的 windows。
- 輸入、目標用 StandardScaler，但 scaler 只 fit 在 forward-validation 年以前的 training samples。

### LSTM

- 預設 backbone。
- 主流程永遠保留三個 LSTM rows：plain、cluster、cluster + conditional adjustment。

### xLSTM

- optional comparison path。
- 使用 `xlstm==2.0.5` 的 `xLSTMBlockStack`。
- `xlstm`：一個 mLSTM block，保留舊 D1 實驗可重現性。
- `xlstm_hybrid`：兩個 block，順序固定為 mLSTM 後接 sLSTM。
- Hybrid 的 sLSTM 使用 `backend="vanilla"`。這是 PyTorch 原生 sLSTM 計算；model 移到 CUDA 時兩個 block 都在 GPU tensor 上執行，但不編譯上游自訂 CUDA extension。
- 在 Streamlit 主流程中，xLSTM 只作為 no-cluster 比較模型，再加一個 `Rolling xLSTM + Conditional Adjustment` row。
- xLSTM 安裝在 `rolling_predict_LSTM/requirements-xlstm.txt`。
- 若本機 CUDA 環境直接 import 官方 package 觸發 `CUDA_HOME` 問題，程式會用 scoped import shim 替換未使用的 sLSTM extension loader；不會改寫 `torch.cuda.is_available`，mLSTM-only 與 native sLSTM 都仍可使用 GPU。
- `selected_params["xlstm_backbone"]` 會記錄實際架構；舊 D1.5～D1.20 證據一律是 mLSTM-only，不得重新標成 Hybrid。
- `batch_xlstm_validation_fallback.py` 可用 prior-year validation 選擇 xLSTM plain 或 balanced adjusted；這是研究輔助，不是 Streamlit 主流程預設。

## 7. 三種狀態

```text
growth_ratio > 0.65 → growth
growth_ratio < 0.40 → decline
其他               → cycle
```

- `growth_ratio`：12 個月中，上升月份的比例。
- `growth_streak`：視窗最後連續上升幾個月。
- 狀態每月重新判斷，不是永久標籤。

## 8. Growth Adjustment

成長訊號：

```text
0.5 × 3 月動能
+ 0.3 × 6 月動能
+ 0.2 × 最新月成長率
```

上修倍率：

```text
1 + alpha × 正成長訊號
```

- cluster adjusted 預設 `alpha=0.8`：控制 LSTM + Cluster 的上修強度。
- xLSTM adjusted 預設 `alpha=0.0`：D1.10 後獨立於 cluster adjusted；D1.15 後預設為 balanced decline cap，不做 growth boost。
- D1.15 後 xLSTM decline cap 使用 balanced gate：必須同時符合 `growth_ratio <= 0.35` 且 xLSTM prediction / last observed `> 1.10`；cluster adjusted 的 decline cap 保留舊行為。
- 預設必須同時符合：
  1. regime 是 growth。
  2. `growth_ratio > 0.65`。
  3. `growth_streak >= 4`。
  4. 最新月成長率 > 0。
  5. growth signal > 0。
- 所以不是把所有股票直接拉高。

若啟用 `regime_strategy`，decline regime 還會把 adjusted prediction 壓回 `last_observed_revenue` 以下。這是 decline cap，不是 growth boost。

明細欄位要分開看：

| 欄位 | 意義 |
|---|---|
| `adjustment_applied` | cluster adjusted 的 growth boost 是否生效 |
| `decline_cap_applied` | cluster adjusted 是否被 decline cap 影響 |
| `xlstm_adjustment_applied` | xLSTM adjusted 的 growth boost 是否生效 |
| `xlstm_decline_cap_applied` | xLSTM adjusted 是否被 decline cap 影響 |
| `adjustment_ratio` | cluster adjusted 相對 cluster prediction 的倍率 |
| `xlstm_adjustment_ratio` | xLSTM adjusted 相對 xLSTM plain prediction 的倍率 |

## 9. 非對稱損失

- `under_weight=2.0`。
- 低估時，平方誤差乘以 2。
- 用途：讓模型更重視低估問題。
- 設太高：可能讓整體預測偏高。

## 10. Trend + Cycle

用途：研究震盪／循環型態。

```text
trend = 12 個月移動平均
cycle = 實際營收 - trend
```

- Trend LSTM 與 Cycle LSTM 分開預測，再相加。
- 這條路徑已從 Streamlit 主流程移除，只保留在研究腳本與研究函式中。
- `trend_slope_beta=0.35`：控制上升趨勢加強幅度。
- `max_volatility_scale=2.5`：Cycle 最多放大 2.5 倍、最少縮到 0.4 倍。
- Trend boost 上限：1.35。
- 最新消融結果顯示月度 Trend + Cycle 會傷害 WMAPE，因此不作為主流程預設方法。

## 11. 季度 target 消融

用途：確認震盪股是否應該從「單月 target」改成「區間／季度 target」。

兩類實驗：

```text
MS*：月預測結果滾動加總成 3 個月營收
Q* ：直接用過去 12 個月預測未來 3 個月營收總和
```

重要限制：

- `MS*` 是 rolling-updated benchmark，3 個月內後續月份會使用已公布的較新月份資訊。
- `Q*` 是 direct 3M target，更接近真正的區間預測。
- 2025 actual 只在預測後合併計算 metrics，不進 training samples。

截至 2026-07-28 完整消融結果：

| 範圍 | Monthly-sum baseline | Direct 3M best | 結論 |
|---|---:|---:|---|
| 全體股票 WMAPE | MS00 = 25.23 | Q03 = 28.75 | direct 3M 較差 |
| cycle dominant WMAPE | MS00 = 27.24 | Q03 = 30.76 | 不支持整批震盪股改季度 target |

股票層級也不支持整體切換：

- cycle dominant 股票中，`Q00` 只有 41/1581 檔 WMAPE 贏過 `MS00`。
- cycle dominant 股票中，`Q03` 只有 38/1581 檔 WMAPE 贏過 `MS00`。

因此季度 target 目前定位是研究輔助或少數股票 fallback，不是主流程替代模型。

## 12. Guardrail

一般上限：

```text
max(最後月營收 × 5,
    最近 12 月最高營收 × 4)
```

- Dynamic guardrail 已移除；不再針對高成長標記放寬到 8 倍。

## 13. 三套門檻不要混淆

| 用途 | 條件 |
|---|---|
| regime growth | `growth_ratio > 0.65` |
| growth phase | `growth_ratio > 0.65` 且 `streak >= 4` |
| 固定 Guardrail | 最後月營收 × 5、最近 12 月最高營收 × 4 |

## 14. 主流程輸出版本

1. Rolling LSTM：只有數值序列。
2. Rolling LSTM + Cluster：加入 KMeans one-hot。
3. Rolling LSTM + Cluster + Conditional Adjustment：符合 growth 條件才成長補償；decline regime 可套用 decline cap。
4. Rolling xLSTM：optional no-cluster comparison；可選歷史 mLSTM-only 或 `mLSTM → sLSTM` Hybrid。
5. Rolling xLSTM + Conditional Adjustment：optional xLSTM post-processing row，預設 alpha 0.0，並使用 balanced decline cap，同時分開標示 growth boost 與 decline cap。

額外研究 row：

- Rolling xLSTM Validation Fallback：由 `batch_xlstm_validation_fallback.py` 產生，使用 2024 validation 在 stock 或 stock-regime scope 選 plain/adjusted，再套到 2025 預測檔；不使用 2025 actual 做選擇。

## 15. xLSTM D1 結論（歷史 mLSTM-only）

目前 xLSTM 是主流程中的 optional research comparison，不取代三個固定 LSTM rows。以下
D1.5～D1.20 結果全部使用 mLSTM-only，不能當作 Hybrid 指標。

已確認：

- xLSTM plain 對 no-cluster LSTM 有訊號；basket-30 中 observation-level MAPE 約從 LSTM plain 33.34% 降到 xLSTM plain 31.09%。
- xLSTM + Cluster 在 D1.7 不穩，暫時不放進 Streamlit 主輸出。
- xLSTM adjusted 在 basket-30 中可把 observation-level MAPE 降到約 19.16%，略優於 cluster adjusted 的 20.73%。
- D1.9 顯示這個改善主要來自 decline cap，而不是大量 growth boost；最佳 post-hoc variant `alpha_0p2_cond_on_regime_on` 的 GrowthBoostRate 約 2.01%、DeclineCapRate 約 8.05%。
- D1.11 進一步把 xLSTM post-processing 拆成 `growth_boost_only`、`decline_cap_only`、`growth_boost_and_decline_cap`，用來確認改善來源到底是 backbone、growth boost，還是 decline cap。
- D1.11 結果顯示 `decline_cap_only` 最佳：MAPE 約 19.16%、GrowthBoostRate 0%、DeclineCapRate 約 8.05%；因此 xLSTM adjusted 預設 alpha 改成 0.0，cluster adjusted 仍使用 alpha 0.8。
- D1.12 新增 `batch_xlstm_main_flow_comparison.py`，專門批次驗證 Streamlit 主流程五個 rows，不再用臨時 inline script。basket-30 結果中，xLSTM adjusted MAPE 約 19.16%，cluster adjusted MAPE 約 20.73%，股票層級 20/29 檔 xLSTM adjusted 勝出。
- D1.13 擴大到 basket-100，有效 99 檔、1 檔 `7631` 因缺 2025 rolling evaluation samples 排除。robust refresh 顯示 xLSTM adjusted 對 cluster adjusted：MAPE `66.77%` vs `72.54%`，WMAPE `16.37%` vs `20.72%`，MedianAPE `11.18%` vs `12.48%`，SMAPE `22.92%` vs `24.48%`；股票層級 MAPE 勝出 60/99 檔，WMAPE 勝出 59/99 檔。
- D1.13 同時顯示 `Rolling xLSTM` plain 的 WMAPE `16.26%` 與 DirectionAccuracy `61.13%` 是整體最佳；xLSTM adjusted 的 decline cap 把 decline regime MAPE 從 `135.72%` 拉到 `27.57%`，但 decline WMAPE 從 `13.62%` 小幅升到 `14.42%`。所以 decline cap 是修正 MAPE outlier 的工具，不是無條件改善總量誤差。
- D1.15 根據 D1.13 的 2025 monthly predictions 與 actual 做 post-hoc 掃描，並據此把 xLSTM adjusted 預設改為 balanced decline cap：`growth_ratio <= 0.35` 且 prediction / last observed `> 1.10`。這個 gate 的 cap rate 約 `7.60%`，相對舊 cap 的 `9.65%`，MAPE 約 `66.77% → 66.76%`，WMAPE 約 `16.37% → 16.29%`，DirectionAccuracy 約 `57.73% → 59.16%`。因為 gate 是看過 2025 結果後選的，這是開發證據，不是獨立 holdout。
- D1.15 main-flow basket-30 已正式重跑新預設；對同一批 29 檔，xLSTM adjusted cap rate 從 `8.05%` 降到 `6.32%`，WMAPE 從 `12.608%` 降到 `12.595%`，DirectionAccuracy 從 `61.21%` 提升到 `62.36%`，MAPE 維持約 `19.16%`。
- D1.16 main-flow basket-100 是在同一個 2025 evaluation set 重跑 D1.15 選出的新預設；有效 99 檔、1 檔 `7631` 排除。balanced xLSTM adjusted 對 cluster adjusted：MAPE `66.76%` vs `72.54%`，WMAPE `16.29%` vs `20.72%`，MedianAPE `11.18%` vs `12.48%`，SMAPE `22.90%` vs `24.48%`；股票層級 MAPE 勝出 60/99 檔，WMAPE 勝出 60/99 檔。這確認實作可重現，但不能當作看不見 2025 的獨立驗證。
- D1.17 新增 validation fallback runner。它用 target year `<= 2023` 訓練驗證模型，用 2024 validation WMAPE 為每檔股票選 `Rolling xLSTM` plain 或 balanced adjusted，再套到 2025。basket-100 中採 strict tie-to-plain，選 11 檔 adjusted、88 檔 plain，其中 13 檔因缺 2024 validation default 回 plain；2025 結果為 MAPE `67.17%`、WMAPE `16.256%`、MedianAPE `10.85%`、DirectionAccuracy `60.95%`。這與 xLSTM plain 的 WMAPE 幾乎打平且微幅較好，但改善不到 `0.001` 個百分點，所以暫不升成主流程預設。
- D1.18 把 validation fallback 延伸成 stock-regime scope，並用 WMAPE 至少改善 5 個百分點才選 adjusted。basket-100 中 158 個 stock-regime groups 只有 5 組選 adjusted，45/1119 個 2025 預測月份使用 adjusted；結果 MAPE `67.17%`、WMAPE `16.25579%`、MedianAPE `10.85%`、SMAPE `22.94%`、DirectionAccuracy `61.22%`。比 xLSTM plain 只微幅改善 WMAPE/DirectionAccuracy，仍不升成主流程預設。
- D1.19 新增 decline cap confidence runner。它不使用 2024 validation 直接選模型，而是用 time-safe score：xLSTM plain prediction / last observed、`growth_ratio`、`growth_streak`、regime。basket-100 fine scan 中 threshold `0.45` 的 WMAPE 最佳：MAE `103,843`、MAPE `67.33%`、WMAPE `16.18359%`、MedianAPE `10.97%`、SMAPE `22.77%`、DirectionAccuracy `60.59%`。這比 xLSTM plain 的 WMAPE `16.25628%` 和 fixed adjusted 的 `16.29188%` 更好，但 MAPE 仍不如 fixed adjusted。
- D1.20 新增 confidence calibration runner，用 2024 validation WMAPE 選 threshold，再套到 2025。validation 選出 `0.55`；2025 calibrated 結果為 MAE `104,193`、MAPE `67.42%`、WMAPE `16.23812%`、MedianAPE `10.85%`、SMAPE `22.78%`、DirectionAccuracy `60.95%`。這比 xLSTM plain 與 fixed adjusted 的 WMAPE 好，但 MAPE 仍輸 fixed adjusted。D1 到這裡收斂：confidence gate 是研究候選，不直接改 Streamlit 預設。
- D1.21 是首次單獨預先登記的 Hybrid basket-100 固定參數 run。100/100 檔成功；Hybrid plain 對同次 LSTM plain 的 WMAPE 為 `17.598%` vs `21.259%`，Hybrid adjusted 對同次 cluster adjusted 為 `17.448%` vs `19.359%`。adjusted 相對 Hybrid plain 的 WMAPE 只改善 `0.151` 個百分點，DirectionAccuracy 下降 `2.213` 個百分點。所有 xLSTM rows 均保留 `xlstm_hybrid` provenance。這使 Hybrid 通過預先登記的 merge recommendation gate，但結果仍是 Tier C；而且 D1.21 與 D1.16 僅重疊 13 檔，不能據此判定 Hybrid 與歷史 mLSTM-only 的直接勝負。
- D1.22 在完全相同的 100 檔、月份與設定下比較 mLSTM-only 和 Hybrid。no-cluster plain 的 Hybrid pooled WMAPE `17.598%` 略差於歷史版本 `17.166%`，但 MedianAPE、SMAPE、DirectionAccuracy 與 63/100 檔股票層級 WMAPE 比較偏向 Hybrid。Hybrid cluster plain 的 WMAPE `15.907%` 明顯低於歷史 cluster `20.823%`。預登記門檻通過，因此保留 Hybrid 預設；但不改 UI、不調 2025 參數，等待歷史 validation 與 unseen-year 驗證。

評估口徑：

- 不能只看 MAPE。低營收月份會放大 percentage error，D1.13 中已有單一個股 MAPE 退步超過 400 個百分點但 MAE 只差約 48 千元的例子。
- batch 與 Streamlit metrics 同步輸出 `MedianAPE`、`WMAPE`、`SMAPE`、`DirectionAccuracy`。判斷「是否真的更準」時，至少同時看 MAPE、WMAPE、MedianAPE 與 MAE。

仍未完成：

- 尚未使用 xlstm 的自訂 sLSTM CUDA extension；Windows 穩定路徑使用 PyTorch 原生 sLSTM backend。
- Hybrid 已完成 D1.21 fixed-parameter 大樣本實驗，但仍缺 newly-unseen-year Tier A 證據，也不可套用舊 mLSTM-only D1 指標。
- D1.22 已完成 exact-cohort backbone 比較；結果混合且仍是 target-year Tier C，不能取代 unseen-year test。
- 尚未證明 xLSTM adjusted 對全股票池、所有 metrics 都穩定優於現有 cluster adjusted 或 xLSTM plain。

## 16. 常見問題快速回答

### 為什麼用 12 個月？

涵蓋完整年度季節性，同時保留近期變化。

### KMeans 在做什麼？

把最近 12 個月的上升／下降型態分群，再把群組當成 LSTM 額外特徵。

### Cluster 和 regime 一樣嗎？

不一樣。Cluster 是無監督型態編號；regime 決定使用哪種修正。

### 有處理震盪股嗎？

有做研究實驗，但沒有放進主流程。月度 Trend + Cycle 會傷害 WMAPE；direct 3M quarterly target 在 cycle dominant 股票也沒有打贏 monthly-sum benchmark，所以目前不建議整批震盪股切換成季度 target。

### Growth Adjustment 是全部上修嗎？

不是，必須同時符合 growth、連續成長、最新月仍上升及正成長訊號。

### xLSTM 是完整版本嗎？

現在有兩種。`xlstm` 是舊 mLSTM-only；`xlstm_hybrid` 是一個 mLSTM block 後接一個 sLSTM block。Hybrid 的 sLSTM 使用 PyTorch 原生 backend，因此是 mLSTM+sLSTM 架構，但不是 xlstm 自訂 CUDA kernel 路徑。

### xLSTM adjusted 為什麼變準？

目前 D1.11 / D1.13 證據顯示，主要改善來自 decline regime 的 cap，讓 xLSTM 在下降狀態不要過度外推，尤其能降低低分母月份造成的 MAPE 爆炸；growth boost only 反而比 xLSTM plain 稍差。D1.15 已把 xLSTM adjusted 預設改成 balanced gate，降低 cap 對 WMAPE/DirectionAccuracy 的副作用。

### xLSTM alpha 跟 Growth Adjustment alpha 一樣嗎？

不一樣。cluster adjusted 預設 alpha 是 0.8；xLSTM adjusted 預設 alpha 是 0.0。xLSTM alpha 0.0 不是關掉 adjusted row，而是讓它預設只做 balanced decline cap、不做 growth boost。

### 5x、4x 是預測倍率嗎？

不是，是 Guardrail 上限的參考倍數。

### 有偷看 2025 嗎？

2025 actual 沒有進入神經網路 training samples；但研究開發過程曾使用 2025 結果比較架構與 post-processing。尤其 D1.15 用 2025 replay 選 balanced decline-cap gate，D1.16 又在同一 evaluation set 重跑，所以兩者只能算 target-year hindsight 的開發證據。D1.17／D1.18／D1.20 的「選擇步驟」改用 2024 validation，較 time-safe；仍需在新的未見年份或 frozen protocol 上重跑，才能升級成獨立 report-ready 證據。

### 自動調參調什麼？

AutoTune 已從主流程移除，只保留在研究腳本；它原本用 2024 驗證選 K 和 max train samples，不使用 2025。

### 目前最大問題？

震盪型態仍不穩定；但最新季度 target 消融顯示，單純改成 direct 3M target 不是解法。比較值得研究的是信心區間、少數股票 fallback，或更精細的 cycle/growth gating。

### 殖利率路徑？

Rolling UI 已加入完整 downstream 鏈。EPS/revenue ratio 只使用 cutoff 前可得且具有完整四季的歷史 EPS；payout 優先使用個股 availability-safe 歷史中位數，個股不足時才使用 cross-sectional 中位數。2025 實際現金股利與月末股價只用於 evaluation。Rolling 月預測會逐月吸收新公布營收，因此 12 個月合計屬於 rolling evaluation total，不是 1 月一次產生的固定全年預測。

### 下一步？

D1 mLSTM-only 已完成。Hybrid 應另開新實驗編號，以 frozen protocol 比較 mLSTM-only 與 mLSTM+sLSTM；在此之前不要把舊 D1 指標當作 Hybrid 成績。自訂 sLSTM CUDA kernel 若要啟用，仍需先修正 xlstm 2.0.5 在 Windows 上的編譯旗標與 batch-size 限制。

## 17. 最後一句

目前主流程的核心不是單一 LSTM，而是 Rolling window、KMeans、動態狀態判斷、條件式 Growth Adjustment、xLSTM comparison 與固定 Guardrail 的組合。
