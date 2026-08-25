# Rolling LSTM 分群營收預測系統

獨立的 Streamlit 研究系統，使用 12 個月 rolling sequences、KMeans pattern clusters 與 PyTorch LSTM 預測台灣股票月營收，並透過自己的薄 adapter 使用 root `financial_forecast/` 估算年營收、EPS、現金股利與殖利率。
主流程也有 optional xLSTM comparison，可在歷史 mLSTM-only D1 路徑與新的 `mLSTM → sLSTM` Hybrid 路徑間切換，並使用相同 rolling samples 比較。

## Method

1. 將月營收轉為 rolling growth-direction patterns。
2. 使用 KMeans 建立非語意化的 pattern cluster ID。
3. 比較無 cluster 與加入 cluster one-hot 的 Rolling LSTM。
4. 每月以過去資料判斷 growth、cycle 或 decline regime。
5. 依設定套用 Growth Adjustment、非對稱 loss 與固定 guardrail。
6. 使用 2025 實際營收評估；2025 actual 只在預測後合併計算 metrics，不進訓練特徵或選參數。
7. 若選擇 xLSTM，可使用 mLSTM-only `xLSTMBlockStack`，或依序堆疊一個 mLSTM block 與一個 sLSTM block 的 Hybrid；舊 D1 證據仍固定代表 mLSTM-only。

Cluster ID 不是永久股票類型；regime 也會隨月份改變。

## Main Outputs

Streamlit 主流程目前保留三個 LSTM 月營收預測版本，並可加入兩個 xLSTM 比較模型：

1. `Rolling LSTM`
2. `Rolling LSTM + Cluster`
3. `Rolling LSTM + Cluster + Conditional Adjustment`
4. `Rolling xLSTM`
5. `Rolling xLSTM + Conditional Adjustment`

每個具有完整 12 個月預測的 row 都會另外產生：

- 預測年營收
- availability-safe 預估 EPS
- availability-safe 歷史 payout ratio
- 預估每股現金股利
- 以 target-year 月末收盤價計算的殖利率回測

系統介面位於 `yield_forecast.py`，共用公式位於 root `financial_forecast/`，且不得 import `ensemble_forecast/`。EPS 與 payout 預設只使用
`2025-01-10` 前可得資料；2025 實際現金股利與月末股價只在估算完成後作為 evaluation
evidence。UI 分開顯示 cutoff 當下真實價格的可部署殖利率，以及 target-year 月末真實價格的回測殖利率；後者不是股價預測。若某模型缺少完整 12 個月，系統會標為
unavailable，不會將部分月份直接當成年營收。

`Rolling xLSTM` 是 no-cluster comparison；實際架構記錄在 `selected_params["xlstm_backbone"]`。`xlstm` 代表舊 mLSTM-only，`xlstm_hybrid` 代表 `mLSTM → sLSTM` Hybrid。既有 D1.5～D1.20 結果全是 mLSTM-only，不可改稱 Hybrid 結果。
`Rolling xLSTM + Conditional Adjustment` 沿用同一套 time-safe Growth Adjustment gates，不使用 2025 target actual 做修正；D1.10 後它有獨立 alpha，D1.11 後預設為 `0.0`，代表預設保留 decline cap、不做 growth boost。D1.15 後 xLSTM decline cap 改成 balanced gate：`growth_ratio <= 0.35` 且 xLSTM 預測高於 `last_observed_revenue * 1.10` 才 cap。cluster adjusted 的 alpha 仍預設為 `0.8`，cluster decline cap 也保留舊行為。明細欄位會分開標出 growth boost 與 decline cap 是否生效。

## xLSTM Current Status

目前 xLSTM 是 Rolling LSTM 系統內的 optional research path：

- package：`xlstm==2.0.5`
- backbone：可選 mLSTM-only，或 `mLSTM → sLSTM` Hybrid `xLSTMBlockStack`
- Hybrid 結構：block 0 為 mLSTM、block 1 為 sLSTM
- sLSTM backend：`vanilla`（PyTorch 原生運算；整個 model 移到 CUDA 時仍在 GPU 執行）
- 主流程位置：Streamlit `xLSTM 架構` selector + `加入 Rolling xLSTM 比較` checkbox
- 輸出 rows：`Rolling xLSTM`、`Rolling xLSTM + Conditional Adjustment`
- adjustment：xLSTM adjusted 有獨立 alpha，預設 `0.0`
- decline cap：xLSTM adjusted 預設使用 balanced gate，`growth_ratio <= 0.35` 且 prediction / last observed `> 1.10`

Windows 預設不啟用 xlstm 2.0.5 的自訂 sLSTM CUDA extension，因為上游編譯旗標包含 Windows `nvcc` 不接受的組合參數。Hybrid 改用功能等價的 PyTorch 原生 sLSTM backend，避開即時編譯，同時保留 CUDA tensor 執行。缺少 CUDA Toolkit 時，Rolling loader 也只會替換未使用的 extension loader，不會改寫或關閉 `torch.cuda.is_available()`。

### Evidence status

目前 D1.15 balanced cap 與 D1.16 basket-100 的 policy 是在看過 2025 replay 後形成，因此：

- D1.5～D1.20 全部使用歷史 mLSTM-only；這些數字不代表目前 Streamlit 預設的 Hybrid；
- prediction-time features 與 gates 仍不讀取 2025 actual；
- 但同一批 2025 headline metrics 屬於 development evidence，不是獨立 holdout；
- D1.20 的 threshold selection 使用 2024 validation，selection step 是 time-safe，但 upstream xLSTM policy 仍帶有 2025 開發歷史；
- 要升級為最終 Tier A 證據，需先凍結 protocol，再對新的未見年份重跑。

正式引用時請先看 [`../docs/experiments/rolling_ablation_index.md`](../docs/experiments/rolling_ablation_index.md) 與 [`../docs/experiments/experiment_registry.md`](../docs/experiments/experiment_registry.md)。

D1 觀察重點（以下 xLSTM 數字全為歷史 mLSTM-only）：

| 比較 | 結果 |
|---|---|
| LSTM plain vs xLSTM plain | xLSTM plain 有訊號，observation-level MAPE 約 `33.34% → 31.09%` |
| LSTM cluster vs xLSTM cluster | xLSTM + Cluster 不穩，暫不放主流程 |
| cluster adjusted vs xLSTM adjusted | D1.12 main-flow basket-30：`20.73% → 19.16%` |
| xLSTM adjusted 改善來源 | D1.11 拆解後，`decline_cap_only` 最佳；growth boost only 稍微變差 |
| stock-level win rate | D1.12 中 xLSTM adjusted 對 cluster adjusted：20/29 檔 MAPE 勝出 |
| robust basket-100 | D1.13 有效 99 檔；xLSTM adjusted 對 cluster adjusted：MAPE `72.54% → 66.77%`，WMAPE `20.72% → 16.37%`，MAPE 勝出 60/99 檔，WMAPE 勝出 59/99 檔 |
| balanced decline cap | D1.15 post-hoc 掃描：balanced gate 相對舊 cap，MAPE 約 `66.77% → 66.76%`，WMAPE 約 `16.37% → 16.29%`，DirectionAccuracy 約 `57.73% → 59.16%` |
| balanced basket-30 validation | D1.15 main-flow basket-30：xLSTM cap rate `8.05% → 6.32%`，WMAPE `12.608% → 12.595%`，DirectionAccuracy `61.21% → 62.36%`，MAPE 幾乎持平 |
| balanced basket-100 validation | D1.16 main-flow basket-100：xLSTM cap rate `9.65% → 7.60%`，WMAPE `16.369% → 16.292%`，DirectionAccuracy `57.73% → 59.16%`，WMAPE 勝出 60/99 檔 |
| validation fallback | D1.17 用 2024 validation 選 xLSTM plain 或 adjusted；結果接近 fixed adjusted，但沒有明顯贏過，暫不升成主流程預設 |
| regime-aware fallback | D1.18 用 stock-regime scope 和 WMAPE 5pp 門檻；WMAPE `16.25628% → 16.25579%`，DirectionAccuracy `61.13% → 61.22%`，改善仍極小 |
| decline confidence cap | D1.19 post-hoc score gate；threshold `0.45` 時 WMAPE `16.25628% → 16.18359%`，MAE `104,309 → 103,843` |
| calibrated confidence cap | D1.20 用 2024 validation 選 threshold `0.55`；2025 WMAPE `16.25628% → 16.23812%`，SMAPE `25.17% → 22.78%` |

D1.16 main-flow basket-100 驗證顯示：`Rolling xLSTM` plain 的 WMAPE `16.26%` 與 DirectionAccuracy `61.13%` 仍是整體最佳；balanced `Rolling xLSTM + Conditional Adjustment` 的 MAPE `66.76%`、MedianAPE `11.18%`、SMAPE `22.90%` 最佳，且比舊 cap 更少傷害 WMAPE/DirectionAccuracy。balanced gate 把 cap rate 從約 `9.65%` 降到 `7.60%`，在保留 MAPE 改善的同時降低 WMAPE 傷害。因此目前判讀是：xLSTM backbone 有價值，decline cap 有助於修掉低分母 MAPE 爆炸，但應用時需要 gate。

D1.17 validation fallback 進一步嘗試用 prior-year validation 做選擇：訓練樣本只用 target year `<= 2023`，用 2024 驗證每檔股票該選 `Rolling xLSTM` plain 還是 balanced `Rolling xLSTM + Conditional Adjustment`，再套到 2025 預測結果。basket-100 中預設 WMAPE 選擇器採 strict tie-to-plain 規則，選了 11 檔 adjusted、88 檔 plain，其中 13 檔因缺 2024 validation default 回 plain。2025 評估為 MAPE `67.17%`、WMAPE `16.256%`、MedianAPE `10.85%`、DirectionAccuracy `60.95%`。它的 WMAPE/MAE 微幅贏 xLSTM plain，MedianAPE 與 SMAPE 優於 plain，但 MAPE 輸 fixed adjusted；訊號太小，所以目前是研究輔助，不是預設策略。

D1.18 將 fallback scope 從整檔股票延伸到 `stock-regime`，讓同一檔股票可在 decline/cycle/growth 月份使用不同來源。正式記錄採 WMAPE 至少改善 5 個百分點才選 adjusted；basket-100 中 158 個 stock-regime groups 只有 5 組選 adjusted，2025 月份 45/1119 筆使用 adjusted。結果 MAPE `67.17%`、WMAPE `16.25579%`、MedianAPE `10.85%`、SMAPE `22.94%`、DirectionAccuracy `61.22%`。這比 xLSTM plain 的 WMAPE `16.25628%` 與 DirectionAccuracy `61.13%` 只微幅好一點，仍不足以升成預設模型，但支持「adjusted 應該非常保守地 gated」這個方向。

D1.19 改用 confidence score，而不是用 2024 validation 直接選模型。score 只使用 xLSTM plain prediction / last observed、過去 12 個月 `growth_ratio`、`growth_streak` 與當月 regime；2025 actual 只在產生 prediction 後評估。basket-100 fine scan 中，threshold `0.45` 的 WMAPE 最佳：MAE `103,843`、MAPE `67.33%`、WMAPE `16.18359%`、MedianAPE `10.97%`、SMAPE `22.77%`、DirectionAccuracy `60.59%`。這比 xLSTM plain 和 fixed adjusted 的 WMAPE 都好，且幅度比 D1.17/D1.18 明顯；但 MAPE 仍不如 fixed adjusted，所以目前仍是 research runner，不直接升主流程。

D1.20 將 D1.19 的 threshold 選擇改成 time-safe calibration：用 target year `<= 2023` 訓練 validation xLSTM，用 2024 validation WMAPE 在 `0.35~0.70` 中選 threshold，再套到 2025。2024 選出 `0.55`；2025 calibrated 結果為 MAE `104,193`、MAPE `67.42%`、WMAPE `16.23812%`、MedianAPE `10.85%`、SMAPE `22.78%`、DirectionAccuracy `60.95%`。這沒有拿到 D1.19 直接看 2025 fine scan 的最佳 WMAPE，但仍比 xLSTM plain 與 fixed adjusted 的 WMAPE 好。因此 D1 結論是：xLSTM backbone 有價值；decline cap 應該保守、time-safe calibrated；但先保留為 research runner，不急著改 Streamlit 預設。

評估準確度時不要只看 MAPE。batch 與 Streamlit metrics 目前同步輸出 `MedianAPE`、`WMAPE`、`SMAPE` 與 `DirectionAccuracy`；低營收月份會放大 MAPE，應同時看 WMAPE/MedianAPE 才能判斷模型是否真的變準。

解讀 xLSTM adjusted 時要分開看：

- `xlstm_adjustment_applied`：growth boost 是否生效
- `xlstm_decline_cap_applied`：decline regime cap 是否生效
- `xlstm_adjustment_ratio`：xLSTM adjusted 相對 xLSTM plain 的倍率

已從主流程移除：

- Dynamic guardrail：保留固定 guardrail，不再對高成長標記放寬上限。
- Direction filter toggle：Growth Adjustment 內固定要求最新月成長率為正。
- Trend + Cycle 月度輸出：保留在研究說明中，但不作為主流程預設結果。
- AutoTune：保留在研究路徑，不放在主流程 UI。

## Environment

The project uses PyTorch only. CUDA is selected automatically when available; otherwise the same implementation runs on CPU.

From the repository root:

```powershell
python --version  # must be Python 3.11.x
python -m venv rolling_predict_LSTM\.venv
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install --upgrade pip
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install -r rolling_predict_LSTM\requirements-gpu.txt
```

For CPU-only installation:

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install -r rolling_predict_LSTM\requirements.txt
```

To recreate the verified Windows CUDA 13.0 package versions:

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install -r rolling_predict_LSTM\requirements-lock.txt
```

此 lockfile 直接使用 CUDA 13.0 wheel index 並固定 `torch==2.11.0+cu130`，因此不適用於
CPU-only 安裝；CPU-only 請使用上方 `requirements.txt`。Lockfile 另固定 `pandas==2.2.3`
與 `pytz==2025.2`。目前 Windows 環境若出現
`DLL load failed while importing period: 應用程式控制原則已封鎖此檔案`，請重建或依 lockfile
重新安裝 Rolling 虛擬環境，不要直接升級到被系統政策封鎖的 pandas wheel。

For the optional xLSTM architectures on the verified CUDA 13.0 path:

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install -r rolling_predict_LSTM\requirements-xlstm.txt
```

`requirements-xlstm.txt` includes `requirements-gpu.txt`, so it installs the cu130 PyTorch build.
For CPU-only xLSTM, first install `requirements.txt` as shown above, then add only the optional package:

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install xlstm==2.0.5
```

On Windows, Hybrid uses the native PyTorch sLSTM backend and therefore does not compile xlstm's optional custom sLSTM CUDA extension. PyTorch still moves the complete mLSTM+sLSTM model to CUDA automatically. The historical D1 path remains mLSTM-only for reproducibility.

## Run

```powershell
.\rolling_predict_LSTM\run_app.ps1
```

The app uses port `8502`.

## Batch Research

Standard batch runners reuse `experiment_metrics.py` for valid-observation filtering, stock counts,
the complete metric record, and grouped summaries. Runner-specific effect tables remain in their
own files; the xLSTM adjustment ablation keeps its specialized boost/cap/guardrail rate summary.

### Sequence Backbone Ablation

比較原本 PyTorch LSTM 與 optional xLSTM backbone。舊 D1.5 指標使用 mLSTM-only；若要跑新 Hybrid，可把 backbone 設為 `xlstm_hybrid`，並以新實驗名稱保存，避免與舊證據混用。

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_sequence_backbone_ablation.py --output-dir rolling_predict_LSTM\outputs\xlstm_hybrid_smoke --backbones xlstm,xlstm_hybrid --epochs 5 --max-train-samples 5000
```

若小跑有訊號，再拉到正式一點的固定參數：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_sequence_backbone_ablation.py --output-dir rolling_predict_LSTM\outputs\xlstm_hybrid_full --backbones xlstm,xlstm_hybrid --k 6 --epochs 35 --max-train-samples 40000
```

擴大到自動抽樣股票池時：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_sequence_backbone_ablation.py --output-dir rolling_predict_LSTM\outputs\xlstm_hybrid_basket_30 --backbones xlstm,xlstm_hybrid --k 6 --epochs 35 --max-train-samples 40000 --stock-limit 30
```

省略 `--backbones` 時仍維持歷史預設 `lstm,xlstm`，用於重現 D1.5 的 LSTM versus mLSTM-only 比較。

重點輸出：

- `stock_accuracy.csv`
- `overall_accuracy.csv`
- `backbone_effects.csv`
- `winner_summary.csv`
- `industry_backbone_accuracy.csv`
- `regime_backbone_accuracy.csv`
- `underestimate_risk.csv`
- `monthly_predictions.csv`

`backbone_effects.csv` 與 `winner_summary.csv` 會依 `--backbones` 指定的兩個架構建立 delta；
例如 `--backbones xlstm,xlstm_hybrid` 會直接輸出 Hybrid minus historical mLSTM-only 的比較，
兩者的 `sequence_backbone` 也會保留在 CSV 中。`winner_summary.csv` 同時提供明確的 MAPE
與 WMAPE wins、ties 和 challenger win rate；無前綴的舊欄位仍保留為 MAPE 相容欄位。

### xLSTM Adjustment Ablation

使用已產生的 `monthly_predictions.csv` 重新播放 xLSTM plain 預測的 post-processing 參數；這不會重新訓練 xLSTM。D1.15 後會分開產生五類 variants：

1. `plain`
2. `growth_boost_only`
3. `decline_cap_only`
4. `decline_cap_balanced`
5. `growth_boost_and_decline_cap`

輸出會分開統計 `GrowthBoostRate` 與 `DeclineCapRate`，避免把 decline regime 的 cap 效果誤讀成成長放大。

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_adjustment_ablation.py --predictions rolling_predict_LSTM\outputs\xlstm_adjustment_basket_30_v2\monthly_predictions.csv --output-dir rolling_predict_LSTM\outputs\xlstm_adjustment_ablation_d1_9
```

可用 `--alphas`、`--conditional-options`、`--regime-options` 掃不同調整強度與 gate 組合。
D1.11 的 basket-30 post-hoc 掃描中，`decline_cap_only` 是 xLSTM adjusted 的較佳起點；D1.15 後主流程預設改用 `decline_cap_balanced`，也就是在 deep decline 且預測明顯高於 last observed 時才 cap。這只用於 xLSTM adjusted，不取代 cluster adjusted 的預設 alpha。

重點輸出：

- `overall_accuracy.csv`
- `component_best_summary.csv`
- `variant_catalog.csv`
- `stock_accuracy.csv`
- `variant_effects.csv`
- `winner_summary.csv`
- `regime_accuracy.csv`
- `baseline_overall_accuracy.csv`
- `monthly_adjustment_predictions.csv`

### xLSTM Main-Flow Comparison

用架構明確的 Streamlit 主流程設定批次驗證五個輸出 rows。這個 runner 固定保留 LSTM main flow，並加入 optional no-cluster xLSTM 與 balanced decline-cap adjusted row；預設 xLSTM 架構與目前 UI 相同，為 `xlstm_hybrid`。

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_main_flow_comparison.py --output-dir rolling_predict_LSTM\outputs\xlstm_main_flow_smoke
```

若要重現 D1.12～D1.20 的歷史 mLSTM-only 設定，必須明確指定：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_main_flow_comparison.py --output-dir rolling_predict_LSTM\outputs\xlstm_main_flow_historical_repro --xlstm-backbone xlstm
```

擴大到自動抽樣 basket：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_main_flow_comparison.py --output-dir rolling_predict_LSTM\outputs\xlstm_main_flow_basket_30 --k 6 --epochs 35 --max-train-samples 40000 --stock-limit 30
```

重點輸出：

- `overall_accuracy.csv`
- `stock_accuracy.csv`
- `model_effects.csv`
- `winner_summary.csv`
- `industry_accuracy.csv`
- `regime_accuracy.csv`
- `monthly_predictions.csv`

`stock_accuracy.csv`、`monthly_predictions.csv`、`failed_runs.csv` 與 `run_config.json` 會記錄 `xlstm_backbone`；benchmark adapter 也會保留 `sequence_backbone` 與 `xlstm_backbone`，避免不同架構在後續比較時被混在一起。

以下 D1.12～D1.16 結果皆由 `--xlstm-backbone xlstm` 的歷史 mLSTM-only 架構產生；不可當作 Hybrid 指標：

D1.12 basket-30 結果：有效 29 檔，`Rolling xLSTM + Conditional Adjustment` observation-level MAPE `19.16%`，`Rolling LSTM + Cluster + Conditional Adjustment` MAPE `20.73%`；股票層級 20/29 檔 xLSTM adjusted 勝出。
D1.13 basket-100 結果：有效 99 檔，1 檔 `7631` 因無 2025 rolling evaluation samples 排除；舊 `Rolling xLSTM + Conditional Adjustment` 對 cluster adjusted 的 MAPE 為 `66.77%` vs `72.54%`，WMAPE 為 `16.37%` vs `20.72%`。同時 `Rolling xLSTM` plain 的 WMAPE `16.26%` 略優於 adjusted。
D1.15 basket-30 validation 已用新預設正式重跑主流程。對同一批 29 檔，balanced cap 相對舊 cap 將 xLSTM adjusted WMAPE 從 `12.608%` 降到 `12.595%`，DirectionAccuracy 從 `61.21%` 提升到 `62.36%`，MAPE 維持約 `19.16%`。
D1.16 basket-100 validation 已用新預設正式重跑主流程。有效 99 檔，1 檔 `7631` 排除；balanced xLSTM adjusted 對 cluster adjusted 的 MAPE 為 `66.76%` vs `72.54%`，WMAPE 為 `16.29%` vs `20.72%`，股票層級 MAPE 勝出 60/99 檔、WMAPE 勝出 60/99 檔。

D1.21 是另外預先登記的 Hybrid fixed-parameter basket-100 run。100/100 檔成功；Hybrid plain 對同次 LSTM plain 的 WMAPE 為 `17.598%` vs `21.259%`，Hybrid adjusted 對同次 cluster adjusted 為 `17.448%` vs `19.359%`。Hybrid adjusted 只比 Hybrid plain 改善 `0.151` 個百分點 WMAPE，DirectionAccuracy 則從 `62.383%` 降至 `60.170%`，所以 adjusted 保留作比較，不能宣稱全面優於 plain。D1.21 與歷史 D1.16 僅重疊 13 檔，不能直接比較兩個 backbone；完整 protocol 與限制見 `docs/experiments/xlstm_hybrid_d1_21_protocol.md`。此結果仍是 Tier C、`report_ready=false`。

D1.22 用同一個 runner、股票 cohort 與 target months 直接比較歷史 mLSTM-only 和 Hybrid。no-cluster plain 的 pooled WMAPE 是 `17.166%` vs `17.598%`，Hybrid 差 `0.433` 個百分點，但股票層級 WMAPE 勝出 63/100 檔並通過預登記的 1.0-point regression gate。Hybrid cluster plain 則為 `15.907%` vs `20.823%`。後者是值得後續以歷史 validation 驗證的研究訊號，不是立即加入 UI 的依據。D1.22 仍是 Tier C、`report_ready=false`；詳見 `docs/experiments/xlstm_backbone_same_cohort_d1_22_protocol.md`。

### xLSTM Validation Fallback

用 prior-year validation 先決定每檔股票的 xLSTM source model，再評估 2025。這個 runner 不用 2025 actual 選擇 fallback；預設會用 target year `<= 2023` 訓練驗證模型，用 2024 validation WMAPE 選 `Rolling xLSTM` 或 `Rolling xLSTM + Conditional Adjustment`。

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_validation_fallback.py --target-predictions rolling_predict_LSTM\outputs\xlstm_main_flow_basket_100_d1_16\monthly_predictions.csv --output-dir rolling_predict_LSTM\outputs\xlstm_validation_fallback_d1_17 --k 6 --epochs 35 --max-train-samples 40000
```

可改成 stock-regime scope，並要求 adjusted 在 validation WMAPE 至少改善 5 個百分點才採用：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_validation_fallback.py --target-predictions rolling_predict_LSTM\outputs\xlstm_main_flow_basket_100_d1_16\monthly_predictions.csv --output-dir rolling_predict_LSTM\outputs\xlstm_validation_fallback_d1_18_stock_regime_wmape5 --selection-scope stock-regime --selection-metric WMAPE --min-improvement 5 --k 6 --epochs 35 --max-train-samples 40000
```

重點輸出：

- `validation_monthly_predictions.csv`
- `validation_accuracy.csv`
- `fallback_selection.csv`
- `fallback_monthly_predictions.csv`
- `combined_monthly_predictions.csv`
- `overall_accuracy.csv`
- `stock_accuracy.csv`
- `regime_accuracy.csv`
- `model_effects.csv`
- `winner_summary.csv`

D1.17 basket-100 結果：預設 WMAPE 選擇器採 strict tie-to-plain，選 11 檔 adjusted、88 檔 plain，整體 MAPE `67.17%`、WMAPE `16.256%`、MedianAPE `10.85%`、DirectionAccuracy `60.95%`。它與 xLSTM plain 的 WMAPE 幾乎打平並微幅較好，但改善不到 `0.001` 個百分點；相對 fixed adjusted 則是 WMAPE 較好、MAPE 較差。因此 validation fallback 目前是判讀輔助，不應直接替換 main-flow xLSTM rows。
D1.18 stock-regime + WMAPE 5pp threshold 結果：選 5 個 stock-regime groups，45/1119 個 2025 預測月份使用 adjusted，整體 MAPE `67.17%`、WMAPE `16.25579%`、MedianAPE `10.85%`、DirectionAccuracy `61.22%`。它比 xLSTM plain 只小幅改善 WMAPE 和方向準確率，但 MAPE 仍不如 fixed adjusted；目前結論是 fallback/gating 有方向，但訊號還太小。

### xLSTM Decline-Cap Confidence

用 xLSTM plain 預測產生 decline cap confidence score，掃不同 score threshold 後重播 capped prediction。這個 runner 不重訓模型，也不使用 2025 actual 來算 score。

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_decline_cap_confidence.py --predictions rolling_predict_LSTM\outputs\xlstm_main_flow_basket_100_d1_16\monthly_predictions.csv --output-dir rolling_predict_LSTM\outputs\xlstm_decline_cap_confidence_d1_19_fine --thresholds 0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7
```

重點輸出：

- `variant_catalog.csv`
- `scored_xlstm_plain.csv`
- `score_distribution.csv`
- `monthly_confidence_predictions.csv`
- `overall_accuracy.csv`
- `stock_accuracy.csv`
- `regime_accuracy.csv`
- `model_effects.csv`
- `winner_summary.csv`

D1.19 fine scan 結果：threshold `0.45` 的 WMAPE 最佳，整體 MAE `103,843`、MAPE `67.33%`、WMAPE `16.18359%`、SMAPE `22.77%`。threshold `0.60` 的 SMAPE 最佳，DirectionAccuracy 則在更高 threshold 接近 xLSTM plain。這表示 score gate 可以比 fixed balanced cap 更好地保護 WMAPE，但仍需要在 MAPE 與 WMAPE 之間取捨。

### xLSTM Confidence Calibration

用 2024 validation 選 D1.19 confidence threshold，再套到 2025。這是 D1 的收斂檢查，用來避免直接看 2025 fine scan 挑 threshold。

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_xlstm_confidence_calibration.py --target-predictions rolling_predict_LSTM\outputs\xlstm_main_flow_basket_100_d1_16\monthly_predictions.csv --output-dir rolling_predict_LSTM\outputs\xlstm_confidence_calibration_d1_20 --thresholds 0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7 --selection-metric WMAPE --k 6 --epochs 35 --max-train-samples 40000
```

重點輸出：

- `validation_confidence_predictions.csv`
- `validation_accuracy.csv`
- `threshold_selection.csv`
- `target_confidence_predictions.csv`
- `target_threshold_accuracy.csv`
- `calibrated_monthly_predictions.csv`
- `overall_accuracy.csv`
- `stock_accuracy.csv`
- `regime_accuracy.csv`
- `model_effects.csv`
- `winner_summary.csv`

D1.20 結果：2024 validation 選 `0.55`，2025 calibrated WMAPE `16.23812%`，優於 xLSTM plain `16.25628%` 與 fixed adjusted `16.29188%`；SMAPE `22.78%` 也優於 fixed adjusted `22.90%`。MAPE `67.42%` 仍輸 fixed adjusted `66.76%`，所以這是研究候選，不是主流程預設替換。

### Method and Feature Ablation

用固定 K、epochs、sample cap 逐一比較方法或特徵是否有效：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_ablation_study.py --output-dir rolling_predict_LSTM\outputs\ablation_full --groups method,feature --k 6 --epochs 35 --max-train-samples 40000
```

### Quarterly Target Ablation

用來確認震盪股是否應改成區間／季度 target。實驗比較：

- `MS*`：現有一月預測滾動加總成 3 個月區間。
- `Q*`：直接用過去 12 個月預測未來 3 個月營收總和。

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe rolling_predict_LSTM\batch_quarterly_target_ablation.py --output-dir rolling_predict_LSTM\outputs\quarterly_target_full --k 6 --epochs 35 --max-train-samples 40000
```

重點輸出：

- `overall_accuracy.csv`
- `overall_effects.csv`
- `dominant_regime_accuracy.csv`
- `dominant_regime_effects.csv`
- `cycle_necessity_summary.csv`
- `quarter_predictions.csv`

截至 2026-07-31 的完整實驗結果：direct 3M target 沒有打贏 monthly-sum benchmark，cycle dominant 股票也不支持整批改成季度 target。季度 target 目前應視為研究輔助或少數股票 fallback，不是主流程替代方案。

## Tests

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m compileall -q rolling_predict_LSTM
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m unittest discover -s rolling_predict_LSTM\tests -v
```

Quick smoke（只驗證流程，不代表正式 accuracy）：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -c "from rolling_predict_LSTM.rolling_lstm_engine import GrowthAdjustmentConfig, RollingExperimentConfig, run_rolling_lstm_experiment; config=RollingExperimentConfig(k=4, epochs=5, max_train_samples=5000, growth=GrowthAdjustmentConfig(enabled=True)); r=run_rolling_lstm_experiment(1101, config=config); print(r.metrics.to_string(index=False))"
```

## Ownership

This system owns:

- `app.py`: Streamlit interface and experiment controls
- `rolling_lstm_engine.py`: sequence, cluster, training, adjustment, and evaluation logic
- `yield_forecast.py`: Rolling compatibility adapter over the shared financial forecast interface
- `batch_*.py`: batch research workflows
- `docs/`: Rolling-specific research notes
- `tests/`: Rolling behavior and isolation tests
- `outputs/`: local experiment results, ignored by Git

It reads source data from the repository root `data/` directory and must not import `ensemble_forecast`.

完整 raw outputs 不會出現在 fresh clone。可引用結論應同步整理到 tracked `docs/experiments/`；需要重現 benchmark 時，先重新產生指定 Rolling output，或用 `--rolling-output-dir` 指向另外取得的 artifact。
