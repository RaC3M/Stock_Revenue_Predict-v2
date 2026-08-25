# Documentation Index

本頁是 Yield-Predict 的文件入口。若不同文件的數字或結論不一致，優先順序如下：

1. `docs/experiments/experiment_registry.md` 的 evidence status
2. 當前系統 README 與 benchmark protocol
3. 各實驗詳細文件
4. 歷史 output 或舊版 diagnostic 文件

## Start here

- [Root README](../README.md)：安裝、執行、驗證與 repository 結構
- [Experiment registry](experiments/experiment_registry.md)：哪些 run 可引用、哪些只屬於 development／legacy
- [Benchmark protocol](experiments/benchmark_protocol.md)：共同 cohort、指標、失敗處理與當前證據限制
- [Rolling ablation index](experiments/rolling_ablation_index.md)：Rolling／xLSTM 實驗地圖與結論邊界
- [D1.21 Hybrid protocol and result](experiments/xlstm_hybrid_d1_21_protocol.md)：預先登記設定、basket-100 結果、merge gate 與限制
- [D1.22 same-cohort backbone result](experiments/xlstm_backbone_same_cohort_d1_22_protocol.md)：mLSTM-only／Hybrid exact-pair 比較與結論限制

## Architecture and ownership

- [ADR 0001 — independent forecast systems](adr/0001-independent-forecast-systems.md)
- [ADR 0002 — canonical data generation](adr/0002-canonical-data-generation.md)
- [ADR 0003 — shared financial forecast module](adr/0003-shared-financial-forecast-module.md)
- [Ensemble Forecast README](../ensemble_forecast/README.md)
- [Rolling LSTM README](../rolling_predict_LSTM/README.md)
- [Forecast Benchmark README](../forecast_benchmark/README.md)
- [Rolling prediction logic](../rolling_predict_LSTM/docs/prediction-logic.md)

Repository 有兩套營收預測系統、一個中立財務轉換模組與一個比較層：

```text
data/ ──> ensemble_forecast/ ──┐
   └────> rolling_predict_LSTM/ ├──> financial_forecast/
                               ┘

ensemble outputs ──┐
                   ├──> forecast_benchmark/
rolling outputs ───┘
```

`financial_forecast/` 只擁有 availability-safe EPS／股利／殖利率轉換；兩套系統透過各自 adapter 使用它。`forecast_benchmark/` 可以讀取兩套系統證據並選擇方法，但兩套預測系統不得互相 import。

## Current experiment documentation

- [Canonical data preprocessing](experiments/free_taiwan_data_preprocessing_zh.md)：資料產生與 replacement gate
- [Direct dividend model](experiments/direct_dividend_model_benchmark_zh.md)：time-safe hurdle／bucket selection
- [Direct dividend diagnostics](experiments/direct_dividend_error_diagnostics_zh.md)：classification 與 amount error 拆解
- [Dividend layer](experiments/dividend_layer_benchmark_zh.md)：legacy、time-safe、announcement-safe 比較
- [Frozen financial ablation](experiments/financial_ablation_20260731.md)：2022–2024 下游選模與 frozen 2025 test

## Historical or supporting diagnostics

以下文件仍保留方法與歷史價值，但其中部分表格來自 2026-07-30 以前的 output。引用精確數字前，先查看文件頂端的 status note 和 experiment registry：

- [EPS benchmark](experiments/eps_benchmark_result_zh.md)
- [EPS diagnostics](experiments/eps_error_diagnostics_zh.md)
- [EPS layer validation](experiments/eps_layer_validation_zh.md)
- [Yield benchmark](experiments/yield_benchmark_result_zh.md)
- [EPS-to-yield transmission](experiments/yield_eps_layer_benchmark_zh.md)

## Evidence policy

- `report_ready=true` 只描述 run 當時寫入的 metadata；是否適合作為最終證據，仍要檢查 selection protocol 與上游模型是否曾看過 target-year 結果。
- 2025 xLSTM 主結果是歷史 mLSTM-only 的 development evidence；D1.15/D1.16 重用了 2025，因此不是獨立 holdout。
- 目前 Streamlit 預設的 `mLSTM → sLSTM` Hybrid 已有 D1.21 fixed-parameter basket-100 結果；100/100 檔成功，Hybrid adjusted WMAPE `17.448%`。因 adjustment policy 帶有 2025 開發歷史，仍列 Tier C、`report_ready=false`，也不得沿用 mLSTM-only D1 數字。
- D1.22 exact-cohort 比較顯示 Hybrid plain 的 pooled WMAPE `17.598%` 略差於歷史 mLSTM-only `17.166%`，但股票層級勝出 63/100 檔；兩者仍是 2025 development evidence，不能取代 unseen-year test。
- `current_system_payout_ratio` 是 legacy hindsight diagnostic，不是目前 Ensemble 正式路徑。
- 2025 actual 只能在 prediction 完成後用於 evaluation，不得用於特徵、regime、threshold 或模型選擇。

## Agent collaboration

- [AGENTS.md](../AGENTS.md)：repository 工作規範
- [CONTEXT.md](../CONTEXT.md)：domain language
- [Issue tracker integration](agents/issue-tracker.md)
- [Triage labels](agents/triage-labels.md)
- [Domain docs guide](agents/domain.md)
