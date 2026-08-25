import unittest

import pandas as pd

from rolling_predict_LSTM import batch_xlstm_main_flow_comparison as comparison
from rolling_predict_LSTM import rolling_lstm_engine


class XLSTMMainFlowComparisonContractTests(unittest.TestCase):
    def test_model_columns_include_streamlit_facing_xlstm_rows(self) -> None:
        self.assertEqual(
            list(comparison.MODEL_COLUMNS),
            [
                "Rolling LSTM",
                "Rolling LSTM + Cluster",
                "Rolling LSTM + Cluster + Conditional Adjustment",
                "Rolling xLSTM",
                "Rolling xLSTM + Conditional Adjustment",
            ],
        )

    def test_default_xlstm_alpha_tracks_engine_balanced_decline_cap_default(self) -> None:
        self.assertEqual(comparison.DEFAULT_XLSTM_ALPHA, rolling_lstm_engine.DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA)
        self.assertEqual(comparison.DEFAULT_XLSTM_ALPHA, 0.0)

    def test_build_run_config_keeps_lstm_main_flow_and_includes_xlstm_plain(self) -> None:
        config = comparison.build_run_config(comparison.MainFlowRunSpec(k=4, epochs=5))

        self.assertEqual(config.sequence_backbone, "lstm")
        self.assertTrue(config.include_xlstm_plain)
        self.assertEqual(config.xlstm_backbone, "xlstm_hybrid")
        self.assertEqual(config.growth.alpha, 0.8)
        self.assertEqual(config.xlstm_growth.alpha, 0.0)
        self.assertEqual(
            config.xlstm_growth.decline_cap_growth_ratio_max,
            rolling_lstm_engine.DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX,
        )
        self.assertEqual(
            config.xlstm_growth.decline_cap_prediction_ratio_min,
            rolling_lstm_engine.DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN,
        )

    def test_historical_mlstm_only_backbone_remains_selectable(self) -> None:
        config = comparison.build_run_config(
            comparison.MainFlowRunSpec(xlstm_backbone="xlstm")
        )

        self.assertEqual(config.xlstm_backbone, "xlstm")

    def test_monthly_export_records_selected_backbone(self) -> None:
        forecast = pd.DataFrame(
            {
                "target_year": [2025],
                "target_month": [1],
                "regime": ["cycle"],
                "actual_revenue": [110.0],
                "last_observed_revenue": [100.0],
                "predicted_revenue_no_cluster": [101.0],
                "predicted_revenue_cluster": [102.0],
                "predicted_revenue_adjusted": [103.0],
                "predicted_revenue_xlstm": [104.0],
                "predicted_revenue_xlstm_adjusted": [105.0],
            }
        )

        exported = comparison.build_monthly_long_frame(
            forecast,
            stock_id=1101,
            stock_name="A",
            industry_category="cement",
            xlstm_backbone="xlstm_hybrid",
        )

        xlstm_rows = exported[exported["model"].isin(rolling_lstm_engine.ROLLING_XLSTM_MODEL_NAMES)]
        lstm_rows = exported[~exported["model"].isin(rolling_lstm_engine.ROLLING_XLSTM_MODEL_NAMES)]
        self.assertEqual(exported["xlstm_backbone"].unique().tolist(), ["xlstm_hybrid"])
        self.assertEqual(xlstm_rows["sequence_backbone"].unique().tolist(), ["xlstm_hybrid"])
        self.assertEqual(lstm_rows["sequence_backbone"].unique().tolist(), ["lstm"])

        overall, industry, regime = comparison.build_accuracy_summaries(exported)
        for summary in (overall, industry, regime):
            self.assertIn("xlstm_backbone", summary.columns)
            self.assertIn("sequence_backbone", summary.columns)
            self.assertEqual(summary["xlstm_backbone"].unique().tolist(), ["xlstm_hybrid"])
            self.assertEqual(set(summary["sequence_backbone"]), {"lstm", "xlstm_hybrid"})

    def test_model_effects_compare_named_baseline_and_challenger_pairs(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "stock_name": ["A", "A", "B", "B"],
                "industry_category": ["X", "X", "Y", "Y"],
                "sequence_backbone": ["xlstm_hybrid"] * 4,
                "xlstm_backbone": ["xlstm_hybrid"] * 4,
                "model": [
                    "Rolling xLSTM",
                    "Rolling xLSTM + Conditional Adjustment",
                    "Rolling xLSTM",
                    "Rolling xLSTM + Conditional Adjustment",
                ],
                "MAPE": [30.0, 20.0, 10.0, 12.0],
                "WMAPE": [25.0, 15.0, 8.0, 18.0],
                "MAE": [300.0, 200.0, 100.0, 120.0],
                "Bias": [5.0, 2.0, -1.0, -3.0],
                "UnderestimateRate": [50.0, 40.0, 20.0, 25.0],
            }
        )

        effects = comparison.build_model_effects(
            stock_accuracy,
            effect_pairs=(("xadj_minus_plain", "Rolling xLSTM", "Rolling xLSTM + Conditional Adjustment"),),
        )
        summary = comparison.build_winner_summary(effects)

        self.assertEqual(float(effects.loc[0, "MAPE_delta_challenger_minus_baseline"]), -10.0)
        self.assertEqual(float(effects.loc[0, "WMAPE_delta_challenger_minus_baseline"]), -10.0)
        self.assertEqual(effects.loc[0, "WMAPE_winner"], "challenger")
        self.assertEqual(float(effects.loc[1, "MAPE_delta_challenger_minus_baseline"]), 2.0)
        self.assertEqual(effects.loc[1, "WMAPE_winner"], "baseline")
        self.assertEqual(int(summary.loc[0, "challenger_wins"]), 1)
        self.assertEqual(int(summary.loc[0, "baseline_wins"]), 1)
        self.assertEqual(int(summary.loc[0, "WMAPE_challenger_wins"]), 1)
        self.assertEqual(int(summary.loc[0, "WMAPE_baseline_wins"]), 1)
        self.assertEqual(effects["xlstm_backbone"].unique().tolist(), ["xlstm_hybrid"])
        self.assertEqual(effects["baseline_sequence_backbone"].unique().tolist(), ["xlstm_hybrid"])
        self.assertEqual(effects["challenger_sequence_backbone"].unique().tolist(), ["xlstm_hybrid"])
        self.assertEqual(summary["xlstm_backbone"].unique().tolist(), ["xlstm_hybrid"])

    def test_model_effects_record_each_compared_architecture(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "stock_id": [1, 1],
                "stock_name": ["A", "A"],
                "industry_category": ["X", "X"],
                "sequence_backbone": ["lstm", "xlstm_hybrid"],
                "xlstm_backbone": ["xlstm_hybrid", "xlstm_hybrid"],
                "model": [
                    rolling_lstm_engine.ROLLING_LSTM_MODEL,
                    rolling_lstm_engine.ROLLING_XLSTM_MODEL,
                ],
                "MAPE": [15.0, 10.0],
            }
        )

        effects = comparison.build_model_effects(
            stock_accuracy,
            effect_pairs=(
                (
                    "xlstm_minus_lstm",
                    rolling_lstm_engine.ROLLING_LSTM_MODEL,
                    rolling_lstm_engine.ROLLING_XLSTM_MODEL,
                ),
            ),
        )
        summary = comparison.build_winner_summary(effects)

        self.assertEqual(effects["baseline_sequence_backbone"].tolist(), ["lstm"])
        self.assertEqual(effects["challenger_sequence_backbone"].tolist(), ["xlstm_hybrid"])
        self.assertEqual(summary["baseline_sequence_backbone"].tolist(), ["lstm"])
        self.assertEqual(summary["challenger_sequence_backbone"].tolist(), ["xlstm_hybrid"])


if __name__ == "__main__":
    unittest.main()
