import unittest

import pandas as pd

from rolling_predict_LSTM import batch_xlstm_decline_cap_confidence as confidence


class XLSTMDeclineCapConfidenceContractTests(unittest.TestCase):
    def test_parse_float_csv_uses_default_for_empty_value(self) -> None:
        self.assertEqual(confidence.parse_float_csv(None, (0.1, 0.2)), (0.1, 0.2))
        self.assertEqual(confidence.parse_float_csv("0, 0.5,1"), (0.0, 0.5, 1.0))

    def test_confidence_score_is_time_safe_and_decline_only(self) -> None:
        frame = pd.DataFrame(
            {
                "stock_id": [1, 1, 1],
                "target_year": [2025, 2025, 2025],
                "target_month": [1, 2, 3],
                "actual_revenue": [80.0, 80.0, 80.0],
                "last_observed_revenue": [100.0, 100.0, 100.0],
                "regime": ["decline", "decline", "cycle"],
                "growth_ratio": [0.0, 0.35, 0.0],
                "growth_streak": [0.0, 3.0, 0.0],
                "model": [confidence.MODEL_XLSTM_PLAIN] * 3,
                "predicted_revenue": [150.0, 105.0, 150.0],
            }
        )

        scored = confidence.calculate_decline_cap_confidence(frame)

        self.assertGreater(scored.loc[0, "decline_cap_confidence"], scored.loc[1, "decline_cap_confidence"])
        self.assertEqual(float(scored.loc[2, "decline_cap_confidence"]), 0.0)
        self.assertAlmostEqual(float(scored.loc[0, "prediction_ratio_to_last"]), 1.5)

    def test_build_confidence_predictions_caps_only_rows_above_threshold(self) -> None:
        monthly = pd.DataFrame(
            {
                "stock_id": [1, 1, 1, 1],
                "stock_name": ["A", "A", "A", "A"],
                "industry_category": ["X", "X", "X", "X"],
                "target_year": [2025, 2025, 2025, 2025],
                "target_month": [1, 1, 2, 2],
                "actual_revenue": [80.0, 80.0, 120.0, 120.0],
                "last_observed_revenue": [100.0, 100.0, 100.0, 100.0],
                "regime": ["decline", "decline", "cycle", "cycle"],
                "growth_ratio": [0.0, 0.0, 0.0, 0.0],
                "growth_streak": [0.0, 0.0, 0.0, 0.0],
                "model": [
                    confidence.MODEL_XLSTM_PLAIN,
                    confidence.MODEL_XLSTM_ADJUSTED,
                    confidence.MODEL_XLSTM_PLAIN,
                    confidence.MODEL_XLSTM_ADJUSTED,
                ],
                "predicted_revenue": [150.0, 100.0, 150.0, 100.0],
            }
        )

        result = confidence.build_confidence_predictions(monthly, thresholds=(0.5,))

        self.assertEqual(result["model"].unique().tolist(), [confidence.confidence_model_name(0.5)])
        self.assertEqual(result.sort_values("target_month")["predicted_revenue"].tolist(), [100.0, 150.0])
        self.assertEqual(result.sort_values("target_month")["confidence_cap_applied"].tolist(), [True, False])
        self.assertEqual(result.sort_values("target_month")["error"].tolist(), [20.0, 30.0])

    def test_build_effect_pairs_compares_confidence_models_to_core_baselines(self) -> None:
        pairs = confidence.build_effect_pairs((confidence.confidence_model_name(0.5),))
        baselines = {baseline for _, baseline, _ in pairs}

        self.assertEqual(
            baselines,
            {
                confidence.MODEL_XLSTM_PLAIN,
                confidence.MODEL_XLSTM_ADJUSTED,
                confidence.MODEL_CLUSTER_ADJUSTED,
            },
        )


if __name__ == "__main__":
    unittest.main()
