import unittest

import pandas as pd

from rolling_predict_LSTM import batch_all_stocks_penalty as penalty


class PenaltyAblationContractTests(unittest.TestCase):
    def test_summarize_uses_finite_observations_and_complete_metric_contract(self) -> None:
        predictions = pd.DataFrame(
            {
                "penalty_setting": ["off_huber"] * 5,
                "stock_id": [1, 2, 3, 4, 5],
                "actual_revenue": [100.0, 200.0, float("nan"), 300.0, 400.0],
                "predicted_revenue": [90.0, 220.0, 150.0, float("nan"), 410.0],
                "last_observed_revenue": [80.0, 210.0, 100.0, 280.0, float("nan")],
            }
        )

        summary = penalty.summarize(predictions, ["penalty_setting"])

        self.assertEqual(["stock_count", "observations"], list(summary.columns[1:3]))
        self.assertEqual(int(summary.loc[0, "observations"]), 2)
        self.assertEqual(int(summary.loc[0, "stock_count"]), 2)
        self.assertAlmostEqual(float(summary.loc[0, "MAE"]), 15.0)
        self.assertAlmostEqual(float(summary.loc[0, "MedianAPE"]), 10.0)
        self.assertAlmostEqual(float(summary.loc[0, "WMAPE"]), 10.0)
        self.assertAlmostEqual(float(summary.loc[0, "SMAPE"]), 10.0250626566)
        self.assertAlmostEqual(float(summary.loc[0, "DirectionAccuracy"]), 50.0)

    def test_build_impact_exports_median_ape_and_smape_deltas(self) -> None:
        summary = pd.DataFrame(
            {
                "model": ["Rolling LSTM", "Rolling LSTM"],
                "penalty_setting": ["off_huber", "on_under_weight_2"],
                "MedianAPE": [20.0, 15.0],
                "SMAPE": [18.0, 16.0],
            }
        )

        impact = penalty.build_impact(summary, ["model"])

        self.assertAlmostEqual(float(impact.loc[0, "MedianAPE_delta_on_minus_off"]), -5.0)
        self.assertAlmostEqual(float(impact.loc[0, "SMAPE_delta_on_minus_off"]), -2.0)


if __name__ == "__main__":
    unittest.main()
