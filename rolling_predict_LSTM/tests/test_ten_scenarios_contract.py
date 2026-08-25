import unittest

import pandas as pd

from rolling_predict_LSTM import batch_ten_scenarios as scenarios


class TenScenariosContractTests(unittest.TestCase):
    def test_summarize_uses_finite_observations_and_complete_metric_contract(self) -> None:
        predictions = pd.DataFrame(
            {
                "scenario_id": ["S01"] * 5,
                "stock_id": [1, 2, 3, 4, 5],
                "actual_revenue": [100.0, 200.0, float("nan"), 300.0, 400.0],
                "predicted_revenue": [90.0, 220.0, 150.0, float("nan"), 410.0],
                "last_observed_revenue": [80.0, 210.0, 100.0, 280.0, float("nan")],
            }
        )

        summary = scenarios.summarize(predictions, ["scenario_id"])

        self.assertEqual(["observations", "stock_count"], list(summary.columns[1:3]))
        self.assertEqual(int(summary.loc[0, "observations"]), 2)
        self.assertEqual(int(summary.loc[0, "stock_count"]), 2)
        self.assertAlmostEqual(float(summary.loc[0, "MAE"]), 15.0)
        self.assertAlmostEqual(float(summary.loc[0, "WMAPE"]), 10.0)
        self.assertAlmostEqual(float(summary.loc[0, "SMAPE"]), 10.0250626566)
        self.assertAlmostEqual(float(summary.loc[0, "DirectionAccuracy"]), 50.0)

    def test_parameter_effects_export_smape_comparison(self) -> None:
        metric_names = [
            "RMSE",
            "MAE",
            "MAPE",
            "MedianAPE",
            "WMAPE",
            "SMAPE",
            "Bias",
            "UnderestimateRate",
            "DirectionAccuracy",
        ]
        baseline = {metric: 10.0 for metric in metric_names}
        treatment = {metric: 8.0 for metric in metric_names}
        baseline["SMAPE"] = 20.0
        treatment["SMAPE"] = 15.0
        summary = pd.DataFrame(
            [
                {"scenario_id": "S01", **baseline},
                {"scenario_id": "S02", **treatment},
            ]
        )

        effects = scenarios.build_parameter_effects(summary, ["scenario_id"])
        growth_effect = effects[effects["effect_id"].eq("E01")].iloc[0]

        self.assertAlmostEqual(float(growth_effect["SMAPE_base"]), 20.0)
        self.assertAlmostEqual(float(growth_effect["SMAPE_treatment"]), 15.0)
        self.assertAlmostEqual(float(growth_effect["SMAPE_delta"]), -5.0)


if __name__ == "__main__":
    unittest.main()
