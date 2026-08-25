import unittest

import pandas as pd

from rolling_predict_LSTM import batch_quarterly_target_ablation as quarterly
from rolling_predict_LSTM import rolling_lstm_engine


class QuarterlyTargetAblationContractTests(unittest.TestCase):
    def test_default_specs_have_unique_ids_and_both_target_families(self) -> None:
        specs = quarterly.DEFAULT_QUARTERLY_SPECS
        experiment_ids = [spec.experiment_id for spec in specs]
        target_families = {spec.target_family for spec in specs}

        self.assertEqual(len(experiment_ids), len(set(experiment_ids)))
        self.assertEqual({"monthly_sum", "direct_3m"}, target_families)

    def test_effect_pairs_reference_known_experiments(self) -> None:
        experiment_ids = {spec.experiment_id for spec in quarterly.DEFAULT_QUARTERLY_SPECS}

        for _, _, baseline_id, treatment_id in quarterly.EFFECT_PAIRS:
            self.assertIn(baseline_id, experiment_ids)
            self.assertIn(treatment_id, experiment_ids)

    def test_quarterly_eval_samples_do_not_include_2025_actual_target(self) -> None:
        dates = pd.date_range("2023-01-01", periods=36, freq="MS")
        raw = pd.DataFrame(
            {
                "stock_id": [1101] * len(dates),
                "revenue_year": dates.year,
                "revenue_month": dates.month,
                "revenue_thousand": range(100, 100 + len(dates)),
                "industry_category": ["cement"] * len(dates),
            }
        )
        prepared = rolling_lstm_engine.prepare_revenue_data(raw)
        prepared["cluster"] = 0
        prepared["growth_ratio"] = 0.5
        prepared["growth_streak"] = 1
        prepared = quarterly.enrich_quarterly_monthly_frame(prepared, horizon=3)

        train, eval_ = quarterly.build_quarterly_sequences_for_stock(
            prepared,
            1101,
            window_size=12,
            horizon=3,
            train_end_year=2024,
            eval_year=2025,
        )

        self.assertTrue(train)
        self.assertTrue(eval_)
        first_eval = eval_[0]
        self.assertNotIn("target_revenue", first_eval)
        self.assertEqual(pd.Timestamp("2024-12-01"), first_eval["sequence_end_date"])
        self.assertEqual(pd.Timestamp("2025-01-01"), first_eval["target_date"])
        self.assertEqual(pd.Timestamp("2025-03-01"), first_eval["target_end_date"])
        self.assertEqual("2025-01~2025-03", first_eval["target_period_label"])

    def test_quarterly_training_target_is_next_three_month_sum(self) -> None:
        dates = pd.date_range("2023-01-01", periods=36, freq="MS")
        raw = pd.DataFrame(
            {
                "stock_id": [1101] * len(dates),
                "revenue_year": dates.year,
                "revenue_month": dates.month,
                "revenue_thousand": range(100, 100 + len(dates)),
                "industry_category": ["cement"] * len(dates),
            }
        )
        prepared = rolling_lstm_engine.prepare_revenue_data(raw)
        prepared["cluster"] = 0
        prepared["growth_ratio"] = 0.5
        prepared["growth_streak"] = 1
        prepared = quarterly.enrich_quarterly_monthly_frame(prepared, horizon=3)

        train, _ = quarterly.build_quarterly_sequences_for_stock(
            prepared,
            1101,
            window_size=12,
            horizon=3,
            train_end_year=2024,
            eval_year=2025,
        )

        first_train = train[0]
        self.assertEqual(pd.Timestamp("2023-12-01"), first_train["sequence_end_date"])
        self.assertEqual(pd.Timestamp("2024-01-01"), first_train["target_date"])
        self.assertEqual(pd.Timestamp("2024-03-01"), first_train["target_end_date"])
        self.assertEqual(112 + 113 + 114, first_train["target_revenue"])
        self.assertEqual(109 + 110 + 111, first_train["last_observed_period_revenue"])

    def test_summarize_uses_finite_observations_and_complete_metric_contract(self) -> None:
        predictions = pd.DataFrame(
            {
                "experiment_id": ["MS00"] * 5,
                "stock_id": [1, 2, 3, 4, 5],
                "actual_revenue": [100.0, 200.0, float("nan"), 300.0, 400.0],
                "predicted_revenue": [90.0, 220.0, 150.0, float("nan"), 410.0],
                "last_observed_revenue": [80.0, 210.0, 100.0, 280.0, float("nan")],
            }
        )

        summary = quarterly.summarize(predictions, ["experiment_id"])

        self.assertEqual(int(summary.loc[0, "observations"]), 2)
        self.assertEqual(int(summary.loc[0, "stock_count"]), 2)
        self.assertAlmostEqual(float(summary.loc[0, "MAE"]), 15.0)
        self.assertAlmostEqual(float(summary.loc[0, "WMAPE"]), 10.0)
        self.assertAlmostEqual(float(summary.loc[0, "SMAPE"]), 10.0250626566)
        self.assertAlmostEqual(float(summary.loc[0, "DirectionAccuracy"]), 50.0)

    def test_overall_effects_work_without_group_columns(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "experiment_id": "MS00",
                    "experiment_name": "baseline",
                    "RMSE": 10.0,
                    "MAE": 8.0,
                    "MAPE": 7.0,
                    "MedianAPE": 6.0,
                    "WMAPE": 5.0,
                    "SMAPE": 4.5,
                    "Bias": 4.0,
                    "UnderestimateRate": 3.0,
                    "DirectionAccuracy": 2.0,
                },
                {
                    "experiment_id": "Q00",
                    "experiment_name": "treatment",
                    "RMSE": 9.0,
                    "MAE": 7.0,
                    "MAPE": 6.0,
                    "MedianAPE": 5.0,
                    "WMAPE": 4.0,
                    "SMAPE": 3.0,
                    "Bias": 3.0,
                    "UnderestimateRate": 2.0,
                    "DirectionAccuracy": 1.0,
                },
            ]
        )

        effects = quarterly.build_pair_effects(summary, [])

        self.assertEqual(1, len(effects[effects["effect_id"].eq("E01")]))
        e01 = effects[effects["effect_id"].eq("E01")].iloc[0]
        self.assertEqual(-1.0, e01["WMAPE_delta"])
        self.assertEqual(4.5, e01["SMAPE_base"])
        self.assertEqual(3.0, e01["SMAPE_treatment"])
        self.assertEqual(-1.5, e01["SMAPE_delta"])


if __name__ == "__main__":
    unittest.main()
