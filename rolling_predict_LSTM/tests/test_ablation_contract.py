import unittest

import pandas as pd

from rolling_predict_LSTM import batch_ablation_study
from rolling_predict_LSTM import rolling_lstm_engine


class BatchAblationContractTests(unittest.TestCase):
    def test_default_experiments_have_unique_ids_and_valid_baselines(self) -> None:
        specs = batch_ablation_study.DEFAULT_ABLATION_SPECS
        experiment_ids = [spec.experiment_id for spec in specs]

        self.assertEqual(len(experiment_ids), len(set(experiment_ids)))
        self.assertEqual({"method", "feature"}, {spec.ablation_group for spec in specs})
        self.assertTrue(all(spec.baseline_id in experiment_ids for spec in specs))

    def test_feature_ablations_remove_one_input_at_a_time(self) -> None:
        specs = {spec.experiment_id: spec for spec in batch_ablation_study.DEFAULT_ABLATION_SPECS}
        base_features = tuple(rolling_lstm_engine.NUMERIC_SEQUENCE_FEATURES)

        self.assertEqual(specs["F00"].numeric_features, base_features)
        for experiment_id in ("F01", "F02", "F03", "F04"):
            removed = set(base_features) - set(specs[experiment_id].numeric_features)
            added = set(specs[experiment_id].numeric_features) - set(base_features)
            self.assertEqual(len(removed), 1)
            self.assertEqual(added, set())

    def test_cluster_feature_ablation_disables_cluster_one_hot_only(self) -> None:
        specs = {spec.experiment_id: spec for spec in batch_ablation_study.DEFAULT_ABLATION_SPECS}

        self.assertFalse(specs["F05"].include_cluster)
        self.assertEqual(specs["F05"].numeric_features, specs["F00"].numeric_features)
        self.assertEqual(specs["F05"].baseline_id, "F00")

    def test_group_filter_keeps_requested_ablation_family(self) -> None:
        method_specs = batch_ablation_study.build_experiment_specs({"method"})
        feature_specs = batch_ablation_study.build_experiment_specs({"feature"})

        self.assertTrue(method_specs)
        self.assertTrue(feature_specs)
        self.assertTrue(all(spec.ablation_group == "method" for spec in method_specs))
        self.assertTrue(all(spec.ablation_group == "feature" for spec in feature_specs))

    def test_removed_method_ablations_are_not_in_default_plan(self) -> None:
        specs = batch_ablation_study.DEFAULT_ABLATION_SPECS
        experiment_ids = {spec.experiment_id for spec in specs}

        self.assertNotIn("M04", experiment_ids)
        self.assertNotIn("M06", experiment_ids)

    def test_summarize_uses_finite_observations_and_complete_metric_contract(self) -> None:
        predictions = pd.DataFrame(
            {
                "experiment_id": ["F00"] * 5,
                "stock_id": [1, 2, 3, 4, 5],
                "actual_revenue": [100.0, 200.0, float("nan"), 300.0, 400.0],
                "predicted_revenue": [90.0, 220.0, 150.0, float("nan"), 410.0],
                "last_observed_revenue": [80.0, 210.0, 100.0, 280.0, float("nan")],
            }
        )

        summary = batch_ablation_study.summarize(predictions, ["experiment_id"])

        self.assertEqual(int(summary.loc[0, "observations"]), 2)
        self.assertEqual(int(summary.loc[0, "stock_count"]), 2)
        self.assertAlmostEqual(float(summary.loc[0, "MAE"]), 15.0)
        self.assertAlmostEqual(float(summary.loc[0, "WMAPE"]), 10.0)
        self.assertAlmostEqual(float(summary.loc[0, "SMAPE"]), 10.0250626566)
        self.assertAlmostEqual(float(summary.loc[0, "DirectionAccuracy"]), 50.0)

    def test_build_effects_exports_smape_comparison(self) -> None:
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
                {
                    "experiment_id": "F00",
                    "experiment_name": "baseline",
                    "baseline_id": "F00",
                    "ablation_group": "feature",
                    **baseline,
                },
                {
                    "experiment_id": "F01",
                    "experiment_name": "treatment",
                    "baseline_id": "F00",
                    "ablation_group": "feature",
                    **treatment,
                },
            ]
        )

        effects = batch_ablation_study.build_effects(summary, [])

        self.assertAlmostEqual(float(effects.loc[0, "SMAPE_base"]), 20.0)
        self.assertAlmostEqual(float(effects.loc[0, "SMAPE_treatment"]), 15.0)
        self.assertAlmostEqual(float(effects.loc[0, "SMAPE_delta"]), -5.0)


if __name__ == "__main__":
    unittest.main()
