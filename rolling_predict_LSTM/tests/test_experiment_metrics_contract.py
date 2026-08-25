import unittest

import numpy as np
import pandas as pd

from rolling_predict_LSTM import batch_ablation_study
from rolling_predict_LSTM import batch_all_stocks_penalty
from rolling_predict_LSTM import batch_quarterly_target_ablation
from rolling_predict_LSTM import batch_sequence_backbone_ablation
from rolling_predict_LSTM import batch_ten_scenarios
from rolling_predict_LSTM import experiment_metrics


class ExperimentMetricsContractTests(unittest.TestCase):
    def test_shared_module_exposes_its_purpose_as_a_module_docstring(self) -> None:
        self.assertIn("Shared metric aggregation", experiment_metrics.__doc__ or "")

    def test_metric_record_uses_complete_direction_rows_when_last_observed_exists(self) -> None:
        frame = pd.DataFrame(
            {
                "stock_id": [1, 2, 3, 4],
                "actual_revenue": [100.0, 200.0, np.nan, 300.0],
                "predicted_revenue": [110.0, np.nan, 10.0, 330.0],
                "last_observed_revenue": [90.0, 180.0, 5.0, np.nan],
            }
        )

        record = experiment_metrics.metric_record(frame)

        self.assertEqual(1, record["observations"])
        self.assertEqual(1, record["stock_count"])
        self.assertEqual(10.0, record["MAE"])
        self.assertEqual(100.0, record["DirectionAccuracy"])

    def test_metric_record_supports_predictions_without_direction_baseline(self) -> None:
        frame = pd.DataFrame(
            {
                "stock_id": [1, 2],
                "actual_revenue": [100.0, 200.0],
                "predicted_revenue": [110.0, 180.0],
            }
        )

        record = experiment_metrics.metric_record(frame)

        self.assertEqual(2, record["observations"])
        self.assertEqual(2, record["stock_count"])
        self.assertTrue(np.isnan(record["DirectionAccuracy"]))

    def test_summarize_keeps_null_groups_and_shared_metric_contract(self) -> None:
        frame = pd.DataFrame(
            {
                "variant": ["plain", np.nan],
                "stock_id": [1, 2],
                "actual_revenue": [100.0, 200.0],
                "predicted_revenue": [110.0, 180.0],
                "last_observed_revenue": [90.0, 190.0],
            }
        )

        summary = experiment_metrics.summarize(frame, ["variant"])

        self.assertEqual(2, len(summary))
        self.assertEqual(2, int(summary["observations"].sum()))
        self.assertIn("WMAPE", summary.columns)
        self.assertEqual(1, int(summary["variant"].isna().sum()))

    def test_standard_batch_runners_reuse_shared_metric_functions(self) -> None:
        modules = (
            batch_ablation_study,
            batch_ten_scenarios,
            batch_sequence_backbone_ablation,
            batch_quarterly_target_ablation,
        )

        for module in modules:
            with self.subTest(module=module.__name__):
                self.assertIs(experiment_metrics.metric_record, module.metric_record)
                self.assertIs(experiment_metrics.summarize, module.summarize)

    def test_penalty_runner_preserves_legacy_count_column_order_through_shared_module(self) -> None:
        frame = pd.DataFrame(
            {
                "penalty_setting": ["off_huber"],
                "stock_id": [1],
                "actual_revenue": [100.0],
                "predicted_revenue": [110.0],
                "last_observed_revenue": [90.0],
            }
        )

        summary = batch_all_stocks_penalty.summarize(frame, ["penalty_setting"])

        self.assertEqual(["stock_count", "observations"], list(summary.columns[1:3]))


if __name__ == "__main__":
    unittest.main()
