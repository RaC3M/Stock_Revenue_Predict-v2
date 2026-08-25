from __future__ import annotations

import unittest

import pandas as pd

from forecast_benchmark.metrics import (
    build_overall_accuracy,
    build_stock_accuracy,
    build_winner_summary,
    compute_metrics,
)


class MetricsTests(unittest.TestCase):
    def test_compute_metrics_matches_rolling_metric_definitions(self) -> None:
        metrics = compute_metrics(
            actual=pd.Series([100.0, 200.0]),
            predicted=pd.Series([110.0, 180.0]),
            last_observed=pd.Series([90.0, 190.0]),
        )

        self.assertAlmostEqual(metrics["MAE"], 15.0)
        self.assertAlmostEqual(metrics["MAPE"], 10.0)
        self.assertAlmostEqual(metrics["MedianAPE"], 10.0)
        self.assertAlmostEqual(metrics["WMAPE"], 10.0)
        self.assertAlmostEqual(metrics["Bias"], -5.0)
        self.assertAlmostEqual(metrics["UnderestimateRate"], 50.0)
        self.assertAlmostEqual(metrics["DirectionAccuracy"], 50.0)

    def test_accuracy_and_winner_summary_use_primary_metric(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_family": ["a", "a", "b", "b"],
                "model": ["A", "A", "B", "B"],
                "stock_id": [1, 2, 1, 2],
                "actual_revenue": [100.0, 200.0, 100.0, 200.0],
                "predicted_revenue": [110.0, 260.0, 105.0, 210.0],
                "last_observed_revenue": [90.0, 190.0, 90.0, 190.0],
            }
        )

        overall = build_overall_accuracy(predictions)
        stock = build_stock_accuracy(predictions)
        winners = build_winner_summary(stock, primary_metric="WMAPE")

        self.assertEqual(overall.iloc[0]["model"], "B")
        self.assertEqual(int(winners[winners["model"].eq("B")]["stock_wins"].iloc[0]), 2)
        self.assertEqual(float(winners[winners["model"].eq("B")]["stock_win_rate"].iloc[0]), 100.0)


if __name__ == "__main__":
    unittest.main()

