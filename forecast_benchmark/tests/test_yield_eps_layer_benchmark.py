from __future__ import annotations

import unittest

import pandas as pd

from forecast_benchmark.yield_eps_layer_benchmark import (
    build_yield_eps_layer_improvement_vs_current,
    build_yield_eps_layer_overall_accuracy,
    build_yield_eps_layer_stock_accuracy,
    build_yield_eps_layer_winner_summary,
)


class YieldEpsLayerBenchmarkTests(unittest.TestCase):
    def test_stock_and_overall_accuracy_keep_eps_method_dimension(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_family": ["a", "a", "a", "a"],
                "model": ["M", "M", "M", "M"],
                "eps_method": ["current_ratio", "current_ratio", "elastic_net_annual", "elastic_net_annual"],
                "stock_id": [1, 1, 1, 1],
                "stock_name": ["S", "S", "S", "S"],
                "industry_category": ["I", "I", "I", "I"],
                "predicted_annual_revenue": [1200.0, 1200.0, 1200.0, 1200.0],
                "actual_annual_revenue": [1000.0, 1000.0, 1000.0, 1000.0],
                "annual_revenue_abs_percent_error": [20.0, 20.0, 20.0, 20.0],
                "estimated_eps": [2.0, 2.0, 1.2, 1.2],
                "actual_annual_eps": [1.0, 1.0, 1.0, 1.0],
                "eps_abs_error": [1.0, 1.0, 0.2, 0.2],
                "estimated_cash_dividend": [1.0, 1.0, 0.6, 0.6],
                "actual_cash_dividend_per_share": [0.5, 0.5, 0.5, 0.5],
                "cash_dividend_abs_error": [0.5, 0.5, 0.1, 0.1],
                "predicted_dividend_yield_percent": [2.0, 4.0, 1.2, 2.4],
                "actual_dividend_yield_percent": [1.0, 2.0, 1.0, 2.0],
                "yield_error_percent_point": [1.0, 2.0, 0.2, 0.4],
            }
        )

        stock_accuracy = build_yield_eps_layer_stock_accuracy(predictions)
        overall = build_yield_eps_layer_overall_accuracy(stock_accuracy)
        winners = build_yield_eps_layer_winner_summary(stock_accuracy)
        improvement = build_yield_eps_layer_improvement_vs_current(overall)

        elastic_stock = stock_accuracy[stock_accuracy["eps_method"].eq("elastic_net_annual")].iloc[0]
        elastic_overall = overall[overall["eps_method"].eq("elastic_net_annual")].iloc[0]
        elastic_improvement = improvement[improvement["eps_method"].eq("elastic_net_annual")].iloc[0]

        self.assertAlmostEqual(float(elastic_stock["yield_mae_percent_point"]), 0.3)
        self.assertAlmostEqual(float(elastic_overall["average_yield_mae_percent_point"]), 0.3)
        self.assertEqual(winners.iloc[0]["eps_method"], "elastic_net_annual")
        self.assertAlmostEqual(float(elastic_improvement["average_yield_mae_delta_vs_current"]), 1.2)
        self.assertAlmostEqual(
            float(elastic_improvement["average_yield_mae_improvement_pct_vs_current"]),
            80.0,
        )


if __name__ == "__main__":
    unittest.main()
