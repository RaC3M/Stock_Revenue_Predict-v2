from __future__ import annotations

import unittest

import pandas as pd

from forecast_benchmark.yield_benchmark import (
    build_error_decomposition,
    build_yield_overall_accuracy,
    build_yield_stock_accuracy,
    build_yield_winner_summary,
    is_observed_stock_price_source,
)


class YieldBenchmarkTests(unittest.TestCase):
    def test_stock_and_overall_accuracy_summarize_yield_errors(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_family": ["a", "a", "b", "b"],
                "model": ["A", "A", "B", "B"],
                "stock_id": [1, 1, 1, 1],
                "stock_name": ["S", "S", "S", "S"],
                "industry_category": ["I", "I", "I", "I"],
                "predicted_annual_revenue": [1200.0, 1200.0, 1100.0, 1100.0],
                "actual_annual_revenue": [1000.0, 1000.0, 1000.0, 1000.0],
                "annual_revenue_abs_percent_error": [20.0, 20.0, 10.0, 10.0],
                "estimated_eps": [2.0, 2.0, 1.5, 1.5],
                "actual_annual_eps": [1.0, 1.0, 1.0, 1.0],
                "eps_abs_error": [1.0, 1.0, 0.5, 0.5],
                "estimated_cash_dividend": [1.0, 1.0, 0.7, 0.7],
                "actual_cash_dividend_per_share": [0.5, 0.5, 0.5, 0.5],
                "cash_dividend_abs_error": [0.5, 0.5, 0.2, 0.2],
                "predicted_dividend_yield_percent": [2.0, 4.0, 1.0, 2.0],
                "actual_dividend_yield_percent": [1.0, 2.0, 1.0, 1.0],
                "yield_error_percent_point": [1.0, 2.0, 0.0, 1.0],
            }
        )

        stock_accuracy = build_yield_stock_accuracy(predictions)
        overall = build_yield_overall_accuracy(stock_accuracy)
        winners = build_yield_winner_summary(stock_accuracy)
        decomposition = build_error_decomposition(overall)

        a_stock = stock_accuracy[stock_accuracy["model"].eq("A")].iloc[0]
        b_overall = overall[overall["model"].eq("B")].iloc[0]
        self.assertAlmostEqual(float(a_stock["yield_mae_percent_point"]), 1.5)
        self.assertAlmostEqual(float(b_overall["average_yield_mae_percent_point"]), 0.5)
        self.assertEqual(winners.iloc[0]["model"], "B")
        self.assertEqual(set(decomposition["error_stage"]), {"revenue", "eps", "cash_dividend", "yield"})

    def test_synthetic_stock_prices_are_not_valid_yield_evidence(self) -> None:
        observed = is_observed_stock_price_source(
            pd.Series(
                [
                    "day K2020~2025.csv",
                    "simulated_stock_prices_2025.csv",
                    "synthetic_fallback",
                ]
            )
        )

        self.assertEqual(observed.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
