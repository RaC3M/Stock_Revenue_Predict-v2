from __future__ import annotations

import unittest

import pandas as pd

from forecast_benchmark.eps_layer_validation import (
    build_selected_test_accuracy,
    score_bucket_methods,
    select_bucket_methods,
    select_stock_methods,
)


class EpsLayerValidationTests(unittest.TestCase):
    def test_select_stock_methods_uses_validation_winner_per_stock(self) -> None:
        validation = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "eps_method": ["current_ratio", "lasso_annual", "current_ratio", "lasso_annual"],
                "eps_abs_error": [2.0, 1.0, 0.5, 0.8],
                "eps_abs_percent_error": [20.0, 10.0, 5.0, 8.0],
                "estimated_eps": [1.0, 2.0, 3.0, 4.0],
                "actual_annual_eps": [2.0, 2.0, 3.5, 3.5],
                "is_oracle": [False, False, False, False],
            }
        )

        selection = select_stock_methods(validation, ["current_ratio", "lasso_annual"])

        self.assertEqual(
            dict(zip(selection["stock_id"], selection["selected_eps_method"], strict=True)),
            {1: "lasso_annual", 2: "current_ratio"},
        )

    def test_bucket_selection_uses_lowest_average_validation_error(self) -> None:
        validation = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "eps_method": ["current_ratio", "lasso_annual", "current_ratio", "lasso_annual"],
                "eps_abs_error": [2.0, 1.0, 4.0, 2.0],
                "is_oracle": [False, False, False, False],
            }
        )
        ratio_stability = pd.DataFrame(
            {
                "stock_id": [1, 2],
                "ratio_stability_bucket": ["unstable_ratio", "unstable_ratio"],
            }
        )

        scores = score_bucket_methods(validation, ratio_stability, ["current_ratio", "lasso_annual"])
        selection = select_bucket_methods(scores)

        self.assertEqual(selection.iloc[0]["selected_eps_method"], "lasso_annual")

    def test_selected_test_accuracy_adds_fixed_and_validation_strategies(self) -> None:
        test = pd.DataFrame(
            {
                "source_family": ["m", "m", "m", "m"],
                "model": ["M", "M", "M", "M"],
                "stock_id": [1, 1, 2, 2],
                "stock_name": ["A", "A", "B", "B"],
                "industry_category": ["I", "I", "I", "I"],
                "eps_method": ["current_ratio", "lasso_annual", "current_ratio", "lasso_annual"],
                "is_oracle": [False, False, False, False],
                "eps_abs_error": [2.0, 1.0, 0.5, 0.8],
                "eps_abs_percent_error": [20.0, 10.0, 5.0, 8.0],
                "estimated_eps": [1.0, 2.0, 3.0, 4.0],
                "actual_annual_eps": [2.0, 2.0, 3.5, 3.5],
                "eps_error": [-1.0, 0.0, -0.5, 0.5],
                "eps_underestimated": [True, False, True, False],
                "annual_revenue_abs_percent_error": [1.0, 1.0, 2.0, 2.0],
            }
        )
        stock_selection = pd.DataFrame(
            {
                "stock_id": [1, 2],
                "selected_eps_method": ["lasso_annual", "current_ratio"],
            }
        )
        bucket_selection = pd.DataFrame(
            {
                "ratio_stability_bucket": ["stable_ratio"],
                "selected_eps_method": ["current_ratio"],
            }
        )
        ratio_stability = pd.DataFrame(
            {
                "stock_id": [1, 2],
                "ratio_stability_bucket": ["stable_ratio", "stable_ratio"],
            }
        )

        selected = build_selected_test_accuracy(
            test,
            stock_selection,
            bucket_selection,
            ratio_stability,
            ["current_ratio", "lasso_annual"],
        )

        self.assertIn("fixed_current_ratio", set(selected["selection_strategy"]))
        self.assertIn("stock_validation_best", set(selected["selection_strategy"]))
        stock_strategy = selected[selected["selection_strategy"].eq("stock_validation_best")]
        self.assertEqual(set(stock_strategy["eps_method"]), {"current_ratio", "lasso_annual"})


if __name__ == "__main__":
    unittest.main()
