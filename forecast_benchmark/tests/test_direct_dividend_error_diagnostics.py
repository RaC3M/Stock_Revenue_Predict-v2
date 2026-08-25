from __future__ import annotations

import unittest

import pandas as pd

from forecast_benchmark.direct_dividend_error_diagnostics import (
    build_bucket_error_summary,
    build_classification_outcomes,
    build_stock_error_comparison,
    classify_dividend_outcome,
)


class DirectDividendErrorDiagnosticsTests(unittest.TestCase):
    def test_stock_error_comparison_computes_bucket_improvement_vs_baseline(self) -> None:
        selected = pd.DataFrame(
            {
                "source_family": ["ensemble_forecast", "ensemble_forecast", "ensemble_forecast", "ensemble_forecast"],
                "model": ["LightGBM", "LightGBM", "LightGBM", "LightGBM"],
                "eps_method": ["time_safe_features"] * 4,
                "dividend_method": [
                    "bucket_validation_best",
                    "bucket_validation_best",
                    "direct_hurdle_ridge_t060",
                    "direct_hurdle_ridge_t060",
                ],
                "stock_id": [1, 2, 1, 2],
                "stock_name": ["A", "B", "A", "B"],
                "industry_category": ["I", "I", "I", "I"],
                "actual_cash_dividend_per_share": [1.0, 2.0, 1.0, 2.0],
                "estimated_cash_dividend": [1.1, 1.5, 0.8, 2.4],
                "cash_dividend_abs_error": [0.1, 0.5, 0.2, 0.4],
                "yield_mae_percent_point": [0.2, 0.7, 0.3, 0.6],
                "dividend_selection_bucket": ["A", "B", "A", "B"],
                "bucket_support_status": ["supported", "fallback_to_global", "supported", "fallback_to_global"],
                "fallback_to_global": [False, True, False, True],
            }
        )
        baseline = pd.DataFrame(
            {
                "source_family": ["ensemble_forecast", "ensemble_forecast"],
                "model": ["LightGBM", "LightGBM"],
                "eps_method": ["current_ratio", "current_ratio"],
                "dividend_method": ["announcement_safe_payout_ratio", "announcement_safe_payout_ratio"],
                "stock_id": [1, 2],
                "estimated_cash_dividend": [1.4, 1.7],
                "cash_dividend_abs_error": [0.4, 0.3],
                "yield_mae_percent_point": [0.8, 0.5],
            }
        )

        comparison = build_stock_error_comparison(selected, baseline, "direct_hurdle_ridge_t060")
        by_stock = comparison.set_index("stock_id")

        self.assertAlmostEqual(float(by_stock.loc[1, "bucket_cash_improvement_vs_baseline"]), 0.3)
        self.assertEqual(by_stock.loc[1, "bucket_vs_baseline_cash_result"], "improved")
        self.assertAlmostEqual(float(by_stock.loc[2, "bucket_cash_improvement_vs_baseline"]), -0.2)
        self.assertEqual(by_stock.loc[2, "bucket_vs_baseline_cash_result"], "worse")

    def test_classify_dividend_outcome_labels_hurdle_errors(self) -> None:
        self.assertEqual(classify_dividend_outcome(True, True), "correct_paid")
        self.assertEqual(classify_dividend_outcome(True, False), "false_negative_missed_dividend")
        self.assertEqual(classify_dividend_outcome(False, True), "false_positive_extra_dividend")
        self.assertEqual(classify_dividend_outcome(False, False), "correct_no_dividend")

    def test_classification_outcomes_use_one_stock_year_row(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_family": ["ensemble_forecast", "ensemble_forecast", "ensemble_forecast"],
                "model": ["LightGBM", "LightGBM", "LightGBM"],
                "eps_method": ["time_safe_features"] * 3,
                "dividend_method": ["bucket_validation_best"] * 3,
                "stock_id": [1, 1, 2],
                "target_month": [1, 2, 1],
                "actual_dividend_paid": [True, True, False],
                "predicted_dividend_paid": [False, False, True],
                "estimated_cash_dividend": [0.0, 0.0, 1.0],
                "actual_cash_dividend_per_share": [2.0, 2.0, 0.0],
                "cash_dividend_abs_error": [2.0, 2.0, 1.0],
            }
        )

        outcomes = build_classification_outcomes(predictions)
        by_stock = outcomes.set_index("stock_id")

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(by_stock.loc[1, "classification_outcome"], "false_negative_missed_dividend")
        self.assertEqual(by_stock.loc[2, "classification_outcome"], "false_positive_extra_dividend")

    def test_bucket_error_summary_aggregates_improvement_counts(self) -> None:
        comparison = pd.DataFrame(
            {
                "stock_id": [1, 2, 3],
                "dividend_selection_bucket": ["A", "A", "B"],
                "fallback_to_global": [False, True, False],
                "bucket_cash_dividend_abs_error": [0.1, 0.5, 0.2],
                "baseline_cash_dividend_abs_error": [0.4, 0.3, 0.6],
                "bucket_cash_improvement_vs_baseline": [0.3, -0.2, 0.4],
                "bucket_yield_mae_percent_point": [0.2, 0.7, 0.3],
                "baseline_yield_mae_percent_point": [0.8, 0.5, 0.9],
                "bucket_yield_improvement_vs_baseline": [0.6, -0.2, 0.6],
                "bucket_vs_baseline_cash_result": ["improved", "worse", "improved"],
            }
        )

        summary = build_bucket_error_summary(comparison).set_index("dividend_selection_bucket")

        self.assertEqual(int(summary.loc["A", "stock_count"]), 2)
        self.assertEqual(int(summary.loc["A", "fallback_stock_count"]), 1)
        self.assertEqual(int(summary.loc["A", "improved_stock_count"]), 1)
        self.assertEqual(int(summary.loc["A", "worse_stock_count"]), 1)

    def test_stock_error_comparison_separates_missing_metrics_from_ties(self) -> None:
        selected = pd.DataFrame(
            {
                "source_family": ["ensemble_forecast"],
                "model": ["LightGBM"],
                "eps_method": ["time_safe_features"],
                "dividend_method": ["bucket_validation_best"],
                "stock_id": [1],
                "cash_dividend_abs_error": [None],
                "yield_mae_percent_point": [None],
            }
        )
        baseline = pd.DataFrame(
            {
                "source_family": ["ensemble_forecast"],
                "model": ["LightGBM"],
                "eps_method": ["current_ratio"],
                "dividend_method": ["announcement_safe_payout_ratio"],
                "stock_id": [1],
                "cash_dividend_abs_error": [None],
                "yield_mae_percent_point": [None],
            }
        )

        comparison = build_stock_error_comparison(selected, baseline, "direct_hurdle_ridge_t060")

        self.assertEqual(comparison.iloc[0]["bucket_vs_baseline_cash_result"], "missing_metric")


if __name__ == "__main__":
    unittest.main()
