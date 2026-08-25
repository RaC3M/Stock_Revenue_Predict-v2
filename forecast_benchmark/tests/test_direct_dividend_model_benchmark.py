from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecast_benchmark.direct_dividend_model_benchmark import (
    BUCKET_SELECTION_STRATEGY,
    BUCKET_SUPPORT_STATUS_FALLBACK,
    BUCKET_SUPPORT_STATUS_SUPPORTED,
    DIRECT_FEATURE_COLUMNS,
    GLOBAL_SELECTION_STRATEGY,
    SELECTED_BUCKET_DIVIDEND_METHOD,
    align_predictions_to_actual_targets,
    attach_dividend_selection_buckets,
    build_actual_cash_dividend_targets,
    build_direct_dividend_estimates,
    build_direct_samples,
    build_threshold_direct_methods,
    build_time_safe_features,
    build_validation_bucket_method_scores,
    build_validation_method_scores,
    direct_method_threshold,
    filter_selected_direct_predictions,
    prepare_time_safe_eps_data,
    prepare_time_safe_revenue_data,
    select_validation_bucket_methods,
    select_validation_direct_method,
    validation_as_of_date_for_year,
)
from forecast_benchmark.dividend_layer_benchmark import build_annual_cash_dividend


class DirectDividendModelBenchmarkTests(unittest.TestCase):
    def test_threshold_methods_are_generated_from_sweep_values(self) -> None:
        methods = build_threshold_direct_methods([0.25, 0.35, 0.5])

        self.assertIn("direct_hurdle_last_known", methods)
        self.assertIn("direct_hurdle_ridge_t025", methods)
        self.assertIn("direct_hurdle_ridge_t035", methods)
        self.assertIn("direct_hurdle_elastic_net_t050", methods)
        self.assertAlmostEqual(direct_method_threshold("direct_hurdle_ridge_t025"), 0.25)
        self.assertAlmostEqual(direct_method_threshold("direct_hurdle_ridge_t060"), 0.60)

    def test_validation_as_of_date_applies_explicit_month_day_to_each_fold_year(self) -> None:
        self.assertEqual(validation_as_of_date_for_year(2022, "2024-02-15"), pd.Timestamp(2022, 2, 15))

    def test_time_safe_features_exclude_post_as_of_dividends_and_eps(self) -> None:
        annual_cash = build_annual_cash_dividend(
            pd.DataFrame(
                {
                    "stock_id": [1, 1],
                    "fiscal_year": [2023, 2024],
                    "TotalCashDividend": [2.0, 9.0],
                    "ex_dividend_year": [2024, 2025],
                    "CashExDividendTradingDate": ["2024-07-01", "2025-07-01"],
                    "AnnouncementDate": ["2024-06-01", "2025-06-01"],
                }
            ),
            target_year=2025,
        )
        eps = prepare_time_safe_eps_data(
            pd.DataFrame(
                {
                    "stock_id": [1, 1],
                    "date": ["2024-09-30", "2024-12-31"],
                    "EPS": [1.5, 10.0],
                    "statement_available_date": ["2024-11-14", "2025-03-31"],
                }
            )
        )
        revenue = prepare_time_safe_revenue_data(
            pd.DataFrame(
                {
                    "stock_id": [1, 1],
                    "revenue_year": [2024, 2024],
                    "revenue_month": [11, 12],
                    "revenue_thousand": [100.0, 200.0],
                    "mom": [0.1, 0.2],
                    "yoy": [0.3, 0.4],
                    "revenue_available_date": ["2024-12-10", "2025-01-10"],
                }
            )
        )

        features = build_time_safe_features(
            revenue,
            eps,
            annual_cash,
            stock_id=1,
            target_year=2025,
            as_of_date=pd.Timestamp(2025, 1, 10),
        )

        self.assertAlmostEqual(float(features["last_known_cash_dividend"]), 2.0)
        self.assertAlmostEqual(float(features["latest_available_eps"]), 1.5)
        self.assertAlmostEqual(float(features["known_revenue_ltm"]), 300.0)
        self.assertEqual(int(features["dividend_history_count"]), 1)

    def test_bucket_labels_are_based_on_time_safe_history_features(self) -> None:
        samples = attach_dividend_selection_buckets(
            pd.DataFrame(
                {
                    "recent_paid_rate": [1.0, 0.33, 0.0, np.nan],
                    "dividend_history_count": [3, 2, 1, 0],
                    "last_known_cash_dividend": [1.0, 0.5, 0.0, np.nan],
                }
            )
        )

        self.assertEqual(samples.loc[0, "paid_rate_bucket"], "paid_high")
        self.assertEqual(samples.loc[1, "paid_rate_bucket"], "paid_mixed")
        self.assertEqual(samples.loc[2, "latest_dividend_bucket"], "latest_zero")
        self.assertEqual(samples.loc[3, "dividend_selection_bucket"], "paid_no_history|history_none|latest_missing")

    def test_actual_targets_treat_missing_target_year_as_zero_for_known_dividend_stocks(self) -> None:
        targets = build_actual_cash_dividend_targets(
            pd.DataFrame(
                {
                    "stock_id": [1, 2],
                    "TotalCashDividend": [1.2, 3.4],
                    "ex_dividend_year": [2025, 2024],
                }
            ),
            stock_ids=[1, 2, 3],
            target_years=2025,
        )

        by_stock = targets.set_index("stock_id")
        self.assertAlmostEqual(float(by_stock.loc[1, "actual_cash_dividend_per_share"]), 1.2)
        self.assertAlmostEqual(float(by_stock.loc[2, "actual_cash_dividend_per_share"]), 0.0)
        self.assertTrue(np.isnan(by_stock.loc[3, "actual_cash_dividend_per_share"]))
        self.assertEqual(
            by_stock.loc[2, "actual_cash_dividend_source"],
            "no target-year ex-dividend record treated as zero",
        )

    def test_heuristic_hurdle_estimate_runs_classification_before_amount(self) -> None:
        samples = pd.DataFrame(
            {
                "source_family": ["x", "x"],
                "model": ["m", "m"],
                "stock_id": [1, 2],
                "stock_name": ["A", "B"],
                "industry_category": ["I", "I"],
                "target_year": [2025, 2025],
                "predicted_annual_revenue": [np.nan, np.nan],
                "actual_annual_revenue": [np.nan, np.nan],
                "annual_revenue_abs_percent_error": [np.nan, np.nan],
                "as_of_date": ["2025-01-10", "2025-01-10"],
                "actual_cash_dividend_per_share": [2.0, 0.0],
                "actual_cash_dividend_record_count": [1, 0],
                "actual_dividend_paid": [True, False],
                "actual_cash_dividend_source": ["target", "zero"],
                "has_known_dividend_data": [True, True],
                "last_known_cash_dividend": [2.0, 4.0],
                "recent_cash_dividend_median": [2.0, 4.0],
                "recent_cash_dividend_mean": [2.0, 4.0],
                "recent_cash_dividend_smoothed": [2.0, 4.0],
                "recent_paid_rate": [1.0, 0.0],
                "dividend_history_count": [3, 3],
                "years_since_last_dividend_reference": [2.0, 2.0],
                "latest_available_eps": [1.0, 1.0],
                "available_eps_ttm": [4.0, 4.0],
                "available_eps_ttm_yoy": [0.1, 0.1],
                "eps_history_count": [4, 4],
                "known_revenue_ltm": [100.0, 100.0],
                "known_revenue_recent_3m": [30.0, 30.0],
                "known_revenue_yoy_mean": [0.1, 0.1],
                "known_revenue_mom_mean": [0.1, 0.1],
                "revenue_history_count": [12, 12],
            }
        )

        estimates = build_direct_dividend_estimates(
            samples,
            training_samples=samples,
            methods=["direct_hurdle_recent_median"],
        )

        self.assertAlmostEqual(float(estimates.loc[0, "estimated_cash_dividend"]), 2.0)
        self.assertAlmostEqual(float(estimates.loc[1, "estimated_cash_dividend"]), 0.0)
        self.assertFalse(bool(estimates.loc[1, "predicted_dividend_paid"]))

    def test_ml_hurdle_falls_back_to_history_when_training_features_are_all_missing(self) -> None:
        samples = pd.DataFrame(
            {
                "source_family": ["x", "x", "x"],
                "model": ["m", "m", "m"],
                "stock_id": [1, 2, 3],
                "stock_name": ["A", "B", "C"],
                "industry_category": ["I", "I", "I"],
                "target_year": [2025, 2025, 2025],
                "predicted_annual_revenue": [np.nan, np.nan, np.nan],
                "actual_annual_revenue": [np.nan, np.nan, np.nan],
                "annual_revenue_abs_percent_error": [np.nan, np.nan, np.nan],
                "as_of_date": ["2025-01-10", "2025-01-10", "2025-01-10"],
                "actual_cash_dividend_per_share": [1.0, 0.0, 3.0],
                "actual_cash_dividend_record_count": [1, 0, 1],
                "actual_dividend_paid": [True, False, True],
                "actual_cash_dividend_source": ["target", "zero", "target"],
                "has_known_dividend_data": [True, True, True],
            }
        )
        for column in DIRECT_FEATURE_COLUMNS:
            samples[column] = np.nan

        estimates = build_direct_dividend_estimates(
            samples,
            training_samples=samples,
            methods=["direct_hurdle_ridge_t050"],
        )

        self.assertAlmostEqual(float(estimates.loc[0, "predicted_dividend_paid_probability"]), 2 / 3)
        self.assertTrue(estimates["predicted_dividend_paid"].all())
        self.assertEqual(estimates["estimated_cash_dividend"].tolist(), [2.0, 2.0, 2.0])

    def test_selection_uses_validation_cash_dividend_error_before_yield(self) -> None:
        selection = select_validation_direct_method(
            pd.DataFrame(
                {
                    "source_family": ["v", "v"],
                    "model": ["m", "m"],
                    "eps_method": ["time_safe_features", "time_safe_features"],
                    "dividend_method": ["direct_hurdle_last_known", "direct_hurdle_smoothed"],
                    "average_cash_dividend_abs_error": [1.0, 0.5],
                    "average_yield_mae_percent_point": [0.1, 9.0],
                }
            )
        )

        self.assertEqual(selection.iloc[0]["selected_dividend_method"], "direct_hurdle_smoothed")
        self.assertEqual(selection.iloc[0]["selection_strategy"], GLOBAL_SELECTION_STRATEGY)

    def test_method_scores_aggregate_multiple_validation_folds(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "stock_id": [1, 1, 1, 1],
                "validation_fold_year": [2023, 2024, 2023, 2024],
                "eps_method": ["time_safe_features"] * 4,
                "dividend_method": [
                    "direct_hurdle_ridge_t050",
                    "direct_hurdle_ridge_t050",
                    "direct_hurdle_ridge_t035",
                    "direct_hurdle_ridge_t035",
                ],
                "cash_dividend_abs_error": [1.0, 1.0, 0.5, 0.5],
                "yield_mae_percent_point": [2.0, 2.0, 3.0, 3.0],
            }
        )

        scores = build_validation_method_scores(stock_accuracy)
        selection = select_validation_direct_method(scores)

        self.assertEqual(selection.iloc[0]["selected_dividend_method"], "direct_hurdle_ridge_t035")
        self.assertEqual(int(selection.iloc[0]["validation_fold_count"]), 2)

    def test_bucket_method_selection_selects_method_per_bucket(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "stock_id": [1, 2, 1, 2],
                "validation_fold_year": [2024, 2024, 2024, 2024],
                "eps_method": ["time_safe_features"] * 4,
                "dividend_selection_bucket": ["A", "B", "A", "B"],
                "paid_rate_bucket": ["paid_high", "paid_none", "paid_high", "paid_none"],
                "dividend_history_bucket": ["history_enough"] * 4,
                "latest_dividend_bucket": ["latest_positive", "latest_zero", "latest_positive", "latest_zero"],
                "dividend_method": [
                    "direct_hurdle_ridge_t050",
                    "direct_hurdle_ridge_t050",
                    "direct_hurdle_ridge_t035",
                    "direct_hurdle_ridge_t035",
                ],
                "cash_dividend_abs_error": [1.0, 0.2, 0.5, 0.8],
                "yield_mae_percent_point": [1.0, 1.0, 1.0, 1.0],
            }
        )

        scores = build_validation_bucket_method_scores(stock_accuracy)
        selection = select_validation_bucket_methods(scores)
        selected = selection.set_index("dividend_selection_bucket")["selected_dividend_method"].to_dict()

        self.assertEqual(selected["A"], "direct_hurdle_ridge_t035")
        self.assertEqual(selected["B"], "direct_hurdle_ridge_t050")
        self.assertEqual(set(selection["selection_strategy"]), {BUCKET_SELECTION_STRATEGY})

    def test_bucket_method_selection_falls_back_to_global_when_support_is_too_small(self) -> None:
        scores = pd.DataFrame(
            {
                "dividend_selection_bucket": ["A", "B"],
                "eps_method": ["time_safe_features", "time_safe_features"],
                "dividend_method": ["direct_hurdle_ridge_t035", "direct_hurdle_ridge_t025"],
                "paid_rate_bucket": ["paid_mixed", "paid_high"],
                "dividend_history_bucket": ["history_sparse", "history_enough"],
                "latest_dividend_bucket": ["latest_positive", "latest_positive"],
                "average_cash_dividend_abs_error": [0.1, 0.2],
                "average_yield_mae_percent_point": [0.3, 0.4],
                "validation_fold_count": [1, 2],
                "validation_stock_year_count": [3, 20],
            }
        )

        selection = select_validation_bucket_methods(
            scores,
            global_selected_method="direct_hurdle_ridge_t060",
            min_bucket_folds=2,
            min_bucket_stock_years=15,
        )
        by_bucket = selection.set_index("dividend_selection_bucket")

        self.assertEqual(by_bucket.loc["A", "selected_dividend_method"], "direct_hurdle_ridge_t060")
        self.assertEqual(by_bucket.loc["A", "bucket_winner_dividend_method"], "direct_hurdle_ridge_t035")
        self.assertEqual(by_bucket.loc["A", "bucket_support_status"], BUCKET_SUPPORT_STATUS_FALLBACK)
        self.assertTrue(bool(by_bucket.loc["A", "fallback_to_global"]))
        self.assertEqual(by_bucket.loc["B", "selected_dividend_method"], "direct_hurdle_ridge_t025")
        self.assertEqual(by_bucket.loc["B", "bucket_support_status"], BUCKET_SUPPORT_STATUS_SUPPORTED)
        self.assertFalse(bool(by_bucket.loc["B", "fallback_to_global"]))

    def test_bucket_selected_predictions_are_aggregated_under_strategy_method(self) -> None:
        predictions = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "source_family": ["x"] * 4,
                "model": ["m"] * 4,
                "target_year": [2025] * 4,
                "target_month": [1] * 4,
                "dividend_selection_bucket": ["A", "A", "B", "B"],
                "dividend_method": [
                    "direct_hurdle_ridge_t050",
                    "direct_hurdle_ridge_t035",
                    "direct_hurdle_ridge_t050",
                    "direct_hurdle_ridge_t035",
                ],
            }
        )
        global_selection = pd.DataFrame(
            {
                "selection_strategy": [GLOBAL_SELECTION_STRATEGY],
                "selection_source": ["global"],
                "selected_dividend_method": ["direct_hurdle_ridge_t050"],
            }
        )
        bucket_selection = pd.DataFrame(
            {
                "dividend_selection_bucket": ["A", "B"],
                "selected_dividend_method": ["direct_hurdle_ridge_t035", "direct_hurdle_ridge_t050"],
            }
        )

        selected = filter_selected_direct_predictions(predictions, global_selection, bucket_selection)
        bucket_selected = selected[selected["selection_strategy"].eq(BUCKET_SELECTION_STRATEGY)]

        self.assertEqual(set(bucket_selected["dividend_method"]), {SELECTED_BUCKET_DIVIDEND_METHOD})
        self.assertEqual(
            bucket_selected.sort_values("stock_id")["selected_underlying_dividend_method"].tolist(),
            ["direct_hurdle_ridge_t035", "direct_hurdle_ridge_t050"],
        )

    def test_bucket_selected_predictions_keep_fallback_metadata(self) -> None:
        predictions = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "source_family": ["x"] * 4,
                "model": ["m"] * 4,
                "target_year": [2025] * 4,
                "target_month": [1] * 4,
                "dividend_selection_bucket": ["A", "A", "B", "B"],
                "dividend_method": [
                    "direct_hurdle_ridge_t060",
                    "direct_hurdle_ridge_t035",
                    "direct_hurdle_ridge_t060",
                    "direct_hurdle_ridge_t025",
                ],
            }
        )
        global_selection = pd.DataFrame(
            {
                "selection_strategy": [GLOBAL_SELECTION_STRATEGY],
                "selection_source": ["global"],
                "selected_dividend_method": ["direct_hurdle_ridge_t060"],
            }
        )
        bucket_selection = pd.DataFrame(
            {
                "dividend_selection_bucket": ["A", "B"],
                "selected_dividend_method": ["direct_hurdle_ridge_t060", "direct_hurdle_ridge_t025"],
                "bucket_winner_dividend_method": ["direct_hurdle_ridge_t035", "direct_hurdle_ridge_t025"],
                "bucket_support_status": [BUCKET_SUPPORT_STATUS_FALLBACK, BUCKET_SUPPORT_STATUS_SUPPORTED],
                "fallback_to_global": [True, False],
            }
        )

        selected = filter_selected_direct_predictions(predictions, global_selection, bucket_selection)
        bucket_selected = (
            selected[selected["selection_strategy"].eq(BUCKET_SELECTION_STRATEGY)]
            .sort_values("stock_id")
            .reset_index(drop=True)
        )

        self.assertEqual(bucket_selected.loc[0, "selected_underlying_dividend_method"], "direct_hurdle_ridge_t060")
        self.assertEqual(bucket_selected.loc[0, "bucket_winner_dividend_method"], "direct_hurdle_ridge_t035")
        self.assertTrue(bool(bucket_selected.loc[0, "fallback_to_global"]))
        self.assertEqual(bucket_selected.loc[1, "selected_underlying_dividend_method"], "direct_hurdle_ridge_t025")
        self.assertFalse(bool(bucket_selected.loc[1, "fallback_to_global"]))

    def test_baseline_alignment_recomputes_zero_label_errors(self) -> None:
        predictions = pd.DataFrame(
            {
                "stock_id": [1],
                "target_year": [2025],
                "estimated_cash_dividend": [1.5],
                "actual_cash_dividend_per_share": [np.nan],
                "stock_price_valid_for_yield": [True],
                "stock_price": [50.0],
                "predicted_dividend_yield_percent": [3.0],
            }
        )
        actual_targets = pd.DataFrame(
            {
                "stock_id": [1],
                "target_year": [2025],
                "actual_cash_dividend_per_share": [0.0],
                "actual_cash_dividend_record_count": [0],
                "actual_dividend_paid": [False],
                "actual_cash_dividend_source": ["no target-year ex-dividend record treated as zero"],
                "has_known_dividend_data": [True],
            }
        )

        aligned = align_predictions_to_actual_targets(predictions, actual_targets)

        self.assertAlmostEqual(float(aligned.loc[0, "cash_dividend_abs_error"]), 1.5)
        self.assertAlmostEqual(float(aligned.loc[0, "actual_dividend_yield_percent"]), 0.0)
        self.assertAlmostEqual(float(aligned.loc[0, "yield_abs_error_percent_point"]), 3.0)

    def test_build_direct_samples_uses_row_specific_as_of_date(self) -> None:
        revenue = prepare_time_safe_revenue_data(
            pd.DataFrame(
                {
                    "stock_id": [1],
                    "revenue_year": [2023],
                    "revenue_month": [12],
                    "revenue_thousand": [100.0],
                }
            )
        )
        eps = prepare_time_safe_eps_data(
            pd.DataFrame({"stock_id": [1], "date": ["2023-09-30"], "EPS": [1.0]})
        )
        annual_cash = build_annual_cash_dividend(
            pd.DataFrame(
                {
                    "stock_id": [1],
                    "fiscal_year": [2022],
                    "TotalCashDividend": [0.5],
                    "ex_dividend_year": [2023],
                    "AnnouncementDate": ["2023-06-01"],
                }
            ),
            target_year=2025,
        )
        context = pd.DataFrame(
            {
                "source_family": ["h"],
                "model": ["m"],
                "stock_id": [1],
                "stock_name": ["A"],
                "industry_category": ["I"],
                "target_year": [2024],
                "predicted_annual_revenue": [np.nan],
                "actual_annual_revenue": [np.nan],
                "annual_revenue_abs_percent_error": [np.nan],
                "as_of_date": ["2024-01-10"],
            }
        )
        actual_targets = build_actual_cash_dividend_targets(
            pd.DataFrame({"stock_id": [1], "TotalCashDividend": [0.0], "ex_dividend_year": [2024]}),
            stock_ids=[1],
            target_years=2024,
        )

        samples = build_direct_samples(context, revenue, eps, annual_cash, actual_targets)

        self.assertEqual(samples.iloc[0]["as_of_date"], "2024-01-10")
        self.assertAlmostEqual(float(samples.iloc[0]["last_known_cash_dividend"]), 0.5)


if __name__ == "__main__":
    unittest.main()
