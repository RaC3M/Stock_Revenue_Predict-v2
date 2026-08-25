import inspect
import unittest

import numpy as np
import pandas as pd

from ensemble_forecast import forecast_engine


class EnsembleModelContractTests(unittest.TestCase):
    def test_public_forecast_interface_contains_only_supported_models(self) -> None:
        self.assertEqual(
            forecast_engine.FORECAST_MODEL_NAMES,
            ("XGBoost", "LightGBM", "CatBoost", "SeasonalQuantile"),
        )
        self.assertEqual(tuple(inspect.signature(forecast_engine.build_forecast).parameters), ("selected_stock",))

    def test_app_summary_contract_uses_formatted_revenue_and_latest_yield(self) -> None:
        forecast = pd.DataFrame(
            {
                "revenue_month": [1, 2],
                "ensemble_revenue": [100.0, 200.0],
                "estimated_cash_dividend": [3.0, 3.0],
                "annual_eps": [5.0, 5.0],
                "stock_price": [50.0, 50.0],
                "stock_price_date": pd.to_datetime(["2025-01-31", "2025-02-28"]),
                "dividend_yield_percent": [6.0, 6.0],
                "payout_ratio": [0.6, 0.6],
                "as_of_price_date": pd.to_datetime(["2024-12-31", "2024-12-31"]),
                "as_of_stock_price": [40.0, 40.0],
                "as_of_price_yield_percent": [7.5, 7.5],
            }
        )
        metrics = pd.DataFrame({"model": ["XGBoost"], "MAPE": [4.5]})
        weights = pd.DataFrame(
            {
                "model": ["LightGBM", "XGBoost"],
                "validation_mape": [3.2, 4.5],
                "weight": [0.7, 0.3],
            }
        )

        revenue_summary = forecast_engine.make_revenue_summary(forecast, metrics, weights)
        yield_summary = forecast_engine.make_yield_summary(forecast)

        self.assertEqual(revenue_summary["annual_total"], "300")
        self.assertEqual(revenue_summary["best_model"], "LightGBM")
        self.assertEqual(revenue_summary["best_mape"], "3.20%")
        self.assertEqual(yield_summary["latest_yield"], "6.00%")
        self.assertEqual(yield_summary["as_of_yield"], "7.50%")
        self.assertEqual(yield_summary["as_of_price"], "40.00")

    def test_revenue_summary_does_not_use_backtest_mape_as_validation_mape(self) -> None:
        forecast = pd.DataFrame(
            {
                "revenue_month": [1, 2],
                "ensemble_revenue": [100.0, 200.0],
            }
        )
        metrics = pd.DataFrame({"model": ["XGBoost"], "MAPE": [4.5]})

        summary = forecast_engine.make_revenue_summary(forecast, metrics)

        self.assertEqual(summary["best_model"], "無")
        self.assertEqual(summary["best_mape"], "無")

    def test_model_recommendation_handles_missing_validation_mape(self) -> None:
        metrics = pd.DataFrame({"model": ["XGBoost"], "MAPE": [12.3]})
        weights = pd.DataFrame(
            {
                "model": ["XGBoost", "LightGBM"],
                "validation_mape": [pd.NA, pd.NA],
                "weight": [0.5, 0.5],
            }
        )

        recommendation = forecast_engine._build_model_recommendation(metrics, weights)

        self.assertEqual(recommendation["historical_best_model"], "無")
        self.assertEqual(recommendation["highest_weight_model"], "XGBoost")
        self.assertEqual(recommendation["actual_best_model"], "XGBoost")

    def test_validation_report_scores_the_actual_weighted_ensemble_predictions(self) -> None:
        rows = []
        for year in [2023, 2024]:
            for month in range(1, 13):
                rows.append(
                    {
                        "stock_id": 1101,
                        "revenue_year": year,
                        "revenue_month": month,
                        "revenue_thousand": 100.0,
                    }
                )
        revenue = pd.DataFrame(rows)

        original_forecaster = forecast_engine._forecast_model_by_name
        try:
            def fake_forecaster(df, selected_stock, target_year, model_name, features=None):
                prediction = 50.0 if model_name == "Low" else 150.0
                return pd.DataFrame(
                    {
                        "revenue_year": target_year,
                        "revenue_month": list(range(1, 13)),
                        "model": model_name,
                        "predicted_revenue": [prediction] * 12,
                    }
                )

            forecast_engine._forecast_model_by_name = fake_forecaster
            report = forecast_engine._build_validation_weights(revenue, 1101, ["Low", "High"])
        finally:
            forecast_engine._forecast_model_by_name = original_forecaster

        self.assertTrue((report["validation_mape"] == 50.0).all())
        self.assertTrue((report["ensemble_validation_mape"] == 0.0).all())
        self.assertTrue((report["validation_year_count"] == 2).all())

        recommendation = forecast_engine._build_model_recommendation(pd.DataFrame(), report)
        self.assertEqual(recommendation["historical_ensemble_mape"], "0.00%")

    def test_seasonal_quantile_falls_back_for_partial_prior_year_history(self) -> None:
        revenue = pd.DataFrame(
            {
                "stock_id": [3150, 3150, 3150],
                "revenue_year": [2024, 2024, 2024],
                "revenue_month": [10, 11, 12],
                "revenue_thousand": [100.0, 120.0, 150.0],
            }
        )

        forecast = forecast_engine._seasonal_quantile_forecast(revenue, 3150, 2025)

        self.assertEqual(forecast["revenue_month"].tolist(), list(range(1, 13)))
        self.assertTrue((forecast["predicted_revenue"] >= 0).all())

    def test_historical_payout_ratio_excludes_target_year_ex_dividend(self) -> None:
        dividends = pd.DataFrame(
            {
                "stock_id": [1101, 1101],
                "fiscal_year": [2023, 2024],
                "ex_dividend_year": [2024, 2025],
                "TotalCashDividend": [2.0, 9.0],
            }
        )
        eps = pd.DataFrame(
            {
                "stock_id": [1101] * 8,
                "eps_year": [2023] * 4 + [2024] * 4,
                "latest_eps": [1.0] * 4 + [2.5] * 4,
            }
        )

        original_dividends = forecast_engine.load_cash_dividend_data
        original_eps = forecast_engine.load_eps_data
        try:
            forecast_engine.load_cash_dividend_data = lambda path=None: dividends
            forecast_engine.load_eps_data = lambda path=None: eps

            payout_ratio, source = forecast_engine._get_historical_payout_ratio(1101, 2025)
        finally:
            forecast_engine.load_cash_dividend_data = original_dividends
            forecast_engine.load_eps_data = original_eps

        self.assertAlmostEqual(payout_ratio, 0.5)
        self.assertIn("time-safe", source)

    def test_missing_stock_payout_uses_historical_cross_section_not_random_policy(self) -> None:
        dividends = pd.DataFrame(
            {
                "stock_id": [2201, 3301],
                "fiscal_year": [2023, 2023],
                "ex_dividend_year": [2024, 2024],
                "TotalCashDividend": [2.0, 3.0],
            }
        )
        eps = pd.DataFrame(
            {
                "stock_id": [2201] * 4 + [3301] * 4,
                "eps_year": [2023] * 8,
                "latest_eps": [1.0] * 4 + [1.5] * 4,
            }
        )

        original_policy = forecast_engine.load_dividend_policy_data
        original_dividends = forecast_engine.load_cash_dividend_data
        original_eps = forecast_engine.load_eps_data
        try:
            forecast_engine.load_dividend_policy_data = lambda path=None: (_ for _ in ()).throw(
                FileNotFoundError
            )
            forecast_engine.load_cash_dividend_data = lambda path=None: dividends
            forecast_engine.load_eps_data = lambda path=None: eps

            policy = forecast_engine._get_dividend_policy(1101, 2025)
        finally:
            forecast_engine.load_dividend_policy_data = original_policy
            forecast_engine.load_cash_dividend_data = original_dividends
            forecast_engine.load_eps_data = original_eps

        self.assertAlmostEqual(policy["payout_ratio"], 0.5)
        self.assertIn("cross-sectional", policy["source"])

    def test_eps_estimate_excludes_statements_unavailable_at_forecast_cutoff(self) -> None:
        revenue_rows = []
        for year, annual_revenue in [(2023, 1_200.0), (2024, 2_400.0)]:
            for month in range(1, 13):
                revenue_rows.append(
                    {
                        "stock_id": 1101,
                        "revenue_year": year,
                        "revenue_month": month,
                        "revenue_thousand": annual_revenue / 12,
                    }
                )
        revenue = pd.DataFrame(revenue_rows)
        eps = pd.DataFrame(
            {
                "stock_id": [1101] * 8,
                "eps_year": [2023] * 4 + [2024] * 4,
                "latest_eps": [3.0] * 4 + [10.0] * 4,
                "available_date": pd.to_datetime(
                    [
                        "2023-05-15",
                        "2023-08-14",
                        "2023-11-14",
                        "2024-03-31",
                        "2024-05-15",
                        "2024-08-14",
                        "2024-11-14",
                        "2025-03-31",
                    ]
                ),
            }
        )

        original_eps = forecast_engine.load_eps_data
        try:
            forecast_engine.load_eps_data = lambda path=None: eps
            estimated_eps, reference_year, _ = forecast_engine._estimate_eps_from_revenue_forecast(
                1101,
                2025,
                2_400.0,
                revenue,
            )
        finally:
            forecast_engine.load_eps_data = original_eps

        self.assertEqual(reference_year, 2023)
        self.assertAlmostEqual(estimated_eps, 24.0)

    def test_stock_price_gaps_use_last_known_real_close_and_never_simulation(self) -> None:
        prices = pd.DataFrame(
            {
                "stock_id": [1101, 1101],
                "price_year": [2024, 2025],
                "price_month": [12, 3],
                "price_date": pd.to_datetime(["2024-12-31", "2025-03-31"]),
                "close_price": [40.0, 50.0],
                "price_source": ["real.csv", "real.csv"],
            }
        )

        original_loader = forecast_engine.load_stock_price_data
        try:
            forecast_engine.load_stock_price_data = lambda **kwargs: prices
            result = forecast_engine._get_stock_prices(pd.DataFrame(), 1101, 2025)
        finally:
            forecast_engine.load_stock_price_data = original_loader

        self.assertEqual(result["close_price"].tolist(), [40.0, 40.0] + [50.0] * 10)
        self.assertIn("last known close", result.loc[0, "price_source"])
        self.assertNotIn("last known close", result.loc[2, "price_source"])
        self.assertFalse(result["price_source"].str.contains("simulat", case=False).any())

    def test_stock_price_is_unavailable_when_no_real_price_exists(self) -> None:
        original_loader = forecast_engine.load_stock_price_data
        try:
            forecast_engine.load_stock_price_data = lambda **kwargs: pd.DataFrame()
            result = forecast_engine._get_stock_prices(pd.DataFrame(), 1101, 2025)
        finally:
            forecast_engine.load_stock_price_data = original_loader

        self.assertEqual(len(result), 12)
        self.assertTrue(np.isnan(result["close_price"]).all())
        self.assertTrue((result["price_source"] == "stock price unavailable").all())


if __name__ == "__main__":
    unittest.main()
