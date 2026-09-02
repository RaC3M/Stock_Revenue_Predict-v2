from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from financial_forecast import FinancialForecastPolicy, forecast_financials
from financial_forecast.contracts import EPS_METHOD_KNOWN_QUARTERS, DIVIDEND_METHOD_FIVE_YEAR_MEAN
from financial_forecast.evidence import FinancialEvidence
from financial_forecast.live_methods import estimate_five_year_dividends


class LiveFinancialPipelineTests(unittest.TestCase):
    def test_known_eps_five_year_average_and_one_as_of_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revenue, eps, dividends = [], [], []
            for year in range(2021, 2026):
                for month in range(1, 13):
                    revenue.append({"stock_id": 1101, "revenue_year": year, "revenue_month": month,
                        "revenue_thousand": 100, "revenue_available_date": pd.Timestamp(year, month, 1) + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)})
                for quarter in range(1, 5):
                    eps.append({"stock_id": 1101, "date": f"{year}-{quarter * 3:02d}-28", "EPS": 1,
                        "statement_available_date": f"{year + (quarter == 4)}-{[5, 8, 11, 3][quarter - 1]:02d}-15"})
                dividends.append({"stock_id": 1101, "fiscal_year": year, "TotalCashDividend": (year - 2020) * 2,
                    "DividendAvailableDate": f"{year + 1}-05-01", "CashExDividendTradingDate": f"{year + 1}-07-01"})
            eps.extend([
                {"stock_id": 1101, "date": "2026-03-31", "EPS": 9, "statement_available_date": "2026-05-15"},
                {"stock_id": 1101, "date": "2026-06-30", "EPS": 99, "statement_available_date": "2026-08-14"},
            ])
            pd.DataFrame(revenue).to_csv(root / "Stock_revenue_2019~2025.csv", index=False)
            pd.DataFrame(eps).to_csv(root / "EPS2020~2025.csv", index=False)
            pd.DataFrame(dividends).to_csv(root / "Dividend2019~2025.csv", index=False)
            pd.DataFrame([{"stock_id": 1101, "date": "2026-07-27", "close": 100},
                {"stock_id": 1101, "date": "2027-01-02", "close": 1}]).to_csv(root / "day K2020~2025.csv", index=False)
            predictions = pd.DataFrame({"source_family": "hybrid", "model": "hybrid", "stock_id": 1101,
                "target_year": 2026, "target_month": range(1, 13), "predicted_revenue": [200] * 12})
            result = forecast_financials(predictions, target_year=2026, as_of_date="2026-07-28", data_dir=root,
                policy=FinancialForecastPolicy(eps_methods=(EPS_METHOD_KNOWN_QUARTERS,),
                    dividend_methods=(DIVIDEND_METHOD_FIVE_YEAR_MEAN,), yield_modes=("as_of_price_yield",)))
            quarters = result.quarterly_eps_estimates.set_index("eps_quarter")
            self.assertEqual(quarters.loc[1, "eps_basis"], "actual")
            self.assertAlmostEqual(float(quarters.loc[1, "quarter_eps"]), 9)
            self.assertEqual(quarters.loc[2, "eps_basis"], "seasonal_estimate")
            self.assertAlmostEqual(float(quarters.loc[2, "quarter_eps"]), 2)
            summary = result.summary.iloc[0]
            self.assertAlmostEqual(float(summary["estimated_eps"]), 15)
            self.assertAlmostEqual(float(summary["payout_ratio"]), 1.5)
            self.assertEqual(int(summary["payout_valid_years"]), 5)
            self.assertAlmostEqual(float(summary["estimated_cash_dividend"]), 22.5)
            self.assertAlmostEqual(float(summary["as_of_price_yield_percent"]), 22.5)
            self.assertEqual(summary["as_of_price_date"], pd.Timestamp("2026-07-27"))
            future_dividend = {"stock_id": 1101, "fiscal_year": 2025, "TotalCashDividend": 9999,
                "DividendAvailableDate": "2026-08-01", "CashExDividendTradingDate": "2026-08-15"}
            pd.DataFrame([*dividends, future_dividend]).to_csv(root / "Dividend2019~2025.csv", index=False)
            predictions["target_year"] = 2027
            next_year = forecast_financials(predictions, target_year=2027, as_of_date="2026-07-28", data_dir=root,
                policy=FinancialForecastPolicy(eps_methods=(EPS_METHOD_KNOWN_QUARTERS,),
                    dividend_methods=(DIVIDEND_METHOD_FIVE_YEAR_MEAN,), yield_modes=("as_of_price_yield",)))
            next_summary = next_year.summary.iloc[0]
            self.assertEqual(next_summary["payout_window"], summary["payout_window"])
            self.assertAlmostEqual(float(next_summary["payout_ratio"]), 1.5)
            self.assertEqual(next_summary["as_of_stock_price"], summary["as_of_stock_price"])
            self.assertEqual(next_summary["as_of_price_date"], summary["as_of_price_date"])
            self.assertFalse(next_year.quarterly_eps_estimates["eps_basis"].eq("actual").any())

    def test_payout_keeps_zero_and_large_ratios_and_excludes_invalid_years(self):
        annual = pd.DataFrame([
            {"stock_id": 1101, "eps_year": y, "annual_eps": e}
            for y, e in [(2021, 1), (2022, 1), (2023, -1), (2025, 1), (2026, 1)]
        ])
        dividends = pd.DataFrame([
            {"stock_id": 1101, "fiscal_year": y, "TotalCashDividend": d}
            for y, d in [(2021, 0), (2022, 4), (2023, 1), (2024, 1), (2026, 99)]
        ])
        evidence = FinancialEvidence(pd.DataFrame(), annual, pd.DataFrame(), dividends, pd.DataFrame())
        inputs = pd.DataFrame([{"stock_id": 1101, "estimated_eps": 3, "status": "ok"}])
        estimates, detail = estimate_five_year_dividends(inputs, evidence, pd.Timestamp("2026-09-02"))
        self.assertAlmostEqual(float(estimates.iloc[0]["payout_ratio"]), 2)
        self.assertEqual(int(estimates.iloc[0]["payout_valid_years"]), 2)
        self.assertEqual(detail.loc[detail["included"], "fiscal_year"].tolist(), [2021, 2022])
        self.assertEqual(detail["fiscal_year"].tolist(), [2021, 2022, 2023, 2024, 2025])


if __name__ == "__main__":
    unittest.main()
