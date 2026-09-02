from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from financial_forecast.dividend_patterns import analyze_dividend_patterns
from financial_forecast.evidence import FinancialEvidence
from financial_forecast.live_methods import estimate_classified_dividends
from financial_forecast.yield_calc import calculate_yields


class DividendPatternTests(unittest.TestCase):
    def evidence(self, cash, eps=None):
        annual, dividends = [], []
        for stock, amounts in cash.items():
            for year, amount in zip(range(2021, 2026), amounts):
                if amount is not None:
                    dividends.append({"stock_id": stock, "fiscal_year": year,
                        "TotalCashDividend": amount, "available_date": pd.Timestamp(year + 1, 6, 1)})
                value = 10 if eps is None else eps[stock][year - 2021]
                if value is not None:
                    annual.append({"stock_id": stock, "eps_year": year, "annual_eps": value,
                        "latest_available_date": pd.Timestamp(year + 1, 3, 31)})
        return FinancialEvidence(pd.DataFrame(), pd.DataFrame(annual, columns=[
            "stock_id", "eps_year", "annual_eps", "latest_available_date"]), pd.DataFrame(),
            pd.DataFrame(dividends, columns=["stock_id", "fiscal_year", "TotalCashDividend", "available_date"]),
            pd.DataFrame({"stock_id": list(cash), "date": pd.Timestamp("2026-07-28"), "close": 50}))

    def test_fixed_cash_does_not_scale_with_eps_or_require_positive_eps(self):
        evidence = self.evidence({1: [2] * 5}, {1: [10, 20, 10, 20, 10]})
        inputs = pd.DataFrame([{"stock_id": 1, "target_year": 2026 + i, "estimated_eps": eps,
            "status": "ok" if np.isfinite(eps) else "EPS unavailable"}
            for i, eps in enumerate([10, 20, -2, np.nan])])
        result, _ = estimate_classified_dividends(inputs, evidence, "2026-09-02")
        self.assertEqual(result.dividend_pattern.tolist(), ["fixed"] * 4)
        self.assertEqual(result.estimated_cash_dividend.tolist(), [2] * 4)
        self.assertEqual(result.status.tolist(), ["ok"] * 4)
        self.assertEqual(result.iloc[-1].eps_status, "EPS unavailable")
        yields = calculate_yields(result, evidence.prices, target_year=2026,
            as_of_date=pd.Timestamp("2026-09-02"), yield_modes=("as_of_price_yield",), min_stock_price=0)
        self.assertEqual(yields.estimated_yield_percent.tolist(), [4] * 4)

    def test_zero_requires_five_explicit_years_even_without_positive_eps(self):
        evidence = self.evidence({1: [0] * 5, 2: [0, 0, 0, 0, None], 3: [None] * 5},
            {1: [-1, None, -2, None, 0], 2: [1] * 5, 3: [1] * 5})
        inputs = pd.DataFrame({"stock_id": [1, 2, 3], "estimated_eps": [np.nan, 10, 10],
            "status": ["EPS unavailable", "ok", "ok"]})
        result, _ = estimate_classified_dividends(inputs, evidence, "2026-09-02")
        self.assertEqual(result.dividend_pattern.tolist(), ["none", "insufficient", "insufficient"])
        self.assertEqual(result.iloc[0].estimated_cash_dividend, 0)
        self.assertTrue(result.iloc[1:].estimated_cash_dividend.isna().all())
        yields = calculate_yields(result, evidence.prices, target_year=2026,
            as_of_date=pd.Timestamp("2026-09-02"), yield_modes=("as_of_price_yield",), min_stock_price=0)
        self.assertEqual(yields.iloc[0].estimated_yield_percent, 0)
        self.assertTrue(yields.iloc[1:].estimated_yield_percent.isna().all())

    def test_normal_mean_includes_zero_excludes_nonpositive_eps(self):
        evidence = self.evidence({1: [0, 2, 4, 6, 8]}, {1: [10, 10, -2, 10, 10]})
        result, detail = estimate_classified_dividends(
            pd.DataFrame([{"stock_id": 1, "estimated_eps": 20, "status": "ok"}]), evidence, "2026-09-02")
        self.assertEqual(result.iloc[0].dividend_pattern, "normal")
        self.assertEqual(result.iloc[0].payout_valid_years, 4)
        self.assertAlmostEqual(result.iloc[0].payout_ratio, .4)
        self.assertAlmostEqual(result.iloc[0].estimated_cash_dividend, 8)
        self.assertTrue(detail.iloc[0].included)
        self.assertFalse(detail.iloc[2].included)

    def test_fixed_tolerance_and_missing_year_do_not_imply_fixed(self):
        evidence = self.evidence({1: [2, 2, 2, 2, 2.1], 2: [2, 2, 2, 2, 2.11],
            3: [2, 2, 2, 2, None], 4: [None, None, None, None, 2]})
        result, _ = estimate_classified_dividends(pd.DataFrame({"stock_id": [1, 2, 3, 4],
            "estimated_eps": 20, "status": "ok"}), evidence, "2026-09-02")
        self.assertEqual(result.dividend_pattern.tolist(), ["fixed", "normal", "normal", "insufficient"])
        self.assertEqual(result.iloc[3].estimated_cash_dividend, 4)
        self.assertIn("有限年度", result.iloc[3].dividend_calculation)

    def test_future_announcements_and_target_year_cash_cannot_change_classification(self):
        evidence = self.evidence({1: [2] * 5})
        expected = analyze_dividend_patterns([1], evidence.annual_eps, evidence.dividends, "2026-09-02")
        extra = pd.DataFrame([
            {"stock_id": 1, "fiscal_year": 2025, "TotalCashDividend": 999, "available_date": pd.Timestamp("2026-09-03")},
            {"stock_id": 1, "fiscal_year": 2026, "TotalCashDividend": 0, "available_date": pd.Timestamp("2026-08-01")},
        ])
        actual = analyze_dividend_patterns([1], evidence.annual_eps,
            pd.concat([evidence.dividends, extra], ignore_index=True), "2026-09-02")
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left, right)

    def test_invalid_installment_does_not_become_complete_year(self):
        evidence = self.evidence({1: [2] * 5})
        for invalid in [np.nan, -1, np.inf]:
            with self.subTest(invalid=invalid):
                extra = pd.DataFrame([{"stock_id": 1, "fiscal_year": 2025,
                    "TotalCashDividend": invalid, "available_date": pd.Timestamp("2026-08-01")}])
                result, detail = analyze_dividend_patterns([1], evidence.annual_eps,
                    pd.concat([evidence.dividends, extra], ignore_index=True), "2026-09-02")
                self.assertEqual(result.iloc[0].dividend_pattern, "normal")
                self.assertEqual(result.iloc[0].cash_history_years, 4)
                self.assertTrue(pd.isna(detail.iloc[-1].cash_dividend))


if __name__ == "__main__":
    unittest.main()
