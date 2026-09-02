from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_forecast.live_engine import build_live_forecast, load_revenue_snapshot


class LiveRevenueSnapshotTests(unittest.TestCase):
    def _write_history(self, root, missing_month=None, include_future=False):
        dates = pd.date_range("2025-01-01", "2026-07-01", freq="MS")
        rows = [{"stock_id": 1101, "revenue_year": d.year, "revenue_month": d.month,
            "revenue_thousand": 100, "revenue_available_date": d + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)}
            for d in dates if d != missing_month]
        if include_future:
            rows.append({"stock_id": 1101, "revenue_year": 2026, "revenue_month": 8,
                "revenue_thousand": 999999999, "revenue_available_date": "2026-09-10"})
        pd.DataFrame(rows).to_csv(root / "Stock_revenue_2019~2025.csv", index=False)

    def test_uses_availability_date_and_reports_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([
                {"stock_id": 1101, "revenue_year": 2026, "revenue_month": 7, "revenue_thousand": 100,
                    "revenue_available_date": "2026-08-10"},
                {"stock_id": 1101, "revenue_year": 2026, "revenue_month": 8, "revenue_thousand": 200,
                    "revenue_available_date": "2026-09-10"},
                {"stock_id": 1101, "revenue_year": 2026, "revenue_month": 6, "revenue_thousand": -1,
                    "revenue_available_date": "2026-07-10"},
            ]).to_csv(root / "Stock_revenue_2019~2025.csv", index=False)
            frame = load_revenue_snapshot(root, "2026-09-02")
            self.assertEqual(frame["revenue_month"].tolist(), [7])
            self.assertEqual(frame.attrs["invalid_rows"], 1)

    def test_missing_financial_files_preserve_revenue_and_report_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_history(root)
            result = build_live_forecast(1101, "2026-09-02", root)
            self.assertEqual(len(result.monthly), 24)
            self.assertEqual(result.summary["actual_months"].tolist(), [7, 0])
            self.assertEqual(result.summary["forecast_months"].tolist(), [5, 12])
            self.assertTrue(result.summary["estimated_eps"].isna().all())
            self.assertTrue(result.summary["as_of_price_yield_percent"].isna().all())
            self.assertTrue(any(n.startswith("eps:") for n in result.notes))
            self._write_history(root, include_future=True)
            future = build_live_forecast(1101, "2026-09-02", root)
            pd.testing.assert_frame_equal(result.monthly, future.monthly)

    def test_interior_gap_does_not_silently_annualize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_history(root, pd.Timestamp("2026-02-01"))
            result = build_live_forecast(1101, "2026-09-02", root)
            self.assertTrue(pd.isna(result.summary.iloc[0]["predicted_annual_revenue"]))
            self.assertTrue(pd.notna(result.summary.iloc[1]["predicted_annual_revenue"]))
            missing = result.monthly[result.monthly["target_date"].eq(pd.Timestamp("2026-02-01"))].iloc[0]
            self.assertEqual(missing["revenue_basis"], "unavailable")


if __name__ == "__main__":
    unittest.main()
