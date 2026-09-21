from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from hybrid_forecast.live_engine import (
    CoreForecastResult,
    ForecastRequest,
    apply_price_scenario,
    build_financial_core_from_predictions,
    build_live_forecast,
    build_llm_zip,
    llm_payload_json,
    load_or_build_forecast,
    load_revenue_snapshot,
    without_price_scenario,
)


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

    def test_persistent_core_cache_excludes_price_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            cache = root / "cache"
            data.mkdir()
            self._write_history(data)
            request = ForecastRequest(1101, "2026-09-02", data)
            first = load_or_build_forecast(request, artifact_root=cache)
            self.assertFalse(first.cache_hit)
            pd.DataFrame([
                {"date": "2026-09-01", "stock_id": 1101, "close": 100},
            ]).to_csv(data / "day K2020~2025.csv", index=False)
            second = load_or_build_forecast(request, artifact_root=cache)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.artifact_id, second.artifact_id)
            pd.testing.assert_frame_equal(
                first.monthly.reset_index(drop=True),
                second.monthly.reset_index(drop=True),
                check_dtype=False,
            )
            pd.DataFrame([{
                "date": "2025-03-31", "stock_id": 1101, "EPS": 1.0,
                "statement_available_date": "2025-05-15",
            }]).to_csv(data / "EPS2020~2025.csv", index=False)
            changed = load_or_build_forecast(request, artifact_root=cache)
            self.assertNotEqual(first.artifact_id, changed.artifact_id)

    def test_repricing_changes_only_price_and_yield_and_exports_strict_json(self):
        summary = pd.DataFrame([{
            "source_family": "hybrid", "model": "test", "stock_id": 1101,
            "target_year": 2026, "as_of_date": pd.Timestamp("2026-09-02"),
            "estimated_cash_dividend": 5.0, "actual_cash_dividend": pd.NA,
            "status": "ok",
        }])
        core = CoreForecastResult(
            monthly=pd.DataFrame([{"stock_id": 1101, "hybrid_method": "weighted_hybrid"}]),
            summary=summary,
            quarterly_eps=pd.DataFrame(), payout_history=pd.DataFrame(),
            data_status=pd.DataFrame(), order_search=pd.DataFrame(), notes=[],
            artifact_id="artifact-test", generated_at="2026-09-02T00:00:00+00:00",
        )
        lower = apply_price_scenario(
            core, stock_price=100, price_date="2026-09-02", price_source="manual_scenario"
        )
        higher = apply_price_scenario(
            core, stock_price=200, price_date="2026-09-02", price_source="manual_scenario"
        )
        self.assertEqual(lower.artifact_id, higher.artifact_id)
        self.assertEqual(float(lower.summary.iloc[0]["as_of_price_yield_percent"]), 5.0)
        self.assertEqual(float(higher.summary.iloc[0]["as_of_price_yield_percent"]), 2.5)
        pd.testing.assert_frame_equal(lower.monthly, higher.monthly)
        payload = json.loads(llm_payload_json(lower))
        self.assertEqual(payload["price_scenario"]["source"], "manual_scenario")
        self.assertNotIn("NaN", llm_payload_json(lower))
        self.assertGreater(len(build_llm_zip(lower)), 0)
        with self.assertRaises(ValueError):
            apply_price_scenario(
                core, stock_price=0, price_date="2026-09-02", price_source="manual_scenario"
            )
        with self.assertRaises(ValueError):
            apply_price_scenario(
                core, stock_price=100, price_date="2026-09-03", price_source="manual_scenario"
            )

    def test_external_prediction_contract_rejects_incomplete_or_duplicate_years(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = ForecastRequest(1101, "2026-09-02", tmp)
            rows = [
                {"source_family": "other", "model": "v1", "stock_id": 1101,
                 "target_year": year, "target_month": month, "predicted_revenue": 100}
                for year in [2026, 2027] for month in range(1, 13)
            ]
            with self.assertRaisesRegex(ValueError, "11/12"):
                build_financial_core_from_predictions(pd.DataFrame(rows[:-1]), request)
            duplicate = pd.DataFrame([*rows, rows[0]])
            with self.assertRaisesRegex(ValueError, "重複月份"):
                build_financial_core_from_predictions(duplicate, request)

    def test_external_artifact_tracks_financial_evidence_and_preserves_model_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = ForecastRequest(1101, "2026-09-02", root)
            rows = pd.DataFrame([
                {"source_family": "partner_team", "model": "partner_v2", "stock_id": 1101,
                 "target_year": year, "target_month": month, "predicted_revenue": 100}
                for year in [2026, 2027] for month in range(1, 13)
            ])
            first = build_financial_core_from_predictions(rows, request)
            pd.DataFrame([{
                "date": "2025-03-31", "stock_id": 1101, "EPS": 1.0,
                "statement_available_date": "2025-05-15",
            }]).to_csv(root / "EPS2020~2025.csv", index=False)
            changed = build_financial_core_from_predictions(rows, request)
            self.assertNotEqual(first.artifact_id, changed.artifact_id)
            payload = json.loads(llm_payload_json(without_price_scenario(changed)))
            self.assertEqual(payload["model"]["source_family"], "partner_team")
            self.assertEqual(payload["model"]["name"], "partner_v2")
            self.assertEqual(payload["model"]["input_contract"], "external_monthly_revenue_predictions")
            self.assertIsNone(payload["formulas"]["hybrid"])
            self.assertIn("外部模型", payload["formulas"]["revenue_formula"])


if __name__ == "__main__":
    unittest.main()
