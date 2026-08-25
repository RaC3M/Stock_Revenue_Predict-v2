from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_preprocessing.preprocessing_audit import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CANDIDATE_DIR,
    DEFAULT_OUTPUT_DIR,
    PreprocessingAuditConfig,
    run_preprocessing_audit,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class PreprocessingAuditTests(unittest.TestCase):
    def test_defaults_share_preprocessing_artifact_directory(self) -> None:
        self.assertEqual(DEFAULT_CANDIDATE_DIR, DEFAULT_ARTIFACT_DIR / "processed")
        self.assertEqual(DEFAULT_OUTPUT_DIR, DEFAULT_ARTIFACT_DIR / "audit")

    def test_audit_reports_coverage_columns_and_numeric_differences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_csv(
                baseline / "Stock_revenue_2019~2025.csv",
                [
                    {
                        "stock_id": 1,
                        "revenue_year": 2025,
                        "revenue_month": 1,
                        "revenue": 1000.0,
                        "revenue_thousand": 1.0,
                    }
                ],
            )
            _write_csv(
                candidate / "Stock_revenue_2019~2025.csv",
                [
                    {
                        "stock_id": 1,
                        "revenue_year": 2025,
                        "revenue_month": 1,
                        "revenue": 1100.0,
                        "revenue_thousand": 1.1,
                        "revenue_available_date": "2025-02-10",
                    }
                ],
            )

            for filename in [
                "target_stocks_2025.csv",
                "EPS2020~2025.csv",
                "Dividend2019~2025.csv",
                "day K2020~2025.csv",
                "stock_list_new.csv",
            ]:
                _write_csv(baseline / filename, [{"stock_id": 1}])
                _write_csv(candidate / filename, [{"stock_id": 1}])

            results = run_preprocessing_audit(
                PreprocessingAuditConfig(
                    baseline_dir=baseline,
                    candidate_dir=candidate,
                    output_dir=root / "audit",
                    abs_tolerance=0.0,
                    rel_tolerance=0.0,
                )
            )

            revenue_coverage = results["coverage_summary"][
                results["coverage_summary"]["dataset"].eq("revenue")
            ].iloc[0]
            revenue_columns = results["column_presence_summary"][
                results["column_presence_summary"]["dataset"].eq("revenue")
            ]
            revenue_diffs = results["numeric_diff_summary"][
                results["numeric_diff_summary"]["dataset"].eq("revenue")
            ]
            readiness = results["replacement_readiness_summary"]

            self.assertEqual(int(revenue_coverage["common_key_count"]), 1)
            self.assertIn("revenue_available_date", set(revenue_columns["column"]))
            self.assertFalse(
                bool(
                    revenue_columns[
                        revenue_columns["column"].eq("revenue_available_date")
                    ]["in_baseline"].iloc[0]
                )
            )
            self.assertEqual(
                int(revenue_diffs[revenue_diffs["column"].eq("revenue")]["mismatch_count"].iloc[0]),
                1,
            )
            revenue_mismatch_gate = readiness[
                readiness["check"].eq("revenue_numeric_mismatch_rate")
            ].iloc[0]
            self.assertEqual(revenue_mismatch_gate["status"], "fail")

    def test_numeric_compare_aligns_duplicate_keys_without_cross_join_false_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"

            for folder, rows in [
                (
                    baseline,
                    [
                        {"stock_id": 1, "year": "113年", "TotalCashDividend": 1.0},
                        {"stock_id": 1, "year": "113年", "TotalCashDividend": 2.0},
                    ],
                ),
                (
                    candidate,
                    [
                        {"stock_id": 1, "year": "113年", "TotalCashDividend": 2.0},
                        {"stock_id": 1, "year": "113年", "TotalCashDividend": 1.0},
                    ],
                ),
            ]:
                _write_csv(folder / "Dividend2019~2025.csv", rows)

            for filename in [
                "Stock_revenue_2019~2025.csv",
                "target_stocks_2025.csv",
                "EPS2020~2025.csv",
                "day K2020~2025.csv",
                "stock_list_new.csv",
            ]:
                _write_csv(baseline / filename, [{"stock_id": 1}])
                _write_csv(candidate / filename, [{"stock_id": 1}])

            results = run_preprocessing_audit(
                PreprocessingAuditConfig(
                    baseline_dir=baseline,
                    candidate_dir=candidate,
                    output_dir=root / "audit",
                    abs_tolerance=0.0,
                    rel_tolerance=0.0,
                )
            )

            dividend_diffs = results["numeric_diff_summary"][
                results["numeric_diff_summary"]["dataset"].eq("dividends")
            ]

            self.assertEqual(
                int(dividend_diffs[dividend_diffs["column"].eq("TotalCashDividend")]["mismatch_count"].iloc[0]),
                0,
            )

    def test_replacement_readiness_fails_missing_columns_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"

            revenue_keys = {
                "stock_id": 1,
                "revenue_year": 2025,
                "revenue_month": 1,
            }
            _write_csv(
                baseline / "Stock_revenue_2019~2025.csv",
                [{**revenue_keys, "revenue": 1000.0, "revenue_thousand": 1.0, "yoy": 0.1}],
            )
            _write_csv(
                candidate / "Stock_revenue_2019~2025.csv",
                [{**revenue_keys, "revenue": 1000.0, "revenue_thousand": 1.0}],
            )
            for filename in [
                "target_stocks_2025.csv",
                "EPS2020~2025.csv",
                "Dividend2019~2025.csv",
                "day K2020~2025.csv",
                "stock_list_new.csv",
            ]:
                _write_csv(baseline / filename, [{"stock_id": 1}])
                _write_csv(candidate / filename, [{"stock_id": 1}])

            results = run_preprocessing_audit(
                PreprocessingAuditConfig(
                    baseline_dir=baseline,
                    candidate_dir=candidate,
                    output_dir=root / "audit",
                )
            )
            readiness = results["replacement_readiness_summary"]

            missing_yoy = readiness[
                readiness["check"].eq("required_column:yoy")
            ]
            self.assertEqual(missing_yoy["status"].tolist(), ["fail"])
            self.assertTrue(
                readiness["check"].str.startswith("canonical_contract:").any()
            )
            self.assertTrue(readiness["status"].eq("fail").any())


if __name__ == "__main__":
    unittest.main()
