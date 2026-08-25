from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_preprocessing.canonical_data_contract import (
    DATA_CONTRACT_VERSION,
    CANONICAL_FILENAMES,
    validate_canonical_data_dir,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class CanonicalDataContractTests(unittest.TestCase):
    def test_validate_canonical_data_dir_checks_files_columns_manifest_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / CANONICAL_FILENAMES["stock_list"],
                [{"stock_id": 1101, "stock_name": "台泥", "industry_category": "水泥工業"}],
            )
            _write_csv(
                root / CANONICAL_FILENAMES["revenue"],
                [
                    {
                        "date": "2025-01-01",
                        "stock_id": 1101,
                        "revenue_year": 2025,
                        "revenue_month": 1,
                        "revenue": 1000,
                        "revenue_thousand": 1,
                        "mom": 0.0,
                        "last_year_revenue": 900,
                        "yoy": 0.1111,
                        "last_3m_revenue": 1000,
                        "last_3m_revenue_yoy": 0.1111,
                        "last_12m_revenue": 1000,
                        "last_12m_revenue_yoy": 0.1111,
                        "acc_revenue": 1000,
                        "acc_revenue_yoy": 0.1111,
                        "industry_category": "水泥工業",
                        "revenue_available_date": "2025-02-10",
                    }
                ],
            )
            _write_csv(
                root / CANONICAL_FILENAMES["target_stocks"],
                [
                    {
                        "date": "2025-01-01",
                        "stock_id": 1101,
                        "country": "Taiwan",
                        "revenue": 1000,
                        "revenue_month": 1,
                        "revenue_year": 2025,
                    }
                ],
            )
            _write_csv(
                root / CANONICAL_FILENAMES["eps"],
                [
                    {
                        "date": "2025-03-31",
                        "stock_id": 1101,
                        "EPS": 0.5,
                        "statement_available_date": "2025-05-15",
                    }
                ],
            )
            _write_csv(
                root / CANONICAL_FILENAMES["dividends"],
                [
                    {
                        "stock_id": 1101,
                        "year": "113年",
                        "TotalCashDividend": 1.2,
                        "AnnouncementDate": "2025-06-10",
                        "DividendAvailableDate": "2025-06-10",
                        "dividend_available_source": "AnnouncementDate",
                    }
                ],
            )
            _write_csv(
                root / CANONICAL_FILENAMES["daily_prices"],
                [
                    {
                        "date": "2025-01-02",
                        "stock_id": 1101,
                        "open": 10,
                        "max": 11,
                        "min": 9,
                        "close": 10.5,
                    }
                ],
            )
            with (root / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump({"data_contract_version": DATA_CONTRACT_VERSION}, stream)

            result = validate_canonical_data_dir(
                root,
                minimum_stock_counts={"stock_list": 1, "revenue": 1, "daily_prices": 1},
                require_manifest=True,
            )

            self.assertTrue(result.is_valid)
            self.assertEqual(result.stock_counts["revenue"], 1)
            self.assertEqual(result.row_counts["daily_prices"], 1)

    def test_validate_canonical_data_dir_fails_when_candidate_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, filename in CANONICAL_FILENAMES.items():
                _write_csv(root / filename, [{"stock_id": 1101}])

            result = validate_canonical_data_dir(root, minimum_stock_counts={"revenue": 2})

            self.assertFalse(result.is_valid)
            self.assertTrue(any("below minimum" in issue for issue in result.issues))

    def test_validate_canonical_data_dir_checks_manifest_counts_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset, filename in CANONICAL_FILENAMES.items():
                _write_csv(root / filename, [{"stock_id": 1101}])

            revenue_path = root / CANONICAL_FILENAMES["revenue"]
            digest = hashlib.sha256(revenue_path.read_bytes()).hexdigest()
            with (root / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "data_contract_version": DATA_CONTRACT_VERSION,
                        "files": CANONICAL_FILENAMES,
                        "row_counts": {"revenue": 99},
                        "stock_counts": {"revenue": 1},
                        "file_sha256": {"revenue": digest},
                    },
                    stream,
                )

            result = validate_canonical_data_dir(root, require_manifest=True)

            self.assertFalse(result.is_valid)
            self.assertTrue(any("manifest row_count" in issue for issue in result.issues))


if __name__ == "__main__":
    unittest.main()
