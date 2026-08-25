from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_preprocessing.free_taiwan_data_preprocessor import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    FreeTaiwanPreprocessConfig,
    build_manifest,
    build_dividend_frame,
    build_eps_frame,
    build_revenue_frame,
    financial_statement_available_date,
    load_stock_info,
    month_revenue_available_date,
    parse_roc_year,
    preprocess_free_taiwan_data,
    write_processed_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class FreeTaiwanDataPreprocessorTests(unittest.TestCase):
    def test_default_output_is_outside_raw_source(self) -> None:
        self.assertFalse(DEFAULT_OUTPUT_DIR.is_relative_to(DEFAULT_SOURCE_DIR))
        self.assertEqual(DEFAULT_OUTPUT_DIR.parts[-3:], ("data_preprocessing", "outputs", "processed"))

    def test_available_date_helpers(self) -> None:
        self.assertEqual(parse_roc_year("113年"), 2024.0)
        self.assertEqual(parse_roc_year("113年後半年度"), 2024.0)
        self.assertEqual(parse_roc_year("113年第4季"), 2024.0)
        self.assertTrue(pd.isna(parse_roc_year("不適用")))
        self.assertEqual(month_revenue_available_date(2025, 1), pd.Timestamp(2025, 2, 10))
        self.assertEqual(month_revenue_available_date(2025, 12), pd.Timestamp(2026, 1, 10))
        self.assertEqual(financial_statement_available_date("2025-03-31"), pd.Timestamp(2025, 5, 15))
        self.assertEqual(financial_statement_available_date("2025-06-30"), pd.Timestamp(2025, 8, 14))
        self.assertEqual(financial_statement_available_date("2025-09-30"), pd.Timestamp(2025, 11, 14))
        self.assertEqual(financial_statement_available_date("2025-12-31"), pd.Timestamp(2026, 3, 31))

    def test_revenue_preprocess_builds_canonical_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "technical" / "TaiwanStockInfo" / "TaiwanStockInfo.csv",
                [
                    {
                        "industry_category": "水泥工業",
                        "stock_id": 1101,
                        "stock_name": "台泥",
                        "type": "twse",
                        "date": "2025-01-01",
                    }
                ],
            )
            _write_csv(
                root / "fundamental" / "TaiwanStockMonthRevenue" / "1101.csv",
                [
                    {
                        "date": "2024-02-01",
                        "stock_id": 1101,
                        "country": "Taiwan",
                        "revenue": 1000,
                        "revenue_month": 1,
                        "revenue_year": 2024,
                        "create_time": "",
                    },
                    {
                        "date": "2025-02-01",
                        "stock_id": 1101,
                        "country": "Taiwan",
                        "revenue": 1500,
                        "revenue_month": 1,
                        "revenue_year": 2025,
                        "create_time": "",
                    },
                ],
            )
            config = FreeTaiwanPreprocessConfig(source_dir=root, start_year=2024, end_year=2025, stock_ids=(1101,))
            revenue = build_revenue_frame(root, load_stock_info(root), config)

            current = revenue[revenue["revenue_year"].eq(2025)].iloc[0]
            self.assertAlmostEqual(float(current["revenue_thousand"]), 1.5)
            self.assertAlmostEqual(float(current["last_year_revenue"]), 1000.0)
            self.assertAlmostEqual(float(current["yoy"]), 0.5)
            self.assertTrue(pd.isna(current["mom"]))
            self.assertAlmostEqual(float(current["last_3m_revenue_yoy"]), 0.5)
            self.assertEqual(current["industry_category"], "水泥工業")
            self.assertEqual(pd.Timestamp(current["revenue_available_date"]), pd.Timestamp(2025, 2, 10))

    def test_eps_and_dividend_preprocess_keep_time_safe_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "fundamental" / "TaiwanStockFinancialStatements" / "1101.csv",
                [
                    {
                        "date": "2025-03-31",
                        "stock_id": 1101,
                        "type": "EPS",
                        "value": 0.5,
                        "origin_name": "基本每股盈餘",
                    }
                ],
            )
            _write_csv(
                root / "fundamental" / "TaiwanStockDividend" / "1101.csv",
                [
                    {
                        "date": "2025-07-08",
                        "stock_id": 1101,
                        "year": "113年",
                        "StockEarningsDistribution": 0.0,
                        "StockExDividendTradingDate": "",
                        "CashEarningsDistribution": 1.0,
                        "CashStatutorySurplus": 0.2,
                        "CashExDividendTradingDate": "2025-07-02",
                        "CashDividendPaymentDate": "2025-07-29",
                        "AnnouncementDate": "2025-06-10",
                        "AnnouncementTime": "09:14:14",
                    },
                    {
                        "date": "2025-07-08",
                        "stock_id": 1101,
                        "year": "不適用",
                        "StockEarningsDistribution": 0.0,
                        "StockExDividendTradingDate": "",
                        "CashEarningsDistribution": 0.0,
                        "CashStatutorySurplus": 0.0,
                        "CashExDividendTradingDate": "",
                        "CashDividendPaymentDate": "",
                        "AnnouncementDate": "",
                        "AnnouncementTime": "",
                    },
                    {
                        "date": "2025-10-08",
                        "stock_id": 1101,
                        "year": "不適用",
                        "StockEarningsDistribution": 0.0,
                        "StockExDividendTradingDate": "",
                        "CashEarningsDistribution": 0.3,
                        "CashStatutorySurplus": 0.1,
                        "CashExDividendTradingDate": "2025-10-02",
                        "CashDividendPaymentDate": "2025-10-29",
                        "AnnouncementDate": "2025-09-10",
                        "AnnouncementTime": "09:14:14",
                    }
                ],
            )
            config = FreeTaiwanPreprocessConfig(source_dir=root, start_year=2025, end_year=2025, stock_ids=(1101,))
            eps = build_eps_frame(root, config)
            dividends = build_dividend_frame(root, config)

            self.assertEqual(pd.Timestamp(eps.iloc[0]["statement_available_date"]), pd.Timestamp(2025, 5, 15))
            self.assertAlmostEqual(float(dividends.iloc[0]["TotalCashDividend"]), 1.2)
            self.assertEqual(pd.Timestamp(dividends.iloc[0]["DividendAvailableDate"]), pd.Timestamp(2025, 6, 10))
            self.assertEqual(dividends.iloc[0]["dividend_available_source"], "AnnouncementDate")
            self.assertEqual(len(dividends), 2)
            retained_non_roc_year = dividends[dividends["year"].eq("不適用")].iloc[0]
            self.assertTrue(pd.isna(retained_non_roc_year["fiscal_year"]))
            self.assertAlmostEqual(float(retained_non_roc_year["TotalCashDividend"]), 0.4)
            self.assertEqual(pd.Timestamp(retained_non_roc_year["DividendAvailableDate"]), pd.Timestamp(2025, 9, 10))

    def test_preprocess_free_taiwan_data_builds_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "technical" / "TaiwanStockInfo" / "TaiwanStockInfo.csv",
                [
                    {
                        "industry_category": "水泥工業",
                        "stock_id": 1101,
                        "stock_name": "台泥",
                        "type": "twse",
                        "date": "2025-01-01",
                    }
                ],
            )
            _write_csv(
                root / "fundamental" / "TaiwanStockMonthRevenue" / "1101.csv",
                [
                    {
                        "date": "2025-02-01",
                        "stock_id": 1101,
                        "country": "Taiwan",
                        "revenue": 1500,
                        "revenue_month": 1,
                        "revenue_year": 2025,
                        "create_time": "",
                    }
                ],
            )
            _write_csv(
                root / "fundamental" / "TaiwanStockFinancialStatements" / "1101.csv",
                [
                    {"date": "2025-03-31", "stock_id": 1101, "type": "EPS", "value": 0.5, "origin_name": "EPS"}
                ],
            )
            _write_csv(
                root / "fundamental" / "TaiwanStockDividend" / "1101.csv",
                [
                    {
                        "date": "2025-07-08",
                        "stock_id": 1101,
                        "year": "113年",
                        "CashEarningsDistribution": 1.0,
                        "CashStatutorySurplus": 0.0,
                        "CashExDividendTradingDate": "2025-07-02",
                        "AnnouncementDate": "2025-06-10",
                        "AnnouncementTime": "09:14:14",
                    }
                ],
            )
            _write_csv(
                root / "technical" / "TaiwanStockPrice" / "1101.csv",
                [
                    {
                        "date": "2025-01-02",
                        "stock_id": 1101,
                        "Trading_Volume": 10,
                        "Trading_money": 100,
                        "open": 10,
                        "max": 11,
                        "min": 9,
                        "close": 10.5,
                        "spread": 0.5,
                        "Trading_turnover": 1,
                    }
                ],
            )
            config = FreeTaiwanPreprocessConfig(source_dir=root, start_year=2025, end_year=2025, stock_ids=(1101,))
            frames = preprocess_free_taiwan_data(config)

            self.assertEqual(set(frames), {"stock_list", "revenue", "target_stocks", "eps", "dividends", "daily_prices"})
            self.assertEqual(len(frames["stock_list"]), 1)
            self.assertEqual(len(frames["target_stocks"]), 1)
            self.assertEqual(len(frames["daily_prices"]), 1)

            manifest = build_manifest(config, frames)

            self.assertEqual(manifest["data_contract_version"], "canonical-csv-v1")
            self.assertEqual(manifest["dataset_role"], "canonical_generated_csv")
            self.assertTrue(manifest["validation"]["is_valid"])
            self.assertEqual(manifest["row_counts"]["revenue"], 1)
            self.assertEqual(manifest["stock_counts"]["stock_list"], 1)
            self.assertIn("revenue_unit_contract", manifest)

            output_config = FreeTaiwanPreprocessConfig(
                source_dir=root,
                output_dir=root / "processed",
                start_year=2025,
                end_year=2025,
                target_year=2025,
                price_start_year=2025,
                stock_ids=(1101,),
            )
            paths = write_processed_outputs(frames, output_config)
            with paths["manifest"].open("r", encoding="utf-8") as stream:
                written_manifest = json.load(stream)

            self.assertEqual(
                written_manifest["files"]["revenue"],
                "Stock_revenue_2025~2025.csv",
            )
            self.assertEqual(
                written_manifest["files"]["daily_prices"],
                "day K2025~2025.csv",
            )
            self.assertIn("revenue", written_manifest["file_sha256"])


if __name__ == "__main__":
    unittest.main()
