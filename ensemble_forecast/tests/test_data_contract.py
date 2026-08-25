from pathlib import Path
import unittest

import numpy as np

from ensemble_forecast import data_contracts
from ensemble_forecast import forecast_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EnsembleDataContractTests(unittest.TestCase):
    def test_default_data_directory_is_the_shared_root_data_directory(self) -> None:
        self.assertEqual(Path(forecast_engine.DATA_DIR).resolve(), (PROJECT_ROOT / "data").resolve())
        self.assertTrue((Path(forecast_engine.DATA_DIR) / forecast_engine.REVENUE_FILENAME).is_file())

    def test_forecast_engine_reuses_data_contract_module(self) -> None:
        self.assertIs(forecast_engine.RevenueDataContract, data_contracts.RevenueDataContract)
        self.assertEqual(
            forecast_engine.MODEL_REVENUE_DATA_CONTRACT,
            data_contracts.MODEL_REVENUE_DATA_CONTRACT,
        )

    def test_relative_data_directory_override_resolves_from_project_root(self) -> None:
        resolved = forecast_engine._resolve_data_dir("free_taiwan_data/processed_benchmark_82")
        self.assertEqual(resolved, PROJECT_ROOT / "free_taiwan_data" / "processed_benchmark_82")

    def test_shared_revenue_contract_converts_raw_monetary_features_to_thousand_units(self) -> None:
        raw = forecast_engine.pd.DataFrame(
            {
                "stock_id": [1, 1],
                "revenue_year": [2024, 2024],
                "revenue_month": [1, 2],
                "revenue": [1_000_000.0, 2_000_000.0],
                "revenue_thousand": [1_000.0, 2_000.0],
                "mom": [0.0, 1.0],
                "last_year_revenue": [900_000.0, 1_800_000.0],
                "yoy": [0.1111, 0.1111],
                "last_3m_revenue": [1_000_000.0, 3_000_000.0],
                "last_3m_revenue_yoy": [0.1111, 0.1111],
                "last_12m_revenue": [1_000_000.0, 3_000_000.0],
                "last_12m_revenue_yoy": [0.1111, 0.1111],
                "acc_revenue": [1_000_000.0, 3_000_000.0],
                "acc_revenue_yoy": [0.1111, 0.1111],
            }
        )

        prepared = forecast_engine.prepare_revenue_data(raw)

        self.assertAlmostEqual(float(prepared.loc[0, "last_year_revenue"]), 900.0)
        self.assertAlmostEqual(float(prepared.loc[1, "last_3m_revenue"]), 3000.0)
        self.assertAlmostEqual(float(prepared.loc[1, "acc_revenue"]), 3000.0)

    def test_model_revenue_contract_keeps_thousand_unit_features_unchanged(self) -> None:
        raw = forecast_engine.pd.DataFrame(
            {
                "stock_id": [1, 1],
                "revenue_year": [2024, 2024],
                "revenue_month": [1, 2],
                "revenue": [1_000_000.0, 2_000_000.0],
                "revenue_thousand": [1_000.0, 2_000.0],
                "mom": [0.0, 1.0],
                "last_year_revenue": [900.0, 1_800.0],
                "yoy": [0.1111, 0.1111],
                "last_3m_revenue": [1_000.0, 3_000.0],
                "last_3m_revenue_yoy": [0.1111, 0.1111],
                "last_12m_revenue": [1_000.0, 3_000.0],
                "last_12m_revenue_yoy": [0.1111, 0.1111],
                "acc_revenue": [1_000.0, 3_000.0],
                "acc_revenue_yoy": [0.1111, 0.1111],
            }
        )

        prepared = forecast_engine.prepare_revenue_data(
            raw,
            data_contract=forecast_engine.MODEL_REVENUE_DATA_CONTRACT,
        )

        self.assertAlmostEqual(float(prepared.loc[0, "last_year_revenue"]), 900.0)
        self.assertAlmostEqual(float(prepared.loc[1, "last_3m_revenue"]), 3000.0)
        self.assertAlmostEqual(float(prepared.loc[1, "acc_revenue"]), 3000.0)

    def test_ratio_parser_only_scales_values_with_an_explicit_percent_marker(self) -> None:
        parsed = forecast_engine._coerce_numeric_column(
            forecast_engine.pd.Series(["11.93%", "0.1193", "50.0"]),
            ratio=True,
        )

        self.assertAlmostEqual(float(parsed.iloc[0]), 0.1193)
        self.assertAlmostEqual(float(parsed.iloc[1]), 0.1193)
        self.assertAlmostEqual(float(parsed.iloc[2]), 50.0)

    def test_feature_imputation_uses_only_prior_rows(self) -> None:
        frame = forecast_engine.pd.DataFrame(
            {
                "stock_id": [1, 1, 1],
                "feature": [np.nan, 10.0, 1_000.0],
            }
        )

        filled = forecast_engine._fill_feature_from_past(frame, "feature")

        self.assertEqual(filled.tolist(), [0.0, 10.0, 1_000.0])

    def test_next_year_target_is_joined_by_calendar_month_not_row_offset(self) -> None:
        frame = forecast_engine.pd.DataFrame(
            {
                "stock_id": [1, 1, 1],
                "revenue_year": [2023, 2023, 2024],
                "revenue_month": [1, 3, 1],
                "log_revenue": [1.0, 3.0, 11.0],
            }
        )

        aligned = forecast_engine._attach_next_year_target(frame)

        self.assertEqual(float(aligned.loc[0, "target_next_year"]), 11.0)
        self.assertTrue(forecast_engine.pd.isna(aligned.loc[1, "target_next_year"]))


if __name__ == "__main__":
    unittest.main()
