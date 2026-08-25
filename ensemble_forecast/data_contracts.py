"""Data contracts and shared file locations for the ensemble forecast system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
DATA_DIR_ENV_VAR = "PREDICT_DATA_DIR"


def _resolve_data_dir(raw_path: str | os.PathLike[str] | None = None) -> Path:
    raw_path = os.environ.get(DATA_DIR_ENV_VAR) if raw_path is None else raw_path
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "data"


DATA_DIR = _resolve_data_dir()
REVENUE_FILENAME = "Stock_revenue_2019~2025.csv"
LEGACY_REVENUE_FILENAME = "stock_revenue_data.csv"
EPS_FILENAME = "EPS2020~2025.csv"
DIVIDEND_CASH_FILENAME = "Dividend2019~2025.csv"
DIVIDEND_POLICY_FILENAME = "dividend_policy_2025.csv"
DAILY_STOCK_PRICE_FILENAME = "day K2020~2025.csv"

MONETARY_REVENUE_FEATURE_COLUMNS = [
    "last_year_revenue",
    "last_3m_revenue",
    "last_12m_revenue",
    "acc_revenue",
]
RevenueAmountUnit = Literal["raw_ntd", "thousand_ntd"]


@dataclass(frozen=True)
class RevenueDataContract:
    monetary_feature_unit: RevenueAmountUnit


SHARED_REVENUE_DATA_CONTRACT = RevenueDataContract(monetary_feature_unit="raw_ntd")
MODEL_REVENUE_DATA_CONTRACT = RevenueDataContract(monetary_feature_unit="thousand_ntd")
REVENUE_FEATURE_UNIT_DIVISORS: dict[RevenueAmountUnit, float] = {
    "raw_ntd": 1000.0,
    "thousand_ntd": 1.0,
}


def apply_revenue_data_contract(df: pd.DataFrame, data_contract: RevenueDataContract) -> pd.DataFrame:
    divisor = REVENUE_FEATURE_UNIT_DIVISORS.get(data_contract.monetary_feature_unit)
    if divisor is None:
        raise ValueError(
            f"Unsupported revenue monetary feature unit: {data_contract.monetary_feature_unit}"
        )
    if divisor == 1:
        return df
    for column in MONETARY_REVENUE_FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce") / divisor
    return df
