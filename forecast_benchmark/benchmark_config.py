"""Default configuration for the cross-system forecast benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_YEAR = 2025
DEFAULT_PRIMARY_METRIC = "WMAPE"
DEFAULT_ROLLING_OUTPUT_DIR = (
    PROJECT_ROOT / "rolling_predict_LSTM" / "outputs" / "xlstm_main_flow_basket_100_d1_16"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "latest"

DEFAULT_ENSEMBLE_MODELS = (
    "ensemble_revenue",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "SeasonalQuantile",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    target_year: int = DEFAULT_TARGET_YEAR
    primary_metric: str = DEFAULT_PRIMARY_METRIC
    rolling_output_dir: Path = DEFAULT_ROLLING_OUTPUT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    ensemble_models: tuple[str, ...] = DEFAULT_ENSEMBLE_MODELS

