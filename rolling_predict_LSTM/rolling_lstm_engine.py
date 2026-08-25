from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pandas as pd

try:
    from .yield_forecast import build_rolling_yield_forecast
except ImportError:
    from yield_forecast import build_rolling_yield_forecast


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
FORECAST_YEAR = 2025
TRAIN_END_YEAR = FORECAST_YEAR - 1
DEFAULT_WINDOW_SIZE = 12
CLUSTER_RANGE = range(4, 9)
MAX_SCALED_PREDICTION = 4.0
MAX_LAST_REVENUE_MULTIPLIER = 5.0
MAX_SEQUENCE_REVENUE_MULTIPLIER = 4.0
DEFAULT_GROWTH_ADJUSTMENT_ALPHA = 0.8
DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA = 0.0
DEFAULT_DECLINE_CAP_PREDICTION_RATIO_MIN = 1.0
DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX = 0.35
DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN = 1.10
DEFAULT_UNDER_WEIGHT = 2.0
HIGH_GROWTH_RATIO_THRESHOLD = 0.6
HIGH_GROWTH_STREAK_THRESHOLD = 5
GROWTH_PHASE_RATIO_THRESHOLD = 0.65
GROWTH_PHASE_STREAK_THRESHOLD = 4
DECLINE_REGIME_RATIO_THRESHOLD = 0.4
CUDA_MIN_BATCH_SIZE = 4096
TORCH_EARLY_STOP_MIN_DELTA = 1e-5
DEFAULT_TREND_SLOPE_BETA = 0.35
DEFAULT_MAX_VOLATILITY_SCALE = 2.5
TRAINING_BACKEND = "torch"
DEFAULT_SEQUENCE_BACKBONE = "lstm"
DEFAULT_XLSTM_BACKBONE = "xlstm"
DEFAULT_STREAMLIT_XLSTM_BACKBONE = "xlstm_hybrid"
SLSTM_BACKEND = "vanilla"


@dataclass(frozen=True)
class XLSTMBackboneSpec:
    key: Literal["xlstm", "xlstm_hybrid"]
    display_name: str
    block_types: tuple[Literal["mlstm", "slstm"], ...]
    slstm_backend: str | None = None


XLSTM_BACKBONE_SPECS = {
    "xlstm": XLSTMBackboneSpec(
        key="xlstm",
        display_name="mLSTM-only（舊 D1 實驗）",
        block_types=("mlstm",),
    ),
    "xlstm_hybrid": XLSTMBackboneSpec(
        key="xlstm_hybrid",
        display_name="mLSTM + sLSTM（Hybrid）",
        block_types=("mlstm", "slstm"),
        slstm_backend=SLSTM_BACKEND,
    ),
}
XLSTM_BACKBONES = tuple(XLSTM_BACKBONE_SPECS)
SEQUENCE_BACKBONES = ("lstm", *XLSTM_BACKBONES)


def _normalize_choice(value: object, choices: tuple[str, ...], option_name: str) -> str:
    normalized = str(value).lower()
    if normalized not in choices:
        raise ValueError(f"Unknown {option_name}={value!r}.")
    return normalized


def get_xlstm_backbone_spec(value: object) -> XLSTMBackboneSpec:
    key = _normalize_choice(value, XLSTM_BACKBONES, "xlstm_backbone")
    return XLSTM_BACKBONE_SPECS[key]


ROLLING_LSTM_MODEL = "Rolling LSTM"
ROLLING_CLUSTER_MODEL = "Rolling LSTM + Cluster"
ROLLING_ADJUSTED_MODEL = "Rolling LSTM + Cluster + Conditional Adjustment"
ROLLING_XLSTM_MODEL = "Rolling xLSTM"
ROLLING_XLSTM_ADJUSTED_MODEL = "Rolling xLSTM + Conditional Adjustment"

ROLLING_MODEL_OUTPUTS = (
    (ROLLING_LSTM_MODEL, "predicted_revenue_no_cluster", "no_cluster_error", "no_cluster_abs_error"),
    (ROLLING_CLUSTER_MODEL, "predicted_revenue_cluster", "cluster_error", "cluster_abs_error"),
    (
        ROLLING_ADJUSTED_MODEL,
        "predicted_revenue_adjusted",
        "adjusted_error",
        "adjusted_abs_error",
    ),
    (ROLLING_XLSTM_MODEL, "predicted_revenue_xlstm", "xlstm_error", "xlstm_abs_error"),
    (
        ROLLING_XLSTM_ADJUSTED_MODEL,
        "predicted_revenue_xlstm_adjusted",
        "xlstm_adjusted_error",
        "xlstm_adjusted_abs_error",
    ),
)
ROLLING_MODEL_NAMES = frozenset(model_name for model_name, *_ in ROLLING_MODEL_OUTPUTS)
ROLLING_XLSTM_MODEL_NAMES = frozenset(
    {
        ROLLING_XLSTM_MODEL,
        ROLLING_XLSTM_ADJUSTED_MODEL,
    }
)


def resolve_model_sequence_backbone(
    model_name: object,
    *,
    main_sequence_backbone: object,
    xlstm_backbone: object,
    include_xlstm_plain: bool,
) -> str:
    normalized_model_name = str(model_name)
    if normalized_model_name not in ROLLING_MODEL_NAMES:
        raise ValueError(f"Unknown rolling model for architecture provenance: {model_name!r}.")
    if normalized_model_name in ROLLING_XLSTM_MODEL_NAMES:
        if not include_xlstm_plain:
            return "disabled"
        return get_xlstm_backbone_spec(xlstm_backbone).key
    return _normalize_choice(
        main_sequence_backbone,
        SEQUENCE_BACKBONES,
        "sequence_backbone",
    )


def attach_model_backbone_provenance(
    frame: pd.DataFrame,
    *,
    main_sequence_backbone: object,
    xlstm_backbone: object,
    include_xlstm_plain: bool,
) -> pd.DataFrame:
    if "model" not in frame.columns:
        raise ValueError("Cannot attach architecture provenance without a model column.")
    result = frame.copy()
    result["sequence_backbone"] = [
        resolve_model_sequence_backbone(
            model_name,
            main_sequence_backbone=main_sequence_backbone,
            xlstm_backbone=xlstm_backbone,
            include_xlstm_plain=include_xlstm_plain,
        )
        for model_name in result["model"]
    ]
    result["xlstm_backbone"] = (
        get_xlstm_backbone_spec(xlstm_backbone).key
        if include_xlstm_plain
        else pd.NA
    )
    return result

NUMERIC_SEQUENCE_FEATURES = [
    "log_revenue",
    "growth_rate",
    "momentum_3m",
    "momentum_6m",
]

TREND_CYCLE_SEQUENCE_FEATURES = [
    *NUMERIC_SEQUENCE_FEATURES,
    "trend_log",
    "cycle_ratio",
    "cycle_volatility_ratio",
    "trend_slope_rate",
]


@dataclass
class RollingLSTMResult:
    forecast: pd.DataFrame
    yield_forecast: pd.DataFrame
    yield_summary: pd.DataFrame
    metrics: pd.DataFrame
    cluster_profile: pd.DataFrame
    cluster_effect: pd.DataFrame
    selected_cluster_timeline: pd.DataFrame
    elbow_scores: pd.DataFrame
    tuning_results: pd.DataFrame
    selected_params: dict[str, object]
    yield_notes: list[str]
    notes: list[str]


@dataclass
class RollingForecastAssemblyResult:
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    clip_counts: dict[str, int]
    xlstm_plain_available: bool


@dataclass(frozen=True)
class GrowthAdjustmentConfig:
    enabled: bool = True
    alpha: float = DEFAULT_GROWTH_ADJUSTMENT_ALPHA
    conditional: bool = True
    regime_strategy: bool = True
    decline_cap_growth_ratio_max: float | None = None
    decline_cap_prediction_ratio_min: float = DEFAULT_DECLINE_CAP_PREDICTION_RATIO_MIN


@dataclass(frozen=True)
class RollingExperimentConfig:
    k: int = 6
    window_size: int = DEFAULT_WINDOW_SIZE
    epochs: int = 40
    max_train_samples: int = 40_000
    sequence_backbone: Literal["lstm", "xlstm", "xlstm_hybrid"] = DEFAULT_SEQUENCE_BACKBONE
    include_xlstm_plain: bool = False
    xlstm_backbone: Literal["xlstm", "xlstm_hybrid"] = DEFAULT_XLSTM_BACKBONE
    include_yield_forecast: bool = False
    yield_as_of_date: str | None = None
    use_asymmetric_loss: bool = True
    under_weight: float = DEFAULT_UNDER_WEIGHT
    growth: GrowthAdjustmentConfig = field(default_factory=GrowthAdjustmentConfig)
    xlstm_growth: GrowthAdjustmentConfig = field(
        default_factory=lambda: GrowthAdjustmentConfig(
            alpha=DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA,
            decline_cap_growth_ratio_max=DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX,
            decline_cap_prediction_ratio_min=DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN,
        )
    )


_LEGACY_EXPERIMENT_OPTIONS = frozenset(
    {
        "k",
        "window_size",
        "epochs",
        "max_train_samples",
        "sequence_backbone",
        "include_xlstm_plain",
        "xlstm_backbone",
        "enable_growth_adjustment",
        "growth_adjustment_alpha",
        "enable_conditional_adjustment",
        "enable_regime_strategy",
        "decline_cap_growth_ratio_max",
        "decline_cap_prediction_ratio_min",
        "xlstm_enable_growth_adjustment",
        "xlstm_growth_adjustment_alpha",
        "xlstm_enable_conditional_adjustment",
        "xlstm_enable_regime_strategy",
        "xlstm_decline_cap_growth_ratio_max",
        "xlstm_decline_cap_prediction_ratio_min",
        "use_asymmetric_loss",
        "under_weight",
    }
)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "off"}:
        return None
    return float(value)


def _normalize_experiment_config(
    config: RollingExperimentConfig | None,
    legacy_options: dict[str, object] | None = None,
) -> RollingExperimentConfig:
    options = dict(legacy_options or {})
    if config is None:
        config = RollingExperimentConfig()
    elif not isinstance(config, RollingExperimentConfig):
        raise TypeError("config must be a RollingExperimentConfig.")

    backend = options.pop("backend", None)
    if backend is not None and str(backend) != TRAINING_BACKEND:
        raise ValueError(f"Rolling LSTM only supports backend={TRAINING_BACKEND!r}.")

    unknown = sorted(set(options) - _LEGACY_EXPERIMENT_OPTIONS)
    if unknown:
        raise TypeError(f"Unknown rolling experiment options: {unknown}")
    if not options:
        sequence_backbone = _normalize_choice(
            config.sequence_backbone,
            SEQUENCE_BACKBONES,
            "sequence_backbone",
        )
        xlstm_backbone = get_xlstm_backbone_spec(config.xlstm_backbone).key
        normalized_updates = {}
        if sequence_backbone != config.sequence_backbone:
            normalized_updates["sequence_backbone"] = sequence_backbone
        if xlstm_backbone != config.xlstm_backbone:
            normalized_updates["xlstm_backbone"] = xlstm_backbone
        if normalized_updates:
            config = replace(config, **normalized_updates)
        return config

    base_updates: dict[str, object] = {}
    for option_name in ["k", "window_size", "epochs", "max_train_samples"]:
        if option_name in options:
            base_updates[option_name] = int(options[option_name])
    if "sequence_backbone" in options:
        sequence_backbone = _normalize_choice(
            options["sequence_backbone"],
            SEQUENCE_BACKBONES,
            "sequence_backbone",
        )
        base_updates["sequence_backbone"] = sequence_backbone
    else:
        sequence_backbone = _normalize_choice(
            config.sequence_backbone,
            SEQUENCE_BACKBONES,
            "sequence_backbone",
        )
        if sequence_backbone != config.sequence_backbone:
            base_updates["sequence_backbone"] = sequence_backbone
    if "include_xlstm_plain" in options:
        base_updates["include_xlstm_plain"] = bool(options["include_xlstm_plain"])
    if "xlstm_backbone" in options:
        xlstm_backbone = get_xlstm_backbone_spec(options["xlstm_backbone"]).key
        base_updates["xlstm_backbone"] = xlstm_backbone
    else:
        xlstm_backbone = get_xlstm_backbone_spec(config.xlstm_backbone).key
        if xlstm_backbone != config.xlstm_backbone:
            base_updates["xlstm_backbone"] = xlstm_backbone
    if "use_asymmetric_loss" in options:
        base_updates["use_asymmetric_loss"] = bool(options["use_asymmetric_loss"])
    if "under_weight" in options:
        base_updates["under_weight"] = float(options["under_weight"])

    growth_updates: dict[str, object] = {}
    if "enable_growth_adjustment" in options:
        growth_updates["enabled"] = bool(options["enable_growth_adjustment"])
    if "growth_adjustment_alpha" in options:
        growth_updates["alpha"] = float(options["growth_adjustment_alpha"])
    if "enable_conditional_adjustment" in options:
        growth_updates["conditional"] = bool(options["enable_conditional_adjustment"])
    if "enable_regime_strategy" in options:
        growth_updates["regime_strategy"] = bool(options["enable_regime_strategy"])
    if "decline_cap_growth_ratio_max" in options:
        growth_updates["decline_cap_growth_ratio_max"] = _optional_float(options["decline_cap_growth_ratio_max"])
    if "decline_cap_prediction_ratio_min" in options:
        growth_updates["decline_cap_prediction_ratio_min"] = float(options["decline_cap_prediction_ratio_min"])

    xlstm_growth_updates: dict[str, object] = {}
    if "xlstm_enable_growth_adjustment" in options:
        xlstm_growth_updates["enabled"] = bool(options["xlstm_enable_growth_adjustment"])
    if "xlstm_growth_adjustment_alpha" in options:
        xlstm_growth_updates["alpha"] = float(options["xlstm_growth_adjustment_alpha"])
    if "xlstm_enable_conditional_adjustment" in options:
        xlstm_growth_updates["conditional"] = bool(options["xlstm_enable_conditional_adjustment"])
    if "xlstm_enable_regime_strategy" in options:
        xlstm_growth_updates["regime_strategy"] = bool(options["xlstm_enable_regime_strategy"])
    if "xlstm_decline_cap_growth_ratio_max" in options:
        xlstm_growth_updates["decline_cap_growth_ratio_max"] = _optional_float(
            options["xlstm_decline_cap_growth_ratio_max"]
        )
    if "xlstm_decline_cap_prediction_ratio_min" in options:
        xlstm_growth_updates["decline_cap_prediction_ratio_min"] = float(
            options["xlstm_decline_cap_prediction_ratio_min"]
        )

    return replace(
        config,
        **base_updates,
        growth=replace(config.growth, **growth_updates),
        xlstm_growth=replace(config.xlstm_growth, **xlstm_growth_updates),
    )


def load_revenue_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = _revenue_file_path()
    return prepare_revenue_data(pd.read_csv(path))


def _revenue_file_path() -> str:
    return str(DATA_DIR / REVENUE_FILENAME)


def _revenue_file_mtime_ns() -> int:
    path = _revenue_file_path()
    return os.stat(path).st_mtime_ns if os.path.exists(path) else 0


@lru_cache(maxsize=8)
def _cached_revenue_and_windows(window_size: int, revenue_mtime_ns: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = revenue_mtime_ns
    df = load_revenue_data()
    windows = build_growth_windows(df, window_size=window_size)
    return df, windows


@lru_cache(maxsize=16)
def _cached_clustered_artifacts(
    k: int,
    window_size: int,
    revenue_mtime_ns: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, windows = _cached_revenue_and_windows(window_size, revenue_mtime_ns)
    elbow_scores = calculate_elbow_scores(windows, train_end_year=TRAIN_END_YEAR)
    _, clustered_windows, cluster_profile = fit_kmeans_clusters(windows, k=k, train_end_year=TRAIN_END_YEAR)
    monthly = attach_clusters_to_monthly(df, clustered_windows)
    return elbow_scores, clustered_windows, cluster_profile, monthly


def prepare_revenue_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    if "revenue_thousand" not in df.columns:
        if "revenue" not in df.columns:
            raise ValueError("Revenue data must include revenue_thousand or revenue.")
        df["revenue_thousand"] = _to_numeric(df["revenue"]) / 1000.0

    numeric_columns = [
        "stock_id",
        "revenue_year",
        "revenue_month",
        "revenue",
        "revenue_thousand",
        "mom",
        "last_year_revenue",
        "yoy",
        "last_3m_revenue",
        "last_3m_revenue_yoy",
        "last_12m_revenue",
        "last_12m_revenue_yoy",
        "acc_revenue",
        "acc_revenue_yoy",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = _to_numeric(df[column])

    required_columns = ["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]
    df = df.dropna(subset=required_columns)
    df = df[np.isfinite(df["revenue_thousand"]) & (df["revenue_thousand"] >= 0)].copy()
    df["stock_id"] = df["stock_id"].astype(int)
    df["revenue_year"] = df["revenue_year"].astype(int)
    df["revenue_month"] = df["revenue_month"].astype(int)
    df["date"] = pd.to_datetime(
        df["revenue_year"].astype(str) + "-" + df["revenue_month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    df["_calendar_month_index"] = df["revenue_year"] * 12 + df["revenue_month"] - 1
    month_gap = df.groupby("stock_id")["_calendar_month_index"].diff().ne(1)
    df["_calendar_segment"] = month_gap.groupby(df["stock_id"]).cumsum().astype(int)
    segment_grouped = df.groupby(["stock_id", "_calendar_segment"], group_keys=False)
    df["_consecutive_month_count"] = segment_grouped.cumcount() + 1

    df["prev_revenue"] = segment_grouped["revenue_thousand"].shift(1)
    df["growth_direction"] = (df["revenue_thousand"] > df["prev_revenue"]).astype(int)
    df.loc[df["prev_revenue"].isna(), "growth_direction"] = 0

    df["growth_rate"] = df["revenue_thousand"] / df["prev_revenue"] - 1
    df["growth_rate"] = df["growth_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["growth_rate"] = df["growth_rate"].clip(lower=-0.95, upper=5.0)
    df["log_revenue"] = np.log1p(df["revenue_thousand"].clip(lower=0))

    df["momentum_3m"] = (
        segment_grouped["growth_rate"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["momentum_6m"] = (
        segment_grouped["growth_rate"]
        .rolling(6, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["trend_component"] = (
        segment_grouped["revenue_thousand"]
        .rolling(12, min_periods=3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["trend_component"] = df["trend_component"].fillna(df["revenue_thousand"]).clip(lower=0)
    df["cycle_component"] = df["revenue_thousand"] - df["trend_component"]
    trend_denom = df["trend_component"].replace(0, np.nan)
    df["trend_log"] = np.log1p(df["trend_component"])
    df["cycle_ratio"] = (df["cycle_component"] / trend_denom).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["cycle_ratio"] = df["cycle_ratio"].clip(lower=-3.0, upper=3.0)
    df["cycle_volatility_6m"] = (
        df.groupby(["stock_id", "_calendar_segment"], group_keys=False)["cycle_component"]
        .rolling(6, min_periods=3)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
    )
    df["cycle_volatility_ratio"] = (
        df["cycle_volatility_6m"] / trend_denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["trend_slope"] = segment_grouped["trend_component"].diff(3).fillna(0)
    trend_slope_base = segment_grouped["trend_component"].shift(3).replace(0, np.nan)
    df["trend_slope_rate"] = (
        df["trend_slope"] / trend_slope_base
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["trend_slope_rate"] = df["trend_slope_rate"].clip(lower=-1.0, upper=1.0)
    return df


def _to_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def get_stock_list(df: pd.DataFrame) -> list[int]:
    return sorted(df["stock_id"].dropna().astype(int).unique().tolist())


def _months_are_consecutive(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    if "_calendar_month_index" in frame.columns:
        month_index = pd.to_numeric(frame["_calendar_month_index"], errors="coerce").to_numpy()
    else:
        month_index = (
            pd.to_numeric(frame["revenue_year"], errors="coerce").to_numpy() * 12
            + pd.to_numeric(frame["revenue_month"], errors="coerce").to_numpy()
            - 1
        )
    return bool(
        np.isfinite(month_index).all()
        and (len(month_index) == 1 or np.all(np.diff(month_index) == 1))
    )


def build_growth_windows(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    vector_columns = [f"g_{i + 1}" for i in range(window_size)]

    for stock_id, stock_df in df.groupby("stock_id", sort=False):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) < window_size:
            continue

        directions = stock_df["growth_direction"].astype(int).to_numpy()
        dates = stock_df["date"].to_numpy()
        years = stock_df["revenue_year"].to_numpy()
        months = stock_df["revenue_month"].to_numpy()

        for end_idx in range(window_size - 1, len(stock_df)):
            window_frame = stock_df.iloc[end_idx - window_size + 1 : end_idx + 1]
            if not _months_are_consecutive(window_frame):
                continue
            vector = directions[end_idx - window_size + 1 : end_idx + 1]
            row = {
                "stock_id": int(stock_id),
                "window_end_date": pd.Timestamp(dates[end_idx]),
                "window_end_year": int(years[end_idx]),
                "window_end_month": int(months[end_idx]),
                "growth_ratio": float(vector.mean()),
                "growth_streak": int(_trailing_ones(vector)),
            }
            row.update({column: int(value) for column, value in zip(vector_columns, vector)})
            rows.append(row)

    return pd.DataFrame(rows)


def _trailing_ones(values: np.ndarray) -> int:
    count = 0
    for value in values[::-1]:
        if value == 1:
            count += 1
        else:
            break
    return count


def fit_kmeans_clusters(
    windows: pd.DataFrame,
    k: int = 6,
    train_end_year: int = TRAIN_END_YEAR,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans

    vector_columns = [column for column in windows.columns if column.startswith("g_")]
    train_windows = windows[windows["window_end_year"] <= train_end_year].copy()
    if train_windows.empty:
        raise ValueError("No training growth-direction windows are available for KMeans.")

    k = int(np.clip(k, min(CLUSTER_RANGE), max(CLUSTER_RANGE)))
    k = min(k, len(train_windows))
    if k < 2:
        raise ValueError("KMeans needs at least two growth windows.")

    model = KMeans(n_clusters=k, n_init=20, random_state=42)
    model.fit(train_windows[vector_columns])

    clustered = windows.copy()
    clustered["cluster"] = model.predict(clustered[vector_columns]).astype(int)
    train_profile_windows = clustered[clustered["window_end_year"] <= train_end_year].copy()
    profile = build_cluster_profile(train_profile_windows, vector_columns)
    if not profile.empty:
        profile["profile_train_end_year"] = int(train_end_year)
    return model, clustered, profile


def calculate_elbow_scores(
    windows: pd.DataFrame,
    train_end_year: int = TRAIN_END_YEAR,
    k_values: range = CLUSTER_RANGE,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    vector_columns = [column for column in windows.columns if column.startswith("g_")]
    train_windows = windows[windows["window_end_year"] <= train_end_year].copy()
    if train_windows.empty:
        return pd.DataFrame(columns=["k", "inertia"])

    rows = []
    max_k = min(max(k_values), len(train_windows))
    for k in k_values:
        if k > max_k:
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        model.fit(train_windows[vector_columns])
        rows.append({"k": int(k), "inertia": float(model.inertia_)})
    return pd.DataFrame(rows)


def build_cluster_profile(clustered_windows: pd.DataFrame, vector_columns: list[str]) -> pd.DataFrame:
    base_profile = (
        clustered_windows.groupby("cluster")
        .agg(
            window_count=("cluster", "size"),
            stock_count=("stock_id", "nunique"),
            avg_growth_ratio=("growth_ratio", "mean"),
            avg_growth_streak=("growth_streak", "mean"),
        )
        .reset_index()
    )
    vector_profile = clustered_windows.groupby("cluster")[vector_columns].mean().reset_index()
    profile = base_profile.merge(vector_profile, on="cluster", how="left")
    profile["pattern"] = profile[vector_columns].apply(
        lambda row: "".join("1" if value >= 0.5 else "0" for value in row),
        axis=1,
    )
    profile["cluster_type"] = profile["avg_growth_ratio"].apply(_describe_cluster_type)
    for column in ["avg_growth_ratio", "avg_growth_streak", *vector_columns]:
        profile[column] = profile[column].round(3)
    return profile.sort_values("avg_growth_ratio", ascending=False).reset_index(drop=True)


def _describe_cluster_type(growth_ratio: float) -> str:
    if growth_ratio >= 0.70:
        return "consistent_growth"
    if growth_ratio >= 0.55:
        return "growth_bias"
    if growth_ratio >= 0.40:
        return "mixed_or_turnaround"
    return "decline_bias"


def attach_clusters_to_monthly(df: pd.DataFrame, clustered_windows: pd.DataFrame) -> pd.DataFrame:
    cluster_map = clustered_windows[["stock_id", "window_end_date", "cluster", "growth_ratio", "growth_streak"]]
    merged = df.merge(
        cluster_map,
        left_on=["stock_id", "date"],
        right_on=["stock_id", "window_end_date"],
        how="left",
    ).drop(columns=["window_end_date"])
    merged["cluster"] = merged["cluster"].fillna(-1).astype(int)
    merged["growth_ratio"] = merged["growth_ratio"].fillna(0.0)
    merged["growth_streak"] = merged["growth_streak"].fillna(0).astype(int)
    return merged.sort_values(["stock_id", "date"]).reset_index(drop=True)


def build_supervised_sequences(
    df: pd.DataFrame,
    selected_stock: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_samples: list[dict[str, object]] = []
    eval_samples: list[dict[str, object]] = []

    for stock_id, stock_df in df.groupby("stock_id", sort=False):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) <= window_size:
            continue

        for end_idx in range(window_size - 1, len(stock_df) - 1):
            target_idx = end_idx + 1
            full_window = stock_df.iloc[end_idx - window_size + 1 : target_idx + 1]
            if not _months_are_consecutive(full_window):
                continue
            target_year = int(stock_df.loc[target_idx, "revenue_year"])
            sequence_frame = stock_df.iloc[end_idx - window_size + 1 : end_idx + 1]
            sample = {
                "stock_id": int(stock_id),
                "sequence_frame": sequence_frame,
                "cluster": int(stock_df.loc[end_idx, "cluster"]),
                "sequence_start_date": stock_df.loc[end_idx - window_size + 1, "date"],
                "sequence_end_date": stock_df.loc[end_idx, "date"],
                "target_date": stock_df.loc[target_idx, "date"],
                "target_year": target_year,
                "target_month": int(stock_df.loc[target_idx, "revenue_month"]),
            }
            if target_year <= TRAIN_END_YEAR:
                target_revenue = float(stock_df.loc[target_idx, "revenue_thousand"])
                if not np.isfinite(target_revenue) or target_revenue < 0:
                    continue
                sample.update(
                    {
                        "target_revenue": target_revenue,
                        "target_trend": float(stock_df.loc[target_idx, "trend_component"]),
                        "target_cycle": float(stock_df.loc[target_idx, "cycle_component"]),
                    }
                )
                train_samples.append(sample)
            elif int(stock_id) == int(selected_stock) and target_year == FORECAST_YEAR:
                eval_samples.append(sample)

    return train_samples, eval_samples


def build_supervised_sequences_for_year(
    df: pd.DataFrame,
    selected_stock: int,
    window_size: int,
    train_end_year: int,
    eval_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_samples: list[dict[str, object]] = []
    eval_samples: list[dict[str, object]] = []

    for stock_id, stock_df in df.groupby("stock_id", sort=False):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) <= window_size:
            continue

        for end_idx in range(window_size - 1, len(stock_df) - 1):
            target_idx = end_idx + 1
            full_window = stock_df.iloc[end_idx - window_size + 1 : target_idx + 1]
            if not _months_are_consecutive(full_window):
                continue
            target_year = int(stock_df.loc[target_idx, "revenue_year"])
            target_revenue = float(stock_df.loc[target_idx, "revenue_thousand"])
            if not np.isfinite(target_revenue) or target_revenue < 0:
                continue
            sequence_frame = stock_df.iloc[end_idx - window_size + 1 : end_idx + 1]
            sample = {
                "stock_id": int(stock_id),
                "sequence_frame": sequence_frame,
                "cluster": int(stock_df.loc[end_idx, "cluster"]),
                "sequence_start_date": stock_df.loc[end_idx - window_size + 1, "date"],
                "sequence_end_date": stock_df.loc[end_idx, "date"],
                "target_date": stock_df.loc[target_idx, "date"],
                "target_year": target_year,
                "target_month": int(stock_df.loc[target_idx, "revenue_month"]),
                "target_revenue": target_revenue,
                "target_trend": float(stock_df.loc[target_idx, "trend_component"]),
                "target_cycle": float(stock_df.loc[target_idx, "cycle_component"]),
            }
            if target_year <= train_end_year:
                train_samples.append(sample)
            elif int(stock_id) == int(selected_stock) and target_year == eval_year:
                eval_samples.append(sample)

    return train_samples, eval_samples


def build_eval_sequences_for_stock(
    df: pd.DataFrame,
    selected_stock: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[dict[str, object]]:
    stock_df = df[df["stock_id"] == int(selected_stock)].sort_values("date").reset_index(drop=True)
    if len(stock_df) <= window_size:
        return []

    eval_samples: list[dict[str, object]] = []
    for end_idx in range(window_size - 1, len(stock_df) - 1):
        target_idx = end_idx + 1
        full_window = stock_df.iloc[end_idx - window_size + 1 : target_idx + 1]
        if not _months_are_consecutive(full_window):
            continue
        target_year = int(stock_df.loc[target_idx, "revenue_year"])
        if target_year != FORECAST_YEAR:
            continue

        sequence_frame = stock_df.iloc[end_idx - window_size + 1 : end_idx + 1]
        eval_samples.append(
            {
                "stock_id": int(selected_stock),
                "sequence_frame": sequence_frame,
                "cluster": int(stock_df.loc[end_idx, "cluster"]),
                "sequence_start_date": stock_df.loc[end_idx - window_size + 1, "date"],
                "sequence_end_date": stock_df.loc[end_idx, "date"],
                "target_date": stock_df.loc[target_idx, "date"],
                "target_year": target_year,
                "target_month": int(stock_df.loc[target_idx, "revenue_month"]),
            }
        )
    return eval_samples


def cap_training_samples(
    train_samples: list[dict[str, object]],
    max_train_samples: int,
    seed: int = 42,
) -> tuple[list[dict[str, object]], bool]:
    if max_train_samples and len(train_samples) > max_train_samples:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(train_samples), size=max_train_samples, replace=False)
        return [train_samples[int(index)] for index in np.sort(selected)], True
    return train_samples, False


def split_forward_validation_samples(
    train_samples: list[dict[str, object]],
    max_train_samples: int,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]], bool]:
    """Reserve the latest target year as a non-overlapping forward validation set."""
    if not train_samples:
        return [], [], False
    target_years = [int(sample["target_year"]) for sample in train_samples]
    validation_year = max(target_years)
    fit_candidates = [
        sample for sample in train_samples if int(sample["target_year"]) < validation_year
    ]
    validation_candidates = [
        sample for sample in train_samples if int(sample["target_year"]) == validation_year
    ]
    if not fit_candidates:
        return cap_training_samples(train_samples, max_train_samples, seed=seed)[0], [], (
            bool(max_train_samples and len(train_samples) > max_train_samples)
        )
    if not max_train_samples or len(train_samples) <= max_train_samples:
        return fit_candidates, validation_candidates, False

    validation_budget = min(
        len(validation_candidates),
        max(1, int(max_train_samples * 0.2)),
    )
    fit_budget = max(1, max_train_samples - validation_budget)
    fit_samples, fit_capped = cap_training_samples(fit_candidates, fit_budget, seed=seed)
    validation_samples, validation_capped = cap_training_samples(
        validation_candidates,
        validation_budget,
        seed=seed + 1,
    )
    return fit_samples, validation_samples, bool(fit_capped or validation_capped)


@lru_cache(maxsize=16)
def _cached_train_samples(
    k: int,
    window_size: int,
    revenue_mtime_ns: int,
) -> tuple[dict[str, object], ...]:
    _, _, _, monthly = _cached_clustered_artifacts(k, window_size, revenue_mtime_ns)
    train_samples, _ = build_supervised_sequences(monthly, selected_stock=-1, window_size=window_size)
    return tuple(train_samples)


@lru_cache(maxsize=128)
def _cached_eval_samples(
    k: int,
    window_size: int,
    selected_stock: int,
    revenue_mtime_ns: int,
) -> tuple[dict[str, object], ...]:
    _, _, _, monthly = _cached_clustered_artifacts(k, window_size, revenue_mtime_ns)
    return tuple(build_eval_sequences_for_stock(monthly, selected_stock, window_size=window_size))


@lru_cache(maxsize=16)
def _cached_all_eval_samples(
    k: int,
    window_size: int,
    revenue_mtime_ns: int,
) -> tuple[dict[str, object], ...]:
    _, _, _, monthly = _cached_clustered_artifacts(k, window_size, revenue_mtime_ns)
    eval_samples: list[dict[str, object]] = []
    for stock_id in get_stock_list(monthly):
        eval_samples.extend(build_eval_sequences_for_stock(monthly, stock_id, window_size=window_size))
    return tuple(eval_samples)


@lru_cache(maxsize=32)
def _cached_training_arrays(
    k: int,
    window_size: int,
    max_train_samples: int,
    include_cluster: bool,
    revenue_mtime_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object, bool]:
    from sklearn.preprocessing import StandardScaler

    clustered_windows = _cached_clustered_artifacts(k, window_size, revenue_mtime_ns)[1]
    cluster_count = int(clustered_windows["cluster"].max()) + 1
    train_samples = list(_cached_train_samples(k, window_size, revenue_mtime_ns))
    fit_samples, validation_samples, was_capped = split_forward_validation_samples(
        train_samples,
        max_train_samples=max_train_samples,
        seed=42,
    )
    numeric_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train, y_train, _ = make_lstm_arrays(
        fit_samples,
        numeric_scaler,
        target_scaler,
        cluster_count=cluster_count,
        include_cluster=include_cluster,
        fit_scalers=True,
    )
    if validation_samples:
        x_validation, y_validation, _ = make_lstm_arrays(
            validation_samples,
            numeric_scaler,
            target_scaler,
            cluster_count=cluster_count,
            include_cluster=include_cluster,
            fit_scalers=False,
            require_target=True,
        )
    else:
        x_validation = np.empty((0, *x_train.shape[1:]), dtype=np.float32)
        y_validation = np.empty((0, 1), dtype=np.float32)
    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        numeric_scaler,
        target_scaler,
        was_capped,
    )


@lru_cache(maxsize=8)
def _cached_lstm_predictions(
    k: int,
    window_size: int,
    max_train_samples: int,
    epochs: int,
    sequence_backbone: str,
    include_xlstm_plain: bool,
    use_asymmetric_loss: bool,
    under_weight: float,
    enable_trend_cycle_model: bool,
    revenue_mtime_ns: int,
    xlstm_backbone: str = DEFAULT_XLSTM_BACKBONE,
) -> tuple[pd.DataFrame, str, str, str, str, str]:
    clustered_windows = _cached_clustered_artifacts(k, window_size, revenue_mtime_ns)[1]
    cluster_count = int(clustered_windows["cluster"].max()) + 1
    train_samples = list(_cached_train_samples(k, window_size, revenue_mtime_ns))
    eval_samples = list(_cached_all_eval_samples(k, window_size, revenue_mtime_ns))
    if not train_samples or not eval_samples:
        raise ValueError("No rolling LSTM training or evaluation samples are available.")

    (
        x_train_cluster,
        y_train_cluster,
        x_validation_cluster,
        y_validation_cluster,
        x_scaler_cluster,
        y_scaler_cluster,
        _,
    ) = _cached_training_arrays(
        k,
        window_size,
        max_train_samples,
        True,
        revenue_mtime_ns,
    )
    x_eval_cluster, _, eval_meta = make_lstm_arrays(
        eval_samples,
        x_scaler_cluster,
        y_scaler_cluster,
        cluster_count=cluster_count,
        include_cluster=True,
        fit_scalers=False,
        require_target=False,
    )
    pred_cluster, backend_used = train_predict_lstm(
        x_train_cluster,
        y_train_cluster,
        x_eval_cluster,
        y_scaler_cluster,
        epochs=epochs,
        sequence_backbone=sequence_backbone,
        use_asymmetric_loss=use_asymmetric_loss,
        under_weight=under_weight,
        x_validation=x_validation_cluster,
        y_validation=y_validation_cluster,
    )

    (
        x_train_plain,
        y_train_plain,
        x_validation_plain,
        y_validation_plain,
        x_scaler_plain,
        y_scaler_plain,
        _,
    ) = _cached_training_arrays(
        k,
        window_size,
        max_train_samples,
        False,
        revenue_mtime_ns,
    )
    x_eval_plain, _, _ = make_lstm_arrays(
        eval_samples,
        x_scaler_plain,
        y_scaler_plain,
        cluster_count=cluster_count,
        include_cluster=False,
        fit_scalers=False,
        require_target=False,
    )
    pred_plain, backend_plain = train_predict_lstm(
        x_train_plain,
        y_train_plain,
        x_eval_plain,
        y_scaler_plain,
        epochs=epochs,
        sequence_backbone=sequence_backbone,
        use_asymmetric_loss=use_asymmetric_loss,
        under_weight=under_weight,
        x_validation=x_validation_plain,
        y_validation=y_validation_plain,
    )

    prediction_frame = eval_meta.copy()
    prediction_frame["raw_pred_cluster"] = pred_cluster
    prediction_frame["raw_pred_plain"] = pred_plain
    backend_xlstm_plain = "disabled"
    if include_xlstm_plain:
        try:
            pred_xlstm_plain, backend_xlstm_plain = train_predict_lstm(
                x_train_plain,
                y_train_plain,
                x_eval_plain,
                y_scaler_plain,
                epochs=epochs,
                sequence_backbone=xlstm_backbone,
                use_asymmetric_loss=use_asymmetric_loss,
                under_weight=under_weight,
                x_validation=x_validation_plain,
                y_validation=y_validation_plain,
            )
            prediction_frame["raw_pred_xlstm"] = pred_xlstm_plain
        except Exception as error:
            prediction_frame["raw_pred_xlstm"] = np.nan
            backend_xlstm_plain = f"unavailable: {type(error).__name__}: {error}"
    else:
        prediction_frame["raw_pred_xlstm"] = np.nan
    trend_backend = "disabled"
    cycle_backend = "disabled"
    if enable_trend_cycle_model:
        trend_pred, cycle_pred, trend_backend, cycle_backend = train_predict_trend_cycle_components(
            list(cap_training_samples(list(_cached_train_samples(k, window_size, revenue_mtime_ns)), max_train_samples)[0]),
            eval_samples,
            cluster_count=cluster_count,
            epochs=epochs,
        )
        prediction_frame["raw_pred_trend"] = trend_pred
        prediction_frame["raw_pred_cycle"] = cycle_pred
    else:
        prediction_frame["raw_pred_trend"] = np.nan
        prediction_frame["raw_pred_cycle"] = 0.0

    return prediction_frame, backend_used, backend_plain, backend_xlstm_plain, trend_backend, cycle_backend


def make_lstm_arrays(
    samples: list[dict[str, object]],
    numeric_scaler,
    target_scaler,
    cluster_count: int,
    include_cluster: bool,
    fit_scalers: bool = False,
    numeric_features: list[str] | tuple[str, ...] | None = None,
    target_column: str = "target_revenue",
    target_transform: Literal["log", "identity"] = "log",
    require_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not samples:
        raise ValueError("No supervised samples are available for LSTM training or evaluation.")

    feature_columns = list(numeric_features or NUMERIC_SEQUENCE_FEATURES)
    numeric_frames = [sample["sequence_frame"][feature_columns] for sample in samples]
    flat_numeric = pd.concat(numeric_frames, ignore_index=True)
    sample_count = len(samples)
    needs_target = bool(require_target or fit_scalers)
    if needs_target:
        missing_target_count = sum(1 for sample in samples if target_column not in sample)
        if missing_target_count:
            raise ValueError(
                f"{missing_target_count} samples are missing required target column {target_column!r}."
            )
        target_values = np.array([sample[target_column] for sample in samples], dtype=np.float32)
        if target_transform == "log":
            target_values = np.clip(target_values, 0, None)
            targets = np.log1p(target_values).reshape(-1, 1)
        elif target_transform == "identity":
            targets = target_values.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported target transform: {target_transform}")
    else:
        targets = None

    if fit_scalers:
        numeric_scaler.fit(flat_numeric)
        target_scaler.fit(targets)

    window_length = len(samples[0]["sequence_frame"])
    scaled_numeric = numeric_scaler.transform(flat_numeric).astype(np.float32)
    numeric_values = scaled_numeric.reshape(sample_count, window_length, len(feature_columns))
    if include_cluster:
        cluster_vector = np.zeros((sample_count, window_length, cluster_count), dtype=np.float32)
        clusters = np.array([int(sample["cluster"]) for sample in samples], dtype=np.int64)
        valid_clusters = (clusters >= 0) & (clusters < cluster_count)
        row_indices = np.where(valid_clusters)[0]
        if len(row_indices):
            cluster_vector[row_indices, :, clusters[valid_clusters]] = 1.0
        x = np.concatenate([numeric_values, cluster_vector], axis=2).astype(np.float32)
    else:
        x = numeric_values.astype(np.float32)

    metadata_rows: list[dict[str, object]] = []

    for sample in samples:
        sequence_frame: pd.DataFrame = sample["sequence_frame"]
        last_row = sequence_frame.iloc[-1]
        last_revenue = float(last_row["revenue_thousand"])
        sequence_max_revenue = float(sequence_frame["revenue_thousand"].max())
        metadata_rows.append(
            {
                "stock_id": sample["stock_id"],
                "input_start_date": sample["sequence_start_date"],
                "input_end_date": sample["sequence_end_date"],
                "target_date": sample["target_date"],
                "target_year": sample["target_year"],
                "target_month": sample["target_month"],
                "last_observed_revenue": last_revenue,
                "sequence_max_revenue": sequence_max_revenue,
                "cluster": sample["cluster"],
                "growth_rate_at_end": float(last_row["growth_rate"]),
                "momentum_3m_at_end": float(last_row["momentum_3m"]),
                "momentum_6m_at_end": float(last_row["momentum_6m"]),
                "growth_ratio": float(last_row["growth_ratio"]),
                "growth_streak": int(last_row["growth_streak"]),
                "trend_component": float(last_row["trend_component"]),
                "cycle_component": float(last_row["cycle_component"]),
                "cycle_volatility_6m": float(last_row["cycle_volatility_6m"]),
                "trend_slope": float(last_row["trend_slope"]),
                "trend_slope_rate": float(last_row["trend_slope_rate"]),
            }
        )
        if require_target and target_column == "target_revenue":
            metadata_rows[-1]["actual_revenue"] = sample["target_revenue"]

    if targets is None:
        y = np.empty((sample_count, 1), dtype=np.float32)
    else:
        y = target_scaler.transform(targets).astype(np.float32)
    return x, y, pd.DataFrame(metadata_rows)


def train_predict_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_scaler,
    epochs: int = 40,
    batch_size: int = 128,
    hidden_units: int = 48,
    sequence_backbone: Literal["lstm", "xlstm", "xlstm_hybrid"] = DEFAULT_SEQUENCE_BACKBONE,
    use_asymmetric_loss: bool = True,
    under_weight: float = DEFAULT_UNDER_WEIGHT,
    target_transform: Literal["log", "identity"] = "log",
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    sequence_backbone = str(sequence_backbone).lower()
    if sequence_backbone not in SEQUENCE_BACKBONES:
        raise ValueError(f"Unknown sequence_backbone={sequence_backbone!r}.")
    try:
        return _train_predict_torch(
            x_train,
            y_train,
            x_eval,
            y_scaler,
            epochs=epochs,
            batch_size=batch_size,
            hidden_units=hidden_units,
            sequence_backbone=sequence_backbone,
            use_asymmetric_loss=use_asymmetric_loss,
            under_weight=under_weight,
            target_transform=target_transform,
            x_validation=x_validation,
            y_validation=y_validation,
        )
    except ImportError as error:
        raise ImportError("PyTorch is required for Rolling LSTM training.") from error


def inverse_scaled_log_prediction(pred_scaled: np.ndarray, y_scaler) -> np.ndarray:
    pred_scaled = np.asarray(pred_scaled, dtype=float).reshape(-1, 1)
    pred_scaled = np.nan_to_num(
        pred_scaled,
        nan=0.0,
        posinf=MAX_SCALED_PREDICTION,
        neginf=-MAX_SCALED_PREDICTION,
    )
    pred_scaled = np.clip(pred_scaled, -MAX_SCALED_PREDICTION, MAX_SCALED_PREDICTION)
    pred_log = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    pred_log = np.nan_to_num(pred_log, nan=0.0, posinf=30.0, neginf=0.0)
    pred_log = np.clip(pred_log, 0.0, 30.0)
    return np.clip(np.expm1(pred_log), 0, None)


def inverse_scaled_prediction(
    pred_scaled: np.ndarray,
    y_scaler,
    target_transform: Literal["log", "identity"] = "log",
) -> np.ndarray:
    if target_transform == "log":
        return inverse_scaled_log_prediction(pred_scaled, y_scaler)
    pred_scaled = np.asarray(pred_scaled, dtype=float).reshape(-1, 1)
    pred_scaled = np.nan_to_num(
        pred_scaled,
        nan=0.0,
        posinf=MAX_SCALED_PREDICTION,
        neginf=-MAX_SCALED_PREDICTION,
    )
    pred_scaled = np.clip(pred_scaled, -MAX_SCALED_PREDICTION, MAX_SCALED_PREDICTION)
    values = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


_XLSTM_CUDA_INIT_MODULE = "xlstm.blocks.slstm.src.cuda_init"
_XLSTM_IMPORT_MODE = "native"


def _clear_partial_xlstm_imports() -> None:
    global _XLSTM_IMPORT_MODE

    for module_name in list(sys.modules):
        if module_name == "xlstm" or module_name.startswith("xlstm."):
            sys.modules.pop(module_name, None)
    _XLSTM_IMPORT_MODE = "native"


def _disabled_slstm_cuda_loader(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "The optional custom sLSTM CUDA kernel is disabled by the xLSTM import compatibility shim. "
        "Rolling xLSTM Hybrid uses the native PyTorch sLSTM backend and does not require this kernel."
    )


def _install_xlstm_mlstm_import_shim() -> None:
    global _XLSTM_IMPORT_MODE

    cuda_init_stub = ModuleType(_XLSTM_CUDA_INIT_MODULE)
    cuda_init_stub.__doc__ = (
        "Compatibility stub used by the Rolling mLSTM-only path when the CUDA extension "
        "toolkit is unavailable."
    )
    cuda_init_stub.load = _disabled_slstm_cuda_loader
    sys.modules[_XLSTM_CUDA_INIT_MODULE] = cuda_init_stub
    _XLSTM_IMPORT_MODE = "mlstm-only-compat"


def _needs_xlstm_mlstm_import_shim() -> bool:
    if importlib.util.find_spec("xlstm") is None:
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    try:
        from torch.utils import cpp_extension
    except (ImportError, AttributeError):
        return False
    return cpp_extension.CUDA_HOME is None


def _import_xlstm_components():
    if _needs_xlstm_mlstm_import_shim():
        _install_xlstm_mlstm_import_shim()

    try:
        from xlstm import (
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
        )

        return xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig
    except ModuleNotFoundError as error:
        if error.name == "xlstm":
            _clear_partial_xlstm_imports()
            raise ImportError("Optional xLSTM dependency is not installed. Install requirements-xlstm.txt.") from error
        raise
    except OSError as error:
        if "CUDA_HOME" not in str(error):
            raise

    _clear_partial_xlstm_imports()
    _install_xlstm_mlstm_import_shim()
    from xlstm import (
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
    )

    return xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig


def _import_xlstm_hybrid_components():
    base_components = _import_xlstm_components()
    from xlstm import FeedForwardConfig, sLSTMBlockConfig, sLSTMLayerConfig

    return (*base_components, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig)


def get_xlstm_backbone_status(
    sequence_backbone: Literal["xlstm", "xlstm_hybrid"] = DEFAULT_XLSTM_BACKBONE,
) -> dict[str, object]:
    try:
        spec = get_xlstm_backbone_spec(sequence_backbone)
    except ValueError as error:
        return {
            "available": False,
            "detail": str(error),
        }
    try:
        if "slstm" in spec.block_types:
            _import_xlstm_hybrid_components()
        else:
            _import_xlstm_components()
    except Exception as error:
        return {
            "available": False,
            "detail": f"{type(error).__name__}: {error}",
        }
    return {
        "available": True,
        "detail": (
            "xLSTM Hybrid (mLSTM + sLSTM) is available; sLSTM uses the native PyTorch "
            "backend and follows the model onto CUDA when GPU training is enabled."
            if "slstm" in spec.block_types
            else
            "xLSTM mLSTM-only backbone is available through the CUDA import compatibility shim; "
            "the PyTorch CUDA runtime remains enabled."
            if _XLSTM_IMPORT_MODE == "mlstm-only-compat"
            else "xLSTM mLSTM-only backbone import is available."
        ),
        "import_mode": _XLSTM_IMPORT_MODE,
    }


def _build_revenue_sequence_model(
    nn,
    input_size: int,
    hidden_units: int,
    window_size: int,
    sequence_backbone: Literal["lstm", "xlstm", "xlstm_hybrid"],
):
    if sequence_backbone == "lstm":
        class RevenueLSTM(nn.Module):
            def __init__(self, input_size: int, hidden_size: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_size, 24), nn.ReLU(), nn.Linear(24, 1))

            def forward(self, x):
                output, _ = self.lstm(x)
                return self.head(output[:, -1, :])

        return RevenueLSTM(input_size, hidden_units)

    spec = get_xlstm_backbone_spec(sequence_backbone)
    if "slstm" in spec.block_types:
        (
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            FeedForwardConfig,
        ) = _import_xlstm_hybrid_components()
    else:
        xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig = _import_xlstm_components()

    class RevenueXLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, context_length: int):
            super().__init__()
            self.input_projection = nn.Linear(input_size, hidden_size)
            slstm_block = None
            slstm_at = []
            num_blocks = len(spec.block_types)
            if "slstm" in spec.block_types:
                slstm_block = sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend=str(spec.slstm_backend),
                        num_heads=4,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                )
                slstm_at = [spec.block_types.index("slstm")]
            self.backbone = xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    mlstm_block=mLSTMBlockConfig(
                        mlstm=mLSTMLayerConfig(
                            conv1d_kernel_size=4,
                            qkv_proj_blocksize=4,
                            num_heads=4,
                            proj_factor=1.0,
                        )
                    ),
                    slstm_block=slstm_block,
                    context_length=context_length,
                    num_blocks=num_blocks,
                    embedding_dim=hidden_size,
                    slstm_at=slstm_at,
                    dropout=0.0,
                )
            )
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden_size, 24), nn.ReLU(), nn.Linear(24, 1))

        def forward(self, x):
            output = self.backbone(self.input_projection(x))
            return self.head(output[:, -1, :])

    return RevenueXLSTM(input_size, hidden_units, window_size)


def _train_predict_torch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_scaler,
    epochs: int,
    batch_size: int,
    hidden_units: int,
    sequence_backbone: Literal["lstm", "xlstm", "xlstm_hybrid"],
    use_asymmetric_loss: bool,
    under_weight: float,
    target_transform: Literal["log", "identity"],
    x_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    import torch
    from torch import nn

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model = _build_revenue_sequence_model(
        nn,
        input_size=x_train.shape[2],
        hidden_units=hidden_units,
        window_size=x_train.shape[1],
        sequence_backbone=sequence_backbone,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def asymmetric_loss(y_pred, y_true):
        error = y_pred - y_true
        weights = torch.where(error < 0.0, torch.full_like(error, float(under_weight)), torch.ones_like(error))
        return torch.mean((error**2) * weights)

    criterion = asymmetric_loss if use_asymmetric_loss else nn.HuberLoss()
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_eval_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
    has_forward_validation = bool(
        x_validation is not None
        and y_validation is not None
        and len(x_validation) > 0
        and len(y_validation) > 0
    )
    x_validation_tensor = (
        torch.as_tensor(x_validation, dtype=torch.float32, device=device)
        if has_forward_validation
        else None
    )
    y_validation_tensor = (
        torch.as_tensor(y_validation, dtype=torch.float32, device=device)
        if has_forward_validation
        else None
    )
    sample_count = x_train_tensor.shape[0]
    effective_batch_size = min(max(batch_size, CUDA_MIN_BATCH_SIZE if device.type == "cuda" else batch_size), sample_count)
    use_amp = device.type == "cuda" and sequence_backbone == "lstm"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)

    def autocast_context():
        if hasattr(torch, "amp"):
            return torch.amp.autocast("cuda", enabled=use_amp)
        return torch.cuda.amp.autocast(enabled=use_amp)

    epoch_count = int(epochs)
    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    patience = min(5, max(2, epoch_count // 6))
    epochs_ran = 0

    model.train()
    for epoch in range(epoch_count):
        order = torch.randperm(sample_count, device=device)
        for start in range(0, order.shape[0], effective_batch_size):
            batch_index = order[start : start + effective_batch_size]
            xb = x_train_tensor[batch_index]
            yb = y_train_tensor[batch_index]
            optimizer.zero_grad()
            with autocast_context():
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        epochs_ran = epoch + 1
        if has_forward_validation:
            model.eval()
            with torch.no_grad(), autocast_context():
                val_pred = model(x_validation_tensor)
                val_loss = float(criterion(val_pred, y_validation_tensor).detach().cpu())
            model.train()
            if val_loss < best_val_loss - TORCH_EARLY_STOP_MIN_DELTA:
                best_val_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad(), autocast_context():
        pred_scaled = model(x_eval_tensor).detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
        backend_label = (
            f"{sequence_backbone}:torch:cuda:{torch.cuda.get_device_name(0)} "
            f"(batch={effective_batch_size}, epochs={epochs_ran}/{epoch_count}, amp={use_amp}, "
            f"forward_val={len(x_validation) if has_forward_validation else 0})"
        )
    else:
        backend_label = (
            f"{sequence_backbone}:torch:cpu "
            f"(batch={effective_batch_size}, epochs={epochs_ran}/{epoch_count}, amp={use_amp}, "
            f"forward_val={len(x_validation) if has_forward_validation else 0})"
        )
    return inverse_scaled_prediction(pred_scaled, y_scaler, target_transform), backend_label


def calculate_growth_signal(metadata: pd.DataFrame) -> np.ndarray:
    growth_signal = (
        0.5 * metadata["momentum_3m_at_end"].to_numpy(dtype=float)
        + 0.3 * metadata["momentum_6m_at_end"].to_numpy(dtype=float)
        + 0.2 * metadata["growth_rate_at_end"].to_numpy(dtype=float)
    )
    return np.nan_to_num(growth_signal, nan=0.0, posinf=0.0, neginf=0.0)


def calculate_high_growth_flag(metadata: pd.DataFrame) -> np.ndarray:
    growth_ratio = metadata["growth_ratio"].to_numpy(dtype=float)
    growth_streak = metadata["growth_streak"].to_numpy(dtype=float)
    return (growth_ratio > HIGH_GROWTH_RATIO_THRESHOLD) | (growth_streak >= HIGH_GROWTH_STREAK_THRESHOLD)


def classify_regime(metadata: pd.DataFrame) -> np.ndarray:
    growth_ratio = metadata["growth_ratio"].to_numpy(dtype=float)
    return np.where(
        growth_ratio > GROWTH_PHASE_RATIO_THRESHOLD,
        "growth",
        np.where(growth_ratio < DECLINE_REGIME_RATIO_THRESHOLD, "decline", "cycle"),
    )


def calculate_growth_phase(metadata: pd.DataFrame) -> np.ndarray:
    growth_ratio = metadata["growth_ratio"].to_numpy(dtype=float)
    growth_streak = metadata["growth_streak"].to_numpy(dtype=float)
    return (growth_ratio > GROWTH_PHASE_RATIO_THRESHOLD) & (growth_streak >= GROWTH_PHASE_STREAK_THRESHOLD)


def calculate_prediction_caps(
    metadata: pd.DataFrame,
) -> np.ndarray:
    max_by_last = metadata["last_observed_revenue"].to_numpy(dtype=float) * MAX_LAST_REVENUE_MULTIPLIER
    max_by_sequence = metadata["sequence_max_revenue"].to_numpy(dtype=float) * MAX_SEQUENCE_REVENUE_MULTIPLIER
    return np.maximum.reduce([max_by_last, max_by_sequence, np.ones(len(metadata), dtype=float)])


def apply_revenue_guardrails(
    predicted: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, int, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    predicted = np.clip(predicted, 0.0, None)
    caps = calculate_prediction_caps(metadata)
    clipped = predicted > caps
    return np.minimum(predicted, caps), int(clipped.sum()), caps


def calculate_decline_cap_mask(
    predicted: np.ndarray,
    metadata: pd.DataFrame,
    regime: np.ndarray,
    enable_regime_strategy: bool = True,
    decline_cap_growth_ratio_max: float | None = None,
    decline_cap_prediction_ratio_min: float = DEFAULT_DECLINE_CAP_PREDICTION_RATIO_MIN,
) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=float)
    if not bool(enable_regime_strategy):
        return np.zeros(len(predicted), dtype=bool)

    last_revenue = metadata["last_observed_revenue"].to_numpy(dtype=float)
    growth_ratio = metadata["growth_ratio"].to_numpy(dtype=float)
    ratio_min = max(float(decline_cap_prediction_ratio_min), 0.0)
    cap_mask = (np.asarray(regime) == "decline") & (predicted > last_revenue * ratio_min)
    if decline_cap_growth_ratio_max is not None:
        cap_mask &= growth_ratio <= float(decline_cap_growth_ratio_max)
    return cap_mask


def apply_growth_adjustment(
    predicted: np.ndarray,
    metadata: pd.DataFrame,
    alpha: float = DEFAULT_GROWTH_ADJUSTMENT_ALPHA,
    enable_growth_adjustment: bool = True,
    enable_conditional_adjustment: bool = True,
    enable_regime_strategy: bool = True,
    decline_cap_growth_ratio_max: float | None = None,
    decline_cap_prediction_ratio_min: float = DEFAULT_DECLINE_CAP_PREDICTION_RATIO_MIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    growth_signal = calculate_growth_signal(metadata)
    regime = classify_regime(metadata)
    is_growth_phase = calculate_growth_phase(metadata)
    direction = metadata["growth_rate_at_end"].to_numpy(dtype=float)
    direction_allowed = direction > 0
    positive_signal = np.clip(growth_signal, 0.0, None)
    raw_adjustment_ratio = 1.0 + (float(alpha) * positive_signal if enable_growth_adjustment else 0.0)

    if enable_regime_strategy:
        adjustment_allowed = regime == "growth"
    else:
        adjustment_allowed = np.ones(len(metadata), dtype=bool)

    if enable_conditional_adjustment:
        adjustment_allowed = adjustment_allowed & is_growth_phase
    adjustment_allowed = adjustment_allowed & direction_allowed & (positive_signal > 0) & bool(enable_growth_adjustment)

    adjusted = np.where(adjustment_allowed, predicted * raw_adjustment_ratio, predicted)
    decline_cap_mask = calculate_decline_cap_mask(
        adjusted,
        metadata,
        regime,
        enable_regime_strategy=enable_regime_strategy,
        decline_cap_growth_ratio_max=decline_cap_growth_ratio_max,
        decline_cap_prediction_ratio_min=decline_cap_prediction_ratio_min,
    )
    if decline_cap_mask.any():
        last_revenue = metadata["last_observed_revenue"].to_numpy(dtype=float)
        adjusted = np.where(decline_cap_mask, np.minimum(adjusted, last_revenue), adjusted)

    adjustment_ratio = np.divide(
        adjusted,
        np.where(predicted == 0, np.nan, predicted),
        out=np.ones_like(adjusted, dtype=float),
        where=predicted != 0,
    )
    return adjusted, growth_signal, adjustment_ratio, regime, is_growth_phase, adjustment_allowed


def train_predict_trend_cycle_components(
    train_samples: list[dict[str, object]],
    eval_samples: list[dict[str, object]],
    cluster_count: int,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    from sklearn.preprocessing import StandardScaler

    x_scaler_trend = StandardScaler()
    y_scaler_trend = StandardScaler()
    x_train_trend, y_train_trend, _ = make_lstm_arrays(
        train_samples,
        x_scaler_trend,
        y_scaler_trend,
        cluster_count=cluster_count,
        include_cluster=True,
        fit_scalers=True,
        numeric_features=TREND_CYCLE_SEQUENCE_FEATURES,
        target_column="target_trend",
        target_transform="log",
    )
    x_eval_trend, _, _ = make_lstm_arrays(
        eval_samples,
        x_scaler_trend,
        y_scaler_trend,
        cluster_count=cluster_count,
        include_cluster=True,
        fit_scalers=False,
        numeric_features=TREND_CYCLE_SEQUENCE_FEATURES,
        target_column="target_trend",
        target_transform="log",
        require_target=False,
    )
    trend_pred, trend_backend = train_predict_lstm(
        x_train_trend,
        y_train_trend,
        x_eval_trend,
        y_scaler_trend,
        epochs=epochs,
        use_asymmetric_loss=False,
        target_transform="log",
    )

    x_scaler_cycle = StandardScaler()
    y_scaler_cycle = StandardScaler()
    x_train_cycle, y_train_cycle, _ = make_lstm_arrays(
        train_samples,
        x_scaler_cycle,
        y_scaler_cycle,
        cluster_count=cluster_count,
        include_cluster=True,
        fit_scalers=True,
        numeric_features=TREND_CYCLE_SEQUENCE_FEATURES,
        target_column="target_cycle",
        target_transform="identity",
    )
    x_eval_cycle, _, _ = make_lstm_arrays(
        eval_samples,
        x_scaler_cycle,
        y_scaler_cycle,
        cluster_count=cluster_count,
        include_cluster=True,
        fit_scalers=False,
        numeric_features=TREND_CYCLE_SEQUENCE_FEATURES,
        target_column="target_cycle",
        target_transform="identity",
        require_target=False,
    )
    cycle_pred, cycle_backend = train_predict_lstm(
        x_train_cycle,
        y_train_cycle,
        x_eval_cycle,
        y_scaler_cycle,
        epochs=epochs,
        use_asymmetric_loss=False,
        target_transform="identity",
    )
    return trend_pred, cycle_pred, trend_backend, cycle_backend


def apply_trend_cycle_adjustment(
    base_prediction: np.ndarray,
    trend_pred: np.ndarray,
    cycle_pred: np.ndarray,
    metadata: pd.DataFrame,
    regime: np.ndarray,
    enable_trend_cycle_model: bool = True,
    trend_slope_beta: float = DEFAULT_TREND_SLOPE_BETA,
    max_volatility_scale: float = DEFAULT_MAX_VOLATILITY_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_prediction = np.asarray(base_prediction, dtype=float)
    trend_pred = np.asarray(trend_pred, dtype=float)
    cycle_pred = np.asarray(cycle_pred, dtype=float)
    regime = np.asarray(regime)

    pred_cycle_std = float(np.nanstd(cycle_pred))
    actual_std = metadata["cycle_volatility_6m"].to_numpy(dtype=float)
    volatility_scale = np.divide(
        actual_std,
        pred_cycle_std + 1e-6,
        out=np.ones_like(actual_std, dtype=float),
        where=np.isfinite(actual_std),
    )
    volatility_scale = np.clip(volatility_scale, 1.0 / max_volatility_scale, max_volatility_scale)
    if not np.isfinite(pred_cycle_std) or pred_cycle_std <= 1e-6:
        volatility_scale = np.ones_like(actual_std, dtype=float)

    adjusted_cycle = cycle_pred * volatility_scale
    trend_slope_rate = metadata["trend_slope_rate"].to_numpy(dtype=float)
    trend_boost = np.where(
        trend_slope_rate > 0,
        1.0 + float(trend_slope_beta) * trend_slope_rate,
        1.0,
    )
    trend_boost = np.clip(trend_boost, 1.0, 1.35)
    trend_cycle_prediction = np.clip((trend_pred * trend_boost) + adjusted_cycle, 0, None)
    apply_mask = (regime == "cycle") & bool(enable_trend_cycle_model)
    final_prediction = np.where(apply_mask, trend_cycle_prediction, base_prediction)
    return final_prediction, adjusted_cycle, volatility_scale, trend_boost, apply_mask


def safe_round_revenue(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, np.iinfo(np.int64).max - 1)
    return np.rint(values).astype(np.int64)


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    last_observed: np.ndarray | None = None,
) -> dict[str, float]:
    empty_metrics = {
        "MSE": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "MAPE": np.nan,
        "MedianAPE": np.nan,
        "WMAPE": np.nan,
        "SMAPE": np.nan,
        "Bias": np.nan,
        "UnderestimateRate": np.nan,
        "DirectionAccuracy": np.nan,
    }
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    last_observed_array = None
    if last_observed is not None:
        last_observed_array = np.asarray(last_observed, dtype=float)
        if last_observed_array.shape != actual.shape:
            raise ValueError("last_observed must have the same shape as actual.")
        valid &= np.isfinite(last_observed_array)
    if not valid.any():
        return empty_metrics
    actual = actual[valid]
    predicted = predicted[valid]
    if last_observed_array is not None:
        last_observed_array = last_observed_array[valid]

    error = predicted - actual
    abs_error = np.abs(error)
    nonzero_actual = actual != 0
    absolute_percentage_error = np.divide(
        abs_error,
        np.abs(actual),
        out=np.full_like(abs_error, np.nan, dtype=float),
        where=nonzero_actual,
    )
    wmape_denominator = float(np.abs(actual).sum())
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2.0 * abs_error,
        smape_denominator,
        out=np.full_like(abs_error, np.nan, dtype=float),
        where=smape_denominator != 0,
    )
    if last_observed_array is None:
        direction_accuracy = np.nan
    else:
        direction_accuracy = float(
            np.mean(np.sign(predicted - last_observed_array) == np.sign(actual - last_observed_array)) * 100
        )

    return {
        "MSE": float(np.mean(error**2)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(abs_error)),
        "MAPE": (
            float(np.nanmean(absolute_percentage_error) * 100)
            if np.isfinite(absolute_percentage_error).any()
            else np.nan
        ),
        "MedianAPE": (
            float(np.nanmedian(absolute_percentage_error) * 100)
            if np.isfinite(absolute_percentage_error).any()
            else np.nan
        ),
        "WMAPE": float(abs_error.sum() / wmape_denominator * 100) if wmape_denominator else np.nan,
        "SMAPE": float(np.nanmean(smape_terms) * 100) if np.isfinite(smape_terms).any() else np.nan,
        "Bias": float(np.mean(error)),
        "UnderestimateRate": float(np.mean(predicted < actual) * 100),
        "DirectionAccuracy": direction_accuracy,
    }


def build_actual_revenue_frame(revenue_data: pd.DataFrame, target_year: int = FORECAST_YEAR) -> pd.DataFrame:
    required_columns = {"stock_id", "revenue_year", "revenue_month", "revenue_thousand"}
    if not required_columns.issubset(revenue_data.columns):
        missing = sorted(required_columns - set(revenue_data.columns))
        raise ValueError(f"Revenue data is missing actual revenue columns: {missing}")

    actual = revenue_data[
        revenue_data["revenue_year"].astype(int).eq(int(target_year))
    ][["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]].copy()
    actual = actual.rename(
        columns={
            "revenue_year": "target_year",
            "revenue_month": "target_month",
            "revenue_thousand": "actual_revenue",
        }
    )
    for column in ["stock_id", "target_year", "target_month", "actual_revenue"]:
        actual[column] = pd.to_numeric(actual[column], errors="coerce")
    actual = actual.dropna(subset=["stock_id", "target_year", "target_month", "actual_revenue"])
    actual = actual[np.isfinite(actual["actual_revenue"]) & (actual["actual_revenue"] >= 0)].copy()
    actual["stock_id"] = actual["stock_id"].astype(int)
    actual["target_year"] = actual["target_year"].astype(int)
    actual["target_month"] = actual["target_month"].astype(int)
    return actual.sort_values(["stock_id", "target_year", "target_month"]).drop_duplicates(
        ["stock_id", "target_year", "target_month"],
        keep="last",
    ).reset_index(drop=True)


def attach_actual_revenue(forecast: pd.DataFrame, actual_revenue: pd.DataFrame) -> pd.DataFrame:
    required_forecast_columns = {"stock_id", "target_year", "target_month"}
    if not required_forecast_columns.issubset(forecast.columns):
        missing = sorted(required_forecast_columns - set(forecast.columns))
        raise ValueError(f"Forecast is missing actual merge columns: {missing}")
    required_actual_columns = {"stock_id", "target_year", "target_month", "actual_revenue"}
    if not required_actual_columns.issubset(actual_revenue.columns):
        missing = sorted(required_actual_columns - set(actual_revenue.columns))
        raise ValueError(f"Actual revenue is missing columns: {missing}")

    actual = actual_revenue[["stock_id", "target_year", "target_month", "actual_revenue"]].copy()
    for column in ["stock_id", "target_year", "target_month", "actual_revenue"]:
        actual[column] = pd.to_numeric(actual[column], errors="coerce")
    actual = actual.dropna(subset=["stock_id", "target_year", "target_month", "actual_revenue"])
    actual["stock_id"] = actual["stock_id"].astype(int)
    actual["target_year"] = actual["target_year"].astype(int)
    actual["target_month"] = actual["target_month"].astype(int)
    actual = actual.sort_values(["stock_id", "target_year", "target_month"]).drop_duplicates(
        ["stock_id", "target_year", "target_month"],
        keep="last",
    )

    forecast_without_actual = forecast.drop(columns=["actual_revenue"], errors="ignore").copy()
    return forecast_without_actual.merge(
        actual,
        on=["stock_id", "target_year", "target_month"],
        how="left",
    )


def evaluate_rolling_forecast(
    forecast: pd.DataFrame,
    actual_revenue: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluated = attach_actual_revenue(forecast, actual_revenue)
    last_observed = evaluated["last_observed_revenue"].to_numpy(dtype=float)
    actual = evaluated["actual_revenue"].to_numpy(dtype=float)
    evaluated["actual_return"] = np.divide(
        actual,
        last_observed,
        out=np.full(len(evaluated), np.nan, dtype=float),
        where=last_observed != 0,
    ) - 1

    for _, prediction_column, error_column, abs_error_column in ROLLING_MODEL_OUTPUTS:
        if prediction_column not in evaluated.columns:
            evaluated[prediction_column] = np.nan
        evaluated[error_column] = evaluated[prediction_column] - evaluated["actual_revenue"]
        evaluated[abs_error_column] = evaluated[error_column].abs()

    metric_rows = []
    valid = evaluated.dropna(subset=["actual_revenue"])
    for model_name, prediction_column, _, _ in ROLLING_MODEL_OUTPUTS:
        if valid.empty:
            metrics = compute_metrics(np.array([], dtype=float), np.array([], dtype=float))
        else:
            metrics = compute_metrics(
                valid["actual_revenue"].to_numpy(),
                valid[prediction_column].to_numpy(),
                valid["last_observed_revenue"].to_numpy(),
            )
        metric_rows.append({"model": model_name, **metrics})
    metrics_frame = pd.DataFrame(metric_rows)
    for column in [
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "MedianAPE",
        "WMAPE",
        "SMAPE",
        "Bias",
        "UnderestimateRate",
        "DirectionAccuracy",
    ]:
        metrics_frame[column] = metrics_frame[column].round(3)
    return evaluated, metrics_frame


def train_cluster_validation_candidate(
    train_samples: list[dict[str, object]],
    eval_samples: list[dict[str, object]],
    cluster_count: int,
    epochs: int,
    max_train_samples: int,
    enable_growth_adjustment: bool,
    growth_adjustment_alpha: float,
    enable_conditional_adjustment: bool,
    enable_regime_strategy: bool,
    use_asymmetric_loss: bool,
    under_weight: float,
) -> dict[str, object]:
    from sklearn.preprocessing import StandardScaler

    sampled_train_samples, capped = cap_training_samples(train_samples, max_train_samples)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train, y_train, _ = make_lstm_arrays(
        sampled_train_samples,
        x_scaler,
        y_scaler,
        cluster_count=cluster_count,
        include_cluster=True,
        fit_scalers=True,
    )
    x_eval, _, eval_meta = make_lstm_arrays(
        eval_samples,
        x_scaler,
        y_scaler,
        cluster_count=cluster_count,
        include_cluster=True,
        fit_scalers=False,
    )
    pred, backend_used = train_predict_lstm(
        x_train,
        y_train,
        x_eval,
        y_scaler,
        epochs=epochs,
        use_asymmetric_loss=use_asymmetric_loss,
        under_weight=under_weight,
    )
    pred, clipped_base_count, _ = apply_revenue_guardrails(
        pred,
        eval_meta,
    )
    pred_adjusted, _, _, _, _, _ = apply_growth_adjustment(
        pred,
        eval_meta,
        alpha=growth_adjustment_alpha,
        enable_growth_adjustment=enable_growth_adjustment,
        enable_conditional_adjustment=enable_conditional_adjustment,
        enable_regime_strategy=enable_regime_strategy,
    )
    pred_adjusted, clipped_adjusted_count, _ = apply_revenue_guardrails(
        pred_adjusted,
        eval_meta,
    )
    metrics = compute_metrics(
        eval_meta["actual_revenue"].to_numpy(),
        pred_adjusted,
        eval_meta["last_observed_revenue"].to_numpy() if "last_observed_revenue" in eval_meta else None,
    )
    metrics.update(
        {
            "backend": backend_used,
            "train_samples_used": len(sampled_train_samples),
            "train_samples_available": len(train_samples),
            "sample_capped": capped,
            "validation_samples": len(eval_samples),
            "clipped_base_count": clipped_base_count,
            "clipped_adjusted_count": clipped_adjusted_count,
        }
    )
    return metrics


def tune_hyperparameters(
    df: pd.DataFrame,
    windows: pd.DataFrame,
    selected_stock: int,
    window_size: int,
    k_candidates: list[int] | tuple[int, ...] | None,
    max_train_sample_candidates: list[int] | tuple[int, ...] | None,
    tuning_year: int,
    tuning_epochs: int,
    tuning_metric: Literal["MAE", "MAPE", "UnderestimateRate"],
    enable_growth_adjustment: bool,
    growth_adjustment_alpha: float,
    enable_conditional_adjustment: bool,
    enable_regime_strategy: bool,
    use_asymmetric_loss: bool,
    under_weight: float,
) -> tuple[int, int, pd.DataFrame, dict[str, object]]:
    if k_candidates is None:
        k_candidates = [4, 6, 8]
    if max_train_sample_candidates is None:
        max_train_sample_candidates = [10_000, 40_000]

    k_values = sorted({int(k) for k in k_candidates if min(CLUSTER_RANGE) <= int(k) <= max(CLUSTER_RANGE)})
    sample_values = sorted({int(value) for value in max_train_sample_candidates if int(value) > 0})
    if not k_values:
        k_values = [6]
    if not sample_values:
        sample_values = [40_000]

    train_end_year = tuning_year - 1
    rows: list[dict[str, object]] = []
    for k_value in k_values:
        k_start = time.perf_counter()
        try:
            _, clustered_windows, _ = fit_kmeans_clusters(windows, k=k_value, train_end_year=train_end_year)
            monthly = attach_clusters_to_monthly(df, clustered_windows)
            train_samples, eval_samples = build_supervised_sequences_for_year(
                monthly,
                selected_stock=selected_stock,
                window_size=window_size,
                train_end_year=train_end_year,
                eval_year=tuning_year,
            )
            if not train_samples or not eval_samples:
                raise ValueError(f"No validation samples for tuning year {tuning_year}.")
            cluster_count = int(clustered_windows["cluster"].max()) + 1
            for sample_value in sample_values:
                start = time.perf_counter()
                try:
                    metrics = train_cluster_validation_candidate(
                        train_samples,
                        eval_samples,
                        cluster_count=cluster_count,
                        epochs=tuning_epochs,
                        max_train_samples=sample_value,
                        enable_growth_adjustment=enable_growth_adjustment,
                        growth_adjustment_alpha=growth_adjustment_alpha,
                        enable_conditional_adjustment=enable_conditional_adjustment,
                        enable_regime_strategy=enable_regime_strategy,
                        use_asymmetric_loss=use_asymmetric_loss,
                        under_weight=under_weight,
                    )
                    row = {
                        "k": k_value,
                        "max_train_samples": sample_value,
                        "tuning_year": tuning_year,
                        "tuning_epochs": tuning_epochs,
                        "duration_sec": round(time.perf_counter() - start, 2),
                        "error": "",
                        **metrics,
                    }
                except Exception as error:
                    row = {
                        "k": k_value,
                        "max_train_samples": sample_value,
                        "tuning_year": tuning_year,
                        "tuning_epochs": tuning_epochs,
                        "duration_sec": round(time.perf_counter() - start, 2),
                        "error": str(error),
                    }
                rows.append(row)
        except Exception as error:
            for sample_value in sample_values:
                rows.append(
                    {
                        "k": k_value,
                        "max_train_samples": sample_value,
                        "tuning_year": tuning_year,
                        "tuning_epochs": tuning_epochs,
                        "duration_sec": round(time.perf_counter() - k_start, 2),
                        "error": str(error),
                    }
                )

    tuning_results = pd.DataFrame(rows)
    valid = tuning_results[tuning_results["error"].fillna("") == ""].copy()
    if valid.empty:
        raise ValueError("All hyperparameter tuning candidates failed.")

    metric_column = tuning_metric if tuning_metric in valid.columns else "MAE"
    best = valid.sort_values([metric_column, "MAE", "MAPE", "max_train_samples"], ascending=True).iloc[0]
    selected_params = {
        "k": int(best["k"]),
        "max_train_samples": int(best["max_train_samples"]),
        "tuning_year": int(tuning_year),
        "tuning_epochs": int(tuning_epochs),
        "tuning_metric": metric_column,
        "validation_MAE": float(best["MAE"]),
        "validation_MAPE": float(best["MAPE"]),
        "validation_WMAPE": float(best["WMAPE"]) if "WMAPE" in valid.columns else np.nan,
        "validation_SMAPE": float(best["SMAPE"]) if "SMAPE" in valid.columns else np.nan,
        "validation_UnderestimateRate": float(best["UnderestimateRate"]),
        "validation_DirectionAccuracy": (
            float(best["DirectionAccuracy"]) if "DirectionAccuracy" in valid.columns else np.nan
        ),
    }
    round_columns = [
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "MedianAPE",
        "WMAPE",
        "SMAPE",
        "Bias",
        "UnderestimateRate",
        "DirectionAccuracy",
        "duration_sec",
    ]
    for column in round_columns:
        if column in tuning_results.columns:
            tuning_results[column] = pd.to_numeric(tuning_results[column], errors="coerce").round(3)
    return selected_params["k"], selected_params["max_train_samples"], tuning_results, selected_params


def assemble_rolling_forecast(
    prediction_frame: pd.DataFrame,
    actual_revenue: pd.DataFrame,
    growth_config: GrowthAdjustmentConfig,
    xlstm_growth_config: GrowthAdjustmentConfig,
    *,
    include_xlstm_plain: bool,
) -> RollingForecastAssemblyResult:
    """Apply all post-model policies and expose one testable forecast contract."""
    required_raw_columns = {"raw_pred_cluster", "raw_pred_plain"}
    missing = sorted(required_raw_columns.difference(prediction_frame.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing required raw columns: {missing}")

    raw_prediction_columns = [
        "raw_pred_cluster",
        "raw_pred_plain",
        "raw_pred_xlstm",
        "raw_pred_trend",
        "raw_pred_cycle",
    ]
    eval_meta = prediction_frame.drop(columns=raw_prediction_columns, errors="ignore").copy()
    pred_cluster = prediction_frame["raw_pred_cluster"].to_numpy(dtype=float)
    pred_plain = prediction_frame["raw_pred_plain"].to_numpy(dtype=float)
    pred_xlstm = pd.to_numeric(
        prediction_frame.get("raw_pred_xlstm", pd.Series(np.nan, index=prediction_frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)

    pred_cluster, clipped_cluster_count, _ = apply_revenue_guardrails(pred_cluster, eval_meta)
    pred_plain, clipped_plain_count, _ = apply_revenue_guardrails(pred_plain, eval_meta)
    xlstm_plain_available = bool(include_xlstm_plain and np.isfinite(pred_xlstm).any())
    if xlstm_plain_available:
        pred_xlstm, clipped_xlstm_count, _ = apply_revenue_guardrails(pred_xlstm, eval_meta)
    else:
        pred_xlstm = np.full(len(eval_meta), np.nan, dtype=float)
        clipped_xlstm_count = 0

    pred_xlstm_adjusted = np.full(len(eval_meta), np.nan, dtype=float)
    xlstm_adjustment_ratio = np.full(len(eval_meta), np.nan, dtype=float)
    xlstm_adjustment_applied = np.zeros(len(eval_meta), dtype=bool)
    clipped_xlstm_adjusted_count = 0
    if xlstm_plain_available:
        (
            pred_xlstm_adjusted,
            _,
            xlstm_adjustment_ratio,
            _,
            _,
            xlstm_adjustment_applied,
        ) = apply_growth_adjustment(
            pred_xlstm,
            eval_meta,
            alpha=float(xlstm_growth_config.alpha),
            enable_growth_adjustment=bool(xlstm_growth_config.enabled),
            enable_conditional_adjustment=bool(xlstm_growth_config.conditional),
            enable_regime_strategy=bool(xlstm_growth_config.regime_strategy),
            decline_cap_growth_ratio_max=xlstm_growth_config.decline_cap_growth_ratio_max,
            decline_cap_prediction_ratio_min=float(
                xlstm_growth_config.decline_cap_prediction_ratio_min
            ),
        )
        pred_xlstm_adjusted, clipped_xlstm_adjusted_count, _ = apply_revenue_guardrails(
            pred_xlstm_adjusted,
            eval_meta,
        )
        xlstm_adjustment_ratio = np.divide(
            pred_xlstm_adjusted,
            np.where(pred_xlstm == 0, np.nan, pred_xlstm),
            out=np.ones_like(pred_xlstm_adjusted, dtype=float),
            where=pred_xlstm != 0,
        )

    (
        pred_adjusted,
        growth_signal,
        adjustment_ratio,
        regime,
        is_growth_phase,
        adjustment_applied,
    ) = apply_growth_adjustment(
        pred_cluster,
        eval_meta,
        alpha=float(growth_config.alpha),
        enable_growth_adjustment=bool(growth_config.enabled),
        enable_conditional_adjustment=bool(growth_config.conditional),
        enable_regime_strategy=bool(growth_config.regime_strategy),
        decline_cap_growth_ratio_max=growth_config.decline_cap_growth_ratio_max,
        decline_cap_prediction_ratio_min=float(growth_config.decline_cap_prediction_ratio_min),
    )
    pred_adjusted, clipped_adjusted_count, adjusted_prediction_cap = apply_revenue_guardrails(
        pred_adjusted,
        eval_meta,
    )
    adjustment_ratio = np.divide(
        pred_adjusted,
        np.where(pred_cluster == 0, np.nan, pred_cluster),
        out=np.ones_like(pred_adjusted, dtype=float),
        where=pred_cluster != 0,
    )
    decline_cap_applied = calculate_decline_cap_mask(
        pred_cluster,
        eval_meta,
        regime,
        enable_regime_strategy=bool(growth_config.regime_strategy),
        decline_cap_growth_ratio_max=growth_config.decline_cap_growth_ratio_max,
        decline_cap_prediction_ratio_min=float(growth_config.decline_cap_prediction_ratio_min),
    )
    xlstm_decline_cap_applied = np.zeros(len(eval_meta), dtype=bool)
    if xlstm_plain_available:
        xlstm_decline_cap_applied = calculate_decline_cap_mask(
            pred_xlstm,
            eval_meta,
            regime,
            enable_regime_strategy=bool(xlstm_growth_config.regime_strategy),
            decline_cap_growth_ratio_max=xlstm_growth_config.decline_cap_growth_ratio_max,
            decline_cap_prediction_ratio_min=float(
                xlstm_growth_config.decline_cap_prediction_ratio_min
            ),
        )

    forecast = eval_meta.copy()
    forecast["regime"] = regime
    forecast["is_growth_phase"] = is_growth_phase
    forecast["growth_signal"] = growth_signal
    forecast["adjustment_applied"] = adjustment_applied
    forecast["decline_cap_applied"] = decline_cap_applied
    forecast["is_high_growth_flag"] = calculate_high_growth_flag(forecast)
    forecast["prediction_cap"] = adjusted_prediction_cap
    forecast["historical_trend_component"] = np.round(
        eval_meta["trend_component"].to_numpy(dtype=float),
        2,
    )
    forecast["historical_cycle_component"] = np.round(
        eval_meta["cycle_component"].to_numpy(dtype=float),
        2,
    )
    forecast["trend_slope"] = np.round(eval_meta["trend_slope"].to_numpy(dtype=float), 2)
    forecast["trend_slope_rate"] = np.round(
        eval_meta["trend_slope_rate"].to_numpy(dtype=float),
        4,
    )
    forecast["predicted_revenue_cluster"] = safe_round_revenue(pred_cluster)
    forecast["predicted_revenue_no_cluster"] = safe_round_revenue(pred_plain)
    forecast["predicted_revenue_xlstm"] = (
        safe_round_revenue(pred_xlstm) if xlstm_plain_available else np.nan
    )
    forecast["predicted_revenue_xlstm_adjusted"] = (
        safe_round_revenue(pred_xlstm_adjusted) if xlstm_plain_available else np.nan
    )
    forecast["predicted_revenue_base"] = forecast["predicted_revenue_cluster"]
    forecast["predicted_revenue_adjusted"] = safe_round_revenue(pred_adjusted)
    forecast["prediction_base"] = forecast["predicted_revenue_base"]
    forecast["prediction_adjusted"] = forecast["predicted_revenue_adjusted"]
    forecast["prediction_xlstm_adjusted"] = forecast["predicted_revenue_xlstm_adjusted"]
    forecast["adjustment_ratio"] = adjustment_ratio
    forecast["xlstm_adjustment_ratio"] = xlstm_adjustment_ratio
    forecast["xlstm_adjustment_applied"] = xlstm_adjustment_applied
    forecast["xlstm_decline_cap_applied"] = xlstm_decline_cap_applied
    forecast["decline_cap_growth_ratio_max"] = growth_config.decline_cap_growth_ratio_max
    forecast["decline_cap_prediction_ratio_min"] = (
        growth_config.decline_cap_prediction_ratio_min
    )
    forecast["xlstm_decline_cap_growth_ratio_max"] = (
        xlstm_growth_config.decline_cap_growth_ratio_max
    )
    forecast["xlstm_decline_cap_prediction_ratio_min"] = (
        xlstm_growth_config.decline_cap_prediction_ratio_min
    )
    for model_suffix in [
        "cluster",
        "no_cluster",
        "xlstm",
        "xlstm_adjusted",
        "adjusted",
    ]:
        forecast[f"predicted_return_{model_suffix}"] = (
            forecast[f"predicted_revenue_{model_suffix}"]
            / forecast["last_observed_revenue"]
            - 1
        )
    forecast = forecast.sort_values("target_date").reset_index(drop=True)
    forecast, metrics = evaluate_rolling_forecast(forecast, actual_revenue)
    return RollingForecastAssemblyResult(
        forecast=forecast,
        metrics=metrics,
        clip_counts={
            "cluster": int(clipped_cluster_count),
            "plain": int(clipped_plain_count),
            "xlstm": int(clipped_xlstm_count),
            "xlstm_adjusted": int(clipped_xlstm_adjusted_count),
            "adjusted": int(clipped_adjusted_count),
        },
        xlstm_plain_available=xlstm_plain_available,
    )


def run_rolling_lstm_experiment(
    selected_stock: int,
    config: RollingExperimentConfig | None = None,
    **legacy_options: object,
) -> RollingLSTMResult:
    config = _normalize_experiment_config(config, legacy_options)
    k = int(config.k)
    window_size = int(config.window_size)
    epochs = int(config.epochs)
    max_train_samples = int(config.max_train_samples)
    sequence_backbone = str(config.sequence_backbone).lower()
    include_xlstm_plain = bool(config.include_xlstm_plain)
    xlstm_backbone = str(config.xlstm_backbone).lower()
    enable_growth_adjustment = bool(config.growth.enabled)
    growth_adjustment_alpha = float(config.growth.alpha)
    enable_conditional_adjustment = bool(config.growth.conditional)
    enable_regime_strategy = bool(config.growth.regime_strategy)
    decline_cap_growth_ratio_max = config.growth.decline_cap_growth_ratio_max
    decline_cap_prediction_ratio_min = float(config.growth.decline_cap_prediction_ratio_min)
    xlstm_enable_growth_adjustment = bool(config.xlstm_growth.enabled)
    xlstm_growth_adjustment_alpha = float(config.xlstm_growth.alpha)
    xlstm_enable_conditional_adjustment = bool(config.xlstm_growth.conditional)
    xlstm_enable_regime_strategy = bool(config.xlstm_growth.regime_strategy)
    xlstm_decline_cap_growth_ratio_max = config.xlstm_growth.decline_cap_growth_ratio_max
    xlstm_decline_cap_prediction_ratio_min = float(config.xlstm_growth.decline_cap_prediction_ratio_min)
    use_asymmetric_loss = bool(config.use_asymmetric_loss)
    under_weight = float(config.under_weight)

    notes: list[str] = []
    revenue_mtime_ns = _revenue_file_mtime_ns()
    df, windows = _cached_revenue_and_windows(window_size, revenue_mtime_ns)
    if int(selected_stock) not in set(get_stock_list(df)):
        raise ValueError(f"Stock {selected_stock} is not available in the revenue data.")

    if windows.empty:
        raise ValueError("No growth-direction windows could be built from the revenue data.")

    tuning_results = pd.DataFrame()
    selected_params: dict[str, object] = {
        "k": int(k),
        "max_train_samples": int(max_train_samples),
        "sequence_backbone": sequence_backbone,
        "include_xlstm_plain": include_xlstm_plain,
        "xlstm_backbone": xlstm_backbone,
        "include_yield_forecast": bool(config.include_yield_forecast),
        "yield_as_of_date": config.yield_as_of_date,
        "growth_alpha": growth_adjustment_alpha,
        "xlstm_growth_alpha": xlstm_growth_adjustment_alpha,
        "decline_cap_growth_ratio_max": decline_cap_growth_ratio_max,
        "decline_cap_prediction_ratio_min": decline_cap_prediction_ratio_min,
        "xlstm_decline_cap_growth_ratio_max": xlstm_decline_cap_growth_ratio_max,
        "xlstm_decline_cap_prediction_ratio_min": xlstm_decline_cap_prediction_ratio_min,
        "auto_tuned": False,
    }

    elbow_scores, clustered_windows, cluster_profile, monthly = _cached_clustered_artifacts(
        int(k),
        int(window_size),
        revenue_mtime_ns,
    )
    train_samples = list(_cached_train_samples(int(k), int(window_size), revenue_mtime_ns))
    eval_samples = list(_cached_eval_samples(int(k), int(window_size), int(selected_stock), revenue_mtime_ns))
    if not eval_samples:
        raise ValueError(f"Stock {selected_stock} has no {FORECAST_YEAR} rolling evaluation samples.")
    if not train_samples:
        raise ValueError("No training samples are available for rolling LSTM.")

    capped_train_sample_count = min(len(train_samples), int(max_train_samples)) if max_train_samples else len(train_samples)
    if max_train_samples and len(train_samples) > max_train_samples:
        notes.append(f"Training samples were capped at {max_train_samples:,} by deterministic random sampling.")
    prediction_frame, backend_used, backend_plain, backend_xlstm_plain, trend_backend, cycle_backend = _cached_lstm_predictions(
        int(k),
        int(window_size),
        capped_train_sample_count,
        int(epochs),
        sequence_backbone,
        include_xlstm_plain,
        use_asymmetric_loss,
        float(under_weight),
        False,
        revenue_mtime_ns,
        xlstm_backbone=xlstm_backbone,
    )
    prediction_frame = prediction_frame[
        prediction_frame["stock_id"] == int(selected_stock)
    ].sort_values("target_date").reset_index(drop=True)
    actual_revenue = build_actual_revenue_frame(df, target_year=FORECAST_YEAR)
    assembly = assemble_rolling_forecast(
        prediction_frame,
        actual_revenue,
        config.growth,
        config.xlstm_growth,
        include_xlstm_plain=include_xlstm_plain,
    )
    forecast = assembly.forecast
    metrics = attach_model_backbone_provenance(
        assembly.metrics,
        main_sequence_backbone=sequence_backbone,
        xlstm_backbone=xlstm_backbone,
        include_xlstm_plain=include_xlstm_plain,
    )
    forecast["sequence_backbone"] = sequence_backbone
    forecast["xlstm_backbone"] = xlstm_backbone if include_xlstm_plain else "disabled"
    clipped_cluster_count = assembly.clip_counts["cluster"]
    clipped_plain_count = assembly.clip_counts["plain"]
    clipped_xlstm_count = assembly.clip_counts["xlstm"]
    clipped_xlstm_adjusted_count = assembly.clip_counts["xlstm_adjusted"]
    clipped_adjusted_count = assembly.clip_counts["adjusted"]

    cluster_effect = build_cluster_effect_table(forecast)
    selected_cluster_timeline = forecast[
        [
            "target_date",
            "target_month",
            "actual_revenue",
            "last_observed_revenue",
            "actual_return",
            "cluster",
            "regime",
            "is_growth_phase",
            "growth_ratio",
            "growth_streak",
            "growth_signal",
            "adjustment_applied",
            "decline_cap_applied",
            "is_high_growth_flag",
            "prediction_cap",
            "trend_slope",
            "trend_slope_rate",
            "prediction_base",
            "prediction_adjusted",
            "predicted_revenue_base",
            "predicted_revenue_adjusted",
            "adjustment_ratio",
            "predicted_revenue_cluster",
            "predicted_revenue_no_cluster",
            "predicted_revenue_xlstm",
            "predicted_revenue_xlstm_adjusted",
            "predicted_return_adjusted",
            "predicted_return_cluster",
            "predicted_return_no_cluster",
            "predicted_return_xlstm",
            "predicted_return_xlstm_adjusted",
            "adjusted_abs_error",
            "cluster_abs_error",
            "no_cluster_abs_error",
            "xlstm_abs_error",
            "xlstm_adjusted_abs_error",
            "xlstm_adjustment_ratio",
            "xlstm_adjustment_applied",
            "xlstm_decline_cap_applied",
        ]
    ].copy()

    yield_forecast = pd.DataFrame()
    yield_summary = pd.DataFrame()
    yield_notes: list[str] = []
    if config.include_yield_forecast:
        try:
            yield_result = build_rolling_yield_forecast(
                forecast,
                selected_stock=int(selected_stock),
                target_year=FORECAST_YEAR,
                model_columns={
                    model_name: prediction_column
                    for model_name, prediction_column, _, _ in ROLLING_MODEL_OUTPUTS
                },
                data_dir=DATA_DIR,
                as_of_date=config.yield_as_of_date,
            )
            yield_forecast = yield_result.monthly
            yield_summary = yield_result.summary
            if not yield_forecast.empty:
                yield_forecast = attach_model_backbone_provenance(
                    yield_forecast,
                    main_sequence_backbone=sequence_backbone,
                    xlstm_backbone=xlstm_backbone,
                    include_xlstm_plain=include_xlstm_plain,
                )
            if not yield_summary.empty:
                yield_summary = attach_model_backbone_provenance(
                    yield_summary,
                    main_sequence_backbone=sequence_backbone,
                    xlstm_backbone=xlstm_backbone,
                    include_xlstm_plain=include_xlstm_plain,
                )
            yield_notes = yield_result.notes
        except (FileNotFoundError, KeyError, ValueError, pd.errors.EmptyDataError) as error:
            yield_notes = [f"Rolling dividend-yield forecast unavailable: {error}"]

    notes.append(
        f"Sequence backbone={sequence_backbone}; with cluster={backend_used}, "
        f"without cluster={backend_plain}; optional xLSTM backbone={xlstm_backbone}; "
        f"xLSTM plain={backend_xlstm_plain}."
    )
    if clipped_cluster_count or clipped_plain_count:
        notes.append(
            "Prediction guardrail clipped "
            f"{clipped_cluster_count} clustered and {clipped_plain_count} plain predictions "
            "that exceeded the recent-revenue cap."
        )
    if clipped_xlstm_count:
        notes.append(
            "Prediction guardrail clipped "
            f"{clipped_xlstm_count} xLSTM plain predictions that exceeded the recent-revenue cap."
        )
    if clipped_xlstm_adjusted_count:
        notes.append(
            "Prediction guardrail clipped "
            f"{clipped_xlstm_adjusted_count} xLSTM adjusted predictions that exceeded the recent-revenue cap."
        )
    if clipped_adjusted_count:
        notes.append(
            "Prediction guardrail clipped "
            f"{clipped_adjusted_count} growth-adjusted predictions that exceeded the recent-revenue cap."
        )
    notes.append(
        "Growth adjustment "
        f"enabled={enable_growth_adjustment}, alpha={growth_adjustment_alpha:.2f}; "
        f"conditional={enable_conditional_adjustment}, direction gate=last growth rate > 0, "
        f"regime_strategy={enable_regime_strategy}; "
        f"decline_cap_growth_ratio_max={decline_cap_growth_ratio_max}, "
        f"decline_cap_prediction_ratio_min={decline_cap_prediction_ratio_min:.2f}; "
        f"asymmetric_loss enabled={use_asymmetric_loss}, under_weight={under_weight:.2f}."
    )
    if include_xlstm_plain:
        notes.append(
            "xLSTM adjustment "
            f"enabled={xlstm_enable_growth_adjustment}, alpha={xlstm_growth_adjustment_alpha:.2f}; "
            f"conditional={xlstm_enable_conditional_adjustment}, direction gate=last growth rate > 0, "
            f"regime_strategy={xlstm_enable_regime_strategy}; "
            f"decline_cap_growth_ratio_max={xlstm_decline_cap_growth_ratio_max}, "
            f"decline_cap_prediction_ratio_min={xlstm_decline_cap_prediction_ratio_min:.2f}."
        )
    if enable_regime_strategy or (include_xlstm_plain and xlstm_enable_regime_strategy):
        notes.append(
            "Regime strategy can cap enabled decline-regime adjusted predictions at last observed revenue; "
            "xLSTM uses a stricter balanced decline-cap gate by default."
        )
    notes.append("KMeans is fit only on windows ending at or before 2024, then applied to 2025 windows.")
    notes.append(
        "Neural early stopping uses the latest historical target year as forward validation; "
        "feature and target scalers are fit only on earlier target years."
    )
    notes.append("Both LSTM models use the same rolling samples; the only difference is the cluster one-hot feature.")
    if include_xlstm_plain:
        notes.append(
            f"Rolling xLSTM uses backbone={xlstm_backbone} as a plain no-cluster comparison model "
            "with the same numeric sequences; its conditional adjustment reuses the same time-safe "
            "growth gates as the clustered adjustment."
        )
    if sequence_backbone == "xlstm":
        notes.append(
            "xLSTM spike uses an mLSTM-only xLSTMBlockStack as the recurrent backbone; "
            "sLSTM CUDA kernels are not part of this D1 path."
        )
    if sequence_backbone == "xlstm_hybrid" or (
        include_xlstm_plain and xlstm_backbone == "xlstm_hybrid"
    ):
        notes.append(
            "xLSTM Hybrid stacks one mLSTM block followed by one sLSTM block. The sLSTM block "
            f"uses backend={SLSTM_BACKEND}; model tensors still run on the selected PyTorch CUDA "
            "device when CUDA is available."
        )
    notes.append("2025 actual revenue is attached only after raw predictions are produced, during evaluation.")
    notes.append(
        "Training arrays and all-stock 2025 LSTM predictions are cached by model parameters; "
        "changing stocks reuses the trained models and only rebuilds stock-specific evaluation outputs."
    )
    notes.append("Return columns are revenue growth from the last observed month to the predicted target month.")

    return RollingLSTMResult(
        forecast=forecast,
        yield_forecast=yield_forecast,
        yield_summary=yield_summary,
        metrics=metrics,
        cluster_profile=cluster_profile,
        cluster_effect=cluster_effect,
        selected_cluster_timeline=selected_cluster_timeline,
        elbow_scores=elbow_scores,
        tuning_results=tuning_results,
        selected_params=selected_params,
        yield_notes=yield_notes,
        notes=notes,
    )


def build_cluster_effect_table(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame()

    rows = []
    for cluster, frame in forecast.groupby("cluster"):
        cluster_mae = float(frame["cluster_abs_error"].mean())
        plain_mae = float(frame["no_cluster_abs_error"].mean())
        adjusted_mae = float(frame["adjusted_abs_error"].mean())
        xlstm_mae = float(frame["xlstm_abs_error"].mean()) if "xlstm_abs_error" in frame else np.nan
        xlstm_adjusted_mae = (
            float(frame["xlstm_adjusted_abs_error"].mean()) if "xlstm_adjusted_abs_error" in frame else np.nan
        )
        rows.append(
            {
                "cluster": int(cluster),
                "months": int(len(frame)),
                "adjusted_model_mae": round(adjusted_mae, 0),
                "cluster_model_mae": round(cluster_mae, 0),
                "no_cluster_model_mae": round(plain_mae, 0),
                "xlstm_plain_model_mae": round(xlstm_mae, 0) if pd.notna(xlstm_mae) else np.nan,
                "xlstm_adjusted_model_mae": (
                    round(xlstm_adjusted_mae, 0) if pd.notna(xlstm_adjusted_mae) else np.nan
                ),
                "adjusted_vs_cluster_mae_improvement": round(cluster_mae - adjusted_mae, 0),
                "cluster_vs_plain_mae_improvement": round(plain_mae - cluster_mae, 0),
                "avg_actual_revenue": round(float(frame["actual_revenue"].mean()), 0),
                "avg_actual_return": round(float(frame["actual_return"].mean()) * 100, 2),
                "high_growth_months": int(frame["is_high_growth_flag"].sum()),
                "growth_phase_months": int(frame["is_growth_phase"].sum()),
                "adjustment_applied_months": int(frame["adjustment_applied"].sum()),
                "decline_cap_applied_months": int(frame["decline_cap_applied"].sum()),
                "xlstm_decline_cap_applied_months": (
                    int(frame["xlstm_decline_cap_applied"].sum())
                    if "xlstm_decline_cap_applied" in frame
                    else 0
                ),
                "cycle_months": int(frame["regime"].eq("cycle").sum()),
                "decline_months": int(frame["regime"].eq("decline").sum()),
                "avg_growth_signal": round(float(frame["growth_signal"].mean()), 4),
                "avg_adjustment_ratio": round(float(frame["adjustment_ratio"].mean()), 4),
                "avg_adjusted_prediction": round(float(frame["predicted_revenue_adjusted"].mean()), 0),
                "avg_cluster_prediction": round(float(frame["predicted_revenue_cluster"].mean()), 0),
                "avg_no_cluster_prediction": round(float(frame["predicted_revenue_no_cluster"].mean()), 0),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
