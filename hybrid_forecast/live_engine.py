from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from financial_forecast import (
    FinancialForecastPolicy,
    calculate_as_of_yields,
    forecast_financial_components,
)
from financial_forecast.contracts import EPS_METHOD_KNOWN_QUARTERS, DIVIDEND_METHOD_CLASSIFIED
from financial_forecast.evidence import load_available_prices, resolve_data_files
from hybrid_forecast.hybrid_engine import HybridConfig, combine_predictions
from revenue_adjustment_formula.formula_engine import FormulaConfig, MIN_FORMULA_HISTORY, _formula_base
from sarima_forecast import sarima_engine


SYSTEM_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = SYSTEM_DIR / "outputs" / "live_artifacts"
SARIMA_WEIGHT = 0.1
FORMULA_CONFIG = FormulaConfig(
    seasonal_weight=0.5, residual_alpha=0.1, residual_strength=0.0,
    growth_log_cap=float(np.log(2.0)), correction_log_cap=0.5,
)
LIVE_CACHE_VERSION = "hybrid_live_v3_price_independent"
LLM_SCHEMA_VERSION = "hybrid-forecast-llm-v1"
CORE_TABLES = {
    "summary": "annual_summary.csv",
    "monthly": "monthly_revenue.csv",
    "quarterly_eps": "quarterly_eps.csv",
    "payout_history": "payout_history.csv",
    "data_status": "data_status.csv",
    "order_search": "sarima_order_search.csv",
}
DATE_COLUMNS = {
    "target_date", "as_of_date", "latest_actual_month", "pattern_as_of_date",
    "price_date", "latest_period", "latest_available_date",
}


@dataclass(frozen=True)
class ForecastRequest:
    stock_id: int
    as_of_date: str | pd.Timestamp
    data_dir: str | Path

    @property
    def cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.as_of_date).normalize()

    @property
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()


@dataclass
class CoreForecastResult:
    monthly: pd.DataFrame
    summary: pd.DataFrame
    quarterly_eps: pd.DataFrame
    payout_history: pd.DataFrame
    data_status: pd.DataFrame
    order_search: pd.DataFrame
    notes: list[str]
    artifact_id: str
    generated_at: str
    cache_hit: bool = False


@dataclass
class LiveForecastResult:
    monthly: pd.DataFrame
    summary: pd.DataFrame
    quarterly_eps: pd.DataFrame
    payout_history: pd.DataFrame
    data_status: pd.DataFrame
    order_search: pd.DataFrame
    notes: list[str]
    artifact_id: str = ""
    generated_at: str = ""
    cache_hit: bool = False
    price_source: str | None = None


def data_fingerprint(data_dir: str | Path) -> tuple:
    root = Path(data_dir).expanduser().resolve()
    paths = [root / "manifest.json", *resolve_data_files(root).values()]
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file()
        else (str(path), None, None) for path in paths
    )


def core_data_fingerprint(data_dir: str | Path) -> tuple:
    root = Path(data_dir).expanduser().resolve()
    resolved = resolve_data_files(root)
    paths = [root / "manifest.json", resolved["revenue"], resolved["eps"], resolved["dividends"]]
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file()
        else (str(path), None, None) for path in paths
    )


def _file_digest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"file": path.name, "sha256": None, "size": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "sha256": digest.hexdigest(), "size": path.stat().st_size}


def _core_identity(request: ForecastRequest) -> tuple[str, dict[str, object]]:
    paths = resolve_data_files(request.resolved_data_dir)
    sources = {
        kind: _file_digest(paths[kind]) for kind in ("revenue", "eps", "dividends")
    }
    identity = {
        "pipeline_version": LIVE_CACHE_VERSION,
        "stock_id": int(request.stock_id),
        "as_of_date": request.cutoff.date().isoformat(),
        "sarima_weight": SARIMA_WEIGHT,
        "formula_config": FORMULA_CONFIG.as_dict(),
        "sources": sources,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20], identity


def artifact_directory(
    request: ForecastRequest,
    artifact_id: str,
    artifact_root: str | Path | None = None,
) -> Path:
    root = Path(artifact_root) if artifact_root is not None else DEFAULT_ARTIFACT_ROOT
    return root / str(int(request.stock_id)) / request.cutoff.date().isoformat() / artifact_id


def load_revenue_snapshot(data_dir: str | Path, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of_date).normalize()
    frame = pd.read_csv(resolve_data_files(data_dir)["revenue"])
    required = {"stock_id", "revenue_year", "revenue_month", "revenue_thousand", "revenue_available_date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"營收 CSV 缺少欄位：{sorted(required - set(frame.columns))}")
    for column in ["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["available_date"] = pd.to_datetime(frame["revenue_available_date"], errors="coerce")
    frame["date"] = pd.to_datetime(dict(
        year=frame["revenue_year"], month=frame["revenue_month"], day=1,
    ), errors="coerce")
    invalid = frame[list(required - {"revenue_available_date"})].isna().any(axis=1) | frame["available_date"].isna() | frame["date"].isna()
    invalid |= ~np.isfinite(frame["revenue_thousand"]) | frame["revenue_thousand"].lt(0)
    skipped = int(invalid.sum())
    frame = frame[~invalid & frame["available_date"].le(cutoff) & frame["date"].lt(cutoff.to_period("M").start_time)].copy()
    for column in ["stock_id", "revenue_year", "revenue_month"]:
        frame[column] = frame[column].astype(int)
    frame = frame.sort_values(["stock_id", "date"]).reset_index(drop=True)
    frame.attrs["invalid_rows"] = skipped
    return frame


def _forecast_components(history: pd.DataFrame, dates: pd.DatetimeIndex):
    values = history["revenue_thousand"].to_numpy(dtype=float)
    order_search = pd.DataFrame()
    sarima_values = np.full(len(dates), np.nan)
    sarima_reason = "連續歷史不足 36 個月"
    if len(values) >= sarima_engine.MIN_HISTORY_MONTHS and len(dates):
        try:
            order, seasonal, order_search = sarima_engine.select_sarima_order(values)
            if order is None:
                sarima_reason = "沒有收斂的 SARIMA 候選模型"
            else:
                if not order_search.empty:
                    order_search = order_search.copy()
                    order_search["selected"] = (
                        order_search["order"].astype(str).eq(str(order))
                        & order_search["seasonal_order"].astype(str).eq(str(seasonal))
                    )
                fitted = sarima_engine._fit_sarima(np.log1p(values), order, seasonal, 100)
                if not fitted.mle_retvals.get("converged", True):
                    raise ValueError("SARIMA 最終擬合未收斂")
                logs = np.asarray(fitted.get_forecast(steps=len(dates)).predicted_mean, dtype=float)
                max_log = float(np.log1p(np.iinfo(np.int64).max - 1))
                safe = np.isfinite(logs) & (logs < max_log)
                sarima_values[safe] = np.expm1(np.maximum(logs[safe], 0))
                sarima_reason = ""
        except (ImportError, ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            sarima_reason = str(error)
    projected = values.tolist()
    formula_rows, sarima_rows = [], []
    for index, date in enumerate(dates):
        previous = projected[-1] if projected else np.nan
        formula, method = np.nan, "unavailable"
        try:
            if not projected or not np.isfinite(previous):
                raise ValueError("無有效公式歷史")
            if len(projected) >= MIN_FORMULA_HISTORY:
                formula, _, _ = _formula_base(np.asarray(projected), FORMULA_CONFIG)
                method = "revenue_adjustment_formula"
            elif len(projected) >= 12:
                formula, method = projected[-12], "seasonal_naive_fallback"
            else:
                formula, method = previous, "last_observed_fallback"
        except (ValueError, ArithmeticError):
            formula = np.nan
        if not np.isfinite(formula) or formula < 0:
            formula, method = np.nan, "unavailable"
        projected.append(formula)
        key = {"stock_id": int(history.iloc[-1]["stock_id"]), "target_date": date,
            "target_year": date.year, "target_month": date.month}
        formula_rows.append({**key, "actual_revenue": np.nan, "last_observed_revenue": previous,
            "formula_adjusted_revenue": formula, "forecast_method": method})
        sarima_rows.append({**key, "predicted_revenue_sarima": sarima_values[index],
            "forecast_method": "sarima" if np.isfinite(sarima_values[index]) else "unavailable",
            "fallback_reason": sarima_reason or ("" if np.isfinite(sarima_values[index]) else "SARIMA 非有限值或超出數值範圍")})
    combined = combine_predictions(pd.DataFrame(formula_rows), pd.DataFrame(sarima_rows), HybridConfig(sarima_weight=SARIMA_WEIGHT))
    combined = combined.rename(columns={"last_observed_revenue": "formula_previous_revenue"})
    combined["formula_history_is_projected"] = np.arange(len(combined)) > 0
    return combined, order_search


def _base_notes() -> list[str]:
    return [
        "SARIMA 10%＋營收公式 90%；沿用 2023–2024 驗證後的固定權重及公式參數。",
        "SARIMA 一次多步預測；公式以自身預測遞推，不使用未公布的實際營收。",
        "年份表示獲利所屬年度；EPS 為公司稅後 EPS，未另扣投資人所得稅或補充保費。",
        "股利依五年歷史分類：固定金額用現金股利中位數；五年明確零股利用零；其餘依有效配息率估計並標記資料不足。",
        "分類非公司未來承諾；年度現金股利為已公告紀錄合計，來源沒有全年已公告完畢標記。",
        "EPS 依已公布季度及歷史 EPS／營收比率推估，尚無稅後淨利與加權平均股數資料可分別建模。",
    ]


def _financial_core(
    monthly: pd.DataFrame,
    request: ForecastRequest,
    *,
    order_search: pd.DataFrame,
    notes: list[str],
    artifact_id: str,
    generated_at: str,
    source_family: str = "hybrid",
    model: str = "SARIMA＋營收公式",
) -> CoreForecastResult:
    summaries, quarters, payouts, statuses = [], [], [], []
    policy = FinancialForecastPolicy(
        eps_methods=(EPS_METHOD_KNOWN_QUARTERS,), dividend_methods=(DIVIDEND_METHOD_CLASSIFIED,),
        yield_modes=(), min_stock_price=0.0,
    )
    for year in [request.cutoff.year, request.cutoff.year + 1]:
        year_frame = monthly[monthly["target_year"].eq(year)]
        normalized = year_frame[["stock_id", "target_year", "target_month", "revenue_used"]].rename(
            columns={"revenue_used": "predicted_revenue"})
        normalized["source_family"], normalized["model"] = source_family, model
        row = {"target_year": year, "actual_months": int(year_frame["revenue_basis"].eq("actual").sum()),
            "forecast_months": int(year_frame["revenue_basis"].eq("forecast").sum()),
            "predicted_annual_revenue": year_frame["revenue_used"].sum(min_count=12),
            "as_of_date": request.cutoff, "latest_actual_month": monthly["latest_actual_month"].max()}
        financial = forecast_financial_components(
            normalized, target_year=year, as_of_date=request.cutoff,
            data_dir=request.resolved_data_dir, policy=policy,
        )
        if not financial.summary.empty:
            row.update(financial.summary.iloc[0].to_dict())
        else:
            failure = (
                str(financial.failures.iloc[0]["status"])
                if not financial.failures.empty and "status" in financial.failures
                else "全年營收不完整"
            )
            row["status"] = f"{failure}，無法估算全年 EPS／股利"
        summaries.append(row)
        quarters.append(financial.quarterly_eps_estimates)
        payouts.append(financial.payout_history)
        statuses.append(financial.data_status)
        notes.extend(issue for issue in financial.notes if any(issue.startswith(k + ":") for k in ["revenue", "eps", "dividends"]))
    quarter_result = pd.concat([q for q in quarters if not q.empty], ignore_index=True) if any(not q.empty for q in quarters) else pd.DataFrame()
    payout_result = pd.concat([p for p in payouts if not p.empty], ignore_index=True).drop_duplicates(["stock_id", "fiscal_year"]) if any(not p.empty for p in payouts) else pd.DataFrame()
    data_status = pd.concat(statuses, ignore_index=True).drop_duplicates("dataset") if statuses else pd.DataFrame()
    return CoreForecastResult(
        monthly=monthly,
        summary=pd.DataFrame(summaries),
        quarterly_eps=quarter_result,
        payout_history=payout_result,
        data_status=data_status,
        order_search=order_search,
        notes=list(dict.fromkeys(notes)),
        artifact_id=artifact_id,
        generated_at=generated_at,
    )


def build_forecast_core(request: ForecastRequest) -> CoreForecastResult:
    """Run the price-independent Hybrid revenue, EPS, and dividend pipeline."""

    artifact_id, _ = _core_identity(request)
    revenue = load_revenue_snapshot(request.resolved_data_dir, request.cutoff)
    stock = revenue[revenue["stock_id"].eq(int(request.stock_id))].copy()
    if stock.empty:
        raise ValueError(f"{request.stock_id} 在 {request.cutoff.date()} 以前沒有可用營收")
    if stock.duplicated("date").any():
        raise ValueError(f"{request.stock_id} 營收有重複月份，請先修正 CSV")
    latest = stock["date"].max()
    history = sarima_engine._trailing_consecutive_history(stock, latest + pd.offsets.MonthBegin(1))
    end = pd.Timestamp(request.cutoff.year + 1, 12, 1)
    dates = pd.date_range(latest + pd.offsets.MonthBegin(1), end, freq="MS")
    components, order_search = _forecast_components(history, dates)
    monthly = pd.DataFrame({"target_date": pd.date_range(f"{request.cutoff.year}-01-01", end, freq="MS")})
    monthly["stock_id"] = int(request.stock_id)
    monthly["target_year"] = monthly["target_date"].dt.year
    monthly["target_month"] = monthly["target_date"].dt.month
    monthly = monthly.merge(stock[["date", "revenue_thousand"]].rename(
        columns={"date": "target_date", "revenue_thousand": "actual_revenue"}), on="target_date", how="left")
    monthly = monthly.merge(components.drop(columns=["actual_revenue", "hybrid_error", "hybrid_abs_error", "hybrid_ape"]),
        on=["stock_id", "target_date", "target_year", "target_month"], how="left")
    monthly["revenue_used"] = monthly["actual_revenue"].fillna(monthly["hybrid_predicted_revenue"])
    monthly["revenue_basis"] = np.select([
        monthly["actual_revenue"].notna(), np.isfinite(monthly["hybrid_predicted_revenue"]),
    ], ["actual", "forecast"], default="unavailable")
    monthly["as_of_date"] = request.cutoff
    monthly["latest_actual_month"] = latest
    notes = _base_notes()
    if revenue.attrs.get("invalid_rows", 0):
        notes.append(f"營收 CSV 有 {revenue.attrs['invalid_rows']} 筆無有效數值或公布日期的資料，未納入。")
    return _financial_core(
        monthly, request, order_search=order_search, notes=notes,
        artifact_id=artifact_id, generated_at=datetime.now(timezone.utc).isoformat(),
    )


def build_financial_core_from_predictions(
    revenue_predictions: pd.DataFrame,
    request: ForecastRequest,
) -> CoreForecastResult:
    """Build price-independent financial outputs from another team's normalized predictions."""

    required = {"source_family", "model", "stock_id", "target_year", "target_month", "predicted_revenue"}
    missing = required - set(revenue_predictions.columns)
    if missing:
        raise ValueError(f"Revenue predictions are missing columns: {sorted(missing)}")
    frame = revenue_predictions.copy()
    for column in ["stock_id", "target_year", "target_month", "predicted_revenue"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["stock_id"].eq(int(request.stock_id))
        & frame["target_year"].isin([request.cutoff.year, request.cutoff.year + 1])
    ].copy()
    if frame.empty:
        raise ValueError("外部預測沒有符合指定股票與年度的資料")
    if frame[["stock_id", "target_year", "target_month", "predicted_revenue"]].isna().any().any():
        raise ValueError("外部預測含有無法轉換的數值或缺值")
    if not np.isfinite(frame["predicted_revenue"]).all():
        raise ValueError("外部預測含有非有限營收")
    source_models = frame[["source_family", "model"]].drop_duplicates()
    if len(source_models) != 1:
        raise ValueError("單檔 artifact 只能包含一組 source_family 與 model")
    for year in [request.cutoff.year, request.cutoff.year + 1]:
        yearly = frame[frame["target_year"].eq(year)]
        if yearly["target_month"].duplicated().any():
            raise ValueError(f"{year} 外部預測有重複月份")
        months = sorted(yearly["target_month"].astype(int).tolist())
        if months != list(range(1, 13)):
            raise ValueError(f"{year} 外部預測月份不完整（{len(months)}/12）")
    serial = frame.sort_values(["target_year", "target_month"]).to_json(orient="records", date_format="iso")
    _, financial_identity = _core_identity(request)
    external_identity = {
        "pipeline_version": LIVE_CACHE_VERSION,
        "stock_id": int(request.stock_id),
        "as_of_date": request.cutoff.date().isoformat(),
        "source_family": str(source_models.iloc[0]["source_family"]),
        "model": str(source_models.iloc[0]["model"]),
        "prediction_sha256": hashlib.sha256(serial.encode("utf-8")).hexdigest(),
        "financial_sources": financial_identity["sources"],
    }
    artifact_id = "external-" + hashlib.sha256(
        json.dumps(external_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:11]
    monthly = frame.rename(columns={"predicted_revenue": "revenue_used"})
    monthly["target_date"] = pd.to_datetime(dict(
        year=monthly["target_year"], month=monthly["target_month"], day=1,
    ), errors="coerce")
    monthly["actual_revenue"] = np.nan
    monthly["revenue_basis"] = "forecast"
    monthly["as_of_date"] = request.cutoff
    monthly["latest_actual_month"] = pd.NaT
    return _financial_core(
        monthly, request, order_search=pd.DataFrame(),
        notes=["月營收預測由外部標準化輸入提供。", *_base_notes()[2:]],
        artifact_id=artifact_id, generated_at=datetime.now(timezone.utc).isoformat(),
        source_family=str(source_models.iloc[0]["source_family"]),
        model=str(source_models.iloc[0]["model"]),
    )


def _read_table(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for column in DATE_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def save_core_artifact(
    core: CoreForecastResult,
    request: ForecastRequest,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    directory = artifact_directory(request, core.artifact_id, artifact_root)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"不完整的 artifact 已存在：{directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for field, filename in CORE_TABLES.items():
        getattr(core, field).to_csv(directory / filename, index=False, encoding="utf-8-sig")
    _, identity = _core_identity(request)
    manifest = {
        "schema_version": LLM_SCHEMA_VERSION,
        "artifact_id": core.artifact_id,
        "generated_at": core.generated_at,
        "request": {
            "stock_id": int(request.stock_id),
            "as_of_date": request.cutoff.date().isoformat(),
            "data_dir": str(request.resolved_data_dir),
        },
        "identity": identity,
        "tables": CORE_TABLES,
        "notes": core.notes,
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return directory


def load_core_artifact(directory: str | Path) -> CoreForecastResult:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8-sig"))
    tables = manifest.get("tables", CORE_TABLES)
    frames = {field: _read_table(path / filename) for field, filename in tables.items()}
    return CoreForecastResult(
        **frames,
        notes=list(manifest.get("notes", [])),
        artifact_id=str(manifest["artifact_id"]),
        generated_at=str(manifest["generated_at"]),
        cache_hit=True,
    )


def load_or_build_forecast(
    request: ForecastRequest,
    *,
    artifact_root: str | Path | None = None,
) -> CoreForecastResult:
    artifact_id, _ = _core_identity(request)
    directory = artifact_directory(request, artifact_id, artifact_root)
    if (directory / "manifest.json").is_file():
        return load_core_artifact(directory)
    core = build_forecast_core(request)
    save_core_artifact(core, request, artifact_root=artifact_root)
    return core


def load_latest_price(
    request: ForecastRequest,
) -> tuple[float, pd.Timestamp] | None:
    prices = load_available_prices(
        request.resolved_data_dir,
        stock_ids={int(request.stock_id)},
        as_of_date=request.cutoff,
    )
    prices = prices[prices["stock_id"].eq(int(request.stock_id))].sort_values("date")
    if prices.empty:
        return None
    row = prices.iloc[-1]
    return float(row["close"]), pd.Timestamp(row["date"]).normalize()


def _live_from_core(core: CoreForecastResult, summary: pd.DataFrame, data_status: pd.DataFrame,
                    notes: list[str], price_source: str | None) -> LiveForecastResult:
    return LiveForecastResult(
        monthly=core.monthly.copy(), summary=summary, quarterly_eps=core.quarterly_eps.copy(),
        payout_history=core.payout_history.copy(), data_status=data_status,
        order_search=core.order_search.copy(), notes=notes, artifact_id=core.artifact_id,
        generated_at=core.generated_at, cache_hit=core.cache_hit, price_source=price_source,
    )


def apply_price_scenario(
    core: CoreForecastResult,
    *,
    stock_price: float,
    price_date: str | pd.Timestamp,
    price_source: str,
) -> LiveForecastResult:
    """Calculate yields from a stored core artifact without rerunning its models."""

    price = float(stock_price)
    date = pd.Timestamp(price_date).normalize()
    cutoff_values = pd.to_datetime(core.summary.get("as_of_date"), errors="coerce").dropna()
    if cutoff_values.empty:
        raise ValueError("core artifact is missing as_of_date")
    cutoff = cutoff_values.max().normalize()
    if not np.isfinite(price) or price <= 0:
        raise ValueError("股價必須是大於零的有限數值")
    if date > cutoff:
        raise ValueError("股價日期不得晚於預測基準日")
    if price_source not in {"observed_csv", "manual_scenario"}:
        raise ValueError("price_source must be observed_csv or manual_scenario")
    yields = calculate_as_of_yields(
        core.summary, stock_price=price, price_date=date,
        price_source=price_source, min_stock_price=0.0,
    )
    summary = core.summary.copy().reset_index(drop=True)
    summary["as_of_stock_price"] = yields["stock_price"].to_numpy()
    summary["as_of_price_date"] = pd.to_datetime(yields["price_date"]).to_numpy()
    summary["as_of_price_source"] = yields["price_source"].to_numpy()
    summary["as_of_price_yield_percent"] = yields["estimated_yield_percent"].to_numpy()
    status = core.data_status[core.data_status["dataset"].ne("daily_prices")].copy() if not core.data_status.empty else pd.DataFrame()
    status = pd.concat([status, pd.DataFrame([{
        "dataset": "daily_prices", "source": price_source,
        "latest_period": date, "latest_available_date": date, "rows": 1,
        "status": "手動情境價格" if price_source == "manual_scenario" else "可用",
    }])], ignore_index=True)
    notes = list(core.notes)
    notes.append(
        "本次殖利率使用手動價格情境，不視為市場觀測值。"
        if price_source == "manual_scenario"
        else "殖利率使用基準日以前 CSV 最新可得收盤價。"
    )
    return _live_from_core(core, summary, status, list(dict.fromkeys(notes)), price_source)


def without_price_scenario(core: CoreForecastResult) -> LiveForecastResult:
    summary = core.summary.copy()
    for column in ["as_of_stock_price", "as_of_price_yield_percent"]:
        summary[column] = np.nan
    summary["as_of_price_date"] = pd.NaT
    summary["as_of_price_source"] = "unavailable"
    if "status" in summary:
        summary.loc[summary["status"].eq("ok"), "status"] = "price unavailable"
    status = pd.concat([core.data_status, pd.DataFrame([{
        "dataset": "daily_prices", "source": "observed_csv", "latest_period": pd.NaT,
        "latest_available_date": pd.NaT, "rows": 0, "status": "無可用資料",
    }])], ignore_index=True)
    return _live_from_core(core, summary, status, [*core.notes, "daily_prices: 無可用股價"], None)


def build_live_forecast(
    selected_stock: int,
    as_of_date: str | pd.Timestamp,
    data_dir: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> LiveForecastResult:
    """Compatibility facade: load/build the core artifact, then apply the latest CSV price."""

    request = ForecastRequest(selected_stock, as_of_date, data_dir)
    core = load_or_build_forecast(request, artifact_root=artifact_root)
    try:
        latest = load_latest_price(request)
    except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError):
        latest = None
    if latest is None:
        return without_price_scenario(core)
    return apply_price_scenario(
        core, stock_price=latest[0], price_date=latest[1], price_source="observed_csv"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return _json_safe(frame.to_dict(orient="records"))


def build_llm_payload(result: LiveForecastResult, *, stock_name: str | None = None) -> dict[str, object]:
    summary = result.summary
    stock_id = int(result.monthly["stock_id"].dropna().iloc[0])
    cutoff = pd.to_datetime(summary["as_of_date"], errors="coerce").max()
    source_family = (
        str(summary["source_family"].dropna().iloc[0])
        if "source_family" in summary and summary["source_family"].notna().any()
        else "hybrid"
    )
    model_name = (
        str(summary["model"].dropna().iloc[0])
        if "model" in summary and summary["model"].notna().any()
        else "SARIMA＋營收公式"
    )
    internal_hybrid = "hybrid_method" in result.monthly.columns
    selected_mask = (
        result.order_search["selected"].astype(str).str.lower().isin({"true", "1"})
        if "selected" in result.order_search else pd.Series(False, index=result.order_search.index)
    )
    selected = result.order_search[selected_mask]
    price_row = summary.dropna(subset=["as_of_stock_price"]).iloc[0] if "as_of_stock_price" in summary and summary["as_of_stock_price"].notna().any() else pd.Series(dtype=object)
    payload = {
        "schema_version": LLM_SCHEMA_VERSION,
        "artifact_id": result.artifact_id,
        "generated_at": result.generated_at,
        "as_of_date": cutoff.date().isoformat() if pd.notna(cutoff) else None,
        "stock": {"stock_id": stock_id, "stock_name": stock_name},
        "model": {
            "source_family": source_family,
            "name": model_name,
            "sarima_weight": SARIMA_WEIGHT if internal_hybrid else None,
            "formula_weight": 1.0 - SARIMA_WEIGHT if internal_hybrid else None,
            "formula_config": FORMULA_CONFIG.as_dict() if internal_hybrid else None,
            "selected_sarima": _records(selected) if internal_hybrid else [],
            "fallback_counts": (
                _json_safe(result.monthly["hybrid_method"].value_counts().to_dict())
                if internal_hybrid else {}
            ),
            "input_contract": "internal_hybrid" if internal_hybrid else "external_monthly_revenue_predictions",
        },
        "formulas": {
            "revenue_formula": (
                "近三個月 YoY log 成長率中位數，套用去年同月後，再與上月營收於 log1p 空間各取 50%"
                if internal_hybrid else "月營收預測由外部模型提供；請依 source_family 與 model 追溯其公式"
            ),
            "hybrid": (
                "0.1 × SARIMA + 0.9 × 營收公式；模型失效時使用有效單一來源"
                if internal_hybrid else None
            ),
            "after_tax_eps": "已公布單季稅後 EPS + 未公布季營收 × 歷史同季 EPS/營收比率中位數",
            "cash_dividend": "固定配息用五年股利中位數；明確零配息用 0；其餘為預估 EPS × 歷史有效平均配息率",
            "cash_yield": "預估每股現金股利 ÷ 指定股價 × 100%",
        },
        "price_scenario": {
            "stock_price": _json_safe(price_row.get("as_of_stock_price")),
            "price_date": _json_safe(price_row.get("as_of_price_date")),
            "source": _json_safe(price_row.get("as_of_price_source", result.price_source)),
        },
        "annual_results": _records(summary),
        "data_cutoffs": _records(result.data_status),
        "warnings": result.notes,
        "limitations": [
            "研究原型，不是投資建議或股價預測。",
            "未建模稅後淨利與加權平均股數。",
            "股利分類為歷史啟發式，非公司未來承諾。",
            "若使用手動價格，該值只是情境輸入，不是觀測市價。",
        ],
        "detail_files": CORE_TABLES,
    }
    return _json_safe(payload)


def llm_payload_json(result: LiveForecastResult, *, stock_name: str | None = None) -> str:
    return json.dumps(
        build_llm_payload(result, stock_name=stock_name),
        ensure_ascii=False, indent=2, allow_nan=False,
    )


def build_llm_zip(result: LiveForecastResult, *, stock_name: str | None = None) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("forecast.json", llm_payload_json(result, stock_name=stock_name))
        for field, filename in CORE_TABLES.items():
            archive.writestr(filename, getattr(result, field).to_csv(index=False).encode("utf-8-sig"))
    return output.getvalue()


def write_llm_bundle(
    result: LiveForecastResult,
    output_root: str | Path,
    *,
    stock_name: str | None = None,
) -> Path:
    price = result.summary.get("as_of_stock_price", pd.Series([np.nan])).iloc[0]
    price_date = result.summary.get("as_of_price_date", pd.Series([pd.NaT])).iloc[0]
    scenario = json.dumps(_json_safe({"price": price, "date": price_date, "source": result.price_source}), sort_keys=True)
    scenario_id = hashlib.sha256(scenario.encode("utf-8")).hexdigest()[:10]
    stock_id = int(result.monthly["stock_id"].dropna().iloc[0])
    destination = Path(output_root) / f"{stock_id}_{result.artifact_id}_{scenario_id}"
    manifest_path = destination / "forecast.json"
    if manifest_path.is_file():
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"不完整的匯出目錄已存在：{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(llm_payload_json(result, stock_name=stock_name), encoding="utf-8")
    for field, filename in CORE_TABLES.items():
        getattr(result, field).to_csv(destination / filename, index=False, encoding="utf-8-sig")
    return destination
