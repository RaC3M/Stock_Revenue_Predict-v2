from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from financial_forecast import FinancialForecastPolicy, forecast_financials
from financial_forecast.contracts import EPS_METHOD_KNOWN_QUARTERS, DIVIDEND_METHOD_CLASSIFIED
from financial_forecast.evidence import resolve_data_files
from hybrid_forecast.hybrid_engine import HybridConfig, combine_predictions
from revenue_adjustment_formula.formula_engine import FormulaConfig, MIN_FORMULA_HISTORY, _formula_base
from sarima_forecast import sarima_engine


SARIMA_WEIGHT = 0.1
FORMULA_CONFIG = FormulaConfig(
    seasonal_weight=0.5, residual_alpha=0.1, residual_strength=0.0,
    growth_log_cap=float(np.log(2.0)), correction_log_cap=0.5,
)
LIVE_CACHE_VERSION = "hybrid_live_v2_dividend_patterns"


@dataclass
class LiveForecastResult:
    monthly: pd.DataFrame
    summary: pd.DataFrame
    quarterly_eps: pd.DataFrame
    payout_history: pd.DataFrame
    data_status: pd.DataFrame
    order_search: pd.DataFrame
    notes: list[str]


def data_fingerprint(data_dir: str | Path) -> tuple:
    root = Path(data_dir).expanduser().resolve()
    paths = [root / "manifest.json", *resolve_data_files(root).values()]
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file()
        else (str(path), None, None) for path in paths
    )


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


def build_live_forecast(
    selected_stock: int, as_of_date: str | pd.Timestamp, data_dir: str | Path,
) -> LiveForecastResult:
    cutoff = pd.Timestamp(as_of_date).normalize()
    revenue = load_revenue_snapshot(data_dir, cutoff)
    stock = revenue[revenue["stock_id"].eq(int(selected_stock))].copy()
    if stock.empty:
        raise ValueError(f"{selected_stock} 在 {cutoff.date()} 以前沒有可用營收")
    if stock.duplicated("date").any():
        raise ValueError(f"{selected_stock} 營收有重複月份，請先修正 CSV")
    latest = stock["date"].max()
    history = sarima_engine._trailing_consecutive_history(stock, latest + pd.offsets.MonthBegin(1))
    end = pd.Timestamp(cutoff.year + 1, 12, 1)
    dates = pd.date_range(latest + pd.offsets.MonthBegin(1), end, freq="MS")
    components, order_search = _forecast_components(history, dates)
    monthly = pd.DataFrame({"target_date": pd.date_range(f"{cutoff.year}-01-01", end, freq="MS")})
    monthly["stock_id"] = int(selected_stock)
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
    monthly["as_of_date"] = cutoff
    monthly["latest_actual_month"] = latest
    notes = [
        "SARIMA 10%＋營收公式 90%；沿用 2023–2024 驗證後的固定權重及公式參數。",
        "SARIMA 一次多步預測；公式以自身預測遞推，不使用未公布的實際營收。",
        "年份表示獲利所屬年度；EPS 為公司稅後 EPS，未另扣投資人所得稅或補充保費。",
        "各年度共用基準日以前的最新有效收盤價，價格日期以 CSV 為準。",
        "股利依五年歷史分類：固定金額用現金股利中位數；五年明確零股利用零；其餘依有效配息率估計並標記資料不足。",
        "分類非公司未來承諾；年度現金股利為已公告紀錄合計，來源沒有全年已公告完畢標記。",
        "EPS 仍依已公布季度及歷史 EPS／營收比率推估，尚無稅後淨利與加權平均股數資料可分別建模。",
    ]
    if revenue.attrs.get("invalid_rows", 0):
        notes.append(f"營收 CSV 有 {revenue.attrs['invalid_rows']} 筆無有效數值或公布日期的資料，未納入。")
    summaries, quarters, payouts, statuses = [], [], [], []
    policy = FinancialForecastPolicy(
        eps_methods=(EPS_METHOD_KNOWN_QUARTERS,), dividend_methods=(DIVIDEND_METHOD_CLASSIFIED,),
        yield_modes=("as_of_price_yield",), min_stock_price=0.0,
    )
    for year in [cutoff.year, cutoff.year + 1]:
        year_frame = monthly[monthly["target_year"].eq(year)]
        normalized = year_frame[["stock_id", "target_year", "target_month", "revenue_used"]].rename(
            columns={"revenue_used": "predicted_revenue"})
        normalized["source_family"], normalized["model"] = "hybrid", "SARIMA＋營收公式"
        row = {"target_year": year, "actual_months": int(year_frame["revenue_basis"].eq("actual").sum()),
            "forecast_months": int(year_frame["revenue_basis"].eq("forecast").sum()),
            "predicted_annual_revenue": year_frame["revenue_used"].sum(min_count=12),
            "as_of_date": cutoff, "latest_actual_month": latest}
        financial = forecast_financials(normalized, target_year=year, as_of_date=cutoff, data_dir=data_dir, policy=policy)
        if not financial.summary.empty:
            row.update(financial.summary.iloc[0].to_dict())
            if pd.isna(row.get("as_of_price_yield_percent")) and row.get("status") == "ok":
                row["status"] = "price unavailable"
        else:
            row["status"] = "全年營收不完整，無法估算全年 EPS／股利"
        summaries.append(row)
        quarters.append(financial.quarterly_eps_estimates)
        payouts.append(financial.payout_history)
        statuses.append(financial.data_status)
        notes.extend(issue for issue in financial.notes if any(issue.startswith(k + ":") for k in ["revenue", "eps", "dividends", "daily_prices"]))
    quarter_result = pd.concat([q for q in quarters if not q.empty], ignore_index=True) if any(not q.empty for q in quarters) else pd.DataFrame()
    payout_result = pd.concat([p for p in payouts if not p.empty], ignore_index=True).drop_duplicates(["stock_id", "fiscal_year"]) if any(not p.empty for p in payouts) else pd.DataFrame()
    data_status = pd.concat(statuses, ignore_index=True).drop_duplicates("dataset")
    return LiveForecastResult(monthly, pd.DataFrame(summaries), quarter_result, payout_result, data_status, order_search, list(dict.fromkeys(notes)))
