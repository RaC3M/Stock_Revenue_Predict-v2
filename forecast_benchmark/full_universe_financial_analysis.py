from __future__ import annotations

"""Evaluate representative full-universe forecasts through the shared financial layer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from financial_forecast import FinancialForecastPolicy, forecast_financials


DEFAULT_MODELS = (
    "Last observed revenue",
    "SeasonalQuantile",
    "LightGBM",
    "ensemble_revenue",
    "Rolling xLSTM + Conditional Adjustment",
)


def parse_models(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_actual_eps(data_dir: Path, target_year: int) -> pd.DataFrame:
    eps = pd.read_csv(data_dir / "EPS2020~2025.csv")
    eps["stock_id"] = pd.to_numeric(eps["stock_id"], errors="coerce")
    eps["date"] = pd.to_datetime(eps["date"], errors="coerce")
    eps["EPS"] = pd.to_numeric(eps["EPS"], errors="coerce")
    eps = eps.dropna(subset=["stock_id", "date", "EPS"])
    eps["stock_id"] = eps["stock_id"].astype(int)
    eps = eps[eps["date"].dt.year.eq(int(target_year))].copy()
    eps["quarter"] = eps["date"].dt.quarter
    annual = eps.groupby("stock_id", as_index=False).agg(
        actual_annual_eps=("EPS", "sum"),
        actual_eps_quarter_count=("quarter", "nunique"),
    )
    annual.loc[annual["actual_eps_quarter_count"].lt(4), "actual_annual_eps"] = np.nan
    return annual


def load_dividend_actuals(data_dir: Path, target_year: int) -> pd.DataFrame:
    dividends = pd.read_csv(data_dir / "Dividend2019~2025.csv")
    dividends["stock_id"] = pd.to_numeric(dividends["stock_id"], errors="coerce")
    dividends["fiscal_year"] = pd.to_numeric(dividends["fiscal_year"], errors="coerce")
    dividends["cash_dividend"] = pd.to_numeric(dividends["TotalCashDividend"], errors="coerce")
    dividends["ex_dividend_date"] = pd.to_datetime(
        dividends["CashExDividendTradingDate"],
        errors="coerce",
    )
    dividends = dividends.dropna(subset=["stock_id", "cash_dividend"])
    dividends["stock_id"] = dividends["stock_id"].astype(int)
    ex_year = dividends["ex_dividend_date"].dt.year
    by_ex_year = (
        dividends[ex_year.eq(int(target_year))]
        .groupby("stock_id", as_index=False)
        .agg(actual_cash_dividend_ex_year=("cash_dividend", "sum"))
    )
    by_fiscal_year = (
        dividends[dividends["fiscal_year"].eq(int(target_year))]
        .groupby("stock_id", as_index=False)
        .agg(actual_cash_dividend_fiscal_year=("cash_dividend", "sum"))
    )
    return by_ex_year.merge(by_fiscal_year, on="stock_id", how="outer")


def build_stock_results(
    predictions: pd.DataFrame,
    financial_summary: pd.DataFrame,
    actual_eps: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    actual_revenue = (
        predictions.drop_duplicates(["stock_id", "target_month"])
        .groupby("stock_id", as_index=False)
        .agg(actual_annual_revenue=("actual_revenue", "sum"))
    )
    labels = predictions[["stock_id", "stock_name", "industry_category"]].drop_duplicates("stock_id")
    results = financial_summary.merge(actual_revenue, on="stock_id", how="left")
    results = results.merge(labels, on="stock_id", how="left", suffixes=("", "_input"))
    results = results.merge(actual_eps, on="stock_id", how="left")
    results = results.merge(dividends, on="stock_id", how="left")
    results["annual_revenue_abs_percent_error"] = np.where(
        results["actual_annual_revenue"].abs().gt(0),
        (results["predicted_annual_revenue"] - results["actual_annual_revenue"]).abs()
        / results["actual_annual_revenue"].abs()
        * 100,
        np.nan,
    )
    results["eps_error"] = results["estimated_eps"] - results["actual_annual_eps"]
    results["eps_abs_error"] = results["eps_error"].abs()
    results["eps_abs_percent_error"] = np.where(
        results["actual_annual_eps"].abs().ge(0.01),
        results["eps_abs_error"] / results["actual_annual_eps"].abs() * 100,
        np.nan,
    )
    results["dividend_error_ex_year"] = (
        results["estimated_cash_dividend"] - results["actual_cash_dividend_ex_year"]
    )
    results["dividend_abs_error_ex_year"] = results["dividend_error_ex_year"].abs()
    results["dividend_error_fiscal_year"] = (
        results["estimated_cash_dividend"] - results["actual_cash_dividend_fiscal_year"]
    )
    results["dividend_abs_error_fiscal_year"] = results["dividend_error_fiscal_year"].abs()
    valid_price = pd.to_numeric(results["as_of_stock_price"], errors="coerce").gt(1.0)
    results["actual_as_of_yield_ex_year_percent"] = np.where(
        valid_price & results["actual_cash_dividend_ex_year"].notna(),
        results["actual_cash_dividend_ex_year"] / results["as_of_stock_price"] * 100,
        np.nan,
    )
    results["actual_as_of_yield_fiscal_year_percent"] = np.where(
        valid_price & results["actual_cash_dividend_fiscal_year"].notna(),
        results["actual_cash_dividend_fiscal_year"] / results["as_of_stock_price"] * 100,
        np.nan,
    )
    results["yield_error_ex_year_percent_point"] = (
        results["as_of_price_yield_percent"] - results["actual_as_of_yield_ex_year_percent"]
    )
    results["yield_abs_error_ex_year_percent_point"] = results[
        "yield_error_ex_year_percent_point"
    ].abs()
    results["yield_error_fiscal_year_percent_point"] = (
        results["as_of_price_yield_percent"] - results["actual_as_of_yield_fiscal_year_percent"]
    )
    results["yield_abs_error_fiscal_year_percent_point"] = results[
        "yield_error_fiscal_year_percent_point"
    ].abs()
    return results


def build_overall(stock_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source_family, model), group in stock_results.groupby(["source_family", "model"]):
        row: dict[str, object] = {
            "source_family": source_family,
            "model": model,
            "stock_count": int(group["stock_id"].nunique()),
        }
        metrics = {
            "annual_revenue_abs_percent_error": "revenue",
            "eps_abs_error": "eps",
            "eps_abs_percent_error": "eps_percent",
            "dividend_abs_error_ex_year": "dividend_ex_year",
            "dividend_abs_error_fiscal_year": "dividend_fiscal_year",
            "yield_abs_error_ex_year_percent_point": "yield_ex_year",
            "yield_abs_error_fiscal_year_percent_point": "yield_fiscal_year",
        }
        for column, label in metrics.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"valid_{label}_stock_count"] = int(values.shape[0])
            row[f"average_{label}_error"] = float(values.mean()) if not values.empty else np.nan
            row[f"median_{label}_error"] = float(values.median()) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("average_yield_fiscal_year_error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = Path(args.data_dir).resolve()
    models = parse_models(args.models)
    predictions = pd.read_csv(args.input_predictions, low_memory=False)
    predictions = predictions[
        predictions["target_year"].eq(int(args.target_year))
        & predictions["model"].isin(models)
    ].copy()
    financial = forecast_financials(
        predictions,
        target_year=int(args.target_year),
        as_of_date=f"{int(args.target_year)}-01-10",
        data_dir=data_dir,
        policy=FinancialForecastPolicy(
            eps_methods=("current_ratio",),
            dividend_methods=("announcement_safe_payout_ratio",),
            yield_modes=("as_of_price_yield", "target_month_end_yield"),
        ),
    )
    actual_eps = load_actual_eps(data_dir, int(args.target_year))
    dividend_actuals = load_dividend_actuals(data_dir, int(args.target_year))
    stock_results = build_stock_results(
        predictions,
        financial.summary,
        actual_eps,
        dividend_actuals,
    )
    overall = build_overall(stock_results)
    alignment_audit = pd.DataFrame(
        [
            {
                "target_year": int(args.target_year),
                "ex_dividend_year_stock_count": int(dividend_actuals["actual_cash_dividend_ex_year"].notna().sum()),
                "fiscal_year_stock_count": int(dividend_actuals["actual_cash_dividend_fiscal_year"].notna().sum()),
                "current_ui_definition": "target-year ex-dividend cash dividend",
                "fiscal_aligned_definition": "cash dividend generated by target fiscal year",
            }
        ]
    )

    financial.eps_estimates.to_csv(output_dir / "eps_estimates.csv", index=False, encoding="utf-8-sig")
    financial.dividend_estimates.to_csv(output_dir / "dividend_estimates.csv", index=False, encoding="utf-8-sig")
    financial.yield_estimates.to_csv(output_dir / "yield_estimates.csv", index=False, encoding="utf-8-sig")
    financial.summary.to_csv(output_dir / "financial_summary.csv", index=False, encoding="utf-8-sig")
    financial.failures.to_csv(output_dir / "financial_failures.csv", index=False, encoding="utf-8-sig")
    stock_results.to_csv(output_dir / "financial_stock_results.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "financial_overall_accuracy.csv", index=False, encoding="utf-8-sig")
    alignment_audit.to_csv(output_dir / "dividend_alignment_audit.csv", index=False, encoding="utf-8-sig")
    config = {
        "target_year": int(args.target_year),
        "as_of_date": f"{int(args.target_year)}-01-10",
        "models": list(models),
        "stock_count": int(predictions["stock_id"].nunique()),
        "eps_method": "current_ratio",
        "dividend_method": "announcement_safe_payout_ratio",
        "yield_denominator": "latest observed close at or before as_of_date",
        "warning": (
            "The current UI ex-dividend-year comparison is timing-misaligned with target-year "
            "revenue. Fiscal-year-aligned columns are the preferred transformation diagnostic."
        ),
        "evidence_tier": "C",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
