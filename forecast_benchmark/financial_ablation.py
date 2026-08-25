"""Select downstream financial methods on historical years, then test frozen forecasts.

The historical validation rows replay observed monthly revenue.  They isolate EPS/dividend
transformation quality and must not be described as validation of the upstream revenue model.
The target-year rows come from an existing frozen prediction file; this runner never trains a
revenue model or neural network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from financial_forecast import FinancialForecastPolicy, forecast_financials
from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)


DEFAULT_INPUT_PREDICTIONS = (
    PROJECT_ROOT
    / "forecast_benchmark"
    / "outputs"
    / "data_migration_revenue_20260730"
    / "comparable_monthly_predictions.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "financial_ablation"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_VALIDATION_YEARS = (2022, 2023, 2024)
DEFAULT_EPS_METHODS = ("current_ratio", "seasonal_quarter_median")
DEFAULT_DIVIDEND_METHODS = (
    "announcement_safe_payout_ratio",
    "announcement_safe_last_cash_dividend",
    "announcement_safe_cash_dividend_median",
    "announcement_safe_smoothed_cash_dividend",
)

PREDICTION_COLUMNS = (
    "source_family",
    "model",
    "stock_id",
    "target_year",
    "target_month",
    "predicted_revenue",
)


def run_financial_ablation(
    frozen_predictions: pd.DataFrame,
    *,
    data_dir: str | Path,
    target_year: int = DEFAULT_TARGET_YEAR,
    validation_years: tuple[int, ...] = DEFAULT_VALIDATION_YEARS,
    eps_methods: tuple[str, ...] = DEFAULT_EPS_METHODS,
    dividend_methods: tuple[str, ...] = DEFAULT_DIVIDEND_METHODS,
    as_of_month: int = 1,
    as_of_day: int = 10,
) -> dict[str, pd.DataFrame]:
    """Run EPS, dividend, and end-to-end stages without retraining revenue models."""

    predictions = _normalize_frozen_predictions(frozen_predictions, int(target_year))
    stock_ids = sorted(int(value) for value in predictions["stock_id"].unique())
    policy = FinancialForecastPolicy(
        eps_methods=tuple(eps_methods),
        dividend_methods=tuple(dividend_methods),
        yield_modes=("as_of_price_yield", "target_month_end_yield"),
    )

    validation_eps_parts: list[pd.DataFrame] = []
    validation_dividend_parts: list[pd.DataFrame] = []
    validation_yield_parts: list[pd.DataFrame] = []
    failure_parts: list[pd.DataFrame] = []
    for year in sorted(set(int(value) for value in validation_years)):
        if year >= int(target_year):
            raise ValueError("Validation years must be earlier than the target test year.")
        replay = _build_actual_revenue_replay(data_dir, stock_ids, year)
        result = forecast_financials(
            replay,
            target_year=year,
            as_of_date=pd.Timestamp(year, as_of_month, as_of_day),
            data_dir=data_dir,
            policy=policy,
        )
        actual_eps = _load_actual_eps(data_dir, year, stock_ids)
        validation_eps_parts.append(
            _decorate_eps(result.eps_estimates, actual_eps, split="validation")
        )
        validation_dividend_parts.append(
            _decorate_dividends(result.dividend_estimates, split="validation")
        )
        validation_yield_parts.append(
            _decorate_yields(result.yield_estimates, split="validation")
        )
        if not result.failures.empty:
            failure = result.failures.copy()
            failure["split"] = "validation"
            failure["validation_source"] = "actual_revenue_replay"
            failure_parts.append(failure)

    validation_eps = _concat(validation_eps_parts)
    validation_dividends = _concat(validation_dividend_parts)
    validation_yields = _concat(validation_yield_parts)
    for frame in (validation_eps, validation_dividends, validation_yields):
        if not frame.empty:
            frame["validation_source"] = "actual_revenue_replay"

    validation_eps_scores = _score_eps(validation_eps)
    validation_dividend_scores = _score_dividends(validation_dividends)
    validation_end_to_end_scores = _score_end_to_end(
        validation_dividends,
        validation_yields,
    )
    method_selection = _select_methods(
        validation_end_to_end_scores,
        validation_years=tuple(sorted(set(int(value) for value in validation_years))),
    )

    test_result = forecast_financials(
        predictions,
        target_year=int(target_year),
        as_of_date=pd.Timestamp(target_year, as_of_month, as_of_day),
        data_dir=data_dir,
        policy=policy,
    )
    test_actual_eps = _load_actual_eps(data_dir, int(target_year), stock_ids)
    test_eps = _decorate_eps(test_result.eps_estimates, test_actual_eps, split="test")
    test_dividends = _decorate_dividends(test_result.dividend_estimates, split="test")
    test_yields = _decorate_yields(test_result.yield_estimates, split="test")
    selected_test = _filter_selected_test(test_yields, method_selection)
    test_scores = _score_end_to_end(
        _filter_selected_test(test_dividends, method_selection),
        selected_test,
    )
    if not test_result.failures.empty:
        failure = test_result.failures.copy()
        failure["split"] = "test"
        failure_parts.append(failure)

    return {
        "validation_eps_estimates": validation_eps,
        "validation_eps_scores": validation_eps_scores,
        "validation_dividend_estimates": validation_dividends,
        "validation_dividend_scores": validation_dividend_scores,
        "validation_yield_estimates": validation_yields,
        "validation_end_to_end_scores": validation_end_to_end_scores,
        "method_selection": method_selection,
        "test_eps_estimates": test_eps,
        "test_dividend_estimates": test_dividends,
        "test_yield_estimates": test_yields,
        "selected_test_estimates": selected_test,
        "selected_test_scores": test_scores,
        "failures": _concat(failure_parts),
    }


def _normalize_frozen_predictions(predictions: pd.DataFrame, target_year: int) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Frozen prediction input missing columns: {sorted(missing)}")
    frame = predictions.loc[:, PREDICTION_COLUMNS].copy()
    for column in ("stock_id", "target_year", "target_month", "predicted_revenue"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(PREDICTION_COLUMNS))
    frame["stock_id"] = frame["stock_id"].astype(int)
    frame["target_year"] = frame["target_year"].astype(int)
    frame["target_month"] = frame["target_month"].astype(int)
    frame = frame[frame["target_year"].eq(int(target_year))]
    if frame.empty:
        raise ValueError(f"No frozen predictions found for target year {target_year}.")
    return frame.sort_values(["stock_id", "source_family", "model", "target_month"])


def _build_actual_revenue_replay(
    data_dir: str | Path,
    stock_ids: list[int],
    target_year: int,
) -> pd.DataFrame:
    revenue = pd.read_csv(
        Path(data_dir) / "Stock_revenue_2019~2025.csv",
        usecols=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"],
    )
    for column in ("stock_id", "revenue_year", "revenue_month", "revenue_thousand"):
        revenue[column] = pd.to_numeric(revenue[column], errors="coerce")
    revenue = revenue.dropna()
    revenue["stock_id"] = revenue["stock_id"].astype(int)
    revenue["revenue_year"] = revenue["revenue_year"].astype(int)
    revenue["revenue_month"] = revenue["revenue_month"].astype(int)
    revenue = revenue[
        revenue["stock_id"].isin(stock_ids) & revenue["revenue_year"].eq(int(target_year))
    ]
    revenue = revenue.groupby(
        ["stock_id", "revenue_year", "revenue_month"], as_index=False
    )["revenue_thousand"].sum()
    complete = revenue.groupby("stock_id")["revenue_month"].nunique()
    revenue = revenue[revenue["stock_id"].isin(complete[complete.eq(12)].index)].copy()
    return pd.DataFrame(
        {
            "source_family": "historical_validation",
            "model": "actual_revenue_replay",
            "stock_id": revenue["stock_id"],
            "target_year": int(target_year),
            "target_month": revenue["revenue_month"],
            "predicted_revenue": revenue["revenue_thousand"],
        }
    )


def _load_actual_eps(
    data_dir: str | Path,
    target_year: int,
    stock_ids: list[int],
) -> pd.DataFrame:
    eps = pd.read_csv(Path(data_dir) / "EPS2020~2025.csv", usecols=["stock_id", "date", "EPS"])
    eps["stock_id"] = pd.to_numeric(eps["stock_id"], errors="coerce")
    eps["date"] = pd.to_datetime(eps["date"], errors="coerce")
    eps["EPS"] = pd.to_numeric(eps["EPS"], errors="coerce")
    eps = eps.dropna(subset=["stock_id", "date", "EPS"])
    eps["stock_id"] = eps["stock_id"].astype(int)
    eps = eps[eps["stock_id"].isin(stock_ids) & eps["date"].dt.year.eq(int(target_year))].copy()
    eps["quarter"] = eps["date"].dt.quarter.astype(int)
    quarterly = eps.groupby(["stock_id", "quarter"], as_index=False)["EPS"].sum()
    annual = quarterly.groupby("stock_id", as_index=False).agg(
        actual_annual_eps=("EPS", "sum"),
        actual_eps_quarter_count=("quarter", "nunique"),
    )
    return annual[annual["actual_eps_quarter_count"].eq(4)].reset_index(drop=True)


def _decorate_eps(estimates: pd.DataFrame, actual: pd.DataFrame, *, split: str) -> pd.DataFrame:
    if estimates.empty:
        return estimates.copy()
    result = estimates.merge(actual, on="stock_id", how="left")
    result["eps_error"] = result["estimated_eps"] - result["actual_annual_eps"]
    result["eps_abs_error"] = result["eps_error"].abs()
    result["split"] = split
    return result


def _decorate_dividends(estimates: pd.DataFrame, *, split: str) -> pd.DataFrame:
    if estimates.empty:
        return estimates.copy()
    result = estimates.copy()
    result["cash_dividend_error"] = (
        result["estimated_cash_dividend"] - result["actual_cash_dividend"]
    )
    result["cash_dividend_abs_error"] = result["cash_dividend_error"].abs()
    result["split"] = split
    return result


def _decorate_yields(estimates: pd.DataFrame, *, split: str) -> pd.DataFrame:
    if estimates.empty:
        return estimates.copy()
    result = estimates.copy()
    result["cash_dividend_error"] = (
        result["estimated_cash_dividend"] - result["actual_cash_dividend"]
    )
    result["cash_dividend_abs_error"] = result["cash_dividend_error"].abs()
    result["yield_abs_error_percent_point"] = result["yield_error_percent_point"].abs()
    result["split"] = split
    return result


def _score_eps(estimates: pd.DataFrame) -> pd.DataFrame:
    if estimates.empty:
        return pd.DataFrame(columns=["eps_method", "eps_observations", "eps_mae"])
    comparable = _exact_method_cohort(
        estimates,
        identity_columns=["source_family", "model", "stock_id", "target_year"],
        method_columns=["eps_method"],
        required_value="eps_abs_error",
    )
    return comparable.groupby("eps_method", as_index=False).agg(
        eps_observations=("eps_abs_error", "count"),
        eps_mae=("eps_abs_error", "mean"),
    )


def _score_dividends(estimates: pd.DataFrame) -> pd.DataFrame:
    columns = ["eps_method", "dividend_method"]
    if estimates.empty:
        return pd.DataFrame(columns=columns + ["cash_dividend_observations", "cash_dividend_mae"])
    comparable = _exact_method_cohort(
        estimates,
        identity_columns=["source_family", "model", "stock_id", "target_year"],
        method_columns=columns,
        required_value="cash_dividend_abs_error",
    )
    return comparable.groupby(columns, as_index=False).agg(
        cash_dividend_observations=("cash_dividend_abs_error", "count"),
        cash_dividend_mae=("cash_dividend_abs_error", "mean"),
    )


def _score_end_to_end(dividends: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    dividend_scores = _score_dividends(dividends)
    key = ["eps_method", "dividend_method"]
    evaluation = (
        yields[yields["yield_mode"].eq("target_month_end_yield")]
        if not yields.empty
        else yields
    )
    if evaluation.empty:
        yield_scores = pd.DataFrame(
            columns=key + ["yield_observations", "yield_mae_percent_point"]
        )
    else:
        comparable_yields = _exact_method_cohort(
            evaluation,
            identity_columns=[
                "source_family",
                "model",
                "stock_id",
                "target_year",
                "target_month",
            ],
            method_columns=key,
            required_value="yield_abs_error_percent_point",
        )
        yield_scores = comparable_yields.groupby(key, as_index=False).agg(
            yield_observations=("yield_abs_error_percent_point", "count"),
            yield_mae_percent_point=("yield_abs_error_percent_point", "mean"),
        )
    return dividend_scores.merge(yield_scores, on=key, how="outer")


def _exact_method_cohort(
    frame: pd.DataFrame,
    *,
    identity_columns: list[str],
    method_columns: list[str],
    required_value: str,
) -> pd.DataFrame:
    """Keep only identities with a valid observation for every requested method."""

    if frame.empty:
        return frame.copy()
    expected_method_count = len(frame[method_columns].drop_duplicates())
    valid = frame.dropna(subset=[required_value]).copy()
    if expected_method_count == 0 or valid.empty:
        return valid.iloc[0:0].copy()
    membership = valid[identity_columns + method_columns].drop_duplicates()
    counts = membership.groupby(identity_columns, dropna=False).size().rename("method_count")
    comparable_keys = counts[counts.eq(expected_method_count)].reset_index()[identity_columns]
    return valid.merge(comparable_keys, on=identity_columns, how="inner", validate="many_to_one")


def _select_methods(
    scores: pd.DataFrame,
    *,
    validation_years: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        "selected_eps_method",
        "selected_dividend_method",
        "selection_min_year",
        "selection_max_year",
        "selection_primary_metric",
        "selection_secondary_metric",
    ]
    candidates = scores.dropna(subset=["cash_dividend_mae"]).copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates["_yield_sort"] = candidates["yield_mae_percent_point"].fillna(np.inf)
    winner = candidates.sort_values(
        ["cash_dividend_mae", "_yield_sort", "eps_method", "dividend_method"]
    ).iloc[0]
    return pd.DataFrame(
        [
            {
                "selected_eps_method": winner["eps_method"],
                "selected_dividend_method": winner["dividend_method"],
                "selection_min_year": min(validation_years),
                "selection_max_year": max(validation_years),
                "selection_primary_metric": "cash_dividend_mae",
                "selection_secondary_metric": "yield_mae_percent_point",
                "validation_cash_dividend_mae": winner["cash_dividend_mae"],
                "validation_yield_mae_percent_point": winner["yield_mae_percent_point"],
            }
        ]
    )


def _filter_selected_test(frame: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or selection.empty:
        return frame.iloc[0:0].copy()
    winner = selection.iloc[0]
    return frame[
        frame["eps_method"].eq(winner["selected_eps_method"])
        & frame["dividend_method"].eq(winner["selected_dividend_method"])
    ].reset_index(drop=True)


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [part for part in parts if not part.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def _parse_csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def _parse_years(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _parse_csv(value))


def load_frozen_predictions(
    path: str | Path,
    *,
    target_year: int,
    models: tuple[str, ...] = (),
    stock_ids: tuple[int, ...] = (),
    stock_limit: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _normalize_frozen_predictions(frame, target_year)
    if models:
        frame = frame[frame["model"].isin(models)]
    if stock_ids:
        frame = frame[frame["stock_id"].isin(stock_ids)]
    selected = sorted(int(value) for value in frame["stock_id"].unique())
    if stock_limit is not None:
        selected = selected[: int(stock_limit)]
    return frame[frame["stock_id"].isin(selected)].reset_index(drop=True)


def write_outputs(
    output_dir: str | Path,
    results: dict[str, pd.DataFrame],
    run_config: dict[str, object],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(root / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_run_config_and_registry(root, run_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions", type=Path, default=DEFAULT_INPUT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument(
        "--validation-years",
        default=",".join(str(year) for year in DEFAULT_VALIDATION_YEARS),
    )
    parser.add_argument("--eps-methods", default=",".join(DEFAULT_EPS_METHODS))
    parser.add_argument("--dividend-methods", default=",".join(DEFAULT_DIVIDEND_METHODS))
    parser.add_argument("--models", help="Optional comma-separated frozen revenue models.")
    parser.add_argument("--stock-ids", help="Optional comma-separated stock IDs.")
    parser.add_argument("--stock-limit", type=int, help="Limit stocks for a smoke run.")
    add_registry_arguments(parser)
    parser.set_defaults(
        selection_protocol="historical-validation",
        report_ready=False,
        evidence_tier="B",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validation_years = _parse_years(args.validation_years)
    eps_methods = _parse_csv(args.eps_methods)
    dividend_methods = _parse_csv(args.dividend_methods)
    predictions = load_frozen_predictions(
        args.input_predictions,
        target_year=args.target_year,
        models=_parse_csv(args.models),
        stock_ids=tuple(int(value) for value in _parse_csv(args.stock_ids)),
        stock_limit=args.stock_limit,
    )
    results = run_financial_ablation(
        predictions,
        data_dir=args.data_dir,
        target_year=args.target_year,
        validation_years=validation_years,
        eps_methods=eps_methods,
        dividend_methods=dividend_methods,
    )
    run_config: dict[str, object] = {
        "input_predictions": str(args.input_predictions),
        "output_dir": str(args.output_dir),
        "data_dir": str(args.data_dir),
        "target_year": int(args.target_year),
        "validation_years": list(validation_years),
        "eps_methods": list(eps_methods),
        "dividend_methods": list(dividend_methods),
        "stock_count": int(predictions["stock_id"].nunique()),
        "frozen_prediction_rows": int(len(predictions)),
        "revenue_model_training_performed": False,
        "validation_source": "actual_revenue_replay",
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="financial_ablation",
        data_dir=args.data_dir,
        report_ready_reason=(
            "Downstream methods selected on historical actual-revenue replay; target-year "
            "revenue predictions remain frozen. Human upstream development audit still required."
        ),
        extra={"input_predictions": str(args.input_predictions)},
    )
    write_outputs(args.output_dir, results, run_config)
    print("Wrote financial ablation outputs to", args.output_dir)
    print(results["method_selection"].to_string(index=False))
    print(results["selected_test_scores"].to_string(index=False))


if __name__ == "__main__":
    main()
