from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import YIELD_MODE_AS_OF_PRICE, YIELD_MODE_TARGET_MONTH_END


def calculate_yields(
    dividend_estimates: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    target_year: int,
    as_of_date: pd.Timestamp,
    yield_modes: tuple[str, ...],
    min_stock_price: float,
) -> pd.DataFrame:
    unknown = set(yield_modes) - {YIELD_MODE_AS_OF_PRICE, YIELD_MODE_TARGET_MONTH_END}
    if unknown:
        raise ValueError(f"Unsupported yield modes: {sorted(unknown)}")

    rows: list[dict[str, object]] = []
    for _, dividend in dividend_estimates.iterrows():
        stock_id = int(dividend["stock_id"])
        stock_prices = prices[prices["stock_id"].eq(stock_id)].sort_values("date")
        if YIELD_MODE_AS_OF_PRICE in yield_modes:
            available = stock_prices[stock_prices["date"].le(as_of_date)]
            price_row = available.iloc[-1] if not available.empty else None
            rows.append(
                _yield_row(
                    dividend,
                    mode=YIELD_MODE_AS_OF_PRICE,
                    target_month=pd.NA,
                    price_row=price_row,
                    min_stock_price=min_stock_price,
                    is_evaluation=False,
                )
            )

        if YIELD_MODE_TARGET_MONTH_END in yield_modes:
            prior = stock_prices[stock_prices["date"].lt(pd.Timestamp(target_year, 1, 1))]
            last_known = prior.iloc[-1] if not prior.empty else None
            target_prices = stock_prices[stock_prices["date"].dt.year.eq(int(target_year))].copy()
            if not target_prices.empty:
                target_prices["target_month"] = target_prices["date"].dt.month.astype(int)
                target_prices = target_prices.groupby("target_month", as_index=False).tail(1)
            lookup = (
                {int(row.target_month): row for row in target_prices.itertuples()}
                if not target_prices.empty
                else {}
            )
            for month in range(1, 13):
                current = lookup.get(month)
                if current is not None:
                    last_known = pd.Series({"date": current.date, "close": current.close})
                rows.append(
                    _yield_row(
                        dividend,
                        mode=YIELD_MODE_TARGET_MONTH_END,
                        target_month=month,
                        price_row=last_known,
                        min_stock_price=min_stock_price,
                        is_evaluation=True,
                    )
                )
    return pd.DataFrame(rows)


def _yield_row(
    dividend: pd.Series,
    *,
    mode: str,
    target_month: object,
    price_row: pd.Series | None,
    min_stock_price: float,
    is_evaluation: bool,
) -> dict[str, object]:
    price = float(price_row["close"]) if price_row is not None else np.nan
    price_date = pd.Timestamp(price_row["date"]) if price_row is not None else pd.NaT
    valid_price = pd.notna(price) and price > float(min_stock_price)
    estimated_cash = dividend["estimated_cash_dividend"]
    actual_cash = dividend["actual_cash_dividend"]
    estimated_yield = (
        float(estimated_cash) / price * 100
        if valid_price and pd.notna(estimated_cash)
        else np.nan
    )
    actual_yield = (
        float(actual_cash) / price * 100
        if valid_price and pd.notna(actual_cash) and is_evaluation
        else np.nan
    )
    row = dividend.to_dict()
    row.update(
        {
            "yield_mode": mode,
            "target_month": target_month,
            "price_date": price_date,
            "stock_price": price,
            "price_source": (
                "latest observed close at or before as_of_date"
                if mode == YIELD_MODE_AS_OF_PRICE
                else "target-year month-end observed close (evaluation)"
            ),
            "estimated_yield_percent": estimated_yield,
            "actual_yield_percent": actual_yield,
            "yield_error_percent_point": (
                estimated_yield - actual_yield
                if pd.notna(estimated_yield) and pd.notna(actual_yield)
                else np.nan
            ),
            "is_evaluation": bool(is_evaluation),
        }
    )
    return row
