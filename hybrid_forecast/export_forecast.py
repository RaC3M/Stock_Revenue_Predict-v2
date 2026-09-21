from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_forecast.live_engine import (
    ForecastRequest,
    apply_price_scenario,
    load_latest_price,
    load_or_build_forecast,
    write_llm_bundle,
)


def _price_overrides(values: list[str]) -> dict[int, float]:
    parsed: dict[int, float] = {}
    for value in values:
        try:
            stock, price = value.split("=", 1)
            parsed[int(stock)] = float(price)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(
                f"價格覆寫格式必須是 STOCK=PRICE：{value}"
            ) from error
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="匯出 SARIMA＋營收公式的 LLM JSON 與 CSV 資料包。"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--stock", action="append", required=True, type=int)
    parser.add_argument(
        "--manual-price", action="append", default=[], metavar="STOCK=PRICE",
        help="選擇性手動價格；價格日期為 --as-of。",
    )
    parser.add_argument(
        "--output-dir", default=str(Path("hybrid_forecast") / "outputs" / "llm_exports")
    )
    args = parser.parse_args()
    try:
        overrides = _price_overrides(args.manual_price)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    failures: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    for stock in args.stock:
        try:
            request = ForecastRequest(stock, args.as_of, args.data_dir)
            core = load_or_build_forecast(request)
            if stock in overrides:
                result = apply_price_scenario(
                    core,
                    stock_price=overrides[stock],
                    price_date=request.cutoff,
                    price_source="manual_scenario",
                )
            else:
                observed = load_latest_price(request)
                if observed is None:
                    raise ValueError("基準日以前沒有可用股價")
                result = apply_price_scenario(
                    core,
                    stock_price=observed[0],
                    price_date=observed[1],
                    price_source="observed_csv",
                )
            destination = write_llm_bundle(result, args.output_dir)
            completed.append({
                "stock_id": stock,
                "artifact_id": result.artifact_id,
                "cache_hit": result.cache_hit,
                "output": str(destination.resolve()),
            })
        except (OSError, ValueError, KeyError, ImportError, ArithmeticError, RuntimeError) as error:
            failures.append({"stock_id": stock, "error": str(error)})
    print(json.dumps({"completed": completed, "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
