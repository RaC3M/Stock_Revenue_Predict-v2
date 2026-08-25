import pandas as pd

from structural_break_engine import (
    StructuralBreakConfig,
    add_structural_break_features,
    apply_structural_break_adjustment,
)


def test_detects_known_collapse_without_target_actual() -> None:
    dates = pd.date_range("2023-12-01", periods=14, freq="MS")
    values = [1000.0] * 12 + [10.0, 8.0]
    revenue = pd.DataFrame(
        {
            "stock_id": 1,
            "date": dates,
            "revenue_thousand": values,
        }
    )
    predictions = pd.DataFrame(
        {
            "stock_id": [1],
            "target_date": [pd.Timestamp("2025-02-01")],
            "last_observed_revenue": [8.0],
            "formula_adjusted_revenue": [500.0],
        }
    )
    featured = add_structural_break_features(predictions, revenue)
    adjusted = apply_structural_break_adjustment(
        featured,
        StructuralBreakConfig(formula_retention=0.0),
    )
    assert bool(adjusted.iloc[0]["structural_break_detected"])
    assert adjusted.iloc[0]["formula_adjusted_revenue"] == 8.0


if __name__ == "__main__":
    test_detects_known_collapse_without_target_actual()
    print("structural break smoke test passed")
