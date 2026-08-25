"""Adapter for existing Rolling LSTM monthly prediction outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROLLING_REQUIRED_COLUMNS = {
    "stock_id",
    "target_year",
    "target_month",
    "model",
    "predicted_revenue",
    "actual_revenue",
}
ROLLING_XLSTM_MODELS = frozenset(
    {
        "Rolling xLSTM",
        "Rolling xLSTM + Conditional Adjustment",
    }
)
VALID_XLSTM_BACKBONES = frozenset({"xlstm", "xlstm_hybrid"})
ARCHITECTURE_COLUMNS = ("sequence_backbone", "xlstm_backbone")


def _validate_xlstm_architecture_provenance(predictions: pd.DataFrame) -> pd.DataFrame:
    xlstm_mask = predictions["model"].isin(ROLLING_XLSTM_MODELS)
    if not xlstm_mask.any():
        return predictions

    missing_columns = [column for column in ARCHITECTURE_COLUMNS if column not in predictions.columns]
    if missing_columns:
        raise ValueError(
            "Rolling xLSTM predictions require architecture provenance columns: "
            f"{missing_columns}"
        )

    normalized = predictions.copy()
    for column in ARCHITECTURE_COLUMNS:
        normalized[column] = normalized[column].astype("string").str.strip().replace("", pd.NA)

    xlstm_rows = normalized.loc[xlstm_mask, ["model", *ARCHITECTURE_COLUMNS]]
    missing_values = [column for column in ARCHITECTURE_COLUMNS if xlstm_rows[column].isna().any()]
    if missing_values:
        raise ValueError(
            "Rolling xLSTM predictions contain missing architecture provenance values: "
            f"{missing_values}"
        )

    invalid_values = {
        column: sorted(set(xlstm_rows[column]).difference(VALID_XLSTM_BACKBONES))
        for column in ARCHITECTURE_COLUMNS
    }
    invalid_values = {column: values for column, values in invalid_values.items() if values}
    if invalid_values:
        raise ValueError(f"Rolling xLSTM predictions contain invalid architecture provenance: {invalid_values}")

    mismatched = xlstm_rows["sequence_backbone"].ne(xlstm_rows["xlstm_backbone"])
    if mismatched.any():
        raise ValueError(
            "Rolling xLSTM sequence_backbone must match xlstm_backbone for every prediction row."
        )

    architecture_pairs = xlstm_rows[list(ARCHITECTURE_COLUMNS)].drop_duplicates()
    if len(architecture_pairs) != 1:
        raise ValueError(
            "Rolling benchmark input mixes xLSTM architectures; evaluate historical mLSTM-only "
            "and Hybrid outputs in separate benchmark runs."
        )
    return normalized


def load_rolling_predictions(
    output_dir: str | Path,
    target_year: int,
    stock_ids: list[int] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    predictions_path = output_dir / "monthly_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Rolling monthly predictions not found: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    missing = ROLLING_REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Rolling predictions missing columns: {sorted(missing)}")

    normalized = predictions.copy()
    normalized["stock_id"] = pd.to_numeric(normalized["stock_id"], errors="coerce")
    normalized["target_year"] = pd.to_numeric(normalized["target_year"], errors="coerce")
    normalized["target_month"] = pd.to_numeric(normalized["target_month"], errors="coerce")
    normalized["predicted_revenue"] = pd.to_numeric(normalized["predicted_revenue"], errors="coerce")
    normalized["actual_revenue"] = pd.to_numeric(normalized["actual_revenue"], errors="coerce")
    if "last_observed_revenue" in normalized.columns:
        normalized["last_observed_revenue"] = pd.to_numeric(
            normalized["last_observed_revenue"], errors="coerce"
        )
    else:
        normalized["last_observed_revenue"] = pd.NA

    normalized = normalized.dropna(
        subset=["stock_id", "target_year", "target_month", "model", "predicted_revenue"]
    )
    normalized["stock_id"] = normalized["stock_id"].astype(int)
    normalized["target_year"] = normalized["target_year"].astype(int)
    normalized["target_month"] = normalized["target_month"].astype(int)
    normalized = normalized[normalized["target_year"].eq(int(target_year))]

    if stock_ids is not None:
        normalized = normalized[normalized["stock_id"].isin([int(stock_id) for stock_id in stock_ids])]
    if model_names is not None:
        requested_models = {str(model) for model in model_names}
        available_models = set(normalized["model"].dropna().astype(str).unique())
        missing_models = sorted(requested_models.difference(available_models))
        if missing_models:
            raise ValueError(
                f"Rolling output is missing requested models for {target_year}: {missing_models}"
            )
        normalized = normalized[normalized["model"].isin(model_names)]

    normalized = _validate_xlstm_architecture_provenance(normalized)
    normalized["source_family"] = "rolling_lstm"
    normalized["source_path"] = str(predictions_path)
    for optional_column in [
        "stock_name",
        "industry_category",
        "sequence_backbone",
        "xlstm_backbone",
    ]:
        if optional_column not in normalized.columns:
            normalized[optional_column] = pd.NA

    return normalized[
        [
            "source_family",
            "model",
            "stock_id",
            "stock_name",
            "industry_category",
            "sequence_backbone",
            "xlstm_backbone",
            "target_year",
            "target_month",
            "predicted_revenue",
            "actual_revenue",
            "last_observed_revenue",
            "source_path",
        ]
    ].sort_values(["stock_id", "target_month", "source_family", "model"])
