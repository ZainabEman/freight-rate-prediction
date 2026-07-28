from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureColumns:
    """
    Centralized column lists for preprocessing.

    Values must match Phase 1 audit findings:
      - identifier: load_id (kept for join/key, usually not model input)
      - categorical: pickup, delivery, equipment
      - datetime: date
      - numeric: pickup_lat, pickup_lon, delivery_lat, delivery_lon, distance,
                 weight, market_index, quote_signal
      - target (train only): posted_rate
    """

    id_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    numeric_cols: list[str]
    target_col: str
    all_input_cols: list[str]


def default_feature_columns() -> FeatureColumns:
    """
    Returns default column configuration derived from Phase 1 audit.

    Note: We do not validate here against the actual CSV schema;
    pipeline builders should verify columns exist.
    """
    id_cols = ["load_id"]
    categorical_cols = ["pickup", "delivery", "equipment"]
    datetime_cols = ["date"]
    numeric_cols = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
    ]
    target_col = "posted_rate"
    all_input_cols = id_cols + categorical_cols + datetime_cols + numeric_cols + [target_col]
    return FeatureColumns(
        id_cols=id_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        numeric_cols=numeric_cols,
        target_col=target_col,
        all_input_cols=all_input_cols,
    )


def ensure_required_columns(df: pd.DataFrame, *, required: list[str], label: str) -> None:
    """
    Ensure the DataFrame contains all required columns.
    Raises ValueError with a readable message on mismatch.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}. Present columns={list(df.columns)}")


def strip_whitespace_in_place(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace in the specified columns.
    Does not attempt case normalization.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()
    return out
