from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from transformers import DateParts, DateToPartsTransformer


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """
    Feature engineering configuration.

    Phase 1 audit indicates `date` is present and should be treated as datetime-like.
    We extract year/month/day as numeric features for modeling.
    """

    date_col: str = "date"
    drop_original_date: bool = True


def build_datetime_feature_transformer(
    *, date_col: str = "date", parts: DateParts | None = None
) -> DateToPartsTransformer:
    """
    Create a sklearn-compatible transformer that extracts year/month/day.
    """
    return DateToPartsTransformer(date_col=date_col, parts=parts)


def infer_engineered_datetime_feature_names(*, parts: DateParts | None = None) -> list[str]:
    """
    Engineered feature names produced by DateToPartsTransformer.
    """
    parts = parts or DateParts()
    return [parts.year, parts.month, parts.day]


def apply_datetime_feature_engineering(
    df: pd.DataFrame, *, cfg: FeatureEngineeringConfig = FeatureEngineeringConfig()
) -> pd.DataFrame:
    """
    Convenience helper for non-sklearn contexts (e.g., report/debugging).
    The main pipeline should rely on the sklearn transformer for consistency.
    """
    transformer = build_datetime_feature_transformer(date_col=cfg.date_col)
    parts_df = transformer.transform(df)

    out = df.join(parts_df)
    if cfg.drop_original_date and cfg.date_col in out.columns:
        out = out.drop(columns=[cfg.date_col])
    return out
