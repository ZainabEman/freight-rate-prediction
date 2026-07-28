from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_engineering import build_datetime_feature_transformer
from preprocessing import (
    CategoricalPreprocessingConfig,
    NumericPreprocessingConfig,
    build_onehot_encoder,
    build_numeric_imputer,
    build_numeric_scalers,
)
from preprocessing_utils import FeatureColumns, ensure_required_columns
from transformers import SafeStringNormalizer


@dataclass(frozen=True)
class PreprocessingPipelineConfig:
    """
    Pipeline configuration for Phase-2 preprocessing.

    Based strictly on Phase-1 audit:
      - Missing:
          * weight, market_index => median imputation
      - Whitespace:
          * categorical columns => strip whitespace
      - Datetime:
          * date => extract year/month/day
      - Encoding:
          * pickup, delivery, equipment => one-hot
      - Scaling:
          * weight => RobustScaler
          * other numeric => StandardScaler
    """

    include_target_in_output: bool = True  # for train only; runner will control per split


def _build_datetime_extraction_transformer(date_col: str):
    """
    Wrapper to keep ColumnTransformer compatibility.
    We extract year/month/day and output as a numeric array.
    """
    transformer = build_datetime_feature_transformer(date_col=date_col)

    def transform_df(x: pd.DataFrame) -> pd.DataFrame:
        parts = transformer.transform(x)
        return parts

    # FunctionTransformer keeps ColumnTransformer happy with DataFrame -> array conversions.
    return FunctionTransformer(transform_df, validate=False)


def build_preprocessing_pipeline(
    *,
    feature_cols: FeatureColumns,
    config: PreprocessingPipelineConfig | None = None,
) -> tuple[Pipeline, list[str]]:
    """
    Build sklearn preprocessing pipeline for model-ready features.

    Returns:
      (pipeline, engineered_feature_names)
    """
    config = config or PreprocessingPipelineConfig()

    # Core preprocessing blocks
    numeric_imputer = build_numeric_imputer()
    weight_scaler, other_scaler = build_numeric_scalers(
        weight_scaler=NumericPreprocessingConfig().weight_scaler,
        other_scaler=NumericPreprocessingConfig().other_scaler,
    )

    categorical_encoder = build_onehot_encoder(
        handle_unknown=CategoricalPreprocessingConfig().handle_unknown,
        sparse_output=CategoricalPreprocessingConfig().sparse_output,
    )

    # date extraction will create engineered numeric columns; ColumnTransformer will output them
    date_extractor = _build_datetime_extraction_transformer(feature_cols.datetime_cols[0])

    # Transformers per column groups
    # We want to keep whitespace stripping and then one-hot encoding.
    # Since Phase-1 audit indicates no missing values in these categoricals, we avoid imputation
    # to keep decisions minimal. If missing appears later, OneHotEncoder will treat NaNs as a category
    # only if handle_unknown applies; we document that later in preprocessing report.

    categorical_pipeline = Pipeline(
        steps=[
            ("string_normalize", SafeStringNormalizer()),
            ("onehot", categorical_encoder),
        ]
    )

    weight_cols = ["weight"]
    other_numeric_cols = [c for c in feature_cols.numeric_cols if c not in weight_cols]

    numeric_weight_pipeline = Pipeline(
        steps=[
            ("impute", numeric_imputer),
            ("scale", weight_scaler),
        ]
    )

    numeric_other_pipeline = Pipeline(
        steps=[
            ("impute", numeric_imputer),
            ("scale", other_scaler),
        ]
    )

    # Assemble ColumnTransformer
    # - date: extracted into 3 numeric cols; we pass date_extractor with date column selection.
    # - numeric: separate for weight vs others to select scaler
    # - categorical: onehot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("date_parts", date_extractor, feature_cols.datetime_cols),
            ("numeric_weight", numeric_weight_pipeline, weight_cols),
            ("numeric_other", numeric_other_pipeline, other_numeric_cols),
            ("categorical", categorical_pipeline, feature_cols.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Feature names produced by date extractor: date_year/date_month/date_day per DateParts defaults
    # The rest of feature names are handled by sklearn's get_feature_names_out during runner.
    engineered_feature_names = ["date_year", "date_month", "date_day"]

    # Wrap in sklearn Pipeline for fit/transform compatibility with one object.
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    return pipeline, engineered_feature_names
