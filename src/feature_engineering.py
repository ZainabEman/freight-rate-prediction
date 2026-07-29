"""Stateless feature construction stage.

This module composes the individual transformers in :mod:`src.transformers`
into a single :class:`FeatureBuilder` that takes the raw feature frame and
returns the raw columns *plus* every engineered column, with names intact.

Keeping construction stateless and separate from encoding means the fitted
(statistical) part of the pipeline is confined to the final
:class:`~sklearn.compose.ColumnTransformer`, which makes leakage auditing
straightforward: only that stage learns anything from the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.transformers import (
    CyclicalDateFeatures,
    GeoFeatureSpec,
    GeoFeatures,
    InteractionFeatureSpec,
    InteractionFeatures,
    MissingFlagger,
    TemporalFeatureSpec,
    _as_frame,
)


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration for the stateless feature-construction stage.

    Attributes:
        date_column: Name of the raw date column.
        temporal: Temporal encoding switches.
        geospatial: Geospatial encoding switches.
        interactions: Interaction encoding switches.
        missing_indicator_columns: Columns to emit missingness flags for.
        drop_date_column: Whether to drop the raw date column from the output.
            It is always dropped before encoding because it is a string.
    """

    date_column: str = "date"
    temporal: TemporalFeatureSpec = TemporalFeatureSpec()
    geospatial: GeoFeatureSpec = GeoFeatureSpec()
    interactions: InteractionFeatureSpec = InteractionFeatureSpec()
    missing_indicator_columns: tuple[str, ...] = ("weight", "market_index")
    drop_date_column: bool = True


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Append engineered features to the raw frame, preserving column names.

    Output layout is ``[raw columns (minus date)] + [temporal] + [geospatial] +
    [interactions] + [missingness flags]``. The order is deterministic, which is
    what makes the train/inference column-alignment assertions in
    :mod:`src.pipeline` meaningful.
    """

    def __init__(self, *, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "FeatureBuilder":
        """Fit the (stateless) sub-transformers and freeze the output schema.

        Args:
            X: Raw feature frame.
            y: Ignored; present for scikit-learn API compatibility.

        Returns:
            The fitted builder.
        """
        frame = _as_frame(X)
        config = self.config

        self.temporal_ = CyclicalDateFeatures(
            date_column=config.date_column, spec=config.temporal
        ).fit(frame)
        self.geospatial_ = GeoFeatures(spec=config.geospatial).fit(frame)
        self.interactions_ = InteractionFeatures(spec=config.interactions).fit(frame)
        self.missing_flagger_ = MissingFlagger(columns=list(config.missing_indicator_columns)).fit(
            frame
        )

        passthrough = [
            column
            for column in frame.columns
            if not (config.drop_date_column and column == config.date_column)
        ]
        self.passthrough_columns_ = passthrough

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = [
            *passthrough,
            *self.temporal_.get_feature_names_out().tolist(),
            *self.geospatial_.get_feature_names_out().tolist(),
            *self.interactions_.get_feature_names_out().tolist(),
            *self.missing_flagger_.get_feature_names_out().tolist(),
        ]

        duplicates = {
            name for name in self.feature_names_out_ if self.feature_names_out_.count(name) > 1
        }
        if duplicates:
            raise ValueError(f"FeatureBuilder produced duplicate feature names: {sorted(duplicates)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the raw + engineered feature frame.

        Raises:
            ValueError: If the produced columns differ from those frozen at fit
                time. This is the guard against silent train/inference schema
                drift (audit finding R-10).
        """
        check_is_fitted(self, "feature_names_out_")
        frame = _as_frame(X)

        blocks = [
            frame[self.passthrough_columns_],
            self.temporal_.transform(frame),
            self.geospatial_.transform(frame),
            self.interactions_.transform(frame),
            self.missing_flagger_.transform(frame),
        ]
        out = pd.concat(blocks, axis=1)

        if list(out.columns) != self.feature_names_out_:
            raise ValueError(
                "FeatureBuilder output schema drifted between fit and transform.\n"
                f"  expected: {self.feature_names_out_}\n"
                f"  actual:   {list(out.columns)}"
            )
        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the full ordered list of built feature names."""
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


def build_feature_engineering_config(
    *,
    date_column: str,
    temporal_options: dict[str, Any],
    geospatial_options: dict[str, Any],
    interaction_options: dict[str, Any],
    missing_indicator_columns: Sequence[str],
) -> FeatureEngineeringConfig:
    """Translate raw YAML option dictionaries into a typed config object.

    Args:
        date_column: Name of the raw date column.
        temporal_options: ``features.temporal`` section of the YAML config.
        geospatial_options: ``features.geospatial`` section.
        interaction_options: ``features.interactions`` section.
        missing_indicator_columns: Columns to flag for missingness.

    Returns:
        A populated :class:`FeatureEngineeringConfig`.
    """
    temporal = TemporalFeatureSpec(
        use_cyclical_day_of_year=bool(temporal_options.get("use_cyclical_day_of_year", True)),
        use_cyclical_day_of_week=bool(temporal_options.get("use_cyclical_day_of_week", True)),
        use_is_weekend=bool(temporal_options.get("use_is_weekend", True)),
        use_days_since_reference=bool(temporal_options.get("use_days_since_reference", False)),
        reference_date=str(temporal_options.get("reference_date", "2025-01-01")),
    )
    geospatial = GeoFeatureSpec(
        use_haversine=bool(geospatial_options.get("use_haversine", True)),
        use_bearing=bool(geospatial_options.get("use_bearing", True)),
        use_lon_delta=bool(geospatial_options.get("use_lon_delta", True)),
        use_lat_delta=bool(geospatial_options.get("use_lat_delta", True)),
    )
    interactions = InteractionFeatureSpec(
        use_weight_per_mile=bool(interaction_options.get("use_weight_per_mile", True)),
        use_log_distance=bool(interaction_options.get("use_log_distance", True)),
    )
    return FeatureEngineeringConfig(
        date_column=date_column,
        temporal=temporal,
        geospatial=geospatial,
        interactions=interactions,
        missing_indicator_columns=tuple(missing_indicator_columns),
    )
