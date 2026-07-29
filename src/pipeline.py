"""End-to-end preprocessing pipeline assembly and integrity guards.

Pipeline layout::

    clean     RawDataCleaner            stateless row-local repairs
    features  FeatureBuilder            stateless feature construction
    encode    ColumnTransformer         the only fitted stage

The two stateless stages run first so that every statistic learned from the
data lives in a single, auditable place.

Audit findings addressed here:
  * C-1  - feature names are preserved end-to-end and verified, never lost to a
           swallowed exception.
  * R-10 - train/inference column alignment is asserted, not assumed.
  * NF-7 - no exception is caught and discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.feature_engineering import FeatureBuilder, FeatureEngineeringConfig
from src.logger import get_logger
from src.preprocessing import (
    CategoricalPreprocessingConfig,
    NumericPreprocessingConfig,
    build_numeric_imputer,
    build_onehot_encoder,
    build_scaler,
)
from src.transformers import RawDataCleaner, UnseenCategoryIndicator

logger = get_logger(__name__)

# Columns produced by the stateless stage that are already binary indicators and
# must therefore bypass imputation and scaling.
_BINARY_FEATURE_SUFFIXES = ("_is_missing", "_is_weekend")


@dataclass(frozen=True)
class PreprocessingPipelineConfig:
    """Configuration for the assembled preprocessing pipeline.

    Attributes:
        categorical_columns: Raw categorical column names.
        numeric_columns: Raw numeric column names.
        date_column: Raw date column name.
        feature_config: Stateless feature-construction configuration.
        numeric_config: Imputation/scaling configuration.
        categorical_config: One-hot encoding configuration.
        weight_column: Column receiving the robust scaler.
        weight_sign_repair: Strategy passed to :class:`RawDataCleaner`.
        weight_min: Lower clip bound for weight.
        weight_max: Upper clip bound for weight.
        add_unknown_category_indicator: Whether to emit unseen-category flags.
    """

    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    date_column: str = "date"
    feature_config: FeatureEngineeringConfig = FeatureEngineeringConfig()
    numeric_config: NumericPreprocessingConfig = NumericPreprocessingConfig()
    categorical_config: CategoricalPreprocessingConfig = CategoricalPreprocessingConfig()
    weight_column: str = "weight"
    weight_sign_repair: str = "abs"
    weight_min: float = 5000.0
    weight_max: float = 47500.0
    add_unknown_category_indicator: bool = True


def _classify_built_features(
    built_names: Sequence[str],
    *,
    categorical_columns: Sequence[str],
    weight_column: str,
) -> tuple[list[str], list[str], list[str]]:
    """Split built feature names into weight / other-numeric / binary groups.

    Args:
        built_names: Ordered names emitted by :class:`FeatureBuilder`.
        categorical_columns: Raw categorical columns, excluded from all groups.
        weight_column: Column that receives the robust scaler.

    Returns:
        Tuple of ``(weight_columns, other_numeric_columns, binary_columns)``.
    """
    categorical = set(categorical_columns)
    weight_columns: list[str] = []
    binary_columns: list[str] = []
    other_numeric: list[str] = []

    for name in built_names:
        if name in categorical:
            continue
        if name == weight_column:
            weight_columns.append(name)
        elif name.endswith(_BINARY_FEATURE_SUFFIXES):
            binary_columns.append(name)
        else:
            other_numeric.append(name)

    return weight_columns, other_numeric, binary_columns


def build_preprocessing_pipeline(config: PreprocessingPipelineConfig) -> Pipeline:
    """Assemble the full preprocessing pipeline.

    The encoder stage is built lazily inside a fitted wrapper because the
    column groups depend on the names produced by :class:`FeatureBuilder`.

    Args:
        config: Pipeline configuration.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline` whose output is a
        :class:`pandas.DataFrame` with meaningful column names.
    """
    cleaner = RawDataCleaner(
        categorical_columns=config.categorical_columns,
        weight_column=config.weight_column,
        weight_sign_repair=config.weight_sign_repair,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
    )
    builder = FeatureBuilder(config=config.feature_config)
    encoder = _EncoderStage(config=config)

    pipeline = Pipeline(
        steps=[
            ("clean", cleaner),
            ("features", builder),
            ("encode", encoder),
        ]
    )
    # Guarantees DataFrame output with names at every stage boundary.
    pipeline.set_output(transform="pandas")
    return pipeline


class _EncoderStage(ColumnTransformer):
    """ColumnTransformer whose column groups are resolved at fit time.

    Subclassing keeps the object a first-class scikit-learn transformer (so it
    supports ``set_output``, cloning and ``get_feature_names_out``) while
    deferring group assignment until the built feature names are known.
    """

    def __init__(self, *, config: PreprocessingPipelineConfig) -> None:
        self.config = config
        super().__init__(transformers=[], remainder="drop", verbose_feature_names_out=False)

    def _resolve_transformers(self, X: pd.DataFrame) -> list[tuple]:
        """Build the transformer list for the given built-feature frame."""
        config = self.config
        built_names = list(X.columns)
        weight_columns, other_numeric, binary_columns = _classify_built_features(
            built_names,
            categorical_columns=config.categorical_columns,
            weight_column=config.weight_column,
        )

        numeric_config = config.numeric_config
        transformers: list[tuple] = []

        if weight_columns:
            transformers.append(
                (
                    "numeric_weight",
                    Pipeline(
                        steps=[
                            ("impute", build_numeric_imputer(numeric_config.missing_strategy)),
                            ("scale", build_scaler(numeric_config.weight_scaler)),
                        ]
                    ),
                    weight_columns,
                )
            )
        if other_numeric:
            transformers.append(
                (
                    "numeric_other",
                    Pipeline(
                        steps=[
                            ("impute", build_numeric_imputer(numeric_config.missing_strategy)),
                            ("scale", build_scaler(numeric_config.default_scaler)),
                        ]
                    ),
                    other_numeric,
                )
            )
        if binary_columns:
            # Binary indicators are already on a 0/1 scale; imputing or scaling
            # them would destroy their interpretability for no benefit.
            transformers.append(("binary", "passthrough", binary_columns))

        categorical_columns = list(config.categorical_columns)
        if categorical_columns:
            categorical_config = config.categorical_config
            transformers.append(
                (
                    "categorical",
                    build_onehot_encoder(
                        handle_unknown=categorical_config.handle_unknown,
                        sparse_output=categorical_config.sparse_output,
                        min_frequency=categorical_config.min_frequency,
                    ),
                    categorical_columns,
                )
            )
            if config.add_unknown_category_indicator:
                transformers.append(
                    (
                        "unknown_category",
                        UnseenCategoryIndicator(columns=categorical_columns),
                        categorical_columns,
                    )
                )

        return transformers

    def fit(self, X, y=None):
        """Resolve column groups from ``X`` then fit as a ColumnTransformer."""
        self.transformers = self._resolve_transformers(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Resolve column groups from ``X`` then fit-transform."""
        self.transformers = self._resolve_transformers(X)
        return super().fit_transform(X, y, **fit_params)


# --------------------------------------------------------------------------- #
# Integrity guards
# --------------------------------------------------------------------------- #

_GENERIC_NAME_PATTERN = ("x0", "x1", "x2")


def assert_feature_names_preserved(frame: pd.DataFrame) -> None:
    """Verify the transformed frame carries real, non-generic feature names.

    This is the direct regression guard for audit finding C-1, where every
    processed column was silently renamed to an integer.

    Args:
        frame: Transformed feature frame.

    Raises:
        TypeError: If ``frame`` is not a DataFrame.
        ValueError: If column names are integer-like, generic or duplicated.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame so feature names are preserved, got {type(frame)!r}."
        )

    names = list(frame.columns)
    if not names:
        raise ValueError("Transformed frame has no columns.")

    integer_like = [name for name in names if isinstance(name, (int, np.integer))]
    if integer_like:
        raise ValueError(
            f"Feature names were lost: {len(integer_like)} column(s) are integers, "
            f"e.g. {integer_like[:5]}."
        )

    stringified = [str(name) for name in names]
    numeric_strings = [name for name in stringified if name.isdigit()]
    if numeric_strings:
        raise ValueError(
            f"Feature names were lost: {len(numeric_strings)} column(s) are numeric strings, "
            f"e.g. {numeric_strings[:5]}."
        )

    generic = [name for name in stringified if name in _GENERIC_NAME_PATTERN]
    if generic:
        raise ValueError(f"Feature names look like sklearn placeholders: {generic[:5]}.")

    duplicates = sorted({name for name in stringified if stringified.count(name) > 1})
    if duplicates:
        raise ValueError(f"Transformed frame contains duplicate feature names: {duplicates[:10]}.")


def assert_frames_aligned(train: pd.DataFrame, other: pd.DataFrame, *, label: str) -> None:
    """Verify two transformed frames share identical columns in identical order.

    Positional-only alignment was the latent failure mode flagged as audit risk
    R-10: with integer column names, a silent reordering would have been
    undetectable.

    Args:
        train: Reference (training) transformed frame.
        other: Frame to compare against the reference.
        label: Human-readable name of ``other`` for error messages.

    Raises:
        ValueError: If columns differ in content or order.
    """
    train_columns = list(train.columns)
    other_columns = list(other.columns)

    if train_columns == other_columns:
        return

    missing = [column for column in train_columns if column not in set(other_columns)]
    extra = [column for column in other_columns if column not in set(train_columns)]
    if missing or extra:
        raise ValueError(
            f"{label} feature columns do not match train.\n"
            f"  missing from {label}: {missing[:10]} (total {len(missing)})\n"
            f"  unexpected in {label}: {extra[:10]} (total {len(extra)})"
        )

    first_mismatch = next(
        index
        for index, (left, right) in enumerate(zip(train_columns, other_columns))
        if left != right
    )
    raise ValueError(
        f"{label} feature columns are ordered differently from train; first mismatch at "
        f"index {first_mismatch}: train={train_columns[first_mismatch]!r}, "
        f"{label}={other_columns[first_mismatch]!r}"
    )


def assert_no_missing_values(frame: pd.DataFrame, *, label: str) -> None:
    """Verify no NaNs survive preprocessing.

    Args:
        frame: Transformed feature frame.
        label: Human-readable name for error messages.

    Raises:
        ValueError: If any column still contains missing values.
    """
    na_counts = frame.isna().sum()
    offending = na_counts[na_counts > 0]
    if not offending.empty:
        raise ValueError(
            f"{label} still contains missing values after preprocessing: "
            f"{offending.to_dict()}"
        )


def assert_no_leakage(pipeline: Pipeline, *, forbidden_columns: Sequence[str]) -> None:
    """Verify no forbidden column reached the fitted stage.

    Guards against the target or an identifier being accidentally routed into
    the encoder, which is the classic silent target-leakage path.

    Args:
        pipeline: A fitted preprocessing pipeline.
        forbidden_columns: Columns that must never appear as model inputs
            (typically the target and ``load_id``).

    Raises:
        ValueError: If any forbidden column is present in the fitted inputs.
    """
    builder: FeatureBuilder = pipeline.named_steps["features"]
    seen = set(map(str, builder.get_feature_names_out()))
    violations = sorted(seen.intersection(set(map(str, forbidden_columns))))
    if violations:
        raise ValueError(
            f"Leakage guard failed: forbidden column(s) reached the feature matrix: {violations}"
        )


def transform_frame(pipeline: Pipeline, frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Transform a frame and run every integrity guard on the result.

    Args:
        pipeline: A fitted preprocessing pipeline.
        frame: Raw feature frame to transform.
        label: Human-readable name for error messages.

    Returns:
        The transformed feature frame.
    """
    transformed = pipeline.transform(frame)
    assert_feature_names_preserved(transformed)
    assert_no_missing_values(transformed, label=label)
    logger.info("Transformed %s: %d rows x %d features", label, *transformed.shape)
    return transformed
