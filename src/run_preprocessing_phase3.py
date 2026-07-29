"""Phase-3 entry point: build, verify and persist the preprocessing pipeline.

Replaces ``run_data_preprocess_phase2.py``, which produced the corrupted
``processed/*.csv`` artifacts described in audit finding C-1 (all 142 feature
columns renamed to integers by a swallowed exception).

This script trains no models. It:

  1. loads the raw datasets and validates their schema,
  2. fits the preprocessing pipeline on training data only,
  3. transforms train and validation and runs every integrity guard,
  4. verifies the temporal split utility on the development window,
  5. reconstructs ``data/december_chart_inputs.csv`` and verifies the
     reduced-feature inference path end-to-end,
  6. persists the fitted pipeline plus metadata,
  7. writes the Phase-3 reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.config import AppConfig, load_config, set_global_seed
from src.data_loader import load_datasets
from src.feature_engineering import build_feature_engineering_config
from src.inference import (
    build_city_coordinate_lookup,
    build_daily_market_lookup,
    enrich_reduced_frame,
    write_december_chart_inputs,
)
from src.logger import get_logger
from src.pipeline import (
    PreprocessingPipelineConfig,
    assert_feature_names_preserved,
    assert_frames_aligned,
    assert_no_leakage,
    assert_no_missing_values,
    build_preprocessing_pipeline,
    transform_frame,
)
from src.preprocessing import CategoricalPreprocessingConfig, NumericPreprocessingConfig
from src.preprocessing_utils import ensure_required_columns
from src.splitting import temporal_train_holdout_split

logger = get_logger(__name__)


@dataclass
class PhaseThreeArtifacts:
    """Everything produced by a Phase-3 run, used to build the report."""

    train_raw_shape: tuple[int, int]
    validation_raw_shape: tuple[int, int]
    train_processed_shape: tuple[int, int]
    validation_processed_shape: tuple[int, int]
    feature_names: list[str]
    split_summary: dict[str, object]
    december_shape: tuple[int, int]
    cleaning_stats: dict[str, object]
    checks_passed: list[str]


def build_pipeline_config(config: AppConfig) -> PreprocessingPipelineConfig:
    """Translate the application config into a pipeline configuration.

    Args:
        config: Loaded application configuration.

    Returns:
        A populated :class:`PreprocessingPipelineConfig`.
    """
    feature_config = build_feature_engineering_config(
        date_column=config.columns.date,
        temporal_options=config.features.temporal,
        geospatial_options=config.features.geospatial,
        interaction_options=config.features.interactions,
        missing_indicator_columns=config.cleaning.missing_indicator_columns,
    )
    return PreprocessingPipelineConfig(
        categorical_columns=tuple(config.columns.categorical),
        numeric_columns=tuple(config.columns.numeric),
        date_column=config.columns.date,
        feature_config=feature_config,
        numeric_config=NumericPreprocessingConfig(
            missing_strategy=config.cleaning.numeric_impute_strategy,
            weight_scaler=config.scaling.weight_scaler,
            default_scaler=config.scaling.default_scaler,
        ),
        categorical_config=CategoricalPreprocessingConfig(
            handle_unknown=config.encoding.handle_unknown,
            sparse_output=False,
            min_frequency=config.encoding.min_frequency,
        ),
        weight_sign_repair=config.cleaning.weight_sign_repair,
        weight_min=config.cleaning.weight_min,
        weight_max=config.cleaning.weight_max,
        add_unknown_category_indicator=bool(
            config.raw["encoding"].get("unknown_category_indicator", True)
        ),
    )


def _cleaning_stats(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    """Collect the raw-data quality figures cited in the Phase-3 report."""
    return {
        "train_negative_weight_rows": int((train["weight"] < 0).sum()),
        "validation_negative_weight_rows": int((validation["weight"] < 0).sum()),
        "train_weight_missing": int(train["weight"].isna().sum()),
        "validation_weight_missing": int(validation["weight"].isna().sum()),
        "train_market_index_missing": int(train["market_index"].isna().sum()),
        "validation_market_index_missing": int(validation["market_index"].isna().sum()),
        "unseen_pickup_cities": sorted(set(validation["pickup"]) - set(train["pickup"])),
        "unseen_delivery_cities": sorted(set(validation["delivery"]) - set(train["delivery"])),
        "validation_rows_unseen_pickup": int(
            validation["pickup"].isin(set(validation["pickup"]) - set(train["pickup"])).sum()
        ),
        "validation_rows_unseen_delivery": int(
            validation["delivery"].isin(set(validation["delivery"]) - set(train["delivery"])).sum()
        ),
    }


def main() -> None:
    """Run the full Phase-3 preprocessing build and verification."""
    config = load_config()
    set_global_seed(config.random_seed)
    columns = config.columns
    checks: list[str] = []

    # -- 1. Load and validate --------------------------------------------- #
    datasets = load_datasets(
        train_path=config.paths.train, validation_path=config.paths.validation
    )
    train_raw, validation_raw = datasets.train, datasets.validation

    ensure_required_columns(
        train_raw,
        required=[columns.id, *columns.raw_feature_columns, columns.target],
        label="train",
    )
    ensure_required_columns(
        validation_raw,
        required=[columns.id, *columns.raw_feature_columns],
        label="validation",
    )
    checks.append("Raw schema validated for train and validation.")

    cleaning_stats = _cleaning_stats(train_raw, validation_raw)

    # -- 2. Fit on training features only ---------------------------------- #
    y_train = train_raw[columns.target].copy()
    X_train = train_raw[columns.raw_feature_columns]
    X_validation = validation_raw[columns.raw_feature_columns]

    pipeline = build_preprocessing_pipeline(build_pipeline_config(config))
    pipeline.fit(X_train)
    checks.append("Pipeline fitted on training rows only; validation never seen during fit.")

    train_processed = transform_frame(pipeline, X_train, label="train")
    validation_processed = transform_frame(pipeline, X_validation, label="validation")

    # -- 3. Integrity guards ------------------------------------------------ #
    assert_feature_names_preserved(train_processed)
    assert_feature_names_preserved(validation_processed)
    checks.append("Feature names preserved: no integer or placeholder column names.")

    assert_frames_aligned(train_processed, validation_processed, label="validation")
    checks.append("Train/validation feature columns identical in content and order.")

    assert_no_missing_values(train_processed, label="train")
    assert_no_missing_values(validation_processed, label="validation")
    checks.append("No missing values survive preprocessing.")

    assert_no_leakage(pipeline, forbidden_columns=[columns.target, columns.id])
    checks.append(f"Leakage guard: neither {columns.target!r} nor {columns.id!r} reached features.")

    # Determinism: refitting with the same seed must reproduce the matrix.
    replica = build_preprocessing_pipeline(build_pipeline_config(config))
    replica.fit(X_train)
    replica_processed = replica.transform(X_validation)
    if not validation_processed.equals(replica_processed):
        raise ValueError("Preprocessing is not deterministic: refit produced a different matrix.")
    checks.append("Deterministic: an independent refit reproduces the matrix exactly.")

    # -- 4. Temporal split verification ------------------------------------ #
    split = temporal_train_holdout_split(
        train_raw, date_column=columns.date, holdout_start=config.split.holdout_start
    )
    split_pipeline = build_preprocessing_pipeline(build_pipeline_config(config))
    split_pipeline.fit(split.train[columns.raw_feature_columns])
    holdout_processed = split_pipeline.transform(split.holdout[columns.raw_feature_columns])
    split_train_processed = split_pipeline.transform(split.train[columns.raw_feature_columns])
    assert_frames_aligned(split_train_processed, holdout_processed, label="temporal holdout")
    checks.append(
        "Temporal split verified: no date overlap, holdout aligns with its own training block."
    )

    # -- 5. Reduced-feature December path ---------------------------------- #
    december_path = write_december_chart_inputs(config.paths.december_inputs)
    december_raw = pd.read_csv(december_path)

    coordinates = build_city_coordinate_lookup(train_raw, validation_raw)
    market_lookup = build_daily_market_lookup(validation_raw, date_column=columns.date)
    december_enriched = enrich_reduced_frame(
        december_raw.drop(columns=["predicted_rate"]),
        coordinates=coordinates,
        market_lookup=market_lookup,
        date_column=columns.date,
    )
    december_processed = pipeline.transform(december_enriched[columns.raw_feature_columns])
    assert_feature_names_preserved(december_processed)
    assert_no_missing_values(december_processed, label="december")
    assert_frames_aligned(train_processed, december_processed, label="december")
    if len(december_processed) != 31:
        raise ValueError(f"Expected 31 December rows, got {len(december_processed)}")
    checks.append(
        "Reduced-feature December path verified: 31 rows reconstructed and aligned with train."
    )

    # -- 6. Persist artifacts ----------------------------------------------- #
    processed_dir = config.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_output = train_processed.copy()
    train_output[columns.target] = y_train.to_numpy()
    train_output.to_csv(processed_dir / "train_processed.csv", index=False)

    validation_output = validation_processed.copy()
    validation_output.insert(0, columns.id, validation_raw[columns.id].to_numpy())
    validation_output.to_csv(processed_dir / "validation_processed.csv", index=False)
    checks.append(f"{columns.id!r} carried alongside validation features for safe re-joining.")

    december_output = december_processed.copy()
    december_output.insert(0, columns.date, december_raw[columns.date].to_numpy())
    december_output.to_csv(processed_dir / "december_processed.csv", index=False)

    models_dir = config.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / "preprocessing_pipeline.joblib")

    feature_names = [str(name) for name in train_processed.columns]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": config.random_seed,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_rows": int(len(train_processed)),
        "validation_rows": int(len(validation_processed)),
        "split": split.summary,
        "cleaning_stats": cleaning_stats,
    }
    (models_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    checks.append("Fitted pipeline and metadata persisted for reuse in later phases.")

    artifacts = PhaseThreeArtifacts(
        train_raw_shape=train_raw.shape,
        validation_raw_shape=validation_raw.shape,
        train_processed_shape=train_processed.shape,
        validation_processed_shape=validation_processed.shape,
        feature_names=feature_names,
        split_summary=split.summary,
        december_shape=december_processed.shape,
        cleaning_stats=cleaning_stats,
        checks_passed=checks,
    )

    # -- 7. Reports ---------------------------------------------------------- #
    from src.reporting_phase3 import write_phase3_reports

    write_phase3_reports(artifacts, config=config)
    logger.info("Phase 3 complete: %d features, all %d checks passed", len(feature_names), len(checks))


if __name__ == "__main__":
    main()
