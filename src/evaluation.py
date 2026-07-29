"""Time-aware evaluation harness for baseline models.

This module owns the *protocol*, not the preprocessing: it calls the Phase-3
pipeline from :mod:`src.pipeline` and never reimplements any cleaning, feature
construction or encoding logic.

The central leakage rule is enforced structurally: for every evaluation - the
holdout and each cross-validation fold - a **fresh** preprocessing pipeline is
fitted on that split's training rows only. Nothing fitted on future data is ever
used to transform past data, and no fitted state is shared across folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from src.baselines import BaselineSpec
from src.config import AppConfig
from src.logger import get_logger
from src.metrics import RegressionMetrics, clip_predictions, compute_metrics
from src.pipeline import (
    PreprocessingPipelineConfig,
    assert_frames_aligned,
    build_preprocessing_pipeline,
)
from src.splitting import temporal_cv_splits, temporal_train_holdout_split

logger = get_logger(__name__)


@dataclass
class BaselineResult:
    """Evaluation outcome for one baseline."""

    name: str
    description: str
    input_kind: str
    target_transform: str
    holdout: RegressionMetrics
    cv_mae_mean: float
    cv_mae_std: float
    cv_fold_mae: list[float] = field(default_factory=list)
    fit_seconds: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flatten to a single row for the comparison table."""
        return {
            "model": self.name,
            "MAE": self.holdout.mae,
            "RMSE": self.holdout.rmse,
            "R2": self.holdout.r2,
            "MAPE_%": self.holdout.mape,
            "CV_MAE_mean": self.cv_mae_mean,
            "CV_MAE_std": self.cv_mae_std,
            "fit_s": self.fit_seconds,
        }


def apply_target_transform(y: np.ndarray, transform: str) -> np.ndarray:
    """Forward-transform the target.

    Raises:
        ValueError: On an unknown transform, or non-positive values under log.
    """
    values = np.asarray(y, dtype=float)
    if transform == "identity":
        return values
    if transform == "log":
        if np.any(values <= 0):
            raise ValueError("Log target transform requires strictly positive values.")
        return np.log(values)
    raise ValueError(f"Unknown target transform: {transform!r}")


def invert_target_transform(y: np.ndarray, transform: str) -> np.ndarray:
    """Invert the target transform applied by :func:`apply_target_transform`."""
    values = np.asarray(y, dtype=float)
    if transform == "identity":
        return values
    if transform == "log":
        return np.exp(np.clip(values, -50.0, 50.0))
    raise ValueError(f"Unknown target transform: {transform!r}")


def fit_predict_split(
    spec: BaselineSpec,
    *,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    config: AppConfig,
    pipeline_config: PreprocessingPipelineConfig,
) -> np.ndarray:
    """Fit a baseline on one training split and predict on the evaluation split.

    A fresh preprocessing pipeline is fitted on ``train_frame`` only. This is
    what makes each fold leakage-free.

    Args:
        spec: The baseline to evaluate.
        train_frame: Raw rows used for fitting (features and target).
        evaluation_frame: Raw rows used for scoring.
        config: Application configuration.
        pipeline_config: Phase-3 preprocessing configuration.

    Returns:
        Predictions on ``evaluation_frame`` in the original target units.
    """
    columns = config.columns
    y_train = apply_target_transform(
        train_frame[columns.target].to_numpy(dtype=float), spec.target_transform
    )

    if spec.input_kind == "raw":
        X_train: Any = train_frame[columns.raw_feature_columns]
        X_evaluation: Any = evaluation_frame[columns.raw_feature_columns]
    elif spec.input_kind == "processed":
        preprocessor = build_preprocessing_pipeline(pipeline_config)
        preprocessor.fit(train_frame[columns.raw_feature_columns])
        X_train = preprocessor.transform(train_frame[columns.raw_feature_columns])
        X_evaluation = preprocessor.transform(evaluation_frame[columns.raw_feature_columns])
        assert_frames_aligned(X_train, X_evaluation, label="evaluation split")
    else:
        raise ValueError(f"Unknown input_kind: {spec.input_kind!r}")

    estimator: BaseEstimator = clone(spec.estimator)
    estimator.fit(X_train, y_train)
    raw_predictions = estimator.predict(X_evaluation)
    return clip_predictions(invert_target_transform(raw_predictions, spec.target_transform))


def evaluate_baseline(
    spec: BaselineSpec,
    *,
    development_frame: pd.DataFrame,
    config: AppConfig,
    pipeline_config: PreprocessingPipelineConfig,
    run_cross_validation: bool = True,
) -> BaselineResult:
    """Evaluate one baseline on the temporal holdout and expanding-window CV.

    Args:
        spec: The baseline to evaluate.
        development_frame: The full labelled development dataset.
        config: Application configuration.
        pipeline_config: Phase-3 preprocessing configuration.
        run_cross_validation: Whether to also run expanding-window CV.

    Returns:
        A populated :class:`BaselineResult`.
    """
    import time

    columns = config.columns
    started = time.perf_counter()

    split = temporal_train_holdout_split(
        development_frame,
        date_column=columns.date,
        holdout_start=config.split.holdout_start,
    )
    predictions = fit_predict_split(
        spec,
        train_frame=split.train,
        evaluation_frame=split.holdout,
        config=config,
        pipeline_config=pipeline_config,
    )
    holdout_metrics = compute_metrics(
        split.holdout[columns.target].to_numpy(dtype=float), predictions
    )

    fold_mae: list[float] = []
    if run_cross_validation:
        for train_positions, validation_positions in temporal_cv_splits(
            development_frame,
            date_column=columns.date,
            n_splits=config.split.n_cv_splits,
        ):
            fold_train = development_frame.iloc[train_positions]
            fold_validation = development_frame.iloc[validation_positions]
            fold_predictions = fit_predict_split(
                spec,
                train_frame=fold_train,
                evaluation_frame=fold_validation,
                config=config,
                pipeline_config=pipeline_config,
            )
            fold_metrics = compute_metrics(
                fold_validation[columns.target].to_numpy(dtype=float), fold_predictions
            )
            fold_mae.append(fold_metrics.mae)

    elapsed = time.perf_counter() - started
    result = BaselineResult(
        name=spec.name,
        description=spec.description,
        input_kind=spec.input_kind,
        target_transform=spec.target_transform,
        holdout=holdout_metrics,
        cv_mae_mean=float(np.mean(fold_mae)) if fold_mae else float("nan"),
        cv_mae_std=float(np.std(fold_mae)) if fold_mae else float("nan"),
        cv_fold_mae=fold_mae,
        fit_seconds=elapsed,
    )
    logger.info(
        "%-32s holdout MAE=%8.2f RMSE=%8.2f R2=%6.4f MAPE=%5.2f%% (%.1fs)",
        spec.name,
        holdout_metrics.mae,
        holdout_metrics.rmse,
        holdout_metrics.r2,
        holdout_metrics.mape,
        elapsed,
    )
    return result


def comparison_table(results: list[BaselineResult]) -> pd.DataFrame:
    """Build the model comparison table, ranked by holdout MAE.

    Args:
        results: Evaluated baselines.

    Returns:
        A DataFrame ordered best-first.
    """
    frame = pd.DataFrame([result.as_row() for result in results])
    return frame.sort_values("MAE", ascending=True).reset_index(drop=True)


def select_best_baseline(results: list[BaselineResult]) -> BaselineResult:
    """Select the reference baseline for Phase 5.

    MAE is the selection criterion: the business cost of a mispriced load is
    roughly linear in dollars, and MAE is not dominated by the small number of
    very high-rate outliers that inflate RMSE.

    Args:
        results: Evaluated baselines.

    Returns:
        The baseline with the lowest holdout MAE.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("Cannot select a best baseline from an empty result list.")
    return min(results, key=lambda result: result.holdout.mae)


def verify_no_leakage(
    *,
    development_frame: pd.DataFrame,
    config: AppConfig,
) -> list[str]:
    """Re-verify the split invariants used by every evaluation.

    Args:
        development_frame: The labelled development dataset.
        config: Application configuration.

    Returns:
        Human-readable confirmations for the report.

    Raises:
        ValueError: If any invariant is violated.
    """
    columns = config.columns
    confirmations: list[str] = []

    split = temporal_train_holdout_split(
        development_frame, date_column=columns.date, holdout_start=config.split.holdout_start
    )
    if split.train_date_range[1] >= split.holdout_date_range[0]:
        raise ValueError("Holdout overlaps the training window.")
    confirmations.append(
        f"Holdout is strictly future: train ends {split.train_date_range[1].date()}, "
        f"holdout starts {split.holdout_date_range[0].date()}."
    )

    if set(split.train.index) & set(split.holdout.index):
        raise ValueError("Train and holdout share rows.")
    confirmations.append("Train and holdout row sets are disjoint.")

    dates = pd.to_datetime(development_frame[columns.date]).to_numpy()
    for index, (train_positions, validation_positions) in enumerate(
        temporal_cv_splits(
            development_frame, date_column=columns.date, n_splits=config.split.n_cv_splits
        ),
        start=1,
    ):
        if dates[train_positions].max() > dates[validation_positions].min():
            raise ValueError(f"CV fold {index} trains on data later than it validates on.")
        if set(train_positions) & set(validation_positions):
            raise ValueError(f"CV fold {index} has overlapping train/validation indices.")
    confirmations.append(
        f"All {config.split.n_cv_splits} CV folds are forward-only and non-overlapping."
    )
    confirmations.append(
        "A fresh preprocessing pipeline is fitted per split; no fitted state crosses folds."
    )
    return confirmations
