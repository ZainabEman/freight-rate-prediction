"""Leakage-safe randomised hyperparameter search.

The search operates on a single composed estimator::

    Pipeline([
        ("preprocess", <Phase-3 preprocessing pipeline>),
        ("model",      TransformedTargetRegressor(<estimator>, log/exp)),
    ])

Because preprocessing is a *step inside* the searched pipeline,
``RandomizedSearchCV`` clones and refits it independently within every
cross-validation fold. No imputation median, scaler statistic, one-hot
vocabulary or unknown-category set is ever learned from data that lies in the
future relative to the fold being scored. This is the same guarantee Phase 4
enforced manually, obtained structurally here.

Folds come from :class:`~sklearn.model_selection.TimeSeriesSplit` applied to
date-sorted rows, so every validation fold is strictly forward of its training
fold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.advanced_models import ModelSpec, log_target_functions
from src.config import AppConfig
from src.logger import get_logger
from src.pipeline import PreprocessingPipelineConfig, build_preprocessing_pipeline
from src.splitting import sort_by_date

logger = get_logger(__name__)

# Prefix applied to every search-space key so it addresses the wrapped estimator
# inside Pipeline -> TransformedTargetRegressor -> regressor.
_PARAM_PREFIX = "model__regressor__"


@dataclass
class TuningResult:
    """Outcome of a randomised search for one model."""

    name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    cv_best_mae: float
    n_candidates: int
    search_seconds: float


def build_full_pipeline(
    spec: ModelSpec,
    *,
    pipeline_config: PreprocessingPipelineConfig,
    seed: int,
) -> Pipeline:
    """Compose preprocessing and the model into one searchable estimator.

    Args:
        spec: The model specification.
        pipeline_config: Phase-3 preprocessing configuration.
        seed: Random seed for the estimator.

    Returns:
        An unfitted pipeline whose ``predict`` returns original-scale rates.
    """
    forward, inverse = log_target_functions()
    model = TransformedTargetRegressor(
        regressor=spec.factory(seed),
        func=forward,
        inverse_func=inverse,
        check_inverse=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(pipeline_config)),
            ("model", model),
        ]
    )


def tune_model(
    spec: ModelSpec,
    *,
    train_frame: pd.DataFrame,
    config: AppConfig,
    pipeline_config: PreprocessingPipelineConfig,
    n_splits: int = 3,
) -> TuningResult:
    """Run a randomised search for one model on the training window.

    Args:
        spec: The model specification and its search space.
        train_frame: Labelled rows strictly before the holdout boundary.
        config: Application configuration.
        pipeline_config: Phase-3 preprocessing configuration.
        n_splits: Number of expanding-window CV folds inside the search.

    Returns:
        A populated :class:`TuningResult` whose ``best_estimator`` is already
        refitted on the whole ``train_frame``.
    """
    columns = config.columns
    ordered = sort_by_date(train_frame, date_column=columns.date)
    X = ordered[columns.raw_feature_columns]
    y = ordered[columns.target].to_numpy(dtype=float)

    estimator = build_full_pipeline(spec, pipeline_config=pipeline_config, seed=config.random_seed)
    search_space = {f"{_PARAM_PREFIX}{key}": values for key, values in spec.param_distributions.items()}

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space,
        n_iter=spec.n_iter,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=config.random_seed,
        # The estimators parallelise internally; keeping the search serial avoids
        # CPU oversubscription and keeps peak memory within the PRD's 4 GB budget.
        n_jobs=1,
        refit=True,
        error_score="raise",
        verbose=0,
    )

    logger.info(
        "Tuning %s: %d candidates x %d folds", spec.name, spec.n_iter, n_splits
    )
    started = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - started

    best_params = {
        key.replace(_PARAM_PREFIX, ""): value for key, value in search.best_params_.items()
    }
    cv_best_mae = float(-search.best_score_)

    logger.info(
        "Tuned %s in %.0fs | CV MAE=%.2f | best params: %s",
        spec.name,
        elapsed,
        cv_best_mae,
        best_params,
    )

    return TuningResult(
        name=spec.name,
        best_estimator=search.best_estimator_,
        best_params=best_params,
        cv_best_mae=cv_best_mae,
        n_candidates=spec.n_iter,
        search_seconds=elapsed,
    )


def predict_original_scale(estimator: Pipeline, frame: pd.DataFrame, *, config: AppConfig) -> np.ndarray:
    """Predict rates in the original dollar scale.

    Args:
        estimator: A fitted full pipeline.
        frame: Raw feature frame.
        config: Application configuration.

    Returns:
        Predicted rates.
    """
    return np.asarray(
        estimator.predict(frame[config.columns.raw_feature_columns]), dtype=float
    )
