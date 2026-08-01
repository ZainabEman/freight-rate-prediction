"""Advanced regression model definitions and their search spaces.

Optional gradient-boosting libraries are imported defensively: if XGBoost,
LightGBM or CatBoost is not installed the corresponding model is skipped with a
warning rather than crashing the run.

Search spaces are deliberately narrow. The PRD constrains the full pipeline to
run on laptop CPU in reasonable time, and every candidate here is evaluated
against a *fresh* preprocessing fit inside each CV fold, so the cost of a wide
space is multiplied by folds x candidates x models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """An advanced model together with its randomised search space.

    Attributes:
        name: Display name used in reports.
        factory: Callable returning a fresh unfitted estimator.
        param_distributions: Search space keyed *without* any pipeline prefix;
            the tuner adds the prefix it needs.
        n_iter: Number of randomised candidates to sample.
        notes: Short rationale shown in the comparison report.
    """

    name: str
    factory: Callable[[int], BaseEstimator]
    param_distributions: dict[str, Any] = field(default_factory=dict)
    n_iter: int = 8
    notes: str = ""


def _random_forest(seed: int) -> BaseEstimator:
    """Random forest tuned for wide, sparse one-hot input."""
    return RandomForestRegressor(random_state=seed, n_jobs=-1, bootstrap=True)


def _hist_gradient_boosting(seed: int) -> BaseEstimator:
    """Histogram-based gradient boosting.

    This is scikit-learn's gradient boosting implementation for large samples.
    ``GradientBoostingRegressor`` is the exact-split variant and is documented by
    scikit-learn as far slower for ``n_samples >= 10_000``; with 38,477 training
    rows and 156 features it would dominate the entire phase runtime for no
    accuracy benefit, so the histogram implementation is used instead.
    """
    return HistGradientBoostingRegressor(random_state=seed, early_stopping=False)


def _build_sklearn_specs(seed: int) -> list[ModelSpec]:
    """Specs for the two always-available scikit-learn ensembles."""
    return [
        ModelSpec(
            name="RandomForest",
            factory=_random_forest,
            param_distributions={
                "n_estimators": [200, 300, 400],
                "max_depth": [None, 16, 24],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.3, 0.5],
            },
            n_iter=6,
            notes="Bagged trees; variance reduction without boosting's sequential cost.",
        ),
        ModelSpec(
            name="HistGradientBoosting",
            factory=_hist_gradient_boosting,
            param_distributions={
                "learning_rate": [0.03, 0.05, 0.1],
                "max_iter": [300, 500, 800],
                "max_leaf_nodes": [31, 63, 127],
                "min_samples_leaf": [20, 40],
                "l2_regularization": [0.0, 1.0],
            },
            n_iter=8,
            notes="scikit-learn histogram gradient boosting (fast variant of GradientBoosting).",
        ),
    ]


def _build_xgboost_spec(seed: int) -> ModelSpec | None:
    """Spec for XGBoost, or ``None`` if the library is unavailable."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.warning("XGBoost is not installed; skipping this model.")
        return None

    def factory(random_state: int) -> BaseEstimator:
        return XGBRegressor(
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        )

    return ModelSpec(
        name="XGBoost",
        factory=factory,
        param_distributions={
            "n_estimators": [400, 700, 1000],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [6, 8, 10],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 5],
            "reg_lambda": [1.0, 5.0],
        },
        n_iter=8,
        notes="Histogram-based boosted trees with column subsampling.",
    )


def _build_lightgbm_spec(seed: int) -> ModelSpec | None:
    """Spec for LightGBM, or ``None`` if the library is unavailable."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.warning("LightGBM is not installed; skipping this model.")
        return None

    def factory(random_state: int) -> BaseEstimator:
        return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)

    return ModelSpec(
        name="LightGBM",
        factory=factory,
        param_distributions={
            "n_estimators": [400, 700, 1000],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [20, 40],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.0, 1.0],
        },
        n_iter=8,
        notes="Leaf-wise boosted trees; fastest of the boosting family on wide input.",
    )


def _build_catboost_spec(seed: int) -> ModelSpec | None:
    """Spec for CatBoost, or ``None`` if the library is unavailable."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        logger.warning("CatBoost is not installed; skipping this model.")
        return None

    def factory(random_state: int) -> BaseEstimator:
        return CatBoostRegressor(
            random_state=random_state,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )

    return ModelSpec(
        name="CatBoost",
        factory=factory,
        param_distributions={
            "iterations": [400, 700, 1000],
            "learning_rate": [0.03, 0.05, 0.1],
            "depth": [6, 8, 10],
            "l2_leaf_reg": [1.0, 3.0, 9.0],
        },
        n_iter=6,
        notes="Ordered boosting with symmetric trees.",
    )


def available_advanced_models(seed: int = 42) -> list[ModelSpec]:
    """Return every advanced model whose library is installed.

    Args:
        seed: Random seed passed to each estimator factory.

    Returns:
        Ordered list of available :class:`ModelSpec` objects. Missing optional
        libraries are skipped with a warning rather than raising.
    """
    specs = _build_sklearn_specs(seed)
    for builder in (_build_xgboost_spec, _build_lightgbm_spec, _build_catboost_spec):
        spec = builder(seed)
        if spec is not None:
            specs.append(spec)

    logger.info("Available advanced models: %s", ", ".join(spec.name for spec in specs))
    return specs


def log_target_functions() -> tuple[Callable, Callable]:
    """Return the forward and inverse target transforms used by every model.

    The log target is carried over from Phase 4, where it produced the best
    baseline (MAE 145.24 against 149.36 for the identity target). It addresses
    the right-skewed target (mean 2,374 against median 2,031, max 25,533) and
    guarantees strictly positive predictions after inversion, which is the
    constraint ``score.py`` enforces at submission time.

    Returns:
        ``(forward, inverse)`` callables.
    """
    return np.log, np.exp
