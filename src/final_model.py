"""Final model assembly and log-target back-transformation.

Phase-6 measured a systematic under-pricing bias of **+$101.81** (t = 15.69) on
the holdout. That is the expected signature of a log-target model: the model is
fitted to the conditional mean of ``log(rate)``, so ``exp()`` returns the
conditional *median*, which sits below the mean for a right-skewed target.

Duan's (1983) smearing estimator is the standard non-parametric correction. It
rescales predictions by the mean of the exponentiated training residuals::

    S = mean(exp(log(y_i) - log(y_hat_i)))
    corrected = exp(log_prediction) * S

Because ``S > 0`` always, the correction cannot produce a non-positive
prediction, so it is safe under the constraint ``score.py`` enforces.

Whether it is *applied* is decided empirically in
:func:`evaluate_back_transformations`, not assumed: smearing targets the
conditional mean, while MAE is minimised by the conditional median. The two
objectives genuinely conflict, so the choice is made on measured holdout
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.logger import get_logger
from src.metrics import RegressionMetrics, compute_metrics
from src.pipeline import PreprocessingPipelineConfig, build_preprocessing_pipeline

logger = get_logger(__name__)


@dataclass(frozen=True)
class BackTransformChoice:
    """The outcome of the smearing decision.

    Attributes:
        use_smearing: Whether to apply the correction to final predictions.
        smearing_factor: Duan's factor measured on the fitting data.
        rationale: Human-readable justification for the decision.
        raw_metrics: Holdout metrics without correction.
        corrected_metrics: Holdout metrics with correction.
    """

    use_smearing: bool
    smearing_factor: float
    rationale: str
    raw_metrics: RegressionMetrics
    corrected_metrics: RegressionMetrics


def build_catboost_pipeline(
    hyperparameters: dict[str, Any],
    *,
    pipeline_config: PreprocessingPipelineConfig,
    seed: int,
) -> Pipeline:
    """Rebuild the Phase-5 winning pipeline with its selected hyperparameters.

    Args:
        hyperparameters: The tuned CatBoost parameters from Phase 5.
        pipeline_config: Phase-3 preprocessing configuration.
        seed: Random seed.

    Returns:
        An unfitted pipeline identical in structure to the Phase-5 winner.
    """
    from catboost import CatBoostRegressor

    regressor = CatBoostRegressor(
        random_state=seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
        **hyperparameters,
    )
    model = TransformedTargetRegressor(
        regressor=regressor, func=np.log, inverse_func=np.exp, check_inverse=False
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(pipeline_config)),
            ("model", model),
        ]
    )


def compute_smearing_factor(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray
) -> float:
    """Compute Duan's smearing factor from a fitted model's own fitting data.

    Args:
        pipeline: A fitted pipeline whose ``predict`` returns dollar rates.
        X: The feature frame the pipeline was fitted on.
        y: The corresponding observed rates.

    Returns:
        The smearing factor, strictly positive.

    Raises:
        ValueError: If the factor is not finite and positive.
    """
    predictions = np.asarray(pipeline.predict(X), dtype=float)
    observed = np.asarray(y, dtype=float)
    # Residuals in log space, which is where the model was actually fitted.
    log_residuals = np.log(observed) - np.log(np.clip(predictions, 1e-9, None))
    factor = float(np.mean(np.exp(log_residuals)))

    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"Smearing factor is not usable: {factor!r}")

    logger.info("Duan smearing factor: %.6f (implies a %+.2f%% shift)", factor, (factor - 1) * 100)
    return factor


def evaluate_back_transformations(
    pipeline: Pipeline,
    *,
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    holdout_features: pd.DataFrame,
    holdout_target: np.ndarray,
) -> BackTransformChoice:
    """Decide empirically whether to apply the smearing correction.

    The factor is derived from the model's fitting data and judged on the
    untouched holdout, so the decision itself does not leak.

    Selection rule: MAE is the project's headline metric (carried from Phases 4
    and 5), so the correction is adopted only if it does not worsen MAE. Bias
    and RMSE are reported either way.

    Args:
        pipeline: A pipeline already fitted on ``fit_features``.
        fit_features: Features the pipeline was fitted on.
        fit_target: Observed rates for the fitting rows.
        holdout_features: Untouched holdout features.
        holdout_target: Observed holdout rates.

    Returns:
        A populated :class:`BackTransformChoice`.
    """
    factor = compute_smearing_factor(pipeline, fit_features, fit_target)

    raw_predictions = np.asarray(pipeline.predict(holdout_features), dtype=float)
    corrected_predictions = raw_predictions * factor

    raw_metrics = compute_metrics(holdout_target, raw_predictions)
    corrected_metrics = compute_metrics(holdout_target, corrected_predictions)

    raw_bias = float(np.mean(holdout_target - raw_predictions))
    corrected_bias = float(np.mean(holdout_target - corrected_predictions))

    improves_mae = corrected_metrics.mae <= raw_metrics.mae
    use_smearing = improves_mae

    if use_smearing:
        rationale = (
            f"Adopted. Smearing (factor {factor:.4f}) reduces holdout MAE from "
            f"${raw_metrics.mae:,.2f} to ${corrected_metrics.mae:,.2f} and cuts mean bias from "
            f"${raw_bias:,.2f} to ${corrected_bias:,.2f}."
        )
    else:
        rationale = (
            f"Rejected. Smearing (factor {factor:.4f}) would cut mean bias from ${raw_bias:,.2f} "
            f"to ${corrected_bias:,.2f}, but it raises holdout MAE from ${raw_metrics.mae:,.2f} to "
            f"${corrected_metrics.mae:,.2f}. This is the expected conflict: smearing targets the "
            "conditional mean while MAE is minimised by the conditional median. Since MAE is this "
            "project's headline metric, the uncorrected median-style prediction is kept and the "
            "residual bias is documented rather than traded for a worse headline error."
        )

    logger.info("Smearing decision: %s", "ADOPTED" if use_smearing else "REJECTED")
    logger.info(
        "  raw       MAE=%.2f RMSE=%.2f bias=%+.2f", raw_metrics.mae, raw_metrics.rmse, raw_bias
    )
    logger.info(
        "  corrected MAE=%.2f RMSE=%.2f bias=%+.2f",
        corrected_metrics.mae,
        corrected_metrics.rmse,
        corrected_bias,
    )

    return BackTransformChoice(
        use_smearing=use_smearing,
        smearing_factor=factor,
        rationale=rationale,
        raw_metrics=raw_metrics,
        corrected_metrics=corrected_metrics,
    )


def apply_back_transform(
    predictions: np.ndarray, choice: BackTransformChoice, *, floor: float = 1.0
) -> np.ndarray:
    """Apply the chosen back-transformation and enforce positivity.

    Args:
        predictions: Raw dollar predictions from the pipeline.
        choice: The decision from :func:`evaluate_back_transformations`.
        floor: Minimum permitted prediction.

    Returns:
        Final predictions, guaranteed finite and strictly positive.
    """
    values = np.asarray(predictions, dtype=float)
    if choice.use_smearing:
        values = values * choice.smearing_factor
    values = np.nan_to_num(values, nan=floor, posinf=floor, neginf=floor)
    return np.clip(values, floor, None)
