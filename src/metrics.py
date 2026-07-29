"""Regression metrics for the freight rate task.

The four headline metrics required by the PRD are MAE, RMSE, R2 and MAPE.
MAPE is well defined here because ``posted_rate`` is strictly positive in the
development data (minimum 57.22), so there is no division-by-zero case to
special-case away.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Predictions are clipped to this floor before scoring. score.py rejects any
# non-positive predicted_rate, so a model that can emit them (an unconstrained
# linear fit, for instance) must be evaluated under the same constraint it will
# face at submission time.
PREDICTION_FLOOR = 1.0


@dataclass(frozen=True)
class RegressionMetrics:
    """Headline regression metrics for a single evaluation."""

    mae: float
    rmse: float
    r2: float
    mape: float
    n: int

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain dictionary."""
        return asdict(self)


def clip_predictions(predictions: np.ndarray, *, floor: float = PREDICTION_FLOOR) -> np.ndarray:
    """Clip predictions to the positive range required by ``score.py``.

    Args:
        predictions: Raw model output.
        floor: Minimum allowed prediction.

    Returns:
        Predictions with non-finite values replaced and a positive floor applied.
    """
    values = np.asarray(predictions, dtype=float)
    values = np.nan_to_num(values, nan=floor, posinf=floor, neginf=floor)
    return np.clip(values, floor, None)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in dollars."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error in dollars."""
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the mean predictor."""
    truth = np.asarray(y_true, dtype=float)
    residual_ss = float(np.sum((truth - np.asarray(y_pred, dtype=float)) ** 2))
    total_ss = float(np.sum((truth - truth.mean()) ** 2))
    if total_ss == 0.0:
        return float("nan")
    return 1.0 - residual_ss / total_ss


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, expressed as a percentage.

    Raises:
        ValueError: If any true value is zero, which would make MAPE undefined.
    """
    truth = np.asarray(y_true, dtype=float)
    if np.any(truth == 0.0):
        raise ValueError("MAPE is undefined when any true value is zero.")
    return float(
        100.0 * np.mean(np.abs((truth - np.asarray(y_pred, dtype=float)) / truth))
    )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, floor: float = PREDICTION_FLOOR
) -> RegressionMetrics:
    """Compute all headline metrics for one set of predictions.

    Args:
        y_true: Observed rates.
        y_pred: Predicted rates (clipped to ``floor`` before scoring).
        floor: Positive prediction floor.

    Returns:
        A populated :class:`RegressionMetrics`.

    Raises:
        ValueError: If the input lengths differ.
    """
    truth = np.asarray(y_true, dtype=float)
    predicted = clip_predictions(y_pred, floor=floor)
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError(
            f"Length mismatch between y_true ({truth.shape[0]}) and y_pred ({predicted.shape[0]})"
        )

    return RegressionMetrics(
        mae=mean_absolute_error(truth, predicted),
        rmse=root_mean_squared_error(truth, predicted),
        r2=r2_score(truth, predicted),
        mape=mean_absolute_percentage_error(truth, predicted),
        n=int(truth.shape[0]),
    )
