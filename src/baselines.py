"""Baseline regression models.

Two families are defined:

* **Raw-input baselines** consume the raw feature frame directly. These are the
  business-sense reference points a freight desk would already have - a flat
  average, and a distance x rate-per-mile quote. They deliberately bypass the
  preprocessing pipeline because their whole value is being simple enough to
  reason about without one.
* **Processed-input baselines** consume the Phase-3 feature matrix. These are
  linear models that establish what a well-specified but non-boosted learner
  achieves on the engineered features.

No hyperparameter search is performed anywhere in this module; every estimator
uses fixed, documented settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_is_fitted

InputKind = Literal["raw", "processed"]
TargetTransform = Literal["identity", "log"]


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predict the training mean rate for every load."""

    def fit(self, X: Any, y: np.ndarray) -> "MeanBaseline":
        """Learn the training mean."""
        self.constant_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the learned constant for every row."""
        check_is_fitted(self, "constant_")
        return np.full(len(X), self.constant_, dtype=float)


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training median rate for every load.

    Included alongside the mean because ``posted_rate`` is right-skewed
    (mean 2,374 vs median 2,031), so the two constants differ materially and
    the median is the stronger reference under MAE.
    """

    def fit(self, X: Any, y: np.ndarray) -> "MedianBaseline":
        """Learn the training median."""
        self.constant_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the learned constant for every row."""
        check_is_fitted(self, "constant_")
        return np.full(len(X), self.constant_, dtype=float)


class RatePerMileBaseline(BaseEstimator, RegressorMixin):
    """Quote ``distance x median rate-per-mile``, optionally per equipment type.

    This is the domain baseline: it is how a broker prices a lane by hand.
    It is a meaningful bar because ``corr(distance, posted_rate) = 0.909``, and
    rate-per-mile differs by equipment (verified medians: Dry Van 2.115,
    Flatbed 2.295, Reefer 2.383 dollars per mile).

    The median rather than the mean is used per group because rate-per-mile has
    a long right tail (max 14.13 against a median of 2.15).
    """

    def __init__(self, *, distance_column: str = "distance", group_column: str | None = "equipment"):
        self.distance_column = distance_column
        self.group_column = group_column

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RatePerMileBaseline":
        """Learn median rate-per-mile overall and per group.

        Raises:
            KeyError: If a required column is absent from ``X``.
        """
        if self.distance_column not in X.columns:
            raise KeyError(f"RatePerMileBaseline requires column {self.distance_column!r}")

        distance = pd.to_numeric(X[self.distance_column], errors="coerce").to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_per_mile = np.divide(
                target, distance, out=np.full_like(target, np.nan), where=distance > 0
            )

        self.global_rate_per_mile_ = float(np.nanmedian(rate_per_mile))

        self.group_rate_per_mile_: dict[str, float] = {}
        if self.group_column is not None:
            if self.group_column not in X.columns:
                raise KeyError(f"RatePerMileBaseline requires column {self.group_column!r}")
            frame = pd.DataFrame(
                {"group": X[self.group_column].astype("string").str.strip(), "rpm": rate_per_mile}
            )
            self.group_rate_per_mile_ = {
                str(name): float(np.nanmedian(part["rpm"]))
                for name, part in frame.groupby("group", dropna=True)
            }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ``distance x rate-per-mile`` using the group rate when known."""
        check_is_fitted(self, "global_rate_per_mile_")
        distance = pd.to_numeric(X[self.distance_column], errors="coerce").to_numpy(dtype=float)

        rates = np.full(len(X), self.global_rate_per_mile_, dtype=float)
        if self.group_column is not None and self.group_rate_per_mile_:
            groups = X[self.group_column].astype("string").str.strip()
            # Unseen equipment types fall back to the global median rather than
            # producing NaN.
            mapped = groups.map(self.group_rate_per_mile_).astype(float)
            rates = mapped.fillna(self.global_rate_per_mile_).to_numpy(dtype=float)

        return np.nan_to_num(distance, nan=0.0) * rates


@dataclass(frozen=True)
class BaselineSpec:
    """A baseline model together with how it must be fed and scored.

    Attributes:
        name: Display name used in reports.
        estimator: An unfitted scikit-learn compatible regressor.
        input_kind: ``"raw"`` to receive the raw feature frame, ``"processed"``
            to receive the Phase-3 feature matrix.
        target_transform: ``"identity"`` or ``"log"``. Log training addresses
            the right-skewed target and guarantees positive predictions after
            inversion.
        description: Short rationale shown in the comparison report.
    """

    name: str
    estimator: BaseEstimator
    input_kind: InputKind
    target_transform: TargetTransform
    description: str


def default_baselines() -> list[BaselineSpec]:
    """Return the fixed set of Phase-4 baselines.

    All settings are defaults or directly justified by EDA; nothing here is
    tuned. ``Ridge`` uses scikit-learn's default ``alpha=1.0``.

    Returns:
        The ordered list of baseline specifications.
    """
    return [
        BaselineSpec(
            name="Mean (constant)",
            estimator=MeanBaseline(),
            input_kind="raw",
            target_transform="identity",
            description="Predicts the training mean rate. Floor for R2 by construction.",
        ),
        BaselineSpec(
            name="Median (constant)",
            estimator=MedianBaseline(),
            input_kind="raw",
            target_transform="identity",
            description="Predicts the training median; stronger than the mean under MAE.",
        ),
        BaselineSpec(
            name="Rate-per-mile (global)",
            estimator=RatePerMileBaseline(group_column=None),
            input_kind="raw",
            target_transform="identity",
            description="distance x global median rate-per-mile.",
        ),
        BaselineSpec(
            name="Rate-per-mile (by equipment)",
            estimator=RatePerMileBaseline(group_column="equipment"),
            input_kind="raw",
            target_transform="identity",
            description="distance x median rate-per-mile within equipment type. Domain baseline.",
        ),
        BaselineSpec(
            name="LinearRegression",
            estimator=LinearRegression(),
            input_kind="processed",
            target_transform="identity",
            description="Ordinary least squares on the 156 engineered features.",
        ),
        BaselineSpec(
            name="Ridge (alpha=1.0)",
            estimator=Ridge(alpha=1.0, random_state=None),
            input_kind="processed",
            target_transform="identity",
            description="L2-regularised least squares; stabilises the 131 sparse one-hot columns.",
        ),
        BaselineSpec(
            name="Ridge (alpha=1.0, log target)",
            estimator=Ridge(alpha=1.0, random_state=None),
            input_kind="processed",
            target_transform="log",
            description=(
                "Ridge on log(rate). Addresses the right-skewed target and cannot emit "
                "non-positive predictions after inversion."
            ),
        ),
    ]
