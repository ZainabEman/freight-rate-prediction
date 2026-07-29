"""Builders for the fitted (statistical) preprocessing components.

Only the objects created here learn anything from the data - imputation
medians, scaler statistics and one-hot vocabularies. Confining all fitted state
to this stage is what makes the leakage audit in
:func:`src.pipeline.assert_no_leakage` tractable.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

_SCALERS = {
    "robust": RobustScaler,
    "standard": StandardScaler,
}


@dataclass(frozen=True)
class NumericPreprocessingConfig:
    """Numeric preprocessing decisions derived from the audit.

    Evidence:
      * ``weight`` is missing in 0.63% of training rows and 1.38% of scoring
        rows; ``market_index`` in 0.78% and 2.08% respectively. Median
        imputation is used because both distributions are skewed and the median
        is unaffected by the extreme tails.
      * ``weight`` remains heavy-tailed after the sign repair applied in
        :class:`~src.transformers.RawDataCleaner`, so it is scaled with
        :class:`~sklearn.preprocessing.RobustScaler` (median/IQR) rather than
        mean/standard deviation.
    """

    missing_strategy: str = "median"
    weight_scaler: str = "robust"
    default_scaler: str = "standard"


@dataclass(frozen=True)
class CategoricalPreprocessingConfig:
    """Categorical preprocessing decisions derived from the audit.

    ``handle_unknown="ignore"`` maps unseen categories to an all-zero vector.
    That is acceptable *only* because the unknown state is separately made
    explicit by :class:`~src.transformers.UnseenCategoryIndicator`; on its own
    it would silently discard ~6% of scoring rows' geography (audit finding
    M-4).

    ``sparse_output=False`` keeps the output dense so that feature names survive
    into a DataFrame. Explosion is bounded: pickup/delivery have 64 levels each
    and equipment has 3.
    """

    handle_unknown: str = "ignore"
    sparse_output: bool = False
    min_frequency: float | int | None = None


def build_numeric_imputer(strategy: str = "median") -> SimpleImputer:
    """Build an imputer for numeric columns.

    Args:
        strategy: A strategy accepted by :class:`~sklearn.impute.SimpleImputer`.

    Returns:
        A configured, unfitted imputer.
    """
    return SimpleImputer(strategy=strategy)


def build_scaler(kind: str):
    """Build a scaler by name.

    Args:
        kind: Either ``"robust"`` or ``"standard"``.

    Returns:
        A configured, unfitted scaler.

    Raises:
        ValueError: If ``kind`` is not a supported scaler name. The previous
            implementation silently fell back to a default on any unknown
            value, which hid configuration typos.
    """
    try:
        return _SCALERS[kind]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported scaler {kind!r}. Expected one of {sorted(_SCALERS)}."
        ) from exc


def build_onehot_encoder(
    *,
    handle_unknown: str = "ignore",
    sparse_output: bool = False,
    min_frequency: float | int | None = None,
) -> OneHotEncoder:
    """Build the categorical one-hot encoder.

    Args:
        handle_unknown: Unseen-category policy.
        sparse_output: Whether to emit a sparse matrix.
        min_frequency: Optional rare-category bucketing threshold.

    Returns:
        A configured, unfitted encoder.
    """
    return OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse_output=sparse_output,
        min_frequency=min_frequency,
    )
