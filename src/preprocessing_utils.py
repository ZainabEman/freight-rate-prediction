"""Shared schema-validation helpers.

The former ``FeatureColumns`` / ``default_feature_columns`` /
``strip_whitespace_in_place`` helpers were removed in Phase 3: column roles are
now owned by :mod:`src.config` (loaded from ``config/config.yaml``) and
whitespace stripping is performed inside the pipeline by
:class:`~src.transformers.RawDataCleaner`. Keeping duplicate definitions of the
same schema in two places was a drift risk.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def ensure_required_columns(
    df: pd.DataFrame, *, required: Sequence[str], label: str
) -> None:
    """Ensure a DataFrame contains every required column.

    Args:
        df: Frame to check.
        required: Column names that must be present.
        label: Human-readable dataset name used in the error message.

    Raises:
        ValueError: If any required column is absent.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}. Present columns={list(df.columns)}"
        )
