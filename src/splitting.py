"""Time-based train/holdout splitting utilities.

Audit findings M-1 and R-3 drive this module. ``train_test.csv`` covers
2025-01-01..2025-10-31 and ``validation.csv`` covers 2025-11-01..2025-12-31 with
zero overlap, so the real task is forward extrapolation. A random split would
measure interpolation skill and report optimistic numbers that collapse on the
actual scoring window.

Nothing here trains or evaluates a model; these are the split primitives Phase 4
will consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TemporalSplit:
    """Result of a time-based split.

    Attributes:
        train: Rows strictly before the holdout boundary.
        holdout: Rows on or after the holdout boundary.
        holdout_start: The boundary timestamp used.
        train_date_range: ``(min, max)`` dates present in ``train``.
        holdout_date_range: ``(min, max)`` dates present in ``holdout``.
    """

    train: pd.DataFrame
    holdout: pd.DataFrame
    holdout_start: pd.Timestamp
    train_date_range: tuple[pd.Timestamp, pd.Timestamp]
    holdout_date_range: tuple[pd.Timestamp, pd.Timestamp]

    @property
    def summary(self) -> dict[str, object]:
        """Return a serialisable summary of the split."""
        return {
            "holdout_start": str(self.holdout_start.date()),
            "train_rows": len(self.train),
            "holdout_rows": len(self.holdout),
            "train_date_min": str(self.train_date_range[0].date()),
            "train_date_max": str(self.train_date_range[1].date()),
            "holdout_date_min": str(self.holdout_date_range[0].date()),
            "holdout_date_max": str(self.holdout_date_range[1].date()),
        }


def temporal_train_holdout_split(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    holdout_start: str | pd.Timestamp = "2025-09-01",
) -> TemporalSplit:
    """Split a frame into a past training block and a future holdout block.

    The holdout is the final contiguous slice of the development window, which
    mirrors the real train -> scoring gap rather than sampling across it.

    Args:
        frame: Development data containing ``date_column``.
        date_column: Name of the date column.
        holdout_start: First date belonging to the holdout.

    Returns:
        A :class:`TemporalSplit`.

    Raises:
        KeyError: If ``date_column`` is absent.
        ValueError: If any date fails to parse, or either side is empty.
    """
    if date_column not in frame.columns:
        raise KeyError(f"date column {date_column!r} not found in frame")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    unparsed = int(dates.isna().sum())
    if unparsed:
        raise ValueError(f"{unparsed} unparseable value(s) in column {date_column!r}")

    boundary = pd.Timestamp(holdout_start)
    train_mask = dates < boundary
    holdout_mask = ~train_mask

    if not train_mask.any():
        raise ValueError(f"Temporal split produced an empty training set at boundary {boundary}")
    if not holdout_mask.any():
        raise ValueError(f"Temporal split produced an empty holdout set at boundary {boundary}")

    train = frame.loc[train_mask]
    holdout = frame.loc[holdout_mask]

    split = TemporalSplit(
        train=train,
        holdout=holdout,
        holdout_start=boundary,
        train_date_range=(dates[train_mask].min(), dates[train_mask].max()),
        holdout_date_range=(dates[holdout_mask].min(), dates[holdout_mask].max()),
    )

    assert_temporal_split_valid(split)
    logger.info(
        "Temporal split at %s: train=%d rows (%s..%s), holdout=%d rows (%s..%s)",
        boundary.date(),
        len(train),
        split.train_date_range[0].date(),
        split.train_date_range[1].date(),
        len(holdout),
        split.holdout_date_range[0].date(),
        split.holdout_date_range[1].date(),
    )
    return split


def assert_temporal_split_valid(split: TemporalSplit) -> None:
    """Verify the split has no temporal overlap between train and holdout.

    Args:
        split: The split to check.

    Raises:
        ValueError: If the training block extends into the holdout window.
    """
    if split.train_date_range[1] >= split.holdout_date_range[0]:
        raise ValueError(
            "Temporal split overlaps: train ends "
            f"{split.train_date_range[1].date()} but holdout starts "
            f"{split.holdout_date_range[0].date()}"
        )


def temporal_cv_splits(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window cross-validation indices ordered by date.

    Rows are sorted by date before splitting so that every validation fold lies
    strictly after its training fold, which is the CV analogue of the holdout
    design above.

    Args:
        frame: Development data containing ``date_column``.
        date_column: Name of the date column.
        n_splits: Number of expanding-window folds.

    Yields:
        ``(train_positions, validation_positions)`` as positional indices into
        the *date-sorted* frame.

    Raises:
        KeyError: If ``date_column`` is absent.
    """
    if date_column not in frame.columns:
        raise KeyError(f"date column {date_column!r} not found in frame")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    order = np.argsort(dates.to_numpy(), kind="stable")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_positions, validation_positions in splitter.split(order):
        yield order[train_positions], order[validation_positions]


def sort_by_date(frame: pd.DataFrame, *, date_column: str = "date") -> pd.DataFrame:
    """Return the frame ordered by date using a stable sort.

    Args:
        frame: Frame to order.
        date_column: Name of the date column.

    Returns:
        A date-ordered copy of ``frame``.
    """
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    return frame.iloc[np.argsort(dates.to_numpy(), kind="stable")]
