"""Tests for the time-based split utilities (audit findings M-1 / R-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.splitting import (
    sort_by_date,
    temporal_cv_splits,
    temporal_train_holdout_split,
)


def test_split_has_no_temporal_overlap(train_raw, config) -> None:
    """The whole point of the split: holdout must lie strictly in the future."""
    split = temporal_train_holdout_split(
        train_raw, date_column=config.columns.date, holdout_start=config.split.holdout_start
    )
    assert split.train_date_range[1] < split.holdout_date_range[0]


def test_split_partitions_every_row(train_raw, config) -> None:
    """No row may be lost or duplicated by the split."""
    split = temporal_train_holdout_split(
        train_raw, date_column=config.columns.date, holdout_start=config.split.holdout_start
    )
    assert len(split.train) + len(split.holdout) == len(train_raw)
    overlap = set(split.train.index) & set(split.holdout.index)
    assert not overlap


def test_split_boundary_is_respected(train_raw, config) -> None:
    """Rows on the boundary date belong to the holdout."""
    boundary = pd.Timestamp(config.split.holdout_start)
    split = temporal_train_holdout_split(
        train_raw, date_column=config.columns.date, holdout_start=boundary
    )
    assert pd.to_datetime(split.train[config.columns.date]).max() < boundary
    assert pd.to_datetime(split.holdout[config.columns.date]).min() >= boundary


def test_split_mirrors_the_real_scoring_gap(train_raw, validation_raw, config) -> None:
    """The holdout must be a forward block, like the real validation window."""
    split = temporal_train_holdout_split(
        train_raw, date_column=config.columns.date, holdout_start=config.split.holdout_start
    )
    validation_start = pd.to_datetime(validation_raw[config.columns.date]).min()
    assert split.holdout_date_range[1] < validation_start


def test_empty_holdout_raises(train_raw, config) -> None:
    """A boundary past the data must fail loudly, not return an empty holdout."""
    with pytest.raises(ValueError, match="empty holdout"):
        temporal_train_holdout_split(
            train_raw, date_column=config.columns.date, holdout_start="2030-01-01"
        )


def test_empty_train_raises(train_raw, config) -> None:
    """A boundary before the data must fail loudly."""
    with pytest.raises(ValueError, match="empty training set"):
        temporal_train_holdout_split(
            train_raw, date_column=config.columns.date, holdout_start="2020-01-01"
        )


def test_missing_date_column_raises(config) -> None:
    """A frame without the date column cannot be split temporally."""
    with pytest.raises(KeyError):
        temporal_train_holdout_split(pd.DataFrame({"a": [1]}), date_column=config.columns.date)


def test_cv_folds_are_forward_only(train_sample, config) -> None:
    """Every validation fold must lie strictly after its training fold."""
    dates = pd.to_datetime(train_sample[config.columns.date]).to_numpy()
    folds = list(
        temporal_cv_splits(
            train_sample, date_column=config.columns.date, n_splits=config.split.n_cv_splits
        )
    )
    assert len(folds) == config.split.n_cv_splits
    for train_positions, validation_positions in folds:
        assert dates[train_positions].max() <= dates[validation_positions].min()


def test_cv_folds_do_not_overlap(train_sample, config) -> None:
    """Training and validation indices within a fold must be disjoint."""
    for train_positions, validation_positions in temporal_cv_splits(
        train_sample, date_column=config.columns.date, n_splits=3
    ):
        assert not set(train_positions) & set(validation_positions)


def test_sort_by_date_is_monotonic(train_sample, config) -> None:
    """The sort helper must produce a non-decreasing date sequence."""
    ordered = sort_by_date(train_sample, date_column=config.columns.date)
    dates = pd.to_datetime(ordered[config.columns.date])
    assert dates.is_monotonic_increasing


def test_split_summary_is_serialisable(train_raw, config) -> None:
    """The summary feeds the report and metadata, so it must be plain data."""
    split = temporal_train_holdout_split(
        train_raw, date_column=config.columns.date, holdout_start=config.split.holdout_start
    )
    summary = split.summary
    assert summary["train_rows"] + summary["holdout_rows"] == len(train_raw)
    assert all(isinstance(value, (str, int, np.integer)) for value in summary.values())
