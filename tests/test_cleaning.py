"""Tests for the row-local cleaning stage (audit finding M-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.transformers import MissingFlagger, RawDataCleaner


@pytest.fixture()
def cleaner() -> RawDataCleaner:
    """A cleaner configured like the production pipeline."""
    return RawDataCleaner(
        categorical_columns=["pickup", "delivery", "equipment"],
        weight_min=5_000.0,
        weight_max=47_500.0,
    )


def _frame(weights: list[float]) -> pd.DataFrame:
    """Build a minimal raw-shaped frame with the given weight column."""
    size = len(weights)
    repeats = size // 2 + 1
    return pd.DataFrame(
        {
            "pickup": (["  Lexington ", "Atlanta"] * repeats)[:size],
            "delivery": (["Fort Wayne", " Mobile"] * repeats)[:size],
            "equipment": (["Dry Van", "Reefer"] * repeats)[:size],
            "weight": weights,
        }
    )


def test_negative_weights_are_sign_repaired(cleaner: RawDataCleaner) -> None:
    """abs() repair converts sign-flipped weights to their true magnitude."""
    frame = _frame([-32_000.0, 25_000.0, -47_500.0, 10_000.0])
    cleaned = cleaner.fit(frame).transform(frame)
    assert (cleaned["weight"] >= 0).all()
    assert cleaned["weight"].tolist() == [32_000.0, 25_000.0, 47_500.0, 10_000.0]


def test_weights_are_clipped_to_physical_envelope(cleaner: RawDataCleaner) -> None:
    """Values outside the observed envelope are clipped, not dropped."""
    frame = _frame([1_000.0, 99_000.0])
    cleaned = cleaner.fit(frame).transform(frame)
    assert cleaned["weight"].tolist() == [5_000.0, 47_500.0]


def test_missing_weight_is_preserved_for_the_imputer(cleaner: RawDataCleaner) -> None:
    """NaN must survive cleaning so the fitted imputer handles it."""
    frame = _frame([np.nan, 30_000.0])
    cleaned = cleaner.fit(frame).transform(frame)
    assert bool(cleaned["weight"].isna().iloc[0])


def test_categorical_whitespace_is_stripped(cleaner: RawDataCleaner) -> None:
    """Leading/trailing whitespace is removed; case is left untouched."""
    frame = _frame([30_000.0, 31_000.0])
    cleaned = cleaner.fit(frame).transform(frame)
    assert cleaned["pickup"].tolist() == ["Lexington", "Atlanta"]
    assert cleaned["delivery"].tolist() == ["Fort Wayne", "Mobile"]


def test_unsupported_repair_strategy_raises() -> None:
    """An unknown strategy fails loudly rather than silently defaulting."""
    cleaner = RawDataCleaner(categorical_columns=["pickup"], weight_sign_repair="magic")
    frame = _frame([30_000.0, 31_000.0])
    with pytest.raises(ValueError, match="Unsupported weight_sign_repair"):
        cleaner.fit(frame).transform(frame)


def test_cleaning_is_row_local_and_order_independent(cleaner: RawDataCleaner) -> None:
    """Cleaning a row must not depend on the other rows present."""
    frame = _frame([-32_000.0, 25_000.0, np.nan, 10_000.0])
    full = cleaner.fit(frame).transform(frame)
    single = cleaner.fit(frame).transform(frame.iloc[[0]])
    assert single["weight"].iloc[0] == full["weight"].iloc[0]


def test_missing_flagger_marks_nulls_before_imputation() -> None:
    """Indicators capture missingness that imputation would otherwise erase."""
    frame = pd.DataFrame({"weight": [np.nan, 1.0], "market_index": [1.0, np.nan]})
    flagger = MissingFlagger(columns=["weight", "market_index"]).fit(frame)
    out = flagger.transform(frame)
    assert list(out.columns) == ["weight_is_missing", "market_index_is_missing"]
    assert out["weight_is_missing"].tolist() == [1.0, 0.0]
    assert out["market_index_is_missing"].tolist() == [0.0, 1.0]


def test_real_training_data_has_no_negative_weights_after_cleaning(train_sample, config) -> None:
    """End-to-end check against the real dataset."""
    cleaner = RawDataCleaner(categorical_columns=config.columns.categorical)
    frame = train_sample[config.columns.raw_feature_columns]
    cleaned = cleaner.fit(frame).transform(frame)
    assert not (cleaned["weight"] < 0).any()
