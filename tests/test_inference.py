"""Tests for the two inference paths (audit finding M-2).

The December path is the one the original pipeline could not serve at all, so
it gets the most coverage here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.inference import (
    DECEMBER_FIXED_DELIVERY,
    DECEMBER_FIXED_DISTANCE,
    DECEMBER_FIXED_EQUIPMENT,
    DECEMBER_FIXED_PICKUP,
    DECEMBER_FIXED_WEIGHT,
    DECEMBER_INPUT_COLUMNS,
    build_city_coordinate_lookup,
    build_daily_market_lookup,
    build_december_chart_inputs,
    enrich_reduced_frame,
)


@pytest.fixture(scope="module")
def coordinates(train_raw, validation_raw):
    """City coordinate lookup built from both raw datasets."""
    return build_city_coordinate_lookup(train_raw, validation_raw)


@pytest.fixture(scope="module")
def market_lookup(validation_raw):
    """Daily market context table covering the scoring window."""
    return build_daily_market_lookup(validation_raw)


def test_december_frame_matches_score_py_contract() -> None:
    """score.py pins the schema exactly; the reconstruction must match it."""
    frame = build_december_chart_inputs()
    assert list(frame.columns) == [*DECEMBER_INPUT_COLUMNS, "predicted_rate"]
    assert len(frame) == 31


def test_december_frame_covers_every_day_of_december() -> None:
    """One row per day from 2025-12-01 to 2025-12-31, no duplicates."""
    frame = build_december_chart_inputs()
    dates = pd.to_datetime(frame["date"])
    assert dates.min() == pd.Timestamp("2025-12-01")
    assert dates.max() == pd.Timestamp("2025-12-31")
    assert not dates.duplicated().any()


def test_december_fixed_inputs_are_constant() -> None:
    """Only the date may vary across the December scenario."""
    frame = build_december_chart_inputs()
    assert (frame["pickup"] == DECEMBER_FIXED_PICKUP).all()
    assert (frame["delivery"] == DECEMBER_FIXED_DELIVERY).all()
    assert (frame["distance"] == DECEMBER_FIXED_DISTANCE).all()
    assert (frame["equipment"] == DECEMBER_FIXED_EQUIPMENT).all()
    assert (frame["weight"] == DECEMBER_FIXED_WEIGHT).all()


def test_city_coordinates_are_deterministic(coordinates) -> None:
    """The lookup is exact only because each city has one coordinate pair."""
    assert coordinates.resolve("Lexington", role="pickup") == coordinates.resolve(
        "Lexington", role="pickup"
    )
    latitude, longitude = coordinates.resolve(DECEMBER_FIXED_PICKUP, role="pickup")
    assert -180.0 <= longitude <= 180.0
    assert -90.0 <= latitude <= 90.0


def test_december_cities_are_resolvable(coordinates) -> None:
    """Lexington and Fort Wayne must both be in the lookup."""
    assert coordinates.resolve(DECEMBER_FIXED_PICKUP, role="pickup")
    assert coordinates.resolve(DECEMBER_FIXED_DELIVERY, role="delivery")


def test_unknown_city_raises(coordinates) -> None:
    """A silent fallback here would corrupt the chart; it must raise."""
    with pytest.raises(KeyError):
        coordinates.resolve("Atlantis", role="pickup")


def test_invalid_role_raises(coordinates) -> None:
    """Guard against typo'd role arguments."""
    with pytest.raises(ValueError, match="role must be"):
        coordinates.resolve("Lexington", role="origin")


def test_market_lookup_covers_all_december_dates(market_lookup) -> None:
    """The December curve depends on this coverage being complete."""
    december = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    assert all(date in market_lookup.index for date in december)


def test_market_lookup_stays_inside_training_support(market_lookup, train_raw) -> None:
    """No extrapolation: December market values sit within the training range."""
    december = market_lookup.loc["2025-12-01":"2025-12-31", "market_index"]
    assert december.min() >= train_raw["market_index"].min()
    assert december.max() <= train_raw["market_index"].max()


def test_enrichment_restores_the_full_schema(coordinates, market_lookup, config) -> None:
    """The reduced frame must gain every column the pipeline requires."""
    reduced = build_december_chart_inputs().drop(columns=["predicted_rate"])
    enriched = enrich_reduced_frame(
        reduced, coordinates=coordinates, market_lookup=market_lookup
    )
    for column in config.columns.raw_feature_columns:
        assert column in enriched.columns
    assert len(enriched) == 31


def test_enrichment_produces_no_missing_values(coordinates, market_lookup) -> None:
    """Missing market context would flatten the December curve."""
    reduced = build_december_chart_inputs().drop(columns=["predicted_rate"])
    enriched = enrich_reduced_frame(
        reduced, coordinates=coordinates, market_lookup=market_lookup
    )
    assert not enriched.isna().any().any()


def test_enrichment_varies_market_context_by_date(coordinates, market_lookup) -> None:
    """Date is the only varying input, so market context must move with it."""
    reduced = build_december_chart_inputs().drop(columns=["predicted_rate"])
    enriched = enrich_reduced_frame(
        reduced, coordinates=coordinates, market_lookup=market_lookup
    )
    assert enriched["market_index"].nunique() > 1
    assert enriched["quote_signal"].nunique() > 1


def test_enrichment_rejects_uncovered_dates(coordinates, market_lookup) -> None:
    """A date with no market context must raise, not silently impute."""
    reduced = pd.DataFrame(
        {
            "pickup": [DECEMBER_FIXED_PICKUP],
            "delivery": [DECEMBER_FIXED_DELIVERY],
            "distance": [DECEMBER_FIXED_DISTANCE],
            "equipment": [DECEMBER_FIXED_EQUIPMENT],
            "weight": [DECEMBER_FIXED_WEIGHT],
            "date": ["2024-06-15"],
        }
    )
    with pytest.raises(KeyError, match="No market context"):
        enrich_reduced_frame(reduced, coordinates=coordinates, market_lookup=market_lookup)


def test_enrichment_rejects_missing_columns(coordinates, market_lookup) -> None:
    """An incomplete reduced frame must be rejected up front."""
    reduced = build_december_chart_inputs().drop(columns=["predicted_rate", "weight"])
    with pytest.raises(KeyError, match="missing required columns"):
        enrich_reduced_frame(reduced, coordinates=coordinates, market_lookup=market_lookup)


def test_december_path_flows_through_the_pipeline(
    fitted_pipeline, coordinates, market_lookup, config
) -> None:
    """End-to-end: the reduced December schema reaches a valid feature matrix."""
    reduced = build_december_chart_inputs().drop(columns=["predicted_rate"])
    enriched = enrich_reduced_frame(
        reduced, coordinates=coordinates, market_lookup=market_lookup
    )
    out = fitted_pipeline.transform(enriched[config.columns.raw_feature_columns])
    assert len(out) == 31
    assert not out.isna().any().any()
    # The curve must be able to move: date-driven features have to vary.
    assert out["market_index"].nunique() > 1
    assert out["doy_sin"].nunique() > 1
