"""Tests for the stateless feature-construction stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.transformers import (
    CyclicalDateFeatures,
    GeoFeatures,
    InteractionFeatures,
    UnseenCategoryIndicator,
)


def test_cyclical_encoding_stays_bounded_outside_the_training_months() -> None:
    """The core fix for audit finding M-1.

    November/December are absent from training. Raw month integers would leave
    the training support entirely; sine/cosine encodings cannot.
    """
    train_dates = pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-10-31", freq="D")})
    score_dates = pd.DataFrame({"date": pd.date_range("2025-11-01", "2025-12-31", freq="D")})

    encoder = CyclicalDateFeatures().fit(train_dates)
    scored = encoder.transform(score_dates)

    for column in ("doy_sin", "doy_cos", "dow_sin", "dow_cos"):
        assert scored[column].between(-1.0, 1.0).all()


def test_cyclical_day_of_year_is_continuous_across_the_year_boundary() -> None:
    """31 Dec and 1 Jan must be adjacent in the encoded space, not opposite."""
    frame = pd.DataFrame({"date": ["2025-12-31", "2026-01-01", "2025-07-01"]})
    encoded = CyclicalDateFeatures().fit(frame).transform(frame)

    def distance(i: int, j: int) -> float:
        return float(
            np.hypot(
                encoded["doy_sin"].iloc[i] - encoded["doy_sin"].iloc[j],
                encoded["doy_cos"].iloc[i] - encoded["doy_cos"].iloc[j],
            )
        )

    assert distance(0, 1) < 0.1
    assert distance(0, 2) > 1.0


def test_no_constant_year_feature_is_emitted() -> None:
    """`date_year` was constant at 2025 and is deliberately not produced."""
    frame = pd.DataFrame({"date": ["2025-03-01", "2025-04-01"]})
    names = set(CyclicalDateFeatures().fit(frame).get_feature_names_out())
    assert "date_year" not in names
    assert "date_month" not in names


def test_weekend_flag_matches_calendar() -> None:
    """2025-12-06 is a Saturday and 2025-12-08 a Monday."""
    frame = pd.DataFrame({"date": ["2025-12-06", "2025-12-07", "2025-12-08"]})
    encoded = CyclicalDateFeatures().fit(frame).transform(frame)
    assert encoded["is_weekend"].tolist() == [1.0, 1.0, 0.0]


def test_unparseable_dates_raise_instead_of_becoming_nan() -> None:
    """Audit finding NF-7: fail loudly rather than leaking NaN into the model."""
    frame = pd.DataFrame({"date": ["2025-01-01", "not-a-date"]})
    encoder = CyclicalDateFeatures().fit(pd.DataFrame({"date": ["2025-01-01"]}))
    with pytest.raises(ValueError, match="could not be parsed"):
        encoder.transform(frame)


def test_haversine_matches_a_known_separation() -> None:
    """One degree of latitude is roughly 69 statute miles."""
    frame = pd.DataFrame(
        {
            "pickup_lat": [35.0],
            "pickup_lon": [-90.0],
            "delivery_lat": [36.0],
            "delivery_lon": [-90.0],
        }
    )
    out = GeoFeatures().fit(frame).transform(frame)
    assert out["haversine_miles"].iloc[0] == pytest.approx(69.09, abs=0.5)


def test_bearing_is_encoded_circularly() -> None:
    """Due north and a hair west of north must be close in encoded space."""
    frame = pd.DataFrame(
        {
            "pickup_lat": [35.0, 35.0],
            "pickup_lon": [-90.0, -90.0],
            "delivery_lat": [36.0, 36.0],
            "delivery_lon": [-90.0, -90.001],
        }
    )
    out = GeoFeatures().fit(frame).transform(frame)
    assert abs(out["bearing_cos"].iloc[0] - out["bearing_cos"].iloc[1]) < 0.01


def test_lon_delta_sign_encodes_direction() -> None:
    """Negative lon_delta means westbound, matching the observed rate premium."""
    frame = pd.DataFrame(
        {
            "pickup_lat": [35.0],
            "pickup_lon": [-80.0],
            "delivery_lat": [35.0],
            "delivery_lon": [-100.0],
        }
    )
    out = GeoFeatures().fit(frame).transform(frame)
    assert out["lon_delta"].iloc[0] < 0


def test_interaction_features_handle_missing_weight() -> None:
    """weight_per_mile must be NaN (not inf/0) when weight is missing."""
    frame = pd.DataFrame({"distance": [100.0, 200.0], "weight": [np.nan, 40_000.0]})
    out = InteractionFeatures().fit(frame).transform(frame)
    assert np.isnan(out["weight_per_mile"].iloc[0])
    assert out["weight_per_mile"].iloc[1] == pytest.approx(200.0)


def test_log_distance_is_monotone() -> None:
    """log1p preserves ordering while compressing the long tail."""
    frame = pd.DataFrame({"distance": [70.0, 900.0, 3_440.0], "weight": [1.0, 1.0, 1.0]})
    out = InteractionFeatures().fit(frame).transform(frame)
    assert out["log_distance"].is_monotonic_increasing


def test_unseen_category_indicator_flags_only_new_values() -> None:
    """The explicit fix for audit finding M-4."""
    train = pd.DataFrame({"pickup": ["Lexington", "Atlanta"]})
    score = pd.DataFrame({"pickup": ["Lexington", "Chicago", "Atlanta"]})
    indicator = UnseenCategoryIndicator(columns=["pickup"]).fit(train)
    out = indicator.transform(score)
    assert out["pickup_is_unknown"].tolist() == [0.0, 1.0, 0.0]


def test_unseen_category_indicator_ignores_surrounding_whitespace() -> None:
    """Whitespace variants must not be misreported as unseen categories."""
    train = pd.DataFrame({"pickup": ["Lexington"]})
    score = pd.DataFrame({"pickup": ["  Lexington  "]})
    indicator = UnseenCategoryIndicator(columns=["pickup"]).fit(train)
    assert indicator.transform(score)["pickup_is_unknown"].tolist() == [0.0]


def test_real_unseen_cities_are_detected(train_sample, validation_raw) -> None:
    """Against the real data, the indicator must fire for genuinely new cities."""
    indicator = UnseenCategoryIndicator(columns=["pickup"]).fit(train_sample[["pickup"]])
    flags = indicator.transform(validation_raw[["pickup"]])["pickup_is_unknown"]
    unseen = set(validation_raw["pickup"]) - set(train_sample["pickup"])
    assert flags.sum() == validation_raw["pickup"].isin(unseen).sum()
