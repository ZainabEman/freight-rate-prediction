"""Tests for the assembled preprocessing pipeline and its integrity guards.

These cover the five verification requirements of Phase 3: reproducibility,
train/validation alignment, feature-name preservation, leakage safety, and a
clean run from the raw CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    assert_feature_names_preserved,
    assert_frames_aligned,
    assert_no_leakage,
    assert_no_missing_values,
    build_preprocessing_pipeline,
)


def test_pipeline_runs_from_raw_csv_columns(fitted_pipeline, train_sample, config) -> None:
    """The pipeline consumes the raw schema directly, with no manual prep."""
    out = fitted_pipeline.transform(train_sample[config.columns.raw_feature_columns])
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(train_sample)
    assert out.shape[1] > len(config.columns.raw_feature_columns)


def test_feature_names_are_preserved(fitted_pipeline, train_sample, config) -> None:
    """Regression guard for audit finding C-1 (names replaced by integers)."""
    out = fitted_pipeline.transform(train_sample[config.columns.raw_feature_columns])
    assert_feature_names_preserved(out)
    assert not any(str(name).isdigit() for name in out.columns)


def test_expected_engineered_features_are_present(fitted_pipeline, train_sample, config) -> None:
    """The engineered feature set actually reaches the output matrix."""
    out = fitted_pipeline.transform(train_sample[config.columns.raw_feature_columns])
    expected = {
        "doy_sin",
        "doy_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "haversine_miles",
        "bearing_sin",
        "bearing_cos",
        "lon_delta",
        "lat_delta",
        "log_distance",
        "weight_per_mile",
        "weight_is_missing",
        "market_index_is_missing",
        "pickup_is_unknown",
        "delivery_is_unknown",
    }
    assert expected.issubset(set(out.columns))


def test_train_and_validation_columns_align(
    fitted_pipeline, train_sample, validation_sample, config
) -> None:
    """Audit risk R-10: alignment must be verified by name, not by position."""
    train_out = fitted_pipeline.transform(train_sample[config.columns.raw_feature_columns])
    validation_out = fitted_pipeline.transform(
        validation_sample[config.columns.raw_feature_columns]
    )
    assert list(train_out.columns) == list(validation_out.columns)
    assert_frames_aligned(train_out, validation_out, label="validation")


def test_alignment_guard_detects_reordering(fitted_pipeline, train_sample, config) -> None:
    """The guard must actually fail when columns are reordered."""
    out = fitted_pipeline.transform(train_sample[config.columns.raw_feature_columns])
    shuffled = out[list(out.columns[::-1])]
    with pytest.raises(ValueError, match="ordered differently"):
        assert_frames_aligned(out, shuffled, label="shuffled")


def test_alignment_guard_detects_missing_columns(fitted_pipeline, train_sample, config) -> None:
    """The guard must fail when a column disappears."""
    out = fitted_pipeline.transform(train_sample[config.columns.raw_feature_columns])
    dropped = out.drop(columns=[out.columns[0]])
    with pytest.raises(ValueError, match="do not match"):
        assert_frames_aligned(out, dropped, label="dropped")


def test_no_missing_values_survive(fitted_pipeline, train_sample, validation_sample, config) -> None:
    """Imputation must leave a fully dense matrix."""
    for name, frame in (("train", train_sample), ("validation", validation_sample)):
        out = fitted_pipeline.transform(frame[config.columns.raw_feature_columns])
        assert_no_missing_values(out, label=name)


def test_no_target_or_id_leakage(fitted_pipeline, config) -> None:
    """Neither the target nor the identifier may reach the feature matrix."""
    assert_no_leakage(fitted_pipeline, forbidden_columns=[config.columns.target, config.columns.id])


def test_target_is_absent_from_output_columns(fitted_pipeline, train_sample, config) -> None:
    """Explicit check that posted_rate never appears as a feature."""
    out = fitted_pipeline.transform(train_sample[config.columns.raw_feature_columns])
    assert config.columns.target not in out.columns
    assert config.columns.id not in out.columns


def test_fitting_ignores_validation_data(
    pipeline_config, train_sample, validation_sample, config
) -> None:
    """A pipeline fitted on train alone must be unchanged by validation rows.

    This is the concrete leakage check: if validation statistics influenced the
    fit, transforming the same training rows would give a different answer.
    """
    columns = config.columns.raw_feature_columns

    train_only = build_preprocessing_pipeline(pipeline_config)
    train_only.fit(train_sample[columns])
    baseline = train_only.transform(train_sample[columns])

    train_only.transform(validation_sample[columns])
    after = train_only.transform(train_sample[columns])

    pd.testing.assert_frame_equal(baseline, after)


def test_preprocessing_is_reproducible(pipeline_config, train_sample, validation_sample, config):
    """Two independent fits with the same seed must agree exactly."""
    columns = config.columns.raw_feature_columns

    first = build_preprocessing_pipeline(pipeline_config)
    first.fit(train_sample[columns])
    second = build_preprocessing_pipeline(pipeline_config)
    second.fit(train_sample[columns])

    pd.testing.assert_frame_equal(
        first.transform(validation_sample[columns]),
        second.transform(validation_sample[columns]),
    )


def test_transform_is_row_order_independent(fitted_pipeline, validation_sample, config) -> None:
    """Transforming a subset must match the corresponding rows of the whole."""
    columns = config.columns.raw_feature_columns
    full = fitted_pipeline.transform(validation_sample[columns])
    subset = fitted_pipeline.transform(validation_sample[columns].iloc[:50])
    np.testing.assert_allclose(full.to_numpy()[:50], subset.to_numpy(), rtol=1e-10)


def test_unseen_categories_do_not_raise(fitted_pipeline, config) -> None:
    """Unseen cities must transform cleanly and be flagged, not crash."""
    frame = pd.DataFrame(
        {
            "pickup": ["Atlantis"],
            "delivery": ["El Dorado"],
            "equipment": ["Dry Van"],
            "date": ["2025-12-15"],
            "pickup_lat": [35.0],
            "pickup_lon": [-90.0],
            "delivery_lat": [36.0],
            "delivery_lon": [-92.0],
            "distance": [360.0],
            "weight": [32_000.0],
            "market_index": [0.95],
            "quote_signal": [2.05],
        }
    )
    out = fitted_pipeline.transform(frame[config.columns.raw_feature_columns])
    assert out["pickup_is_unknown"].iloc[0] == 1.0
    assert out["delivery_is_unknown"].iloc[0] == 1.0
    assert not out.isna().any().any()


def test_binary_indicators_are_not_scaled(fitted_pipeline, validation_sample, config) -> None:
    """Indicator columns must stay interpretable 0/1 values."""
    out = fitted_pipeline.transform(validation_sample[config.columns.raw_feature_columns])
    for column in ("weight_is_missing", "market_index_is_missing", "pickup_is_unknown"):
        assert set(np.unique(out[column])).issubset({0.0, 1.0})


def test_feature_name_guard_rejects_integer_columns() -> None:
    """The guard itself must catch the exact corruption from audit finding C-1."""
    corrupt = pd.DataFrame(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Feature names were lost"):
        assert_feature_names_preserved(corrupt)


def test_feature_name_guard_rejects_non_dataframe() -> None:
    """A bare ndarray carries no names and must be rejected."""
    with pytest.raises(TypeError):
        assert_feature_names_preserved(np.zeros((2, 3)))
