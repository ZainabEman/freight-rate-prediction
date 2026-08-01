"""Basic Phase-5 validation.

Checks that tuning completes, that metrics are produced, and that the persisted
best model reloads and predicts. Kept to a smoke suite by design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.advanced_models import ModelSpec, available_advanced_models, log_target_functions
from src.metrics import compute_metrics
from src.tuning import build_full_pipeline, predict_original_scale, tune_model


def test_at_least_one_advanced_model_is_available() -> None:
    """The two scikit-learn ensembles are always present."""
    names = {spec.name for spec in available_advanced_models()}
    assert {"RandomForest", "HistGradientBoosting"} <= names


def test_missing_optional_libraries_are_skipped_not_fatal() -> None:
    """Building the spec list must never raise on a missing optional library."""
    specs = available_advanced_models()
    assert all(isinstance(spec, ModelSpec) for spec in specs)
    assert len(specs) >= 2


def test_log_target_round_trips() -> None:
    """The forward/inverse pair must reconstruct the original rates."""
    forward, inverse = log_target_functions()
    values = np.array([57.22, 2030.76, 25533.0])
    np.testing.assert_allclose(inverse(forward(values)), values, rtol=1e-10)


def test_full_pipeline_trains_and_predicts_positive(config, pipeline_config, train_raw) -> None:
    """Training must complete and yield strictly positive dollar predictions."""
    spec = next(s for s in available_advanced_models() if s.name == "HistGradientBoosting")
    pipeline = build_full_pipeline(spec, pipeline_config=pipeline_config, seed=config.random_seed)

    sample = train_raw.head(2_000)
    pipeline.fit(
        sample[config.columns.raw_feature_columns],
        sample[config.columns.target].to_numpy(dtype=float),
    )
    predictions = predict_original_scale(pipeline, sample.head(100), config=config)
    assert len(predictions) == 100
    assert np.all(predictions > 0)
    assert np.all(np.isfinite(predictions))


def test_tuning_completes_and_produces_metrics(config, pipeline_config, train_raw) -> None:
    """A minimal search must return a fitted estimator and a finite CV score."""
    base = next(s for s in available_advanced_models() if s.name == "HistGradientBoosting")
    tiny = ModelSpec(
        name=base.name,
        factory=base.factory,
        param_distributions={"learning_rate": [0.1], "max_iter": [50]},
        n_iter=1,
        notes=base.notes,
    )
    sample = train_raw.head(3_000)
    result = tune_model(
        tiny,
        train_frame=sample,
        config=config,
        pipeline_config=pipeline_config,
        n_splits=2,
    )
    assert np.isfinite(result.cv_best_mae)
    assert result.best_params

    predictions = predict_original_scale(result.best_estimator, sample.head(200), config=config)
    metrics = compute_metrics(
        sample.head(200)[config.columns.target].to_numpy(dtype=float), predictions
    )
    for value in (metrics.mae, metrics.rmse, metrics.r2, metrics.mape):
        assert np.isfinite(value)


def test_saved_best_model_reloads_and_predicts(config) -> None:
    """The Phase-7 handoff artifact must survive a round trip to disk."""
    import joblib

    path = config.paths.models_dir / "best_model.joblib"
    if not path.is_file():
        pytest.skip("Run `python -m src.run_advanced_models_phase5` first.")

    model = joblib.load(path)
    validation = pd.read_csv(config.paths.validation).head(100)
    predictions = predict_original_scale(model, validation, config=config)
    assert len(predictions) == 100
    assert np.all(predictions > 0)
    assert np.all(np.isfinite(predictions))
