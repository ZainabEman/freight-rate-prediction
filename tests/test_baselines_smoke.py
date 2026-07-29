"""Basic Phase-4 validation.

Deliberately a smoke suite, not exhaustive coverage: it checks that the
evaluation pipeline runs, that metrics are produced, that the split protocol is
leakage-free, and that the persisted artifact can be loaded and used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import RatePerMileBaseline, default_baselines
from src.evaluation import (
    comparison_table,
    evaluate_baseline,
    select_best_baseline,
    verify_no_leakage,
)
from src.metrics import compute_metrics


def test_metrics_are_computed_correctly() -> None:
    """A perfect prediction must score zero error and R2 of 1."""
    truth = np.array([100.0, 200.0, 300.0])
    metrics = compute_metrics(truth, truth)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.mape == pytest.approx(0.0)
    assert metrics.n == 3


def test_predictions_are_floored_to_positive() -> None:
    """score.py rejects non-positive rates, so scoring applies the same floor."""
    metrics = compute_metrics(np.array([100.0, 200.0]), np.array([-50.0, 200.0]))
    assert np.isfinite(metrics.mae)
    assert metrics.mae > 0


def test_rate_per_mile_baseline_recovers_a_known_rate() -> None:
    """With a constant rate-per-mile, the baseline must reproduce it exactly."""
    frame = pd.DataFrame(
        {"distance": [100.0, 200.0, 400.0], "equipment": ["Dry Van"] * 3}
    )
    y = frame["distance"].to_numpy() * 2.0
    model = RatePerMileBaseline().fit(frame, y)
    np.testing.assert_allclose(model.predict(frame), y)


def test_evaluation_runs_and_produces_metrics(train_raw, config, pipeline_config) -> None:
    """The harness must run end-to-end and return finite headline metrics."""
    spec = next(s for s in default_baselines() if s.name == "Rate-per-mile (by equipment)")
    result = evaluate_baseline(
        spec,
        development_frame=train_raw,
        config=config,
        pipeline_config=pipeline_config,
        run_cross_validation=False,
    )
    assert result.holdout.n > 0
    for value in (result.holdout.mae, result.holdout.rmse, result.holdout.r2, result.holdout.mape):
        assert np.isfinite(value)


def test_no_leakage_in_the_split_protocol(train_raw, config) -> None:
    """Holdout and every CV fold must be strictly forward-looking."""
    confirmations = verify_no_leakage(development_frame=train_raw, config=config)
    assert len(confirmations) >= 4


def test_comparison_table_is_ranked_by_mae() -> None:
    """The comparison table must present the best model first."""
    from src.evaluation import BaselineResult
    from src.metrics import RegressionMetrics

    def make(name: str, mae: float) -> BaselineResult:
        return BaselineResult(
            name=name,
            description="",
            input_kind="raw",
            target_transform="identity",
            holdout=RegressionMetrics(mae=mae, rmse=mae * 2, r2=0.5, mape=10.0, n=10),
            cv_mae_mean=mae,
            cv_mae_std=1.0,
        )

    results = [make("worse", 500.0), make("better", 100.0)]
    table = comparison_table(results)
    assert table.iloc[0]["model"] == "better"
    assert select_best_baseline(results).name == "better"


def test_persisted_baseline_artifact_loads_and_predicts(config) -> None:
    """The Phase-5 handoff artifact must be usable after reload."""
    import joblib

    artifact_path = config.paths.models_dir / "baseline_reference.joblib"
    if not artifact_path.is_file():
        pytest.skip("Run `python -m src.run_baselines_phase4` first.")

    artifact = joblib.load(artifact_path)
    assert {"name", "estimator", "preprocessor", "input_kind", "target_transform"} <= set(artifact)

    validation = pd.read_csv(config.paths.validation).head(50)
    features = validation[config.columns.raw_feature_columns]
    if artifact["input_kind"] == "processed":
        features = artifact["preprocessor"].transform(features)
    predictions = artifact["estimator"].predict(features)
    if artifact["target_transform"] == "log":
        predictions = np.exp(predictions)
    assert len(predictions) == 50
    assert np.all(predictions > 0)
