"""Phase-5 entry point: tune and compare advanced regression models.

Scope is deliberately narrow. This script tunes each available advanced model
with a randomised search, scores every tuned model on the same temporal holdout
Phase 4 used, selects a winner, and persists only that winner.

It produces no SHAP output, no error analysis, no submission files and no
December predictions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.advanced_models import available_advanced_models
from src.config import load_config, set_global_seed
from src.data_loader import load_csv_safe
from src.evaluation import verify_no_leakage
from src.logger import get_logger
from src.metrics import RegressionMetrics, compute_metrics
from src.run_preprocessing_phase3 import build_pipeline_config
from src.splitting import temporal_train_holdout_split
from src.tuning import TuningResult, predict_original_scale, tune_model

logger = get_logger(__name__)

CV_SPLITS = 3


def _load_baseline_rows(models_dir: Path) -> list[dict[str, object]]:
    """Load Phase-4 baseline metrics for the combined comparison table.

    Args:
        models_dir: Directory holding ``baseline_metrics.json``.

    Returns:
        Comparison rows for each baseline, or an empty list if Phase 4 has not
        been run.
    """
    path = models_dir / "baseline_metrics.json"
    if not path.is_file():
        logger.warning("No Phase-4 baseline metrics found at %s", path)
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "model": result["name"],
            "family": "baseline",
            "MAE": result["mae"],
            "RMSE": result["rmse"],
            "R2": result["r2"],
            "MAPE_%": result["mape"],
        }
        for result in payload.get("results", [])
    ]


def _build_report(
    table: pd.DataFrame,
    tuning_results: list[TuningResult],
    holdout_metrics: dict[str, RegressionMetrics],
    best_name: str,
    *,
    split_summary: dict[str, object],
    skipped: list[str],
) -> str:
    """Render ``reports/model_comparison.md``."""
    best_metrics = holdout_metrics[best_name]
    best_tuning = next(result for result in tuning_results if result.name == best_name)

    lines: list[str] = []
    lines.append("# Model Comparison (Phase 5)\n")
    lines.append(
        "Regenerate with `python -m src.run_advanced_models_phase5`. Every number is measured "
        "on the same temporal holdout Phase 4 used, so baselines and advanced models are "
        "directly comparable.\n"
    )

    lines.append("## Protocol\n")
    lines.append(
        f"- Tuning: `RandomizedSearchCV` with `TimeSeriesSplit(n_splits={CV_SPLITS})` over "
        "date-sorted training rows, scored by negative MAE."
    )
    lines.append(
        "- **Leakage control:** preprocessing is a *step inside* the searched pipeline, so it is "
        "cloned and refitted independently within every CV fold. No imputation median, scaler "
        "statistic or one-hot vocabulary is ever learned from data later than the fold it scores."
    )
    lines.append(
        "- All models train on `log(posted_rate)` via `TransformedTargetRegressor`, carried over "
        "from Phase 4 where the log target won. Predictions are inverted to dollars and are "
        "strictly positive by construction."
    )
    lines.append("")
    lines.append("| Split | rows | dates |")
    lines.append("|:--|--:|:--|")
    lines.append(
        f"| tuning (fit + CV) | {split_summary['train_rows']:,} | "
        f"{split_summary['train_date_min']} .. {split_summary['train_date_max']} |"
    )
    lines.append(
        f"| holdout (scored once) | {split_summary['holdout_rows']:,} | "
        f"{split_summary['holdout_date_min']} .. {split_summary['holdout_date_max']} |"
    )
    lines.append("")
    if skipped:
        lines.append(
            f"- Skipped (library not installed): {', '.join(skipped)}.\n"
        )

    lines.append("## Results\n")
    display = table.copy()
    for column in ("MAE", "RMSE"):
        display[column] = display[column].map(lambda value: f"{value:,.2f}")
    display["R2"] = display["R2"].map(lambda value: f"{value:.4f}")
    display["MAPE_%"] = display["MAPE_%"].map(lambda value: f"{value:.2f}")
    lines.append(display.to_markdown(index=False))
    lines.append("")
    lines.append(
        "Ranked by holdout MAE, the selection criterion carried over from Phase 4: the business "
        "cost of a mispriced load is roughly linear in dollars.\n"
    )

    lines.append("## Selected hyperparameters\n")
    for result in tuning_results:
        lines.append(
            f"- **{result.name}** (CV MAE {result.cv_best_mae:,.2f}, "
            f"{result.n_candidates} candidates, {result.search_seconds:.0f}s): "
            f"`{result.best_params}`"
        )
    lines.append("")

    lines.append("## Selected model\n")
    lines.append(f"**{best_name}**\n")
    lines.append("| Metric | Holdout value |")
    lines.append("|:--|--:|")
    lines.append(f"| MAE | ${best_metrics.mae:,.2f} |")
    lines.append(f"| RMSE | ${best_metrics.rmse:,.2f} |")
    lines.append(f"| R2 | {best_metrics.r2:.4f} |")
    lines.append(f"| MAPE | {best_metrics.mape:.2f}% |")
    lines.append(f"| Holdout rows | {best_metrics.n:,} |")
    lines.append("")
    lines.append(f"Hyperparameters: `{best_tuning.best_params}`\n")
    lines.append(
        "Persisted to `models/best_model.joblib` as a complete pipeline "
        "(preprocessing + log-target model), so Phase 7 can load it and predict directly from "
        "raw feature frames.\n"
    )
    return "\n".join(lines)


def main() -> None:
    """Run the Phase-5 advanced model comparison."""
    config = load_config()
    set_global_seed(config.random_seed)

    development_frame = load_csv_safe(config.paths.train, label="train")
    pipeline_config = build_pipeline_config(config)

    for confirmation in verify_no_leakage(
        development_frame=development_frame, config=config
    ):
        logger.info("Leakage check: %s", confirmation)

    split = temporal_train_holdout_split(
        development_frame,
        date_column=config.columns.date,
        holdout_start=config.split.holdout_start,
    )

    specs = available_advanced_models(config.random_seed)
    if not specs:
        raise RuntimeError("No advanced models are available; cannot run Phase 5.")

    expected = {"RandomForest", "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"}
    skipped = sorted(expected - {spec.name for spec in specs})

    tuning_results: list[TuningResult] = []
    holdout_metrics: dict[str, RegressionMetrics] = {}

    for spec in specs:
        result = tune_model(
            spec,
            train_frame=split.train,
            config=config,
            pipeline_config=pipeline_config,
            n_splits=CV_SPLITS,
        )
        tuning_results.append(result)

        predictions = predict_original_scale(result.best_estimator, split.holdout, config=config)
        metrics = compute_metrics(
            split.holdout[config.columns.target].to_numpy(dtype=float), predictions
        )
        holdout_metrics[spec.name] = metrics
        logger.info(
            "%-22s holdout MAE=%8.2f RMSE=%8.2f R2=%6.4f MAPE=%5.2f%%",
            spec.name,
            metrics.mae,
            metrics.rmse,
            metrics.r2,
            metrics.mape,
        )

    advanced_rows = [
        {
            "model": name,
            "family": "advanced",
            "MAE": metrics.mae,
            "RMSE": metrics.rmse,
            "R2": metrics.r2,
            "MAPE_%": metrics.mape,
        }
        for name, metrics in holdout_metrics.items()
    ]
    table = (
        pd.DataFrame(advanced_rows + _load_baseline_rows(config.paths.models_dir))
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    best_name = min(holdout_metrics, key=lambda name: holdout_metrics[name].mae)
    best_tuning = next(result for result in tuning_results if result.name == best_name)
    logger.info(
        "Best advanced model: %s (holdout MAE=%.2f)", best_name, holdout_metrics[best_name].mae
    )

    models_dir = config.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_tuning.best_estimator, models_dir / "best_model.joblib")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": config.random_seed,
        "selection_metric": "holdout MAE",
        "best_model": best_name,
        "best_params": best_tuning.best_params,
        "target_transform": "log",
        "cv_splits": CV_SPLITS,
        "split": split.summary,
        "skipped_models": skipped,
        "holdout_metrics": {
            name: metrics.as_dict() for name, metrics in holdout_metrics.items()
        },
        "tuning": [
            {
                "name": result.name,
                "cv_best_mae": result.cv_best_mae,
                "n_candidates": result.n_candidates,
                "search_seconds": result.search_seconds,
                "best_params": result.best_params,
            }
            for result in tuning_results
        ],
    }
    (models_dir / "best_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "model_comparison.md"
    report_path.write_text(
        _build_report(
            table,
            tuning_results,
            holdout_metrics,
            best_name,
            split_summary=split.summary,
            skipped=skipped,
        ),
        encoding="utf-8",
    )

    logger.info("Wrote %s", report_path)
    logger.info("Phase 5 complete: %d advanced models tuned", len(tuning_results))


if __name__ == "__main__":
    main()
