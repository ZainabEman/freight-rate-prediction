"""Phase-4 entry point: train, evaluate and compare baseline models.

Trains no advanced models, performs no hyperparameter search and produces no
submission files. It establishes the reference bar that Phase 5 must beat.

The selected baseline is refitted on the full development window and persisted
so Phase 5 can load it for a like-for-like comparison without re-running Phase 4.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.base import clone

from src.baselines import BaselineSpec, default_baselines
from src.config import load_config, set_global_seed
from src.data_loader import load_csv_safe
from src.evaluation import (
    BaselineResult,
    apply_target_transform,
    comparison_table,
    evaluate_baseline,
    select_best_baseline,
    verify_no_leakage,
)
from src.logger import get_logger
from src.pipeline import build_preprocessing_pipeline
from src.run_preprocessing_phase3 import build_pipeline_config

logger = get_logger(__name__)


def _build_report(
    results: list[BaselineResult],
    table: pd.DataFrame,
    best: BaselineResult,
    leakage_confirmations: list[str],
    *,
    split_summary: dict[str, object],
) -> str:
    """Render ``reports/baseline_models.md``."""
    lines: list[str] = []
    lines.append("# Baseline Models (Phase 4)\n")
    lines.append(
        "Regenerate with `python -m src.run_baselines_phase4`. Every number below is measured "
        "on the temporal holdout defined in `config/config.yaml`.\n"
    )

    lines.append("## Evaluation protocol\n")
    lines.append(
        "Development data is 2025-01-01..2025-10-31; the real scoring window is "
        "2025-11-01..2025-12-31. Because the two do not overlap, every evaluation here is "
        "**forward-in-time**:\n"
    )
    lines.append("| Split | rows | dates |")
    lines.append("|:--|--:|:--|")
    lines.append(
        f"| fit | {split_summary['train_rows']:,} | "
        f"{split_summary['train_date_min']} .. {split_summary['train_date_max']} |"
    )
    lines.append(
        f"| holdout | {split_summary['holdout_rows']:,} | "
        f"{split_summary['holdout_date_min']} .. {split_summary['holdout_date_max']} |"
    )
    lines.append("")
    lines.append(
        "A secondary expanding-window cross-validation (`TimeSeriesSplit`) reports MAE stability "
        "across folds. **A fresh preprocessing pipeline is fitted inside every split**, so no "
        "statistic learned from later data ever touches earlier data.\n"
    )
    lines.append(
        "Predictions are clipped to a positive floor before scoring, matching the constraint "
        "`score.py` enforces at submission time.\n"
    )

    lines.append("## Model comparison\n")
    display = table.copy()
    for column in ("MAE", "RMSE", "CV_MAE_mean", "CV_MAE_std"):
        display[column] = display[column].map(lambda value: f"{value:,.2f}")
    display["R2"] = display["R2"].map(lambda value: f"{value:.4f}")
    display["MAPE_%"] = display["MAPE_%"].map(lambda value: f"{value:.2f}")
    display["fit_s"] = display["fit_s"].map(lambda value: f"{value:.1f}")
    lines.append(display.to_markdown(index=False))
    lines.append("")
    lines.append(
        "Ranked by holdout MAE. MAE is the selection criterion because the business cost of a "
        "mispriced load is roughly linear in dollars, and unlike RMSE it is not dominated by the "
        "small number of very high-rate loads (target max 25,533 against a median of 2,031).\n"
    )

    lines.append("## Model descriptions\n")
    for result in results:
        lines.append(
            f"- **{result.name}** ({result.input_kind} input, {result.target_transform} target): "
            f"{result.description}"
        )
    lines.append("")

    lines.append("## Selected reference baseline\n")
    lines.append(f"**{best.name}**\n")
    lines.append("| Metric | Holdout value |")
    lines.append("|:--|--:|")
    lines.append(f"| MAE | ${best.holdout.mae:,.2f} |")
    lines.append(f"| RMSE | ${best.holdout.rmse:,.2f} |")
    lines.append(f"| R2 | {best.holdout.r2:.4f} |")
    lines.append(f"| MAPE | {best.holdout.mape:.2f}% |")
    lines.append(f"| CV MAE (mean +/- std) | {best.cv_mae_mean:,.2f} +/- {best.cv_mae_std:,.2f} |")
    lines.append(f"| Holdout rows | {best.holdout.n:,} |")
    lines.append("")
    lines.append(
        "This is the bar Phase 5 must beat. A gradient-boosted model that cannot improve on it "
        "is not worth the additional complexity.\n"
    )

    lines.append("## Leakage verification\n")
    for confirmation in leakage_confirmations:
        lines.append(f"- [x] {confirmation}")
    lines.append("")

    lines.append("## Scope\n")
    lines.append(
        "Phase 4 covers splitting, baseline training, evaluation and comparison only. No "
        "hyperparameter search, no advanced models, no explainability and no submission files "
        "were produced.\n"
    )
    return "\n".join(lines)


def _refit_best_on_full_development(
    spec: BaselineSpec, development_frame: pd.DataFrame, config, pipeline_config
):
    """Refit the winning baseline on the entire development window.

    Returns:
        ``(fitted_estimator, fitted_preprocessor_or_None)``.
    """
    columns = config.columns
    y = apply_target_transform(
        development_frame[columns.target].to_numpy(dtype=float), spec.target_transform
    )
    features = development_frame[columns.raw_feature_columns]

    preprocessor = None
    if spec.input_kind == "processed":
        preprocessor = build_preprocessing_pipeline(pipeline_config)
        preprocessor.fit(features)
        features = preprocessor.transform(features)

    estimator = clone(spec.estimator)
    estimator.fit(features, y)
    return estimator, preprocessor


def main() -> None:
    """Run the Phase-4 baseline comparison."""
    config = load_config()
    set_global_seed(config.random_seed)

    development_frame = load_csv_safe(config.paths.train, label="train")
    pipeline_config = build_pipeline_config(config)

    leakage_confirmations = verify_no_leakage(
        development_frame=development_frame, config=config
    )
    for confirmation in leakage_confirmations:
        logger.info("Leakage check: %s", confirmation)

    specs = default_baselines()
    logger.info("Evaluating %d baselines", len(specs))

    results = [
        evaluate_baseline(
            spec,
            development_frame=development_frame,
            config=config,
            pipeline_config=pipeline_config,
        )
        for spec in specs
    ]

    table = comparison_table(results)
    best = select_best_baseline(results)
    best_spec = next(spec for spec in specs if spec.name == best.name)

    logger.info("Best baseline: %s (holdout MAE=%.2f)", best.name, best.holdout.mae)

    # Persist the reference baseline for Phase 5.
    estimator, preprocessor = _refit_best_on_full_development(
        best_spec, development_frame, config, pipeline_config
    )
    models_dir = config.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": best.name,
            "estimator": estimator,
            "preprocessor": preprocessor,
            "input_kind": best_spec.input_kind,
            "target_transform": best_spec.target_transform,
        },
        models_dir / "baseline_reference.joblib",
    )

    from src.splitting import temporal_train_holdout_split

    split = temporal_train_holdout_split(
        development_frame,
        date_column=config.columns.date,
        holdout_start=config.split.holdout_start,
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": config.random_seed,
        "selection_metric": "holdout MAE",
        "best_baseline": best.name,
        "split": split.summary,
        "results": [
            {
                "name": result.name,
                "input_kind": result.input_kind,
                "target_transform": result.target_transform,
                **result.holdout.as_dict(),
                "cv_mae_mean": result.cv_mae_mean,
                "cv_mae_std": result.cv_mae_std,
                "cv_fold_mae": result.cv_fold_mae,
            }
            for result in results
        ],
    }
    (models_dir / "baseline_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "baseline_models.md"
    report_path.write_text(
        _build_report(
            results, table, best, leakage_confirmations, split_summary=split.summary
        ),
        encoding="utf-8",
    )

    logger.info("Wrote %s", report_path)
    logger.info("Phase 4 complete: %d baselines evaluated", len(results))


if __name__ == "__main__":
    main()
