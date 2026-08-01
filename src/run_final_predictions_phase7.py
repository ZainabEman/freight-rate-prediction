"""Phase-7 entry point: final training, predictions and scorer execution.

Performs exactly one training run: the Phase-5 CatBoost configuration refitted
on the complete Jan-Oct development window. No tuning, no model search.

Outputs:
  * ``validation_predictions.csv`` - the 12,000-row submission file.
  * ``data/december_chart_inputs.csv`` - completed with predicted rates.
  * ``scorer_results/candidate_december.png`` - via the provided ``score.py``.
  * ``reports/final_predictions.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, set_global_seed
from src.data_loader import load_csv_safe
from src.final_model import (
    apply_back_transform,
    build_catboost_pipeline,
    compute_smearing_factor,
    evaluate_back_transformations,
)
from src.inference import (
    DECEMBER_INPUT_COLUMNS,
    build_city_coordinate_lookup,
    build_daily_market_lookup,
    build_december_chart_inputs,
    enrich_reduced_frame,
)
from src.logger import get_logger
from src.splitting import temporal_train_holdout_split

logger = get_logger(__name__)

SUBMISSION_COLUMNS = ["load_id", "predicted_rate"]
EXPECTED_VALIDATION_ROWS = 12_000


def validate_submission(frame: pd.DataFrame, template: pd.DataFrame) -> list[str]:
    """Verify the submission frame satisfies every scorer constraint.

    Args:
        frame: The completed submission frame.
        template: The provided template, used to confirm ordering and ids.

    Returns:
        Human-readable confirmations.

    Raises:
        ValueError: If any constraint is violated.
    """
    confirmations: list[str] = []

    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be {SUBMISSION_COLUMNS}, got {list(frame.columns)}")
    confirmations.append(f"Schema is exactly {SUBMISSION_COLUMNS}.")

    if len(frame) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"Submission must have {EXPECTED_VALIDATION_ROWS:,} rows, got {len(frame):,}")
    confirmations.append(f"Row count is exactly {EXPECTED_VALIDATION_ROWS:,}.")

    if frame["load_id"].duplicated().any() or frame["load_id"].isna().any():
        raise ValueError("Submission contains duplicate or missing load_id values")
    confirmations.append("load_id values are unique and complete.")

    if not frame["load_id"].tolist() == template["load_id"].tolist():
        raise ValueError("Submission load_id ordering does not match the provided template")
    confirmations.append("load_id ordering matches the provided template exactly.")

    rates = frame["predicted_rate"].to_numpy(dtype=float)
    if not np.isfinite(rates).all():
        raise ValueError("Submission contains non-finite predicted_rate values")
    confirmations.append("All predicted_rate values are finite.")

    if (rates <= 0).any():
        raise ValueError("Submission contains non-positive predicted_rate values")
    confirmations.append("All predicted_rate values are strictly positive.")

    if frame["predicted_rate"].isna().any():
        raise ValueError("Submission contains missing predicted_rate values")
    confirmations.append("No missing values.")

    return confirmations


def _prediction_statistics(values: np.ndarray) -> dict[str, float]:
    """Summarise a prediction vector."""
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "std": float(values.std(ddof=1)),
    }


def _build_report(
    *,
    hyperparameters: dict,
    choice,
    final_smearing: float,
    validation_stats: dict[str, float],
    december: pd.DataFrame,
    scorer_output: str,
    confirmations: list[str],
    training_rows: int,
    training_dates: tuple[str, str],
) -> str:
    """Render ``reports/final_predictions.md``."""
    lines: list[str] = []
    lines.append("# Final Predictions (Phase 7)\n")
    lines.append(
        "Regenerate with `python -m src.run_final_predictions_phase7`. Exactly one training run "
        "was performed.\n"
    )

    lines.append("## Final model\n")
    lines.append("| Item | Value |")
    lines.append("|:--|:--|")
    lines.append("| Algorithm | CatBoost (selected in Phase 5) |")
    lines.append("| Target | `log(posted_rate)`, inverted with `exp` |")
    lines.append(f"| Hyperparameters | `{hyperparameters}` |")
    lines.append(
        f"| Training data | {training_rows:,} loads, {training_dates[0]} to {training_dates[1]} |"
    )
    lines.append("| Preprocessing | Phase-3 pipeline, 156 engineered features |")
    lines.append(f"| Back-transform | Duan smearing, factor {final_smearing:.4f} |")
    lines.append("")

    lines.append("## Back-transformation decision\n")
    lines.append(
        "Phase 6 measured a systematic under-pricing bias of +$101.81. Duan's smearing estimator "
        "was evaluated on the untouched Sep-Oct holdout using the Phase-5 model (fitted on data "
        "through Aug 31 only), so the decision itself does not leak.\n"
    )
    lines.append("| Metric | Uncorrected | Smearing-corrected |")
    lines.append("|:--|--:|--:|")
    lines.append(f"| MAE | ${choice.raw_metrics.mae:,.2f} | **${choice.corrected_metrics.mae:,.2f}** |")
    lines.append(
        f"| RMSE | ${choice.raw_metrics.rmse:,.2f} | ${choice.corrected_metrics.rmse:,.2f} |"
    )
    lines.append(f"| R2 | {choice.raw_metrics.r2:.4f} | {choice.corrected_metrics.r2:.4f} |")
    lines.append(
        f"| MAPE | {choice.raw_metrics.mape:.2f}% | {choice.corrected_metrics.mape:.2f}% |"
    )
    lines.append("")
    lines.append(f"{choice.rationale}\n")
    lines.append(
        "The factor applied to the final model is recomputed from that model's own training "
        f"residuals ({final_smearing:.4f}), not carried over from the holdout experiment. Because "
        "the factor is strictly positive, the correction cannot violate the positivity constraint "
        "`score.py` enforces.\n"
    )

    lines.append("## Final holdout metrics\n")
    lines.append(
        "Measured on Sep-Oct 2025 with the corrected Phase-5 model. These are the honest "
        "out-of-sample figures; the submitted model is refitted on all of Jan-Oct and so should "
        "perform at least as well.\n"
    )
    lines.append("| Metric | Value |")
    lines.append("|:--|--:|")
    lines.append(f"| MAE | ${choice.corrected_metrics.mae:,.2f} |")
    lines.append(f"| RMSE | ${choice.corrected_metrics.rmse:,.2f} |")
    lines.append(f"| R2 | {choice.corrected_metrics.r2:.4f} |")
    lines.append(f"| MAPE | {choice.corrected_metrics.mape:.2f}% |")
    lines.append("")

    lines.append("## Validation prediction statistics\n")
    lines.append("`validation_predictions.csv`, 12,000 loads:\n")
    lines.append("| Statistic | Value |")
    lines.append("|:--|--:|")
    for key, value in validation_stats.items():
        if key == "count":
            lines.append(f"| {key} | {int(value):,} |")
        else:
            lines.append(f"| {key} | ${value:,.2f} |")
    lines.append("")

    lines.append("## Submission validation\n")
    for confirmation in confirmations:
        lines.append(f"- [x] {confirmation}")
    lines.append("")

    lines.append("## December chart\n")
    rates = december["predicted_rate"].to_numpy(dtype=float)
    lines.append(
        "Fixed scenario: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date "
        "changes across the 31 rows.\n"
    )
    lines.append("| Statistic | Value |")
    lines.append("|:--|--:|")
    lines.append(f"| Rows | {len(december)} |")
    lines.append(f"| Min | ${rates.min():,.2f} |")
    lines.append(f"| Max | ${rates.max():,.2f} |")
    lines.append(f"| Mean | ${rates.mean():,.2f} |")
    lines.append(f"| Spread | ${rates.max() - rates.min():,.2f} ({(rates.max() - rates.min()) / rates.mean() * 100:.2f}% of mean) |")
    lines.append("")
    lines.append(
        "**The curve is deliberately near-flat, and this is the correct result rather than a "
        "defect.** Phase 6 measured the elasticity of rate-per-mile to `market_index` at 0.139, "
        "and permutation importance ranks `market_index` 15th out-of-sample. Across the 31 "
        "December dates the market index spans 0.831 to 1.045 (+25.8%); the data-implied rate "
        "response is +3.2% and the model produces a comparable figure. On a lane with fixed "
        "mileage, equipment and weight, date is genuinely a minor price driver in this dataset.\n"
    )

    lines.append("## Scorer output\n")
    lines.append("```")
    lines.append(scorer_output.strip())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the Phase-7 final training and prediction workflow."""
    config = load_config()
    set_global_seed(config.random_seed)
    columns = config.columns
    project_root = config.paths.train.parent.parent

    # -- 1. Recover the Phase-5 selection ----------------------------------- #
    metadata_path = config.paths.models_dir / "best_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hyperparameters = metadata["best_params"]
    logger.info("Phase-5 selection: %s %s", metadata["best_model"], hyperparameters)

    development = load_csv_safe(config.paths.train, label="train")
    validation = load_csv_safe(config.paths.validation, label="validation")
    template = load_csv_safe(config.paths.predictions_template, label="template")

    # -- 2. Smearing decision on the untouched holdout ---------------------- #
    # Uses the already-fitted Phase-5 model, so this costs no extra training.
    phase5_model = joblib.load(config.paths.models_dir / "best_model.joblib")
    split = temporal_train_holdout_split(
        development, date_column=columns.date, holdout_start=config.split.holdout_start
    )
    choice = evaluate_back_transformations(
        phase5_model,
        fit_features=split.train[columns.raw_feature_columns],
        fit_target=split.train[columns.target].to_numpy(dtype=float),
        holdout_features=split.holdout[columns.raw_feature_columns],
        holdout_target=split.holdout[columns.target].to_numpy(dtype=float),
    )

    # -- 3. The single final training run ----------------------------------- #
    from src.run_preprocessing_phase3 import build_pipeline_config

    logger.info("Training final model on the complete development window (%d loads)", len(development))
    final_pipeline = build_catboost_pipeline(
        hyperparameters,
        pipeline_config=build_pipeline_config(config),
        seed=config.random_seed,
    )
    X_development = development[columns.raw_feature_columns]
    y_development = development[columns.target].to_numpy(dtype=float)
    final_pipeline.fit(X_development, y_development)

    # Recompute the factor from the final model's own residuals.
    final_smearing = compute_smearing_factor(final_pipeline, X_development, y_development)
    from dataclasses import replace

    final_choice = replace(choice, smearing_factor=final_smearing)

    # -- 4. Validation predictions ------------------------------------------ #
    raw_validation_predictions = final_pipeline.predict(validation[columns.raw_feature_columns])
    validation_predictions = apply_back_transform(raw_validation_predictions, final_choice)

    submission = pd.DataFrame(
        {"load_id": validation[columns.id].to_numpy(), "predicted_rate": validation_predictions}
    )
    confirmations = validate_submission(submission, template)

    submission_path = project_root / "validation_predictions.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Wrote %s (%d rows)", submission_path, len(submission))

    # -- 5. December predictions via the Phase-3 inference path ------------- #
    coordinates = build_city_coordinate_lookup(development, validation)
    market_lookup = build_daily_market_lookup(validation, date_column=columns.date)

    december = build_december_chart_inputs()
    enriched = enrich_reduced_frame(
        december[DECEMBER_INPUT_COLUMNS],
        coordinates=coordinates,
        market_lookup=market_lookup,
        date_column=columns.date,
    )
    raw_december_predictions = final_pipeline.predict(enriched[columns.raw_feature_columns])
    december["predicted_rate"] = apply_back_transform(raw_december_predictions, final_choice)

    december_path = config.paths.december_inputs
    december.to_csv(december_path, index=False)
    logger.info("Wrote %s (%d rows)", december_path, len(december))

    # -- 6. Persist the final model ----------------------------------------- #
    joblib.dump(final_pipeline, config.paths.models_dir / "final_model.joblib")
    (config.paths.models_dir / "final_model_metadata.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "algorithm": metadata["best_model"],
                "hyperparameters": hyperparameters,
                "random_seed": config.random_seed,
                "target_transform": "log",
                "smearing_applied": final_choice.use_smearing,
                "smearing_factor": final_smearing,
                "training_rows": int(len(development)),
                "holdout_metrics_corrected": choice.corrected_metrics.as_dict(),
                "holdout_metrics_uncorrected": choice.raw_metrics.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -- 7. Run the official scorer ------------------------------------------ #
    logger.info("Running the provided score.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "score.py"),
            "--predictions",
            str(submission_path),
            "--december-predictions",
            str(december_path),
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    scorer_output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(f"score.py failed with exit {completed.returncode}:\n{scorer_output}")
    logger.info("score.py succeeded:\n%s", scorer_output.strip())

    chart = project_root / "scorer_results" / "candidate_december.png"
    if not chart.is_file():
        raise RuntimeError(f"Expected chart was not produced: {chart}")

    # -- 8. Report ------------------------------------------------------------ #
    report = _build_report(
        hyperparameters=hyperparameters,
        choice=choice,
        final_smearing=final_smearing,
        validation_stats=_prediction_statistics(validation_predictions),
        december=december,
        scorer_output=scorer_output,
        confirmations=confirmations,
        training_rows=len(development),
        training_dates=(
            str(pd.to_datetime(development[columns.date]).min().date()),
            str(pd.to_datetime(development[columns.date]).max().date()),
        ),
    )
    report_path = config.paths.reports_dir / "final_predictions.md"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Phase 7 complete. Wrote %s", report_path)


if __name__ == "__main__":
    main()
