"""Phase-6 entry point: explainability, error analysis and business insights.

Loads the persisted Phase-5 model and analyses it. Trains nothing, tunes
nothing, and produces no submission files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config, set_global_seed
from src.data_loader import load_csv_safe
from src.error_analysis import (
    build_error_frame,
    diagnose_residuals,
    outlier_rows,
    plot_residual_diagnostics,
    plot_segment_errors,
    segment_summary,
)
from src.explainability import (
    build_importance_table,
    compute_shap_values,
    load_model_parts,
    native_feature_importance,
    permutation_feature_importance,
    plot_importances,
    plot_shap_figures,
)
from src.logger import get_logger
from src.reporting_phase6 import (
    build_business_insights_report,
    build_error_analysis_report,
    build_explainability_report,
    build_residual_section,
)
from src.splitting import temporal_train_holdout_split

logger = get_logger(__name__)

N_DEPENDENCE_PLOTS = 6
SHAP_SAMPLE_SIZE = 2_000


def _business_evidence(development: pd.DataFrame) -> dict[str, object]:
    """Measure the business-facing effects quoted in the insights report."""
    frame = development.copy()
    frame["rate_per_mile"] = frame["posted_rate"] / frame["distance"]
    dates = pd.to_datetime(frame["date"])
    frame["month"] = dates.dt.strftime("%Y-%m")
    frame["day_of_week"] = dates.dt.day_name()

    equipment_effects = (
        frame.groupby("equipment")
        .agg(
            loads=("posted_rate", "size"),
            median_rate=("posted_rate", "median"),
            median_rate_per_mile=("rate_per_mile", "median"),
            mean_distance=("distance", "mean"),
        )
        .sort_values("median_rate_per_mile", ascending=False)
        .reset_index()
    )

    from src.error_analysis import DISTANCE_BINS, DISTANCE_LABELS

    frame["distance_band"] = pd.cut(
        frame["distance"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False
    )
    distance_effects = (
        frame.groupby("distance_band", observed=True)
        .agg(
            loads=("posted_rate", "size"),
            median_rate=("posted_rate", "median"),
            median_rate_per_mile=("rate_per_mile", "median"),
        )
        .reset_index()
    )

    frame["market_quintile"] = pd.qcut(
        frame["market_index"], q=5, labels=[f"Q{i}" for i in range(1, 6)]
    )
    market_effects = (
        frame.groupby("market_quintile", observed=True)
        .agg(
            loads=("posted_rate", "size"),
            mean_market_index=("market_index", "mean"),
            median_rate_per_mile=("rate_per_mile", "median"),
        )
        .reset_index()
    )

    monthly = (
        frame.groupby("month")
        .agg(
            loads=("posted_rate", "size"),
            mean_rate=("posted_rate", "mean"),
            mean_rate_per_mile=("rate_per_mile", "mean"),
            mean_market_index=("market_index", "mean"),
        )
        .reset_index()
    )

    weekday_effects = (
        frame.groupby("day_of_week")
        .agg(loads=("posted_rate", "size"), mean_rate_per_mile=("rate_per_mile", "mean"))
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        .reset_index()
    )

    lanes = (
        frame.groupby(["pickup", "delivery"])
        .agg(
            loads=("posted_rate", "size"),
            mean_distance=("distance", "mean"),
            median_rate_per_mile=("rate_per_mile", "median"),
        )
        .reset_index()
    )
    top_lanes = (
        lanes[lanes["loads"] >= 30]
        .sort_values("median_rate_per_mile", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    daily = frame.groupby(dates.dt.date).agg(
        market_index=("market_index", "mean"), rate_per_mile=("rate_per_mile", "mean")
    )

    # Elasticity of rate-per-mile with respect to market index, in logs. This is
    # the sensitivity measure; the daily correlation above only measures
    # co-movement and materially overstates how much market index moves price.
    log_market = np.log(daily["market_index"].to_numpy(dtype=float))
    log_rpm = np.log(daily["rate_per_mile"].to_numpy(dtype=float))
    market_elasticity = float(np.polyfit(log_market, log_rpm, 1)[0])

    # Same relationship restricted to the December chart's distance band, where
    # mileage is effectively held constant.
    band = frame[(frame["distance"] > 340) & (frame["distance"] < 380)]
    within_band_market_corr = float(band["market_index"].corr(band["rate_per_mile"]))

    correlations = {
        "distance": float(frame["distance"].corr(frame["posted_rate"])),
        "market_index": float(frame["market_index"].corr(frame["posted_rate"])),
        "quote_signal": float(frame["quote_signal"].corr(frame["posted_rate"])),
        "weight": float(frame["weight"].abs().corr(frame["posted_rate"])),
        "delivery_lon": float(frame["delivery_lon"].corr(frame["posted_rate"])),
        "distance_vs_rpm": float(frame["distance"].corr(frame["rate_per_mile"])),
        "daily_market_vs_rpm": float(daily["market_index"].corr(daily["rate_per_mile"])),
        "market_elasticity": market_elasticity,
        "within_band_market_corr": within_band_market_corr,
    }

    return {
        "equipment_effects": equipment_effects,
        "distance_effects": distance_effects,
        "market_effects": market_effects,
        "monthly": monthly,
        "weekday_effects": weekday_effects,
        "top_lanes": top_lanes,
        "correlations": correlations,
    }


def main() -> None:
    """Run the full Phase-6 analysis."""
    config = load_config()
    set_global_seed(config.random_seed)

    figures_dir = config.paths.figures_dir
    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load model and holdout ----------------------------------------- #
    parts = load_model_parts(config.paths.models_dir / "best_model.joblib")
    development = load_csv_safe(config.paths.train, label="train")
    split = temporal_train_holdout_split(
        development,
        date_column=config.columns.date,
        holdout_start=config.split.holdout_start,
    )
    holdout = split.holdout.reset_index(drop=True)

    X_holdout = holdout[config.columns.raw_feature_columns]
    predictions = np.asarray(parts.pipeline.predict(X_holdout), dtype=float)
    X_transformed = parts.preprocessor.transform(X_holdout)
    y_log = np.log(holdout[config.columns.target].to_numpy(dtype=float))

    # -- 2. Feature importance --------------------------------------------- #
    native = native_feature_importance(parts)
    permutation = permutation_feature_importance(
        parts, X_transformed, y_log, n_repeats=5, seed=config.random_seed
    )
    importance = build_importance_table(native, permutation)
    importance_path = reports_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    logger.info("Wrote %s", importance_path)

    importance_figures = plot_importances(
        native, permutation, output_dir=figures_dir / "importance"
    )

    # -- 3. SHAP ------------------------------------------------------------ #
    explanation, sample = compute_shap_values(
        parts, X_transformed, sample_size=SHAP_SAMPLE_SIZE, seed=config.random_seed
    )
    mean_absolute_shap = np.abs(explanation.values).mean(axis=0)
    shap_ranking = (
        pd.DataFrame({"feature": parts.feature_names, "mean_abs_shap": mean_absolute_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Dependence plots for the strongest continuous drivers.
    top_features = [
        name
        for name in shap_ranking["feature"].tolist()
        if sample[name].nunique() > 2
    ][:N_DEPENDENCE_PLOTS]

    # Waterfall plots for three genuinely representative loads: a typical case
    # and the two worst failures in each direction.
    sample_positions = {index: position for position, index in enumerate(sample.index)}
    residual = holdout[config.columns.target].to_numpy(dtype=float) - predictions
    candidate_indices = list(sample.index)
    candidate_residuals = pd.Series(residual[candidate_indices], index=candidate_indices)
    waterfall_indices = {
        "typical": sample_positions[candidate_residuals.abs().idxmin()],
        "worst_underprediction": sample_positions[candidate_residuals.idxmax()],
        "worst_overprediction": sample_positions[candidate_residuals.idxmin()],
    }

    shap_figures = plot_shap_figures(
        explanation,
        sample,
        output_dir=figures_dir / "shap",
        top_features=top_features,
        waterfall_indices=waterfall_indices,
    )

    # -- 4 & 5. Error and residual analysis --------------------------------- #
    errors = build_error_frame(holdout, predictions, target=config.columns.target)
    diagnostics = diagnose_residuals(errors)

    segments = {
        "equipment": segment_summary(errors, "equipment"),
        "distance_band": segment_summary(errors, "distance_band"),
        "weight_band": segment_summary(errors, "weight_band"),
        "month": segment_summary(errors, "month"),
        "day_of_week": segment_summary(errors, "day_of_week"),
        "prediction_quintile": segment_summary(errors, "prediction_quintile"),
        "pickup": segment_summary(errors, "pickup", min_rows=30).head(15),
        "delivery": segment_summary(errors, "delivery", min_rows=30).head(15),
    }
    worst_rows = outlier_rows(errors, top_n=20)

    residual_figures = plot_residual_diagnostics(errors, output_dir=figures_dir / "residuals")
    segment_figures = plot_segment_errors(errors, output_dir=figures_dir / "error_analysis")

    # -- 6 & 7. Reports ------------------------------------------------------ #
    def relative(paths: list) -> list[str]:
        return [str(path.relative_to(config.paths.figures_dir.parent)) for path in paths]

    explainability_report = build_explainability_report(
        importance=importance,
        native_top=native,
        permutation_top=permutation,
        shap_ranking=shap_ranking,
        model_name=type(parts.regressor).__name__,
        n_features=len(parts.feature_names),
        n_shap_rows=len(sample),
        figure_paths=relative(importance_figures + shap_figures),
    )
    (reports_dir / "explainability_report.md").write_text(explainability_report, encoding="utf-8")

    error_report = build_error_analysis_report(
        errors=errors,
        diagnostics=diagnostics,
        segments=segments,
        worst_rows=worst_rows,
        figure_paths=relative(residual_figures + segment_figures),
        holdout_dates=(
            split.summary["holdout_date_min"],
            split.summary["holdout_date_max"],
        ),
    ) + "\n" + build_residual_section(diagnostics)
    (reports_dir / "error_analysis.md").write_text(error_report, encoding="utf-8")

    evidence = _business_evidence(development)
    insights_report = build_business_insights_report(
        development=development,
        errors=errors,
        shap_ranking=shap_ranking,
        equipment_effects=evidence["equipment_effects"],
        distance_effects=evidence["distance_effects"],
        market_effects=evidence["market_effects"],
        monthly=evidence["monthly"],
        weekday_effects=evidence["weekday_effects"],
        top_lanes=evidence["top_lanes"],
        correlations=evidence["correlations"],
    )
    (reports_dir / "business_insights.md").write_text(insights_report, encoding="utf-8")

    total_figures = len(
        importance_figures + shap_figures + residual_figures + segment_figures
    )
    logger.info(
        "Phase 6 complete: 3 reports, 1 CSV, %d figures. Holdout MAE=$%.2f",
        total_figures,
        errors.mae,
    )


if __name__ == "__main__":
    main()
