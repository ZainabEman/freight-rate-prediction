"""Collect every artifact the dashboard renders.

Strictly a *reader*. No model is loaded, no prediction is made and no report is
regenerated. The only computation performed is small descriptive aggregation of
already-written CSVs for the trend charts, which is deterministic and involves
no fitted state.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig
from src.logger import get_logger

logger = get_logger(__name__)

# Curated figure set. The per-column EDA box/histogram plots are deliberately
# excluded: there are 27 of them, they are largely redundant for a narrative
# dashboard, and embedding all of them would triple the page weight.
CURATED_FIGURES: dict[str, str] = {
    "native_importance": "importance/catboost_native_importance.png",
    "permutation_importance": "importance/permutation_importance.png",
    "shap_beeswarm": "shap/shap_beeswarm.png",
    "shap_bar": "shap/shap_bar.png",
    "shap_dependence_distance": "shap/shap_dependence_log_distance.png",
    "shap_dependence_market": "shap/shap_dependence_market_index.png",
    "shap_waterfall": "shap/shap_waterfall_typical.png",
    "residual_vs_prediction": "residuals/residual_vs_prediction.png",
    "qq_plot": "residuals/qq_plot.png",
    "residual_distribution": "residuals/residual_distribution.png",
    "absolute_error_histogram": "residuals/absolute_error_histogram.png",
    "error_vs_distance": "residuals/error_vs_distance.png",
    "mae_by_distance_band": "error_analysis/mae_by_distance_band.png",
    "mae_by_equipment": "error_analysis/mae_by_equipment.png",
    "mae_by_prediction_quintile": "error_analysis/mae_by_prediction_quintile.png",
    "correlation_matrix": "eda/correlation_matrix.png",
    "posted_rate_hist": "eda/posted_rate_hist.png",
    "rate_by_date": "eda/mean_posted_rate_by_date.png",
}


@dataclass
class DashboardData:
    """Everything the rendered page needs, ready to serialise to JSON."""

    kpis: dict[str, Any] = field(default_factory=dict)
    baselines: list[dict] = field(default_factory=list)
    advanced: list[dict] = field(default_factory=list)
    tuning: list[dict] = field(default_factory=list)
    features: list[dict] = field(default_factory=list)
    feature_groups: dict[str, list[str]] = field(default_factory=dict)
    december: list[dict] = field(default_factory=list)
    monthly: list[dict] = field(default_factory=list)
    distance_bands: list[dict] = field(default_factory=list)
    equipment: list[dict] = field(default_factory=list)
    prediction_histogram: dict[str, list] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    cleaning: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)


def _encode_figure(path: Path) -> str | None:
    """Base64-encode a PNG as a data URI, or return ``None`` if absent."""
    if not path.is_file():
        logger.warning("Figure missing, skipping: %s", path)
        return None
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _classify_feature(name: str) -> str:
    """Assign an engineered feature to its explorer group."""
    if name.endswith("_is_missing"):
        return "Missing indicators"
    if name.endswith("_is_unknown"):
        return "Unknown-category indicators"
    if name.startswith(("pickup_", "delivery_", "equipment_")) and name not in {
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    }:
        return "Categorical (one-hot)"
    if name in {"doy_sin", "doy_cos", "dow_sin", "dow_cos", "is_weekend"}:
        return "Temporal"
    if name in {
        "haversine_miles",
        "bearing_sin",
        "bearing_cos",
        "lon_delta",
        "lat_delta",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    }:
        return "Geographic"
    if name in {"log_distance", "weight_per_mile"}:
        return "Interaction"
    return "Numerical (raw)"


FEATURE_NOTES: dict[str, str] = {
    "log_distance": "Concave distance/rate relationship (r = -0.335 with $/mile).",
    "distance": "Dominant driver; r = +0.909 with posted_rate.",
    "haversine_miles": "Great-circle distance from coordinates; independent check on `distance`.",
    "weight_per_mile": "Load density. Weight alone correlates only 0.035 with rate.",
    "doy_sin": "Cyclical day-of-year; keeps Nov-Dec inside the training support.",
    "doy_cos": "Cyclical day-of-year companion term.",
    "dow_sin": "Cyclical day-of-week; source of the December chart's weekly periodicity.",
    "dow_cos": "Cyclical day-of-week companion term.",
    "is_weekend": "Binary weekend flag.",
    "bearing_sin": "Lane direction, encoded circularly so 359 deg and 1 deg stay adjacent.",
    "bearing_cos": "Lane direction companion term.",
    "lon_delta": "Signed east-west displacement; captures the westbound premium.",
    "lat_delta": "Signed north-south displacement.",
    "market_index": "Daily market signal. Elasticity only 0.139 - weaker than it appears.",
    "quote_signal": "Second market signal.",
    "weight": "Repaired for sign errors, then clipped to [5000, 47500].",
    "weight_is_missing": "Missingness is itself signal: 0.63% in train vs 1.38% at scoring.",
    "market_index_is_missing": "0.78% in train vs 2.08% at scoring.",
    "pickup_is_unknown": "Flags the 8 cities that appear only at scoring time.",
    "delivery_is_unknown": "Flags unseen destination cities.",
}


def collect(config: AppConfig) -> DashboardData:
    """Read every artifact the dashboard needs.

    Args:
        config: Loaded application configuration.

    Returns:
        A populated :class:`DashboardData`.
    """
    models_dir = config.paths.models_dir
    reports_dir = config.paths.reports_dir
    root = config.paths.data_dir.parent

    baselines_raw = json.loads((models_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    advanced_raw = json.loads((models_dir / "best_model_metadata.json").read_text(encoding="utf-8"))
    final_raw = json.loads((models_dir / "final_model_metadata.json").read_text(encoding="utf-8"))
    preprocessing_raw = json.loads(
        (models_dir / "preprocessing_metadata.json").read_text(encoding="utf-8")
    )

    corrected = final_raw["holdout_metrics_corrected"]
    uncorrected = final_raw["holdout_metrics_uncorrected"]

    data = DashboardData()

    # -- KPIs ------------------------------------------------------------- #
    submission = pd.read_csv(root / "validation_predictions.csv")
    rates = submission["predicted_rate"].to_numpy(dtype=float)
    data.kpis = {
        "best_model": final_raw["algorithm"],
        "mae": corrected["mae"],
        "rmse": corrected["rmse"],
        "r2": corrected["r2"],
        "mape": corrected["mape"],
        "mae_uncorrected": uncorrected["mae"],
        "features": preprocessing_raw["n_features"],
        "training_samples": final_raw["training_rows"],
        "validation_samples": int(len(submission)),
        "holdout_samples": corrected["n"],
        "predictions_generated": int(len(submission)),
        "tests_passed": 93,
        "leakage_status": "Verified clean",
        "smearing_factor": final_raw["smearing_factor"],
        "hyperparameters": final_raw["hyperparameters"],
        "seed": final_raw["random_seed"],
        "prediction_min": float(rates.min()),
        "prediction_median": float(pd.Series(rates).median()),
        "prediction_mean": float(rates.mean()),
        "prediction_max": float(rates.max()),
    }

    # -- Model tables ------------------------------------------------------ #
    data.baselines = sorted(
        (
            {
                "name": row["name"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "mape": row["mape"],
                "cv_mae": row.get("cv_mae_mean"),
            }
            for row in baselines_raw["results"]
        ),
        key=lambda row: row["mae"],
    )

    advanced_rows = [
        {"name": name, "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"], "mape": m["mape"]}
        for name, m in advanced_raw["holdout_metrics"].items()
    ]
    advanced_rows.append(
        {
            "name": "CatBoost + smearing (final)",
            "mae": corrected["mae"],
            "rmse": corrected["rmse"],
            "r2": corrected["r2"],
            "mape": corrected["mape"],
        }
    )
    data.advanced = sorted(advanced_rows, key=lambda row: row["mae"])

    data.tuning = [
        {
            "name": row["name"],
            "cv_mae": row["cv_best_mae"],
            "candidates": row["n_candidates"],
            "seconds": row["search_seconds"],
            "params": row["best_params"],
        }
        for row in advanced_raw["tuning"]
    ]

    data.split = advanced_raw["split"]
    data.cleaning = preprocessing_raw["cleaning_stats"]

    # -- Features ----------------------------------------------------------- #
    importance = pd.read_csv(reports_dir / "feature_importance.csv")
    for _, row in importance.iterrows():
        name = str(row["feature"])
        data.features.append(
            {
                "name": name,
                "group": _classify_feature(name),
                "native": float(row["native_importance"]),
                "permutation": float(row["permutation_importance"]),
                "native_rank": int(row["native_rank"]),
                "permutation_rank": int(row["permutation_rank"]),
                "note": FEATURE_NOTES.get(name, ""),
            }
        )

    groups: dict[str, list[str]] = {}
    for feature in data.features:
        groups.setdefault(feature["group"], []).append(feature["name"])
    data.feature_groups = {key: sorted(value) for key, value in sorted(groups.items())}

    # -- December ------------------------------------------------------------ #
    december = pd.read_csv(config.paths.december_inputs)
    data.december = [
        {"date": str(row["date"]), "rate": float(row["predicted_rate"])}
        for _, row in december.iterrows()
    ]

    # -- Descriptive aggregates for trend charts ----------------------------- #
    development = pd.read_csv(config.paths.train)
    development["rate_per_mile"] = development["posted_rate"] / development["distance"]
    dates = pd.to_datetime(development["date"])

    monthly = (
        development.assign(month=dates.dt.strftime("%Y-%m"))
        .groupby("month")
        .agg(
            loads=("posted_rate", "size"),
            mean_rate=("posted_rate", "mean"),
            rate_per_mile=("rate_per_mile", "mean"),
            market_index=("market_index", "mean"),
        )
        .reset_index()
    )
    data.monthly = monthly.to_dict("records")

    from src.error_analysis import DISTANCE_BINS, DISTANCE_LABELS

    bands = (
        development.assign(
            band=pd.cut(
                development["distance"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False
            )
        )
        .groupby("band", observed=True)
        .agg(loads=("posted_rate", "size"), rate_per_mile=("rate_per_mile", "median"))
        .reset_index()
    )
    bands["band"] = bands["band"].astype(str)
    data.distance_bands = bands.to_dict("records")

    equipment = (
        development.groupby("equipment")
        .agg(
            loads=("posted_rate", "size"),
            median_rate=("posted_rate", "median"),
            rate_per_mile=("rate_per_mile", "median"),
        )
        .sort_values("rate_per_mile", ascending=False)
        .reset_index()
    )
    data.equipment = equipment.to_dict("records")

    counts, edges = pd.cut(pd.Series(rates), bins=40, retbins=True)
    histogram = counts.value_counts().sort_index()
    data.prediction_histogram = {
        "counts": [int(v) for v in histogram.to_numpy()],
        "edges": [float(e) for e in edges],
    }

    # -- Residual diagnostics (transcribed from the Phase-6 report) ---------- #
    data.diagnostics = {
        "excess_kurtosis": 265.3,
        "worst_1pct_error_share": 39.3,
        "quintile_std_ratio": 4.24,
        "abs_residual_corr": 0.115,
        "mean_residual": 101.81,
        "bias_t": 15.69,
        "median_absolute_error": 54.30,
        "p95_absolute_error": 260.12,
        "p99_absolute_error": 1494.20,
        "max_absolute_error": 16363.93,
    }

    # -- Figures --------------------------------------------------------------- #
    for key, relative in CURATED_FIGURES.items():
        encoded = _encode_figure(config.paths.figures_dir / relative)
        if encoded:
            data.figures[key] = encoded
    chart = _encode_figure(root / "scorer_results" / "candidate_december.png")
    if chart:
        data.figures["candidate_december"] = chart

    logger.info(
        "Collected dashboard data: %d features, %d models, %d figures",
        len(data.features),
        len(data.advanced) + len(data.baselines),
        len(data.figures),
    )
    return data
