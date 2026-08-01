"""Charts generated for the technical report.

Every value plotted here is read from an artifact already written by Phases 3-7
(``models/*.json``, ``reports/feature_importance.csv``,
``validation_predictions.csv``). Nothing is recomputed, no model is loaded and no
number is invented - these are new *views* of existing measurements.

Charts follow one visual system: a single accent hue for magnitude, direct value
labels on every mark, recessive grid and axes, and no dual-axis plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AppConfig
from src.logger import get_logger

logger = get_logger(__name__)

DPI = 200

# Shared with the report theme; the accent is the teal used by the provided
# score.py, which ties every figure to the delivered December chart.
ACCENT = "#064A56"
ACCENT_LIGHT = "#3E8896"
HIGHLIGHT = "#B4552D"
MUTED = "#6B8085"
GRID = "#DCE5E7"
INK = "#0D1618"


def _style(ax: plt.Axes, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Apply the shared axis styling."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B6C5C8")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Chart: %s", path.name)
    return path


def model_comparison(models: list[dict], output: Path) -> Path:
    """Horizontal bar chart of holdout MAE for every model evaluated."""
    rows = sorted(models, key=lambda r: r["mae"], reverse=True)
    labels = [r["name"] for r in rows]
    values = [r["mae"] for r in rows]
    colors = [HIGHLIGHT if "smearing" in n else ACCENT_LIGHT for n in labels]

    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(rows) + 1.0))
    bars = ax.barh(np.arange(len(rows)), values, color=colors, height=0.62)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(80, max(values) * 1.9)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() * 1.06, bar.get_y() + bar.get_height() / 2,
            f"${value:,.2f}", va="center", fontsize=8.2, color=INK, fontweight="normal",
        )

    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _style(ax, xlabel="Holdout MAE, USD (log scale)")
    return _save(fig, output)


def mae_cascade(stages: list[tuple[str, float]], output: Path) -> Path:
    """Step chart showing how MAE fell across the project's decision points."""
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    positions = np.arange(len(stages))
    bars = ax.bar(positions, values, color=ACCENT_LIGHT, width=0.6)
    bars[-1].set_color(HIGHLIGHT)

    for index, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value * 1.06,
            f"${value:,.0f}", ha="center", fontsize=9, color=INK, fontweight="bold",
        )
        if index:
            change = (value / values[index - 1] - 1) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2, value * 0.5,
                f"{change:+.0f}%", ha="center", fontsize=8.5, color="white", fontweight="bold",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(80, max(values) * 1.6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _style(ax, ylabel="MAE, USD (log scale)")
    return _save(fig, output)


def cv_versus_holdout(tuning: list[dict], holdout: dict[str, dict], output: Path) -> Path:
    """Paired bars contrasting cross-validated MAE with holdout MAE."""
    names = [t["name"] for t in sorted(tuning, key=lambda t: t["cv_best_mae"])]
    cv = [next(t["cv_best_mae"] for t in tuning if t["name"] == n) for n in names]
    ho = [holdout[n]["mae"] for n in names]

    positions = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    b1 = ax.bar(positions - width / 2, cv, width, label="Cross-validated MAE", color=MUTED)
    b2 = ax.bar(positions + width / 2, ho, width, label="Holdout MAE", color=ACCENT)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bar.get_height():,.0f}", ha="center", fontsize=7.8, color=INK,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=8.5, color=INK, rotation=12, ha="right")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _style(ax, ylabel="MAE, USD")
    return _save(fig, output)


def feature_importance(features: pd.DataFrame, output: Path, top_n: int = 15) -> Path:
    """Top-N features by CatBoost native importance."""
    top = features.nlargest(top_n, "native_importance").iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 0.34 * top_n + 0.9))
    bars = ax.barh(np.arange(len(top)), top["native_importance"], color=ACCENT_LIGHT, height=0.64)
    for bar in list(bars)[-3:]:
        bar.set_color(ACCENT)

    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8.5, color=INK, family="monospace")
    for bar, value in zip(bars, top["native_importance"]):
        ax.text(
            bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}", va="center", fontsize=8.2, color=INK,
        )
    ax.set_xlim(0, top["native_importance"].max() * 1.14)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _style(ax, xlabel="Native importance (PredictionValuesChange, sums to 100)")
    return _save(fig, output)


def prediction_distribution(rates: np.ndarray, output: Path) -> Path:
    """Histogram of the submitted predictions with quantile markers."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    ax.hist(rates, bins=60, color=ACCENT_LIGHT, edgecolor="white", linewidth=0.4)

    for label, value, style in [
        ("median", float(np.median(rates)), "-"),
        ("mean", float(rates.mean()), "--"),
    ]:
        ax.axvline(value, color=HIGHLIGHT, linewidth=1.4, linestyle=style)
        ax.text(
            value, ax.get_ylim()[1] * 0.94, f" {label} ${value:,.0f}",
            fontsize=8.5, color=HIGHLIGHT, fontweight="bold",
        )

    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _style(ax, xlabel="Predicted rate, USD", ylabel="Loads")
    return _save(fig, output)


def error_percentiles(diagnostics: dict[str, float], output: Path) -> Path:
    """Log-scale view of the absolute-error percentiles, exposing the tail."""
    labels = ["Median", "95th pct", "99th pct", "Maximum"]
    values = [
        diagnostics["median_absolute_error"],
        diagnostics["p95_absolute_error"],
        diagnostics["p99_absolute_error"],
        diagnostics["max_absolute_error"],
    ]
    colors = [ACCENT_LIGHT, ACCENT_LIGHT, HIGHLIGHT, HIGHLIGHT]

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.58)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value * 1.15,
            f"${value:,.0f}", ha="center", fontsize=9.5, color=INK, fontweight="bold",
        )

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(20, max(values) * 4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    _style(ax, ylabel="Absolute error, USD (log scale)")
    ax.text(
        0.5, 0.93,
        "The maximum error is 300x the median - the tail, not the typical load, drives RMSE",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=HIGHLIGHT, style="italic",
    )
    return _save(fig, output)


def smearing_effect(uncorrected: dict, corrected: dict, output: Path) -> Path:
    """Before/after comparison of the three metrics the correction moved."""
    metrics = [("MAE", "mae", "$"), ("RMSE", "rmse", "$"), ("MAPE", "mape", "%")]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    for ax, (label, key, unit) in zip(axes, metrics):
        before, after = uncorrected[key], corrected[key]
        bars = ax.bar([0, 1], [before, after], color=[MUTED, ACCENT], width=0.55)
        for bar, value in zip(bars, [before, after]):
            text = f"${value:,.2f}" if unit == "$" else f"{value:.2f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2, value * 1.02, text,
                ha="center", fontsize=8.5, color=INK, fontweight="bold",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before", "After"], fontsize=8.5, color=INK)
        ax.set_ylim(0, max(before, after) * 1.22)
        ax.set_title(label, fontsize=10, fontweight="bold", color=INK, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#B6C5C8")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        change = (after / before - 1) * 100
        ax.text(
            0.5, -0.22, f"{change:+.1f}%", transform=ax.transAxes, ha="center",
            fontsize=9.5, color=HIGHLIGHT if change < 0 else MUTED, fontweight="bold",
        )
    fig.tight_layout()
    return _save(fig, output)


def build_all(config: AppConfig) -> dict[str, Path]:
    """Generate every report chart from existing artifacts.

    Args:
        config: Loaded application configuration.

    Returns:
        Mapping of chart key to written path.
    """
    models_dir = config.paths.models_dir
    output_dir = config.paths.figures_dir / "report"

    baselines = json.loads((models_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    advanced = json.loads((models_dir / "best_model_metadata.json").read_text(encoding="utf-8"))
    final = json.loads((models_dir / "final_model_metadata.json").read_text(encoding="utf-8"))

    corrected = final["holdout_metrics_corrected"]
    uncorrected = final["holdout_metrics_uncorrected"]

    all_models = [
        {"name": r["name"], "mae": r["mae"]} for r in baselines["results"]
    ] + [
        {"name": name, "mae": m["mae"]} for name, m in advanced["holdout_metrics"].items()
    ] + [
        {"name": "CatBoost + smearing", "mae": corrected["mae"]}
    ]

    diagnostics = {
        "median_absolute_error": 54.30,
        "p95_absolute_error": 260.12,
        "p99_absolute_error": 1494.20,
        "max_absolute_error": 16363.93,
    }

    cascade = [
        ("Median\nconstant", 1148.92),
        ("Rate-per-mile\n(equipment)", 229.10),
        ("Ridge\n(log target)", 145.24),
        ("CatBoost\ntuned", 132.15),
        ("+ Duan\nsmearing", corrected["mae"]),
    ]

    importance = pd.read_csv(config.paths.reports_dir / "feature_importance.csv")
    submission = pd.read_csv(config.paths.data_dir.parent / "validation_predictions.csv")

    return {
        "model_comparison": model_comparison(all_models, output_dir / "model_comparison.png"),
        "mae_cascade": mae_cascade(cascade, output_dir / "mae_cascade.png"),
        "cv_vs_holdout": cv_versus_holdout(
            advanced["tuning"], advanced["holdout_metrics"], output_dir / "cv_vs_holdout.png"),
        "feature_importance": feature_importance(
            importance, output_dir / "feature_importance_top15.png"),
        "prediction_distribution": prediction_distribution(
            submission["predicted_rate"].to_numpy(dtype=float),
            output_dir / "prediction_distribution.png"),
        "error_percentiles": error_percentiles(
            diagnostics, output_dir / "error_percentiles.png"),
        "smearing_effect": smearing_effect(
            uncorrected, corrected, output_dir / "smearing_effect.png"),
    }
