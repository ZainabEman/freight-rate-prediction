"""Segment error analysis and residual diagnostics.

All analysis runs on the **temporal holdout** (2025-09-01 to 2025-10-31), which
the Phase-5 model never saw: ``RandomizedSearchCV`` was fitted on rows up to
2025-08-31 only. Errors reported here are therefore genuine out-of-sample
errors, not in-sample optimism.

Sign convention throughout: ``residual = actual - predicted``. A positive
residual means the model **under-priced** the load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 150

# Distance bands chosen to match freight operating categories rather than
# equal-width bins: short-haul, regional, medium, long-haul, transcontinental.
DISTANCE_BINS = [0, 250, 500, 1000, 2000, np.inf]
DISTANCE_LABELS = ["<250mi", "250-500mi", "500-1000mi", "1000-2000mi", ">2000mi"]

WEIGHT_BINS = [0, 15000, 25000, 35000, 45000, np.inf]
WEIGHT_LABELS = ["<15k lb", "15-25k lb", "25-35k lb", "35-45k lb", ">45k lb"]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class ErrorFrame:
    """Holdout rows enriched with predictions and error columns."""

    frame: pd.DataFrame

    @property
    def mae(self) -> float:
        """Overall mean absolute error in dollars."""
        return float(self.frame["absolute_error"].mean())

    @property
    def bias(self) -> float:
        """Mean residual; positive means systematic under-pricing."""
        return float(self.frame["residual"].mean())


def build_error_frame(
    holdout: pd.DataFrame, predictions: np.ndarray, *, target: str = "posted_rate"
) -> ErrorFrame:
    """Attach predictions, residuals and segment columns to the holdout.

    Args:
        holdout: Raw holdout rows.
        predictions: Model predictions in dollars, aligned row-wise.
        target: Name of the target column.

    Returns:
        A populated :class:`ErrorFrame`.
    """
    frame = holdout.copy().reset_index(drop=True)
    frame["predicted"] = np.asarray(predictions, dtype=float)
    frame["residual"] = frame[target] - frame["predicted"]
    frame["absolute_error"] = frame["residual"].abs()
    # Percentage error is the scale-free view; the target is strictly positive.
    frame["percentage_error"] = 100.0 * frame["residual"] / frame[target]
    frame["absolute_percentage_error"] = frame["percentage_error"].abs()

    dates = pd.to_datetime(frame["date"])
    frame["month"] = dates.dt.strftime("%Y-%m")
    frame["day_of_week"] = dates.dt.day_name()

    frame["distance_band"] = pd.cut(
        frame["distance"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False
    )
    # Weight is repaired inside the pipeline, so mirror that repair here purely
    # for banding; the raw column is left untouched.
    frame["weight_band"] = pd.cut(
        frame["weight"].abs(), bins=WEIGHT_BINS, labels=WEIGHT_LABELS, right=False
    )
    frame["prediction_quintile"] = pd.qcut(
        frame["predicted"], q=5, labels=[f"Q{i}" for i in range(1, 6)]
    )

    logger.info(
        "Built error frame: %d rows | MAE=%.2f | bias=%.2f",
        len(frame),
        frame["absolute_error"].mean(),
        frame["residual"].mean(),
    )
    return ErrorFrame(frame=frame)


def segment_summary(
    errors: ErrorFrame, by: str, *, min_rows: int = 1, sort_by: str = "MAE"
) -> pd.DataFrame:
    """Aggregate error statistics within one segment column.

    Args:
        errors: The enriched holdout.
        by: Column to group on.
        min_rows: Drop groups smaller than this.
        sort_by: Column to sort the result by, descending.

    Returns:
        A per-segment summary frame.
    """
    grouped = errors.frame.groupby(by, observed=True).agg(
        rows=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
        MAPE=("absolute_percentage_error", "mean"),
        bias=("residual", "mean"),
        median_rate=("posted_rate", "median"),
    )
    grouped = grouped[grouped["rows"] >= min_rows]
    return grouped.sort_values(sort_by, ascending=False).reset_index()


def outlier_rows(errors: ErrorFrame, *, top_n: int = 20) -> pd.DataFrame:
    """Return the worst absolute-error rows.

    Args:
        errors: The enriched holdout.
        top_n: Number of rows to return.

    Returns:
        The worst-predicted loads with their key features.
    """
    columns = [
        "load_id",
        "pickup",
        "delivery",
        "equipment",
        "distance",
        "weight",
        "date",
        "market_index",
        "posted_rate",
        "predicted",
        "residual",
        "absolute_percentage_error",
    ]
    available = [column for column in columns if column in errors.frame.columns]
    return errors.frame.nlargest(top_n, "absolute_error")[available].reset_index(drop=True)


def diagnose_residuals(errors: ErrorFrame) -> dict[str, float | bool | str]:
    """Measure heteroscedasticity, tail weight and bias.

    Returns:
        A dictionary of measured diagnostics with boolean verdicts.
    """
    frame = errors.frame
    residual = frame["residual"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)

    # Heteroscedasticity: compare residual spread in the lowest and highest
    # prediction quintiles.
    quintile_std = frame.groupby("prediction_quintile", observed=True)["residual"].std()
    spread_ratio = float(quintile_std.max() / quintile_std.min())

    # Correlation between |residual| and prediction is the direct test.
    absolute_vs_prediction = float(
        np.corrcoef(np.abs(residual), predicted)[0, 1]
    )

    # Tail weight: excess kurtosis, and how much of total absolute error comes
    # from the worst 1% of rows.
    centred = residual - residual.mean()
    kurtosis = float(np.mean(centred**4) / (np.mean(centred**2) ** 2) - 3.0)
    absolute = np.abs(residual)
    worst_one_percent = int(max(1, round(0.01 * len(absolute))))
    tail_share = float(
        np.sort(absolute)[-worst_one_percent:].sum() / absolute.sum() * 100.0
    )

    mean_residual = float(residual.mean())
    median_residual = float(np.median(residual))
    # Standard error of the mean residual gives a scale for judging bias.
    standard_error = float(residual.std(ddof=1) / np.sqrt(len(residual)))
    bias_t = mean_residual / standard_error if standard_error else 0.0

    return {
        "mean_residual": mean_residual,
        "median_residual": median_residual,
        "residual_std": float(residual.std(ddof=1)),
        "bias_t_statistic": float(bias_t),
        "bias_is_significant": bool(abs(bias_t) > 2.0),
        "quintile_std_ratio": spread_ratio,
        "abs_residual_vs_prediction_corr": absolute_vs_prediction,
        "heteroscedastic": bool(spread_ratio > 1.5 or abs(absolute_vs_prediction) > 0.2),
        "excess_kurtosis": kurtosis,
        "worst_1pct_share_of_total_error": tail_share,
        "heavy_tailed": bool(kurtosis > 3.0),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _save(figure: plt.Figure, path: Path) -> Path:
    """Write a figure to disk and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure: %s", path)
    return path


def plot_residual_diagnostics(errors: ErrorFrame, *, output_dir: Path) -> list[Path]:
    """Save the six residual diagnostic figures.

    Args:
        errors: The enriched holdout.
        output_dir: Directory for ``figures/residuals``.

    Returns:
        Paths of the saved figures.
    """
    frame = errors.frame
    residual = frame["residual"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    saved: list[Path] = []

    # 1. Residual vs prediction.
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.scatter(predicted, residual, s=6, alpha=0.25, color="#1f6f78", edgecolors="none")
    axis.axhline(0, color="#b4552d", linewidth=1.2)
    axis.set_xlabel("Predicted rate ($)")
    axis.set_ylabel("Residual (actual - predicted, $)")
    axis.set_title("Residual vs prediction", loc="left", fontweight="bold")
    axis.grid(alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    saved.append(_save(figure, output_dir / "residual_vs_prediction.png"))

    # 2. Residual distribution.
    figure, axis = plt.subplots(figsize=(9, 5))
    limit = float(np.percentile(np.abs(residual), 99))
    axis.hist(residual, bins=120, range=(-limit, limit), color="#1f6f78")
    axis.axvline(0, color="#b4552d", linewidth=1.2)
    axis.set_xlabel("Residual ($, central 99% shown)")
    axis.set_ylabel("Loads")
    axis.set_title("Residual distribution", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    saved.append(_save(figure, output_dir / "residual_distribution.png"))

    # 3. QQ plot against the normal distribution.
    standardised = (residual - residual.mean()) / residual.std(ddof=1)
    ordered = np.sort(standardised)
    probabilities = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
    # Inverse normal CDF via the error function, avoiding a scipy dependency.
    from math import sqrt

    theoretical = sqrt(2.0) * _erfinv(2.0 * probabilities - 1.0)
    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    axis.scatter(theoretical, ordered, s=5, alpha=0.4, color="#1f6f78", edgecolors="none")
    line = np.linspace(theoretical.min(), theoretical.max(), 10)
    axis.plot(line, line, color="#b4552d", linewidth=1.3, label="normal reference")
    axis.set_xlabel("Theoretical normal quantiles")
    axis.set_ylabel("Standardised residual quantiles")
    axis.set_title("QQ plot of residuals", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    saved.append(_save(figure, output_dir / "qq_plot.png"))

    # 4. Absolute error histogram.
    figure, axis = plt.subplots(figsize=(9, 5))
    absolute = frame["absolute_error"].to_numpy(dtype=float)
    axis.hist(absolute, bins=120, range=(0, float(np.percentile(absolute, 99))), color="#4a7c59")
    axis.set_xlabel("Absolute error ($, central 99% shown)")
    axis.set_ylabel("Loads")
    axis.set_title("Absolute error distribution", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    saved.append(_save(figure, output_dir / "absolute_error_histogram.png"))

    # 5 & 6. Error against the two dominant physical drivers.
    for column, label, filename in [
        ("distance", "Distance (miles)", "error_vs_distance.png"),
        ("weight", "Weight (lb)", "error_vs_weight.png"),
    ]:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        values = frame[column].abs().to_numpy(dtype=float)
        axis.scatter(values, residual, s=6, alpha=0.25, color="#1f6f78", edgecolors="none")
        axis.axhline(0, color="#b4552d", linewidth=1.2)
        # Rolling median makes any systematic trend visible through the scatter.
        order = np.argsort(values)
        window = max(50, len(values) // 40)
        rolling = pd.Series(residual[order]).rolling(window, center=True, min_periods=10).median()
        axis.plot(values[order], rolling, color="#000000", linewidth=1.6, label="rolling median")
        axis.set_xlabel(label)
        axis.set_ylabel("Residual ($)")
        axis.set_title(f"Residual vs {column}", loc="left", fontweight="bold")
        axis.legend(frameon=False)
        axis.grid(alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
        saved.append(_save(figure, output_dir / filename))

    return saved


def _erfinv(y: np.ndarray) -> np.ndarray:
    """Inverse error function via Newton refinement of a rational seed.

    Avoids adding SciPy purely for QQ-plot quantiles.

    Args:
        y: Values in ``(-1, 1)``.

    Returns:
        ``erfinv(y)``.
    """
    from math import pi, sqrt

    y = np.clip(y, -0.999999, 0.999999)
    a = 0.147
    ln_term = np.log(1.0 - y**2)
    first = 2.0 / (pi * a) + ln_term / 2.0
    seed = np.sign(y) * np.sqrt(np.sqrt(first**2 - ln_term / a) - first)

    # Two Newton steps against the true erf are enough for plotting precision.
    from math import erf

    erf_vectorised = np.vectorize(erf)
    for _ in range(2):
        error = erf_vectorised(seed) - y
        derivative = 2.0 / sqrt(pi) * np.exp(-(seed**2))
        seed = seed - error / derivative
    return seed


def plot_segment_errors(errors: ErrorFrame, *, output_dir: Path, top_cities: int = 15) -> list[Path]:
    """Save the per-segment MAE bar charts.

    Args:
        errors: The enriched holdout.
        output_dir: Directory for ``figures/error_analysis``.
        top_cities: Number of worst cities to chart.

    Returns:
        Paths of the saved figures.
    """
    saved: list[Path] = []

    ordered_segments = {
        "equipment": (None, "mae_by_equipment.png", "Equipment"),
        "distance_band": (DISTANCE_LABELS, "mae_by_distance_band.png", "Distance band"),
        "weight_band": (WEIGHT_LABELS, "mae_by_weight_band.png", "Weight band"),
        "month": (None, "mae_by_month.png", "Month"),
        "day_of_week": (DAY_ORDER, "mae_by_day_of_week.png", "Day of week"),
        "prediction_quintile": (
            [f"Q{i}" for i in range(1, 6)],
            "mae_by_prediction_quintile.png",
            "Prediction quintile (Q1 = cheapest)",
        ),
    }

    for column, (order, filename, label) in ordered_segments.items():
        summary = segment_summary(errors, column)
        if order is not None:
            summary["__order"] = summary[column].astype(str).map(
                {name: index for index, name in enumerate(order)}
            )
            summary = summary.sort_values("__order").drop(columns="__order")
        else:
            summary = summary.sort_values(column)

        figure, axis = plt.subplots(figsize=(9, 5))
        positions = np.arange(len(summary))
        axis.bar(positions, summary["MAE"], color="#1f6f78")
        axis.set_xticks(positions)
        axis.set_xticklabels(summary[column].astype(str), rotation=30, ha="right")
        axis.axhline(
            errors.mae, color="#b4552d", linewidth=1.3, linestyle="--", label="overall MAE"
        )
        for position, (value, rows) in enumerate(zip(summary["MAE"], summary["rows"])):
            axis.text(position, value, f"n={rows:,}", ha="center", va="bottom", fontsize=7.5)
        axis.set_ylabel("MAE ($)")
        axis.set_xlabel(label)
        axis.set_title(f"Holdout MAE by {label.lower()}", loc="left", fontweight="bold")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
        saved.append(_save(figure, output_dir / filename))

    # Worst cities by MAE, restricted to segments with enough rows to be stable.
    for column, filename in [("pickup", "worst_pickup_cities.png"), ("delivery", "worst_delivery_cities.png")]:
        summary = segment_summary(errors, column, min_rows=30).head(top_cities)
        figure, axis = plt.subplots(figsize=(9, max(4.0, 0.34 * len(summary))))
        positions = np.arange(len(summary))
        axis.barh(positions, summary["MAE"], color="#b4552d")
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [f"{city} (n={rows:,})" for city, rows in zip(summary[column], summary["rows"])],
            fontsize=9,
        )
        axis.invert_yaxis()
        axis.axvline(errors.mae, color="#1f6f78", linewidth=1.3, linestyle="--", label="overall MAE")
        axis.set_xlabel("MAE ($)")
        axis.set_title(
            f"Worst {top_cities} {column} cities by MAE (min 30 loads)", loc="left", fontweight="bold"
        )
        axis.legend(frameon=False)
        axis.grid(axis="x", alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
        saved.append(_save(figure, output_dir / filename))

    return saved
