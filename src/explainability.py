"""Feature importance and SHAP explainability for the selected model.

Nothing here retrains or retunes. The persisted Phase-5 pipeline is loaded and
interrogated as-is.

**Interpretation caveat that applies to every output in this module:** the model
predicts ``log(posted_rate)``, so native importances, permutation importances and
SHAP values are all expressed in log-dollar space. Contributions are therefore
*additive in logs*, which means *multiplicative in dollars*: a SHAP value of
``+0.10`` is roughly a ``+10.5%`` effect on the rate, not ``+$0.10``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.logger import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 150


@dataclass
class ModelParts:
    """The decomposed pieces of the persisted Phase-5 pipeline.

    Attributes:
        pipeline: The full fitted pipeline (raw frame -> dollar prediction).
        preprocessor: The fitted Phase-3 preprocessing pipeline.
        regressor: The fitted CatBoost model operating in log space.
        feature_names: The 156 engineered feature names.
    """

    pipeline: Pipeline
    preprocessor: Pipeline
    regressor: Any
    feature_names: list[str]


def load_model_parts(model_path: str | Path) -> ModelParts:
    """Load the persisted best model and decompose it.

    Args:
        model_path: Path to ``models/best_model.joblib``.

    Returns:
        A populated :class:`ModelParts`.

    Raises:
        FileNotFoundError: If the model artifact is absent.
    """
    import joblib

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Best model not found at {path}. Run `python -m src.run_advanced_models_phase5` first."
        )

    pipeline = joblib.load(path)
    preprocessor = pipeline.named_steps["preprocess"]
    regressor = pipeline.named_steps["model"].regressor_
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    logger.info(
        "Loaded %s with %d features and %d trees",
        type(regressor).__name__,
        len(feature_names),
        getattr(regressor, "tree_count_", -1),
    )
    return ModelParts(
        pipeline=pipeline,
        preprocessor=preprocessor,
        regressor=regressor,
        feature_names=feature_names,
    )


def _save(figure: plt.Figure, path: Path) -> Path:
    """Write a figure to disk and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure: %s", path)
    return path


def _barh(
    labels: list[str],
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    color: str,
) -> plt.Figure:
    """Render a horizontal bar chart, largest value at the top."""
    height = max(4.0, 0.32 * len(labels))
    figure, axis = plt.subplots(figsize=(9.0, height))
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=color)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def native_feature_importance(parts: ModelParts) -> pd.DataFrame:
    """Extract CatBoost's built-in feature importance.

    CatBoost's default ``PredictionValuesChange`` measures how much each feature
    changes the prediction when its split values are altered, normalised to sum
    to 100.

    Args:
        parts: The decomposed model.

    Returns:
        A frame of ``feature`` and ``native_importance``, ranked descending.
    """
    values = np.asarray(parts.regressor.get_feature_importance(), dtype=float)
    frame = pd.DataFrame({"feature": parts.feature_names, "native_importance": values})
    return frame.sort_values("native_importance", ascending=False).reset_index(drop=True)


def permutation_feature_importance(
    parts: ModelParts,
    X_transformed: pd.DataFrame,
    y_log: np.ndarray,
    *,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on the engineered feature space.

    Permutation is run against the *inner* regressor on already-transformed
    features so the result is directly comparable with the native importance and
    SHAP values, which live in the same 156-feature space.

    Args:
        parts: The decomposed model.
        X_transformed: Preprocessed holdout features.
        y_log: Holdout target in log space.
        n_repeats: Shuffles per feature.
        seed: Random seed.

    Returns:
        A frame of ``feature``, ``permutation_importance`` and
        ``permutation_std``, ranked descending.
    """
    logger.info(
        "Computing permutation importance over %d features x %d repeats",
        X_transformed.shape[1],
        n_repeats,
    )
    result = permutation_importance(
        parts.regressor,
        X_transformed,
        y_log,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": parts.feature_names,
            "permutation_importance": result.importances_mean,
            "permutation_std": result.importances_std,
        }
    )
    return frame.sort_values("permutation_importance", ascending=False).reset_index(drop=True)


def build_importance_table(native: pd.DataFrame, permutation: pd.DataFrame) -> pd.DataFrame:
    """Merge the two importance views into one ranked table.

    Args:
        native: Output of :func:`native_feature_importance`.
        permutation: Output of :func:`permutation_feature_importance`.

    Returns:
        A merged frame ranked by native importance, with both ranks attached.
    """
    merged = native.merge(permutation, on="feature", how="outer")
    merged["native_rank"] = merged["native_importance"].rank(ascending=False).astype(int)
    merged["permutation_rank"] = merged["permutation_importance"].rank(ascending=False).astype(int)
    return merged.sort_values("native_importance", ascending=False).reset_index(drop=True)


def plot_importances(
    native: pd.DataFrame,
    permutation: pd.DataFrame,
    *,
    output_dir: Path,
    top_n: int = 25,
) -> list[Path]:
    """Save the native and permutation importance bar charts.

    Args:
        native: Native importance frame.
        permutation: Permutation importance frame.
        output_dir: Directory for ``figures/importance``.
        top_n: Number of features to display.

    Returns:
        Paths of the saved figures.
    """
    top_native = native.head(top_n)
    figure = _barh(
        top_native["feature"].tolist(),
        top_native["native_importance"].to_numpy(),
        title=f"CatBoost native feature importance (top {top_n})",
        xlabel="PredictionValuesChange (normalised, sums to 100)",
        color="#1f6f78",
    )
    native_path = _save(figure, output_dir / "catboost_native_importance.png")

    top_permutation = permutation.head(top_n)
    height = max(4.0, 0.32 * len(top_permutation))
    figure, axis = plt.subplots(figsize=(9.0, height))
    positions = np.arange(len(top_permutation))
    axis.barh(
        positions,
        top_permutation["permutation_importance"].to_numpy(),
        xerr=top_permutation["permutation_std"].to_numpy(),
        color="#b4552d",
        error_kw={"ecolor": "#5c5c5c", "elinewidth": 0.8},
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(top_permutation["feature"].tolist(), fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("Increase in log-space MAE when shuffled")
    axis.set_title(f"Permutation importance (top {top_n})", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    permutation_path = _save(figure, output_dir / "permutation_importance.png")

    return [native_path, permutation_path]


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


def compute_shap_values(
    parts: ModelParts, X_transformed: pd.DataFrame, *, sample_size: int = 2_000, seed: int = 42
) -> tuple[Any, pd.DataFrame]:
    """Compute exact tree SHAP values on a sample of the holdout.

    A sample is used because SHAP interaction cost grows with rows; 2,000 rows is
    ample for stable global rankings on a 400-tree model.

    Args:
        parts: The decomposed model.
        X_transformed: Preprocessed holdout features.
        sample_size: Number of rows to explain.
        seed: Sampling seed.

    Returns:
        ``(shap_explanation, sampled_features)``.

    Raises:
        ImportError: If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for Phase 6. Install it with `python -m pip install shap`."
        ) from exc

    sample = X_transformed
    if len(X_transformed) > sample_size:
        sample = X_transformed.sample(n=sample_size, random_state=seed).sort_index()

    logger.info("Computing SHAP values for %d rows", len(sample))
    explainer = shap.TreeExplainer(parts.regressor)
    explanation = explainer(sample)
    explanation.feature_names = parts.feature_names
    return explanation, sample


def plot_shap_figures(
    explanation: Any,
    sample: pd.DataFrame,
    *,
    output_dir: Path,
    top_features: list[str],
    waterfall_indices: dict[str, int],
) -> list[Path]:
    """Save the full set of SHAP figures.

    Args:
        explanation: SHAP explanation object.
        sample: The explained feature frame.
        output_dir: Directory for ``figures/shap``.
        top_features: Features to produce dependence plots for.
        waterfall_indices: Mapping of label -> positional index into ``sample``.

    Returns:
        Paths of every saved figure.
    """
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Beeswarm (the canonical "summary" view).
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP beeswarm - effect on log(rate)", loc="left", fontweight="bold")
    saved.append(_save(plt.gcf(), output_dir / "shap_beeswarm.png"))

    # Summary plot (dot form, kept separate as the task lists both).
    plt.figure()
    shap.summary_plot(
        explanation.values, sample, feature_names=list(sample.columns), max_display=20, show=False
    )
    plt.title("SHAP summary - effect on log(rate)", loc="left", fontweight="bold")
    saved.append(_save(plt.gcf(), output_dir / "shap_summary.png"))

    # Mean absolute SHAP bar chart.
    plt.figure()
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("Mean |SHAP| - average impact on log(rate)", loc="left", fontweight="bold")
    saved.append(_save(plt.gcf(), output_dir / "shap_bar.png"))

    # Dependence plots for the most important features.
    for feature in top_features:
        if feature not in sample.columns:
            logger.warning("Skipping dependence plot for absent feature %r", feature)
            continue
        plt.figure()
        shap.dependence_plot(
            feature,
            explanation.values,
            sample,
            feature_names=list(sample.columns),
            show=False,
        )
        plt.title(f"SHAP dependence - {feature}", loc="left", fontweight="bold")
        saved.append(_save(plt.gcf(), output_dir / f"shap_dependence_{feature}.png"))

    # Waterfall plots for representative individual predictions.
    for label, position in waterfall_indices.items():
        plt.figure()
        shap.plots.waterfall(explanation[position], max_display=15, show=False)
        plt.title(f"SHAP waterfall - {label}", loc="left", fontweight="bold")
        saved.append(_save(plt.gcf(), output_dir / f"shap_waterfall_{label}.png"))

    return saved
