from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EDAConfig:
    """
    Configuration for Phase 2 exploratory data analysis.

    Attributes:
        train_path: Path to data/train_test.csv.
        figures_dir: Directory where EDA figures will be saved.
        top_k_categories: For categorical top-k frequency displays.
        outlier_iqr_multiplier: Multiplier for IQR outlier bounds.
        random_seed: Seed for any stochastic operations (kept for reproducibility).
    """

    train_path: str | Path = "data/train_test.csv"
    figures_dir: str | Path = "figures/eda"
    top_k_categories: int = 10
    outlier_iqr_multiplier: float = 1.5
    random_seed: int = 42


@dataclass(frozen=True)
class EDAResults:
    """
    Outputs created by the EDA run.
    """

    exploratory_report_markdown: str
    business_insights_markdown: str


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _hist_series(
    df: pd.DataFrame, col: str, out_path: Path, *, bins: int = 50, color: str = "#064A56"
) -> None:
    x = df[col].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(x, bins=bins, color=color, alpha=0.85)
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    _save_fig(fig, out_path)


def _box_series(df: pd.DataFrame, col: str, out_path: Path) -> None:
    x = df[col].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(x, vert=True, showfliers=True)
    ax.set_title(f"Boxplot: {col}")
    ax.set_ylabel(col)
    _save_fig(fig, out_path)


def _scatter_geo(df: pd.DataFrame, lon_col: str, lat_col: str, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[lon_col].to_numpy(), df[lat_col].to_numpy(), s=1, alpha=0.25)
    ax.set_title(f"Geo scatter: {label}")
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    _save_fig(fig, out_path)


def _missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_counts / len(df) * 100.0).round(4)
    missing_df = pd.DataFrame({"missing_count": missing_counts, "missing_pct": missing_pct})
    return missing_df[missing_df.missing_count > 0]


def _outliers_iqr(df: pd.DataFrame, numeric_cols: list[str], *, multiplier: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) < 10:
            continue
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo = q1 - multiplier * iqr
        hi = q3 + multiplier * iqr
        outlier_mask = (x < lo) | (x > hi)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = 100.0 * outlier_count / len(x) if len(x) else 0.0
        rows.append(
            {"feature": c, "outlier_count": outlier_count, "outlier_pct": outlier_pct, "lo": float(lo), "hi": float(hi)}
        )
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)


def _correlation_matrix(df: pd.DataFrame, numeric_cols: list[str], out_path: Path) -> pd.DataFrame:
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix (numerical features)")
    _save_fig(fig, out_path)
    return corr


def _categorical_topk(df: pd.DataFrame, cat_col: str, top_k: int) -> pd.Series:
    s = df[cat_col].astype("string")
    return s.value_counts(dropna=True).head(top_k)


def run_eda_phase2(config: EDAConfig) -> EDAResults:
    """
    Run Phase 2 EDA from raw training data only.

    Generates:
      - reports/exploratory_data_analysis.md
      - reports/business_insights.md
      - figures/eda/*
    """
    train_path = Path(config.train_path)
    figures_dir = Path(config.figures_dir)
    reports_dir = Path("reports")

    _ensure_dir(figures_dir)
    _ensure_dir(reports_dir)

    if not train_path.is_file():
        raise FileNotFoundError(f"train CSV not found: {train_path.resolve()}")

    df = pd.read_csv(train_path)

    # Identify groups
    if "posted_rate" not in df.columns:
        raise ValueError("Expected target column 'posted_rate' in training data.")
    target_col = "posted_rate"
    load_id_col = "load_id" if "load_id" in df.columns else None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # "object" includes python strings; sometimes pandas may infer other types, keep it explicit
    object_cols = [c for c in df.columns if c not in numeric_cols]

    # Date handling (temporal analysis)
    date_col = "date" if "date" in df.columns else None
    df_date_parsed = None
    date_summary: dict[str, Any] = {}
    if date_col is not None:
        df_date_parsed = _safe_to_datetime(df[date_col])
        date_summary = {
            "date_parse_missing": int(df_date_parsed.isna().sum()),
            "date_min": str(df_date_parsed.min()) if df_date_parsed.notna().any() else "n/a",
            "date_max": str(df_date_parsed.max()) if df_date_parsed.notna().any() else "n/a",
        }

    # 1) Target analysis
    target_missing = int(df[target_col].isna().sum())
    target_desc = df[target_col].describe()

    # Figures: target hist/box
    if df[target_col].notna().any():
        _hist_series(df, target_col, figures_dir / "posted_rate_hist.png")
        _box_series(df, target_col, figures_dir / "posted_rate_box.png")

    # 2) Numeric feature analysis
    numeric_summary = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

    # key numeric columns to visualize (only if present)
    key_numeric_candidates = [
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]
    for c in key_numeric_candidates:
        if c in df.columns and c in numeric_cols:
            _hist_series(df, c, figures_dir / f"{c}_hist.png")

    # 3) Categorical analysis
    # Exclude load_id from categorical top-k to keep focus
    categorical_cols = [c for c in object_cols if c != load_id_col and c != date_col]
    cat_topk_payload: list[dict[str, Any]] = []
    for c in categorical_cols:
        vc = _categorical_topk(df, c, config.top_k_categories)
        cat_topk_payload.append(
            {"feature": c, "n_unique": int(df[c].astype("string").nunique(dropna=True)), "topk": vc.to_dict()}
        )
        # plot
        fig, ax = plt.subplots(figsize=(10, 4))
        vc.plot(kind="bar", color="#064A56", alpha=0.85, ax=ax)
        ax.set_title(f"Top categories: {c}")
        ax.set_xlabel(c)
        ax.set_ylabel("count")
        _save_fig(fig, figures_dir / f"{c}_top_categories.png")

    # 4) Missing value analysis
    missing_df = _missingness_table(df)
    if not missing_df.empty:
        top_missing = missing_df.head(30)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(top_missing.index.astype(str), top_missing["missing_pct"].to_numpy(), color="#064A56", alpha=0.85)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Missingness % by feature (top 30)")
        ax.set_ylabel("missing %")
        _save_fig(fig, figures_dir / "missingness_pct_top30.png")

    # 5) Outlier analysis (IQR)
    outliers_df = _outliers_iqr(df, numeric_cols, multiplier=config.outlier_iqr_multiplier)
    if not outliers_df.empty:
        # bar plot of outlier percentages
        top_out = outliers_df.head(25).copy()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(top_out["feature"].astype(str), top_out["outlier_pct"].to_numpy(), color="#064A56", alpha=0.85)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Outlier prevalence by feature (IQR rule, top 25)")
        ax.set_ylabel("outlier %")
        _save_fig(fig, figures_dir / "outliers_top25.png")

    # also boxplots for target + key numeric (if present)
    for c in ([target_col] + [x for x in key_numeric_candidates if x in df.columns])[:10]:
        if c in df.columns and c in numeric_cols:
            _box_series(df, c, figures_dir / f"{c}_box.png")

    # 6) Correlation analysis
    corr_path = figures_dir / "correlation_matrix.png"
    corr = pd.DataFrame()
    if numeric_cols:
        corr = _correlation_matrix(df, numeric_cols, corr_path)

    # 7) Geographic analysis
    # Use scatter on lat/lon pairs if present
    if {"pickup_lat", "pickup_lon"}.issubset(df.columns):
        _scatter_geo(df, "pickup_lon", "pickup_lat", "pickup", figures_dir / "geo_scatter_pickup.png")
    if {"delivery_lat", "delivery_lon"}.issubset(df.columns):
        _scatter_geo(df, "delivery_lon", "delivery_lat", "delivery", figures_dir / "geo_scatter_delivery.png")

    # 8) Temporal analysis
    if date_col is not None and df_date_parsed is not None and df_date_parsed.notna().any():
        daily_mean = df.loc[df_date_parsed.notna()].groupby(df_date_parsed)[target_col].mean().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_mean.index, daily_mean.values, color="#064A56")
        ax.set_title("Mean posted_rate by date")
        ax.set_xlabel("date")
        ax.set_ylabel("mean posted_rate")
        ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _save_fig(fig, figures_dir / "mean_posted_rate_by_date.png")

    # 9) Business relationship analysis
    business_payload: dict[str, Any] = {"group_stats": {}}
    for c in ["pickup", "delivery", "equipment"]:
        if c in df.columns:
            grp = df.groupby(c)[target_col].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
            business_payload["group_stats"][c] = grp.head(10)

    # 10) EDA writing
    # Correlation with target (top absolute correlations)
    corr_with_target_md = "- n/a"
    if (not corr.empty) and target_col in corr.columns:
        corr_with_target = corr[target_col].dropna().sort_values(ascending=False)
        top = corr_with_target.head(20)
        corr_with_target_md = top.to_frame(name="corr").reset_index().rename(columns={"index": "feature"}).to_markdown(index=False)

    missing_md = (
        missing_df.head(40).to_markdown()
        if not missing_df.empty
        else "- None detected"
    )
    outliers_md = (
        outliers_df.head(25).to_markdown(index=False)
        if not outliers_df.empty
        else "- None detected"
    )
    numeric_summary_md = numeric_summary.head(25).to_markdown()

    # Categorical top-k in markdown
    cat_md_parts: list[str] = []
    if categorical_cols:
        for payload in cat_topk_payload:
            feat = payload["feature"]
            cat_md_parts.append(f"### {feat}\n")
            cat_md_parts.append(pd.Series(payload["topk"]).to_frame("count").to_markdown())
            cat_md_parts.append("\n")
    cat_md = "\n".join(cat_md_parts) if cat_md_parts else "- None"

    # Business payload md
    business_md_parts: list[str] = []
    for c, grp in business_payload["group_stats"].items():
        business_md_parts.append(f"### Relationship: {c} -> posted_rate (top 10 by mean)\n")
        business_md_parts.append(grp.to_markdown())
        business_md_parts.append("\n")
    business_md = "\n".join(business_md_parts) if business_md_parts else "- None"

    date_md = (
        f"- date_parse_missing: **{date_summary.get('date_parse_missing')}**\n"
        f"- date_min: **{date_summary.get('date_min')}**\n"
        f"- date_max: **{date_summary.get('date_max')}**\n"
        if date_col is not None
        else "- date column not present"
    )

    exploratory_report_lines: list[str] = []
    exploratory_report_lines.append("# Exploratory Data Analysis (Phase 2) - data/train_test.csv\n")
    exploratory_report_lines.append(f"- Shape: **{df.shape[0]:,} rows x {df.shape[1]:,} columns**\n")
    exploratory_report_lines.append(f"- Target column: **{target_col}**\n")
    exploratory_report_lines.append(f"- Target missing values: **{target_missing}**\n")

    exploratory_report_lines.append("## Target variable analysis\n")
    exploratory_report_lines.append(target_desc.to_frame(name="value").to_markdown())
    exploratory_report_lines.append("\n")

    exploratory_report_lines.append("## Numerical feature analysis\n")
    exploratory_report_lines.append(numeric_summary_md)
    exploratory_report_lines.append("\n")

    exploratory_report_lines.append("## Categorical feature analysis\n")
    exploratory_report_lines.append(cat_md)
    exploratory_report_lines.append("\n")

    exploratory_report_lines.append("## Missing value analysis\n")
    exploratory_report_lines.append(missing_md)
    exploratory_report_lines.append("\n")

    exploratory_report_lines.append("## Outlier analysis (IQR rule)\n")
    exploratory_report_lines.append(outliers_md)
    exploratory_report_lines.append("\n")

    exploratory_report_lines.append("## Correlation analysis\n")
    exploratory_report_lines.append("Top correlations with target (by value):\n")
    exploratory_report_lines.append(corr_with_target_md)
    exploratory_report_lines.append("\n")
    exploratory_report_lines.append("- Full correlation heatmap: `figures/eda/correlation_matrix.png`\n")

    exploratory_report_lines.append("## Geographic analysis\n")
    exploratory_report_lines.append("- Pickup geo scatter: `figures/eda/geo_scatter_pickup.png` (if available)\n")
    exploratory_report_lines.append("- Delivery geo scatter: `figures/eda/geo_scatter_delivery.png` (if available)\n")

    exploratory_report_lines.append("## Temporal analysis\n")
    exploratory_report_lines.append(date_md)
    exploratory_report_lines.append("- Mean by date plot: `figures/eda/mean_posted_rate_by_date.png` (if parseable)\n")

    exploratory_report_lines.append("## Business relationship analysis\n")
    exploratory_report_lines.append(business_md)
    exploratory_report_lines.append("\n")

    exploratory_report_lines.append("## Modeling implications (descriptive only)\n")
    exploratory_report_lines.append("- Use missing-value imputation for features with missingness.\n")
    exploratory_report_lines.append("- Use robust scaling for features with outliers (if supported by EDA).\n")
    exploratory_report_lines.append("- Consider time-derived features if temporal plots show structure.\n")
    exploratory_report_lines.append("- Consider interaction features if correlations/relationships indicate non-linear effects.\n")

    # Feature engineering opportunities: evidence-based and conservative
    opps: list[str] = []
    if date_col is not None:
        opps.append("Date feature extraction: year/month/day/weekday/weekend (validated after observing temporal patterns).")
    if {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}.issubset(df.columns):
        opps.append("Geographic distance feature(s): e.g., haversine distance from pickup to delivery (validated using geo scatter and distance relationships).")
    if {"weight", "distance"}.issubset(df.columns):
        opps.append("Weight-per-distance interaction (business rationale: shipping intensity relative to distance).")
    if {"market_index", "quote_signal"}.issubset(df.columns):
        opps.append("Market/quote interactions (validated using correlation analysis with posted_rate).")
    if "equipment" in df.columns:
        opps.append("Equipment category impacts on posted_rate (validated via group-by relationship).")

    exploratory_report_lines.append("## Feature engineering opportunities (evidence to be validated in Phase 3 implementation)\n")
    for o in opps:
        exploratory_report_lines.append(f"- {o}\n")

    exploratory_report = "".join(exploratory_report_lines)

    # Business insights report (focused)
    business_lines: list[str] = []
    business_lines.append("# Business Insights (Phase 2) - from EDA\n")
    business_lines.append(f"- Target: **{target_col}**\n\n")

    business_lines.append("## Interpretable patterns to validate in Phase 3\n")
    business_lines.append("- Missingness locations: decide whether to impute with median (numeric) and how to treat categorical missing/unknown.\n")
    business_lines.append("- Outlier prevalence: decide robust scaling vs StandardScaler and whether to clip (only if extreme outliers distort).\n")
    business_lines.append("- Temporal structure: decide which date parts to extract if the mean-by-date plot shows seasonality/trends.\n")
    business_lines.append("- Geographic structure: decide whether distance-from-coordinates features add value beyond provided `distance`.\n")
    business_lines.append("- Category effects: decide encoding strategy for pickup/delivery/equipment based on frequency and group-by differences.\n")
    business_lines.append("\n## Feature candidates and business justification\n")
    for o in opps:
        business_lines.append(f"- {o}\n")

    business_insights = "".join(business_lines)

    return EDAResults(
        exploratory_report_markdown=exploratory_report,
        business_insights_markdown=business_insights,
    )
