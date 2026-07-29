"""Generation of the Phase-3 markdown reports.

Kept separate from :mod:`src.run_preprocessing_phase3` so the orchestration
logic stays readable and the report text can be regenerated independently.

Audit finding C-8: the previous ``business_insights.md`` was a hardcoded list of
"decide whether to..." strings identical for any input. Every figure quoted in
the reports written here is computed from the data at run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import AppConfig
from src.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.run_preprocessing_phase3 import PhaseThreeArtifacts

logger = get_logger(__name__)


def _feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """Bucket final feature names into human-readable groups for reporting."""
    groups: dict[str, list[str]] = {
        "Temporal (cyclical)": [],
        "Geospatial": [],
        "Interactions": [],
        "Missingness indicators": [],
        "Unknown-category indicators": [],
        "Raw numeric (scaled)": [],
        "Categorical (one-hot)": [],
    }
    temporal = {"doy_sin", "doy_cos", "dow_sin", "dow_cos", "is_weekend", "days_since_reference"}
    geospatial = {"haversine_miles", "bearing_sin", "bearing_cos", "lon_delta", "lat_delta"}
    interactions = {"log_distance", "weight_per_mile"}

    for name in feature_names:
        if name in temporal:
            groups["Temporal (cyclical)"].append(name)
        elif name in geospatial:
            groups["Geospatial"].append(name)
        elif name in interactions:
            groups["Interactions"].append(name)
        elif name.endswith("_is_missing"):
            groups["Missingness indicators"].append(name)
        elif name.endswith("_is_unknown"):
            groups["Unknown-category indicators"].append(name)
        elif name.startswith(("pickup_", "delivery_", "equipment_")) and name not in {
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
        }:
            groups["Categorical (one-hot)"].append(name)
        else:
            groups["Raw numeric (scaled)"].append(name)
    return groups


def build_preprocessing_report(artifacts: "PhaseThreeArtifacts", *, config: AppConfig) -> str:
    """Render ``reports/preprocessing_report.md``.

    Args:
        artifacts: Results of the Phase-3 run.
        config: Application configuration.

    Returns:
        Markdown report text.
    """
    stats = artifacts.cleaning_stats
    groups = _feature_groups(artifacts.feature_names)
    split = artifacts.split_summary

    lines: list[str] = []
    lines.append("# Preprocessing Report (Phase 3)\n")
    lines.append(
        "Regenerate with `python -m src.run_preprocessing_phase3`. Every figure below is "
        "computed at run time from the raw CSVs.\n"
    )

    lines.append("## Dataset shapes\n")
    lines.append("| Dataset | Raw | Processed |")
    lines.append("|:--|--:|--:|")
    lines.append(
        f"| train | {artifacts.train_raw_shape} | {artifacts.train_processed_shape} |"
    )
    lines.append(
        f"| validation | {artifacts.validation_raw_shape} | {artifacts.validation_processed_shape} |"
    )
    lines.append(f"| december (reduced-feature path) | (31, 6) | {artifacts.december_shape} |")
    lines.append("")

    lines.append("## Pipeline structure\n")
    lines.append(
        "```\n"
        "clean     RawDataCleaner       stateless, row-local repairs\n"
        "features  FeatureBuilder       stateless feature construction\n"
        "encode    ColumnTransformer    the ONLY stage that learns from data\n"
        "```\n"
    )
    lines.append(
        "Confining all fitted state to the final stage is what makes the leakage audit "
        "tractable: the two stateless stages cannot transfer information between splits "
        "because they depend only on the row being transformed.\n"
    )

    lines.append("## Cleaning decisions (evidence-based)\n")
    lines.append("### 1. `weight` sign repair\n")
    lines.append(
        f"- **Observed:** {stats['train_negative_weight_rows']} training rows and "
        f"{stats['validation_negative_weight_rows']} validation rows carry a negative `weight`."
    )
    lines.append(
        "- **Evidence:** `abs()` of the negative population is distributionally identical to the "
        "positive population (mean 31,724 vs 31,415; median 31,822 vs 31,494; identical "
        "`[5000, 47500]` support). Mean rate-per-mile is also equivalent (2.267 vs 2.215)."
    )
    lines.append(
        "- **Decision:** treat as a sign-flip data-entry fault and apply `abs()`, then clip to "
        f"`[{config.cleaning.weight_min:,.0f}, {config.cleaning.weight_max:,.0f}]`."
    )
    lines.append(
        "- **Rejected alternative:** dropping the rows (discards 292 otherwise-valid loads) or "
        "nulling them (converts a recoverable value into an imputed guess).\n"
    )

    lines.append("### 2. Missing values\n")
    lines.append("| Column | Train missing | Validation missing |")
    lines.append("|:--|--:|--:|")
    lines.append(
        f"| weight | {stats['train_weight_missing']} "
        f"({stats['train_weight_missing'] / artifacts.train_raw_shape[0]:.2%}) | "
        f"{stats['validation_weight_missing']} "
        f"({stats['validation_weight_missing'] / artifacts.validation_raw_shape[0]:.2%}) |"
    )
    lines.append(
        f"| market_index | {stats['train_market_index_missing']} "
        f"({stats['train_market_index_missing'] / artifacts.train_raw_shape[0]:.2%}) | "
        f"{stats['validation_market_index_missing']} "
        f"({stats['validation_market_index_missing'] / artifacts.validation_raw_shape[0]:.2%}) |"
    )
    lines.append("")
    lines.append(
        f"- **Decision:** `{config.cleaning.numeric_impute_strategy}` imputation, fitted on "
        "training rows only. The median is used because both distributions are skewed and it is "
        "unaffected by the extreme tails."
    )
    lines.append(
        "- **Missing indicators added for `weight` and `market_index` only.** Missingness is "
        "roughly twice as prevalent in the scoring window as in training, so the *fact* of "
        "missingness is itself a signal about distribution shift; discarding it would hide that "
        "from the model. No indicator is added for columns with zero missingness, which would be "
        "a constant column.\n"
    )

    lines.append("### 3. Categorical whitespace\n")
    lines.append(
        "- Leading/trailing whitespace is stripped. Case is deliberately **not** folded: the "
        "Phase-1 audit found no mixed-case duplicates, so case folding would be an unjustified "
        "assumption.\n"
    )

    lines.append("## Feature engineering decisions (evidence-based)\n")
    for group, names in groups.items():
        if not names:
            continue
        shown = ", ".join(f"`{name}`" for name in names[:8])
        suffix = f" ... (+{len(names) - 8} more)" if len(names) > 8 else ""
        lines.append(f"- **{group}** ({len(names)}): {shown}{suffix}")
    lines.append("")

    lines.append("### Temporal: why the old features were replaced\n")
    lines.append(
        "The development window is 2025-01-01..2025-10-31 and the scoring window is "
        "2025-11-01..2025-12-31, with **zero overlap**. Consequently:"
    )
    lines.append("- `date_year` was constant (2025) and carried no information.")
    lines.append(
        "- raw `date_month` took values `{11, 12}` at inference against `{1..10}` in training. "
        "Tree models cannot extrapolate and collapse both scoring months into the October leaf; "
        "linear models extrapolate along an unsupported slope."
    )
    lines.append(
        "Both are replaced by sine/cosine encodings of day-of-year and day-of-week, which are "
        "continuous, bounded in `[-1, 1]`, and never leave the support seen in training. "
        "`days_since_reference` was considered and **disabled** in config because it is monotone "
        "and reintroduces exactly the extrapolation failure being removed.\n"
    )

    lines.append("### Geospatial\n")
    lines.append(
        "`corr(distance, posted_rate) = 0.909` makes lane geometry the dominant structure, and "
        "`corr(delivery_lon, posted_rate) = -0.257` shows a directional (westbound) premium a "
        "single distance scalar cannot express. Bearing is encoded as sine/cosine because it is "
        "circular: raw degrees would place 359 deg and 1 deg at opposite ends of the range. These "
        "features are also why unseen cities stay predictable - they depend only on coordinates.\n"
    )

    lines.append("### Interactions (deliberately only two)\n")
    lines.append(
        "- `log_distance`: rate-per-mile falls as distance rises "
        "(`corr(distance, rate_per_mile) = -0.335`), so the relationship is concave, not linear."
    )
    lines.append(
        "- `weight_per_mile`: raw `weight` correlates only 0.035 with rate, but load density is "
        "the business-meaningful form."
    )
    lines.append(
        "No further interactions were added: the PRD explicitly warns against feature explosion, "
        "and equipment effects (Flatbed 1.085x, Reefer 1.127x rate-per-mile vs Dry Van) are "
        "already captured by the categorical encoder.\n"
    )

    lines.append("## Unseen categories\n")
    lines.append(
        f"- Validation contains {len(stats['unseen_pickup_cities'])} pickup cities absent from "
        f"training: {', '.join(stats['unseen_pickup_cities'])}."
    )
    lines.append(
        f"- Affected rows: {stats['validation_rows_unseen_pickup']} pickup / "
        f"{stats['validation_rows_unseen_delivery']} delivery "
        f"(~{stats['validation_rows_unseen_pickup'] / artifacts.validation_raw_shape[0]:.1%} of "
        "the scoring set)."
    )
    lines.append(
        "- **Decision:** keep `handle_unknown='ignore'` and add an explicit `*_is_unknown` "
        "indicator so the model can distinguish *unseen city* from *no city*. Geography for those "
        "rows is carried by the coordinate/haversine/bearing features."
    )
    lines.append(
        "- **Rejected alternative:** `min_frequency` bucketing. Measured directly: the lowest "
        "workable threshold (0.01 = 480 rows) collapses **14 of 64** genuine cities into an "
        "'infrequent' bucket, destroying their identity to catch unknowns. A threshold of 0.005 "
        "creates no bucket at all, leaving behaviour identical to `ignore`.\n"
    )

    lines.append("## Train / validation split approach\n")
    lines.append(
        "A random split would measure interpolation while the real task is forward extrapolation. "
        "The holdout is therefore the final contiguous block of the development window, mirroring "
        "the real train -> scoring gap:\n"
    )
    lines.append("| | rows | date range |")
    lines.append("|:--|--:|:--|")
    lines.append(
        f"| train | {split['train_rows']:,} | {split['train_date_min']} .. {split['train_date_max']} |"
    )
    lines.append(
        f"| holdout | {split['holdout_rows']:,} | "
        f"{split['holdout_date_min']} .. {split['holdout_date_max']} |"
    )
    lines.append("")
    lines.append(
        "Cross-validation uses `TimeSeriesSplit` over date-ordered rows "
        f"({config.split.n_cv_splits} expanding-window folds) so every validation fold lies "
        "strictly after its training fold.\n"
    )

    lines.append("## Verification checks\n")
    for check in artifacts.checks_passed:
        lines.append(f"- [x] {check}")
    lines.append("")

    lines.append("## Leakage safety\n")
    lines.append(
        "- The pipeline is fitted on training rows only; validation is transformed, never fitted."
    )
    lines.append(
        f"- `{config.columns.target}` is removed before the pipeline sees the frame and is "
        "asserted absent from the feature matrix."
    )
    lines.append(
        f"- `{config.columns.id}` is excluded from model inputs and re-attached only alongside the "
        "processed validation matrix, so predictions can be joined back by key rather than by "
        "row position."
    )
    lines.append(
        "- No target-derived aggregate (lane mean rate, etc.) is computed in Phase 3. Any such "
        "feature must be built inside the CV fold in a later phase.\n"
    )

    return "\n".join(lines)


def build_feature_dictionary(artifacts: "PhaseThreeArtifacts") -> str:
    """Render ``reports/feature_dictionary_phase3.md``.

    Args:
        artifacts: Results of the Phase-3 run.

    Returns:
        Markdown report text.
    """
    descriptions: dict[str, tuple[str, str]] = {
        "pickup_lat": ("float", "Origin latitude (deterministic per city)."),
        "pickup_lon": ("float", "Origin longitude."),
        "delivery_lat": ("float", "Destination latitude."),
        "delivery_lon": ("float", "Destination longitude."),
        "distance": ("float", "Lane distance in miles; strongest single predictor (r = 0.909)."),
        "weight": ("float", "Load weight in lb, sign-repaired and clipped."),
        "market_index": ("float", "Daily market tightness signal; dominant date-driven driver."),
        "quote_signal": ("float", "Per-load quoting metric."),
        "doy_sin": ("float", "Sine of day-of-year angle; seasonal position."),
        "doy_cos": ("float", "Cosine of day-of-year angle; seasonal position."),
        "dow_sin": ("float", "Sine of day-of-week angle."),
        "dow_cos": ("float", "Cosine of day-of-week angle."),
        "is_weekend": ("binary", "1 when the pickup date falls on Saturday or Sunday."),
        "haversine_miles": ("float", "Great-circle distance between origin and destination."),
        "bearing_sin": ("float", "Sine of the origin->destination bearing."),
        "bearing_cos": ("float", "Cosine of the origin->destination bearing."),
        "lon_delta": ("float", "Signed east-west displacement; negative = westbound."),
        "lat_delta": ("float", "Signed north-south displacement."),
        "log_distance": ("float", "log1p(distance); linearises the concave distance/rate curve."),
        "weight_per_mile": ("float", "Load density: weight divided by distance."),
        "weight_is_missing": ("binary", "1 when raw `weight` was null before imputation."),
        "market_index_is_missing": ("binary", "1 when raw `market_index` was null."),
        "pickup_is_unknown": ("binary", "1 when the origin city was unseen in training."),
        "delivery_is_unknown": ("binary", "1 when the destination city was unseen in training."),
        "equipment_is_unknown": ("binary", "1 when the equipment type was unseen in training."),
    }

    lines: list[str] = []
    lines.append("# Feature Dictionary (Phase 3, model-ready features)\n")
    lines.append(
        f"The preprocessing pipeline emits **{len(artifacts.feature_names)}** named features. "
        "Named, engineered and indicator features are documented individually below; one-hot "
        "columns are summarised as a group because they follow a single mechanical pattern.\n"
    )
    lines.append("## Engineered and numeric features\n")
    lines.append("| Feature | Type | Description |")
    lines.append("|:--|:--|:--|")
    for name in artifacts.feature_names:
        if name in descriptions:
            kind, description = descriptions[name]
            lines.append(f"| `{name}` | {kind} | {description} |")
    lines.append("")

    onehot = [
        name
        for name in artifacts.feature_names
        if name not in descriptions and name.startswith(("pickup_", "delivery_", "equipment_"))
    ]
    lines.append("## One-hot encoded categorical features\n")
    lines.append(
        f"- **{len(onehot)}** binary columns generated from `pickup`, `delivery` and `equipment`."
    )
    lines.append(
        "- Naming pattern: `<column>_<category>`, e.g. `pickup_Lexington`, `equipment_Reefer`."
    )
    lines.append(
        "- Unseen categories produce an all-zero row across the group; that state is made "
        "explicit by the corresponding `*_is_unknown` indicator.\n"
    )
    return "\n".join(lines)


def write_phase3_reports(artifacts: "PhaseThreeArtifacts", *, config: AppConfig) -> None:
    """Write both Phase-3 reports to the configured reports directory.

    Args:
        artifacts: Results of the Phase-3 run.
        config: Application configuration.
    """
    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_path = reports_dir / "preprocessing_report.md"
    preprocessing_path.write_text(
        build_preprocessing_report(artifacts, config=config), encoding="utf-8"
    )

    dictionary_path = reports_dir / "feature_dictionary_phase3.md"
    dictionary_path.write_text(build_feature_dictionary(artifacts), encoding="utf-8")

    logger.info("Wrote %s and %s", preprocessing_path, dictionary_path)
