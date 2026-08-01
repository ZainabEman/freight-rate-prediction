"""Content and layout for the dashboard's panels.

Holds the narrative constants (decision log, timeline, repository map) and the
renderers that turn :class:`~src.dashboard_data.DashboardData` into HTML.
:mod:`src.build_dashboard` assembles the surrounding document.
"""

from __future__ import annotations

import json

from src.dashboard_data import DashboardData

# (id, sidebar label, group). Groups mirror the real delivery phases, so the
# numbering encodes sequence rather than decorating the nav.
SECTIONS: list[tuple[str, str, str]] = [
    ("exec", "Executive Summary", "Overview"),
    ("overview", "Project Overview", "Overview"),
    ("dataset", "Dataset", "Overview"),
    ("audit", "Data Audit", "Analysis"),
    ("eda", "Exploratory Analysis", "Analysis"),
    ("features", "Feature Engineering", "Build"),
    ("preprocessing", "Preprocessing Pipeline", "Build"),
    ("baselines", "Baseline Models", "Modelling"),
    ("advanced", "Advanced Models", "Modelling"),
    ("tuning", "Hyperparameter Tuning", "Modelling"),
    ("comparison", "Model Comparison", "Modelling"),
    ("explain", "Explainability", "Evaluation"),
    ("errors", "Error Analysis", "Evaluation"),
    ("insights", "Business Insights", "Evaluation"),
    ("predictions", "Final Predictions", "Delivery"),
    ("december", "December Analysis", "Delivery"),
    ("decisions", "Decision Log", "Reference"),
    ("architecture", "Architecture", "Reference"),
    ("repo", "Repository Explorer", "Reference"),
    ("timeline", "Project Timeline", "Reference"),
    ("modelcard", "Model Card", "Reference"),
    ("deployment", "Deployment Readiness", "Reference"),
    ("conclusion", "Conclusion", "Reference"),
]

DECISIONS: list[dict] = [
    {
        "title": "Time-based split instead of a random split",
        "tag": ("accent", "Foundational"),
        "decision": "Hold out the final contiguous block of the development window "
                    "(2025-09-01 to 2025-10-31) rather than sampling rows at random.",
        "why": "Training data ends 2025-10-31 and scoring begins 2025-11-01 with zero overlap. The "
               "task is forward extrapolation, so a random split would measure interpolation skill "
               "the model will never be asked to use.",
        "alternatives": "Random k-fold; grouped split by lane; a single random holdout.",
        "evidence": "The two windows are strictly disjoint. Every fold was verified to train only "
                    "on rows earlier than the rows it scores.",
        "impact": "All reported metrics are honest forward-in-time estimates. This was the single "
                  "most consequential decision in the project.",
    },
    {
        "title": "Leakage prevented structurally, not by convention",
        "tag": ("accent", "Foundational"),
        "decision": "Make preprocessing a step <i>inside</i> the searched pipeline so scikit-learn "
                    "clones and refits it independently within every CV fold.",
        "why": "Fitting the preprocessor once on all training data leaks future feature "
               "distributions into earlier folds. Relying on discipline to avoid this fails "
               "silently and is invisible in the metrics.",
        "alternatives": "Fit once and reuse (faster, but leaks); manual per-fold refitting (correct "
                        "but easy to get wrong).",
        "evidence": "<code>verify_no_leakage()</code> asserts fold ordering and disjointness; "
                    "<code>assert_no_leakage()</code> confirms neither the target nor "
                    "<code>load_id</code> reaches the feature matrix.",
        "impact": "No imputation median, scaler statistic or one-hot vocabulary is ever learned "
                  "from data later than the rows it scores. Costs runtime, buys correctness.",
    },
    {
        "title": "Cyclical date encoding, and no monotone time feature",
        "tag": ("accent", "Foundational"),
        "decision": "Encode date as <code>doy_sin/cos</code> and <code>dow_sin/cos</code>. Drop "
                    "<code>date_year</code> and raw <code>date_month</code>; explicitly disable "
                    "<code>days_since_reference</code>.",
        "why": "<code>date_year</code> is constant (2025) and carries zero information. Raw "
               "<code>date_month</code> takes values {11, 12} at scoring against {1..10} in "
               "training — tree models cannot extrapolate and collapse both scoring months into "
               "the October leaf. A monotone day counter has the same defect.",
        "alternatives": "Raw month/day integers (the inherited implementation); one-hot months "
                        "(same failure); a linear time trend.",
        "evidence": "Training months {1..10} vs scoring {11, 12}, verified from both CSVs. Cyclical "
                    "encodings are bounded in [-1, 1] and never leave the training support.",
        "impact": "Removes the extrapolation failure mode entirely, and produces the weekly "
                  "periodicity visible in the December chart.",
    },
    {
        "title": "Negative weights repaired by sign flip, not discarded",
        "tag": ("neutral", "Data quality"),
        "decision": "Apply <code>abs()</code> to 292 negative <code>weight</code> values, then clip "
                    "to the observed envelope [5,000, 47,500].",
        "why": "The absolute values of the negative population are distributionally identical to "
               "the positive population, identifying a sign-flip data-entry fault rather than a "
               "distinct physical population.",
        "alternatives": "Drop the rows (loses usable data); treat as missing and impute (discards a "
                        "recoverable true magnitude); leave as-is (corrupts the scaler).",
        "evidence": "Mean of |negative| = 31,724 vs positive mean 31,415; both span an identical "
                    "[5,000, 47,500] range.",
        "impact": "Recovers 292 usable rows with their true magnitudes intact.",
    },
    {
        "title": "Missing-value indicators alongside imputation",
        "tag": ("neutral", "Data quality"),
        "decision": "Emit <code>weight_is_missing</code> and <code>market_index_is_missing</code> "
                    "before median imputation.",
        "why": "Missingness rates roughly double between training and scoring. Imputing without "
               "flagging would silently hide a real distribution shift from the model.",
        "alternatives": "Impute silently; drop rows; model-based imputation.",
        "evidence": "<code>weight</code> missing 0.63% train vs 1.38% validation; "
                    "<code>market_index</code> 0.78% vs 2.08%.",
        "impact": "Two extra features; the model can condition on the fact of missingness.",
    },
    {
        "title": "Explicit unknown-category flag over frequency bucketing",
        "tag": ("neutral", "Data quality"),
        "decision": "Keep <code>handle_unknown='ignore'</code> and add explicit "
                    "<code>*_is_unknown</code> flags rather than bucketing rare categories.",
        "why": "With plain one-hot, an unseen city becomes an all-zero vector indistinguishable "
               "from 'no city at all' — the encoder fails silently by design.",
        "alternatives": "<code>min_frequency</code> bucketing was measured and rejected: the lowest "
                        "workable threshold collapses 14 of 64 genuine cities to catch unknowns.",
        "evidence": "8 cities appear only at scoring time, affecting ~6% of scoring rows. "
                    "Coordinate features remain fully informative for those rows.",
        "impact": "Preserves all 64 city identities while making the unknown state learnable.",
    },
    {
        "title": "Log target chosen on measured evidence",
        "tag": ("neutral", "Modelling"),
        "decision": "Train every advanced model on <code>log(posted_rate)</code> via "
                    "<code>TransformedTargetRegressor</code>.",
        "why": "The target is right-skewed (median $2,031, mean $2,374, max $25,533). Log training "
               "also guarantees strictly positive predictions after inversion, which the scorer "
               "requires.",
        "alternatives": "Identity target; Box-Cox; square-root.",
        "evidence": "In Phase 4 the log-target Ridge beat identity Ridge on MAE ($145.24 vs "
                    "$149.36) and MAPE (6.27% vs 8.65%).",
        "impact": "Better MAE and structural positivity, at the cost of the back-transformation "
                  "bias addressed below.",
    },
    {
        "title": "Duan smearing correction, tested before adoption",
        "tag": ("good", "Highest impact"),
        "decision": "Rescale predictions by the mean of exponentiated training residuals "
                    "(factor 1.0128).",
        "why": "A log-target model fitted to the conditional mean of log(rate) returns the "
               "conditional <i>median</i> after <code>exp()</code>, which sits below the mean for a "
               "right-skewed target. Phase 6 measured the resulting bias at +$101.81 (t = 15.69).",
        "alternatives": "No correction (accept the bias); a parametric log-normal correction, which "
                        "assumes normal residuals — the QQ plot rejects that assumption.",
        "evidence": "Evaluated on the untouched holdout <i>before</i> adoption, because smearing "
                    "targets the conditional mean while MAE is minimised by the median — the two "
                    "genuinely conflict. Here they did not: MAE $132.15 → $114.99, RMSE $641.48 → "
                    "$636.25, bias $101.81 → $73.37.",
        "impact": "A 13.0% MAE improvement for one multiplication. The factor is strictly positive, "
                  "so positivity cannot be violated.",
    },
    {
        "title": "CatBoost selected — with an explicit caveat",
        "tag": ("warn", "Within noise"),
        "decision": "Select CatBoost on holdout MAE from five tuned advanced models.",
        "why": "Lowest holdout MAE and MAPE of every model evaluated.",
        "alternatives": "LightGBM ($135.78) and HistGradientBoosting ($136.21) are effectively tied; "
                        "RandomForest and XGBoost were clearly behind.",
        "evidence": "CatBoost $132.15 vs LightGBM $135.78 — a $3.63 gap on a $2,280 mean "
                    "prediction, which is inside fold-to-fold variation.",
        "impact": "A defensible pick, but one of three near-equivalent models rather than a decisive "
                  "winner. Reporting it otherwise would overstate the evidence.",
    },
    {
        "title": "MAE as the headline metric, not RMSE",
        "tag": ("neutral", "Modelling"),
        "decision": "Rank and select every model on holdout MAE.",
        "why": "The business cost of a mispriced load is roughly linear in dollars. RMSE is "
               "dominated by a small number of very high-rate loads no model predicts well.",
        "alternatives": "RMSE; R²; MAPE.",
        "evidence": "RMSE varied only $640–$654 across six model families while MAE spanned "
                    "$132–$158 — RMSE could not discriminate between them.",
        "impact": "Selection tracked genuine differences in typical-load accuracy.",
    },
    {
        "title": "Holdout-first evaluation, scored once",
        "tag": ("neutral", "Evaluation"),
        "decision": "Tune with cross-validation on the training block only; score the holdout once "
                    "per model and never tune against it.",
        "why": "Repeatedly consulting a holdout turns it into a training set and inflates the final "
               "estimate.",
        "alternatives": "Select on CV score alone; iterate against the holdout.",
        "evidence": "CV and holdout rankings genuinely disagreed (RandomForest CV $167.68 → holdout "
                    "$144.85) — exactly the information a contaminated holdout would have destroyed.",
        "impact": "The reported $114.99 MAE is an honest estimate rather than an optimised one.",
    },
    {
        "title": "Three independent explainability methods",
        "tag": ("neutral", "Evaluation"),
        "decision": "Report CatBoost native importance, permutation importance and SHAP together.",
        "why": "Each has a different blind spot: native is in-sample and over-credits "
               "high-cardinality splits; permutation is out-of-sample but under-credits correlated "
               "features; SHAP is additive but splits credit between them.",
        "alternatives": "Native importance alone — the common shortcut.",
        "evidence": "The disagreement was itself the finding: <code>market_index</code> ranks 7th "
                    "natively but 15th by permutation, with a measured elasticity of 0.139.",
        "impact": "Prevented overstating market context as a driver, and explained the shape of the "
                  "December chart.",
    },
    {
        "title": "Stateless-first pipeline architecture",
        "tag": ("neutral", "Engineering"),
        "decision": "Order the pipeline as cleaning → feature construction → encoding, with all "
                    "fitted state confined to the final <code>ColumnTransformer</code>.",
        "why": "If every learned statistic lives in one stage, the leakage audit reduces to "
               "inspecting that stage. Scattered fitted state cannot be reasoned about.",
        "alternatives": "A monolithic transformer; cleaning outside the pipeline (the inherited "
                        "approach, which risks train/inference divergence).",
        "evidence": "Integrity guards assert feature-name preservation, column alignment, no "
                    "residual NaNs, and exact reproducibility on refit.",
        "impact": "Training and inference are provably identical, and the December path reuses the "
                  "same object rather than duplicating logic.",
    },
    {
        "title": "Reconstructing the missing December input file",
        "tag": ("warn", "Recovered"),
        "decision": "Rebuild <code>data/december_chart_inputs.csv</code> from the constants in "
                    "<code>score.py</code>, and reconstruct the six columns its schema omits.",
        "why": "The file the scorer requires was absent from the inherited repository, making the "
               "second deliverable impossible to produce.",
        "alternatives": "Skip the deliverable; guess the scenario.",
        "evidence": "<code>score.py</code> pins pickup, delivery, distance, equipment, weight and "
                    "the date range exactly. Coordinates come from a verified city lookup (exactly "
                    "one coordinate pair per city); market context from a per-date table covering "
                    "all 31 December dates, all inside the training range.",
        "impact": "Unblocked the deliverable with an exact rather than approximate reconstruction.",
    },
]

TIMELINE: list[tuple[str, str, str, str]] = [
    ("Phase 1", "Data audit",
     "Schema profiling, validation and quality checks across both datasets.",
     "reports/data_audit.md"),
    ("Phase 2", "Exploratory analysis",
     "Distributions, correlations, geography and temporal structure; 27 figures.",
     "reports/exploratory_data_analysis.md"),
    ("Phase 3", "Foundation repair &amp; feature engineering",
     "Fixed the broken import structure and a swallowed exception that had silently renamed all "
     "142 processed columns to integers. Built the production pipeline: 156 named features, "
     "integrity guards, time-split utility and the dual inference path.",
     "reports/preprocessing_report.md"),
    ("Phase 4", "Baselines &amp; evaluation framework",
     "Seven baselines from constant to linear; time-aware harness with per-fold preprocessing.",
     "reports/baseline_models.md"),
    ("Phase 5", "Advanced models &amp; tuning",
     "Five model families tuned by randomised search over expanding-window folds (~33 min).",
     "reports/model_comparison.md"),
    ("Phase 6", "Explainability &amp; error analysis",
     "Native, permutation and SHAP importance; segment errors; residual diagnostics; 28 figures.",
     "reports/explainability_report.md"),
    ("Phase 7", "Final training &amp; predictions",
     "One training run on the full window, smearing correction, submission and scorer execution.",
     "reports/final_predictions.md"),
    ("Phase 8", "Documentation &amp; packaging",
     "Professional README, 9-page technical report PDF, submission checklist, binary-safe git "
     "configuration.",
     "reports/Freight_Rate_Prediction_Technical_Report.pdf"),
    ("Phase 9", "Project dashboard",
     "This interactive case study, assembled entirely from existing artifacts.",
     "dashboard/index.html"),
]

REPO_TREE: list[tuple[str, str, str]] = [
    ("config/", "dir", "Single source of truth: paths, column roles, split dates, feature switches."),
    ("  config.yaml", "file", "Every tunable constant. Editing this changes behaviour everywhere."),
    ("data/", "dir", "Raw inputs, tracked in git."),
    ("  train_test.csv", "file", "48,000 labelled loads, Jan–Oct 2025."),
    ("  validation.csv", "file", "12,000 unlabelled loads, Nov–Dec 2025."),
    ("  december_chart_inputs.csv", "file", "31-row fixed scenario, reconstructed from score.py."),
    ("src/", "dir", "All source, imported as a package."),
    ("  config.py", "file", "Typed YAML loader and global seeding."),
    ("  logger.py", "file", "Central logging; replaced scattered print() calls."),
    ("  data_loader.py", "file", "Raw CSV loading with clear failure messages."),
    ("  data_profiler.py", "file", "Per-feature profiling for the audit report."),
    ("  data_validator.py", "file", "Schema and quality checks."),
    ("  eda.py", "file", "EDA computation and figure generation."),
    ("  transformers.py", "file", "Custom transformers; all implement get_feature_names_out."),
    ("  feature_engineering.py", "file", "Stateless feature construction; schema frozen at fit."),
    ("  preprocessing.py", "file", "Imputer, scaler and encoder builders — the only fitted parts."),
    ("  pipeline.py", "file", "Pipeline assembly plus integrity guards (names, alignment, leakage)."),
    ("  splitting.py", "file", "Temporal holdout and expanding-window CV utilities."),
    ("  inference.py", "file", "Full and reduced-feature paths; city and market lookups."),
    ("  metrics.py", "file", "MAE, RMSE, R², MAPE with a positive-prediction floor."),
    ("  baselines.py", "file", "Constant, rate-per-mile and linear baselines."),
    ("  evaluation.py", "file", "Time-aware harness; refits preprocessing per fold."),
    ("  advanced_models.py", "file", "Model specs and search spaces; skips missing libraries."),
    ("  tuning.py", "file", "Leakage-safe randomised search over the composed pipeline."),
    ("  explainability.py", "file", "Native and permutation importance, SHAP computation."),
    ("  error_analysis.py", "file", "Segment errors and residual diagnostics."),
    ("  final_model.py", "file", "Final fit and Duan smearing correction."),
    ("  build_technical_report.py", "file", "Generates the technical report PDF from JSON."),
    ("  build_dashboard.py", "file", "Generates this dashboard."),
    ("  run_*.py", "file", "One entry point per phase."),
    ("tests/", "dir", "93 tests: config, cleaning, features, pipeline integrity, splitting."),
    ("reports/", "dir", "11 markdown reports, ranked importance CSV, technical report PDF."),
    ("figures/", "dir", "55 figures across eda, shap, importance, residuals, error_analysis."),
    ("models/", "dir", "Fitted pipelines and metadata (git-ignored, regenerable)."),
    ("score.py", "file", "The provided scorer. Unmodified."),
    ("validation_predictions.csv", "file", "The 12,000-row submission file."),
]

READINESS: list[tuple[str, str, str, str]] = [
    ("Configuration", "good", "Ready",
     "All paths, column roles, split dates and feature switches live in "
     "<code>config/config.yaml</code>. No hardcoded literals in the pipeline."),
    ("Logging", "good", "Ready",
     "Central <code>logger.py</code>; every stage emits structured progress. No bare "
     "<code>print()</code> in library code."),
    ("Reproducibility", "good", "Ready",
     "Global seed 42 across <code>random</code>, <code>numpy</code> and "
     "<code>PYTHONHASHSEED</code>. An independent refit is asserted to reproduce the feature "
     "matrix exactly."),
    ("Testing", "good", "Ready",
     "93 tests covering config, cleaning, features, pipeline integrity, splitting and inference. "
     "<code>ruff</code> clean."),
    ("Model persistence", "good", "Ready",
     "Complete pipeline serialised to <code>models/final_model.joblib</code> with a metadata record "
     "(seed, hyperparameters, metrics, smearing factor)."),
    ("Documentation", "good", "Ready",
     "README, 9-page technical report PDF, 11 markdown reports, and this dashboard."),
    ("Prediction pipeline", "good", "Ready",
     "Both the full and reduced-feature paths converge on the same fitted object; verified "
     "end-to-end against the official scorer."),
    ("Submission artifacts", "good", "Ready",
     "12,000-row CSV and the December chart, both scorer-validated."),
    ("Monitoring", "warn", "Not built",
     "No drift detection or realised-vs-predicted tracking exists. Required before production use."),
    ("Prediction intervals", "warn", "Not built",
     "Heteroscedasticity is confirmed, but the model emits a point estimate only."),
    ("Serving layer", "warn", "Not built",
     "No API, container or batch scheduler. The model is a library artifact, not a service."),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def esc(text: object) -> str:
    """Escape a value for safe HTML text content."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def kpi(label: str, value: str, note: str = "", hero: bool = False, small: bool = False) -> str:
    """Render a KPI tile."""
    classes = "kpi hero" if hero else "kpi"
    vclass = "k-value sm" if small else "k-value"
    note_html = f'<div class="k-note">{note}</div>' if note else ""
    return (
        f'<div class="{classes}"><div class="k-label">{label}</div>'
        f'<div class="{vclass}">{value}</div>{note_html}</div>'
    )


def figure(data: DashboardData, key: str, caption: str) -> str:
    """Embed a curated figure, or return nothing if it is unavailable."""
    src = data.figures.get(key)
    if not src:
        return ""
    return (
        f'<figure><img src="{src}" alt="{esc(caption)}" loading="lazy">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def metric_table(rows: list[dict], highlight: str | None = None) -> str:
    """Render a model metric table, optionally highlighting one row."""
    cells = []
    for row in rows:
        attr = ' class="highlight"' if highlight and row["name"] == highlight else ""
        cells.append(
            f"<tr{attr}><td>{esc(row['name'])}</td>"
            f'<td class="num">${row["mae"]:,.2f}</td>'
            f'<td class="num">${row["rmse"]:,.2f}</td>'
            f'<td class="num">{row["r2"]:.4f}</td>'
            f'<td class="num">{row["mape"]:.2f}%</td></tr>'
        )
    body = "".join(cells)
    return (
        '<div class="tablewrap"><table><thead><tr><th>Model</th>'
        '<th class="num">MAE</th><th class="num">RMSE</th>'
        '<th class="num">R&sup2;</th><th class="num">MAPE</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def section(sid: str, eyebrow: str, title: str, lede: str, body: str) -> str:
    """Wrap content in a navigable panel."""
    return (
        f'<section class="panel" id="{sid}">'
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<h2 class="title">{title}</h2>'
        f'<p class="lede">{lede}</p>{body}</section>'
    )


def chartbox(cid: str, title: str, subtitle: str, controls: str = "", sub_id: str = "") -> str:
    """Render a chart container that the JavaScript draws into."""
    sub_attr = f' id="{sub_id}"' if sub_id else ""
    return (
        f'<div class="chartbox"><h4>{title}</h4>'
        f'<p class="sub"{sub_attr}>{subtitle}</p>{controls}'
        f'<div class="chart-scroll"><div id="{cid}"></div></div></div>'
    )


def _params(params: dict) -> str:
    """Render a hyperparameter dict compactly."""
    return esc(json.dumps(params)[1:-1].replace('"', ""))
