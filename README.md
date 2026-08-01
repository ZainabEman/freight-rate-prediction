# Freight Rate Prediction

Predicting freight `posted_rate` (USD) per load from lane, equipment, weight and
market-context features.

**Final model: CatBoost on a log target with Duan smearing correction —
MAE $114.99, MAPE 5.03%, R² 0.830** on a strictly out-of-sample temporal holdout
of 9,523 loads.

> **Status: complete and validated against the official scorer.**
> `validation_predictions.csv` (12,000 rows) and
> `scorer_results/candidate_december.png` are generated and verified.

---

## Table of contents

- [Business problem](#business-problem)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [Pipeline overview](#pipeline-overview)
- [Validation strategy](#validation-strategy)
- [Data quality issues](#data-quality-issues)
- [Feature engineering](#feature-engineering)
- [Models evaluated](#models-evaluated)
- [Final model](#final-model)
- [Explainability](#explainability)
- [Error analysis](#error-analysis)
- [Submission artifacts](#submission-artifacts)
- [Reports](#reports)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

---

## Business problem

Given a load's lane, equipment type, weight, date and market context, predict the
rate it will post at. Accurate pricing lets a broker quote competitively without
eroding margin.

The business cost of a mispriced load is approximately **linear in dollars**, so
**MAE is the headline metric** throughout this project rather than RMSE, which
would be dominated by a handful of very high-rate loads.

**The defining constraint:** the labelled window (Jan–Oct 2025) and the scoring
window (Nov–Dec 2025) do not overlap. This is a **forward-extrapolation**
problem, not an i.i.d. tabular regression, and every design decision follows from
that.

## Dataset

| File | Rows | Cols | Date range | Labelled |
|:--|--:|--:|:--|:--|
| `data/train_test.csv` | 48,000 | 14 | 2025-01-01 → 2025-10-31 | Yes |
| `data/validation.csv` | 12,000 | 13 | 2025-11-01 → 2025-12-31 | No |
| `data/validation_predictions_template.csv` | 12,000 | 2 | — | — |
| `data/december_chart_inputs.csv` | 31 | 7 | 2025-12-01 → 2025-12-31 | — |

**Target** `posted_rate` is strictly positive and right-skewed (median $2,031,
mean $2,374, max $25,533).

**Features:** 3 categorical (`pickup`, `delivery`, `equipment`), 1 date, and 8
numeric (origin/destination lat-lon, `distance`, `weight`, `market_index`,
`quote_signal`).

> `data/december_chart_inputs.csv` was **absent** from the original repository.
> It is reconstructed exactly from the constants in `score.py`.

## Installation

Requires **Python 3.11+**.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

The gradient-boosting libraries (XGBoost, LightGBM, CatBoost) are optional at
runtime — `src/advanced_models.py` skips any that are missing rather than
failing.

## Usage

All entry points are module invocations from the repository root.

```bash
# Phase 1 — schema audit, profiling and validation
python -m src.run_data_audit_phase1

# Phase 2 — exploratory data analysis and figures
python -m src.run_eda_phase2

# Phase 3 — build, verify and persist the preprocessing pipeline
python -m src.run_preprocessing_phase3

# Phase 4 — baseline models and the time-aware evaluation framework
python -m src.run_baselines_phase4

# Phase 5 — tune and compare advanced models  (~33 min)
python -m src.run_advanced_models_phase5

# Phase 6 — explainability, error analysis, business insights
python -m src.run_explainability_phase6

# Phase 7 — final training, predictions and scorer execution
python -m src.run_final_predictions_phase7

# Technical report (PDF)
python -m src.build_technical_report

# Interactive dashboard (self-contained HTML)
python -m src.build_dashboard

# Test suite (93 tests)
python -m pytest tests -q
```

To reproduce only the submission from a clean checkout, run phases 3 → 7.

**Scoring** (run automatically by Phase 7, or manually):

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

## Repository structure

```
.
├── config/config.yaml              Paths, column roles, split dates, feature
│                                   switches. Single source of truth.
├── data/                           Raw inputs (tracked)
├── src/
│   ├── config.py                   Typed YAML loader + global seeding
│   ├── logger.py                   Central logging
│   ├── data_loader.py              Raw CSV loading
│   ├── data_profiler.py            Per-feature profiling
│   ├── data_validator.py           Schema and quality checks
│   ├── eda.py                      EDA computation and figures
│   ├── transformers.py             Custom sklearn transformers
│   ├── feature_engineering.py      Stateless feature construction
│   ├── preprocessing.py            Imputer / scaler / encoder builders
│   ├── pipeline.py                 Pipeline assembly + integrity guards
│   ├── splitting.py                Time-based split utilities
│   ├── inference.py                Full and reduced-feature inference paths
│   ├── metrics.py                  MAE / RMSE / R² / MAPE
│   ├── baselines.py                Baseline regressors
│   ├── evaluation.py               Time-aware evaluation harness
│   ├── advanced_models.py          Advanced model specs + search spaces
│   ├── tuning.py                   Leakage-safe randomised search
│   ├── explainability.py           Feature importance + SHAP
│   ├── error_analysis.py           Segment errors + residual diagnostics
│   ├── final_model.py              Final fit + Duan smearing correction
│   ├── reporting_phase3.py         Phase-3 report generation
│   ├── reporting_phase6.py         Phase-6 report generation
│   ├── build_technical_report.py   Technical report PDF
│   ├── dashboard_data.py           Dashboard artifact collection
│   ├── dashboard_assets.py         Dashboard CSS + interactive charts
│   ├── dashboard_sections.py       Decision log, timeline, repo map
│   ├── dashboard_panels.py         Dashboard panel renderers
│   ├── build_dashboard.py          Builds dashboard/index.html
│   └── run_*.py                    Phase entry points
├── dashboard/index.html            Self-contained interactive case study
├── notebooks/                      Exploratory notebooks (phases 1–2)
├── reports/                        Generated reports + technical report PDF
├── figures/                        eda, shap, importance, residuals,
│                                   error_analysis
├── tests/                          93 tests
├── processed/                      Derived matrices (git-ignored)
├── models/                         Fitted pipelines + metadata (git-ignored)
├── scorer_results/                 Scorer output (git-ignored)
├── score.py                        Provided scorer (unmodified)
└── validation_predictions.csv      Submission file
```

## Pipeline overview

```
raw CSV
  └─ RawDataCleaner        stateless row-local repairs (weight sign, whitespace)
  └─ FeatureBuilder        stateless feature construction → 156 named features
  └─ ColumnTransformer     the ONLY fitted stage (impute, scale, one-hot)
  └─ CatBoost              log target, inverted with exp
  └─ Duan smearing         bias correction, strictly positive
```

Stateless stages run first so that **every statistic learned from data lives in a
single auditable place**, which makes the leakage audit tractable. Integrity
guards assert feature-name preservation, train/inference column alignment, no
residual missing values, and that neither the target nor `load_id` reaches the
feature matrix.

## Validation strategy

A random split would measure interpolation skill while the real task is forward
extrapolation. The holdout is instead the **final contiguous block** of the
development window:

| Split | Rows | Dates | Purpose |
|:--|--:|:--|:--|
| Training | 38,477 | 2025-01-01 → 2025-08-31 | Fitting + hyperparameter search |
| Holdout | 9,523 | 2025-09-01 → 2025-10-31 | Scored once, never tuned against |

Cross-validation uses `TimeSeriesSplit` over date-ordered rows so every
validation fold lies strictly after its training fold.

**Leakage is prevented structurally, not by convention:** preprocessing is a step
*inside* the searched pipeline, so scikit-learn clones and refits it
independently within every fold.

## Data quality issues

| Issue | Evidence | Resolution |
|:--|:--|:--|
| 292 negative `weight` values | `abs()` of the negative population is distributionally identical to the positive one (mean 31,724 vs 31,415; identical `[5000, 47500]` support) | Sign-flip repair via `abs()`, then clip |
| Missing `weight` / `market_index` | 0.63% / 0.78% in train; **1.38% / 2.08%** in validation | Median imputation (fit on train) **plus** missingness indicators — the rate shift is itself signal |
| 8 cities in validation absent from training | Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego — ~6% of scoring rows | Explicit `*_is_unknown` indicators; geography carried by coordinate features that work regardless |
| `date_year` constant, `date_month` out of range | Train months `{1..10}` vs scoring `{11, 12}` | Replaced with cyclical day-of-year / day-of-week encodings |

## Feature engineering

**156 model-ready features**, all named end-to-end:

| Group | N | Detail |
|:--|--:|:--|
| Temporal | 5 | `doy_sin/cos`, `dow_sin/cos`, `is_weekend` — cyclical, so Nov–Dec never leave the training support |
| Geospatial | 5 | `haversine_miles`, `bearing_sin/cos`, `lon_delta`, `lat_delta` — captures the westbound premium (`r = −0.257`) |
| Interactions | 2 | `log_distance` (concave distance/rate, `r = −0.335`), `weight_per_mile` |
| Indicators | 5 | Missingness and unknown-category flags |
| Raw numeric | 8 | Coordinates, distance, weight, market signals |
| One-hot | 131 | pickup (64), delivery (64), equipment (3) |

## Models evaluated

Holdout MAE, Sep–Oct 2025 (9,523 loads):

| Model | MAE | RMSE | R² | MAPE |
|:--|--:|--:|--:|--:|
| **CatBoost + smearing** ← final | **$114.99** | **$636.25** | **0.8302** | **5.03%** |
| CatBoost | $132.15 | $641.48 | 0.8233 | 5.68% |
| LightGBM | $135.78 | $642.24 | 0.8229 | 5.88% |
| HistGradientBoosting | $136.21 | $640.69 | 0.8237 | 5.93% |
| RandomForest | $144.85 | $646.24 | 0.8207 | 6.27% |
| Ridge (log target) — best baseline | $145.24 | $644.04 | 0.8219 | 6.27% |
| LinearRegression | $149.37 | $641.36 | 0.8234 | 8.65% |
| XGBoost | $157.60 | $654.23 | 0.8162 | 6.84% |
| Rate-per-mile by equipment | $229.10 | $670.66 | 0.8069 | 10.49% |
| Median (constant) | $1,148.92 | $1,569.42 | −0.0576 | 70.15% |

Each advanced model was tuned with `RandomizedSearchCV` over 3 expanding-window
folds, scored by negative MAE.

> **Honest caveat:** the top three advanced models sit within $4 of each other,
> which is inside fold-to-fold noise. CatBoost is the defensible pick, but it is
> one of three near-equivalent models rather than a decisive winner.

## Final model

| Item | Value |
|:--|:--|
| Algorithm | CatBoost |
| Hyperparameters | `learning_rate=0.03, depth=6, iterations=400, l2_leaf_reg=1.0` |
| Target | `log(posted_rate)`, inverted with `exp` |
| Back-transform | Duan smearing, factor **1.0128** |
| Training data | 48,000 loads, full Jan–Oct window |
| Features | 156 |

**Why smearing.** Phase 6 measured a systematic under-pricing bias of **+$101.81**
(t = 15.69) — the expected signature of `exp()` back-transformation returning a
conditional *median* for a right-skewed target. Duan's estimator was **evaluated
before adoption**, since smearing targets the conditional mean while MAE is
minimised by the median, so the two can conflict. Here they did not:

| Metric | Uncorrected | Corrected | Change |
|:--|--:|--:|--:|
| MAE | $132.15 | $114.99 | **−13.0%** |
| RMSE | $641.48 | $636.25 | −0.8% |
| MAPE | 5.68% | 5.03% | −0.65 pp |

The factor is strictly positive, so the correction cannot violate the scorer's
positivity constraint.

## Explainability

Three independent methods, each with a different blind spot, all agreeing:

| Feature | Native | Permutation rank |
|:--|--:|--:|
| `log_distance` | 30.75 | 1 |
| `distance` | 30.65 | 3 |
| `haversine_miles` | 30.40 | 2 |
| `equipment_Dry Van` | 2.06 | 4 |
| `market_index` | 0.68 | 15 |

**Distance is the price.** Three distance features carry over 90% of importance.

**The `market_index` discrepancy is informative:** 7th natively but 15th
out-of-sample, with a measured elasticity of only **0.139**. The often-quoted
+0.577 daily correlation measures co-movement of *averages*, not sensitivity —
a distinction that directly explains the shape of the December chart.

SHAP values are additive in logs and therefore **multiplicative in dollars**: a
SHAP value of +0.10 is roughly a +10.5% rate effect.

## Error analysis

| Diagnostic | Measurement | Verdict |
|:--|:--|:--|
| Heavy tails | Excess kurtosis 265; worst 1% of loads carry **39.3%** of all error | **Present, dominant** |
| Heteroscedasticity | Residual SD varies **4.24×** across prediction quintiles | **Present** |
| Bias | Mean residual +$101.81 (t = 15.69) | **Present → corrected** |

Median absolute error is **$54** against a mean of $132. That gap is the whole
story: typical loads are priced very accurately and a small minority are not.
**This is why RMSE stayed pinned near $640 across all six model families** — the
tail is not something hyperparameters can fix.

## Submission artifacts

| Artifact | Description | Status |
|:--|:--|:--|
| `validation_predictions.csv` | 12,000 rows, `load_id,predicted_rate` | ✅ Validated |
| `data/december_chart_inputs.csv` | 31 rows with `predicted_rate` filled | ✅ Validated |
| `scorer_results/candidate_december.png` | Chart from `score.py` | ✅ Generated |
| `reports/Freight_Rate_Prediction_Technical_Report.pdf` | 9-page technical report | ✅ Generated |
| `dashboard/index.html` | Interactive case study, 23 sections | ✅ Generated |
| `models/final_model.joblib` | Complete fitted pipeline | ✅ Persisted |

Prediction statistics: min $201.65 · median $1,975.49 · mean $2,280.56 · max
$6,465.66. All 12,000 finite, strictly positive, template ordering preserved.

**Scorer output:**

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
```

**December chart:** $785.70–$799.22, spread $13.52 (1.71%). The near-flat curve is
**correct, not a defect** — market index swings +25.8% across those dates but the
measured elasticity of 0.139 implies only a +3.2% rate response. The visible
weekly periodicity comes from the cyclical day-of-week features.

## Reports

| Report | Contents |
|:--|:--|
| `reports/Freight_Rate_Prediction_Technical_Report.pdf` | **Full technical report** (9 pages) |
| `reports/data_audit.md` | Schema, dtypes, missingness, duplicates, quality issues |
| `reports/feature_dictionary.md` | Raw column inventory |
| `reports/exploratory_data_analysis.md` | Distributions, correlations, geography, temporal structure |
| `reports/preprocessing_report.md` | Every cleaning and feature decision with its evidence |
| `reports/feature_dictionary_phase3.md` | The 156 model-ready features |
| `reports/baseline_models.md` | Baseline comparison on the temporal holdout |
| `reports/model_comparison.md` | Advanced model tuning and final comparison |
| `reports/explainability_report.md` | Native, permutation and SHAP importance |
| `reports/error_analysis.md` | Segment errors, outliers, residual diagnostics |
| `reports/business_insights.md` | Measured pricing drivers and recommendations |
| `reports/final_predictions.md` | Final model, submission stats, scorer output |
| `reports/feature_importance.csv` | Ranked importance, all 156 features |

## Reproducibility

- Global seed (**42**) applied to `random`, `numpy` and `PYTHONHASHSEED`.
- The pipeline is fitted on training rows only; an independent refit is asserted
  to reproduce the feature matrix **exactly**.
- Fitted pipelines and metadata (seed, feature names, split summary, cleaning
  statistics) are persisted to `models/`.
- Derived artifacts are git-ignored and fully regenerable from the raw CSVs.
- 93 tests cover config, cleaning, features, pipeline integrity, splitting,
  inference and model smoke paths.

## Limitations

- **The heavy tail is unresolved.** RMSE ($636) is 5.5× MAE ($115) and did not
  respond to any model family tried. A minority of loads are priced by mechanisms
  not present in these features.
- **No labelled data for the scoring window.** Nov–Dec performance is inferred
  from Sep–Oct holdout behaviour and cannot be verified before submission.
- **Model-selection margin is within noise** — top three models differ by $4.
- **Eight cities appear only at scoring time** and are priced from geography alone.
- **December market context is reconstructed** from validation-set aggregates,
  since the scorer's December schema omits those columns.

## Future improvements

1. **Target the tail directly** — quantile regression or a two-stage
   normal/exceptional classifier. Single largest remaining opportunity.
2. **Prediction intervals** via quantile models, given confirmed heteroscedasticity.
3. **Native categorical handling** to replace 131 sparse one-hot columns.
4. **Lane-level aggregate features** (historical median $/mile per lane) computed
   time-aware to avoid leakage.
5. **Drift monitoring** on `market_index` and rate distributions in production.
