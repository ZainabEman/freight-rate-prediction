# Freight Rate Prediction

Predicting freight `posted_rate` (USD) per load from lane, equipment, weight and
market-context features.

Development data covers **2025-01-01 to 2025-10-31** (48,000 labelled loads).
Scoring data covers **2025-11-01 to 2025-12-31** (12,000 unlabelled loads).
The two windows do not overlap, so this is a **forward-extrapolation** problem
rather than an i.i.d. tabular regression, and the whole pipeline is built around
that fact.

> **Project status: Phase 5 of 9 complete.** Data audit, EDA, cleaning, feature
> engineering, the production preprocessing pipeline, baselines and tuned
> advanced models are done and tested. No submission files exist yet - see
> [Roadmap](#roadmap).

---

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Running

All entry points are module invocations from the repository root, so `src` is
importable as a package.

```bash
# Phase 1 - schema audit, profiling and validation
python -m src.run_data_audit_phase1

# Phase 2 - exploratory data analysis and figures
python -m src.run_eda_phase2

# Phase 3 - build, verify and persist the preprocessing pipeline
python -m src.run_preprocessing_phase3

# Phase 4 - train, evaluate and compare baseline models
python -m src.run_baselines_phase4

# Phase 5 - tune and compare advanced models
python -m src.run_advanced_models_phase5

# Test suite
python -m pytest tests -q
```

`run_preprocessing_phase3` builds the feature pipeline: it loads the raw CSVs,
fits on training data only, runs every integrity guard, verifies the temporal
split and the reduced-feature December path, and persists the fitted pipeline.

`run_baselines_phase4` then evaluates seven baselines against the temporal
holdout and expanding-window CV, writes the comparison report, and persists the
winning baseline as the reference bar for Phase 5.

## Repository structure

```
.
├── config/config.yaml          All paths, column roles, split dates and
│                               feature switches. Single source of truth.
├── data/                       Raw inputs (tracked)
│   ├── train_test.csv                  48,000 x 14, labelled
│   ├── validation.csv                  12,000 x 13, unlabelled
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv       reconstructed from score.py's spec
├── src/
│   ├── config.py               Typed YAML config loader + global seeding
│   ├── logger.py               Central logging setup
│   ├── data_loader.py          Raw CSV loading
│   ├── data_profiler.py        Per-feature profiling
│   ├── data_validator.py       Schema and quality checks
│   ├── eda.py                  EDA computation and figures
│   ├── transformers.py         Custom sklearn transformers
│   ├── feature_engineering.py  Stateless feature construction
│   ├── preprocessing.py        Imputer / scaler / encoder builders
│   ├── pipeline.py             Pipeline assembly + integrity guards
│   ├── splitting.py            Time-based split utilities
│   ├── inference.py            Full and reduced-feature inference paths
│   ├── metrics.py              MAE / RMSE / R2 / MAPE
│   ├── baselines.py            Baseline regressors
│   ├── evaluation.py           Time-aware evaluation harness
│   ├── advanced_models.py      Advanced model specs + search spaces
│   ├── tuning.py               Leakage-safe randomised search
│   ├── reporting_phase3.py     Phase-3 report generation
│   └── run_*.py                Phase entry points
├── notebooks/                  Exploratory notebooks
├── reports/                    Generated markdown reports
├── figures/                    Generated figures
├── tests/                      93 tests
├── processed/                  Derived matrices (git-ignored)
├── models/                     Fitted pipeline + metadata (git-ignored)
└── score.py                    Provided scorer (unmodified)
```

## Approach

### Train / validation split

Because the scoring window sits entirely after the development window, a random
split would measure interpolation skill and report optimistic numbers. The
holdout is instead the **final contiguous block** of the development window:

| | rows | dates |
|:--|--:|:--|
| train | 38,477 | 2025-01-01 .. 2025-08-31 |
| holdout | 9,523 | 2025-09-01 .. 2025-10-31 |

Cross-validation uses `TimeSeriesSplit` over date-ordered rows, so every
validation fold lies strictly after its training fold.

### Data quality issues found and addressed

| Issue | Evidence | Resolution |
|:--|:--|:--|
| 292 negative `weight` values | `abs()` of the negative population is distributionally identical to the positive one (mean 31,724 vs 31,415, identical `[5000, 47500]` support) | Sign-flip repair via `abs()`, then clip |
| Missing `weight` / `market_index` | 0.63% / 0.78% in train; 1.38% / 2.08% in validation | Median imputation (fit on train) **plus** explicit missingness indicators, since missingness is itself a shift signal |
| 8 cities in validation absent from training | Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego - ~6% of scoring rows | Explicit `*_is_unknown` indicators; geography carried by coordinate features that work regardless |
| `date_year` constant, `date_month` out-of-range at inference | Train months `{1..10}` vs scoring months `{11, 12}` | Replaced with cyclical day-of-year / day-of-week encodings |

### Feature engineering

156 model-ready features, all named:

- **Temporal (5)** - `doy_sin/cos`, `dow_sin/cos`, `is_weekend`. Cyclical so
  November and December never leave the training support.
- **Geospatial (5)** - `haversine_miles`, `bearing_sin/cos`, `lon_delta`,
  `lat_delta`. `corr(delivery_lon, rate) = -0.257` shows a westbound premium a
  single distance scalar cannot express.
- **Interactions (2)** - `log_distance` (rate-per-mile falls with distance,
  `r = -0.335`), `weight_per_mile` (load density).
- **Indicators (5)** - missingness and unknown-category flags.
- **Raw numeric (8)** and **one-hot categorical (131)**.

### Two inference paths

`score.py` requires predictions on two different schemas:

1. **Full validation** - all 13 raw feature columns present.
2. **Fixed December chart** - only `pickup, delivery, distance, equipment,
   weight, date`. Coordinates, `market_index` and `quote_signal` are absent.

The second is served by reconstructing the missing columns:

- **Coordinates** from an exact city lookup (verified: one coordinate pair per
  city).
- **Market context** from a per-date table built from `validation.csv`, which
  covers all 31 December dates. Their daily `market_index` spans 0.831..1.045,
  inside the training range of 0.676..1.468, so no extrapolation is needed.

This matters because daily market index is the dominant date-driven signal
(`r = 0.577` against daily mean rate-per-mile) - it is what allows the December
curve to vary meaningfully when only the date changes.

## Reproducibility

- Global seed (42) applied to `random`, `numpy` and `PYTHONHASHSEED`.
- The pipeline is fitted on training rows only; an independent refit is asserted
  to reproduce the feature matrix exactly.
- The fitted pipeline and a metadata record (seed, feature names, split summary,
  cleaning statistics) are persisted to `models/`.
- Derived artifacts are git-ignored and regenerable from the raw CSVs.

## Reports

| Report | Contents |
|:--|:--|
| `reports/data_audit.md` | Schema, dtypes, missingness, duplicates, quality issues |
| `reports/feature_dictionary.md` | Raw column inventory |
| `reports/exploratory_data_analysis.md` | Distributions, correlations, geography, temporal structure |
| `reports/preprocessing_report.md` | Every cleaning and feature decision with its evidence |
| `reports/feature_dictionary_phase3.md` | The 156 model-ready features |
| `reports/baseline_models.md` | Baseline comparison on the temporal holdout |
| `reports/model_comparison.md` | Advanced model tuning and final comparison |

## Results

Measured on the temporal holdout (Sep-Oct 2025, 9,523 rows), ranked by MAE.
Advanced models are tuned by `RandomizedSearchCV` over `TimeSeriesSplit(3)`:

| Model | MAE | RMSE | R2 | MAPE |
|:--|--:|--:|--:|--:|
| **CatBoost** (selected) | **$132.15** | **$641.48** | **0.8233** | **5.68%** |
| LightGBM | $135.78 | $642.24 | 0.8229 | 5.88% |
| HistGradientBoosting | $136.21 | $640.69 | 0.8237 | 5.93% |
| RandomForest | $144.85 | $646.24 | 0.8207 | 6.27% |
| Ridge (log target) - best baseline | $145.24 | $644.04 | 0.8219 | 6.27% |
| XGBoost | $157.60 | $654.23 | 0.8162 | 6.84% |

All advanced models train on `log(posted_rate)` and are persisted as complete
pipelines (preprocessing + model), so they predict directly from raw frames.

### Baseline detail

| Model | MAE | RMSE | R2 | MAPE |
|:--|--:|--:|--:|--:|
| **Ridge (log target)** | **$145.24** | **$644.04** | **0.8219** | **6.27%** |
| Ridge (alpha=1.0) | $149.36 | $641.36 | 0.8234 | 8.65% |
| LinearRegression | $149.37 | $641.36 | 0.8234 | 8.65% |
| Rate-per-mile (by equipment) | $229.10 | $670.66 | 0.8069 | 10.49% |
| Rate-per-mile (global) | $256.95 | $684.25 | 0.7990 | 11.63% |
| Median (constant) | $1,148.92 | $1,569.42 | -0.0576 | 70.15% |
| Mean (constant) | $1,178.80 | $1,526.23 | -0.0002 | 83.43% |

## Roadmap

| Phase | Status |
|:--|:--|
| 0 - Setup | Complete |
| 1 - Data audit and validation | Complete |
| 2 - EDA | Complete |
| 3 - Foundation repair and feature engineering | Complete |
| 4 - Baselines and validation framework | Complete |
| 5 - Advanced models and tuning | Complete |
| 6 - Explainability and error analysis | Not started |
| 7 - Final training and predictions | Not started |
| 8 - Documentation and technical report | Not started |
| 9 - Loom walkthrough and submission | Not started |

## Scoring

Once Phase 7 produces the prediction files:

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and writes
`scorer_results/candidate_december.png`.

---

## Assessment brief (original instructions)

The original task description from `freight-rate-ml-assessment.pdf`:

1. Train and validate using `data/train_test.csv`.
2. Predict every load in `data/validation.csv` (each has a unique `load_id`).
3. Fill `predicted_rate` in `data/validation_predictions_template.csv` and save
   as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv`.
5. Run `score.py` as shown above.

**Submit:** GitHub repository, `validation_predictions.csv`, a PDF/DOCX report
covering the validation and split approach plus `candidate_december.png`, and a
2-3 minute Loom walkthrough.
