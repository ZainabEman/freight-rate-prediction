# Submission Checklist

Verified against the repository on completion of Phase 8.

## Required by the assessment brief

| # | Requirement | Artifact | Status |
|:--|:--|:--|:--|
| 1 | Accessible GitHub repository with code, dependencies and run instructions | This repository + `README.md` + `requirements.txt` | ⚠️ **Not yet pushed** |
| 2 | `validation_predictions.csv` with exactly `load_id,predicted_rate` | `validation_predictions.csv` | ✅ Complete |
| 3 | PDF/DOCX report: validation + split approach, and `candidate_december.png` | `reports/Freight_Rate_Prediction_Technical_Report.pdf` | ✅ Complete |
| 4 | 2–3 minute Loom walkthrough | — | ❌ **Not recorded** |

## Artifact verification

| Item | Check | Result |
|:--|:--|:--|
| **Submission CSV** | Exactly 12,000 rows | ✅ |
| | Columns exactly `[load_id, predicted_rate]` | ✅ |
| | `load_id` order matches provided template | ✅ |
| | All values finite | ✅ |
| | All values strictly positive (min $201.65) | ✅ |
| | No missing values | ✅ |
| **December inputs** | 31 rows, one per December date | ✅ |
| | Original 7-column schema and order preserved | ✅ |
| | All `predicted_rate` positive | ✅ |
| **Scorer** | `score.py` exits 0 | ✅ |
| | `scorer_results/candidate_december.png` produced | ✅ |
| **Technical report** | PDF opens and parses (9 pages) | ✅ |
| | All 18 required sections present | ✅ |
| | `candidate_december.png` embedded | ✅ |
| **README** | All 22 referenced file paths resolve | ✅ |
| | All 21 listed `src/` modules exist | ✅ |
| | All 8 documented entry points exist | ✅ |
| | No broken TOC anchors | ✅ |
| **Model** | `models/final_model.joblib` reloads and predicts | ✅ |
| | Metadata records seed, hyperparameters, metrics | ✅ |
| **Reports** | 11 markdown reports + 1 CSV + 1 PDF | ✅ |
| **Figures** | 55 figures across 5 categories | ✅ |
| **Tests** | 93 passed | ✅ |
| **Lint** | `ruff` clean (E9, F, W) | ✅ |

## Scorer output

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

## Final performance

| Metric | Holdout (Sep–Oct 2025, 9,523 loads) |
|:--|--:|
| MAE | **$114.99** |
| RMSE | $636.25 |
| R² | 0.8302 |
| MAPE | **5.03%** |

## Remaining before submission

1. **Push to GitHub** — everything is committed locally; no remote is configured.
   The grader needs an accessible repository URL.
2. **Record the Loom** (2–3 min), covering the five points the brief asks for:
   key EDA findings, data-quality issues and fixes, model reasoning, the
   train/validation split approach, and a code walkthrough.

Suggested Loom structure:

| Time | Topic | Key points |
|:--|:--|:--|
| 0:00–0:30 | Data findings | Distance drives price (r = 0.909); rate-per-mile falls with distance; equipment premiums; train/score windows do not overlap |
| 0:30–1:00 | Data quality | 292 negative weights (sign-flip, verified distributionally); missingness higher at scoring time; 8 unseen cities; degenerate `date_year`/`date_month` |
| 1:00–1:30 | Split approach | Why a random split would be wrong here; temporal holdout Sep–Oct; `TimeSeriesSplit`; preprocessing refit inside every fold |
| 1:30–2:15 | Model reasoning | Baselines → 5 tuned advanced models; CatBoost selected; smearing correction measured before adoption (−13% MAE) |
| 2:15–3:00 | Code walkthrough | `config.yaml` → `pipeline.py` integrity guards → `tuning.py` leakage control → `inference.py` dual path; note the near-flat December curve is correct |
