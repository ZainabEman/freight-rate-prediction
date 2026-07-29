# Baseline Models (Phase 4)

Regenerate with `python -m src.run_baselines_phase4`. Every number below is measured on the temporal holdout defined in `config/config.yaml`.

## Evaluation protocol

Development data is 2025-01-01..2025-10-31; the real scoring window is 2025-11-01..2025-12-31. Because the two do not overlap, every evaluation here is **forward-in-time**:

| Split | rows | dates |
|:--|--:|:--|
| fit | 38,477 | 2025-01-01 .. 2025-08-31 |
| holdout | 9,523 | 2025-09-01 .. 2025-10-31 |

A secondary expanding-window cross-validation (`TimeSeriesSplit`) reports MAE stability across folds. **A fresh preprocessing pipeline is fitted inside every split**, so no statistic learned from later data ever touches earlier data.

Predictions are clipped to a positive floor before scoring, matching the constraint `score.py` enforces at submission time.

## Model comparison

| model                         |     MAE |    RMSE |      R2 |   MAPE_% |   CV_MAE_mean |   CV_MAE_std |   fit_s |
|:------------------------------|--------:|--------:|--------:|---------:|--------------:|-------------:|--------:|
| Ridge (alpha=1.0, log target) |  145.24 |  644.04 |  0.8219 |     6.27 |        122.41 |        15.35 |     3.6 |
| Ridge (alpha=1.0)             |  149.36 |  641.36 |  0.8234 |     8.65 |        148.78 |         7.55 |     3.4 |
| LinearRegression              |  149.37 |  641.36 |  0.8234 |     8.65 |        148.81 |         7.6  |     7.3 |
| Rate-per-mile (by equipment)  |  229.1  |  670.66 |  0.8069 |    10.49 |        210.4  |        20.34 |     0.2 |
| Rate-per-mile (global)        |  256.95 |  684.25 |  0.799  |    11.63 |        242.14 |        16.12 |     0.1 |
| Median (constant)             | 1148.92 | 1569.42 | -0.0576 |    70.15 |       1141.83 |        22.39 |     0.1 |
| Mean (constant)               | 1178.8  | 1526.23 | -0.0002 |    83.43 |       1162.79 |        21.76 |     0.1 |

Ranked by holdout MAE. MAE is the selection criterion because the business cost of a mispriced load is roughly linear in dollars, and unlike RMSE it is not dominated by the small number of very high-rate loads (target max 25,533 against a median of 2,031).

## Model descriptions

- **Mean (constant)** (raw input, identity target): Predicts the training mean rate. Floor for R2 by construction.
- **Median (constant)** (raw input, identity target): Predicts the training median; stronger than the mean under MAE.
- **Rate-per-mile (global)** (raw input, identity target): distance x global median rate-per-mile.
- **Rate-per-mile (by equipment)** (raw input, identity target): distance x median rate-per-mile within equipment type. Domain baseline.
- **LinearRegression** (processed input, identity target): Ordinary least squares on the 156 engineered features.
- **Ridge (alpha=1.0)** (processed input, identity target): L2-regularised least squares; stabilises the 131 sparse one-hot columns.
- **Ridge (alpha=1.0, log target)** (processed input, log target): Ridge on log(rate). Addresses the right-skewed target and cannot emit non-positive predictions after inversion.

## Selected reference baseline

**Ridge (alpha=1.0, log target)**

| Metric | Holdout value |
|:--|--:|
| MAE | $145.24 |
| RMSE | $644.04 |
| R2 | 0.8219 |
| MAPE | 6.27% |
| CV MAE (mean +/- std) | 122.41 +/- 15.35 |
| Holdout rows | 9,523 |

This is the bar Phase 5 must beat. A gradient-boosted model that cannot improve on it is not worth the additional complexity.

## Leakage verification

- [x] Holdout is strictly future: train ends 2025-08-31, holdout starts 2025-09-01.
- [x] Train and holdout row sets are disjoint.
- [x] All 5 CV folds are forward-only and non-overlapping.
- [x] A fresh preprocessing pipeline is fitted per split; no fitted state crosses folds.

## Scope

Phase 4 covers splitting, baseline training, evaluation and comparison only. No hyperparameter search, no advanced models, no explainability and no submission files were produced.
