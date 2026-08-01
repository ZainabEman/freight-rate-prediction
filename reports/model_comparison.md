# Model Comparison (Phase 5)

Regenerate with `python -m src.run_advanced_models_phase5`. Every number is measured on the same temporal holdout Phase 4 used, so baselines and advanced models are directly comparable.

## Protocol

- Tuning: `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=3)` over date-sorted training rows, scored by negative MAE.
- **Leakage control:** preprocessing is a *step inside* the searched pipeline, so it is cloned and refitted independently within every CV fold. No imputation median, scaler statistic or one-hot vocabulary is ever learned from data later than the fold it scores.
- All models train on `log(posted_rate)` via `TransformedTargetRegressor`, carried over from Phase 4 where the log target won. Predictions are inverted to dollars and are strictly positive by construction.

| Split | rows | dates |
|:--|--:|:--|
| tuning (fit + CV) | 38,477 | 2025-01-01 .. 2025-08-31 |
| holdout (scored once) | 9,523 | 2025-09-01 .. 2025-10-31 |

## Results

| model                         | family   |     MAE |    RMSE |      R2 |   MAPE_% |
|:------------------------------|:---------|--------:|--------:|--------:|---------:|
| CatBoost                      | advanced |  132.15 |  641.48 |  0.8233 |     5.68 |
| LightGBM                      | advanced |  135.78 |  642.24 |  0.8229 |     5.88 |
| HistGradientBoosting          | advanced |  136.21 |  640.69 |  0.8237 |     5.93 |
| RandomForest                  | advanced |  144.85 |  646.24 |  0.8207 |     6.27 |
| Ridge (alpha=1.0, log target) | baseline |  145.24 |  644.04 |  0.8219 |     6.27 |
| Ridge (alpha=1.0)             | baseline |  149.36 |  641.36 |  0.8234 |     8.65 |
| LinearRegression              | baseline |  149.37 |  641.36 |  0.8234 |     8.65 |
| XGBoost                       | advanced |  157.6  |  654.23 |  0.8162 |     6.84 |
| Rate-per-mile (by equipment)  | baseline |  229.1  |  670.66 |  0.8069 |    10.49 |
| Rate-per-mile (global)        | baseline |  256.95 |  684.25 |  0.799  |    11.63 |
| Median (constant)             | baseline | 1148.92 | 1569.42 | -0.0576 |    70.15 |
| Mean (constant)               | baseline | 1178.8  | 1526.23 | -0.0002 |    83.43 |

Ranked by holdout MAE, the selection criterion carried over from Phase 4: the business cost of a mispriced load is roughly linear in dollars.

## Selected hyperparameters

- **RandomForest** (CV MAE 167.68, 6 candidates, 328s): `{'n_estimators': 200, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}`
- **HistGradientBoosting** (CV MAE 165.92, 8 candidates, 797s): `{'min_samples_leaf': 40, 'max_leaf_nodes': 31, 'max_iter': 800, 'learning_rate': 0.03, 'l2_regularization': 1.0}`
- **XGBoost** (CV MAE 175.15, 8 candidates, 504s): `{'subsample': 1.0, 'reg_lambda': 5.0, 'n_estimators': 1000, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.6}`
- **LightGBM** (CV MAE 158.88, 8 candidates, 177s): `{'subsample': 0.8, 'reg_lambda': 0.0, 'num_leaves': 31, 'n_estimators': 400, 'min_child_samples': 20, 'learning_rate': 0.05, 'colsample_bytree': 0.6}`
- **CatBoost** (CV MAE 124.13, 6 candidates, 186s): `{'learning_rate': 0.03, 'l2_leaf_reg': 1.0, 'iterations': 400, 'depth': 6}`

## Selected model

**CatBoost**

| Metric | Holdout value |
|:--|--:|
| MAE | $132.15 |
| RMSE | $641.48 |
| R2 | 0.8233 |
| MAPE | 5.68% |
| Holdout rows | 9,523 |

Hyperparameters: `{'learning_rate': 0.03, 'l2_leaf_reg': 1.0, 'iterations': 400, 'depth': 6}`

Persisted to `models/best_model.joblib` as a complete pipeline (preprocessing + log-target model), so Phase 7 can load it and predict directly from raw feature frames.
