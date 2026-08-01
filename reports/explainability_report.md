# Explainability Report (Phase 6)

Model: **CatBoostRegressor** (156 engineered features), loaded from `models/best_model.joblib`. Nothing was retrained or retuned.

> **Reading the numbers.** The model predicts `log(posted_rate)`, so all importances and SHAP values are in log-dollar space. Contributions are additive in logs, which means *multiplicative* in dollars: a SHAP value of `+0.10` is about a **+10.5%** effect on the rate, not `+$0.10`.

All values are computed on the temporal holdout (2025-09-01 to 2025-10-31, out-of-sample).

## Three views of feature importance

Three independent methods are reported because each has a different blind spot: CatBoost's native score is computed on the training data and can over-credit high-cardinality splits; permutation importance is measured out-of-sample but under-credits correlated features (shuffling `distance` matters less when `log_distance` and `haversine_miles` remain); SHAP is out-of-sample and additive but splits credit between correlated features.

### CatBoost native importance (top 15)

| feature           |   native_importance |
|:------------------|--------------------:|
| log_distance      |               30.75 |
| distance          |               30.65 |
| haversine_miles   |               30.4  |
| equipment_Dry Van |                2.06 |
| weight_per_mile   |                1.23 |
| weight            |                0.87 |
| market_index      |                0.68 |
| doy_cos           |                0.58 |
| equipment_Reefer  |                0.51 |
| lon_delta         |                0.4  |
| pickup_lat        |                0.27 |
| delivery_lat      |                0.26 |
| quote_signal      |                0.26 |
| delivery_lon      |                0.2  |
| pickup_lon        |                0.18 |

### Permutation importance (top 15)

Measured as the increase in log-space MAE when a column is shuffled, 5 repeats.

| feature           |   permutation_importance |   permutation_std |
|:------------------|-------------------------:|------------------:|
| log_distance      |                   0.233  |            0.0025 |
| haversine_miles   |                   0.2146 |            0.0023 |
| distance          |                   0.213  |            0.0024 |
| equipment_Dry Van |                   0.0224 |            0.0002 |
| weight            |                   0.0106 |            0.0002 |
| equipment_Reefer  |                   0.0043 |            0.0001 |
| weight_per_mile   |                   0.0031 |            0.0001 |
| pickup_lat        |                   0.002  |            0.0001 |
| lon_delta         |                   0.002  |            0.0001 |
| delivery_lat      |                   0.0018 |            0.0001 |
| delivery_lon      |                   0.0014 |            0.0001 |
| bearing_sin       |                   0.001  |            0      |
| pickup_lon        |                   0.0009 |            0      |
| quote_signal      |                   0.0006 |            0.0001 |
| market_index      |                   0.0006 |            0      |

### Mean |SHAP| (top 15)

Computed on 2,000 sampled holdout rows with exact TreeSHAP.

| feature           |   mean_abs_shap |
|:------------------|----------------:|
| log_distance      |          0.1787 |
| haversine_miles   |          0.172  |
| distance          |          0.1622 |
| equipment_Dry Van |          0.0362 |
| weight            |          0.0225 |
| market_index      |          0.0192 |
| equipment_Reefer  |          0.0131 |
| weight_per_mile   |          0.0126 |
| pickup_lat        |          0.0095 |
| delivery_lat      |          0.0089 |
| delivery_lon      |          0.0045 |
| lon_delta         |          0.0043 |
| pickup_lon        |          0.0033 |
| quote_signal      |          0.0032 |
| doy_cos           |          0.0025 |

### Where the three methods agree

Features in the top 10 of the native ranking: `log_distance`, `distance`, `haversine_miles`, `equipment_Dry Van`, `weight_per_mile`, `weight`, `market_index`, `doy_cos`, `equipment_Reefer`, `lon_delta`.

## Figures

- `figures\importance\catboost_native_importance.png`
- `figures\importance\permutation_importance.png`
- `figures\shap\shap_beeswarm.png`
- `figures\shap\shap_summary.png`
- `figures\shap\shap_bar.png`
- `figures\shap\shap_dependence_log_distance.png`
- `figures\shap\shap_dependence_haversine_miles.png`
- `figures\shap\shap_dependence_distance.png`
- `figures\shap\shap_dependence_weight.png`
- `figures\shap\shap_dependence_market_index.png`
- `figures\shap\shap_dependence_weight_per_mile.png`
- `figures\shap\shap_waterfall_typical.png`
- `figures\shap\shap_waterfall_worst_underprediction.png`
- `figures\shap\shap_waterfall_worst_overprediction.png`

## What the model actually learned

The dependence and waterfall plots show the mechanism behind the rankings above; see `reports/business_insights.md` for the business reading of these effects, measured directly from the data rather than inferred from the model.
