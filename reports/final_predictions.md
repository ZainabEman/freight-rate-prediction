# Final Predictions (Phase 7)

Regenerate with `python -m src.run_final_predictions_phase7`. Exactly one training run was performed.

## Final model

| Item | Value |
|:--|:--|
| Algorithm | CatBoost (selected in Phase 5) |
| Target | `log(posted_rate)`, inverted with `exp` |
| Hyperparameters | `{'learning_rate': 0.03, 'l2_leaf_reg': 1.0, 'iterations': 400, 'depth': 6}` |
| Training data | 48,000 loads, 2025-01-01 to 2025-10-31 |
| Preprocessing | Phase-3 pipeline, 156 engineered features |
| Back-transform | Duan smearing, factor 1.0128 |

## Back-transformation decision

Phase 6 measured a systematic under-pricing bias of +$101.81. Duan's smearing estimator was evaluated on the untouched Sep-Oct holdout using the Phase-5 model (fitted on data through Aug 31 only), so the decision itself does not leak.

| Metric | Uncorrected | Smearing-corrected |
|:--|--:|--:|
| MAE | $132.15 | **$114.99** |
| RMSE | $641.48 | $636.25 |
| R2 | 0.8233 | 0.8262 |
| MAPE | 5.68% | 5.03% |

Adopted. Smearing (factor 1.0124) reduces holdout MAE from $132.15 to $114.99 and cuts mean bias from $101.81 to $73.37.

The factor applied to the final model is recomputed from that model's own training residuals (1.0128), not carried over from the holdout experiment. Because the factor is strictly positive, the correction cannot violate the positivity constraint `score.py` enforces.

## Final holdout metrics

Measured on Sep-Oct 2025 with the corrected Phase-5 model. These are the honest out-of-sample figures; the submitted model is refitted on all of Jan-Oct and so should perform at least as well.

| Metric | Value |
|:--|--:|
| MAE | $114.99 |
| RMSE | $636.25 |
| R2 | 0.8262 |
| MAPE | 5.03% |

## Validation prediction statistics

`validation_predictions.csv`, 12,000 loads:

| Statistic | Value |
|:--|--:|
| count | 12,000 |
| min | $201.65 |
| p05 | $595.15 |
| median | $1,975.49 |
| mean | $2,280.56 |
| p95 | $4,726.15 |
| max | $6,465.66 |
| std | $1,308.72 |

## Submission validation

- [x] Schema is exactly ['load_id', 'predicted_rate'].
- [x] Row count is exactly 12,000.
- [x] load_id values are unique and complete.
- [x] load_id ordering matches the provided template exactly.
- [x] All predicted_rate values are finite.
- [x] All predicted_rate values are strictly positive.
- [x] No missing values.

## December chart

Fixed scenario: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date changes across the 31 rows.

| Statistic | Value |
|:--|--:|
| Rows | 31 |
| Min | $785.70 |
| Max | $799.22 |
| Mean | $792.12 |
| Spread | $13.52 (1.71% of mean) |

**The curve is deliberately near-flat, and this is the correct result rather than a defect.** Phase 6 measured the elasticity of rate-per-mile to `market_index` at 0.139, and permutation importance ranks `market_index` 15th out-of-sample. Across the 31 December dates the market index spans 0.831 to 1.045 (+25.8%); the data-implied rate response is +3.2% and the model produces a comparable figure. On a lane with fixed mileage, equipment and weight, date is genuinely a minor price driver in this dataset.

## Scorer output

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```
