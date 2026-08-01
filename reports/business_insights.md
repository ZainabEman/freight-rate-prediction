# Business Insights (Phase 6)

Every figure below is measured from the 48,000 labelled loads in `data/train_test.csv` (2025-01-01 to 2025-10-31) or from out-of-sample model behaviour on the Sep-Oct holdout. Nothing is assumed.

## 1. What actually drives price

Ranked by mean |SHAP| on held-out loads, the top drivers are:

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

Measured correlations with `posted_rate` across the development set:

| Feature | Pearson r |
|:--|--:|
| `distance` | +0.909 |
| `market_index` | +0.034 |
| `quote_signal` | -0.040 |
| `weight` | +0.041 |
| `delivery_lon` | -0.257 |
| `distance_vs_rpm` | -0.335 |
| `daily_market_vs_rpm` | +0.577 |
| `market_elasticity` | +0.139 |
| `within_band_market_corr` | +0.055 |

**Distance is the price.** At r = +0.909 it explains the overwhelming majority of rate variation. Everything else in this report is a modifier on top of the mileage.

## 2. Distance effects

| distance_band   |   loads |   median_rate |   median_rate_per_mile |
|:----------------|--------:|--------------:|-----------------------:|
| <250mi          |    2712 |        506.05 |                   2.75 |
| 250-500mi       |    7575 |        924.05 |                   2.42 |
| 500-1000mi      |   15057 |       1632.13 |                   2.2  |
| 1000-2000mi     |   15108 |       2872.15 |                   2.05 |
| >2000mi         |    7548 |       4531.8  |                   1.91 |

Rate-per-mile falls steadily as hauls get longer (r = -0.335 between distance and $/mile). Short hauls carry fixed costs - loading, positioning, driver time - across few miles, so they price at a premium per mile. **Operational implication:** quoting a flat $/mile across the network systematically overprices long hauls and underprices short ones.

## 3. Equipment effects

| equipment   |   loads |   median_rate |   median_rate_per_mile |   mean_distance |
|:------------|--------:|--------------:|-----------------------:|----------------:|
| Reefer      |   12045 |       2196.67 |                   2.31 |         1135.07 |
| Flatbed     |    8753 |       2076.81 |                   2.22 |         1132.51 |
| Dry Van     |   27202 |       1953.04 |                   2.05 |         1137.28 |

Taking Dry Van as the base rate: **Reefer 113.0%** of Dry Van; **Flatbed 108.5%** of Dry Van. Reefer commands the largest premium, consistent with temperature-controlled capacity being scarcer and more costly to operate.

## 4. Market effects

| market_quintile   |   loads |   mean_market_index |   median_rate_per_mile |
|:------------------|--------:|--------------------:|-----------------------:|
| Q1                |    9526 |                0.86 |                   2.1  |
| Q2                |    9526 |                0.97 |                   2.1  |
| Q3                |    9525 |                1.06 |                   2.13 |
| Q4                |    9524 |                1.19 |                   2.17 |
| Q5                |    9525 |                1.33 |                   2.22 |

`market_index` correlates +0.034 with rate at the individual-load level. In aggregate the correlation looks far stronger - daily mean market index against daily mean rate-per-mile correlates +0.577 - **but that headline number is misleading, and it is worth being precise about why.**

A correlation between daily *averages* measures co-movement, not sensitivity. The measured elasticity is only **0.139**: a 1% rise in market index moves rate-per-mile by 0.14%. Within a narrow distance band (340-380 miles, the December chart lane) the load-level correlation collapses to +0.055.

The permutation importance in `reports/explainability_report.md` agrees: `market_index` ranks 7th on CatBoost's native score but only 15th out-of-sample, with a permutation effect two orders of magnitude below the distance features.

**Consequence for the December chart.** Across the 31 December dates the market index spans 0.831 to 1.045 (+25.8%). The data-implied rate response is +3.2%; the model produces +2.7%. The fixed-lane December curve will therefore be close to flat - roughly a $21 spread on a ~$777 rate. That is the honest signal in this dataset, not a modelling failure: on a fixed lane with fixed equipment and weight, **date genuinely is a minor price driver here.**

## 5. Seasonal observations

| month   |   loads |   mean_rate |   mean_rate_per_mile |   mean_market_index |
|:--------|--------:|------------:|---------------------:|--------------------:|
| 2025-01 |    4918 |     2255.97 |                 2.1  |                0.93 |
| 2025-02 |    4337 |     2273.8  |                 2.12 |                1    |
| 2025-03 |    5036 |     2372.27 |                 2.21 |                1.07 |
| 2025-04 |    4819 |     2372.16 |                 2.21 |                1.21 |
| 2025-05 |    4913 |     2421.78 |                 2.26 |                1.3  |
| 2025-06 |    4783 |     2497.03 |                 2.33 |                1.28 |
| 2025-07 |    4912 |     2415.16 |                 2.26 |                1.2  |
| 2025-08 |    4759 |     2338.41 |                 2.19 |                0.98 |
| 2025-09 |    4670 |     2406.37 |                 2.23 |                0.89 |
| 2025-10 |    4853 |     2379.05 |                 2.24 |                0.96 |

Mean rate-per-mile moves between $2.100 and $2.332 across the ten observed months, a 11.1% swing. Note the scoring window (November-December) is **not** represented in the development data, so any seasonal peak in those months cannot be learned directly - the model reaches them through `market_index` and cyclical date encodings instead.

### Day of week

| day_of_week   |   loads |   mean_rate_per_mile |
|:--------------|--------:|---------------------:|
| Monday        |    6769 |                 2.2  |
| Tuesday       |    6737 |                 2.21 |
| Wednesday     |    7007 |                 2.25 |
| Thursday      |    6934 |                 2.24 |
| Friday        |    6975 |                 2.21 |
| Saturday      |    6770 |                 2.2  |
| Sunday        |    6808 |                 2.19 |

## 6. Lane observations

Highest rate-per-mile lanes with at least 30 loads:

| pickup        | delivery    |   loads |   mean_distance |   median_rate_per_mile |
|:--------------|:------------|--------:|----------------:|-----------------------:|
| Nashville     | Cincinnati  |      33 |          187.66 |                   2.89 |
| Atlanta       | Nashville   |      33 |          143.41 |                   2.83 |
| Lexington     | Cincinnati  |      37 |          133.33 |                   2.82 |
| Atlanta       | Cincinnati  |      30 |          206    |                   2.77 |
| Cincinnati    | Lexington   |      31 |          133.63 |                   2.75 |
| Lexington     | Atlanta     |      39 |          208.94 |                   2.69 |
| Shreveport    | Nashville   |      31 |          376.11 |                   2.59 |
| Nashville     | Lexington   |      32 |          258.37 |                   2.55 |
| Lexington     | Nashville   |      30 |          261.66 |                   2.54 |
| Nashville     | Mobile      |      32 |          299.61 |                   2.54 |
| Tucson        | Bakersfield |      33 |          322.18 |                   2.51 |
| Oklahoma City | Baton Rouge |      33 |          437.82 |                   2.48 |
| Fort Wayne    | Lexington   |      35 |          363.72 |                   2.43 |
| Columbia      | Montgomery  |      35 |          341.92 |                   2.42 |
| Baton Rouge   | Memphis     |      30 |          476.11 |                   2.41 |

Directional pricing is real: `delivery_lon` correlates -0.257 with rate, meaning westbound deliveries price higher than eastbound ones at comparable distance. This is why the model carries `bearing_sin`/`bearing_cos` and `lon_delta` rather than distance alone.

## 7. Where the model is weakest

| prediction_quintile   |   rows |    MAE |   MAPE |   bias |   median_rate |
|:----------------------|-------:|-------:|-------:|-------:|--------------:|
| Q5                    |   1905 | 246.02 |   5.45 | 196.3  |       4389.3  |
| Q4                    |   1904 | 172.64 |   5.09 | 135.16 |       2954.26 |
| Q3                    |   1905 | 115.51 |   5.48 |  86.92 |       2043.1  |
| Q2                    |   1904 |  77.58 |   6.03 |  56.67 |       1408.05 |
| Q1                    |   1905 |  48.99 |   6.34 |  33.97 |        820.16 |

Absolute error grows with rate while percentage error stays comparatively flat, which is the practical meaning of the heteroscedasticity documented in `reports/error_analysis.md`.

## 8. Practical recommendations

1. **Quote uncertainty as a percentage, not a dollar band.** Error scales with rate, so a flat +/- $X interval is far too wide on cheap loads and far too narrow on expensive ones.
2. **Do not use a flat $/mile.** Rate-per-mile is strongly distance-dependent; use the banded figures in section 2 as a sanity check on any manual quote.
3. **Refresh `market_index` daily.** It is the dominant time-varying driver. A stale market index degrades every prediction on that day simultaneously.
4. **Route the worst segments to human review.** The tail is concentrated: the worst 1% of loads carry a disproportionate share of total error, so a small manual-review queue captures most of the residual risk.
5. **Treat November-December predictions as extrapolation.** No labelled data exists for those months. Monitor realised rates against predictions weekly and be prepared to recalibrate.
6. **Collect data for the 8 unseen cities.** Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk and San Diego appear only at scoring time; they are currently priced from geography alone.
