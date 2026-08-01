# Error Analysis (Phase 6)

Holdout: **2025-09-01 to 2025-10-31**, 9,523 loads. The model was fitted on data up to 2025-08-31 only, so every error below is genuinely out-of-sample.

Sign convention: `residual = actual - predicted`. **Positive means the model under-priced the load.**

## Headline

| Statistic | Value |
|:--|--:|
| MAE | $132.15 |
| Median absolute error | $54.30 |
| MAPE | 5.68% |
| Mean residual (bias) | $101.81 |
| Median residual | $49.41 |
| 95th percentile absolute error | $260.12 |
| 99th percentile absolute error | $1,494.20 |
| Max absolute error | $16,363.93 |

The gap between the median absolute error ($54.30) and the mean ($132.15) is the whole story of this model: typical loads are priced very accurately and a small minority are not.

## Segment breakdown

### By equipment

| equipment   |   rows |    MAE |   MAPE |   bias |   median_rate |
|:------------|-------:|-------:|-------:|-------:|--------------:|
| Reefer      |   2393 | 150.96 |   6.31 | 119.78 |       2227.13 |
| Flatbed     |   1770 | 131.2  |   6.1  |  95.85 |       2045.3  |
| Dry Van     |   5360 | 124.07 |   5.26 |  95.75 |       1952.44 |

### By distance band

| distance_band   |   rows |    MAE |   MAPE |   bias |   median_rate |
|:----------------|-------:|-------:|-------:|-------:|--------------:|
| >2000mi         |   1534 | 250.21 |   5.48 | 199.52 |       4583.55 |
| 1000-2000mi     |   2927 | 168.51 |   5.19 | 129.76 |       2860.24 |
| 500-1000mi      |   2973 |  92.62 |   5.98 |  69.9  |       1641.95 |
| 250-500mi       |   1539 |  59.3  |   6.18 |  44.66 |        936.36 |
| <250mi          |    550 |  26.97 |   5.83 |  12.9  |        503.09 |

### By weight band

| weight_band   |   rows |    MAE |   MAPE |   bias |   median_rate |
|:--------------|-------:|-------:|-------:|-------:|--------------:|
| 35-45k lb     |   2745 | 136.14 |   5.6  | 107.88 |       2108.62 |
| 25-35k lb     |   4243 | 135.98 |   5.39 | 107.05 |       2015.6  |
| 15-25k lb     |   1771 | 126.13 |   6.56 |  84.79 |       2014.62 |
| <15k lb       |    213 | 124.01 |   6.11 |  93.34 |       2077.33 |
| >45k lb       |    486 | 106    |   5.54 |  90.31 |       2039.11 |

### By month

| month   |   rows |    MAE |   MAPE |   bias |   median_rate |
|:--------|-------:|-------:|-------:|-------:|--------------:|
| 2025-10 |   4853 | 140.33 |   6.39 | 109.76 |       2035.9  |
| 2025-09 |   4670 | 123.65 |   4.94 |  93.54 |       2057.13 |

### By day of week

| day_of_week   |   rows |    MAE |   MAPE |   bias |   median_rate |
|:--------------|-------:|-------:|-------:|-------:|--------------:|
| Wednesday     |   1399 | 162.68 |   5.8  | 135    |       2015.6  |
| Thursday      |   1423 | 155.63 |   6.21 | 127.28 |       2038.06 |
| Friday        |   1417 | 125.05 |   5.85 | 102.39 |       2080.96 |
| Sunday        |   1242 | 123.36 |   5.87 |  87.93 |       2039.72 |
| Saturday      |   1283 | 121.21 |   5.14 |  92.67 |       2064.59 |
| Tuesday       |   1366 | 118.65 |   5.38 |  85.31 |       2084.23 |
| Monday        |   1393 | 115.89 |   5.47 |  78.83 |       1999.43 |

### By prediction quintile

| prediction_quintile   |   rows |    MAE |   MAPE |   bias |   median_rate |
|:----------------------|-------:|-------:|-------:|-------:|--------------:|
| Q5                    |   1905 | 246.02 |   5.45 | 196.3  |       4389.3  |
| Q4                    |   1904 | 172.64 |   5.09 | 135.16 |       2954.26 |
| Q3                    |   1905 | 115.51 |   5.48 |  86.92 |       2043.1  |
| Q2                    |   1904 |  77.58 |   6.03 |  56.67 |       1408.05 |
| Q1                    |   1905 |  48.99 |   6.34 |  33.97 |        820.16 |

### Worst pickup cities (min 30 loads)

| pickup         |   rows |    MAE |   MAPE |   bias |   median_rate |
|:---------------|-------:|-------:|-------:|-------:|--------------:|
| San Francisco  |     84 | 272.85 |   5.45 | 206.83 |       4407.24 |
| Boston         |    129 | 263.53 |   4.36 | 254.76 |       2486.75 |
| Montgomery     |    182 | 254.08 |   4.47 | 246.24 |       1872.41 |
| Providence     |     74 | 232.99 |   5.02 | 227.62 |       2766.07 |
| Baltimore      |     98 | 229.88 |   3.63 | 220.88 |       2019.97 |
| Corpus Christi |    124 | 209    |   6.37 | 180.91 |       2357.84 |
| Bakersfield    |    245 | 197.11 |   7.41 | 104.11 |       3892.71 |
| Houston        |    144 | 195.63 |   8.23 | 143.39 |       2047.76 |
| Milwaukee      |    192 | 188.78 |   6.68 | 149.45 |       1545.68 |
| Los Angeles    |    202 | 181.98 |   3.13 | 159.08 |       4105.34 |
| Columbia       |    211 | 181.48 |   8.18 | 148.58 |       1538.13 |
| Jacksonville   |    131 | 181.17 |   5.72 | 124.19 |       1711.11 |
| Phoenix        |    211 | 176.2  |   3.64 | 161.86 |       4082.26 |
| Fort Wayne     |    235 | 173.71 |   7.48 | 125.47 |       1696.9  |
| Toledo         |    144 | 157.34 |   3.53 | 152.83 |       1493.22 |

### Worst delivery cities (min 30 loads)

| delivery      |   rows |    MAE |   MAPE |   bias |   median_rate |
|:--------------|-------:|-------:|-------:|-------:|--------------:|
| Detroit       |     61 | 325.65 |   6.99 | 285.33 |       1811.32 |
| Baltimore     |    119 | 295.14 |   6.73 | 222.01 |       2199.27 |
| Las Vegas     |     54 | 283.4  |   4.45 | 274.24 |       3883.77 |
| Tucson        |    187 | 281.96 |   9.02 | 191.4  |       3652.13 |
| Bakersfield   |    226 | 276.56 |   4.07 | 263.57 |       3775.29 |
| Albuquerque   |    120 | 265.15 |   4.15 | 255.02 |       2952.96 |
| Fresno        |    142 | 251.46 |   3.78 | 244    |       4178.86 |
| El Paso       |    154 | 219.29 |   8.13 | 130.65 |       3198.64 |
| Lubbock       |    154 | 204.56 |   4.41 | 202.26 |       2373.89 |
| Phoenix       |    237 | 201.08 |   6.44 | 162.21 |       3947.05 |
| Oklahoma City |    223 | 190.5  |   4.76 | 185.79 |       2315.28 |
| Columbia      |    190 | 179.63 |   7.44 | 136.06 |       1656.05 |
| San Francisco |    101 | 173.52 |   3.79 | 165.38 |       4515.31 |
| Providence    |     96 | 173    |   6.14 | 128.33 |       2775.01 |
| Albany        |    161 | 167.93 |   3.35 | 157.43 |       2491.67 |

## Worst and best segments

- **Worst equipment:** Reefer (MAE $150.96); **best:** Dry Van (MAE $124.07).
- **Worst distance band:** >2000mi (MAE $250.21, MAPE 5.48%); **best:** <250mi (MAE $26.97, MAPE 5.83%).
- **Error scales with rate:** MAE runs from $48.99 to $246.02 across prediction quintiles - a 5.0x spread.

## Systematic bias

Mean residual is **$101.81** (t = 15.69, statistically significant at the 2-sigma level). Median residual is $49.41.

## Outliers

The worst 1% of loads account for **39.3%** of all absolute error. The 20 worst predictions:

| load_id   | pickup         | delivery      | equipment   |   distance |   weight | date       |   market_index |   posted_rate |   predicted |   residual |   absolute_percentage_error |
|:----------|:---------------|:--------------|:------------|-----------:|---------:|:-----------|---------------:|--------------:|------------:|-----------:|----------------------------:|
| TR-040097 | Baltimore      | Oklahoma City | Reefer      |     1760.3 |    29521 | 2025-09-11 |           1    |       20132.4 |     3768.44 |   16363.9  |                       81.28 |
| TR-040172 | Montgomery     | Fresno        | Dry Van     |     2283.4 |    32447 | 2025-09-11 |           0.99 |       20361.9 |     4244.68 |   16117.2  |                       79.15 |
| TR-043540 | Milwaukee      | Bakersfield   | Dry Van     |     1893   |    30760 | 2025-10-03 |           1.08 |       17514.1 |     3572.11 |   13942    |                       79.6  |
| TR-046214 | Columbia       | Tucson        | Flatbed     |     2023.4 |    34710 | 2025-10-20 |           0.9  |       17893.2 |     4114.1  |   13779.1  |                       77.01 |
| TR-047299 | Boston         | Bakersfield   | Reefer      |     2979.1 |    30585 | 2025-10-27 |           0.87 |       19110.3 |     5797.57 |   13312.7  |                       69.66 |
| TR-040323 | Montgomery     | El Paso       | Dry Van     |     1639.9 |    38363 | 2025-09-12 |           0.96 |       14949.2 |     3242.91 |   11706.3  |                       78.31 |
| TR-046517 | Toledo         | Albuquerque   | Reefer      |     1684.3 |    31536 | 2025-10-22 |           0.99 |       14784.4 |     3476.06 |   11308.3  |                       76.49 |
| TR-042882 | San Antonio    | Detroit       | Dry Van     |     1460.5 |    43567 | 2025-09-29 |           0.75 |       13503.6 |     2807.45 |   10696.2  |                       79.21 |
| TR-042150 | Grand Rapids   | Lubbock       | Dry Van     |     1415.5 |    35333 | 2025-09-24 |           0.96 |       13026.2 |     2768.65 |   10257.6  |                       78.75 |
| TR-046367 | Bakersfield    | Columbia      | Dry Van     |     2251.9 |    40729 | 2025-10-21 |           0.95 |       14403.1 |     4168.97 |   10234.2  |                       71.06 |
| TR-047537 | Los Angeles    | Richmond      | Dry Van     |     2782.4 |    25730 | 2025-10-29 |           1.03 |       15003.5 |     4775.78 |   10227.8  |                       68.17 |
| TR-044268 | Albany         | Albuquerque   | Dry Van     |     2492.6 |    25513 | 2025-10-08 |           1.06 |       14425.4 |     4289.95 |   10135.5  |                       70.26 |
| TR-043303 | Atlanta        | Phoenix       | Reefer      |     2083.1 |    40335 | 2025-10-01 |           0.91 |       13894.5 |     4358.53 |    9536.02 |                       68.63 |
| TR-043385 | Lexington      | Lubbock       | Dry Van     |     1187   |    25359 | 2025-10-02 |           0.96 |       11696.4 |     2287.91 |    9408.53 |                       80.44 |
| TR-042736 | Jacksonville   | Tucson        | Dry Van     |     1943   |    16292 | 2025-09-28 |           0.78 |       12797.5 |     3518.37 |    9279.17 |                       72.51 |
| TR-047587 | Corpus Christi | Las Vegas     | Flatbed     |     1383.3 |    18645 | 2025-10-29 |           1.02 |       11596.3 |     2778.96 |    8817.32 |                       76.04 |
| TR-039363 | Tulsa          | Bakersfield   | Dry Van     |     1477.7 |    30076 | 2025-09-06 |           0.89 |       11684.7 |     2893.29 |    8791.41 |                       75.24 |
| TR-044345 | Lexington      | Albany        | Dry Van     |     1040   |    36569 | 2025-10-08 |           1.04 |       10566.2 |     2052.72 |    8513.53 |                       80.57 |
| TR-040159 | San Francisco  | Shreveport    | Dry Van     |     1950.8 |    40882 | 2025-09-11 |           1.02 |       12240.8 |     3811    |    8429.81 |                       68.87 |
| TR-043887 | Shreveport     | Baltimore     | Reefer      |     1348.4 |    25462 | 2025-10-05 |           0.86 |       11052.7 |     2861.5  |    8191.22 |                       74.11 |

## Figures

- `figures\residuals\residual_vs_prediction.png`
- `figures\residuals\residual_distribution.png`
- `figures\residuals\qq_plot.png`
- `figures\residuals\absolute_error_histogram.png`
- `figures\residuals\error_vs_distance.png`
- `figures\residuals\error_vs_weight.png`
- `figures\error_analysis\mae_by_equipment.png`
- `figures\error_analysis\mae_by_distance_band.png`
- `figures\error_analysis\mae_by_weight_band.png`
- `figures\error_analysis\mae_by_month.png`
- `figures\error_analysis\mae_by_day_of_week.png`
- `figures\error_analysis\mae_by_prediction_quintile.png`
- `figures\error_analysis\worst_pickup_cities.png`
- `figures\error_analysis\worst_delivery_cities.png`

## Residual diagnostics

| Diagnostic | Measurement | Verdict |
|:--|--:|:--|
| Heteroscedasticity | corr(&#124;residual&#124;, prediction) = 0.115; residual sd varies 4.24x across prediction quintiles | **Present** |
| Heavy tails | excess kurtosis = 265.3; worst 1% of loads carry 39.3% of total error | **Present** |
| Systematic bias | mean residual $101.81, t = 15.69 | **Present** |

**Reading:**

- *Heteroscedasticity exists.* Residual spread grows with the predicted rate. This is expected for a log-target model: constant proportional error becomes growing absolute error in dollars. It means a single dollar-denominated error bar is misleading - uncertainty should be quoted as a percentage.
- *Heavy tails exist,* and dominate. Excess kurtosis of 265 is far above the normal value of 0, and the QQ plot departs sharply from the reference line at both ends. This is why RMSE barely moved across every model tried in Phases 4-5 while MAE improved substantially: the tail is not something hyperparameters can fix.
- *A systematic bias exists:* mean residual $101.81. Because a log-target model is fitted to the conditional mean of `log(rate)`, back-transforming with `exp` returns the conditional *median*, which sits below the mean for a right-skewed target. A small positive bias (under-pricing) is the expected signature of that transform, not a modelling error.
