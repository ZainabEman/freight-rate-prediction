# Exploratory Data Analysis (Phase 2) - data/train_test.csv
- Shape: **48,000 rows x 14 columns**
- Target column: **posted_rate**
- Target missing values: **0**
## Target variable analysis
|       |    value |
|:------|---------:|
| count | 48000    |
| mean  |  2373.98 |
| std   |  1486.49 |
| min   |    57.22 |
| 25%   |  1251.55 |
| 50%   |  2030.76 |
| 75%   |  3330.75 |
| max   | 25533    |
## Numerical feature analysis
|              |   count |        mean |         std |          min |         25% |         50% |         75% |         max |
|:-------------|--------:|------------:|------------:|-------------:|------------:|------------:|------------:|------------:|
| pickup_lat   |   48000 |    35.6475  |    4.31528  |     28.3576  |    31.9869  |    35.2948  |    39.411   |    44.303   |
| pickup_lon   |   48000 |   -90.929   |   13.4824   |   -121.698   |   -98.4006  |   -88.0892  |   -83.2851  |   -69.5     |
| delivery_lat |   48000 |    35.6412  |    4.3172   |     28.3576  |    31.9869  |    35.2948  |    39.411   |    44.303   |
| delivery_lon |   48000 |   -90.8573  |   13.4766   |   -121.698   |   -98.4006  |   -87.5287  |   -83.2851  |   -69.5     |
| distance     |   48000 |  1135.86    |  728.564    |     70       |   550.4     |   953.3     |  1645.53    |  3439.8     |
| weight       |   47700 | 31028.8     | 9391.44     | -47500       | 25800       | 31436.5     | 37018       | 47500       |
| market_index |   47626 |     1.08339 |    0.168091 |      0.67639 |     0.94967 |     1.0558  |     1.21959 |     1.46778 |
| quote_signal |   48000 |     2.06247 |    0.291391 |      0.69228 |     1.89103 |     2.05575 |     2.22168 |     3.61035 |
| posted_rate  |   48000 |  2373.98    | 1486.49     |     57.22    |  1251.55    |  2030.76    |  3330.75    | 25533       |
## Categorical feature analysis
### pickup

|               |   count |
|:--------------|--------:|
| Oklahoma City |    1242 |
| Lexington     |    1209 |
| Bakersfield   |    1193 |
| Fort Wayne    |    1170 |
| Hartford      |    1150 |
| Richmond      |    1140 |
| Nashville     |    1124 |
| Phoenix       |    1121 |
| Baton Rouge   |    1115 |
| Mobile        |    1094 |


### delivery

|               |   count |
|:--------------|--------:|
| Lexington     |    1197 |
| Fort Wayne    |    1176 |
| Baton Rouge   |    1167 |
| Bakersfield   |    1156 |
| Hartford      |    1143 |
| Oklahoma City |    1140 |
| Richmond      |    1109 |
| Atlanta       |    1096 |
| Phoenix       |    1090 |
| Mobile        |    1089 |


### equipment

|         |   count |
|:--------|--------:|
| Dry Van |   27202 |
| Reefer  |   12045 |
| Flatbed |    8753 |


## Missing value analysis
|              |   missing_count |   missing_pct |
|:-------------|----------------:|--------------:|
| market_index |             374 |        0.7792 |
| weight       |             300 |        0.625  |
## Outlier analysis (IQR rule)
| feature      |   outlier_count |   outlier_pct |          lo |          hi |
|:-------------|----------------:|--------------:|------------:|------------:|
| quote_signal |            1956 |     4.075     |     1.39505 |     2.71767 |
| delivery_lon |             467 |     0.972917  |  -121.074   |   -60.6118  |
| pickup_lon   |             453 |     0.94375   |  -121.074   |   -60.6118  |
| weight       |             424 |     0.888889  |  8973       | 53845       |
| posted_rate  |             260 |     0.541667  | -1867.24    |  6449.54    |
| distance     |              32 |     0.0666667 | -1092.29    |  3288.21    |
| pickup_lat   |               0 |     0         |    20.8507  |    50.5472  |
| delivery_lat |               0 |     0         |    20.8507  |    50.5472  |
| market_index |               0 |     0         |     0.54479 |     1.62447 |
## Correlation analysis
Top correlations with target (by value):
| feature      |       corr |
|:-------------|-----------:|
| posted_rate  |  1         |
| distance     |  0.908519  |
| weight       |  0.0348399 |
| market_index |  0.0341651 |
| quote_signal | -0.039858  |
| pickup_lat   | -0.0908734 |
| delivery_lat | -0.0919699 |
| pickup_lon   | -0.255058  |
| delivery_lon | -0.257086  |
- Full correlation heatmap: `figures/eda/correlation_matrix.png`
## Geographic analysis
- Pickup geo scatter: `figures/eda/geo_scatter_pickup.png` (if available)
- Delivery geo scatter: `figures/eda/geo_scatter_delivery.png` (if available)
## Temporal analysis
- date_parse_missing: **0**
- date_min: **2025-01-01 00:00:00**
- date_max: **2025-10-31 00:00:00**
- Mean by date plot: `figures/eda/mean_posted_rate_by_date.png` (if parseable)
## Business relationship analysis
### Relationship: pickup -> posted_rate (top 10 by mean)

| pickup         |   count |    mean |   median |
|:---------------|--------:|--------:|---------:|
| San Francisco  |     453 | 4071.21 |  4301.81 |
| Fresno         |     827 | 3878.91 |  4160.28 |
| Reno           |     642 | 3765.39 |  3836.93 |
| Phoenix        |    1121 | 3764.77 |  3935.62 |
| Los Angeles    |     945 | 3729.64 |  3996.61 |
| Bakersfield    |    1193 | 3519.37 |  3804.88 |
| Tucson         |     944 | 3311    |  3535.99 |
| Las Vegas      |     277 | 3292.82 |  3633.06 |
| Salt Lake City |     394 | 3144.76 |  3316.26 |
| Providence     |     425 | 3134.07 |  2603.08 |


### Relationship: delivery -> posted_rate (top 10 by mean)

| delivery       |   count |    mean |   median |
|:---------------|--------:|--------:|---------:|
| San Francisco  |     467 | 4064.47 |  4356.6  |
| Fresno         |     763 | 3865.38 |  4140.17 |
| Phoenix        |    1090 | 3751.47 |  3921.98 |
| Los Angeles    |     955 | 3742.25 |  4046.09 |
| Reno           |     653 | 3679.87 |  3783.88 |
| Bakersfield    |    1156 | 3532.2  |  3742.53 |
| Las Vegas      |     292 | 3428.38 |  3707.76 |
| Tucson         |     997 | 3363.19 |  3591.79 |
| Salt Lake City |     406 | 3164.35 |  3316.06 |
| El Paso        |     769 | 3008.98 |  3089.55 |


### Relationship: equipment -> posted_rate (top 10 by mean)

| equipment   |   count |    mean |   median |
|:------------|--------:|--------:|---------:|
| Reefer      |   12045 | 2553.64 |  2196.67 |
| Flatbed     |    8753 | 2445.09 |  2076.81 |
| Dry Van     |   27202 | 2271.55 |  1953.04 |


## Modeling implications (descriptive only)
- Use missing-value imputation for features with missingness.
- Use robust scaling for features with outliers (if supported by EDA).
- Consider time-derived features if temporal plots show structure.
- Consider interaction features if correlations/relationships indicate non-linear effects.
## Feature engineering opportunities (evidence to be validated in Phase 3 implementation)
- Date feature extraction: year/month/day/weekday/weekend (validated after observing temporal patterns).
- Geographic distance feature(s): e.g., haversine distance from pickup to delivery (validated using geo scatter and distance relationships).
- Weight-per-distance interaction (business rationale: shipping intensity relative to distance).
- Market/quote interactions (validated using correlation analysis with posted_rate).
- Equipment category impacts on posted_rate (validated via group-by relationship).
