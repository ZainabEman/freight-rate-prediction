# Initial Data Audit Report (Phase 1)

## Concise summary
- Training shape: (48000, 14)
- Validation shape: (12000, 13)
- Number of features (train): 14
- Target variable (heuristic): posted_rate
 - Data quality issues found: train=2, validation=2, cross-schema=1

## Identified feature groups (heuristic, documented uncertainty)
- Target column: posted_rate
- Identifier columns: load_id

## Numerical statistical summary (train)
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
## Categorical statistical summary (train)
| feature   |   count_non_missing |   unique_values | top_value     |
|:----------|--------------------:|----------------:|:--------------|
| load_id   |               48000 |           48000 | TR-000001     |
| pickup    |               48000 |              64 | Oklahoma City |
| delivery  |               48000 |              64 | Lexington     |
| equipment |               48000 |               3 | Dry Van       |
| date      |               48000 |             304 | 2025-06-17    |
## Inspect every feature individually (train)
- **load_id** | dtype=object | missing=0.000% | purpose=identifier | observations=High cardinality: 48000 unique non-missing values.
- **pickup** | dtype=object | missing=0.000% | purpose=categorical | observations=none
- **delivery** | dtype=object | missing=0.000% | purpose=categorical | observations=none
- **pickup_lat** | dtype=float64 | missing=0.000% | purpose=numeric | observations=Numeric range: [28.3576, 44.303]
- **pickup_lon** | dtype=float64 | missing=0.000% | purpose=numeric | observations=Numeric range: [-121.698, -69.5]
- **delivery_lat** | dtype=float64 | missing=0.000% | purpose=numeric | observations=Numeric range: [28.3576, 44.303]
- **delivery_lon** | dtype=float64 | missing=0.000% | purpose=numeric | observations=Numeric range: [-121.698, -69.5]
- **distance** | dtype=float64 | missing=0.000% | purpose=numeric | observations=High cardinality: 21204 unique non-missing values.; Numeric range: [70, 3439.8]
- **equipment** | dtype=object | missing=0.000% | purpose=categorical | observations=none
- **weight** | dtype=float64 | missing=0.625% | purpose=numeric | observations=High cardinality: 23678 unique non-missing values.; Numeric range: [-47500, 47500]
- **date** | dtype=object | missing=0.000% | purpose=datetime | observations=none
- **market_index** | dtype=float64 | missing=0.779% | purpose=numeric | observations=High cardinality: 32884 unique non-missing values.; Numeric range: [0.67639, 1.46778]
- **quote_signal** | dtype=float64 | missing=0.000% | purpose=numeric | observations=High cardinality: 37633 unique non-missing values.; Numeric range: [0.69228, 3.61035]
- **posted_rate** | dtype=float64 | missing=0.000% | purpose=numeric | observations=High cardinality: 45398 unique non-missing values.; Numeric range: [57.22, 25533]
## Data quality issues
### Training issues
- **[warning]** quality | column=weight: Missing values: 300/48000 (0.625%).
- **[warning]** quality | column=market_index: Missing values: 374/48000 (0.779%).

### Validation issues
- **[warning]** quality | column=weight: Missing values: 165/12000 (1.375%).
- **[warning]** quality | column=market_index: Missing values: 249/12000 (2.075%).

### Cross-dataset schema issues
- **[error]** schema | column=(dataset): Validation is missing columns present in training: ['posted_rate']
## Potential risks & recommendations (initial)
- Risks identified:
  - Train/validation schema mismatch; Phase 2 must align preprocessing and encoding consistently.
  - Missingness present; Phase 2 must decide imputation strategy or missing-indicator usage.
- Recommendations before Phase 2 (EDA):
  - Validate numeric coercion policy and enforce consistent types across splits.
  - If categorical values show whitespace/case issues, normalize categories in Phase 2 (fit on training only).
  - If datetime columns exist, derive safe time features without leaking validation information.
  - If coordinate-like columns exist, verify ranges and confirm correct units before modeling.
