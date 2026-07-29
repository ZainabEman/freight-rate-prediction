# Preprocessing Report (Phase 3)

Regenerate with `python -m src.run_preprocessing_phase3`. Every figure below is computed at run time from the raw CSVs.

## Dataset shapes

| Dataset | Raw | Processed |
|:--|--:|--:|
| train | (48000, 14) | (48000, 156) |
| validation | (12000, 13) | (12000, 156) |
| december (reduced-feature path) | (31, 6) | (31, 156) |

## Pipeline structure

```
clean     RawDataCleaner       stateless, row-local repairs
features  FeatureBuilder       stateless feature construction
encode    ColumnTransformer    the ONLY stage that learns from data
```

Confining all fitted state to the final stage is what makes the leakage audit tractable: the two stateless stages cannot transfer information between splits because they depend only on the row being transformed.

## Cleaning decisions (evidence-based)

### 1. `weight` sign repair

- **Observed:** 292 training rows and 145 validation rows carry a negative `weight`.
- **Evidence:** `abs()` of the negative population is distributionally identical to the positive population (mean 31,724 vs 31,415; median 31,822 vs 31,494; identical `[5000, 47500]` support). Mean rate-per-mile is also equivalent (2.267 vs 2.215).
- **Decision:** treat as a sign-flip data-entry fault and apply `abs()`, then clip to `[5,000, 47,500]`.
- **Rejected alternative:** dropping the rows (discards 292 otherwise-valid loads) or nulling them (converts a recoverable value into an imputed guess).

### 2. Missing values

| Column | Train missing | Validation missing |
|:--|--:|--:|
| weight | 300 (0.62%) | 165 (1.38%) |
| market_index | 374 (0.78%) | 249 (2.08%) |

- **Decision:** `median` imputation, fitted on training rows only. The median is used because both distributions are skewed and it is unaffected by the extreme tails.
- **Missing indicators added for `weight` and `market_index` only.** Missingness is roughly twice as prevalent in the scoring window as in training, so the *fact* of missingness is itself a signal about distribution shift; discarding it would hide that from the model. No indicator is added for columns with zero missingness, which would be a constant column.

### 3. Categorical whitespace

- Leading/trailing whitespace is stripped. Case is deliberately **not** folded: the Phase-1 audit found no mixed-case duplicates, so case folding would be an unjustified assumption.

## Feature engineering decisions (evidence-based)

- **Temporal (cyclical)** (5): `doy_sin`, `doy_cos`, `dow_sin`, `dow_cos`, `is_weekend`
- **Geospatial** (5): `haversine_miles`, `bearing_sin`, `bearing_cos`, `lon_delta`, `lat_delta`
- **Interactions** (2): `log_distance`, `weight_per_mile`
- **Missingness indicators** (2): `weight_is_missing`, `market_index_is_missing`
- **Unknown-category indicators** (3): `pickup_is_unknown`, `delivery_is_unknown`, `equipment_is_unknown`
- **Raw numeric (scaled)** (8): `weight`, `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`, `distance`, `market_index`, `quote_signal`
- **Categorical (one-hot)** (131): `pickup_Albany`, `pickup_Albuquerque`, `pickup_Amarillo`, `pickup_Atlanta`, `pickup_Austin`, `pickup_Bakersfield`, `pickup_Baltimore`, `pickup_Baton Rouge` ... (+123 more)

### Temporal: why the old features were replaced

The development window is 2025-01-01..2025-10-31 and the scoring window is 2025-11-01..2025-12-31, with **zero overlap**. Consequently:
- `date_year` was constant (2025) and carried no information.
- raw `date_month` took values `{11, 12}` at inference against `{1..10}` in training. Tree models cannot extrapolate and collapse both scoring months into the October leaf; linear models extrapolate along an unsupported slope.
Both are replaced by sine/cosine encodings of day-of-year and day-of-week, which are continuous, bounded in `[-1, 1]`, and never leave the support seen in training. `days_since_reference` was considered and **disabled** in config because it is monotone and reintroduces exactly the extrapolation failure being removed.

### Geospatial

`corr(distance, posted_rate) = 0.909` makes lane geometry the dominant structure, and `corr(delivery_lon, posted_rate) = -0.257` shows a directional (westbound) premium a single distance scalar cannot express. Bearing is encoded as sine/cosine because it is circular: raw degrees would place 359 deg and 1 deg at opposite ends of the range. These features are also why unseen cities stay predictable - they depend only on coordinates.

### Interactions (deliberately only two)

- `log_distance`: rate-per-mile falls as distance rises (`corr(distance, rate_per_mile) = -0.335`), so the relationship is concave, not linear.
- `weight_per_mile`: raw `weight` correlates only 0.035 with rate, but load density is the business-meaningful form.
No further interactions were added: the PRD explicitly warns against feature explosion, and equipment effects (Flatbed 1.085x, Reefer 1.127x rate-per-mile vs Dry Van) are already captured by the categorical encoder.

## Unseen categories

- Validation contains 8 pickup cities absent from training: Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego.
- Affected rows: 725 pickup / 722 delivery (~6.0% of the scoring set).
- **Decision:** keep `handle_unknown='ignore'` and add an explicit `*_is_unknown` indicator so the model can distinguish *unseen city* from *no city*. Geography for those rows is carried by the coordinate/haversine/bearing features.
- **Rejected alternative:** `min_frequency` bucketing. Measured directly: the lowest workable threshold (0.01 = 480 rows) collapses **14 of 64** genuine cities into an 'infrequent' bucket, destroying their identity to catch unknowns. A threshold of 0.005 creates no bucket at all, leaving behaviour identical to `ignore`.

## Train / validation split approach

A random split would measure interpolation while the real task is forward extrapolation. The holdout is therefore the final contiguous block of the development window, mirroring the real train -> scoring gap:

| | rows | date range |
|:--|--:|:--|
| train | 38,477 | 2025-01-01 .. 2025-08-31 |
| holdout | 9,523 | 2025-09-01 .. 2025-10-31 |

Cross-validation uses `TimeSeriesSplit` over date-ordered rows (5 expanding-window folds) so every validation fold lies strictly after its training fold.

## Verification checks

- [x] Raw schema validated for train and validation.
- [x] Pipeline fitted on training rows only; validation never seen during fit.
- [x] Feature names preserved: no integer or placeholder column names.
- [x] Train/validation feature columns identical in content and order.
- [x] No missing values survive preprocessing.
- [x] Leakage guard: neither 'posted_rate' nor 'load_id' reached features.
- [x] Deterministic: an independent refit reproduces the matrix exactly.
- [x] Temporal split verified: no date overlap, holdout aligns with its own training block.
- [x] Reduced-feature December path verified: 31 rows reconstructed and aligned with train.
- [x] 'load_id' carried alongside validation features for safe re-joining.
- [x] Fitted pipeline and metadata persisted for reuse in later phases.

## Leakage safety

- The pipeline is fitted on training rows only; validation is transformed, never fitted.
- `posted_rate` is removed before the pipeline sees the frame and is asserted absent from the feature matrix.
- `load_id` is excluded from model inputs and re-attached only alongside the processed validation matrix, so predictions can be joined back by key rather than by row position.
- No target-derived aggregate (lane mean rate, etc.) is computed in Phase 3. Any such feature must be built inside the CV fold in a later phase.
