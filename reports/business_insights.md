# Business Insights (Phase 2) - from EDA
- Target: **posted_rate**

## Interpretable patterns to validate in Phase 3
- Missingness locations: decide whether to impute with median (numeric) and how to treat categorical missing/unknown.
- Outlier prevalence: decide robust scaling vs StandardScaler and whether to clip (only if extreme outliers distort).
- Temporal structure: decide which date parts to extract if the mean-by-date plot shows seasonality/trends.
- Geographic structure: decide whether distance-from-coordinates features add value beyond provided `distance`.
- Category effects: decide encoding strategy for pickup/delivery/equipment based on frequency and group-by differences.

## Feature candidates and business justification
- Date feature extraction: year/month/day/weekday/weekend (validated after observing temporal patterns).
- Geographic distance feature(s): e.g., haversine distance from pickup to delivery (validated using geo scatter and distance relationships).
- Weight-per-distance interaction (business rationale: shipping intensity relative to distance).
- Market/quote interactions (validated using correlation analysis with posted_rate).
- Equipment category impacts on posted_rate (validated via group-by relationship).
