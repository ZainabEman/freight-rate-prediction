# Preprocessing Report (Phase 2)

## Dataset overview
- Train raw shape: **(48000, 14)**
- Validation raw shape: **(12000, 13)**
- Train processed shape: **(48000, 143)**
- Validation processed shape: **(12000, 142)**
- Final processed feature count: **142**

## Cleaning steps performed (based on Phase-1 audit)
- Missing numeric values (weight, market_index): imputed with **median** (per-column).
- Categorical whitespace: stripped leading/trailing whitespace.
- Datetime: extracted **date_year, date_month, date_day**; original `date` dropped.

## Feature engineering performed
- `date` -> (`date_year`, `date_month`, `date_day`). No other interactions created.

## Encoding decisions
- Categorical columns: `pickup`, `delivery`, `equipment`
- Encoding: **OneHotEncoder(handle_unknown='ignore')**

## Scaling decisions
- `weight`: **RobustScaler**
- other numeric columns: **StandardScaler**

## Output missingness check
- None detected: no missing values remain in processed feature matrices.

## Leakage-safety notes
- Pipeline fit uses train split only (no use of validation statistics/encodings).
- Target `posted_rate` is excluded from preprocessing fit/transform.

## Schema/compatibility notes
- load_id is treated as an identifier and is dropped from the model feature matrix by the preprocessor.

## Risks & recommendations for Phase 3
- Ensure model training uses the processed feature matrices only; do not re-join raw identifiers unless needed for evaluation.
- If any categorical missingness appears in later data, document how OneHotEncoder treats NaNs for that case.
