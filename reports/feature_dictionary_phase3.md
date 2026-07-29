# Feature Dictionary (Phase 3, model-ready features)

The preprocessing pipeline emits **156** named features. Named, engineered and indicator features are documented individually below; one-hot columns are summarised as a group because they follow a single mechanical pattern.

## Engineered and numeric features

| Feature | Type | Description |
|:--|:--|:--|
| `weight` | float | Load weight in lb, sign-repaired and clipped. |
| `pickup_lat` | float | Origin latitude (deterministic per city). |
| `pickup_lon` | float | Origin longitude. |
| `delivery_lat` | float | Destination latitude. |
| `delivery_lon` | float | Destination longitude. |
| `distance` | float | Lane distance in miles; strongest single predictor (r = 0.909). |
| `market_index` | float | Daily market tightness signal; dominant date-driven driver. |
| `quote_signal` | float | Per-load quoting metric. |
| `doy_sin` | float | Sine of day-of-year angle; seasonal position. |
| `doy_cos` | float | Cosine of day-of-year angle; seasonal position. |
| `dow_sin` | float | Sine of day-of-week angle. |
| `dow_cos` | float | Cosine of day-of-week angle. |
| `is_weekend` | binary | 1 when the pickup date falls on Saturday or Sunday. |
| `haversine_miles` | float | Great-circle distance between origin and destination. |
| `bearing_sin` | float | Sine of the origin->destination bearing. |
| `bearing_cos` | float | Cosine of the origin->destination bearing. |
| `lon_delta` | float | Signed east-west displacement; negative = westbound. |
| `lat_delta` | float | Signed north-south displacement. |
| `log_distance` | float | log1p(distance); linearises the concave distance/rate curve. |
| `weight_per_mile` | float | Load density: weight divided by distance. |
| `weight_is_missing` | binary | 1 when raw `weight` was null before imputation. |
| `market_index_is_missing` | binary | 1 when raw `market_index` was null. |
| `pickup_is_unknown` | binary | 1 when the origin city was unseen in training. |
| `delivery_is_unknown` | binary | 1 when the destination city was unseen in training. |
| `equipment_is_unknown` | binary | 1 when the equipment type was unseen in training. |

## One-hot encoded categorical features

- **131** binary columns generated from `pickup`, `delivery` and `equipment`.
- Naming pattern: `<column>_<category>`, e.g. `pickup_Lexington`, `equipment_Reefer`.
- Unseen categories produce an all-zero row across the group; that state is made explicit by the corresponding `*_is_unknown` indicator.
