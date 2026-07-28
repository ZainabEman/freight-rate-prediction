# Feature Dictionary (inferred, Phase 1)

Generated from train-test.csv with heuristic descriptions based on inferred datatypes and column names.

| Feature Name   | Type    | Description (inferred)                       | Missing %   | Recommended Usage                                             |
|:---------------|:--------|:---------------------------------------------|:------------|:--------------------------------------------------------------|
| load_id        | object  | identifier feature inferred from name/type.  | 0.000%      | Use as join key only; usually not a model input               |
| pickup         | object  | categorical feature inferred from name/type. | 0.000%      | Categorical feature (encode/clean in Phase 2)                 |
| delivery       | object  | categorical feature inferred from name/type. | 0.000%      | Categorical feature (encode/clean in Phase 2)                 |
| pickup_lat     | float64 | numeric feature inferred from name/type.     | 0.000%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| pickup_lon     | float64 | numeric feature inferred from name/type.     | 0.000%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| delivery_lat   | float64 | numeric feature inferred from name/type.     | 0.000%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| delivery_lon   | float64 | numeric feature inferred from name/type.     | 0.000%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| distance       | float64 | numeric feature inferred from name/type.     | 0.000%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| equipment      | object  | categorical feature inferred from name/type. | 0.000%      | Categorical feature (encode/clean in Phase 2)                 |
| weight         | float64 | numeric feature inferred from name/type.     | 0.625%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| date           | object  | datetime feature inferred from name/type.    | 0.000%      | Use for time features; keep raw until Phase 2                 |
| market_index   | float64 | numeric feature inferred from name/type.     | 0.779%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| quote_signal   | float64 | numeric feature inferred from name/type.     | 0.000%      | Numeric feature (validate ranges; scale/transform in Phase 2) |
| posted_rate    | float64 | numeric feature inferred from name/type.     | 0.000%      | Numeric feature (validate ranges; scale/transform in Phase 2) |