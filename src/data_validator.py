from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    """
    A single data validation issue.

    Attributes:
        issue_type: High-level category (schema, quality, numeric, categorical, etc.)
        column: Column name if applicable; otherwise None.
        message: Human-readable description of the issue.
        severity: Suggested severity label ('error' | 'warning' | 'info').
    """

    issue_type: str
    column: str | None
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class DatasetSchemaReport:
    """
    Summary of dataset schema-related information.

    Attributes:
        columns: Column order as read from the CSV.
        dtypes: Inferred pandas dtypes.
        memory_bytes: pandas memory usage.
    """

    columns: list[str]
    dtypes: dict[str, str]
    memory_bytes: int


def _normalize_column_list(df: pd.DataFrame) -> list[str]:
    return list(map(str, df.columns))


def get_schema_report(df: pd.DataFrame) -> DatasetSchemaReport:
    """
    Compute a schema report with inferred dtypes and total memory usage.

    No data is modified.
    """
    columns = _normalize_column_list(df)
    dtypes = {str(col): str(df[col].dtype) for col in columns}
    # deep=True includes object memory; safe for audit purposes.
    memory_bytes = int(df.memory_usage(deep=True).sum())
    return DatasetSchemaReport(columns=columns, dtypes=dtypes, memory_bytes=memory_bytes)


def compare_schemas(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> list[ValidationIssue]:
    """
    Compare schema of training and validation datasets.

    Reports:
      - missing columns in validation
      - additional columns in validation
      - dtype mismatches (inferred)
    """
    issues: list[ValidationIssue] = []

    train_cols = set(map(str, train_df.columns))
    val_cols = set(map(str, validation_df.columns))

    missing = sorted(train_cols - val_cols)
    additional = sorted(val_cols - train_cols)

    if missing:
        issues.append(
            ValidationIssue(
                issue_type="schema",
                column=None,
                message=f"Validation is missing columns present in training: {missing}",
                severity="error",
            )
        )
    if additional:
        issues.append(
            ValidationIssue(
                issue_type="schema",
                column=None,
                message=f"Validation has additional columns not present in training: {additional}",
                severity="warning",
            )
        )

    # dtype mismatches only for intersection
    for col in sorted(train_cols & val_cols):
        train_dtype = str(train_df[col].dtype)
        val_dtype = str(validation_df[col].dtype)
        if train_dtype != val_dtype:
            issues.append(
                ValidationIssue(
                    issue_type="schema",
                    column=col,
                    message=f"Dtype mismatch for column '{col}': train={train_dtype}, validation={val_dtype}",
                    severity="warning",
                )
            )

    return issues


def find_duplicate_rows(df: pd.DataFrame) -> ValidationIssue | None:
    """
    Check whether duplicate full rows exist.

    Returns a single issue if found; otherwise None.
    """
    dup_mask = df.duplicated(keep=False)
    if bool(dup_mask.any()):
        count = int(dup_mask.sum())
        return ValidationIssue(
            issue_type="quality",
            column=None,
            message=f"Found duplicate rows: {count} rows are part of duplicate sets (keep=False).",
            severity="warning",
        )
    return None


def find_duplicate_load_ids(df: pd.DataFrame, *, load_id_col: str = "load_id") -> ValidationIssue | None:
    """
    Check duplicates for identifier column 'load_id'.

    Notes:
      - If load_id_col is absent, issue is info (not error) because audit must not assume.
    """
    if load_id_col not in df.columns:
        return ValidationIssue(
            issue_type="quality",
            column=load_id_col,
            message=f"'{load_id_col}' column not found. Cannot validate duplicate load IDs.",
            severity="info",
        )

    series = df[load_id_col]
    missing_mask = series.isna()
    dup_mask = series.duplicated(keep=False)

    if bool(missing_mask.any()) or bool(dup_mask.any()):
        missing_count = int(missing_mask.sum())
        dup_count = int(dup_mask.sum())
        return ValidationIssue(
            issue_type="quality",
            column=load_id_col,
            message=(
                f"Duplicate load IDs / missing load_id detected: missing={missing_count}, "
                f"rows in duplicate sets={dup_count}."
            ),
            severity="error" if dup_count > 0 else "warning",
        )
    return None


def missing_values_report(df: pd.DataFrame) -> list[ValidationIssue]:
    """
    For every column with missing values, emit one issue with counts and percentages.
    """
    issues: list[ValidationIssue] = []
    n_rows = len(df)

    for col in df.columns:
        miss_count = int(df[col].isna().sum())
        if miss_count == 0:
            continue
        miss_pct = 100.0 * miss_count / n_rows if n_rows else 0.0
        issues.append(
            ValidationIssue(
                issue_type="quality",
                column=str(col),
                message=f"Missing values: {miss_count}/{n_rows} ({miss_pct:.3f}%).",
                severity="warning",
            )
        )
    return issues


def _coerce_numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Attempt numeric coercion without mutating original series.

    Returns:
        (numeric_values, invalid_mask)
    """
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = numeric.isna() & series.notna()
    return numeric, invalid_mask


def invalid_numeric_values_report(
    df: pd.DataFrame, *, numeric_candidate_min_unique: int = 5
) -> list[ValidationIssue]:
    """
    Detect invalid numeric values.

    IMPORTANT (fixing Phase-1 false positives):
    - We only treat a column as "numeric to validate" if:
        (a) pandas already inferred a numeric dtype, OR
        (b) the column name strongly suggests numeric content (heuristic).
    - We do NOT attempt numeric coercion on arbitrary object/string columns
      (e.g., IDs, categoricals like pickup/delivery/date) to avoid incorrect failures.
    """
    issues: list[ValidationIssue] = []

    numeric_name_hints = {
        "lat",
        "lon",
        "lng",
        "latitude",
        "longitude",
        "distance",
        "weight",
        "index",
        "signal",
        "rate",
        "amount",
        "price",
        "quote",
    }

    for col in df.columns:
        series = df[col]
        col_name = str(col).lower()
        is_numeric_dtype = pd.api.types.is_numeric_dtype(series.dtype)

        name_suggests_numeric = any(hint in col_name for hint in numeric_name_hints)
        if not is_numeric_dtype and not name_suggests_numeric:
            continue

        # Optional guard: for object columns with numeric-looking names, still avoid flagging
        # if the column is actually low-cardinality categorical disguised as object.
        if not is_numeric_dtype:
            if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
                if series.nunique(dropna=True) < numeric_candidate_min_unique:
                    continue

        numeric, invalid_mask = _coerce_numeric(series)
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            total_non_missing = int(series.notna().sum())
            invalid_pct = 100.0 * invalid_count / total_non_missing if total_non_missing else 0.0
            issues.append(
                ValidationIssue(
                    issue_type="numeric",
                    column=str(col),
                    message=(
                        f"Invalid numeric values found for '{col}': "
                        f"{invalid_count}/{total_non_missing} coercions failed ({invalid_pct:.3f}%)."
                    ),
                    severity="error",
                )
            )
            continue

        non_finite = (~np.isfinite(numeric.astype(float))) & numeric.notna()
        if bool(non_finite.any()):
            count = int(non_finite.sum())
            issues.append(
                ValidationIssue(
                    issue_type="numeric",
                    column=str(col),
                    message=f"Non-finite numeric values found for '{col}': count={count}.",
                    severity="error",
                )
            )

    return issues


def _find_coordinate_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Identify likely latitude/longitude column pairs based on names.

    This is a heuristic and may not match the dataset; we document findings elsewhere.
    """
    cols = list(map(str, df.columns))
    lat_candidates = [
        c
        for c in cols
        if c.lower() in {"lat", "latitude"} or "latitude" in c.lower() or c.lower().endswith("_lat")
    ]
    lon_candidates = [
        c
        for c in cols
        if c.lower() in {"lon", "lng", "longitude"} or "longitude" in c.lower() or c.lower().endswith("_lon")
    ]

    pairs: list[tuple[str, str]] = []
    for lat in lat_candidates:
        for lon in lon_candidates:
            pairs.append((lat, lon))
    return pairs


def impossible_coordinates_report(df: pd.DataFrame) -> list[ValidationIssue]:
    """
    Validate coordinate ranges if latitude/longitude-like columns exist.

    latitude: [-90, 90]
    longitude: [-180, 180]
    """
    issues: list[ValidationIssue] = []
    pairs = _find_coordinate_columns(df)
    if not pairs:
        return issues

    # Phase-3 fix (audit finding C-3): this was previously wrapped in an outer
    # loop whose variables were discarded and which always broke on its first
    # iteration, so it re-checked every pair once and then exited. Each distinct
    # coordinate column is now checked exactly once.
    checked: set[str] = set()
    for lat_col, lon_col in pairs:
        for col, lo, hi in [(lat_col, -90.0, 90.0), (lon_col, -180.0, 180.0)]:
            if col in checked:
                continue
            checked.add(col)

            numeric, invalid_mask = _coerce_numeric(df[col])
            out_of_range = numeric.notna() & ((numeric < lo) | (numeric > hi))
            if bool(out_of_range.any()):
                count = int(out_of_range.sum())
                issues.append(
                    ValidationIssue(
                        issue_type="quality",
                        column=col,
                        message=(
                            f"Impossible coordinate values in '{col}': {count} rows outside [{lo}, {hi}]."
                        ),
                        severity="warning",
                    )
                )

            if bool(invalid_mask.any()):
                count = int(invalid_mask.sum())
                issues.append(
                    ValidationIssue(
                        issue_type="numeric",
                        column=col,
                        message=f"Non-numeric coordinate values in '{col}': {count} coercions failed.",
                        severity="warning",
                    )
                )

    return issues


def categorical_inconsistency_report(df: pd.DataFrame) -> list[ValidationIssue]:
    """
    Heuristic checks for categorical inconsistencies on object/string columns:
      - leading/trailing whitespace
      - empty strings as a category
      - mixed casing (high uniqueness with multiple case variants)
    """
    issues: list[ValidationIssue] = []
    object_cols = [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c].dtype) or pd.api.types.is_string_dtype(df[c].dtype)
    ]

    for col in object_cols:
        s = df[col].astype("string")
        non_missing = s.dropna()
        if len(non_missing) == 0:
            continue

        whitespace_mask = non_missing.str.contains(r"^\s|\s$", regex=True)
        if bool(whitespace_mask.any()):
            issues.append(
                ValidationIssue(
                    issue_type="categorical",
                    column=str(col),
                    message=f"Categorical inconsistencies: leading/trailing whitespace detected in '{col}'.",
                    severity="warning",
                )
            )

        empty_mask = non_missing.eq("")
        if bool(empty_mask.any()):
            issues.append(
                ValidationIssue(
                    issue_type="categorical",
                    column=str(col),
                    message=f"Categorical value includes empty strings for '{col}'.",
                    severity="warning",
                )
            )

        uniq_raw = non_missing.nunique(dropna=True)
        uniq_casefold = non_missing.str.casefold().nunique(dropna=True)
        if uniq_raw >= 2 and uniq_casefold < uniq_raw:
            if uniq_casefold / uniq_raw <= 0.7:
                issues.append(
                    ValidationIssue(
                        issue_type="categorical",
                        column=str(col),
                        message=(
                            f"Mixed casing likely in '{col}' (case-fold reduces unique count from {uniq_raw} to {uniq_casefold})."
                        ),
                        severity="info",
                    )
                )

    return issues


def validate_all(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    load_id_col: str = "load_id",
) -> dict[str, Any]:
    """
    Run all Phase 1 validation/audit checks.

    Returns:
        Dictionary containing:
          - schema reports
          - per-dataset issues list
          - cross-dataset schema comparison issues
    """
    train_schema = get_schema_report(train_df)
    val_schema = get_schema_report(validation_df)

    train_issues: list[ValidationIssue] = []
    val_issues: list[ValidationIssue] = []

    dup_issue = find_duplicate_rows(train_df)
    if dup_issue:
        train_issues.append(dup_issue)
    dup_issue = find_duplicate_rows(validation_df)
    if dup_issue:
        val_issues.append(dup_issue)

    issue = find_duplicate_load_ids(train_df, load_id_col=load_id_col)
    if issue:
        train_issues.append(issue)
    issue = find_duplicate_load_ids(validation_df, load_id_col=load_id_col)
    if issue:
        val_issues.append(issue)

    train_issues.extend(missing_values_report(train_df))
    val_issues.extend(missing_values_report(validation_df))

    train_issues.extend(invalid_numeric_values_report(train_df))
    val_issues.extend(invalid_numeric_values_report(validation_df))

    train_issues.extend(impossible_coordinates_report(train_df))
    val_issues.extend(impossible_coordinates_report(validation_df))

    train_issues.extend(categorical_inconsistency_report(train_df))
    val_issues.extend(categorical_inconsistency_report(validation_df))

    schema_mismatch_issues = compare_schemas(train_df, validation_df)

    return {
        "train_schema": train_schema,
        "validation_schema": val_schema,
        "train_issues": train_issues,
        "validation_issues": val_issues,
        "cross_dataset_schema_issues": schema_mismatch_issues,
    }
