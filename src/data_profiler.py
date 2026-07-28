from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureAudit:
    """
    Per-feature audit result.

    Attributes:
        feature_name: Column name
        dtype: Inferred pandas dtype (string)
        unique_values: List of example/complete unique values (may be truncated)
        missing_count: Missing value count
        missing_pct: Missing value percentage
        possible_purpose: Inferred purpose (identifier/target/datetime/numeric/categorical)
        observations: List of noteworthy observations
    """

    feature_name: str
    dtype: str
    unique_values: list[str]
    missing_count: int
    missing_pct: float
    possible_purpose: str
    observations: list[str]


@dataclass(frozen=True)
class ProfileSummary:
    """
    Dataset-level profile outputs.
    """

    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    feature_audits: list[FeatureAudit]
    inferred_target: str | None
    inferred_identifier_columns: list[str]
    inferred_datetime_features: list[str]
    inferred_numerical_features: list[str]
    inferred_categorical_features: list[str]


def _is_datetime_like_column(col: str, dtype: str) -> bool:
    name = col.lower()
    if any(k in name for k in ["date", "time", "datetime"]):
        return True
    # pandas dtype 'datetime64[...]' may already exist if CSV parsed as such
    return dtype.startswith("datetime64")


def _infer_identifier_columns(columns: list[str]) -> list[str]:
    cols = [c for c in columns]
    identifiers: list[str] = []
    for c in cols:
        lc = c.lower()
        if lc == "load_id" or lc.endswith("_id"):
            identifiers.append(c)
    return identifiers


def _infer_target_column(columns: list[str]) -> str | None:
    """
    Infer likely target from column name heuristics (documented).

    This does not guarantee correctness; notebook will cross-check.
    """
    candidates = []
    for c in columns:
        lc = c.lower()
        if lc in {"target", "y", "rate", "freight_rate", "predicted_rate"}:
            candidates.append(c)
        if "rate" in lc and lc not in {"predicted_rate"}:
            # e.g. 'freight_rate' / 'rate'
            candidates.append(c)

    # Prefer 'rate'/'freight_rate' type over others by ordering preference.
    preferred = ["rate", "freight_rate", "freight rate", "target", "y"]
    for p in preferred:
        for c in candidates:
            if c.lower() == p.replace(" ", "_") or c.lower() == p:
                return c
    if candidates:
        # fallback: first candidate
        return candidates[0]
    return None


def _is_numeric_col(dtype: str) -> bool:
    return dtype.startswith("int") or dtype.startswith("float") or dtype.startswith("number")


def _is_categorical_col(dtype: str) -> bool:
    return dtype in {"object", "string"} or dtype.startswith("category")


def _truncate_unique(values: list[Any], *, max_values: int = 50) -> list[str]:
    as_str = []
    for v in values[:max_values]:
        if pd.isna(v):
            continue
        as_str.append(str(v))
    if len(values) > max_values:
        as_str.append(f"... (truncated, total_unique={len(values)})")
    return as_str


def _profile_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).describe().T


def _profile_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_df = df.select_dtypes(include=["object", "string", "category"])
    rows = []
    for col in cat_df.columns:
        s = cat_df[col]
        rows.append(
            {
                "feature": col,
                "count_non_missing": int(s.notna().sum()),
                "unique_values": int(s.nunique(dropna=True)),
                "top_value": s.mode(dropna=True).iloc[0] if s.dropna().shape[0] and s.mode(dropna=True).shape[0] else None,
            }
        )
    return pd.DataFrame(rows).set_index("feature") if rows else pd.DataFrame(columns=["count_non_missing", "unique_values"])


def audit_features(
    df: pd.DataFrame,
    *,
    unique_value_max: int = 50,
) -> list[FeatureAudit]:
    """
    Inspect every feature individually.

    Notes:
      - We do not modify data.
      - We cap unique values output to avoid huge reports.
    """
    audits: list[FeatureAudit] = []
    n_rows = len(df)

    inferred_identifiers = set(_infer_identifier_columns(list(df.columns)))

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)

        missing_count = int(series.isna().sum())
        missing_pct = (100.0 * missing_count / n_rows) if n_rows else 0.0

        non_missing = series.dropna()
        unique_vals = non_missing.unique().tolist()
        unique_strs = _truncate_unique(unique_vals, max_values=unique_value_max)

        observations: list[str] = []

        # Possible purpose
        lc = col.lower()
        possible_purpose = "unknown"
        if col in inferred_identifiers:
            possible_purpose = "identifier"
        elif _is_datetime_like_column(col, dtype):
            possible_purpose = "datetime"
        elif _is_numeric_col(dtype):
            possible_purpose = "numeric"
        elif _is_categorical_col(dtype) or pd.api.types.is_object_dtype(series.dtype):
            possible_purpose = "categorical"

        # Observations heuristics
        if non_missing.empty:
            observations.append("All values are missing.")
        else:
            # constant column
            if non_missing.nunique(dropna=True) <= 1:
                observations.append("Nearly constant (<=1 unique non-missing value).")

            # cardinality
            nunique = int(non_missing.nunique(dropna=True))
            if nunique > 0:
                if nunique == 2:
                    observations.append("Binary-like categorical/numeric feature (2 unique values).")
                if nunique > 1000:
                    observations.append(f"High cardinality: {nunique} unique non-missing values.")

        # Whitespace/casing observations for categorical-like columns
        if possible_purpose == "categorical":
            non_missing_str = non_missing.astype("string")
            if bool(non_missing_str.str.contains(r"^\s|\s$", regex=True).any()):
                observations.append("Contains leading/trailing whitespace in some values (heuristic).")
            # empty strings
            if bool(non_missing_str.eq("").any()):
                observations.append("Contains empty-string category.")

        # Numeric range observation if numeric-like
        if possible_purpose == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_non_missing = numeric.dropna()
            if not numeric_non_missing.empty:
                min_v = float(numeric_non_missing.min())
                max_v = float(numeric_non_missing.max())
                observations.append(f"Numeric range: [{min_v:g}, {max_v:g}]")

        audits.append(
            FeatureAudit(
                feature_name=str(col),
                dtype=dtype,
                unique_values=unique_strs,
                missing_count=missing_count,
                missing_pct=missing_pct,
                possible_purpose=possible_purpose,
                observations=observations,
            )
        )

    return audits


def build_profile(
    df: pd.DataFrame,
) -> ProfileSummary:
    """
    Build dataset profile: summaries + per-feature audits + inferred roles.

    Inference is heuristic-based and will be documented downstream.
    """
    columns = list(map(str, df.columns))
    dtypes = {c: str(df[c].dtype) for c in columns}

    inferred_target = _infer_target_column(columns)
    identifier_cols = _infer_identifier_columns(columns)

    datetime_features = [c for c in columns if _is_datetime_like_column(c, dtypes[c])]
    numeric_features = [c for c in columns if _is_numeric_col(dtypes[c])]
    categorical_features = [c for c in columns if _is_categorical_col(dtypes[c])]

    numeric_summary = _profile_numeric(df)
    categorical_summary = _profile_categorical(df)

    audits = audit_features(df)

    return ProfileSummary(
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        feature_audits=audits,
        inferred_target=inferred_target,
        inferred_identifier_columns=identifier_cols,
        inferred_datetime_features=datetime_features,
        inferred_numerical_features=numeric_features,
        inferred_categorical_features=categorical_features,
    )
