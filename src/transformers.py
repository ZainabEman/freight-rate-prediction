from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class DateParts:
    """Extracted datetime parts column names."""

    year: str = "date_year"
    month: str = "date_month"
    day: str = "date_day"


class DateToPartsTransformer(BaseEstimator, TransformerMixin):
    """
    Convert a date-like string column into separate parts (year/month/day).

    - Fit does not learn from data.
    - Transform parses dates with pandas.to_datetime(errors="coerce").
    - Unparseable dates become NaT and then NaNs for extracted parts.

    This transformer returns a DataFrame with only the extracted parts.
    """

    def __init__(self, date_col: str, *, parts: DateParts | None = None) -> None:
        self.date_col = date_col
        self.parts = parts or DateParts()

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "DateToPartsTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.date_col not in X.columns:
            raise KeyError(f"date_col '{self.date_col}' not found in input columns")

        df = X[[self.date_col]].copy()
        parsed = pd.to_datetime(df[self.date_col], errors="coerce")

        return pd.DataFrame(
            {
                self.parts.year: parsed.dt.year,
                self.parts.month: parsed.dt.month,
                self.parts.day: parsed.dt.day,
            },
            index=X.index,
        )


class SafeStringNormalizer(BaseEstimator, TransformerMixin):
    """
    Minimal categorical string cleaning:
      - cast to string (pandas StringDtype)
      - strip leading/trailing whitespace

    No case normalization is applied (only whitespace stripping) to avoid assumptions.
    """

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "SafeStringNormalizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in out.columns:
            s = out[col].astype("string")
            out[col] = s.str.strip()
        return out
