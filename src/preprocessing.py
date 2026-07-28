from __future__ import annotations

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


@dataclass(frozen=True)
class NumericPreprocessingConfig:
    """
    Numeric preprocessing decisions derived from Phase 1 audit findings:
      - Missing values exist for:
          * weight (~0.625% train)
          * market_index (~0.779% train)
      - Numeric columns also include potentially heavy-tailed values (weight can be negative).

    Decisions:
      - Impute missing numeric values using median (robust to outliers).
      - Scale:
          * use StandardScaler for most continuous numeric features
          * use RobustScaler for weight (more robust to outliers)
    """

    missing_strategy: str = "median"
    weight_scaler: str = "robust"
    other_scaler: str = "standard"


@dataclass(frozen=True)
class CategoricalPreprocessingConfig:
    """
    Categorical preprocessing decisions derived from Phase 1 audit findings:
      - No missing values detected for pickup/delivery/equipment in Phase 1.
      - Possible whitespace/casing inconsistencies were checked heuristically.
    Decisions:
      - Strip whitespace is handled by SafeStringNormalizer (separate transformer).
      - Encode using OneHotEncoder(handle_unknown="ignore", sparse_output=False).

    Notes:
      - We keep encoding as sparse_output=False to simplify writing processed CSVs.
        (Feature explosion is bounded: pickup/delivery each ~64 unique, equipment ~3.)
    """

    handle_unknown: str = "ignore"
    sparse_output: bool = False


def build_numeric_imputer(strategy: str = "median") -> SimpleImputer:
    """
    Build an imputer for numeric columns.
    """
    return SimpleImputer(strategy=strategy)


def build_numeric_scalers(*, weight_scaler: str = "robust", other_scaler: str = "standard"):
    """
    Build scalers for numeric columns.

    Returns:
      (weight_scaler_obj, other_scaler_obj)
    """
    weight_scaler_obj = RobustScaler() if weight_scaler == "robust" else StandardScaler()
    other_scaler_obj = StandardScaler() if other_scaler == "standard" else RobustScaler()
    return weight_scaler_obj, other_scaler_obj


def build_onehot_encoder(*, handle_unknown: str = "ignore", sparse_output: bool = False) -> OneHotEncoder:
    """
    Build OneHotEncoder for categorical columns.
    """
    return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)
