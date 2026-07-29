"""Stateless and stateful scikit-learn transformers for the freight pipeline.

Every transformer here implements ``get_feature_names_out`` so that feature
names survive the whole pipeline. Audit finding C-1: the previous date
transformer was wrapped in a bare ``FunctionTransformer`` with no
``feature_names_out``, which made ``ColumnTransformer.get_feature_names_out()``
raise. That exception was swallowed by a bare ``except`` and every processed
column was silently renamed to an integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Mean Earth radius in statute miles; `distance` in the dataset is in miles.
EARTH_RADIUS_MILES = 3958.7613


def _as_frame(X: Any, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Coerce transformer input to a DataFrame without copying when possible."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=list(columns) if columns is not None else None)


class SafeStringNormalizer(BaseEstimator, TransformerMixin):
    """Cast categorical columns to string and strip surrounding whitespace.

    Case is deliberately left untouched: the Phase-1 audit found no mixed-case
    duplicates, so case folding would be an unjustified assumption.
    """

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "SafeStringNormalizer":
        """Record input columns; no statistics are learned."""
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with whitespace-stripped string columns."""
        check_is_fitted(self, "feature_names_in_")
        frame = _as_frame(X).copy()
        for column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()
        return frame

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return unchanged column names."""
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)


class RawDataCleaner(BaseEstimator, TransformerMixin):
    """Apply deterministic, row-local repairs to the raw feature frame.

    All repairs are stateless (they depend only on the row being transformed),
    so applying them inside the pipeline cannot leak information between splits
    while guaranteeing training and inference are cleaned identically.

    Repairs performed:
      * ``weight`` sign correction (audit finding M-3). 292 training rows carry
        negative weights whose absolute values are distributionally identical
        to the positive population (mean 31,724 vs 31,415; identical
        ``[5000, 47500]`` support), identifying a sign-flip data-entry fault
        rather than a distinct population.
      * ``weight`` clipping to the observed physical envelope.
      * Whitespace stripping on categorical columns.
    """

    def __init__(
        self,
        *,
        categorical_columns: Sequence[str],
        weight_column: str = "weight",
        weight_sign_repair: str = "abs",
        weight_min: float = 5000.0,
        weight_max: float = 47500.0,
    ) -> None:
        # sklearn's clone contract requires __init__ to store parameters
        # unmodified; any normalisation happens in fit().
        self.categorical_columns = categorical_columns
        self.weight_column = weight_column
        self.weight_sign_repair = weight_sign_repair
        self.weight_min = weight_min
        self.weight_max = weight_max

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "RawDataCleaner":
        """Record input columns; no statistics are learned."""
        frame = _as_frame(X)
        self.categorical_columns_ = list(self.categorical_columns)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of ``X`` with identical columns.

        Raises:
            ValueError: If ``weight_sign_repair`` is not a supported strategy.
        """
        check_is_fitted(self, "feature_names_in_")
        frame = _as_frame(X).copy()

        if self.weight_column in frame.columns:
            weight = pd.to_numeric(frame[self.weight_column], errors="coerce")
            if self.weight_sign_repair == "abs":
                weight = weight.abs()
            elif self.weight_sign_repair == "nan":
                weight = weight.where(weight >= 0, other=np.nan)
            else:
                raise ValueError(
                    f"Unsupported weight_sign_repair strategy: {self.weight_sign_repair!r}. "
                    "Expected 'abs' or 'nan'."
                )
            # Preserve NaN (handled downstream by the imputer) while clipping
            # only genuinely observed values into the physical envelope.
            frame[self.weight_column] = weight.clip(lower=self.weight_min, upper=self.weight_max)

        for column in self.categorical_columns_:
            if column in frame.columns:
                frame[column] = frame[column].astype("string").str.strip()

        return frame

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return unchanged column names."""
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)


@dataclass(frozen=True)
class TemporalFeatureSpec:
    """Switches controlling which temporal representations are produced."""

    use_cyclical_day_of_year: bool = True
    use_cyclical_day_of_week: bool = True
    use_is_weekend: bool = True
    use_days_since_reference: bool = False
    reference_date: str = "2025-01-01"


class CyclicalDateFeatures(BaseEstimator, TransformerMixin):
    """Encode a date column as extrapolation-safe cyclical features.

    Audit finding M-1 drives this design. Training covers 2025-01-01..2025-10-31
    and scoring covers 2025-11-01..2025-12-31, so:

      * ``date_year`` is constant (2025) and carries zero information.
      * raw ``date_month`` takes values ``{11, 12}`` at inference against
        ``{1..10}`` in training. Tree models cannot extrapolate and collapse
        both scoring months into the October leaf; linear models extrapolate
        along an unsupported slope.

    Sine/cosine encodings of day-of-year and day-of-week are continuous, bounded
    in ``[-1, 1]``, and never leave the support seen during training, which
    removes the extrapolation failure mode entirely.
    """

    def __init__(self, *, date_column: str = "date", spec: TemporalFeatureSpec | None = None) -> None:
        self.date_column = date_column
        self.spec = spec or TemporalFeatureSpec()

    def _output_names(self) -> list[str]:
        spec = self.spec
        names: list[str] = []
        if spec.use_cyclical_day_of_year:
            names += ["doy_sin", "doy_cos"]
        if spec.use_cyclical_day_of_week:
            names += ["dow_sin", "dow_cos"]
        if spec.use_is_weekend:
            names.append("is_weekend")
        if spec.use_days_since_reference:
            names.append("days_since_reference")
        return names

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "CyclicalDateFeatures":
        """Validate the date column is present; no statistics are learned."""
        frame = _as_frame(X)
        if self.date_column not in frame.columns:
            raise KeyError(
                f"date column {self.date_column!r} not found; present columns={list(frame.columns)}"
            )
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = self._output_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the engineered temporal columns.

        Raises:
            ValueError: If any date fails to parse. Audit finding NF-7 requires
                loud failure rather than silent NaN propagation into the model.
        """
        check_is_fitted(self, "feature_names_out_")
        frame = _as_frame(X)
        parsed = pd.to_datetime(frame[self.date_column], errors="coerce")

        unparsed = int(parsed.isna().sum())
        if unparsed:
            raise ValueError(
                f"{unparsed} value(s) in column {self.date_column!r} could not be parsed as dates."
            )

        spec = self.spec
        out: dict[str, np.ndarray] = {}

        if spec.use_cyclical_day_of_year:
            # Normalise by the true year length so the encoding stays aligned
            # across leap and non-leap years.
            days_in_year = np.where(parsed.dt.is_leap_year.to_numpy(), 366.0, 365.0)
            angle = 2.0 * np.pi * (parsed.dt.dayofyear.to_numpy() - 1) / days_in_year
            out["doy_sin"] = np.sin(angle)
            out["doy_cos"] = np.cos(angle)

        if spec.use_cyclical_day_of_week:
            angle = 2.0 * np.pi * parsed.dt.dayofweek.to_numpy() / 7.0
            out["dow_sin"] = np.sin(angle)
            out["dow_cos"] = np.cos(angle)

        if spec.use_is_weekend:
            out["is_weekend"] = (parsed.dt.dayofweek.to_numpy() >= 5).astype(float)

        if spec.use_days_since_reference:
            reference = pd.Timestamp(spec.reference_date)
            out["days_since_reference"] = (parsed - reference).dt.days.to_numpy().astype(float)

        return pd.DataFrame(out, index=frame.index)[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the engineered temporal feature names."""
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


@dataclass(frozen=True)
class GeoFeatureSpec:
    """Switches controlling which geospatial features are produced."""

    use_haversine: bool = True
    use_bearing: bool = True
    use_lon_delta: bool = True
    use_lat_delta: bool = True


class GeoFeatures(BaseEstimator, TransformerMixin):
    """Derive lane geometry from pickup/delivery coordinates.

    Justified by the audit: ``corr(distance, posted_rate) = 0.909`` makes lane
    geometry the dominant structure, and ``corr(delivery_lon, posted_rate) =
    -0.257`` shows a directional (westbound) premium that a raw distance scalar
    cannot express.

    Bearing is encoded as sine/cosine because it is a circular quantity: raw
    degrees would place 359 deg and 1 deg at opposite ends of the range.

    These features are also the reason unseen cities (audit finding M-4) remain
    predictable: they depend only on coordinates, not on category membership.
    """

    def __init__(
        self,
        *,
        pickup_lat: str = "pickup_lat",
        pickup_lon: str = "pickup_lon",
        delivery_lat: str = "delivery_lat",
        delivery_lon: str = "delivery_lon",
        spec: GeoFeatureSpec | None = None,
    ) -> None:
        self.pickup_lat = pickup_lat
        self.pickup_lon = pickup_lon
        self.delivery_lat = delivery_lat
        self.delivery_lon = delivery_lon
        self.spec = spec or GeoFeatureSpec()

    def _output_names(self) -> list[str]:
        spec = self.spec
        names: list[str] = []
        if spec.use_haversine:
            names.append("haversine_miles")
        if spec.use_bearing:
            names += ["bearing_sin", "bearing_cos"]
        if spec.use_lon_delta:
            names.append("lon_delta")
        if spec.use_lat_delta:
            names.append("lat_delta")
        return names

    @property
    def _required_columns(self) -> list[str]:
        return [self.pickup_lat, self.pickup_lon, self.delivery_lat, self.delivery_lon]

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "GeoFeatures":
        """Validate coordinate columns are present; no statistics are learned."""
        frame = _as_frame(X)
        missing = [column for column in self._required_columns if column not in frame.columns]
        if missing:
            raise KeyError(f"GeoFeatures missing required coordinate columns: {missing}")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = self._output_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the engineered geospatial columns."""
        check_is_fitted(self, "feature_names_out_")
        frame = _as_frame(X)

        lat1 = np.radians(frame[self.pickup_lat].to_numpy(dtype=float))
        lon1 = np.radians(frame[self.pickup_lon].to_numpy(dtype=float))
        lat2 = np.radians(frame[self.delivery_lat].to_numpy(dtype=float))
        lon2 = np.radians(frame[self.delivery_lon].to_numpy(dtype=float))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        out: dict[str, np.ndarray] = {}
        spec = self.spec

        if spec.use_haversine:
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            # clip guards against float error pushing `a` marginally above 1.
            out["haversine_miles"] = (
                2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            )

        if spec.use_bearing:
            y_component = np.sin(dlon) * np.cos(lat2)
            x_component = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            bearing = np.arctan2(y_component, x_component)
            out["bearing_sin"] = np.sin(bearing)
            out["bearing_cos"] = np.cos(bearing)

        if spec.use_lon_delta:
            out["lon_delta"] = frame[self.delivery_lon].to_numpy(dtype=float) - frame[
                self.pickup_lon
            ].to_numpy(dtype=float)

        if spec.use_lat_delta:
            out["lat_delta"] = frame[self.delivery_lat].to_numpy(dtype=float) - frame[
                self.pickup_lat
            ].to_numpy(dtype=float)

        return pd.DataFrame(out, index=frame.index)[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the engineered geospatial feature names."""
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


@dataclass(frozen=True)
class InteractionFeatureSpec:
    """Switches controlling which interaction features are produced."""

    use_weight_per_mile: bool = True
    use_log_distance: bool = True


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Derive the small set of interactions supported by EDA evidence.

    Kept deliberately narrow to avoid the feature explosion the PRD warns
    against. Only two are produced:

      * ``log_distance`` - ``distance`` spans 70..3,440 miles and rate-per-mile
        falls as distance rises (``corr(distance, rate_per_mile) = -0.335``),
        i.e. the distance/rate relationship is concave rather than linear.
      * ``weight_per_mile`` - shipping intensity; ``weight`` alone correlates
        only 0.035 with rate, but load density is the business-meaningful form.
    """

    def __init__(
        self,
        *,
        distance_column: str = "distance",
        weight_column: str = "weight",
        spec: InteractionFeatureSpec | None = None,
    ) -> None:
        self.distance_column = distance_column
        self.weight_column = weight_column
        self.spec = spec or InteractionFeatureSpec()

    def _output_names(self) -> list[str]:
        names: list[str] = []
        if self.spec.use_log_distance:
            names.append("log_distance")
        if self.spec.use_weight_per_mile:
            names.append("weight_per_mile")
        return names

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "InteractionFeatures":
        """Validate source columns are present; no statistics are learned."""
        frame = _as_frame(X)
        missing = [
            column
            for column in (self.distance_column, self.weight_column)
            if column not in frame.columns
        ]
        if missing:
            raise KeyError(f"InteractionFeatures missing required columns: {missing}")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = self._output_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the engineered interaction columns."""
        check_is_fitted(self, "feature_names_out_")
        frame = _as_frame(X)

        distance = pd.to_numeric(frame[self.distance_column], errors="coerce").to_numpy(dtype=float)
        weight = pd.to_numeric(frame[self.weight_column], errors="coerce").to_numpy(dtype=float)

        out: dict[str, np.ndarray] = {}
        if self.spec.use_log_distance:
            # distance is strictly positive in both datasets (min 70), so log1p
            # is safe; it is used rather than log for defensive symmetry.
            out["log_distance"] = np.log1p(np.clip(distance, 0.0, None))
        if self.spec.use_weight_per_mile:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(weight, distance, out=np.full_like(weight, np.nan), where=distance > 0)
            # NaN here means missing weight; the downstream imputer handles it.
            out["weight_per_mile"] = ratio

        return pd.DataFrame(out, index=frame.index)[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the engineered interaction feature names."""
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class MissingFlagger(BaseEstimator, TransformerMixin):
    """Emit binary missingness indicators for selected columns.

    Missingness is row-local and therefore stateless, so the flags are produced
    before imputation without any risk of leakage.

    Justified by the audit: missingness is low but systematically *higher* in
    the scoring window (``weight`` 0.63% train vs 1.38% validation;
    ``market_index`` 0.78% vs 2.08%). Discarding the fact of missingness would
    hide a real distribution shift from the model.
    """

    def __init__(self, *, columns: Sequence[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "MissingFlagger":
        """Validate flagged columns are present; no statistics are learned."""
        frame = _as_frame(X)
        self.columns_ = list(self.columns)
        missing = [column for column in self.columns_ if column not in frame.columns]
        if missing:
            raise KeyError(f"MissingFlagger missing required columns: {missing}")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = [f"{column}_is_missing" for column in self.columns_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the binary missingness indicator columns."""
        check_is_fitted(self, "feature_names_out_")
        frame = _as_frame(X)
        out = {
            f"{column}_is_missing": frame[column].isna().to_numpy().astype(float)
            for column in self.columns_
        }
        return pd.DataFrame(out, index=frame.index)[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the missingness indicator names."""
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class UnseenCategoryIndicator(BaseEstimator, TransformerMixin):
    """Flag rows whose category value was not present during training.

    Audit finding M-4: ``validation.csv`` contains 8 cities absent from
    ``train_test.csv`` (Allentown, Charlotte, Chicago, Jackson, Knoxville,
    Laredo, Norfolk, San Diego) affecting roughly 6% of scoring rows. With
    ``OneHotEncoder(handle_unknown="ignore")`` those rows become an all-zero
    vector that is indistinguishable from "no city at all", and the encoder
    fails silently by design.

    This transformer makes the unknown state explicit and learnable. Measured
    ``min_frequency`` bucketing was rejected as the alternative because the
    lowest workable threshold collapses 14 of 64 genuine cities.
    """

    def __init__(self, *, columns: Sequence[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> "UnseenCategoryIndicator":
        """Learn the set of category values observed per column."""
        frame = _as_frame(X)
        self.columns_ = list(self.columns)
        missing = [column for column in self.columns_ if column not in frame.columns]
        if missing:
            raise KeyError(f"UnseenCategoryIndicator missing required columns: {missing}")

        self.known_categories_: dict[str, set[str]] = {
            column: set(frame[column].astype("string").str.strip().dropna().unique())
            for column in self.columns_
        }
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = [f"{column}_is_unknown" for column in self.columns_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the binary unknown-category indicator columns."""
        check_is_fitted(self, "known_categories_")
        frame = _as_frame(X)
        out: dict[str, np.ndarray] = {}
        for column in self.columns_:
            values = frame[column].astype("string").str.strip()
            known = self.known_categories_[column]
            out[f"{column}_is_unknown"] = (~values.isin(known)).to_numpy().astype(float)
        return pd.DataFrame(out, index=frame.index)[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the unknown-category indicator names."""
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)
