"""Inference-time feature reconstruction for both scoring modes.

The assessment requires two different inference paths, and audit finding M-2
records that the second was architecturally impossible under the original
pipeline:

1. **Full validation scoring** - ``data/validation.csv`` carries all 13 raw
   feature columns and needs no reconstruction.

2. **Fixed December chart** - ``score.py`` pins the input schema to exactly
   ``pickup, delivery, distance, equipment, weight, date, predicted_rate``.
   Four coordinate columns, ``market_index`` and ``quote_signal`` are absent,
   yet the pipeline requires all of them.

Both gaps are closable from data already in the repository:

* **Coordinates** - verified deterministic: every city maps to exactly one
  ``(lat, lon)`` pair across both datasets, so a lookup table is exact rather
  than approximate.
* **Market context** - all 31 December dates appear in ``validation.csv`` with
  163-227 loads each. Their daily mean ``market_index`` spans 0.831..1.045,
  comfortably inside the training support of 0.676..1.468, so no extrapolation
  is required. This matters because daily market index is the dominant
  date-driven signal (``corr = 0.577`` against daily mean rate-per-mile) and is
  therefore what makes the December curve vary at all when only the date moves.

This module builds those lookups and reconstructs the full raw schema. It does
not predict anything; Phase 7 consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# The exact input schema required by score.py::validate_december, excluding the
# `predicted_rate` column the candidate fills in.
DECEMBER_INPUT_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]

# Fixed December scenario, transcribed from score.py module constants.
DECEMBER_FIXED_PICKUP = "Lexington"
DECEMBER_FIXED_DELIVERY = "Fort Wayne"
DECEMBER_FIXED_DISTANCE = 360.0
DECEMBER_FIXED_EQUIPMENT = "Dry Van"
DECEMBER_FIXED_WEIGHT = 32_000.0
DECEMBER_DATE_START = "2025-12-01"
DECEMBER_DATE_END = "2025-12-31"


@dataclass(frozen=True)
class CityCoordinateLookup:
    """City -> ``(latitude, longitude)`` mapping for pickup and delivery roles.

    Attributes:
        pickup: Mapping used when a city appears as an origin.
        delivery: Mapping used when a city appears as a destination.
    """

    pickup: dict[str, tuple[float, float]]
    delivery: dict[str, tuple[float, float]]

    def resolve(self, city: str, *, role: str) -> tuple[float, float]:
        """Look up coordinates for a city in a given role.

        Args:
            city: City name.
            role: Either ``"pickup"`` or ``"delivery"``.

        Returns:
            ``(latitude, longitude)``.

        Raises:
            ValueError: If ``role`` is invalid.
            KeyError: If the city is unknown in that role and in the fallback.
        """
        if role == "pickup":
            primary, fallback = self.pickup, self.delivery
        elif role == "delivery":
            primary, fallback = self.delivery, self.pickup
        else:
            raise ValueError(f"role must be 'pickup' or 'delivery', got {role!r}")

        key = str(city).strip()
        if key in primary:
            return primary[key]
        if key in fallback:
            # Cities appear in both roles with identical coordinates; falling
            # back is exact, not an approximation.
            return fallback[key]
        raise KeyError(f"No coordinates known for city {key!r} in role {role!r}")


def _role_lookup(frames: list[pd.DataFrame], *, city_col: str, lat_col: str, lon_col: str) -> dict:
    """Build a city -> coordinate mapping, verifying it is single-valued."""
    combined = pd.concat(
        [frame[[city_col, lat_col, lon_col]] for frame in frames if city_col in frame.columns],
        ignore_index=True,
    )
    combined[city_col] = combined[city_col].astype("string").str.strip()
    combined = combined.dropna()

    counts = combined.groupby(city_col)[[lat_col, lon_col]].nunique()
    ambiguous = counts[(counts[lat_col] > 1) | (counts[lon_col] > 1)]
    if not ambiguous.empty:
        raise ValueError(
            "City -> coordinate mapping is not deterministic for: "
            f"{sorted(ambiguous.index.tolist())[:10]}"
        )

    unique = combined.drop_duplicates(subset=[city_col])
    return {
        str(row[city_col]): (float(row[lat_col]), float(row[lon_col]))
        for _, row in unique.iterrows()
    }


def build_city_coordinate_lookup(*frames: pd.DataFrame) -> CityCoordinateLookup:
    """Build the city -> coordinate lookup from one or more raw frames.

    Args:
        *frames: Raw datasets containing city and coordinate columns.

    Returns:
        A :class:`CityCoordinateLookup`.

    Raises:
        ValueError: If no frames are supplied or a city maps to multiple
            coordinate pairs.
    """
    frame_list = [frame for frame in frames if frame is not None]
    if not frame_list:
        raise ValueError("At least one frame is required to build the coordinate lookup")

    pickup = _role_lookup(frame_list, city_col="pickup", lat_col="pickup_lat", lon_col="pickup_lon")
    delivery = _role_lookup(
        frame_list, city_col="delivery", lat_col="delivery_lat", lon_col="delivery_lon"
    )
    logger.info(
        "Built city coordinate lookup: %d pickup cities, %d delivery cities",
        len(pickup),
        len(delivery),
    )
    return CityCoordinateLookup(pickup=pickup, delivery=delivery)


def build_daily_market_lookup(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    market_columns: tuple[str, ...] = ("market_index", "quote_signal"),
) -> pd.DataFrame:
    """Aggregate per-load market signals into a per-date lookup table.

    Both signals vary within a day (per-day standard deviation ~0.025 for
    ``market_index``), so the daily mean is the natural summary of the market
    state on that date.

    Args:
        frame: Raw dataset containing dates and market signal columns.
        date_column: Name of the date column.
        market_columns: Signal columns to aggregate.

    Returns:
        A frame indexed by ``date`` with one mean column per signal.

    Raises:
        KeyError: If a required column is absent.
    """
    missing = [
        column for column in (date_column, *market_columns) if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Cannot build market lookup; missing columns: {missing}")

    working = frame[[date_column, *market_columns]].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    if working[date_column].isna().any():
        raise ValueError(f"Unparseable dates encountered in column {date_column!r}")

    lookup = working.groupby(date_column)[list(market_columns)].mean()
    logger.info(
        "Built daily market lookup covering %d dates (%s..%s)",
        len(lookup),
        lookup.index.min().date(),
        lookup.index.max().date(),
    )
    return lookup


def build_december_chart_inputs() -> pd.DataFrame:
    """Construct the fixed December scenario frame required by ``score.py``.

    The assessment's ``data/december_chart_inputs.csv`` is absent from this
    repository (audit finding M-2), but ``score.py`` pins every value, so the
    file is reconstructed exactly rather than guessed: one row per day from
    2025-12-01 to 2025-12-31 with all non-date inputs held constant.

    Returns:
        A 31-row frame with columns :data:`DECEMBER_INPUT_COLUMNS` plus an empty
        ``predicted_rate`` column for Phase 7 to fill.
    """
    dates = pd.date_range(DECEMBER_DATE_START, DECEMBER_DATE_END, freq="D")
    frame = pd.DataFrame(
        {
            "pickup": DECEMBER_FIXED_PICKUP,
            "delivery": DECEMBER_FIXED_DELIVERY,
            "distance": DECEMBER_FIXED_DISTANCE,
            "equipment": DECEMBER_FIXED_EQUIPMENT,
            "weight": DECEMBER_FIXED_WEIGHT,
            "date": dates.strftime("%Y-%m-%d"),
        }
    )
    frame["predicted_rate"] = np.nan
    return frame


def enrich_reduced_frame(
    frame: pd.DataFrame,
    *,
    coordinates: CityCoordinateLookup,
    market_lookup: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Expand a reduced-schema frame to the full raw feature schema.

    Args:
        frame: Frame containing at least :data:`DECEMBER_INPUT_COLUMNS`.
        coordinates: City -> coordinate lookup.
        market_lookup: Per-date market signal table from
            :func:`build_daily_market_lookup`.
        date_column: Name of the date column.

    Returns:
        A frame carrying every raw feature column the pipeline expects.

    Raises:
        KeyError: If required reduced columns are absent, or a city or date
            cannot be resolved. Failing loudly here is deliberate: a silently
            missing market value would flow through imputation and produce a
            flat, meaningless December curve.
    """
    missing = [column for column in DECEMBER_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Reduced frame is missing required columns: {missing}")

    out = frame.copy()

    pickup_coords = [coordinates.resolve(city, role="pickup") for city in out["pickup"]]
    delivery_coords = [coordinates.resolve(city, role="delivery") for city in out["delivery"]]
    out["pickup_lat"] = [lat for lat, _ in pickup_coords]
    out["pickup_lon"] = [lon for _, lon in pickup_coords]
    out["delivery_lat"] = [lat for lat, _ in delivery_coords]
    out["delivery_lon"] = [lon for _, lon in delivery_coords]

    dates = pd.to_datetime(out[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Unparseable dates in column {date_column!r}")

    unresolved = sorted({str(value.date()) for value in dates[~dates.isin(market_lookup.index)]})
    if unresolved:
        raise KeyError(
            f"No market context available for {len(unresolved)} date(s): {unresolved[:10]}"
        )

    for column in market_lookup.columns:
        out[column] = market_lookup.loc[dates, column].to_numpy()

    logger.info("Enriched reduced frame: %d rows -> %d columns", len(out), out.shape[1])
    return out


def write_december_chart_inputs(path: str | Path) -> Path:
    """Write the reconstructed December input file to disk.

    Args:
        path: Destination CSV path.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = build_december_chart_inputs()
    frame.to_csv(destination, index=False)
    logger.info("Wrote December chart inputs: %s (%d rows)", destination, len(frame))
    return destination
