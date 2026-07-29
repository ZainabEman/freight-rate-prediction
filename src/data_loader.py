"""Raw dataset loading.

Performs no preprocessing or feature engineering; schema and content validation
live in :mod:`src.data_validator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedDatasets:
    """Both raw datasets together with the paths they were read from."""

    train: pd.DataFrame
    validation: pd.DataFrame
    train_path: Path
    validation_path: Path


def load_csv_safe(path: str | Path, *, label: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame with a clear error on failure.

    Args:
        path: File path to the CSV.
        label: Human-readable label used in error messages.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read {label} CSV: {csv_path} ({exc})") from exc

    logger.info("Loaded %s: %d rows x %d columns from %s", label, *frame.shape, csv_path)
    return frame


def load_datasets(
    *,
    train_path: str | Path,
    validation_path: str | Path,
) -> LoadedDatasets:
    """Load the training and validation datasets.

    Args:
        train_path: Path to the labelled development dataset.
        validation_path: Path to the unlabelled scoring dataset.

    Returns:
        A :class:`LoadedDatasets` holding both frames and their paths.
    """
    train_df = load_csv_safe(train_path, label="train")
    validation_df = load_csv_safe(validation_path, label="validation")

    return LoadedDatasets(
        train=train_df,
        validation=validation_df,
        train_path=Path(train_path),
        validation_path=Path(validation_path),
    )
