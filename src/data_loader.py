from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadedDatasets:
    train: pd.DataFrame
    validation: pd.DataFrame
    train_path: Path
    validation_path: Path


def load_csv_safe(path: str | Path, *, label: str) -> pd.DataFrame:
    """
    Safely load a CSV into a pandas DataFrame.

    Notes:
        - Does not perform any preprocessing or feature engineering.
        - Uses pandas' dtype inference.
        - Validation of schema/content happens in data_validator.py.

    Args:
        path: File path to the CSV.
        label: Human-readable label for error messages.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as CSV.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {csv_path.resolve()}")

    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not read {label} CSV: {csv_path.resolve()} ({exc})") from exc


def load_datasets(
    *,
    train_path: str | Path = "train-test.csv",
    validation_path: str | Path = "validation.csv",
) -> LoadedDatasets:
    """
    Load Phase 1 datasets.

    Args:
        train_path: Path to the training dataset CSV.
        validation_path: Path to the unseen validation dataset CSV.

    Returns:
        LoadedDatasets containing both DataFrames and their paths.
    """
    train_df = load_csv_safe(train_path, label="train-test")
    validation_df = load_csv_safe(validation_path, label="validation")

    return LoadedDatasets(
        train=train_df,
        validation=validation_df,
        train_path=Path(train_path),
        validation_path=Path(validation_path),
    )
