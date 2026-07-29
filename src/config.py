"""Typed configuration loader backed by ``config/config.yaml``.

Phase-3 audit findings addressed:
  * TD-18 - path and column literals were hardcoded across modules.
  * C-5    - ``PreprocessingPipelineConfig`` existed but its single field was
             never read, so configuration was decorative.
  * C-12   - a ``random_seed`` field was declared but never applied.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, resolved to absolute paths against the repo root."""

    data_dir: Path
    train: Path
    validation: Path
    predictions_template: Path
    december_inputs: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class ColumnsConfig:
    """Raw-schema column roles."""

    id: str
    target: str
    date: str
    categorical: list[str]
    numeric: list[str]
    december_absent: list[str]

    @property
    def raw_feature_columns(self) -> list[str]:
        """All raw model-input columns, excluding id and target."""
        return [*self.categorical, self.date, *self.numeric]


@dataclass(frozen=True)
class CleaningConfig:
    """Data-repair decisions derived from the Phase-1/2 audit."""

    weight_sign_repair: str
    weight_min: float
    weight_max: float
    numeric_impute_strategy: str
    missing_indicator_columns: list[str]


@dataclass(frozen=True)
class FeaturesConfig:
    """Feature-engineering switches."""

    temporal: dict[str, Any]
    geospatial: dict[str, Any]
    interactions: dict[str, Any]


@dataclass(frozen=True)
class EncodingConfig:
    """Categorical encoding policy."""

    categorical_strategy: str
    handle_unknown: str
    min_frequency: float | int | None


@dataclass(frozen=True)
class ScalingConfig:
    """Numeric scaling policy."""

    weight_scaler: str
    default_scaler: str


@dataclass(frozen=True)
class SplitConfig:
    """Train/holdout split policy."""

    strategy: str
    holdout_start: str
    n_cv_splits: int


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object."""

    name: str
    random_seed: int
    paths: PathsConfig
    columns: ColumnsConfig
    cleaning: CleaningConfig
    features: FeaturesConfig
    encoding: EncodingConfig
    scaling: ScalingConfig
    split: SplitConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _resolve(root: Path, value: str) -> Path:
    """Resolve a configured path against the project root."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate)


def load_config(path: str | Path | None = None, *, project_root: Path | None = None) -> AppConfig:
    """Load and validate the project configuration.

    Args:
        path: Optional explicit path to a YAML config file.
        project_root: Optional root used to resolve relative paths.

    Returns:
        A fully populated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required section or key is missing.
    """
    root = project_root or PROJECT_ROOT
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    required_sections = [
        "project",
        "paths",
        "columns",
        "cleaning",
        "features",
        "encoding",
        "scaling",
        "split",
    ]
    missing = [section for section in required_sections if section not in raw]
    if missing:
        raise ValueError(f"Configuration is missing required sections: {missing}")

    paths_raw = raw["paths"]
    paths = PathsConfig(
        data_dir=_resolve(root, paths_raw["data_dir"]),
        train=_resolve(root, paths_raw["train"]),
        validation=_resolve(root, paths_raw["validation"]),
        predictions_template=_resolve(root, paths_raw["predictions_template"]),
        december_inputs=_resolve(root, paths_raw["december_inputs"]),
        processed_dir=_resolve(root, paths_raw["processed_dir"]),
        models_dir=_resolve(root, paths_raw["models_dir"]),
        reports_dir=_resolve(root, paths_raw["reports_dir"]),
        figures_dir=_resolve(root, paths_raw["figures_dir"]),
    )

    columns_raw = raw["columns"]
    columns = ColumnsConfig(
        id=columns_raw["id"],
        target=columns_raw["target"],
        date=columns_raw["date"],
        categorical=list(columns_raw["categorical"]),
        numeric=list(columns_raw["numeric"]),
        december_absent=list(columns_raw["december_absent"]),
    )

    cleaning_raw = raw["cleaning"]
    cleaning = CleaningConfig(
        weight_sign_repair=cleaning_raw["weight_sign_repair"],
        weight_min=float(cleaning_raw["weight_min"]),
        weight_max=float(cleaning_raw["weight_max"]),
        numeric_impute_strategy=cleaning_raw["numeric_impute_strategy"],
        missing_indicator_columns=list(cleaning_raw["missing_indicator_columns"]),
    )

    features_raw = raw["features"]
    features = FeaturesConfig(
        temporal=dict(features_raw["temporal"]),
        geospatial=dict(features_raw["geospatial"]),
        interactions=dict(features_raw["interactions"]),
    )

    encoding_raw = raw["encoding"]
    encoding = EncodingConfig(
        categorical_strategy=encoding_raw["categorical_strategy"],
        handle_unknown=encoding_raw["handle_unknown"],
        min_frequency=encoding_raw.get("min_frequency"),
    )

    scaling_raw = raw["scaling"]
    scaling = ScalingConfig(
        weight_scaler=scaling_raw["weight_scaler"],
        default_scaler=scaling_raw["default_scaler"],
    )

    split_raw = raw["split"]
    split = SplitConfig(
        strategy=split_raw["strategy"],
        holdout_start=str(split_raw["holdout_start"]),
        n_cv_splits=int(split_raw["n_cv_splits"]),
    )

    return AppConfig(
        name=raw["project"]["name"],
        random_seed=int(raw["project"]["random_seed"]),
        paths=paths,
        columns=columns,
        cleaning=cleaning,
        features=features,
        encoding=encoding,
        scaling=scaling,
        split=split,
        raw=raw,
    )


def set_global_seed(seed: int) -> None:
    """Seed every stochastic source used by the project.

    Addresses audit finding ML-8 / C-12: no seed was previously set anywhere,
    so runs were not reproducible.

    Args:
        seed: Seed value applied to ``random``, ``numpy`` and ``PYTHONHASHSEED``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
