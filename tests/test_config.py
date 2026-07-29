"""Tests for configuration loading and repository import structure."""

from __future__ import annotations

import importlib

import pytest

from src.config import load_config, set_global_seed


def test_config_loads_and_resolves_paths(config) -> None:
    """Configured data paths must resolve to files that exist."""
    assert config.paths.train.is_file()
    assert config.paths.validation.is_file()
    assert config.paths.predictions_template.is_file()


def test_config_column_roles_are_consistent(config) -> None:
    """The raw feature list must exclude the identifier and the target."""
    raw_features = config.columns.raw_feature_columns
    assert config.columns.id not in raw_features
    assert config.columns.target not in raw_features
    assert config.columns.date in raw_features


def test_missing_config_file_raises() -> None:
    """A wrong path must fail loudly rather than fall back to defaults."""
    with pytest.raises(FileNotFoundError):
        load_config("config/does_not_exist.yaml")


def test_global_seed_is_applied() -> None:
    """Seeding must make numpy draws reproducible."""
    import numpy as np

    set_global_seed(42)
    first = np.random.rand(5)
    set_global_seed(42)
    second = np.random.rand(5)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "module",
    [
        "src.config",
        "src.data_loader",
        "src.data_profiler",
        "src.data_validator",
        "src.eda",
        "src.feature_engineering",
        "src.inference",
        "src.logger",
        "src.pipeline",
        "src.preprocessing",
        "src.preprocessing_utils",
        "src.splitting",
        "src.transformers",
    ],
)
def test_every_module_is_importable_as_a_package(module: str) -> None:
    """Audit finding A-1: `import src.pipeline` used to raise ModuleNotFoundError."""
    assert importlib.import_module(module) is not None
