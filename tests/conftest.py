"""Shared pytest fixtures.

Fixtures load a bounded sample of the real CSVs rather than synthetic data so
the tests exercise the actual quirks the audit found (negative weights, missing
values, unseen cities, the temporal split).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import AppConfig, load_config
from src.data_loader import load_csv_safe
from src.pipeline import PreprocessingPipelineConfig, build_preprocessing_pipeline
from src.run_preprocessing_phase3 import build_pipeline_config

# Large enough to span the full year and contain every city, small enough to
# keep the suite fast.
_SAMPLE_ROWS = 6_000


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """Load the project configuration once per session."""
    return load_config()


@pytest.fixture(scope="session")
def train_raw(config: AppConfig) -> pd.DataFrame:
    """Full training dataset."""
    return load_csv_safe(config.paths.train, label="train")


@pytest.fixture(scope="session")
def validation_raw(config: AppConfig) -> pd.DataFrame:
    """Full validation dataset."""
    return load_csv_safe(config.paths.validation, label="validation")


@pytest.fixture(scope="session")
def train_sample(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Evenly spaced sample of training rows spanning the whole date range."""
    step = max(1, len(train_raw) // _SAMPLE_ROWS)
    return train_raw.iloc[::step].reset_index(drop=True)


@pytest.fixture(scope="session")
def validation_sample(validation_raw: pd.DataFrame) -> pd.DataFrame:
    """Evenly spaced sample of validation rows spanning the whole date range."""
    step = max(1, len(validation_raw) // _SAMPLE_ROWS)
    return validation_raw.iloc[::step].reset_index(drop=True)


@pytest.fixture(scope="session")
def pipeline_config(config: AppConfig) -> PreprocessingPipelineConfig:
    """Pipeline configuration derived from the project config."""
    return build_pipeline_config(config)


@pytest.fixture()
def fitted_pipeline(pipeline_config: PreprocessingPipelineConfig, train_sample, config):
    """A pipeline fitted on the training sample only."""
    pipeline = build_preprocessing_pipeline(pipeline_config)
    pipeline.fit(train_sample[config.columns.raw_feature_columns])
    return pipeline
