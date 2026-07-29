"""Freight rate prediction - source package.

Modules are imported as ``from src.<module> import ...``. Phase-3 audit finding
A-1/A-2: the package previously mixed three import conventions and used bare
top-level imports internally, so ``import src.pipeline`` raised
``ModuleNotFoundError`` and the package only worked when a script happened to
put ``src/`` on ``sys.path``.
"""

__all__ = [
    "config",
    "data_loader",
    "data_profiler",
    "data_validator",
    "eda",
    "feature_engineering",
    "inference",
    "logger",
    "pipeline",
    "preprocessing",
    "preprocessing_utils",
    "splitting",
    "transformers",
]
