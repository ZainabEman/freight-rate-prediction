from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Local imports to support running as:
#   python src/run_data_preprocess_phase2.py
from data_loader import load_datasets
from pipeline import PreprocessingPipelineConfig, build_preprocessing_pipeline
from preprocessing_utils import default_feature_columns, ensure_required_columns


@dataclass(frozen=True)
class Phase2Paths:
    processed_dir: Path = Path("processed")
    reports_dir: Path = Path("reports")


def _write_processed_csv(df_features: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_path, index=False)


def _schema_smoke_check(original_df: pd.DataFrame, processed_df: pd.DataFrame, *, label: str) -> list[str]:
    issues: list[str] = []

    # No leakage: load_id should never be removed from output? Actually it is dropped by preprocessor.
    # We keep output feature matrix only, so we only check for row count.
    if len(processed_df) != len(original_df):
        issues.append(f"{label}: row count mismatch (original={len(original_df)}, processed={len(processed_df)}).")

    # Ensure no missing values remain after preprocessing
    if processed_df.isna().any().any():
        missing_cols = processed_df.columns[processed_df.isna().any()].tolist()
        issues.append(f"{label}: remaining missing values in columns: {missing_cols}.")

    return issues


def _build_preprocessing_report_text(
    *,
    train_shape: tuple[int, int],
    val_shape: tuple[int, int],
    processed_train_shape: tuple[int, int],
    processed_val_shape: tuple[int, int],
    feature_count: int,
    leakage_notes: list[str],
    missingness_issues: list[str],
    schema_notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Preprocessing Report (Phase 2)\n")
    lines.append("## Dataset overview")
    lines.append(f"- Train raw shape: **{train_shape}**")
    lines.append(f"- Validation raw shape: **{val_shape}**")
    lines.append(f"- Train processed shape: **{processed_train_shape}**")
    lines.append(f"- Validation processed shape: **{processed_val_shape}**")
    lines.append(f"- Final processed feature count: **{feature_count}**\n")

    lines.append("## Cleaning steps performed (based on Phase-1 audit)")
    lines.append("- Missing numeric values (weight, market_index): imputed with **median** (per-column).")
    lines.append("- Categorical whitespace: stripped leading/trailing whitespace.")
    lines.append("- Datetime: extracted **date_year, date_month, date_day**; original `date` dropped.\n")

    lines.append("## Feature engineering performed")
    lines.append("- `date` -> (`date_year`, `date_month`, `date_day`). No other interactions created.\n")

    lines.append("## Encoding decisions")
    lines.append("- Categorical columns: `pickup`, `delivery`, `equipment`")
    lines.append("- Encoding: **OneHotEncoder(handle_unknown='ignore')**\n")

    lines.append("## Scaling decisions")
    lines.append("- `weight`: **RobustScaler**")
    lines.append("- other numeric columns: **StandardScaler**\n")

    lines.append("## Output missingness check")
    if missingness_issues:
        lines.append("- Remaining issues:")
        for i in missingness_issues:
            lines.append(f"  - {i}")
    else:
        lines.append("- None detected: no missing values remain in processed feature matrices.\n")

    lines.append("## Leakage-safety notes")
    for n in leakage_notes:
        lines.append(f"- {n}")
    if not leakage_notes:
        lines.append("- None.\n")

    if schema_notes:
        lines.append("\n## Schema/compatibility notes")
        for n in schema_notes:
            lines.append(f"- {n}")

    lines.append("\n## Risks & recommendations for Phase 3")
    lines.append("- Ensure model training uses the processed feature matrices only; do not re-join raw identifiers unless needed for evaluation.")
    lines.append("- If any categorical missingness appears in later data, document how OneHotEncoder treats NaNs for that case.\n")

    return "\n".join(lines)


def main() -> None:
    paths = Phase2Paths()
    feature_cols = default_feature_columns()

    datasets = load_datasets(train_path="train-test.csv", validation_path="validation.csv")
    train_df_raw = datasets.train
    val_df_raw = datasets.validation

    # Required columns
    ensure_required_columns(
        train_df_raw,
        required=feature_cols.id_cols + feature_cols.categorical_cols + feature_cols.datetime_cols + feature_cols.numeric_cols + [feature_cols.target_col],
        label="train-test",
    )
    ensure_required_columns(
        val_df_raw,
        required=feature_cols.id_cols + feature_cols.categorical_cols + feature_cols.datetime_cols + feature_cols.numeric_cols,
        label="validation",
    )

    # Split inputs/target; keep target out of preprocessing feature matrix
    y_train = train_df_raw[feature_cols.target_col].copy()
    X_train = train_df_raw.drop(columns=[feature_cols.target_col])
    X_val = val_df_raw.copy()

    # Build pipeline and fit on train only
    pipeline, _engineered_names = build_preprocessing_pipeline(
        feature_cols=feature_cols,
        config=PreprocessingPipelineConfig(),
    )

    pipeline.fit(X_train)

    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)

    # Obtain feature names from sklearn ColumnTransformer
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        train_features_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        val_features_df = pd.DataFrame(X_val_processed, columns=feature_names, index=X_val.index)
    except Exception:
        # Fallback: numeric columns with generic names (still valid for shapes)
        train_features_df = pd.DataFrame(X_train_processed, index=X_train.index)
        val_features_df = pd.DataFrame(X_val_processed, index=X_val.index)

        feature_names = list(train_features_df.columns)

    # For train, include target column in processed output for convenience
    train_processed_out = train_features_df.copy()
    train_processed_out[feature_cols.target_col] = y_train.values

    val_processed_out = val_features_df.copy()

    # Write CSVs
    _write_processed_csv(train_processed_out, out_path=paths.processed_dir / "train_processed.csv")
    _write_processed_csv(val_processed_out, out_path=paths.processed_dir / "validation_processed.csv")

    # Validation checks
    missingness_issues: list[str] = []
    missingness_issues.extend(_schema_smoke_check(train_df_raw, train_processed_out.drop(columns=[feature_cols.target_col]), label="train"))
    missingness_issues.extend(_schema_smoke_check(val_df_raw, val_processed_out, label="validation"))

    leakage_notes = [
        "Pipeline fit uses train split only (no use of validation statistics/encodings).",
        "Target `posted_rate` is excluded from preprocessing fit/transform.",
    ]

    schema_notes: list[str] = []
    schema_notes.append(
        "load_id is treated as an identifier and is dropped from the model feature matrix by the preprocessor."
    )

    report_text = _build_preprocessing_report_text(
        train_shape=train_df_raw.shape,
        val_shape=val_df_raw.shape,
        processed_train_shape=train_processed_out.shape,
        processed_val_shape=val_processed_out.shape,
        feature_count=train_features_df.shape[1],
        leakage_notes=leakage_notes,
        missingness_issues=missingness_issues,
        schema_notes=schema_notes,
    )

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.reports_dir / "preprocessing_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote: {report_path}")
    print(f"Wrote: {paths.processed_dir / 'train_processed.csv'}")
    print(f"Wrote: {paths.processed_dir / 'validation_processed.csv'}")


if __name__ == "__main__":
    main()
