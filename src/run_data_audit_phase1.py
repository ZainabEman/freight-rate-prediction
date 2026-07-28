from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_datasets
from data_profiler import build_profile
from data_validator import validate_all, ValidationIssue


def issues_to_md(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "- None detected."
    lines: list[str] = []
    for i in issues:
        col = i.column if i.column is not None else "(dataset)"
        lines.append(
            "- **[" + i.severity + "]** " + i.issue_type + " | column=" + col + ": " + i.message
        )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(".").resolve()
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(train_path="train-test.csv", validation_path="validation.csv")
    train_df = datasets.train
    val_df = datasets.validation

    train_profile = build_profile(train_df)
    _val_profile = build_profile(val_df)

    audit = validate_all(train_df=train_df, validation_df=val_df, load_id_col="load_id")
    train_issues = audit["train_issues"]
    val_issues = audit["validation_issues"]
    cross_issues = audit["cross_dataset_schema_issues"]

    # Feature dictionary (train)
    rows: list[dict[str, str]] = []
    for fa in train_profile.feature_audits:
        if fa.possible_purpose == "target":
            rec = "Model target"
        elif fa.possible_purpose == "identifier":
            rec = "Use as join key only; usually not a model input"
        elif fa.possible_purpose == "datetime":
            rec = "Use for time features; keep raw until Phase 2"
        elif fa.possible_purpose == "numeric":
            rec = "Numeric feature (validate ranges; scale/transform in Phase 2)"
        elif fa.possible_purpose == "categorical":
            rec = "Categorical feature (encode/clean in Phase 2)"
        else:
            rec = "Unknown"

        rows.append(
            {
                "Feature Name": fa.feature_name,
                "Type": fa.dtype,
                "Description (inferred)": fa.possible_purpose + " feature inferred from name/type.",
                "Missing %": f"{fa.missing_pct:.3f}%",
                "Recommended Usage": rec,
            }
        )

    feature_dictionary_df = pd.DataFrame(rows)
    feature_dictionary_md = (
        "# Feature Dictionary (inferred, Phase 1)\n\n"
        + "Generated from train-test.csv with heuristic descriptions based on inferred datatypes and column names.\n\n"
        + feature_dictionary_df.to_markdown(index=False)
    )

    # Concise summary
    concise = (
        "# Initial Data Audit Report (Phase 1)\n\n"
        + "## Concise summary\n"
        + f"- Training shape: {train_df.shape}\n"
        + f"- Validation shape: {val_df.shape}\n"
        + f"- Number of features (train): {train_df.shape[1]}\n"
        + f"- Target variable (heuristic): {train_profile.inferred_target}\n"
        + " - Data quality issues found: "
        + f"train={len(train_issues)}, validation={len(val_issues)}, cross-schema={len(cross_issues)}\n"
    )

    identifiers = ", ".join(train_profile.inferred_identifier_columns) if train_profile.inferred_identifier_columns else "None detected"

    roles = (
        "## Identified feature groups (heuristic, documented uncertainty)\n"
        + f"- Target column: {train_profile.inferred_target}\n"
        + f"- Identifier columns: {identifiers}\n"
    )

    numeric_md = "## Numerical statistical summary (train)\n" + (
        train_profile.numeric_summary.to_markdown()
        if not train_profile.numeric_summary.empty
        else "- No numeric columns detected."
    )

    categorical_md = "## Categorical statistical summary (train)\n" + (
        train_profile.categorical_summary.reset_index().to_markdown(index=False)
        if not train_profile.categorical_summary.empty
        else "- No categorical/object/string columns detected."
    )

    feat_lines: list[str] = []
    for fa in train_profile.feature_audits:
        obs = "; ".join(fa.observations) if fa.observations else "none"
        feat_lines.append(
            f"- **{fa.feature_name}** | dtype={fa.dtype} | missing={fa.missing_pct:.3f}% | purpose={fa.possible_purpose} | observations={obs}"
        )
    feat_md = "## Inspect every feature individually (train)\n" + "\n".join(feat_lines)

    quality_md = (
        "## Data quality issues\n"
        + "### Training issues\n"
        + issues_to_md(train_issues)
        + "\n\n### Validation issues\n"
        + issues_to_md(val_issues)
        + "\n\n### Cross-dataset schema issues\n"
        + issues_to_md(cross_issues)
    )

    # Risks & recommendations (initial heuristics)
    risks: list[str] = []
    if any(i.issue_type == "numeric" for i in train_issues + val_issues):
        risks.append("Invalid numeric values found; Phase 2 must strictly coerce/validate and handle failures.")
    if any(i.issue_type == "schema" for i in cross_issues):
        risks.append("Train/validation schema mismatch; Phase 2 must align preprocessing and encoding consistently.")
    if any((i.message or "").lower().find("duplicate") >= 0 for i in train_issues + val_issues):
        risks.append("Duplicate records/IDs detected; Phase 2 must decide whether to deduplicate and avoid leakage.")
    if any((i.message or "").lower().find("missing") >= 0 for i in train_issues + val_issues):
        risks.append("Missingness present; Phase 2 must decide imputation strategy or missing-indicator usage.")
    if not risks:
        risks = ["No major data-quality risks detected by heuristics; still re-check edge cases in Phase 2."]

    recommendations = [
        "Validate numeric coercion policy and enforce consistent types across splits.",
        "If categorical values show whitespace/case issues, normalize categories in Phase 2 (fit on training only).",
        "If datetime columns exist, derive safe time features without leaking validation information.",
        "If coordinate-like columns exist, verify ranges and confirm correct units before modeling.",
    ]

    risks_md = (
        "## Potential risks & recommendations (initial)\n"
        + "- Risks identified:\n"
        + "".join(["  - " + r + "\n" for r in risks])
        + "- Recommendations before Phase 2 (EDA):\n"
        + "".join(["  - " + r + "\n" for r in recommendations])
    )

    data_audit_md = "\n".join([concise, roles, numeric_md, categorical_md, feat_md, quality_md, risks_md])

    (reports_dir / "data_audit.md").write_text(data_audit_md, encoding="utf-8")
    (reports_dir / "feature_dictionary.md").write_text(feature_dictionary_md, encoding="utf-8")

    print("Wrote reports/data_audit.md and reports/feature_dictionary.md")


if __name__ == "__main__":
    main()
