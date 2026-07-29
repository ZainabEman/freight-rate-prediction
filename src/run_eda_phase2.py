from __future__ import annotations

from src.config import load_config, set_global_seed
from src.eda import EDAConfig, run_eda_phase2
from src.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """
    Entry point to run Phase 2 EDA from raw training data only.

    Generates:
      - reports/exploratory_data_analysis.md
      - reports/business_insights.md
      - figures/eda/ (all EDA plots)
    """
    app_config = load_config()
    set_global_seed(app_config.random_seed)

    config = EDAConfig(
        train_path=app_config.paths.train,
        figures_dir=app_config.paths.figures_dir / "eda",
        top_k_categories=10,
        outlier_iqr_multiplier=1.5,
    )

    results = run_eda_phase2(config=config)

    reports_dir = app_config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "exploratory_data_analysis.md").write_text(
        results.exploratory_report_markdown, encoding="utf-8"
    )
    (reports_dir / "business_insights.md").write_text(
        results.business_insights_markdown, encoding="utf-8"
    )
    logger.info("Wrote EDA reports to %s", reports_dir)


if __name__ == "__main__":
    main()
