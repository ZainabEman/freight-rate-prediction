"""Build the self-contained interactive project dashboard.

Reads artifacts written by Phases 3-8 and renders a single HTML file with all
CSS, JavaScript and figures inlined. No network dependency, no model loading,
no report regeneration.

Run with ``python -m src.build_dashboard``.
"""

from __future__ import annotations

import json

from src.config import load_config
from src.dashboard_assets import CSS, JS
from src.dashboard_data import DashboardData, collect
from src.dashboard_panels import build_sections
from src.dashboard_sections import SECTIONS, esc
from src.logger import get_logger

logger = get_logger(__name__)


def build_nav() -> str:
    """Render the grouped sidebar navigation."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for sid, label, group in SECTIONS:
        groups.setdefault(group, []).append((sid, label))

    parts: list[str] = []
    index = 1
    for group, items in groups.items():
        parts.append(f'<div class="navgroup"><span>{esc(group)}</span></div>')
        for sid, label in items:
            parts.append(
                f'<a href="#{sid}" data-target="{sid}" data-label="{esc(label)}">'
                f'<span class="n">{index:02d}</span><span>{esc(label)}</span></a>'
            )
            index += 1
    return "".join(parts)


def build_payload(data: DashboardData) -> str:
    """Serialise the chart data blob consumed by the inline JavaScript."""
    payload = {
        "kpis": data.kpis,
        "baselines": data.baselines,
        "advanced": data.advanced,
        "tuning": data.tuning,
        "features": data.features,
        "feature_groups": data.feature_groups,
        "december": data.december,
        "monthly": data.monthly,
        "distance_bands": data.distance_bands,
        "equipment": data.equipment,
        "prediction_histogram": data.prediction_histogram,
        "diagnostics": data.diagnostics,
    }
    # Escaping "</" prevents a literal </script> inside the data terminating the
    # surrounding script element.
    return json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")


def build_page(data: DashboardData) -> str:
    """Assemble the complete HTML document."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Freight Rate Prediction &mdash; ML Case Study</title>
<meta name="description" content="Interactive case study: freight rate prediction with CatBoost, \
MAE ${data.kpis['mae']:,.2f} on out-of-sample data.">
<style>{CSS}</style>
</head>
<body>
<div class="shell">
  <nav class="sidebar" aria-label="Sections">
    <div class="brand">
      <h1>Freight Rate Prediction</h1>
      <p>ML Engineer Assessment &middot; Case Study</p>
    </div>
    {build_nav()}
  </nav>
  <div class="main">
    <header class="topbar">
      <div style="display:flex;align-items:center;gap:12px">
        <button class="ctl menu-btn" id="menu" aria-label="Toggle navigation">Menu</button>
        <div class="crumb" id="crumb"><b>Executive Summary</b></div>
      </div>
      <div class="toolbar">
        <span class="badge good">Scorer passed</span>
        <span class="badge accent">MAE ${data.kpis['mae']:,.2f}</span>
        <button class="ctl" id="theme">Dark</button>
      </div>
    </header>
    <main class="content">
{build_sections(data)}
    </main>
  </div>
</div>
<script>window.__DASH__ = {build_payload(data)};</script>
<script>{JS}</script>
</body>
</html>
"""


def main() -> None:
    """Build the dashboard into ``dashboard/index.html``."""
    config = load_config()
    data = collect(config)
    html = build_page(data)

    output = config.paths.data_dir.parent / "dashboard" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info(
        "Wrote dashboard: %s (%.2f MB, %d sections, %d figures)",
        output,
        output.stat().st_size / 1e6,
        len(SECTIONS),
        len(data.figures),
    )


if __name__ == "__main__":
    main()
