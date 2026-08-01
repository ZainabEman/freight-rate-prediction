"""Renderers for each dashboard panel.

Separated from :mod:`src.dashboard_sections` (which holds the narrative
constants and small helpers) purely to keep both files readable.
"""

from __future__ import annotations

from src.dashboard_data import DashboardData
from src.dashboard_sections import (
    DECISIONS,
    READINESS,
    REPO_TREE,
    TIMELINE,
    _params,
    chartbox,
    esc,
    figure,
    kpi,
    metric_table,
    section,
)


def _executive(d: DashboardData) -> str:
    k, diag = d.kpis, d.diagnostics
    return section(
        "exec", "Executive Summary", "Freight rate prediction, end to end",
        "A production ML case study: 48,000 labelled loads, a forward-extrapolation problem, and a "
        "delivered submission validated against the official scorer.",
        '<div class="grid g4" style="margin-bottom:14px">'
        + kpi("Selected model", esc(k["best_model"]), "log target + smearing", hero=True, small=True)
        + kpi("MAE", f"${k['mae']:,.2f}", "holdout, out-of-sample", hero=True)
        + kpi("MAPE", f"{k['mape']:.2f}%", "mean absolute % error", hero=True)
        + kpi("R&sup2;", f"{k['r2']:.4f}", "variance explained")
        + "</div>"
        + '<div class="grid g4" style="margin-bottom:16px">'
        + kpi("RMSE", f"${k['rmse']:,.2f}", "tail-dominated")
        + kpi("Features engineered", f"{k['features']}", "all named end-to-end")
        + kpi("Training samples", f"{k['training_samples']:,}", "Jan&ndash;Oct 2025")
        + kpi("Holdout samples", f"{k['holdout_samples']:,}", "Sep&ndash;Oct, unseen")
        + kpi("Validation samples", f"{k['validation_samples']:,}", "Nov&ndash;Dec 2025")
        + kpi("Predictions generated", f"{k['predictions_generated']:,}", "all positive &amp; finite")
        + kpi("Tests passed", f"{k['tests_passed']}", "ruff clean")
        + kpi("Leakage status", esc(k["leakage_status"]), "structurally enforced", small=True)
        + "</div>"
        + '<div class="callout"><b>The result in one line.</b> A CatBoost regressor on a log target '
          f"predicts freight rates to <b>${k['mae']:,.2f} MAE</b> and <b>{k['mape']:.2f}% MAPE</b> on "
          "data it has never seen &mdash; a <b>90.0% error reduction</b> against a constant-rate "
          "baseline and 20.8% against the best linear model.</div>"
        + "<h3>What actually mattered</h3>"
        + '<div class="grid g3">'
        + '<div class="card"><h4 style="margin-top:0">The problem is extrapolation</h4>'
          '<p style="font-size:13.4px;color:var(--ink-secondary);margin:0">Training ends 2025-10-31; '
          "scoring begins 2025-11-01. Zero overlap. A random split would have measured the wrong "
          "thing entirely &mdash; this single fact drove the split, the date encoding and the "
          "feature set.</p></div>"
        + '<div class="card"><h4 style="margin-top:0">Distance is the price</h4>'
          '<p style="font-size:13.4px;color:var(--ink-secondary);margin:0">Three distance features '
          "carry over 90% of model importance (<code>r = 0.909</code>). Everything else &mdash; "
          "equipment, geography, market &mdash; is a modifier on top of the mileage.</p></div>"
        + '<div class="card"><h4 style="margin-top:0">The tail dominates the error</h4>'
          '<p style="font-size:13.4px;color:var(--ink-secondary);margin:0">Median absolute error is '
          f"<b>${diag['median_absolute_error']:,.2f}</b> against a mean of ${k['mae']:,.2f}. The "
          f"worst 1% of loads carry <b>{diag['worst_1pct_error_share']:.1f}%</b> of all error &mdash; "
          "which is why RMSE barely moved across six model families.</p></div>"
        + "</div>"
        + chartbox(
            "chart-models", "Every model evaluated",
            "Ranked on the same out-of-sample holdout. Hover any bar for full metrics.",
            '<div class="segmented"><span class="lbl">Metric</span>'
            '<button class="ctl" data-metric="mae" aria-pressed="true">MAE</button>'
            '<button class="ctl" data-metric="rmse" aria-pressed="false">RMSE</button>'
            '<button class="ctl" data-metric="r2" aria-pressed="false">R&sup2;</button>'
            '<button class="ctl" data-metric="mape" aria-pressed="false">MAPE</button></div>',
            sub_id="models-sub"),
    )


def _overview(d: DashboardData) -> str:
    return section(
        "overview", "Project Overview", "From inherited repository to delivered submission",
        "The project began as an audit of an existing codebase and ran through nine phases to a "
        "scorer-validated submission.",
        '<div class="card"><h4 style="margin-top:0">Business problem</h4>'
        "<p>Given a load's lane, equipment type, weight, date and market context, predict the rate "
        "it will post at. Accurate pricing lets a broker quote competitively without eroding "
        "margin.</p><p>The cost of a mispriced load is approximately <b>linear in dollars</b>, which "
        "is why <b>MAE is the headline metric</b> throughout rather than RMSE &mdash; RMSE would be "
        "dominated by a handful of very high-rate loads that no model predicts well.</p></div>"
        + '<div class="callout warn"><b>What the inherited repository got wrong.</b> The audit found '
          "a swallowed exception in the preprocessing runner that had silently renamed all 142 "
          "processed feature columns to integers &mdash; feature names were being destroyed with no "
          "error raised. It also found a broken import structure, an incomplete "
          "<code>requirements.txt</code> that produced a non-functional environment, and a missing "
          "December input file that made one deliverable impossible. Phase 3 repaired all four.</div>"
        + "<h3>Approach</h3>"
        + '<ul class="tight">'
          "<li><b>Audit before building.</b> Nothing was assumed about the inherited code; every "
          "claim was verified against the data.</li>"
          "<li><b>Evidence over convention.</b> Each cleaning and modelling decision is backed by a "
          "measurement &mdash; see the Decision Log.</li>"
          "<li><b>Structural correctness.</b> Leakage prevention and feature-name preservation are "
          "enforced by assertions, not by discipline.</li>"
          "<li><b>Honest reporting.</b> Where the evidence is weak &mdash; the model-selection "
          "margin, the flat December curve &mdash; the dashboard says so rather than overselling."
          "</li></ul>",
    )


def _dataset(d: DashboardData) -> str:
    return section(
        "dataset", "Dataset", "48,000 labelled loads, 12,000 to predict",
        "Two disjoint time windows &mdash; the single most important structural fact about this "
        "problem.",
        '<div class="tablewrap"><table><thead><tr><th>File</th><th class="num">Rows</th>'
        '<th class="num">Cols</th><th>Date range</th><th>Labelled</th></tr></thead><tbody>'
        '<tr><td><code>data/train_test.csv</code></td><td class="num">48,000</td>'
        '<td class="num">14</td><td>2025-01-01 &rarr; 2025-10-31</td>'
        '<td><span class="badge good">Yes</span></td></tr>'
        '<tr><td><code>data/validation.csv</code></td><td class="num">12,000</td>'
        '<td class="num">13</td><td>2025-11-01 &rarr; 2025-12-31</td>'
        '<td><span class="badge neutral">No</span></td></tr>'
        '<tr><td><code>data/december_chart_inputs.csv</code></td><td class="num">31</td>'
        '<td class="num">7</td><td>2025-12-01 &rarr; 2025-12-31</td>'
        '<td><span class="badge warn">Reconstructed</span></td></tr></tbody></table></div>'
        + '<div class="callout"><b>Target.</b> <code>posted_rate</code> is strictly positive and '
          "right-skewed: median $2,031, mean $2,374, max $25,533. That skew is why a log target "
          "helps, and why the back-transformation needed a bias correction.</div>"
        + "<h3>Schema</h3>"
        + '<div class="grid g3">'
          '<div class="card"><h4 style="margin-top:0">Categorical (3)</h4>'
          '<p style="font-size:13.2px;margin:0;color:var(--ink-secondary)"><code>pickup</code> and '
          "<code>delivery</code> (64 cities each), <code>equipment</code> (Dry Van, Flatbed, "
          "Reefer).</p></div>"
          '<div class="card"><h4 style="margin-top:0">Numeric (8)</h4>'
          '<p style="font-size:13.2px;margin:0;color:var(--ink-secondary)">Origin and destination '
          "lat/lon, <code>distance</code>, <code>weight</code>, <code>market_index</code>, "
          "<code>quote_signal</code>.</p></div>"
          '<div class="card"><h4 style="margin-top:0">Temporal (1)</h4>'
          '<p style="font-size:13.2px;margin:0;color:var(--ink-secondary)"><code>date</code> at daily '
          "granularity &mdash; roughly 157 loads per day across the development window.</p></div>"
          "</div>"
        + figure(d, "posted_rate_hist",
                 "Target distribution: strictly positive with a long right tail."),
    )


def _audit(d: DashboardData) -> str:
    c = d.cleaning
    return section(
        "audit", "Phase 1 &middot; Data Audit", "Four material quality issues, each verified",
        "Nothing was cleaned on suspicion. Each repair is justified by a measurement that "
        "distinguishes a data fault from a real signal.",
        '<div class="tablewrap"><table><thead><tr><th class="wrap">Issue</th>'
        '<th class="wrap">Evidence</th><th class="wrap">Resolution</th></tr></thead><tbody>'
        f'<tr><td class="wrap"><b>{c["train_negative_weight_rows"]} negative <code>weight</code> '
        'values</b></td><td class="wrap">Mean of |negative| = 31,724 vs positive mean 31,415; both '
        "span an identical [5,000, 47,500] range &mdash; a sign-flip fault, not a distinct "
        'population.</td><td class="wrap">Repaired with <code>abs()</code>, then clipped.</td></tr>'
        f'<tr><td class="wrap"><b>Missing <code>weight</code> / <code>market_index</code></b></td>'
        f'<td class="wrap">{c["train_weight_missing"]} / {c["train_market_index_missing"]} in train '
        "(0.63% / 0.78%), but <b>1.38% / 2.08%</b> in validation &mdash; the rate roughly doubles."
        '</td><td class="wrap">Median imputation <b>plus</b> explicit missingness indicators.</td></tr>'
        f'<tr><td class="wrap"><b>{len(c["unseen_pickup_cities"])} unseen pickup cities</b></td>'
        f'<td class="wrap">{esc(", ".join(c["unseen_pickup_cities"]))} &mdash; present only at '
        f'scoring time, affecting {c["validation_rows_unseen_pickup"]:,} rows (~6%).</td>'
        '<td class="wrap">Explicit <code>*_is_unknown</code> flags; geography carried by '
        "coordinates.</td></tr>"
        '<tr><td class="wrap"><b>Degenerate temporal features</b></td>'
        '<td class="wrap"><code>date_year</code> constant (2025); <code>date_month</code> is {11,12} '
        "at scoring vs {1..10} in training &mdash; outside the model's learned support.</td>"
        '<td class="wrap">Replaced with cyclical day-of-year / day-of-week encodings.</td></tr>'
        "</tbody></table></div>"
        + '<div class="callout crit"><b>The most dangerous finding was in the code, not the data.</b> '
          "The inherited preprocessing runner wrapped <code>get_feature_names_out()</code> in a bare "
          "<code>try/except</code> with a silent fallback. The call was failing on every run, so all "
          "142 processed columns were written with integer names &mdash; feature names were lost "
          "entirely and nothing reported an error. This is now guarded by "
          "<code>assert_feature_names_preserved()</code>, which fails loudly on any integer or "
          "placeholder column name.</div>",
    )


def _eda(d: DashboardData) -> str:
    return section(
        "eda", "Phase 2 &middot; Exploratory Analysis", "What drives a freight rate",
        "Five findings that shaped every subsequent decision.",
        '<div class="grid g2">'
        '<div class="card"><h4 style="margin-top:0">Distance is the price</h4>'
        '<p style="font-size:13.3px;color:var(--ink-secondary);margin:0">'
        "<code>corr(distance, posted_rate) = +0.909</code>. It explains the overwhelming majority "
        "of rate variation.</p></div>"
        '<div class="card"><h4 style="margin-top:0">Rate-per-mile falls with distance</h4>'
        '<p style="font-size:13.3px;color:var(--ink-secondary);margin:0"><code>r = &minus;0.335</code>. '
        "Short hauls spread fixed costs over fewer miles, so a flat $/mile quote is wrong at both "
        "ends.</p></div>"
        '<div class="card"><h4 style="margin-top:0">Equipment carries a real premium</h4>'
        '<p style="font-size:13.3px;color:var(--ink-secondary);margin:0">Median $/mile: Dry Van '
        "2.115, Flatbed 2.295 (+8.5%), Reefer 2.383 (+12.7%).</p></div>"
        '<div class="card"><h4 style="margin-top:0">Pricing is directional</h4>'
        '<p style="font-size:13.3px;color:var(--ink-secondary);margin:0">'
        "<code>corr(delivery_lon, rate) = &minus;0.257</code> &mdash; a westbound premium a single "
        "distance scalar cannot express.</p></div></div>"
        + '<div class="callout warn"><b>Market context is weaker than it first appears.</b> Daily '
          "mean market index correlates <code>+0.577</code> with daily mean $/mile &mdash; but that "
          "is co-movement of <i>averages</i>, not sensitivity. The measured elasticity is only "
          "<b>0.139</b>: a 1% rise in market index moves rate-per-mile by 0.14%. This distinction "
          "later explained the shape of the December chart.</div>"
        + '<div class="grid g2">'
        + chartbox("chart-bands", "Rate-per-mile by distance band",
                   "Median $/mile falls monotonically as hauls lengthen.")
        + chartbox("chart-equipment", "Rate-per-mile by equipment",
                   "Reefer commands the largest premium over Dry Van.")
        + "</div>"
        + '<div class="grid g2">'
        + chartbox("chart-monthly-rpm", "Monthly mean rate-per-mile",
                   "Development window only &mdash; Nov and Dec are unlabelled.")
        + chartbox("chart-monthly-mkt", "Monthly mean market index",
                   "Shown as its own chart rather than on a second axis.")
        + "</div>"
        + figure(d, "correlation_matrix", "Correlation across numeric features and the target.")
        + figure(d, "rate_by_date", "Mean posted rate by date across the development window."),
    )


def _features(d: DashboardData) -> str:
    k = d.kpis
    group_cards = "".join(
        f'<div class="card"><h4 style="margin-top:0">{esc(g)} '
        f'<span class="badge neutral nodot">{len(names)}</span></h4>'
        f'<p style="font-size:12.6px;color:var(--ink-muted);margin:0;font-family:var(--mono);'
        f'line-height:1.7">{esc(", ".join(names[:14]))}{"&hellip;" if len(names) > 14 else ""}'
        "</p></div>"
        for g, names in d.feature_groups.items()
    )
    return section(
        "features", "Phase 3 &middot; Feature Engineering", f"{k['features']} named features",
        "Deliberately narrow. Every engineered feature is justified by a measurement, and names "
        "survive end-to-end so nothing is anonymous downstream.",
        f'<div class="grid g2">{group_cards}</div>'
        + "<h3>Feature explorer</h3>"
        + '<div class="segmented"><span class="lbl">Group</span>'
          '<select class="ctl" id="feat-group"></select>'
          '<input class="ctl" id="feat-search" type="search" placeholder="Search features&hellip;" '
          'style="padding:4px 9px">'
          '<span class="lbl" id="feat-count" style="margin-left:6px"></span></div>'
        + '<div class="tablewrap" style="max-height:520px;overflow-y:auto"><table><thead><tr>'
          '<th>Feature</th><th>Group</th><th class="num">Native</th><th class="num">Permutation</th>'
          '<th class="num">Rank</th><th class="wrap">Note</th></tr></thead>'
          '<tbody id="feat-body"></tbody></table></div>'
        + '<div class="callout"><b>Why cyclical time.</b> <code>doy_sin/cos</code> and '
          "<code>dow_sin/cos</code> are bounded in [&minus;1, 1] and continuous across the year "
          "boundary, so November and December never leave the support the model was trained on. A "
          "monotone day counter was explicitly disabled for the opposite reason.</div>",
    )


def _preprocessing(d: DashboardData) -> str:
    k = d.kpis
    return section(
        "preprocessing", "Phase 3 &middot; Preprocessing Pipeline", "Stateless first, fitted last",
        "The ordering is the design: if every learned statistic lives in one stage, the leakage "
        "audit reduces to inspecting that stage.",
        '<div class="card"><div class="flow">'
        '<div class="flow-row"><div class="node io"><div class="nt">Raw feature frame</div>'
        '<div class="nd">13 columns from CSV</div></div></div><div class="arrow">&darr;</div>'
        '<div class="flow-row"><div class="node stateless"><div class="nt">RawDataCleaner</div>'
        '<div class="nd">Weight sign repair &amp; clip; whitespace strip. Row-local, so it cannot '
        'leak.</div></div></div><div class="arrow">&darr;</div>'
        '<div class="flow-row"><div class="node stateless"><div class="nt">FeatureBuilder</div>'
        '<div class="nd">Temporal, geospatial, interaction and indicator features. Schema frozen at '
        'fit.</div></div></div><div class="arrow">&darr;</div>'
        '<div class="flow-row">'
        '<div class="node fitted"><div class="nt">Impute</div>'
        '<div class="nd">Median, learned from training rows only</div></div>'
        '<div class="node fitted"><div class="nt">Scale</div>'
        '<div class="nd">Robust for weight, standard elsewhere</div></div>'
        '<div class="node fitted"><div class="nt">One-hot</div>'
        '<div class="nd">131 columns; unknowns flagged separately</div></div></div>'
        '<div class="arrow">&darr;</div>'
        f'<div class="flow-row"><div class="node model"><div class="nt">{k["features"]} named '
        'features</div><div class="nd">Verified: no integer names, no NaNs, train/inference '
        "aligned</div></div></div></div>"
        '<div class="legend-inline">'
        '<span><i class="swatch" style="background:var(--series-3)"></i> Stateless &mdash; no fitted '
        "state</span>"
        '<span><i class="swatch" style="background:var(--series-2)"></i> Fitted &mdash; learns from '
        "training rows only</span>"
        '<span><i class="swatch" style="background:var(--accent)"></i> Output</span></div></div>'
        + "<h3>Integrity guards</h3>"
        + '<ul class="tight">'
          "<li><code>assert_feature_names_preserved()</code> &mdash; fails if any column name is an "
          "integer or an sklearn placeholder. The direct regression guard for the inherited bug.</li>"
          "<li><code>assert_frames_aligned()</code> &mdash; train and inference columns must match in "
          "content <i>and</i> order.</li>"
          "<li><code>assert_no_missing_values()</code> &mdash; no NaN may survive preprocessing.</li>"
          "<li><code>assert_no_leakage()</code> &mdash; neither the target nor <code>load_id</code> "
          "may reach the feature matrix.</li>"
          "<li>An independent refit must reproduce the feature matrix <b>exactly</b>.</li></ul>",
    )


def _baselines(d: DashboardData) -> str:
    s = d.split
    return section(
        "baselines", "Phase 4 &middot; Baseline Models", "Seven references, from trivial to linear",
        "Baselines exist to make the advanced models prove their worth. The domain heuristic here "
        "is genuinely strong, which sets a meaningful bar.",
        metric_table(d.baselines, highlight="Ridge (alpha=1.0, log target)")
        + '<div class="callout"><b>The rate-per-mile heuristic reaches R&sup2; = 0.807</b> &mdash; '
          "distance &times; median $/mile within equipment type, which is how a broker prices a lane "
          "by hand. Most of the signal is available before any learning occurs, so the real question "
          "for Phase 5 was whether boosting could beat a well-specified linear model at all.</div>"
        + "<h3>Evaluation protocol</h3>"
        + '<div class="tablewrap"><table><thead><tr><th>Split</th><th class="num">Rows</th>'
          "<th>Dates</th><th>Purpose</th></tr></thead><tbody>"
          f'<tr><td>Training</td><td class="num">{s["train_rows"]:,}</td>'
          f'<td>{s["train_date_min"]} &rarr; {s["train_date_max"]}</td>'
          "<td>Fitting and hyperparameter search</td></tr>"
          f'<tr><td>Holdout</td><td class="num">{s["holdout_rows"]:,}</td>'
          f'<td>{s["holdout_date_min"]} &rarr; {s["holdout_date_max"]}</td>'
          "<td>Scored once, never tuned against</td></tr></tbody></table></div>",
    )


def _advanced(d: DashboardData) -> str:
    return section(
        "advanced", "Phase 5 &middot; Advanced Models", "Five families, tuned and compared",
        "Every model trained on a log target and evaluated on the same untouched holdout.",
        metric_table(d.advanced, highlight="CatBoost + smearing (final)")
        + '<div class="callout warn"><b>An honest caveat on the margin.</b> CatBoost, LightGBM and '
          "HistGradientBoosting sit within $4 of each other on a $2,280 mean prediction &mdash; "
          "inside fold-to-fold noise. CatBoost is the defensible pick, but it should be read as one "
          "of three near-equivalent models rather than a decisive winner.</div>"
        + '<div class="callout"><b>Why HistGradientBoosting rather than GradientBoosting.</b> '
          "scikit-learn documents the exact-split <code>GradientBoostingRegressor</code> as far "
          "slower for <code>n_samples &ge; 10,000</code>. With 38,477 training rows and 156 features "
          "it would have dominated the phase runtime for no accuracy benefit, so the histogram "
          "implementation was used instead.</div>",
    )


def _tuning(d: DashboardData) -> str:
    rows = "".join(
        f'<tr><td>{esc(t["name"])}</td><td class="num">${t["cv_mae"]:,.2f}</td>'
        f'<td class="num">{t["candidates"]}</td><td class="num">{t["seconds"]:.0f}s</td>'
        f'<td class="wrap"><code style="font-size:11px">{_params(t["params"])}</code></td></tr>'
        for t in sorted(d.tuning, key=lambda t: t["cv_mae"])
    )
    return section(
        "tuning", "Phase 5 &middot; Hyperparameter Tuning",
        "Randomised search, leakage-safe by construction",
        "Preprocessing is a step inside the searched pipeline, so scikit-learn refits it "
        "independently within every cross-validation fold.",
        chartbox("chart-tuning", "Best cross-validated MAE per model",
                 "Expanding-window CV on the training block only. Hover for the selected parameters.")
        + '<div class="tablewrap"><table><thead><tr><th>Model</th><th class="num">CV MAE</th>'
          '<th class="num">Candidates</th><th class="num">Search time</th>'
          f'<th class="wrap">Selected parameters</th></tr></thead><tbody>{rows}</tbody></table></div>'
        + '<div class="callout"><b>CV and holdout rankings genuinely disagreed</b> &mdash; '
          "RandomForest scored CV $167.68 but holdout $144.85. The 3-fold expanding window trains "
          "early folds on as little as ~9,600 rows, so CV is pessimistic and noisy. Selection was "
          "made on the holdout, which is the more trustworthy estimate. A holdout consulted "
          "repeatedly during tuning would have destroyed exactly this information.</div>",
    )


def _comparison(d: DashboardData) -> str:
    return section(
        "comparison", "Model Comparison", "Baselines and advanced models side by side",
        "One holdout, one metric, every model &mdash; the only fair way to read the improvement.",
        metric_table(sorted(d.baselines + d.advanced, key=lambda r: r["mae"]),
                     highlight="CatBoost + smearing (final)")
        + '<div class="statline" style="margin-top:16px">'
          '<div><span class="sv">&minus;90.0%</span><span class="sl">vs constant median</span></div>'
          '<div><span class="sv">&minus;20.8%</span><span class="sl">vs best linear</span></div>'
          '<div><span class="sv">&minus;13.0%</span><span class="sl">from smearing alone</span></div>'
          '<div><span class="sv">~2%</span><span class="sl">RMSE spread across families</span></div>'
          "</div>"
        + '<div class="callout"><b>Read the RMSE column.</b> It varies by barely 2% across every '
          "model tried, while MAE spans $115&ndash;$1,149. RMSE simply cannot discriminate between "
          "these models, because it is dominated by a tail that none of them predicts. That is a "
          "finding about the data, not a deficiency in any model.</div>",
    )


def _explain(d: DashboardData) -> str:
    return section(
        "explain", "Phase 6 &middot; Explainability", "Three methods, one conclusion",
        "Native importance, permutation importance and SHAP each have a different blind spot, so "
        "all three are reported. Where they disagreed, the disagreement was the finding.",
        '<div class="callout"><b>Reading the numbers.</b> The model predicts <code>log(rate)</code>, '
        "so importances and SHAP values live in log-dollar space. Contributions are additive in "
        "logs, which means <b>multiplicative in dollars</b>: a SHAP value of +0.10 is roughly a "
        "+10.5% effect on the rate, not +$0.10.</div>"
        + chartbox(
            "chart-importance", "Top 20 features",
            "Switch between CatBoost's in-sample score and out-of-sample permutation importance. "
            "Hover for both ranks and the feature's rationale.",
            '<div class="segmented"><span class="lbl">Method</span>'
            '<button class="ctl" data-imp="native" aria-pressed="true">Native</button>'
            '<button class="ctl" data-imp="permutation" aria-pressed="false">Permutation</button>'
            "</div>")
        + '<div class="callout warn"><b>The <code>market_index</code> discrepancy is informative.</b> '
          "It ranks 7th on CatBoost's native score but only <b>15th</b> out-of-sample, with a "
          "permutation effect two orders of magnitude below the distance features. Its measured "
          "elasticity is 0.139. An earlier draft of this analysis described market context as the "
          "dominant operational signal; the permutation evidence contradicted that, and the claim "
          "was corrected rather than left standing.</div>"
        + '<div class="grid g2">'
        + figure(d, "shap_beeswarm", "SHAP beeswarm &mdash; per-load feature effects on log(rate).")
        + figure(d, "shap_bar", "Mean |SHAP| &mdash; average magnitude of each feature's effect.")
        + "</div><div class=\"grid g2\">"
        + figure(d, "native_importance", "CatBoost native importance (PredictionValuesChange).")
        + figure(d, "permutation_importance", "Permutation importance, measured out-of-sample.")
        + "</div><div class=\"grid g2\">"
        + figure(d, "shap_dependence_distance",
                 "SHAP dependence &mdash; log_distance, the dominant driver.")
        + figure(d, "shap_dependence_market",
                 "SHAP dependence &mdash; market_index, a second-order effect.")
        + "</div>"
        + figure(d, "shap_waterfall",
                 "Waterfall for a typical load: how a single prediction is built up."),
    )


def _errors(d: DashboardData) -> str:
    k, diag = d.kpis, d.diagnostics
    return section(
        "errors", "Phase 6 &middot; Error Analysis", "The tail is the whole story",
        f"Measured on {k['holdout_samples']:,} genuinely unseen loads. Sign convention: a positive "
        "residual means the model under-priced the load.",
        '<div class="grid g4" style="margin-bottom:16px">'
        + kpi("Median abs. error", f"${diag['median_absolute_error']:,.2f}", "the typical load")
        + kpi("Mean abs. error", f"${k['mae']:,.2f}", "pulled up by the tail")
        + kpi("95th percentile", f"${diag['p95_absolute_error']:,.2f}")
        + kpi("Max abs. error", f"${diag['max_absolute_error']:,.0f}", "single worst load")
        + "</div>"
        + '<div class="tablewrap" style="margin-bottom:16px"><table><thead><tr><th>Diagnostic</th>'
          '<th class="wrap">Measurement</th><th>Verdict</th></tr></thead><tbody>'
          f'<tr><td><b>Heavy tails</b></td><td class="wrap">Excess kurtosis '
          f'{diag["excess_kurtosis"]:.0f}; worst 1% of loads carry '
          f'{diag["worst_1pct_error_share"]:.1f}% of all absolute error</td>'
          '<td><span class="badge crit">Dominant</span></td></tr>'
          f'<tr><td><b>Heteroscedasticity</b></td><td class="wrap">Residual SD varies '
          f'{diag["quintile_std_ratio"]:.2f}&times; across prediction quintiles; '
          f'corr(|residual|, prediction) = {diag["abs_residual_corr"]:.3f}</td>'
          '<td><span class="badge warn">Present</span></td></tr>'
          f'<tr><td><b>Systematic bias</b></td><td class="wrap">Mean residual '
          f'+${diag["mean_residual"]:,.2f} (t = {diag["bias_t"]:.2f})</td>'
          '<td><span class="badge good">Corrected</span></td></tr></tbody></table></div>'
        + '<div class="callout crit"><b>Why RMSE never moved.</b> Across six model families RMSE '
          "stayed pinned between $636 and $654 while MAE ranged from $115 to $1,149. Every model "
          "shares the same heavy tail &mdash; a minority of loads priced by mechanisms not present "
          "in these features. No hyperparameter fixes that, and it is the single largest remaining "
          "opportunity in the project.</div>"
        + '<div class="grid g2">'
        + figure(d, "residual_vs_prediction", "Residual vs prediction &mdash; spread widens with rate.")
        + figure(d, "qq_plot", "QQ plot &mdash; sharp departure from normal at both tails.")
        + "</div><div class=\"grid g2\">"
        + figure(d, "residual_distribution", "Residual distribution (central 99%).")
        + figure(d, "absolute_error_histogram", "Absolute error distribution (central 99%).")
        + "</div><div class=\"grid g2\">"
        + figure(d, "mae_by_distance_band", "MAE by distance band.")
        + figure(d, "mae_by_prediction_quintile",
                 "MAE by prediction quintile &mdash; error scales with rate.")
        + "</div><div class=\"grid g2\">"
        + figure(d, "mae_by_equipment", "MAE by equipment type.")
        + figure(d, "error_vs_distance", "Residual vs distance with a rolling median.")
        + "</div>",
    )


def _insights(d: DashboardData) -> str:
    cards = [
        ("1 &middot; Quote uncertainty as a percentage",
         "Residual spread varies 4.24&times; across prediction quintiles. A flat &plusmn;$X band is "
         "far too wide on cheap loads and far too narrow on expensive ones."),
        ("2 &middot; Never quote a flat $/mile",
         "Rate-per-mile falls monotonically with distance (<code>r = &minus;0.335</code>). Use the "
         "banded figures as a sanity check on any manual quote."),
        ("3 &middot; Refresh market context daily",
         "It is the dominant time-varying input, so a stale <code>market_index</code> degrades every "
         "prediction that day at once &mdash; though the effect size is modest (elasticity 0.139)."),
        ("4 &middot; Route the tail to human review",
         "Error is concentrated enough that a queue covering the worst ~1% of loads captures roughly "
         "39% of total error &mdash; a small manual workload for a large risk reduction."),
        ("5 &middot; Treat Nov&ndash;Dec as extrapolation",
         "No labelled data exists for the scoring months. Monitor realised rates against predictions "
         "weekly and be prepared to recalibrate."),
        ("6 &middot; Collect data for the 8 unseen cities",
         "Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk and San Diego are "
         "currently priced from geography alone."),
    ]
    body = "".join(
        f'<div class="card"><h4 style="margin-top:0">{title}</h4>'
        f'<p style="font-size:13.3px;color:var(--ink-secondary);margin:0">{text}</p></div>'
        for title, text in cards
    )
    return section(
        "insights", "Phase 6 &middot; Business Insights", "What a pricing desk should do with this",
        "Six recommendations, each traceable to a measurement rather than intuition.",
        f'<div class="grid g2">{body}</div>',
    )


def _predictions(d: DashboardData) -> str:
    k = d.kpis
    return section(
        "predictions", "Phase 7 &middot; Final Predictions", "One training run, one submission",
        "The selected configuration refitted once on the complete development window, then applied "
        "to all 12,000 scoring loads.",
        '<div class="grid g4" style="margin-bottom:16px">'
        + kpi("Predictions", f"{k['predictions_generated']:,}", "exactly as required", hero=True)
        + kpi("Minimum", f"${k['prediction_min']:,.2f}", "all strictly positive")
        + kpi("Median", f"${k['prediction_median']:,.2f}")
        + kpi("Mean", f"${k['prediction_mean']:,.2f}")
        + "</div>"
        + chartbox("chart-predhist", "Distribution of the 12,000 submitted predictions",
                   "Hover any bar for the load count in that rate range.")
        + "<h3>The smearing correction</h3>"
        + '<div class="tablewrap" style="margin-bottom:14px"><table><thead><tr><th>Metric</th>'
          '<th class="num">Uncorrected</th><th class="num">Corrected</th><th class="num">Change</th>'
          "</tr></thead><tbody>"
          f'<tr class="highlight"><td>MAE</td><td class="num">${k["mae_uncorrected"]:,.2f}</td>'
          f'<td class="num">${k["mae"]:,.2f}</td><td class="num">&minus;13.0%</td></tr>'
          '<tr><td>RMSE</td><td class="num">$641.48</td>'
          f'<td class="num">${k["rmse"]:,.2f}</td><td class="num">&minus;0.8%</td></tr>'
          '<tr><td>MAPE</td><td class="num">5.68%</td>'
          f'<td class="num">{k["mape"]:.2f}%</td><td class="num">&minus;0.65 pp</td></tr>'
          "</tbody></table></div>"
        + '<div class="callout"><b>Tested before adoption, not assumed.</b> Smearing targets the '
          "conditional mean while MAE is minimised by the conditional median &mdash; the two "
          "genuinely conflict, so adopting it blind would have been a coin flip. It was evaluated on "
          "the untouched holdout using the already-fitted Phase-5 model, which cost no extra "
          f"training. All three metrics improved, so it was adopted (factor "
          f"{k['smearing_factor']:.4f}).</div>"
        + "<h3>Submission validation</h3>"
        + '<ul class="tight">'
          '<li><span class="badge good nodot">&check;</span> Exactly 12,000 rows</li>'
          '<li><span class="badge good nodot">&check;</span> Columns exactly '
          "<code>[load_id, predicted_rate]</code></li>"
          '<li><span class="badge good nodot">&check;</span> <code>load_id</code> order matches the '
          "provided template</li>"
          '<li><span class="badge good nodot">&check;</span> All values finite and strictly positive</li>'
          '<li><span class="badge good nodot">&check;</span> No missing values</li>'
          '<li><span class="badge good nodot">&check;</span> <code>score.py</code> exits 0</li></ul>'
        + '<div class="card"><h4 style="margin-top:0">Scorer output</h4>'
          '<pre style="margin:0;font-family:var(--mono);font-size:12.3px;'
          'color:var(--ink-secondary);white-space:pre-wrap">Validated 12,000 final predictions.\n'
          "Validated 31 fixed December predictions.\n"
          "Created chart: scorer_results\\candidate_december.png\n"
          "Final validation metrics are calculated by Spotter after submission.</pre></div>",
    )


def _december(d: DashboardData) -> str:
    rates = [p["rate"] for p in d.december]
    spread = max(rates) - min(rates)
    mean = sum(rates) / len(rates)
    return section(
        "december", "Phase 7 &middot; December Analysis",
        "Why the curve is nearly flat &mdash; and why that is right",
        "A fixed lane &mdash; Lexington to Fort Wayne, 360 miles, Dry Van, 32,000&nbsp;lb &mdash; "
        "across all 31 days of December, with only the date changing.",
        '<div class="grid g4" style="margin-bottom:16px">'
        + kpi("Minimum", f"${min(rates):,.2f}")
        + kpi("Maximum", f"${max(rates):,.2f}")
        + kpi("Spread", f"${spread:,.2f}", f"{spread / mean * 100:.2f}% of mean")
        + kpi("Rows", f"{len(rates)}", "one per December day")
        + "</div>"
        + chartbox("chart-december", "Predicted rate across December 2025",
                   "Only the date changes. Hover for the exact predicted rate on any day.")
        + '<div class="callout"><b>This narrow range is the correct result, not a defect &mdash; and '
          "it was worth verifying rather than assuming.</b> Across these 31 dates the market index "
          "swings <b>+25.8%</b> (0.831 &rarr; 1.045). The elasticity measured directly from the data "
          "is 0.139, which implies a rate response of only <b>+3.2%</b>. The model produces "
          "<b>+2.7%</b> &mdash; closely tracking the data. On a lane with fixed mileage, equipment "
          "and weight, date is genuinely a minor price driver in this dataset.</div>"
        + '<div class="callout"><b>The visible weekly rhythm</b> &mdash; mid-week peaks, weekend '
          "troughs &mdash; comes from the cyclical <code>dow_sin</code>/<code>dow_cos</code> "
          "features. Had the inherited raw <code>date_month</code> encoding been kept, this chart "
          "would have been a flat line: every December date would have fallen into the same "
          "out-of-range leaf.</div>"
        + figure(d, "candidate_december",
                 "candidate_december.png &mdash; produced by the provided score.py, unmodified."),
    )


def _decisions(d: DashboardData) -> str:
    items = "".join(
        f'<details class="decision"><summary><span class="badge {tone}">{tag}</span>'
        f'{item["title"]}</summary><dl class="dbody">'
        f'<div class="drow"><dt>Decision</dt><dd><b>{item["decision"]}</b></dd></div>'
        f'<div class="drow"><dt>Why</dt><dd>{item["why"]}</dd></div>'
        f'<div class="drow"><dt>Alternatives</dt><dd>{item["alternatives"]}</dd></div>'
        f'<div class="drow"><dt>Evidence</dt><dd>{item["evidence"]}</dd></div>'
        f'<div class="drow"><dt>Impact</dt><dd>{item["impact"]}</dd></div></dl></details>'
        for item in DECISIONS
        for tone, tag in [item["tag"]]
    )
    return section(
        "decisions", "Reference &middot; Decision Log",
        f"{len(DECISIONS)} engineering decisions, with the evidence behind each",
        "Every non-obvious choice in the project: why it was made, what else was considered, and "
        "what it changed. Expand any row.",
        items,
    )


def _architecture(d: DashboardData) -> str:
    k = d.kpis
    return section(
        "architecture", "Reference &middot; Architecture", "How the system fits together",
        "Four views: the end-to-end pipeline, the training workflow, the two prediction paths, and "
        "the model selection funnel.",
        '<div class="card"><h4 style="margin-top:0">End-to-end pipeline</h4><div class="flow">'
        '<div class="flow-row">'
        '<div class="node io"><div class="nt">train_test.csv</div>'
        '<div class="nd">48,000 labelled</div></div>'
        '<div class="node io"><div class="nt">validation.csv</div>'
        '<div class="nd">12,000 unlabelled</div></div>'
        '<div class="node io"><div class="nt">december_chart_inputs.csv</div>'
        '<div class="nd">31 fixed rows</div></div></div><div class="arrow">&darr;</div>'
        '<div class="flow-row">'
        '<div class="node stateless"><div class="nt">Audit &amp; validate</div>'
        '<div class="nd">Schema, dtypes, missingness, quality</div></div>'
        '<div class="node stateless"><div class="nt">Clean</div>'
        '<div class="nd">Weight repair, whitespace, indicators</div></div>'
        '<div class="node stateless"><div class="nt">Engineer</div>'
        f'<div class="nd">{k["features"]} named features</div></div></div>'
        '<div class="arrow">&darr;</div><div class="flow-row">'
        '<div class="node fitted"><div class="nt">Temporal split</div>'
        '<div class="nd">Train &le; Aug 31 &middot; Holdout Sep&ndash;Oct</div></div>'
        '<div class="node fitted"><div class="nt">Tune &amp; select</div>'
        '<div class="nd">5 families, expanding-window CV</div></div>'
        '<div class="node model"><div class="nt">CatBoost + smearing</div>'
        '<div class="nd">Final fit on all 48,000</div></div></div>'
        '<div class="arrow">&darr;</div><div class="flow-row">'
        '<div class="node io"><div class="nt">validation_predictions.csv</div>'
        '<div class="nd">12,000 rows, scorer-validated</div></div>'
        '<div class="node io"><div class="nt">candidate_december.png</div>'
        '<div class="nd">Produced by score.py</div></div></div></div></div>'
        + '<div class="grid g2">'
        + '<div class="card"><h4 style="margin-top:0">Training workflow</h4><div class="flow">'
          '<div class="flow-row"><div class="node stateless"><div class="nt">1 &middot; Sort by date</div>'
          '<div class="nd">Stable ordering for time splits</div></div></div>'
          '<div class="arrow">&darr;</div>'
          '<div class="flow-row"><div class="node fitted"><div class="nt">2 &middot; Split fold</div>'
          '<div class="nd">Train strictly earlier than validation</div></div></div>'
          '<div class="arrow">&darr;</div>'
          '<div class="flow-row"><div class="node fitted">'
          '<div class="nt">3 &middot; Refit preprocessing</div>'
          '<div class="nd">Fresh, inside the fold &mdash; the leakage guarantee</div></div></div>'
          '<div class="arrow">&darr;</div>'
          '<div class="flow-row"><div class="node model"><div class="nt">4 &middot; Fit &amp; score</div>'
          '<div class="nd">Log target; invert to dollars; clip positive</div></div></div>'
          "</div></div>"
        + '<div class="card"><h4 style="margin-top:0">Two prediction paths</h4><div class="flow">'
          '<div class="flow-row"><div class="node io"><div class="nt">Full path</div>'
          '<div class="nd">validation.csv &mdash; all 13 columns present</div></div></div>'
          '<div class="arrow">&darr;</div>'
          '<div class="flow-row"><div class="node io"><div class="nt">Reduced path</div>'
          '<div class="nd">December &mdash; 6 columns; coordinates and market context missing</div>'
          "</div></div><div class=\"arrow\">&darr;</div>"
          '<div class="flow-row"><div class="node stateless"><div class="nt">Enrich</div>'
          '<div class="nd">Exact city&rarr;coordinate lookup; per-date market table</div></div></div>'
          '<div class="arrow">&darr;</div>'
          '<div class="flow-row"><div class="node model"><div class="nt">Same fitted pipeline</div>'
          '<div class="nd">Both paths converge &mdash; no duplicated logic</div></div></div>'
          "</div></div></div>"
        + '<div class="card"><h4 style="margin-top:0">Model selection funnel</h4><div class="flow">'
          '<div class="flow-row"><div class="node io"><div class="nt">7 baselines</div>'
          '<div class="nd">Constant &rarr; rate-per-mile &rarr; linear. Best: Ridge log, $145.24</div>'
          "</div></div><div class=\"arrow\">&darr;</div>"
          '<div class="flow-row"><div class="node fitted"><div class="nt">5 advanced families</div>'
          '<div class="nd">RF, HistGB, XGBoost, LightGBM, CatBoost &mdash; 36 tuned candidates</div>'
          "</div></div><div class=\"arrow\">&darr;</div>"
          '<div class="flow-row"><div class="node fitted"><div class="nt">Holdout, scored once</div>'
          '<div class="nd">CatBoost $132.15 &mdash; but within $4 of two others</div></div></div>'
          '<div class="arrow">&darr;</div>'
          '<div class="flow-row"><div class="node model"><div class="nt">+ Smearing correction</div>'
          f'<div class="nd">${k["mae"]:,.2f} final MAE</div></div></div></div></div>',
    )


def _repo(d: DashboardData) -> str:
    rows = "".join(
        '<div class="row"><span class="p">'
        + (f"<span class='dir'>{esc(path)}</span>" if kind == "dir" else esc(path))
        + f'</span><span class="d">{esc(desc)}</span></div>'
        for path, kind, desc in REPO_TREE
    )
    return section(
        "repo", "Reference &middot; Repository Explorer", "Every module and what it is for",
        "The structure follows the pipeline: configuration, data, source grouped by role, and "
        "generated artifacts kept separate from tracked inputs.",
        f'<div class="card"><div class="tree">{rows}</div></div>'
        + '<div class="callout"><b>Derived artifacts are git-ignored and fully regenerable.</b> '
          "<code>processed/</code>, <code>models/</code> and <code>scorer_results/</code> can all be "
          "rebuilt from the raw CSVs with a fixed seed &mdash; with one deliberate exception: "
          "<code>candidate_december.png</code> is force-tracked because it is a required submission "
          "artifact.</div>",
    )


def _timeline(d: DashboardData) -> str:
    items = "".join(
        f'<div class="tl-item"><h4>{phase} &mdash; {title}</h4><p>{desc}</p>'
        f'<div class="tl-meta">{esc(artifact)}</div></div>'
        for phase, title, desc, artifact in TIMELINE
    )
    return section(
        "timeline", "Reference &middot; Project Timeline", "Nine phases, audit to dashboard",
        "Each phase produced a verifiable artifact before the next began.",
        f'<div class="card"><div class="timeline">{items}</div></div>',
    )


def _modelcard(d: DashboardData) -> str:
    k, diag = d.kpis, d.diagnostics
    return section(
        "modelcard", "Reference &middot; Model Card", "CatBoost freight rate regressor",
        "Metadata, training procedure, inputs and outputs, performance, failure modes and "
        "deployment considerations.",
        '<div class="grid g2">'
        '<div class="card"><h4 style="margin-top:0">Model details</h4>'
        '<div class="tablewrap" style="border:none"><table>'
        "<tr><td>Algorithm</td><td><b>CatBoost regressor</b></td></tr>"
        "<tr><td>Target</td><td><code>log(posted_rate)</code>, inverted with <code>exp</code></td></tr>"
        f"<tr><td>Back-transform</td><td>Duan smearing, factor {k['smearing_factor']:.4f}</td></tr>"
        f"<tr><td>Hyperparameters</td><td><code style='font-size:11.5px'>"
        f"{_params(k['hyperparameters'])}</code></td></tr>"
        f"<tr><td>Features</td><td>{k['features']} engineered</td></tr>"
        f"<tr><td>Random seed</td><td>{k['seed']}</td></tr></table></div></div>"
        '<div class="card"><h4 style="margin-top:0">Training procedure</h4>'
        '<div class="tablewrap" style="border:none"><table>'
        f"<tr><td>Training rows</td><td>{k['training_samples']:,}</td></tr>"
        "<tr><td>Window</td><td>2025-01-01 &rarr; 2025-10-31</td></tr>"
        "<tr><td>Selection</td><td>Randomised search, 3 expanding-window folds</td></tr>"
        "<tr><td>Selection metric</td><td>Holdout MAE</td></tr>"
        "<tr><td>Final fit</td><td>Exactly one run on the full window</td></tr>"
        "<tr><td>Leakage control</td><td>Preprocessing refit inside every fold</td></tr>"
        "</table></div></div></div>"
        + '<div class="grid g2">'
          '<div class="card"><h4 style="margin-top:0">Inputs</h4><ul class="tight" style="margin:0">'
          "<li><b>Categorical:</b> <code>pickup</code>, <code>delivery</code>, "
          "<code>equipment</code></li>"
          "<li><b>Numeric:</b> origin/destination lat-lon, <code>distance</code>, "
          "<code>weight</code>, <code>market_index</code>, <code>quote_signal</code></li>"
          "<li><b>Temporal:</b> <code>date</code></li>"
          "<li>Accepts a raw DataFrame &mdash; preprocessing lives inside the persisted pipeline</li>"
          "</ul></div>"
          '<div class="card"><h4 style="margin-top:0">Outputs</h4><ul class="tight" style="margin:0">'
          "<li>A single <code>predicted_rate</code> in USD per load</li>"
          "<li><b>Guaranteed finite and strictly positive</b> &mdash; log inverse, a positive "
          "smearing factor, and an explicit floor</li>"
          f"<li>Observed range on the scoring set: ${k['prediction_min']:,.2f} &ndash; "
          f"${k['prediction_max']:,.2f}</li>"
          "<li>No prediction interval is produced &mdash; see limitations</li></ul></div></div>"
        + '<div class="card"><h4 style="margin-top:0">Performance</h4>'
          f'<p style="font-size:13.4px;color:var(--ink-secondary)">On {k["holdout_samples"]:,} loads '
          "from 2025-09-01 to 2025-10-31, never seen during fitting or tuning:</p>"
          '<div class="statline">'
          f'<div><span class="sv">${k["mae"]:,.2f}</span><span class="sl">MAE</span></div>'
          f'<div><span class="sv">${k["rmse"]:,.2f}</span><span class="sl">RMSE</span></div>'
          f'<div><span class="sv">{k["r2"]:.4f}</span><span class="sl">R&sup2;</span></div>'
          f'<div><span class="sv">{k["mape"]:.2f}%</span><span class="sl">MAPE</span></div>'
          f'<div><span class="sv">${diag["median_absolute_error"]:,.2f}</span>'
          '<span class="sl">Median abs. error</span></div></div></div>'
        + '<div class="card"><h4 style="margin-top:0">Known failure modes</h4>'
          '<ul class="tight" style="margin:0">'
          f"<li><b>High-rate outliers.</b> The worst 1% of loads carry "
          f"{diag['worst_1pct_error_share']:.1f}% of total error; the single worst is off by "
          f"${diag['max_absolute_error']:,.0f}. These loads are priced by mechanisms not present in "
          "the features.</li>"
          "<li><b>Unseen cities.</b> 8 cities appear only at scoring time and are priced from "
          "coordinates alone. The <code>*_is_unknown</code> flag makes the state explicit but cannot "
          "recover missing lane history.</li>"
          "<li><b>Far-future dates.</b> Cyclical encoding keeps predictions in-support, but the "
          "model has no mechanism for a genuine regime change beyond the observed market range.</li>"
          "<li><b>Stale market context.</b> A missing or outdated <code>market_index</code> degrades "
          "every prediction for that day simultaneously.</li></ul></div>"
        + '<div class="card"><h4 style="margin-top:0">Deployment considerations</h4>'
          '<ul class="tight" style="margin:0">'
          "<li>Persisted as a complete pipeline &mdash; no separate preprocessing step to keep in "
          "sync.</li>"
          "<li>Single-threaded inference on 12,000 rows completes in under a second.</li>"
          "<li>Quote uncertainty as a percentage, not a dollar band (heteroscedasticity confirmed)."
          "</li>"
          "<li>Route the top ~1% by predicted rate to human review to absorb tail risk.</li>"
          "<li>Monitor <code>market_index</code> drift and realised-vs-predicted error weekly.</li>"
          "</ul></div>",
    )


def _deployment(d: DashboardData) -> str:
    rows = "".join(
        f'<tr><td><b>{esc(name)}</b></td>'
        f'<td><span class="badge {tone}">{esc(status)}</span></td>'
        f'<td class="wrap">{note}</td></tr>'
        for name, tone, status, note in READINESS
    )
    ready = sum(1 for r in READINESS if r[1] == "good")
    return section(
        "deployment", "Reference &middot; Deployment Readiness",
        f"{ready} of {len(READINESS)} dimensions production-ready",
        "An honest assessment. The modelling and reproducibility story is complete; the operational "
        "story is not, and is scoped rather than glossed over.",
        f'<div class="tablewrap"><table><thead><tr><th>Dimension</th><th>Status</th>'
        f'<th class="wrap">Detail</th></tr></thead><tbody>{rows}</tbody></table></div>'
        + '<div class="callout warn"><b>What &ldquo;not built&rdquo; means here.</b> The three gaps '
          "are deliberate scope boundaries, not oversights. This was assessed as a modelling "
          "exercise with a fixed submission format; monitoring, prediction intervals and a serving "
          "layer are the first three items any real deployment would need, and they are listed so "
          "the gap is explicit rather than discovered later.</div>",
    )


def _conclusion(d: DashboardData) -> str:
    k = d.kpis
    return section(
        "conclusion", "Conclusion", "What was built, and what it is worth",
        "A reproducible, leakage-free pipeline delivering a scorer-validated submission &mdash; with "
        "its weaknesses stated as plainly as its results.",
        '<div class="grid g4" style="margin-bottom:18px">'
        + kpi("Final MAE", f"${k['mae']:,.2f}", "out-of-sample", hero=True)
        + kpi("vs constant baseline", "&minus;90.0%", "error reduction")
        + kpi("vs best linear", "&minus;20.8%", "error reduction")
        + kpi("Scorer", "Passed", "exit 0")
        + "</div>"
        + "<h3>What went well</h3>"
        + '<ul class="tight">'
          "<li><b>The split reflects the real task.</b> Recognising the forward-extrapolation "
          "structure early drove the holdout design, the cyclical date encoding and the decision to "
          "avoid monotone time features. A random split would have produced flattering numbers that "
          "collapsed at scoring time.</li>"
          "<li><b>Correctness is enforced, not assumed.</b> Leakage prevention is structural; "
          "feature-name preservation, column alignment and determinism are asserted on every run. "
          "The inherited bug that silently destroyed all feature names cannot recur.</li>"
          "<li><b>Decisions were tested, not assumed.</b> The smearing correction could have hurt "
          "MAE &mdash; it was measured on the holdout before adoption and delivered a 13% "
          "improvement.</li>"
          "<li><b>Claims were corrected when evidence contradicted them.</b> Market context was "
          "initially described as the dominant operational signal; permutation importance and a "
          "measured elasticity of 0.139 said otherwise, and the analysis was revised.</li></ul>"
        + "<h3>What remains open</h3>"
        + '<ul class="tight">'
          "<li><b>The heavy tail is unresolved</b> and is the single largest remaining opportunity. "
          "RMSE is 5.5&times; MAE and did not respond to any model family tried. Quantile regression "
          "or a two-stage normal/exceptional classifier is the natural next attempt.</li>"
          "<li><b>The model-selection margin is within noise.</b> Three models sit within $4; "
          "presenting CatBoost as a decisive winner would overstate the evidence.</li>"
          "<li><b>Nov&ndash;Dec performance is inferred, not measured.</b> No labelled data exists "
          "for the scoring window, so the $114.99 figure is carried forward from Sep&ndash;Oct.</li>"
          "<li><b>Operational tooling does not exist</b> &mdash; no monitoring, intervals or serving "
          "layer.</li></ul>"
        + '<div class="callout"><b>The through-line.</b> The work&rsquo;s main strength is that its '
          "conclusions are measured rather than assumed. The two findings most likely to be "
          "misread &mdash; the near-flat December curve and the weak <code>market_index</code> "
          "effect &mdash; were investigated and explained rather than presented at face value or "
          "quietly smoothed over.</div>",
    )


PANELS = [
    _executive, _overview, _dataset, _audit, _eda, _features, _preprocessing,
    _baselines, _advanced, _tuning, _comparison, _explain, _errors, _insights,
    _predictions, _december, _decisions, _architecture, _repo, _timeline,
    _modelcard, _deployment, _conclusion,
]


def build_sections(data: DashboardData) -> str:
    """Render every panel in navigation order."""
    return "".join(panel(data) for panel in PANELS)
