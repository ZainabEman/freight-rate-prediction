"""CSS and JavaScript for the project dashboard.

Kept as module-level constants so :mod:`src.build_dashboard` stays focused on
content assembly. Both are inlined into the output page - the dashboard is a
single self-contained file with no network dependencies.

Design tokens
-------------
The accent ``#064A56`` is taken from the provided ``score.py``: it is the exact
teal the official December chart is drawn in, which ties the dashboard visually
to the delivered artifact rather than importing an unrelated brand colour.

Series colours come from the validated data-visualisation palette and were run
through the palette validator for both light and dark surfaces (all checks pass;
the light-mode aqua sits below 3:1 contrast, which is why every bar carries a
visible value label).
"""

from __future__ import annotations

CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  color-scheme: light;

  /* Neutrals carry a slight teal bias so they read as chosen, not defaulted. */
  --bg:            #f4f7f7;
  --surface:       #ffffff;
  --surface-sunk:  #eef2f3;
  --border:        #d5dfe1;
  --border-strong: #b6c5c8;

  --ink:           #0d1618;
  --ink-secondary: #40565a;
  --ink-muted:     #6b8085;

  --accent:        #064A56;
  --accent-soft:   #e4eef0;
  --accent-line:   #0d6d7d;

  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;

  --good:     #16794f;
  --warning:  #8a5a00;
  --critical: #b3322f;
  --good-bg:     #e2f3ea;
  --warning-bg:  #fbf0d8;
  --critical-bg: #fbe6e5;

  --grid: #e3eaeb;

  --radius: 4px;
  --radius-lg: 6px;
  --maxw: 1180px;

  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg:            #0b1214;
    --surface:       #111b1d;
    --surface-sunk:  #0e1719;
    --border:        #24363a;
    --border-strong: #365055;

    --ink:           #eef4f5;
    --ink-secondary: #a9bfc3;
    --ink-muted:     #7e969b;

    --accent:        #4bb8c9;
    --accent-soft:   #122e34;
    --accent-line:   #4bb8c9;

    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;

    --good:     #46c08a;
    --warning:  #d8a53c;
    --critical: #ef7472;
    --good-bg:     #122a20;
    --warning-bg:  #2b2413;
    --critical-bg: #2d1817;

    --grid: #1c2b2e;
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --bg:            #0b1214;
  --surface:       #111b1d;
  --surface-sunk:  #0e1719;
  --border:        #24363a;
  --border-strong: #365055;
  --ink:           #eef4f5;
  --ink-secondary: #a9bfc3;
  --ink-muted:     #7e969b;
  --accent:        #4bb8c9;
  --accent-soft:   #122e34;
  --accent-line:   #4bb8c9;
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --good:     #46c08a;
  --warning:  #d8a53c;
  --critical: #ef7472;
  --good-bg:     #122a20;
  --warning-bg:  #2b2413;
  --critical-bg: #2d1817;
  --grid: #1c2b2e;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

/* ---------------------------------------------------------------- layout */
.shell { display: flex; min-height: 100vh; }

.sidebar {
  width: 268px;
  flex: 0 0 268px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  height: 100vh;
  position: sticky;
  top: 0;
  overflow-y: auto;
  padding: 22px 0 40px;
}

.brand { padding: 0 22px 18px; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
.brand h1 {
  margin: 0;
  font-size: 16px;
  font-weight: 650;
  letter-spacing: -0.015em;
  line-height: 1.3;
}
.brand p { margin: 5px 0 0; font-size: 12.5px; color: var(--ink-muted); }

.navgroup { padding: 12px 22px 4px; }
.navgroup span {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ink-muted);
  font-weight: 600;
}

.sidebar a {
  display: flex;
  align-items: baseline;
  gap: 9px;
  padding: 6px 22px;
  color: var(--ink-secondary);
  text-decoration: none;
  font-size: 13.5px;
  border-left: 2px solid transparent;
}
.sidebar a:hover { background: var(--surface-sunk); color: var(--ink); }
.sidebar a.active {
  color: var(--accent);
  border-left-color: var(--accent);
  background: var(--accent-soft);
  font-weight: 560;
}
.sidebar a .n {
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--ink-muted);
  min-width: 15px;
  font-variant-numeric: tabular-nums;
}
.sidebar a.active .n { color: var(--accent); }

.main { flex: 1 1 auto; min-width: 0; }
.topbar {
  position: sticky; top: 0; z-index: 20;
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px;
  padding: 11px 30px;
  background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(9px);
  border-bottom: 1px solid var(--border);
}
.topbar .crumb { font-size: 12.5px; color: var(--ink-muted); }
.topbar .crumb b { color: var(--ink); font-weight: 580; }
.toolbar { display: flex; gap: 8px; align-items: center; }

.content { max-width: var(--maxw); padding: 26px 30px 90px; margin: 0 auto; }

section.panel { display: none; }
section.panel.active { display: block; animation: fade .22s ease; }
@keyframes fade { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) {
  section.panel.active { animation: none; }
  * { transition: none !important; }
}

/* ------------------------------------------------------------- typography */
.eyebrow {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--accent); font-weight: 660; margin-bottom: 7px;
  font-family: var(--mono);
}
h2.title {
  margin: 0 0 8px; font-size: 26px; font-weight: 650;
  letter-spacing: -0.022em; line-height: 1.2; text-wrap: balance;
}
p.lede { margin: 0 0 22px; color: var(--ink-secondary); max-width: 68ch; font-size: 15.5px; }
h3 {
  margin: 30px 0 12px; font-size: 15.5px; font-weight: 640;
  letter-spacing: -0.01em;
  padding-bottom: 7px; border-bottom: 1px solid var(--border);
}
h4 { margin: 20px 0 8px; font-size: 13.5px; font-weight: 640; color: var(--ink-secondary); }
p { max-width: 74ch; }
a { color: var(--accent-line); }
code {
  font-family: var(--mono); font-size: 0.87em;
  background: var(--surface-sunk); padding: 1.5px 5px;
  border-radius: 3px; border: 1px solid var(--border);
}

/* ------------------------------------------------------------------ cards */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 18px 20px; margin-bottom: 16px;
}
.card > :first-child { margin-top: 0; }
.card > :last-child { margin-bottom: 0; }

.grid { display: grid; gap: 14px; }
.g2 { grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); }
.g3 { grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); }
.g4 { grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); }

/* KPI tiles: value first, label under, delta as a chip. */
.kpi {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 15px 16px;
  display: flex; flex-direction: column; gap: 3px;
}
.kpi .k-label {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.075em;
  color: var(--ink-muted); font-weight: 600;
}
.kpi .k-value {
  font-size: 25px; font-weight: 650; letter-spacing: -0.025em;
  font-variant-numeric: tabular-nums; line-height: 1.15;
}
.kpi .k-value.sm { font-size: 18px; letter-spacing: -0.015em; }
.kpi .k-note { font-size: 11.5px; color: var(--ink-muted); }
.kpi.hero { border-color: var(--accent); background: var(--accent-soft); }
.kpi.hero .k-value { color: var(--accent); }

/* ------------------------------------------------------------------ chips */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 600; padding: 2px 8px;
  border-radius: 100px; border: 1px solid transparent;
  letter-spacing: 0.01em; white-space: nowrap;
}
.badge.good     { background: var(--good-bg);     color: var(--good);     border-color: color-mix(in srgb, var(--good) 30%, transparent); }
.badge.warn     { background: var(--warning-bg);  color: var(--warning);  border-color: color-mix(in srgb, var(--warning) 30%, transparent); }
.badge.crit     { background: var(--critical-bg); color: var(--critical); border-color: color-mix(in srgb, var(--critical) 30%, transparent); }
.badge.neutral  { background: var(--surface-sunk); color: var(--ink-secondary); border-color: var(--border); }
.badge.accent   { background: var(--accent-soft); color: var(--accent); border-color: color-mix(in srgb, var(--accent) 32%, transparent); }
.badge::before {
  content: ""; width: 5px; height: 5px; border-radius: 50%;
  background: currentColor; flex: 0 0 5px;
}
.badge.nodot::before { display: none; }

/* ------------------------------------------------------------------ table */
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 8px 13px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
th {
  background: var(--surface-sunk); font-weight: 620; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.055em; color: var(--ink-secondary);
  position: sticky; top: 0;
}
td.num, th.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 12.3px; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: var(--surface-sunk); }
tr.highlight { background: var(--accent-soft); }
tr.highlight td { font-weight: 600; }
tr.highlight:hover { background: var(--accent-soft); }
td.wrap, th.wrap { white-space: normal; min-width: 220px; }

/* ---------------------------------------------------------------- buttons */
button.ctl {
  font: inherit; font-size: 12px; font-weight: 550;
  padding: 4px 11px; border-radius: 100px;
  border: 1px solid var(--border-strong); background: var(--surface);
  color: var(--ink-secondary); cursor: pointer;
}
button.ctl:hover { border-color: var(--accent); color: var(--accent); }
/* The accent is dark in light mode and light in dark mode, so the pressed-state
   label is derived from the page background rather than hardcoded to white. */
button.ctl[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--surface); }
button.ctl:focus-visible, .sidebar a:focus-visible, select:focus-visible { outline: 2px solid var(--accent-line); outline-offset: 2px; }

.segmented { display: inline-flex; gap: 5px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
.segmented .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-muted); font-weight: 600; margin-right: 3px; }

select.ctl {
  font: inherit; font-size: 12.5px; padding: 4px 9px;
  border-radius: var(--radius); border: 1px solid var(--border-strong);
  background: var(--surface); color: var(--ink);
}

/* ----------------------------------------------------------------- charts */
.chartbox { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px 18px; margin-bottom: 16px; }
.chartbox h4 { margin: 0 0 2px; font-size: 14px; color: var(--ink); font-weight: 620; }
.chartbox .sub { font-size: 12px; color: var(--ink-muted); margin: 0 0 12px; max-width: 72ch; }
.chartbox svg { display: block; width: 100%; overflow: visible; }
.chart-scroll { overflow-x: auto; }

.tip {
  position: fixed; pointer-events: none; z-index: 60;
  background: var(--ink); color: var(--bg);
  font-size: 11.5px; line-height: 1.45;
  padding: 6px 9px; border-radius: var(--radius);
  opacity: 0; transition: opacity .1s; max-width: 260px;
  font-variant-numeric: tabular-nums;
}
.tip.on { opacity: 1; }
.tip b { color: var(--bg); }

/* ------------------------------------------------------------------ media */
figure { margin: 0 0 16px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 14px; }
figure img { width: 100%; height: auto; display: block; border-radius: var(--radius); background: #fff; }
figure figcaption { font-size: 12px; color: var(--ink-muted); margin-top: 9px; }

/* --------------------------------------------------------------- timeline */
.timeline { position: relative; padding-left: 26px; }
.timeline::before { content: ""; position: absolute; left: 7px; top: 6px; bottom: 6px; width: 1.5px; background: var(--border); }
.tl-item { position: relative; padding-bottom: 18px; }
.tl-item::before {
  content: ""; position: absolute; left: -23px; top: 6px;
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--surface);
  box-shadow: 0 0 0 1.5px var(--accent);
}
.tl-item h4 { margin: 0 0 3px; font-size: 14px; color: var(--ink); }
.tl-item p { margin: 0 0 5px; font-size: 13.2px; color: var(--ink-secondary); }
.tl-meta { font-family: var(--mono); font-size: 11px; color: var(--ink-muted); }

/* --------------------------------------------------------------- diagrams */
.flow { display: flex; flex-direction: column; gap: 0; align-items: stretch; }
.flow-row { display: flex; gap: 10px; flex-wrap: wrap; }
.node {
  flex: 1 1 150px; min-width: 145px;
  background: var(--surface); border: 1px solid var(--border-strong);
  border-radius: var(--radius); padding: 10px 12px;
}
.node .nt { font-size: 12.5px; font-weight: 620; margin-bottom: 2px; }
.node .nd { font-size: 11.5px; color: var(--ink-muted); line-height: 1.45; }
.node.stateless { border-left: 3px solid var(--series-3); }
.node.fitted    { border-left: 3px solid var(--series-2); }
.node.model     { border-left: 3px solid var(--accent); }
.node.io        { border-left: 3px solid var(--series-1); }
.arrow { text-align: center; color: var(--ink-muted); font-size: 15px; line-height: 1; padding: 6px 0; }
.legend-inline { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: var(--ink-secondary); margin-top: 12px; }
.legend-inline span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

/* ---------------------------------------------------------------- decision */
.decision { border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--surface); margin-bottom: 12px; overflow: hidden; }
.decision > summary {
  cursor: pointer; padding: 13px 16px; font-weight: 600; font-size: 14px;
  display: flex; align-items: center; gap: 10px; list-style: none;
}
.decision > summary::-webkit-details-marker { display: none; }
.decision > summary::after { content: "+"; margin-left: auto; color: var(--ink-muted); font-family: var(--mono); font-size: 15px; }
.decision[open] > summary::after { content: "\\2212"; }
.decision[open] > summary { border-bottom: 1px solid var(--border); }
.decision > summary:hover { background: var(--surface-sunk); }
.dbody { padding: 14px 16px; display: grid; gap: 11px; }
.drow { display: grid; grid-template-columns: 116px 1fr; gap: 12px; align-items: start; }
.drow dt {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--ink-muted); font-weight: 660; padding-top: 2px;
}
.drow dd { margin: 0; font-size: 13.3px; color: var(--ink-secondary); }
.drow dd b { color: var(--ink); }
@media (max-width: 620px) { .drow { grid-template-columns: 1fr; gap: 3px; } }

/* ------------------------------------------------------------------- tree */
.tree { font-family: var(--mono); font-size: 12.3px; line-height: 1.85; }
.tree .row { display: flex; gap: 12px; padding: 1px 6px; border-radius: 3px; }
.tree .row:hover { background: var(--surface-sunk); }
.tree .p { color: var(--ink); white-space: pre; }
.tree .d { color: var(--ink-muted); font-family: var(--sans); font-size: 12px; }
.tree .p .dir { color: var(--accent); font-weight: 600; }

ul.tight { margin: 8px 0; padding-left: 19px; }
ul.tight li { margin-bottom: 5px; color: var(--ink-secondary); font-size: 13.6px; }
ul.tight li b, ul.tight li strong { color: var(--ink); }

.callout {
  border-left: 3px solid var(--accent); background: var(--accent-soft);
  padding: 12px 15px; border-radius: 0 var(--radius) var(--radius) 0;
  margin: 14px 0; font-size: 13.6px; color: var(--ink-secondary);
}
.callout b { color: var(--ink); }
.callout.warn { border-left-color: var(--warning); background: var(--warning-bg); }
.callout.crit { border-left-color: var(--critical); background: var(--critical-bg); }

.statline { display: flex; flex-wrap: wrap; gap: 22px; margin: 10px 0 4px; }
.statline div { display: flex; flex-direction: column; }
.statline .sv { font-size: 17px; font-weight: 640; font-variant-numeric: tabular-nums; letter-spacing: -0.015em; }
.statline .sl { font-size: 11px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.06em; }

/* ---------------------------------------------------------------- responsive */
.menu-btn { display: none; }
@media (max-width: 900px) {
  .sidebar {
    position: fixed; left: 0; top: 0; bottom: 0; z-index: 50;
    transform: translateX(-100%); transition: transform .2s ease;
    box-shadow: 0 0 30px rgba(0,0,0,.18);
  }
  .sidebar.open { transform: none; }
  .menu-btn { display: inline-flex; }
  .content { padding: 20px 18px 70px; }
  .topbar { padding: 10px 18px; }
  h2.title { font-size: 22px; }
}
"""


JS = r"""
const D = window.__DASH__;
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const fmt = {
  usd:  v => '$' + v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}),
  usd0: v => '$' + v.toLocaleString('en-US', {maximumFractionDigits: 0}),
  n:    v => v.toLocaleString('en-US'),
  pct:  v => v.toFixed(2) + '%',
  r2:   v => v.toFixed(4)
};

/* ------------------------------------------------------------- navigation */
function show(id) {
  $$('section.panel').forEach(p => p.classList.toggle('active', p.id === id));
  $$('.sidebar a').forEach(a => a.classList.toggle('active', a.dataset.target === id));
  const link = $('.sidebar a[data-target="' + id + '"]');
  if (link) $('#crumb').innerHTML = '<b>' + link.dataset.label + '</b>';
  window.scrollTo({top: 0, behavior: 'instant'});
  if (history.replaceState) history.replaceState(null, '', '#' + id);
  $('.sidebar').classList.remove('open');
  drawAll();
}

/* ------------------------------------------------------------------ theme */
function initTheme() {
  const btn = $('#theme');
  const stored = null; // no storage available; follow OS then toggle in-session
  btn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const osDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = cur ? (cur === 'dark' ? 'light' : 'dark') : (osDark ? 'light' : 'dark');
    document.documentElement.setAttribute('data-theme', next);
    btn.textContent = next === 'dark' ? 'Light' : 'Dark';
    drawAll();
  });
}

/* --------------------------------------------------------------- tooltips */
const tip = document.createElement('div');
tip.className = 'tip';
document.body.appendChild(tip);
function showTip(html, ev) {
  tip.innerHTML = html;
  tip.classList.add('on');
  const pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
}
function hideTip() { tip.classList.remove('on'); }

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
const SVGNS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

/* ------------------------------------------------- horizontal bar chart */
function barChart(host, rows, opts) {
  host.innerHTML = '';
  const o = Object.assign({fmt: fmt.usd, labelW: 190, rowH: 26, pad: 12, highlight: null}, opts || {});
  const W = Math.max(host.clientWidth || 700, 460);
  const H = o.pad * 2 + rows.length * o.rowH;
  const svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img'}, host);
  const plotX = o.labelW + 8;
  const plotW = Math.max(W - plotX - 74, 60);
  const max = Math.max(...rows.map(r => r.value)) || 1;
  const ink = css('--ink'), muted = css('--ink-muted'), grid = css('--grid');
  const accent = css('--accent'), s1 = css('--series-1');

  [0, .25, .5, .75, 1].forEach(t => {
    el('line', {x1: plotX + plotW * t, x2: plotX + plotW * t, y1: o.pad - 4, y2: H - o.pad + 2,
                stroke: grid, 'stroke-width': 1}, svg);
  });

  rows.forEach((r, i) => {
    const y = o.pad + i * o.rowH;
    const bh = 13;
    const w = Math.max((r.value / max) * plotW, 1.5);
    const isHi = o.highlight && r.label === o.highlight;

    const t = el('text', {x: o.labelW, y: y + bh / 2 + 4, 'text-anchor': 'end',
                          'font-size': 12, fill: isHi ? accent : ink,
                          'font-weight': isHi ? 640 : 450}, svg);
    t.textContent = r.label.length > 30 ? r.label.slice(0, 29) + '…' : r.label;

    const rect = el('rect', {x: plotX, y: y, width: w, height: bh, rx: 3,
                             fill: isHi ? accent : s1,
                             opacity: isHi ? 1 : .82}, svg);

    const v = el('text', {x: plotX + w + 7, y: y + bh / 2 + 4, 'font-size': 11.5,
                          fill: isHi ? accent : muted,
                          'font-weight': isHi ? 640 : 500,
                          'font-family': css('--mono')}, svg);
    v.textContent = o.fmt(r.value);

    const hit = el('rect', {x: 0, y: y - 4, width: W, height: o.rowH, fill: 'transparent',
                            style: 'cursor:pointer'}, svg);
    hit.addEventListener('mousemove', e => showTip(
      '<b>' + r.label + '</b><br>' + (r.tip || (o.name || 'Value') + ': ' + o.fmt(r.value)), e));
    hit.addEventListener('mouseleave', hideTip);
    rect.style.transition = 'opacity .12s';
    hit.addEventListener('mouseenter', () => rect.setAttribute('opacity', 1));
    hit.addEventListener('mouseleave', () => rect.setAttribute('opacity', isHi ? 1 : .82));
  });
}

/* ---------------------------------------------------- line / area chart */
function lineChart(host, points, opts) {
  host.innerHTML = '';
  const o = Object.assign({fmt: fmt.usd, height: 250, yLabel: '', xEvery: 5, color: null}, opts || {});
  const W = Math.max(host.clientWidth || 760, 460), H = o.height;
  const m = {t: 14, r: 16, b: 40, l: 62};
  const svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img'}, host);
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const vals = points.map(p => p.y);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const padv = (hi - lo) * 0.18 || Math.abs(hi) * 0.05 || 1;
  lo -= padv; hi += padv;
  const X = i => m.l + (points.length === 1 ? pw / 2 : (i / (points.length - 1)) * pw);
  const Y = v => m.t + ph - ((v - lo) / (hi - lo)) * ph;
  const grid = css('--grid'), muted = css('--ink-muted');
  const color = o.color || css('--accent');

  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * (k / 4), y = Y(v);
    el('line', {x1: m.l, x2: m.l + pw, y1: y, y2: y, stroke: grid, 'stroke-width': 1}, svg);
    const t = el('text', {x: m.l - 9, y: y + 3.5, 'text-anchor': 'end', 'font-size': 10.5,
                          fill: muted, 'font-family': css('--mono')}, svg);
    t.textContent = o.fmt(v);
  }

  const d = points.map((p, i) => (i ? 'L' : 'M') + X(i) + ' ' + Y(p.y)).join(' ');
  el('path', {d: d + ' L ' + X(points.length - 1) + ' ' + (m.t + ph) + ' L ' + X(0) + ' ' + (m.t + ph) + ' Z',
              fill: color, opacity: .09}, svg);
  el('path', {d: d, fill: 'none', stroke: color, 'stroke-width': 2,
              'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, svg);

  points.forEach((p, i) => {
    if (i % o.xEvery === 0 || i === points.length - 1) {
      const t = el('text', {x: X(i), y: H - 16, 'text-anchor': 'middle', 'font-size': 10,
                            fill: muted, 'font-family': css('--mono')}, svg);
      t.textContent = p.x;
    }
  });

  const focus = el('circle', {r: 4.5, fill: color, stroke: css('--surface'),
                              'stroke-width': 2, opacity: 0}, svg);
  const vline = el('line', {y1: m.t, y2: m.t + ph, stroke: color, 'stroke-width': 1,
                            'stroke-dasharray': '3 3', opacity: 0}, svg);
  const overlay = el('rect', {x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent',
                              style: 'cursor:crosshair'}, svg);
  overlay.addEventListener('mousemove', e => {
    const box = svg.getBoundingClientRect();
    const rel = (e.clientX - box.left) / box.width * W;
    let idx = Math.round(((rel - m.l) / pw) * (points.length - 1));
    idx = Math.max(0, Math.min(points.length - 1, idx));
    const p = points[idx];
    focus.setAttribute('cx', X(idx)); focus.setAttribute('cy', Y(p.y));
    focus.setAttribute('opacity', 1);
    vline.setAttribute('x1', X(idx)); vline.setAttribute('x2', X(idx));
    vline.setAttribute('opacity', .6);
    showTip('<b>' + p.x + '</b><br>' + (o.yLabel ? o.yLabel + ': ' : '') + o.fmt(p.y), e);
  });
  overlay.addEventListener('mouseleave', () => {
    focus.setAttribute('opacity', 0); vline.setAttribute('opacity', 0); hideTip();
  });
}

/* ------------------------------------------------------ histogram (bars) */
function histogram(host, counts, edges, opts) {
  host.innerHTML = '';
  const o = Object.assign({height: 230, color: null}, opts || {});
  const W = Math.max(host.clientWidth || 760, 460), H = o.height;
  const m = {t: 12, r: 14, b: 38, l: 54};
  const svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img'}, host);
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const max = Math.max(...counts) || 1;
  const bw = pw / counts.length;
  const grid = css('--grid'), muted = css('--ink-muted');
  const color = o.color || css('--series-1');

  for (let k = 0; k <= 3; k++) {
    const y = m.t + ph - (k / 3) * ph;
    el('line', {x1: m.l, x2: m.l + pw, y1: y, y2: y, stroke: grid, 'stroke-width': 1}, svg);
    const t = el('text', {x: m.l - 8, y: y + 3.5, 'text-anchor': 'end', 'font-size': 10.5,
                          fill: muted, 'font-family': css('--mono')}, svg);
    t.textContent = fmt.n(Math.round(max * k / 3));
  }

  counts.forEach((c, i) => {
    const h = (c / max) * ph;
    const x = m.l + i * bw;
    el('rect', {x: x + 0.8, y: m.t + ph - h, width: Math.max(bw - 1.6, 1), height: h,
                fill: color, opacity: .85, rx: 1.5}, svg);
    const hit = el('rect', {x: x, y: m.t, width: bw, height: ph, fill: 'transparent'}, svg);
    hit.addEventListener('mousemove', e => showTip(
      '<b>' + fmt.usd0(edges[i]) + ' – ' + fmt.usd0(edges[i + 1]) + '</b><br>' +
      fmt.n(c) + ' loads', e));
    hit.addEventListener('mouseleave', hideTip);
  });

  [0, Math.floor(counts.length / 2), counts.length - 1].forEach(i => {
    const t = el('text', {x: m.l + i * bw + bw / 2, y: H - 14, 'text-anchor': 'middle',
                          'font-size': 10, fill: muted, 'font-family': css('--mono')}, svg);
    t.textContent = fmt.usd0(edges[i]);
  });
}

/* ------------------------------------------------------------- registry */
const charts = [];
function register(fn) { charts.push(fn); }
function drawAll() { charts.forEach(f => { try { f(); } catch (e) { console.warn(e); } }); }

/* ----------------------------------------------------------- model chart */
let modelMetric = 'mae';
const METRIC = {
  mae:  {label: 'MAE (lower is better)',  fmt: fmt.usd,  asc: true},
  rmse: {label: 'RMSE (lower is better)', fmt: fmt.usd,  asc: true},
  r2:   {label: 'R² (higher is better)', fmt: fmt.r2, asc: false},
  mape: {label: 'MAPE (lower is better)', fmt: fmt.pct,  asc: true}
};

function drawModels() {
  const host = $('#chart-models'); if (!host || !host.offsetParent) return;
  const all = D.baselines.map(r => Object.assign({fam: 'Baseline'}, r))
    .concat(D.advanced.map(r => Object.assign({fam: 'Advanced'}, r)));
  const cfg = METRIC[modelMetric];
  const rows = all
    .filter(r => r.name !== 'Mean (constant)' && r.name !== 'Median (constant)')
    .sort((a, b) => cfg.asc ? a[modelMetric] - b[modelMetric] : b[modelMetric] - a[modelMetric])
    .map(r => ({
      label: r.name, value: r[modelMetric],
      tip: r.fam + '<br>MAE ' + fmt.usd(r.mae) + ' · RMSE ' + fmt.usd(r.rmse) +
           '<br>R² ' + fmt.r2(r.r2) + ' · MAPE ' + fmt.pct(r.mape)
    }));
  barChart(host, rows, {fmt: cfg.fmt, labelW: 200, highlight: 'CatBoost + smearing (final)'});
  $('#models-sub').textContent = cfg.label +
    ' · constant baselines omitted for scale (MAE > $1,100)';
}

/* ------------------------------------------------------ importance chart */
let impMetric = 'native';
function drawImportance() {
  const host = $('#chart-importance'); if (!host || !host.offsetParent) return;
  const key = impMetric;
  const rows = D.features.slice()
    .sort((a, b) => b[key] - a[key]).slice(0, 20)
    .map(f => ({
      label: f.name, value: Math.max(f[key], 0),
      tip: 'Native ' + f.native.toFixed(3) + ' (rank ' + f.native_rank + ')<br>' +
           'Permutation ' + f.permutation.toFixed(4) + ' (rank ' + f.permutation_rank + ')' +
           (f.note ? '<br><i>' + f.note + '</i>' : '')
    }));
  barChart(host, rows, {
    fmt: key === 'native' ? (v => v.toFixed(2)) : (v => v.toFixed(4)),
    labelW: 175
  });
}

/* ---------------------------------------------------------- other charts */
function drawDecember() {
  const host = $('#chart-december'); if (!host || !host.offsetParent) return;
  lineChart(host, D.december.map(d => ({x: d.date.slice(5), y: d.rate})),
            {yLabel: 'Predicted rate', xEvery: 4, height: 260});
}
function drawMonthly() {
  const a = $('#chart-monthly-rpm'), b = $('#chart-monthly-mkt');
  if (a && a.offsetParent) {
    lineChart(a, D.monthly.map(m => ({x: m.month.slice(5), y: m.rate_per_mile})),
      {fmt: v => '$' + v.toFixed(3), yLabel: 'Mean $/mile', xEvery: 1, height: 210});
  }
  if (b && b.offsetParent) {
    lineChart(b, D.monthly.map(m => ({x: m.month.slice(5), y: m.market_index})),
      {fmt: v => v.toFixed(3), yLabel: 'Mean market index', xEvery: 1, height: 210,
       color: css('--series-2')});
  }
}
function drawPredHist() {
  const host = $('#chart-predhist'); if (!host || !host.offsetParent) return;
  histogram(host, D.prediction_histogram.counts, D.prediction_histogram.edges);
}
function drawBands() {
  const host = $('#chart-bands'); if (!host || !host.offsetParent) return;
  barChart(host, D.distance_bands.map(b => ({
    label: b.band, value: b.rate_per_mile,
    tip: fmt.n(b.loads) + ' loads<br>median $' + b.rate_per_mile.toFixed(3) + '/mile'
  })), {fmt: v => '$' + v.toFixed(3), labelW: 105, rowH: 30});
}
function drawEquipment() {
  const host = $('#chart-equipment'); if (!host || !host.offsetParent) return;
  barChart(host, D.equipment.map(e => ({
    label: e.equipment, value: e.rate_per_mile,
    tip: fmt.n(e.loads) + ' loads<br>median rate ' + fmt.usd(e.median_rate)
  })), {fmt: v => '$' + v.toFixed(3), labelW: 105, rowH: 30});
}
function drawTuning() {
  const host = $('#chart-tuning'); if (!host || !host.offsetParent) return;
  barChart(host, D.tuning.slice().sort((a, b) => a.cv_mae - b.cv_mae).map(t => ({
    label: t.name, value: t.cv_mae,
    tip: t.candidates + ' candidates · ' + Math.round(t.seconds) + 's<br>' +
         '<i>' + JSON.stringify(t.params).replace(/[{}"]/g, '') + '</i>'
  })), {fmt: fmt.usd, labelW: 175, rowH: 30});
}

register(drawModels); register(drawImportance); register(drawDecember);
register(drawMonthly); register(drawPredHist); register(drawBands);
register(drawEquipment); register(drawTuning);

/* ------------------------------------------------------- feature explorer */
function renderFeatures() {
  const group = $('#feat-group').value;
  const q = $('#feat-search').value.trim().toLowerCase();
  const body = $('#feat-body');
  const rows = D.features
    .filter(f => group === 'all' || f.group === group)
    .filter(f => !q || f.name.toLowerCase().includes(q))
    .sort((a, b) => b.native - a.native);
  body.innerHTML = rows.map(f =>
    '<tr><td><code>' + f.name + '</code></td>' +
    '<td><span class="badge neutral nodot">' + f.group + '</span></td>' +
    '<td class="num">' + f.native.toFixed(3) + '</td>' +
    '<td class="num">' + f.permutation.toFixed(4) + '</td>' +
    '<td class="num">' + f.native_rank + '</td>' +
    '<td class="wrap" style="color:var(--ink-muted);font-size:12.5px">' + (f.note || '—') + '</td></tr>'
  ).join('') || '<tr><td colspan="6" style="color:var(--ink-muted)">No features match.</td></tr>';
  $('#feat-count').textContent = rows.length + ' of ' + D.features.length + ' features';
}

/* -------------------------------------------------------------- bootstrap */
function init() {
  $$('.sidebar a').forEach(a =>
    a.addEventListener('click', e => { e.preventDefault(); show(a.dataset.target); }));
  $('#menu').addEventListener('click', () => $('.sidebar').classList.toggle('open'));
  initTheme();

  $$('[data-metric]').forEach(b => b.addEventListener('click', () => {
    modelMetric = b.dataset.metric;
    $$('[data-metric]').forEach(x => x.setAttribute('aria-pressed', x === b));
    drawModels();
  }));
  $$('[data-imp]').forEach(b => b.addEventListener('click', () => {
    impMetric = b.dataset.imp;
    $$('[data-imp]').forEach(x => x.setAttribute('aria-pressed', x === b));
    drawImportance();
  }));

  const gsel = $('#feat-group');
  gsel.innerHTML = '<option value="all">All groups</option>' +
    Object.keys(D.feature_groups).sort().map(g =>
      '<option value="' + g + '">' + g + ' (' + D.feature_groups[g].length + ')</option>').join('');
  gsel.addEventListener('change', renderFeatures);
  $('#feat-search').addEventListener('input', renderFeatures);
  renderFeatures();

  const start = (location.hash || '').replace('#', '');
  show(document.getElementById(start) ? start : 'exec');

  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(drawAll, 140); });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
}
document.addEventListener('DOMContentLoaded', init);
"""
