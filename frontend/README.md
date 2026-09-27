# Forecast Trust Console — frontend

Four screens for the SIH 2026 PS 26079 dashboard: Dashboard (`index.html`),
Model Trust, Feedback Log, and Methodology (under `pages/`). Plain HTML/CSS/JS,
no build step or framework.

## Run locally

```
npm install
npm run dev
```

Then open http://localhost:5173.

## Structure

- `index.html`, `pages/*.html` — one shell per screen, each mounting an `#app` root.
- `scripts/pages/*.js` — per-screen state and rendering logic.
- `scripts/components/nav.js` — shared command bar / navigation.
- `scripts/lib/` — the bust-risk formula, the deterministic pseudo-random
  generator behind the mock data, and a small state → render base class.
- `scripts/data/` — mock subdivisions, contributing factors, historical
  analogs, and feedback-log rows.
- `styles/tokens.css` — design tokens: the IMD green/yellow/orange/red
  bust-risk scale, and the separate violet scale for model self-confidence.
- `styles/base.css` — shared component styles (nav, buttons, tables, chips).

## Provenance

This was translated from a Claude Design canvas prototype (4 `.dc.html`
artboards using Claude's internal Design Component format, which depends on
a `support.js` runtime that isn't distributed with the artifact and can't run
standalone). All design tokens, layout, copy, and mock-data logic — the
bust-probability formula, tier thresholds, the 33 subdivisions, contributing
factors, historical analogs, and feedback rows — were carried over exactly;
only the templating mechanism was rewritten as plain JS.
