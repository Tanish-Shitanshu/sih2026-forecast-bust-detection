# Vishwas — frontend

Forecast Trust Console, for the SIH 2026 PS 26079 project. A landing page
(`index.html`) plus four screens under `pages/`: Dashboard, Model Trust,
Feedback Log, and Methodology. Plain HTML/CSS/JS, no build step or framework.

## Run locally

```
npm install
npm run dev
```

Then open http://localhost:5173.

## Structure

- `index.html` — landing page: hero, "Enter Dashboard" CTA, and nav cards to
  the four screens.
- `pages/*.html` — one shell per screen, each mounting an `#app` root.
- `scripts/pages/*.js` — per-screen state and rendering logic. Dashboard and
  Model Trust manage their region-grid tiles (and Dashboard its detail
  drawer) as persistent DOM nodes updated in place, rather than fully
  re-rendered, so their CSS transitions can animate.
- `scripts/components/nav.js` — shared command bar: brand (links to the
  landing page) + navigation.
- `scripts/lib/` — the bust-risk formula, the deterministic pseudo-random
  generator behind the mock data, a small state → render base class used by
  the simpler screens, and the count-up animation helper.
- `scripts/data/` — mock subdivisions, contributing factors, historical
  analogs, and feedback-log rows.
- `styles/tokens.css` — design tokens: the IMD green/yellow/orange/red
  bust-risk scale, and the separate violet scale for model self-confidence.
- `styles/base.css` — shared component styles (nav, buttons, tables, chips,
  region-tile/drawer transitions).
- `styles/landing.css` — landing-page-only layout (hero, CTA, nav cards,
  footer), built from the same tokens.

## Provenance

This was translated from a Claude Design canvas prototype (4 `.dc.html`
artboards using Claude's internal Design Component format, which depends on
a `support.js` runtime that isn't distributed with the artifact and can't run
standalone). All design tokens, layout, copy, and mock-data logic — the
bust-probability formula, tier thresholds, the 33 subdivisions, contributing
factors, historical analogs, and feedback rows — were carried over exactly;
only the templating mechanism was rewritten as plain JS.
