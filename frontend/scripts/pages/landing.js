import { renderNav } from '../components/nav.js';
import { bustFor, tierKeyFor, TIER_META } from '../lib/bust-risk.js';
import { LEAD_DAYS } from '../lib/constants.js';
import { SUBDIVISIONS } from '../data/subdivisions.js';
import { ROWS, OUTCOME_LABELS } from '../data/feedback-rows.js';
import { computeFeedbackStats } from '../lib/feedback-stats.js';
import { animateCountUps } from '../lib/animate.js';

const PREVIEW_DAYS = [3, 6, 9];
const PREVIEW_CYCLE_MS = 3500;

const HOW_IT_WORKS = [
  'An official NCMRWF/IMD forecast is issued for Day 1–10.',
  'The current pattern is compared against historical forecast-error behavior.',
  'Bust-risk and confidence are scored per subdivision and lead day.',
  'The forecaster gets a recommended action and can log the real outcome afterward.',
];

const FEATURES = [
  {
    title: 'Explainable, not a black box',
    desc: 'Every bust-risk score comes with its top contributing factors and how much each one pushes the estimate up or down, not just a single number.',
  },
  {
    title: 'Historical analog matching',
    desc: 'Each flagged subdivision is matched against real past events with a similar pattern, and how those forecasts actually verified.',
  },
  {
    title: 'Model self-confidence, kept separate',
    desc: 'A separate violet scale shows where the model itself lacks historical precedent, always distinct from the bust-risk warning colors.',
  },
  {
    title: 'Actionable, not just a number',
    desc: 'Every flagged subdivision comes with a specific recommended action, from a routine cross-check to immediate escalation.',
  },
  {
    title: 'A real forecaster feedback loop',
    desc: 'Duty forecasters mark whether a flagged prediction was correct, incorrect, or partial, building a verified record over time.',
  },
  {
    title: "Built on IMD's own warning-color convention",
    desc: 'Bust-risk uses the same Green, Yellow, Orange, Red scale IMD already uses for weather warnings, so it reads instantly to a duty forecaster.',
  },
];

function previewTileHTML(r, day) {
  const bust = bustFor(r.seed, day);
  const meta = TIER_META[tierKeyFor(bust)];
  return `<div class="preview-tile" data-id="${r.id}" style="background:${meta.color};color:${meta.text}"><span>${r.code}</span></div>`;
}

// One-time reveal: run `callback` the first time `el` scrolls into view,
// then stop watching. Used for the stats-bar count-up and the how-it-works
// connecting line — both are a single reveal, never a repeating effect.
function onceInView(el, threshold, callback) {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        callback();
        observer.disconnect();
        break;
      }
    }
  }, { threshold });
  observer.observe(el);
}

function activityRowHTML(r) {
  return `
    <div class="activity-row">
      <div class="activity-date">${r.date}</div>
      <div class="activity-region">${r.region}</div>
      <span class="badge ${r.outcome}">${OUTCOME_LABELS[r.outcome]}</span>
    </div>`;
}

const subdivisionCount = SUBDIVISIONS.length;
const fbStats = computeFeedbackStats();
const previewTiles = SUBDIVISIONS.slice(0, 12);
const recentRows = ROWS.slice(0, 4);
const initialPreviewDay = PREVIEW_DAYS[0];

document.getElementById('app').innerHTML = `
${renderNav({
  active: null,
  homeHref: 'index.html', dashboardHref: 'pages/dashboard.html', modelTrustHref: 'pages/model-trust.html',
  feedbackLogHref: 'pages/feedback-log.html', methodologyHref: 'pages/methodology.html',
  showTimestamp: false,
})}

<div class="hero">
  <div class="hero-grid">
    <div class="hero-left">
      <div class="hero-name">Vishwas</div>
      <div class="hero-tagline face-mono">Forecast Trust Console — NCMRWF, Ministry of Earth Sciences</div>
      <div class="hero-desc">Estimates how likely an already-issued NCMRWF/IMD forecast is to bust, so duty forecasters can weigh official guidance with calibrated confidence instead of treating every forecast as equally certain.</div>
      <div class="hero-buttons">
        <a href="pages/dashboard.html" class="cta-btn">Enter Dashboard</a>
        <a href="pages/methodology.html" class="cta-btn-secondary">View Methodology</a>
      </div>
    </div>
    <div class="hero-preview">
      <div class="preview-grid" id="preview-grid">
        ${previewTiles.map((r) => previewTileHTML(r, initialPreviewDay)).join('')}
      </div>
      <div class="preview-caption face-mono" id="preview-caption">Bust-risk cartogram, Day ${initialPreviewDay} preview — sample of ${previewTiles.length} of ${subdivisionCount} subdivisions</div>
    </div>
  </div>
</div>

<div class="stats-bar" id="stats-bar">
  <div class="stat-item">
    <div class="stat-value" data-stat="s-subdivisions" data-value="${subdivisionCount}">0</div>
    <div class="stat-label">Meteorological subdivisions monitored</div>
  </div>
  <div class="stat-item">
    <div class="stat-value" data-stat="s-days" data-value="${LEAD_DAYS}">0</div>
    <div class="stat-label">Day forecast horizon covered</div>
  </div>
  <div class="stat-item">
    <div class="stat-value" data-stat="s-entries" data-value="${fbStats.total}">0</div>
    <div class="stat-label">Feedback entries logged</div>
  </div>
  <div class="stat-item">
    <div class="stat-value" data-stat="s-correct" data-value="${fbStats.correctPct}" data-suffix="%">0%</div>
    <div class="stat-label">Confirmed correct in the feedback log</div>
  </div>
</div>

<div class="how-it-works">
  <div class="how-it-works-inner">
    <div class="section-header">
      <div class="section-title">How it works</div>
      <div class="section-subtitle">From an issued forecast to a logged outcome, in four steps.</div>
    </div>
    <div class="steps" id="steps">
      ${HOW_IT_WORKS.map((text, i) => `
        <div class="step">
          <div class="step-num">${i + 1}</div>
          <div class="step-text">${text}</div>
        </div>`).join('')}
    </div>
  </div>
</div>

<div class="feature-section">
  <div class="feature-section-inner">
    <div class="section-header">
      <div class="section-title">Why it's different</div>
      <div class="section-subtitle">What sets this apart from a plain risk score.</div>
    </div>
    <div class="feature-cards">
      ${FEATURES.map((f) => `
        <div class="feature-card">
          <div class="feature-card-title">${f.title}</div>
          <div class="feature-card-desc">${f.desc}</div>
        </div>`).join('')}
    </div>
  </div>
</div>

<div class="activity-section">
  <div class="activity-grid">
    <div class="activity-left">
      <div class="activity-left-title">Why the feedback loop matters</div>
      <div class="activity-left-desc">Every flagged prediction can be marked correct, incorrect, or partial by the duty forecaster who reviewed it. That record is what turns a bust-risk estimate into something verified over time, not just a static score.</div>
    </div>
    <div class="activity-box">
      <div class="activity-header">
        <div class="activity-title">Sample feedback log entries</div>
        <a href="pages/feedback-log.html" class="activity-link">View full log</a>
      </div>
      ${recentRows.map(activityRowHTML).join('')}
    </div>
  </div>
</div>

<div class="landing-footer">
  <div class="footer-inner">
    <div class="footer-col">
      <div class="footer-col-title">Sections</div>
      <a href="pages/dashboard.html">Dashboard</a>
      <a href="pages/model-trust.html">Model Trust</a>
      <a href="pages/feedback-log.html">Feedback Log</a>
      <a href="pages/methodology.html">Methodology</a>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">Data sources</div>
      <div class="footer-col-text">ERA5 reanalysis</div>
      <div class="footer-col-text">IMD Gridded Rainfall</div>
      <div class="footer-col-text">NCMRWF NWP archives</div>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">About</div>
      <div class="footer-col-text">Ministry of Earth Sciences (MoES)</div>
      <div class="footer-col-text">National Centre for Medium-Range Weather Forecasting (NCMRWF)</div>
      <div class="footer-col-text">Smart India Hackathon 2026, Problem Statement 26079, AI-based Forecast Bust Detection</div>
    </div>
  </div>
  <div class="footer-bottom">UI shown with illustrative data, not connected to a live feed.</div>
</div>
`;

// Count the stats bar up from 0 once it actually scrolls into view — not on
// page load, and only once (this is a one-time reveal, not a decorative loop).
const statsBar = document.getElementById('stats-bar');
onceInView(statsBar, 0.4, () => animateCountUps(statsBar, {}, 800));

// How it works: the connecting line draws itself left-to-right once, the
// first time the section scrolls into view.
const steps = document.getElementById('steps');
onceInView(steps, 0.4, () => steps.classList.add('line-drawn'));

// Hero cartogram preview: cycle through a few sample lead-time days using
// the same real bustFor/tierKeyFor formula the Dashboard uses, so it reads
// as a live sample of real logic rather than a static graphic — the caption
// always names which day is currently shown, and it's explicitly a preview.
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (!reducedMotion) {
  const previewCaption = document.getElementById('preview-caption');
  const previewTileEls = new Map();
  document.querySelectorAll('#preview-grid [data-id]').forEach((el) => previewTileEls.set(el.dataset.id, el));

  let previewIndex = 0;
  setInterval(() => {
    previewIndex = (previewIndex + 1) % PREVIEW_DAYS.length;
    const day = PREVIEW_DAYS[previewIndex];
    previewTiles.forEach((r) => {
      const el = previewTileEls.get(String(r.id));
      if (!el) return;
      const meta = TIER_META[tierKeyFor(bustFor(r.seed, day))];
      el.style.background = meta.color;
      el.style.color = meta.text;
    });
    previewCaption.textContent = `Bust-risk cartogram, Day ${day} preview — sample of ${previewTiles.length} of ${subdivisionCount} subdivisions`;
  }, PREVIEW_CYCLE_MS);
}
