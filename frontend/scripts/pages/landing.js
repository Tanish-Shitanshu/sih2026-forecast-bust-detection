import { renderNav } from '../components/nav.js';

const cards = [
  {
    href: 'pages/dashboard.html',
    title: 'Dashboard',
    desc: 'Bust-risk cartogram for all 33 meteorological subdivisions, Day 1 to 10, with region-level detail and outcome logging.',
  },
  {
    href: 'pages/model-trust.html',
    title: 'Model Trust',
    desc: 'Where the model itself is uncertain, on a separate violet scale so it is never confused with bust-risk.',
  },
  {
    href: 'pages/feedback-log.html',
    title: 'Feedback Log',
    desc: 'Outcomes duty forecasters have logged against flagged predictions.',
  },
  {
    href: 'pages/methodology.html',
    title: 'Methodology',
    desc: 'How the bust-risk estimate is built, and the data sources behind it.',
  },
];

document.getElementById('app').innerHTML = `
${renderNav({
  active: null,
  homeHref: 'index.html', dashboardHref: 'pages/dashboard.html', modelTrustHref: 'pages/model-trust.html',
  feedbackLogHref: 'pages/feedback-log.html', methodologyHref: 'pages/methodology.html',
  showTimestamp: false,
})}

<div class="hero">
  <div class="hero-name">Vishwas</div>
  <div class="hero-tagline face-mono">Forecast Trust Console — NCMRWF, Ministry of Earth Sciences</div>
  <div class="hero-desc">Estimates how likely an already-issued NCMRWF/IMD forecast is to bust, so duty forecasters can weigh official guidance with calibrated confidence instead of treating every forecast as equally certain.</div>
  <a href="pages/dashboard.html" class="cta-btn">Enter Dashboard</a>
</div>

<div class="nav-cards">
  ${cards.map((c) => `
    <a href="${c.href}" class="nav-card">
      <div class="nav-card-title">${c.title}</div>
      <div class="nav-card-desc">${c.desc}</div>
    </a>`).join('')}
</div>

<div class="landing-footer">
  Ministry of Earth Sciences (MoES) and National Centre for Medium-Range Weather Forecasting (NCMRWF). Built for Smart India Hackathon 2026, Problem Statement 26079, AI-based Forecast Bust Detection. UI shown with illustrative data, not connected to a live feed.
</div>
`;
