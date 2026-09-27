import { renderNav } from '../components/nav.js';

document.getElementById('app').innerHTML = `
${renderNav({
  active: 'methodology',
  dashboardHref: '../index.html', modelTrustHref: 'model-trust.html',
  feedbackLogHref: 'feedback-log.html', methodologyHref: 'methodology.html',
  showTimestamp: false,
})}

<div style="flex:1;padding:40px;display:flex;justify-content:center">
  <div style="width:820px;display:flex;flex-direction:column;gap:32px">

    <div>
      <div class="face-display" style="font-size:26px;font-weight:700;margin-bottom:6px">About this tool</div>
      <p style="color:var(--ink-dim)">Forecast Trust Console, built for NCMRWF and the Ministry of Earth Sciences under Smart India Hackathon 2026, Problem Statement 26079.</p>
    </div>

    <div style="display:flex;gap:20px">
      <div style="flex:1;border:1px solid var(--imd-green);background:#fff;padding:18px">
        <h2 style="color:var(--imd-green)">What this does</h2>
        <p>It looks at an already-issued official NCMRWF and IMD forecast for Day 1 to 10 and estimates how likely that forecast is to bust, a large deviation between the forecast and what actually happens, before the outcome is known. It flags which subdivisions and lead days deserve extra caution, so a duty forecaster can weigh the official guidance with calibrated confidence rather than treating every forecast as equally certain.</p>
      </div>
      <div style="flex:1;border:1px solid var(--structural-dim);background:#fff;padding:18px">
        <h2>What this does not do</h2>
        <p>It does not forecast the weather itself, and it does not replace or override IMD or NCMRWF guidance. It does not issue public warnings, and it is not a substitute for the duty forecaster's judgment, it is a second opinion on how much to trust the first one.</p>
      </div>
    </div>

    <div>
      <h2>How the bust-risk estimate is built</h2>
      <p>For each subdivision and lead day, the system compares characteristics of the current forecast pattern (moisture flux, upper-air trough position, sea-surface temperature anomalies, ensemble spread, and related fields) against a historical record of past forecasts and their verified outcomes. Where the current pattern closely resembles situations that busted before, the bust-risk estimate rises. This is reported separately from the model's own self-confidence, which reflects how much historical precedent exists for the current pattern, not how severe the weather is expected to be. The two are shown in different colors throughout this tool so they are never confused.</p>
    </div>

    <div>
      <h2>Data sources</h2>
      <div style="display:flex;gap:16px">
        <div class="src">
          <div class="face-mono" style="font-weight:600;font-size:13px;margin-bottom:6px">ERA5 reanalysis</div>
          <div style="font-size:12px;color:var(--ink-dim);line-height:1.5">ECMWF's global reanalysis, used as a verification reference for past atmospheric states.</div>
        </div>
        <div class="src">
          <div class="face-mono" style="font-weight:600;font-size:13px;margin-bottom:6px">IMD Gridded Rainfall</div>
          <div style="font-size:12px;color:var(--ink-dim);line-height:1.5">India Meteorological Department's gridded daily rainfall archive, used to verify past forecast outcomes at subdivision level.</div>
        </div>
        <div class="src">
          <div class="face-mono" style="font-weight:600;font-size:13px;margin-bottom:6px">NCMRWF NWP archives</div>
          <div style="font-size:12px;color:var(--ink-dim);line-height:1.5">Archived numerical weather prediction runs, used to build the historical bust record this tool learns from.</div>
        </div>
      </div>
    </div>

    <div style="border-top:1px solid var(--line);padding-top:20px;font-size:12px;color:var(--ink-dim)">
      Ministry of Earth Sciences (MoES) and National Centre for Medium-Range Weather Forecasting (NCMRWF). Built for Smart India Hackathon 2026, Problem Statement 26079, AI-based Forecast Bust Detection. UI shown with illustrative data, not connected to a live feed.
    </div>

  </div>
</div>
`;
