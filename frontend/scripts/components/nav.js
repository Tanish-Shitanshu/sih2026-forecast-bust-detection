// Shared command bar: brand (links back to the landing page), nav links, and
// (on the Dashboard and Model Trust screens only, matching the original) the
// forecast-issued timestamp.
export function renderNav({ active, homeHref, dashboardHref, modelTrustHref, feedbackLogHref, methodologyHref, showTimestamp }) {
  const link = (href, label, key) =>
    `<a href="${href}" class="navlink${active === key ? ' active' : ''}">${label}</a>`;

  const timestamp = showTimestamp
    ? `<div class="face-mono" style="margin-left:auto;display:flex;flex-direction:column;justify-content:center;align-items:flex-end;font-size:12px;color:#CBD6DE;line-height:1.5">
        <div>Forecast issued 27 Sep 2026, 00 UTC</div>
        <div style="color:#8FA3B0">NCMRWF GFS 12km, IMD synthesis</div>
      </div>`
    : '';

  return `
    <div class="face-display" style="flex:0 0 auto;min-height:64px;background:var(--structural-deep);color:#F5F7F4;display:flex;align-items:stretch;gap:0;padding:10px 24px;border-bottom:1px solid #0B1D2E">
      <a href="${homeHref}" style="display:flex;flex-direction:column;justify-content:center;padding-right:24px;border-right:1px solid #2E4A63;color:inherit;text-decoration:none">
        <div style="font-size:18px;font-weight:700;letter-spacing:.2px">Vishwas</div>
        <div style="font-size:11px;color:#9FB2BF;font-family:'IBM Plex Sans',sans-serif">Forecast Trust Console — NCMRWF, Ministry of Earth Sciences</div>
      </a>
      <nav style="display:flex;gap:6px;padding-left:8px">
        ${link(dashboardHref, 'Dashboard', 'dashboard')}
        ${link(modelTrustHref, 'Model Trust', 'model-trust')}
        ${link(feedbackLogHref, 'Feedback Log', 'feedback-log')}
        ${link(methodologyHref, 'Methodology', 'methodology')}
      </nav>
      ${timestamp}
    </div>`;
}
