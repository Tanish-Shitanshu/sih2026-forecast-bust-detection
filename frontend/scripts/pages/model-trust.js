import { pseudo } from '../lib/pseudo-random.js';
import { SUBDIVISIONS } from '../data/subdivisions.js';
import { REASONS } from '../data/model-trust-reasons.js';
import { renderNav } from '../components/nav.js';

// Grid cells are created once and mutated in place on day-switch (rather
// than replaced) so the CSS background-color transition can actually animate.
class ModelTrust {
  constructor(root) {
    this.root = root;
    this.state = { day: 3 };
    this.mount();
  }

  computeVals() {
    const { day } = this.state;
    const days = Array.from({ length: 10 }, (_, i) => i + 1)
      .map((d) => ({ n: d, activeClass: d === day ? 'active' : '' }));

    const regions = SUBDIVISIONS.map((r) => {
      const u = pseudo(r.seed * 3.3 + 1, day * 0.7);
      const confidencePct = Math.round((1 - u) * 100);
      let color, text;
      if (u < 0.35) { color = '#F3F0FA'; text = '#12181C'; }
      else if (u < 0.55) { color = '#D8CFEF'; text = '#12181C'; }
      else if (u < 0.72) { color = '#A692D1'; text = '#12181C'; }
      else { color = '#6B5CA5'; text = '#F5F7F4'; }
      const reason = REASONS[r.seed % REASONS.length];
      return { id: r.id, name: r.name, code: r.code, col: r.col, row: r.row, confidencePct, color, text, reason, u };
    });

    const ranked = [...regions].sort((a, b) => b.u - a.u).slice(0, 8);
    return { day, days, regions, ranked };
  }

  cellHTML(r) {
    return `<div class="cell" data-id="${r.id}" style="grid-column:${r.col};grid-row:${r.row};background:${r.color};color:${r.text}">
      <span class="face-mono" style="font-size:11px;font-weight:600">${r.code}</span>
      <span class="face-mono" style="font-size:10px;opacity:.85" data-pct>${r.confidencePct}%</span>
    </div>`;
  }

  rowHTML(r) {
    return `<tr class="row">
      <td style="font-weight:500">${r.name}</td>
      <td class="face-mono" style="font-weight:600;color:var(--model-uncertain)">${r.confidencePct}%</td>
      <td style="color:var(--ink-dim)">${r.reason}</td>
    </tr>`;
  }

  shell(v) {
    return `
    ${renderNav({
      active: 'model-trust',
      homeHref: '../index.html', dashboardHref: 'dashboard.html', modelTrustHref: 'model-trust.html',
      feedbackLogHref: 'feedback-log.html', methodologyHref: 'methodology.html',
      showTimestamp: true,
    })}

    <div style="flex:0 0 56px;display:flex;align-items:center;gap:20px;padding:0 24px;border-bottom:1px solid var(--line);background:#fff">
      <div class="face-display" style="font-size:13px;font-weight:600;color:var(--ink-dim)">Lead time</div>
      <div id="day-buttons" style="display:flex;gap:6px">
        ${v.days.map((d) => `<button class="daybtn ${d.activeClass}" data-role="day" data-day="${d.n}">D${d.n}</button>`).join('')}
      </div>
      <div style="width:1px;height:28px;background:var(--line)"></div>
      <div id="lead-label" class="face-mono" style="font-size:12px;color:var(--ink-dim)">Model self-confidence for Day ${v.day}, independent of the bust-risk map</div>
    </div>

    <div style="flex:0 0 auto;padding:14px 24px;border-bottom:1px solid var(--line);background:var(--model-uncertain-soft);display:flex;gap:16px;align-items:center">
      <div style="width:14px;height:14px;background:var(--model-uncertain);flex:none;margin-top:1px"></div>
      <div style="font-size:12px;line-height:1.5;color:var(--ink)">
        This panel shows where the <strong>model itself</strong> is unsure, because of sparse historical analogs, an unfamiliar synoptic pattern, or disagreement among ensemble members, not how severe the weather is expected to be. It uses a single violet scale on purpose, never the Green, Yellow, Orange, Red warning code, so it can never be misread as a weather flag.
      </div>
    </div>

    <div style="flex:1;display:flex;min-height:0">
      <div style="flex:1;padding:20px;display:flex;flex-direction:column;gap:12px;min-width:0">
        <div class="face-display" style="font-size:13px;font-weight:600;color:var(--ink-dim)">Model self-confidence, Day ${v.day}</div>
        <div style="flex:1;background:var(--paper-raised);border:1px solid var(--line);padding:8px">
          <div id="confidence-grid" style="display:grid;grid-template-columns:repeat(6, minmax(0,1fr));grid-template-rows:repeat(9, minmax(0,1fr));gap:3px;width:100%;height:100%">
            ${v.regions.map((r) => this.cellHTML(r)).join('')}
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;font-size:12px;color:var(--ink-dim)">
          <span>Model confidence:</span>
          <div style="display:flex;align-items:center;gap:2px">
            <div style="width:22px;height:12px;background:#F3F0FA;border:1px solid var(--line)"></div>
            <div style="width:22px;height:12px;background:#D8CFEF"></div>
            <div style="width:22px;height:12px;background:#A692D1"></div>
            <div style="width:22px;height:12px;background:#6B5CA5"></div>
          </div>
          <span>Typical</span><span style="margin-left:auto"></span><span>Substantially reduced</span>
        </div>
      </div>

      <div style="flex:0 0 420px;border-left:1px solid var(--line);background:#fff;display:flex;flex-direction:column;overflow-y:auto">
        <div style="padding:20px">
          <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:4px">Lowest model confidence, Day ${v.day}</div>
          <div style="font-size:11px;color:var(--ink-dim);margin-bottom:12px">Ranked by self-confidence, most reduced first</div>
          <table class="tight">
            <thead>
              <tr>
                <th>Subdivision</th>
                <th>Confidence</th>
                <th>Primary reason</th>
              </tr>
            </thead>
            <tbody id="ranked-body">
              ${v.ranked.map((r) => this.rowHTML(r)).join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>
    `;
  }

  mount() {
    const v = this.computeVals();
    this.root.innerHTML = this.shell(v);

    this.dayButtonsEl = this.root.querySelector('#day-buttons');
    this.leadLabelEl = this.root.querySelector('#lead-label');
    this.rankedBodyEl = this.root.querySelector('#ranked-body');

    this.cellEls = new Map();
    this.root.querySelectorAll('#confidence-grid [data-id]').forEach((el) => {
      this.cellEls.set(el.dataset.id, el);
    });

    this.bind();
  }

  setState(patch) {
    this.state = { ...this.state, ...patch };
    this.update(this.computeVals());
  }

  update(v) {
    this.dayButtonsEl.innerHTML = v.days.map((d) => `<button class="daybtn ${d.activeClass}" data-role="day" data-day="${d.n}">D${d.n}</button>`).join('');
    this.leadLabelEl.textContent = `Model self-confidence for Day ${v.day}, independent of the bust-risk map`;
    this.rankedBodyEl.innerHTML = v.ranked.map((r) => this.rowHTML(r)).join('');

    v.regions.forEach((r) => {
      const el = this.cellEls.get(String(r.id));
      if (!el) return;
      el.style.background = r.color;
      el.style.color = r.text;
      const pctEl = el.querySelector('[data-pct]');
      if (pctEl) pctEl.textContent = `${r.confidencePct}%`;
    });
  }

  bind() {
    this.root.addEventListener('click', (e) => {
      const dayBtn = e.target.closest('[data-role="day"]');
      if (dayBtn) this.setState({ day: Number(dayBtn.dataset.day) });
    });
  }
}

new ModelTrust(document.getElementById('app'));
