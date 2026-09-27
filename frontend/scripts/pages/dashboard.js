import { pseudo } from '../lib/pseudo-random.js';
import { bustFor, tierKeyFor, actionFor, TIER_META } from '../lib/bust-risk.js';
import { LEAD_DAYS } from '../lib/constants.js';
import { SUBDIVISIONS } from '../data/subdivisions.js';
import { FACTORS } from '../data/factors.js';
import { ANALOGS } from '../data/analogs.js';
import { renderNav } from '../components/nav.js';
import { animateCountUps } from '../lib/animate.js';

const TIER_KEYS = ['green', 'yellow', 'orange', 'red'];

// Region tiles and the detail drawer are created once and mutated in place
// (rather than replaced on every render) so their CSS transitions — tile
// background-color, and the drawer's slide-in translateX — can animate.
class Dashboard {
  constructor(root) {
    this.root = root;
    this.state = { day: 3, selectedId: null, outcomes: {}, noteDraft: '' };
    this._statValues = {};
    this.mount();
  }

  computeVals() {
    const { day, selectedId, outcomes, noteDraft } = this.state;

    const days = Array.from({ length: LEAD_DAYS }, (_, i) => i + 1)
      .map((d) => ({ n: d, activeClass: d === day ? 'active' : '' }));

    const regions = SUBDIVISIONS.map((r) => {
      const bust = bustFor(r.seed, day);
      const key = tierKeyFor(bust);
      const meta = TIER_META[key];
      return {
        id: r.id, name: r.name, code: r.code, col: r.col, row: r.row, seed: r.seed,
        bustPct: Math.round(bust * 100),
        tierKey: key, tierColor: meta.color, tierText: meta.text, tierLabel: meta.label,
        selectedClass: r.id === selectedId ? 'selected' : '',
      };
    });

    const counts = { green: 0, yellow: 0, orange: 0, red: 0 };
    regions.forEach((r) => counts[r.tierKey]++);
    const tierStats = TIER_KEYS.map((k) => ({
      key: k, color: TIER_META[k].color, count: counts[k], label: TIER_META[k].label,
    }));
    const flagged = counts.orange + counts.red;
    const modelReducedCount = regions.filter((r) => pseudo(r.seed * 3.3 + 1, day * 0.7) > 0.72).length;

    let drawer = null;
    const sel = regions.find((r) => r.id === selectedId);
    if (sel) {
      const ciLow = Math.max(0, sel.bustPct - Math.round((0.05 + pseudo(sel.seed + 9, day) * 0.10) * 100));
      const ciHigh = Math.min(100, sel.bustPct + Math.round((0.05 + pseudo(sel.seed + 11, day) * 0.10) * 100));
      const factors = FACTORS.map((name, i) => {
        const m = (pseudo(sel.seed + i * 13.7, day * 2.3 + i) - 0.5) * 36;
        return { name, val: m };
      }).sort((a, b) => Math.abs(b.val) - Math.abs(a.val)).slice(0, 4).map((f) => ({
        name: f.name,
        sign: f.val >= 0 ? '+' : '−',
        absPct: Math.round(Math.abs(f.val)),
        color: f.val >= 0 ? 'var(--imd-red)' : 'var(--structural)',
        barLeft: f.val >= 0 ? 50 : Math.max(0, 50 - Math.abs(f.val)),
        barWidth: Math.min(50, Math.abs(f.val)),
      }));
      const a1idx = sel.seed % ANALOGS.length;
      let a2idx = (sel.seed + 4) % ANALOGS.length;
      if (a2idx === a1idx) a2idx = (a2idx + 1) % ANALOGS.length;
      const outcomeKey = sel.id + '-d' + day;
      const marked = outcomes[outcomeKey];
      drawer = {
        region: { name: sel.name }, day, bustPct: sel.bustPct, ciLow, ciHigh,
        tierColor: sel.tierColor, tierText: sel.tierText, tierLabel: sel.tierLabel,
        factors, analogs: [ANALOGS[a1idx], ANALOGS[a2idx]],
        action: actionFor(sel.tierKey),
        correctClass: marked === 'correct' ? 'picked' : '',
        incorrectClass: marked === 'incorrect' ? 'picked' : '',
        partialClass: marked === 'partial' ? 'picked' : '',
        note: noteDraft,
        hasMark: !!marked,
        markedLabel: marked ? marked.charAt(0).toUpperCase() + marked.slice(1) : '',
      };
    }

    return {
      day, days, regions, tierStats, flagged, total: regions.length,
      modelReducedCount, hasDrawer: !!drawer, drawer,
    };
  }

  tileHTML(r) {
    return `<button class="regionbtn" data-role="region" data-id="${r.id}" style="grid-column:${r.col};grid-row:${r.row};background:${r.tierColor};color:${r.tierText}" title="${r.name}" aria-label="${r.name}, ${r.bustPct} percent bust probability, ${r.tierLabel}">
      <span class="face-mono" style="font-size:11px;font-weight:600">${r.code}</span>
      <span class="face-mono" style="font-size:10px;opacity:.85" data-pct>${r.bustPct}%</span>
    </button>`;
  }

  drawerInnerHTML(v) {
    if (!v.drawer) return '';
    const d = v.drawer;
    return `
      <div style="height:6px;background:${d.tierColor};flex:none"></div>
      <div style="padding:20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
        <div>
          <div class="face-display" style="font-size:19px;font-weight:700">${d.region.name}</div>
          <div class="face-mono" style="font-size:12px;color:var(--ink-dim);margin-top:2px">Day ${d.day} lead time</div>
        </div>
        <button data-role="close-drawer" aria-label="Close region detail" style="border:1px solid var(--line);background:#fff;width:32px;height:32px;display:flex;align-items:center;justify-content:center;border-radius:2px">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"><path d="M1 1L13 13M13 1L1 13" stroke="#12181C" stroke-width="1.5" stroke-linecap="round"/></svg>
        </button>
      </div>

      <div style="padding:20px;border-bottom:1px solid var(--line);display:flex;gap:20px">
        <div>
          <div style="font-size:11px;color:var(--ink-dim);text-transform:uppercase;letter-spacing:.04em">Bust probability</div>
          <div class="face-mono" style="font-size:36px;font-weight:600;color:${d.tierColor}" data-stat="drawer-bust" data-value="${d.bustPct}" data-suffix="%">${d.bustPct}%</div>
        </div>
        <div style="border-left:1px solid var(--line);padding-left:20px">
          <div style="font-size:11px;color:var(--ink-dim);text-transform:uppercase;letter-spacing:.04em">Confidence band</div>
          <div class="face-mono" style="font-size:16px;font-weight:600;margin-top:8px">${d.ciLow}% to ${d.ciHigh}%</div>
          <div style="font-size:11px;color:var(--ink-dim);margin-top:2px">${d.tierLabel}</div>
        </div>
      </div>

      <div style="padding:20px;border-bottom:1px solid var(--line)">
        <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Top contributing factors</div>
        <div style="display:flex;flex-direction:column;gap:10px">
          ${d.factors.map((f) => `
            <div>
              <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
                <span>${f.name}</span>
                <span class="face-mono" style="font-weight:600;color:${f.color}">${f.sign}${f.absPct}%</span>
              </div>
              <div style="height:6px;background:var(--paper-raised);position:relative">
                <div style="position:absolute;top:0;bottom:0;left:${f.barLeft}%;width:${f.barWidth}%;background:${f.color}"></div>
              </div>
            </div>`).join('')}
        </div>
        <div style="font-size:11px;color:var(--ink-dim);margin-top:10px">Positive values push bust probability up, negative values pull it down.</div>
      </div>

      <div style="padding:20px;border-bottom:1px solid var(--line)">
        <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Historical analogs</div>
        <div style="display:flex;flex-direction:column;gap:10px">
          ${d.analogs.map((an) => `
            <div style="border:1px solid var(--line);padding:12px">
              <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:600">
                <span>This resembles the ${an.date} event</span>
              </div>
              <div style="font-size:12px;color:var(--ink-dim);margin-top:4px">${an.desc}</div>
              <div style="font-size:12px;margin-top:6px"><strong>Outcome:</strong> ${an.outcome}</div>
            </div>`).join('')}
        </div>
      </div>

      <div style="padding:20px;border-bottom:1px solid var(--line)">
        <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Recommended action</div>
        <div style="border:1px solid ${d.tierColor};background:${d.tierColor};color:${d.tierText};padding:12px 14px;font-size:13px;font-weight:600">${d.action}</div>
      </div>

      <div style="padding:20px">
        <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Mark outcome</div>
        <div style="display:flex;gap:8px;margin-bottom:10px">
          <button class="actionbtn ${d.correctClass}" data-role="mark" data-outcome="correct">Correct</button>
          <button class="actionbtn ${d.incorrectClass}" data-role="mark" data-outcome="incorrect">Incorrect</button>
          <button class="actionbtn ${d.partialClass}" data-role="mark" data-outcome="partial">Partial</button>
        </div>
        <label for="outcome-note" style="display:block;font-size:11px;color:var(--ink-dim);margin-bottom:6px">Note for the feedback log (optional)</label>
        <textarea id="outcome-note" data-role="note" placeholder="e.g. nowcast confirmed guidance was reliable" rows="3" style="width:100%">${d.note}</textarea>
        ${d.hasMark ? `<div style="font-size:12px;color:var(--ink-dim);margin-top:8px">Logged as <strong>${d.markedLabel}</strong> for this session. Visible to the duty team in the Feedback Log.</div>` : ''}
      </div>`;
  }

  shell(v) {
    return `
    ${renderNav({
      active: 'dashboard',
      homeHref: '../index.html', dashboardHref: 'dashboard.html', modelTrustHref: 'model-trust.html',
      feedbackLogHref: 'feedback-log.html', methodologyHref: 'methodology.html',
      showTimestamp: true,
    })}

    <div style="position:relative;flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden">

      <div style="flex:0 0 56px;display:flex;align-items:center;gap:20px;padding:0 24px;border-bottom:1px solid var(--line);background:#fff">
        <div class="face-display" style="font-size:13px;font-weight:600;color:var(--ink-dim)">Lead time</div>
        <div id="day-buttons" style="display:flex;gap:6px">
          ${v.days.map((d) => `<button class="daybtn ${d.activeClass}" data-role="day" data-day="${d.n}">D${d.n}</button>`).join('')}
        </div>
        <div style="width:1px;height:28px;background:var(--line)"></div>
        <div id="lead-label" class="face-mono" style="font-size:12px;color:var(--ink-dim)">Showing bust-risk guidance for Day ${v.day}, valid 27 Sep 2026 + ${v.day} days</div>
      </div>

      <div style="flex:0 0 64px;display:flex;align-items:stretch;border-bottom:1px solid var(--line);background:var(--paper)">
        <div style="flex:0 0 340px;display:flex;flex-direction:column;justify-content:center;padding:0 24px;border-right:1px solid var(--line)">
          <div style="font-size:20px;font-weight:700"><span class="face-mono" data-stat="flagged" data-value="${v.flagged}">${v.flagged}</span> of <span class="face-mono" data-stat="total" data-value="${v.total}">${v.total}</span> subdivisions flagged</div>
          <div style="font-size:12px;color:var(--ink-dim)">Orange or Red bust-risk on Day ${v.day}, cross-check or escalate</div>
        </div>
        <div id="tier-stats" style="flex:1;display:flex">
          ${v.tierStats.map((t) => this.tierStatHTML(t)).join('')}
        </div>
      </div>

      <div style="flex:1;display:flex;min-height:0">
        <div style="flex:1;padding:20px;display:flex;flex-direction:column;gap:12px;min-width:0">
          <div class="face-display" style="font-size:13px;font-weight:600;color:var(--ink-dim)">Meteorological subdivisions, Day ${v.day} bust-risk cartogram</div>
          <div style="flex:1;background:var(--paper-raised);border:1px solid var(--line);padding:8px">
            <div id="region-grid" style="display:grid;grid-template-columns:repeat(6, minmax(0,1fr));grid-template-rows:repeat(9, minmax(0,1fr));gap:3px;width:100%;height:100%">
              ${v.regions.map((r) => this.tileHTML(r)).join('')}
            </div>
          </div>
        </div>

        <div style="flex:0 0 340px;border-left:1px solid var(--line);background:#fff;display:flex;flex-direction:column;overflow-y:auto">
          <div style="padding:20px;border-bottom:1px solid var(--line)">
            <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:12px">Bust-risk legend</div>
            <div style="display:flex;flex-direction:column;gap:10px">
              <div style="display:flex;gap:10px;align-items:flex-start">
                <div style="width:14px;height:14px;background:var(--imd-green);margin-top:2px;flex:none"></div>
                <div style="font-size:12px;line-height:1.4"><strong>Green, no warning.</strong> Bust probability under 18%. Forecast is reliable at this lead time.</div>
              </div>
              <div style="display:flex;gap:10px;align-items:flex-start">
                <div style="width:14px;height:14px;background:var(--imd-yellow);margin-top:2px;flex:none"></div>
                <div style="font-size:12px;line-height:1.4"><strong>Yellow, watch.</strong> 18 to 38%. Stay informed, minor deviations possible.</div>
              </div>
              <div style="display:flex;gap:10px;align-items:flex-start">
                <div style="width:14px;height:14px;background:var(--imd-orange);margin-top:2px;flex:none"></div>
                <div style="font-size:12px;line-height:1.4"><strong>Orange, alert.</strong> 38 to 62%. Cross-check against the short-range nowcast.</div>
              </div>
              <div style="display:flex;gap:10px;align-items:flex-start">
                <div style="width:14px;height:14px;background:var(--imd-red);margin-top:2px;flex:none"></div>
                <div style="font-size:12px;line-height:1.4"><strong>Red, warning.</strong> Above 62%. Treat forecast with strong caution, escalate.</div>
              </div>
            </div>
          </div>
          <div id="model-blurb" style="padding:20px;border-bottom:1px solid var(--line);background:var(--model-uncertain-soft)">
            ${this.modelBlurbHTML(v)}
          </div>
          <div style="padding:20px">
            <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Selected region</div>
            <div id="selected-mini">${this.selectedMiniHTML(v)}</div>
          </div>
        </div>
      </div>

      <div id="drawer" class="drawer${v.hasDrawer ? ' drawer-open' : ''}">
        <div id="drawer-inner" style="display:flex;flex-direction:column">${this.drawerInnerHTML(v)}</div>
      </div>

    </div>
    `;
  }

  tierStatHTML(t) {
    return `
      <div style="flex:1;display:flex;align-items:center;gap:10px;padding:0 20px;border-right:1px solid var(--line)">
        <div style="width:12px;height:12px;background:${t.color};border:1px solid rgba(0,0,0,0.15)"></div>
        <div>
          <div class="face-mono" style="font-size:18px;font-weight:600;line-height:1" data-stat="tier-${t.key}" data-value="${t.count}">${t.count}</div>
          <div style="font-size:11px;color:var(--ink-dim)">${t.label}</div>
        </div>
      </div>`;
  }

  modelBlurbHTML(v) {
    return `
      <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:6px;color:var(--model-uncertain)">Model self-confidence</div>
      <div style="font-size:12px;line-height:1.5;color:var(--ink)">Reduced for <span class="face-mono" style="font-weight:600" data-stat="model-reduced" data-value="${v.modelReducedCount}">${v.modelReducedCount}</span> subdivisions on Day ${v.day} — sparse history or an unfamiliar pattern. This is separate from bust-risk color above.</div>
      <a href="model-trust.html" style="display:inline-block;margin-top:10px;font-size:12px;font-weight:600;color:var(--model-uncertain);border-bottom:1px solid var(--model-uncertain)">View Model Trust panel</a>`;
  }

  selectedMiniHTML(v) {
    return v.hasDrawer
      ? `<div style="font-size:12px;color:var(--ink-dim)">Bust-risk detail for ${v.drawer.region.name} is open on the right.</div>`
      : `<div style="font-size:12px;color:var(--ink-dim)">Select a subdivision to view its bust-risk detail, contributing factors, and recommended action.</div>`;
  }

  mount() {
    const v = this.computeVals();
    this.root.innerHTML = this.shell(v);

    this.dayButtonsEl = this.root.querySelector('#day-buttons');
    this.leadLabelEl = this.root.querySelector('#lead-label');
    this.tierStatsEl = this.root.querySelector('#tier-stats');
    this.modelBlurbEl = this.root.querySelector('#model-blurb');
    this.selectedMiniEl = this.root.querySelector('#selected-mini');
    this.drawerEl = this.root.querySelector('#drawer');
    this.drawerInnerEl = this.root.querySelector('#drawer-inner');

    this.tileEls = new Map();
    this.root.querySelectorAll('#region-grid [data-role="region"]').forEach((el) => {
      this.tileEls.set(el.dataset.id, el);
    });

    this.bind();
    // First paint: animate every stat from 0 up to its value.
    this._statValues = animateCountUps(this.root, this._statValues);
  }

  setState(patch) {
    this.state = { ...this.state, ...patch };
    this.update(this.computeVals());
  }

  update(v) {
    this.dayButtonsEl.innerHTML = v.days.map((d) => `<button class="daybtn ${d.activeClass}" data-role="day" data-day="${d.n}">D${d.n}</button>`).join('');
    this.leadLabelEl.textContent = `Showing bust-risk guidance for Day ${v.day}, valid 27 Sep 2026 + ${v.day} days`;
    this.tierStatsEl.innerHTML = v.tierStats.map((t) => this.tierStatHTML(t)).join('');
    this.modelBlurbEl.innerHTML = this.modelBlurbHTML(v);
    this.selectedMiniEl.innerHTML = this.selectedMiniHTML(v);

    v.regions.forEach((r) => {
      const el = this.tileEls.get(String(r.id));
      if (!el) return;
      el.style.background = r.tierColor;
      el.style.color = r.tierText;
      el.classList.toggle('selected', r.id === this.state.selectedId);
      el.title = r.name;
      el.setAttribute('aria-label', `${r.name}, ${r.bustPct} percent bust probability, ${r.tierLabel}`);
      const pctEl = el.querySelector('[data-pct]');
      if (pctEl) pctEl.textContent = `${r.bustPct}%`;
    });

    this.drawerEl.classList.toggle('drawer-open', v.hasDrawer);
    if (v.hasDrawer) {
      this.drawerInnerEl.innerHTML = this.drawerInnerHTML(v);
    }

    this._statValues = animateCountUps(this.root, this._statValues);
  }

  bind() {
    this.root.addEventListener('click', (e) => {
      const dayBtn = e.target.closest('[data-role="day"]');
      if (dayBtn) { this.setState({ day: Number(dayBtn.dataset.day), selectedId: null }); return; }

      const regionBtn = e.target.closest('[data-role="region"]');
      if (regionBtn) { this.setState({ selectedId: Number(regionBtn.dataset.id) }); return; }

      const closeBtn = e.target.closest('[data-role="close-drawer"]');
      if (closeBtn) { this.setState({ selectedId: null }); return; }

      const markBtn = e.target.closest('[data-role="mark"]');
      if (markBtn) {
        const outcomeKey = this.state.selectedId + '-d' + this.state.day;
        this.setState({ outcomes: { ...this.state.outcomes, [outcomeKey]: markBtn.dataset.outcome } });
      }
    });

    // Track the note as the user types without re-rendering per keystroke
    // (re-rendering the drawer would blow away the textarea's cursor position).
    this.root.addEventListener('input', (e) => {
      if (e.target.dataset.role === 'note') this.state.noteDraft = e.target.value;
    });
  }
}

new Dashboard(document.getElementById('app'));
