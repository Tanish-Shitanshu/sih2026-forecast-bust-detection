import { Screen } from '../lib/screen.js';
import { pseudo } from '../lib/pseudo-random.js';
import { bustFor, tierKeyFor, actionFor, TIER_META } from '../lib/bust-risk.js';
import { SUBDIVISIONS } from '../data/subdivisions.js';
import { FACTORS } from '../data/factors.js';
import { ANALOGS } from '../data/analogs.js';
import { renderNav } from '../components/nav.js';

class Dashboard extends Screen {
  renderVals() {
    const { day, selectedId, outcomes, noteDraft } = this.state;

    const days = Array.from({ length: 10 }, (_, i) => i + 1)
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
    const tierStats = ['green', 'yellow', 'orange', 'red'].map((k) => ({
      color: TIER_META[k].color, count: counts[k], label: TIER_META[k].label,
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
        region: { name: sel.name }, bustPct: sel.bustPct, ciLow, ciHigh,
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

  template(v) {
    return `
    ${renderNav({
      active: 'dashboard',
      dashboardHref: 'index.html', modelTrustHref: 'pages/model-trust.html',
      feedbackLogHref: 'pages/feedback-log.html', methodologyHref: 'pages/methodology.html',
      showTimestamp: true,
    })}

    <div style="flex:0 0 56px;display:flex;align-items:center;gap:20px;padding:0 24px;border-bottom:1px solid var(--line);background:#fff">
      <div class="face-display" style="font-size:13px;font-weight:600;color:var(--ink-dim)">Lead time</div>
      <div style="display:flex;gap:6px">
        ${v.days.map((d) => `<button class="daybtn ${d.activeClass}" data-role="day" data-day="${d.n}">D${d.n}</button>`).join('')}
      </div>
      <div style="width:1px;height:28px;background:var(--line)"></div>
      <div class="face-mono" style="font-size:12px;color:var(--ink-dim)">Showing bust-risk guidance for Day ${v.day}, valid 27 Sep 2026 + ${v.day} days</div>
    </div>

    <div style="flex:0 0 64px;display:flex;align-items:stretch;border-bottom:1px solid var(--line);background:var(--paper)">
      <div style="flex:0 0 340px;display:flex;flex-direction:column;justify-content:center;padding:0 24px;border-right:1px solid var(--line)">
        <div style="font-size:20px;font-weight:700"><span class="face-mono">${v.flagged}</span> of <span class="face-mono">${v.total}</span> subdivisions flagged</div>
        <div style="font-size:12px;color:var(--ink-dim)">Orange or Red bust-risk on Day ${v.day}, cross-check or escalate</div>
      </div>
      <div style="flex:1;display:flex">
        ${v.tierStats.map((t) => `
          <div style="flex:1;display:flex;align-items:center;gap:10px;padding:0 20px;border-right:1px solid var(--line)">
            <div style="width:12px;height:12px;background:${t.color};border:1px solid rgba(0,0,0,0.15)"></div>
            <div>
              <div class="face-mono" style="font-size:18px;font-weight:600;line-height:1">${t.count}</div>
              <div style="font-size:11px;color:var(--ink-dim)">${t.label}</div>
            </div>
          </div>`).join('')}
      </div>
    </div>

    <div style="flex:1;display:flex;min-height:0">
      <div style="flex:1;padding:20px;display:flex;flex-direction:column;gap:12px;min-width:0">
        <div class="face-display" style="font-size:13px;font-weight:600;color:var(--ink-dim)">Meteorological subdivisions, Day ${v.day} bust-risk cartogram</div>
        <div style="flex:1;background:var(--paper-raised);border:1px solid var(--line);padding:8px">
          <div style="display:grid;grid-template-columns:repeat(6, minmax(0,1fr));grid-template-rows:repeat(9, minmax(0,1fr));gap:3px;width:100%;height:100%">
            ${v.regions.map((r) => `
              <button class="regionbtn ${r.selectedClass}" style="grid-column:${r.col};grid-row:${r.row};background:${r.tierColor};color:${r.tierText}" data-role="region" data-id="${r.id}" title="${r.name}" aria-label="${r.name}, ${r.bustPct} percent bust probability, ${r.tierLabel}">
                <span class="face-mono" style="font-size:11px;font-weight:600">${r.code}</span>
                <span class="face-mono" style="font-size:10px;opacity:.85">${r.bustPct}%</span>
              </button>`).join('')}
          </div>
        </div>
      </div>

      <div style="flex:0 0 340px;border-left:1px solid var(--line);background:#fff;display:flex;flex-direction:column;overflow-y:auto">
        <div style="padding:20px;border-bottom:1px solid var(--line)">
          <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:12px">Bust-risk legend</div>
          <div style="display:flex;flex-direction:column;gap:10px">
            <div style="display:flex;gap:10px;align-items:flex-start">
              <div style="width:14px;height:14px;background:var(--imd-green);margin-top:2px;flex:none"></div>
              <div style="font-size:12px;line-height:1.4"><strong>Green, no warning.</strong> Bust probability under 18%. Forecast can be trusted at this lead time.</div>
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
        <div style="padding:20px;border-bottom:1px solid var(--line);background:var(--model-uncertain-soft)">
          <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:6px;color:var(--model-uncertain)">Model self-confidence</div>
          <div style="font-size:12px;line-height:1.5;color:var(--ink)">Reduced for <span class="face-mono" style="font-weight:600">${v.modelReducedCount}</span> subdivisions on Day ${v.day} — sparse history or an unfamiliar pattern. This is separate from bust-risk color above.</div>
          <a href="pages/model-trust.html" style="display:inline-block;margin-top:10px;font-size:12px;font-weight:600;color:var(--model-uncertain);border-bottom:1px solid var(--model-uncertain)">View Model Trust panel</a>
        </div>
        <div style="padding:20px">
          <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Selected region</div>
          ${v.hasDrawer
            ? `<div style="font-size:12px;color:var(--ink-dim)">${v.drawer.region.name} is open in the detail panel.</div>`
            : `<div style="font-size:12px;color:var(--ink-dim)">Click any subdivision on the map to open its bust-risk detail.</div>`}
        </div>
      </div>
    </div>

    ${v.hasDrawer ? `
    <div style="position:absolute;top:64px;right:0;bottom:0;width:460px;background:#fff;border-left:2px solid var(--structural);box-shadow:none;display:flex;flex-direction:column;overflow-y:auto">
      <div style="height:6px;background:${v.drawer.tierColor};flex:none"></div>
      <div style="padding:20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
        <div>
          <div class="face-display" style="font-size:19px;font-weight:700">${v.drawer.region.name}</div>
          <div class="face-mono" style="font-size:12px;color:var(--ink-dim);margin-top:2px">Day ${v.day} lead time</div>
        </div>
        <button data-role="close-drawer" aria-label="Close region detail" style="border:1px solid var(--line);background:#fff;width:32px;height:32px;display:flex;align-items:center;justify-content:center;border-radius:2px">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"><path d="M1 1L13 13M13 1L1 13" stroke="#12181C" stroke-width="1.5" stroke-linecap="round"/></svg>
        </button>
      </div>

      <div style="padding:20px;border-bottom:1px solid var(--line);display:flex;gap:20px">
        <div>
          <div style="font-size:11px;color:var(--ink-dim);text-transform:uppercase;letter-spacing:.04em">Bust probability</div>
          <div class="face-mono" style="font-size:36px;font-weight:600;color:${v.drawer.tierColor}">${v.drawer.bustPct}%</div>
        </div>
        <div style="border-left:1px solid var(--line);padding-left:20px">
          <div style="font-size:11px;color:var(--ink-dim);text-transform:uppercase;letter-spacing:.04em">Confidence band</div>
          <div class="face-mono" style="font-size:16px;font-weight:600;margin-top:8px">${v.drawer.ciLow}% to ${v.drawer.ciHigh}%</div>
          <div style="font-size:11px;color:var(--ink-dim);margin-top:2px">${v.drawer.tierLabel}</div>
        </div>
      </div>

      <div style="padding:20px;border-bottom:1px solid var(--line)">
        <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Top contributing factors</div>
        <div style="display:flex;flex-direction:column;gap:10px">
          ${v.drawer.factors.map((f) => `
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
          ${v.drawer.analogs.map((an) => `
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
        <div style="border:1px solid ${v.drawer.tierColor};background:${v.drawer.tierColor};color:${v.drawer.tierText};padding:12px 14px;font-size:13px;font-weight:600">${v.drawer.action}</div>
      </div>

      <div style="padding:20px">
        <div class="face-display" style="font-size:13px;font-weight:600;margin-bottom:10px">Mark outcome</div>
        <div style="display:flex;gap:8px;margin-bottom:10px">
          <button class="actionbtn ${v.drawer.correctClass}" data-role="mark" data-outcome="correct">Correct</button>
          <button class="actionbtn ${v.drawer.incorrectClass}" data-role="mark" data-outcome="incorrect">Incorrect</button>
          <button class="actionbtn ${v.drawer.partialClass}" data-role="mark" data-outcome="partial">Partial</button>
        </div>
        <label for="outcome-note" style="display:block;font-size:11px;color:var(--ink-dim);margin-bottom:6px">Note for the feedback log (optional)</label>
        <textarea id="outcome-note" data-role="note" placeholder="e.g. nowcast confirmed guidance was reliable" rows="3" style="width:100%">${v.drawer.note}</textarea>
        ${v.drawer.hasMark ? `<div style="font-size:12px;color:var(--ink-dim);margin-top:8px">Logged as <strong>${v.drawer.markedLabel}</strong> for this session. Visible to the duty team in the Feedback Log.</div>` : ''}
      </div>
    </div>` : ''}
    `;
  }

  bind(root) {
    root.addEventListener('click', (e) => {
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
    // (a full re-render would blow away the textarea's cursor position).
    root.addEventListener('input', (e) => {
      if (e.target.dataset.role === 'note') this.state.noteDraft = e.target.value;
    });
  }
}

new Dashboard(document.getElementById('app'), { day: 3, selectedId: null, outcomes: {}, noteDraft: '' });
