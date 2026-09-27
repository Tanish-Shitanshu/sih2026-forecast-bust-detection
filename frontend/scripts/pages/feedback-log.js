import { Screen } from '../lib/screen.js';
import { ROWS, OUTCOME_LABELS } from '../data/feedback-rows.js';
import { computeFeedbackStats } from '../lib/feedback-stats.js';
import { renderNav } from '../components/nav.js';

class FeedbackLog extends Screen {
  renderVals() {
    const { filter } = this.state;
    const filters = [
      { key: 'all', label: 'All' },
      { key: 'correct', label: 'Correct' },
      { key: 'incorrect', label: 'Incorrect' },
      { key: 'partial', label: 'Partial' },
    ].map((f) => ({ key: f.key, label: f.label, activeClass: filter === f.key ? 'active' : '' }));

    const rows = ROWS.filter((r) => filter === 'all' || r.outcome === filter).map((r) => ({
      date: r.date, region: r.region, leadDay: r.leadDay, predicted: r.predicted,
      outcomeClass: r.outcome, outcomeLabel: OUTCOME_LABELS[r.outcome], by: r.by,
    }));

    return { filters, rows, stats: computeFeedbackStats() };
  }

  template(v) {
    return `
    ${renderNav({
      active: 'feedback-log',
      homeHref: '../index.html', dashboardHref: 'dashboard.html', modelTrustHref: 'model-trust.html',
      feedbackLogHref: 'feedback-log.html', methodologyHref: 'methodology.html',
      showTimestamp: false,
    })}

    <div style="flex:0 0 auto;padding:24px;border-bottom:1px solid var(--line);background:#fff;display:flex;align-items:center;gap:32px">
      <div>
        <div class="face-display" style="font-size:20px;font-weight:700">Feedback log</div>
        <div style="font-size:12px;color:var(--ink-dim)">Outcomes marked by duty forecasters against flagged predictions</div>
      </div>
      <div style="display:flex;gap:24px;margin-left:auto">
        <div>
          <div class="face-mono" style="font-size:22px;font-weight:600" data-stat="fb-total" data-value="${v.stats.total}">${v.stats.total}</div>
          <div style="font-size:11px;color:var(--ink-dim)">Entries logged</div>
        </div>
        <div>
          <div class="face-mono" style="font-size:22px;font-weight:600;color:var(--imd-green)" data-stat="fb-correct" data-value="${v.stats.correctPct}" data-suffix="%">${v.stats.correctPct}%</div>
          <div style="font-size:11px;color:var(--ink-dim)">Confirmed correct</div>
        </div>
        <div>
          <div class="face-mono" style="font-size:22px;font-weight:600;color:var(--imd-red)" data-stat="fb-incorrect" data-value="${v.stats.incorrectPct}" data-suffix="%">${v.stats.incorrectPct}%</div>
          <div style="font-size:11px;color:var(--ink-dim)">Confirmed incorrect</div>
        </div>
      </div>
    </div>

    <div style="flex:0 0 auto;padding:16px 24px;border-bottom:1px solid var(--line);display:flex;gap:8px">
      ${v.filters.map((f) => `<button class="chip ${f.activeClass}" data-role="filter" data-key="${f.key}">${f.label}</button>`).join('')}
    </div>

    <div style="flex:1;padding:0 24px 24px">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Subdivision</th>
            <th>Lead day</th>
            <th>Predicted bust probability</th>
            <th>Outcome</th>
            <th>Submitted by</th>
          </tr>
        </thead>
        <tbody>
          ${v.rows.map((r) => `
            <tr>
              <td class="face-mono" style="color:var(--ink-dim)">${r.date}</td>
              <td style="font-weight:500">${r.region}</td>
              <td class="face-mono">D${r.leadDay}</td>
              <td class="face-mono">${r.predicted}%</td>
              <td><span class="badge ${r.outcomeClass}">${r.outcomeLabel}</span></td>
              <td style="color:var(--ink-dim)">${r.by}</td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>
    `;
  }

  bind(root) {
    root.addEventListener('click', (e) => {
      const chip = e.target.closest('[data-role="filter"]');
      if (chip) this.setState({ filter: chip.dataset.key });
    });
  }
}

new FeedbackLog(document.getElementById('app'), { filter: 'all' });
