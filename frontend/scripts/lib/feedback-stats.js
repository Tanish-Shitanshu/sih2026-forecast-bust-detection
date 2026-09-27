import { ROWS } from '../data/feedback-rows.js';

// Single source of truth for the Feedback Log's aggregate numbers, so every
// screen that cites them (the Feedback Log page and the landing page's
// stats bar) reads the exact same computation over the exact same dataset.
export function computeFeedbackStats() {
  const total = ROWS.length;
  const correctPct = Math.round(100 * ROWS.filter((r) => r.outcome === 'correct').length / total);
  const incorrectPct = Math.round(100 * ROWS.filter((r) => r.outcome === 'incorrect').length / total);
  return { total, correctPct, incorrectPct };
}
