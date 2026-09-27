import { pseudo } from './pseudo-random.js';

// IMD warning scale: bust-risk color, text color, and label per tier.
export const TIER_META = {
  green: { color: 'var(--imd-green)', text: '#F5F7F4', label: 'Green, no warning' },
  yellow: { color: 'var(--imd-yellow)', text: '#12181C', label: 'Yellow, watch' },
  orange: { color: 'var(--imd-orange)', text: '#12181C', label: 'Orange, alert' },
  red: { color: 'var(--imd-red)', text: '#F5F7F4', label: 'Red, warning' },
};

export function bustFor(seed, day) {
  const v = pseudo(seed, day * 3.17) * 0.6 + pseudo(seed * 2.31, day * 1.11) * 0.4;
  return Math.min(0.97, Math.max(0.02, v));
}

export function tierKeyFor(bust) {
  if (bust < 0.18) return 'green';
  if (bust < 0.38) return 'yellow';
  if (bust < 0.62) return 'orange';
  return 'red';
}

export function actionFor(key) {
  if (key === 'green') return 'No action needed. Forecast reliable at this lead time.';
  if (key === 'yellow') return 'Cross-check short-range nowcast before briefing.';
  if (key === 'orange') return 'Escalate to duty forecaster for manual review.';
  return 'Escalate immediately. Treat forecast with strong caution.';
}
