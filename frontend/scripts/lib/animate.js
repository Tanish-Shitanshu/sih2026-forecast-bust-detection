// Counts every [data-stat] element in `root` from its cached previous value
// (0 on first render) up to its current data-value, over ~500ms. Returns the
// updated value cache to pass back in on the next call.
export function animateCountUps(root, previousValues = {}, duration = 500) {
  const nextValues = { ...previousValues };

  root.querySelectorAll('[data-stat]').forEach((el) => {
    const key = el.dataset.stat;
    const target = Number(el.dataset.value);
    const suffix = el.dataset.suffix || '';
    const start = previousValues[key] !== undefined ? previousValues[key] : 0;
    nextValues[key] = target;

    if (start === target) {
      el.textContent = target + suffix;
      return;
    }

    // Set the starting frame synchronously, before the browser paints,
    // so there's no flash of the final value.
    el.textContent = start + suffix;
    const startTime = performance.now();

    function frame(now) {
      const t = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const val = Math.round(start + (target - start) * eased);
      el.textContent = val + suffix;
      if (t < 1) requestAnimationFrame(frame);
      else el.textContent = target + suffix;
    }
    requestAnimationFrame(frame);
  });

  return nextValues;
}
