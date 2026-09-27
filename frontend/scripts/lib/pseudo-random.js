// Deterministic pseudo-random in [0, 1), seeded by two numbers.
// Same formula the prototype used so every screen's mock data stays reproducible.
export function pseudo(a, b) {
  const x = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
  return x - Math.floor(x);
}
