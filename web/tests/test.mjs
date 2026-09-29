// Node tests for the pure JS modules:  node web/tests/test.mjs
import assert from "node:assert/strict";
import { centroid, homography, portraitOrder } from "../js/geometry.js";
import { completeSet, findSets, isSet } from "../js/sets.js";

let passed = 0;
const test = (name, fn) => { fn(); passed++; };

// --- sets: the full 81-card deck contains exactly 1080 sets
const deck = [];
for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++)
  deck.push({ id: deck.length, idx: { number: a, color: b, shape: c, shading: d } });
test("full deck has 1080 sets", () => assert.equal(findSets(deck).length, 1080));
test("every pair completes to a set", () => {
  for (let i = 0; i < 81; i += 7) for (let j = i + 1; j < 81; j += 5) {
    const idx = completeSet(deck[i], deck[j]);
    assert.ok(isSet(deck[i], deck[j], { idx }));
  }
});
test("known non-set", () => assert.equal(isSet(
  { idx: { number: 0, color: 0, shape: 0, shading: 0 } },
  { idx: { number: 0, color: 0, shape: 0, shading: 1 } },
  { idx: { number: 0, color: 0, shape: 0, shading: 1 } }), false));
test("uncertain flag propagates", () => {
  const cards = deck.slice(0, 3).map((c, i) => ({ ...c, uncertain: i === 1 }));  // differ only in shading: a set
  const s = findSets(cards);
  assert.equal(s.length, 1);
  assert.equal(s[0].uncertain, true);
});

// --- geometry: same rules as src/setsolver/crops.py portrait_order
const card = (deg, w = 171, h = 267, cx = 300, cy = 300) => {
  const a = (deg * Math.PI) / 180;
  return [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    .map(([x, y]) => [cx + x * Math.cos(a) - y * Math.sin(a), cy + x * Math.sin(a) + y * Math.cos(a)]);
};
const near = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6;
test("portraitOrder: any start, either direction", () => {
  for (const deg of [0, 17, 45, 90, 133, 180, 260, 315]) for (let s = 0; s < 4; s++) for (const rev of [false, true]) {
    let c = card(deg);
    c = [...c.slice(s), ...c.slice(0, s)];
    if (rev) c = c.reverse();
    const p = portraitOrder(c);
    const e = p.map((q, i) => Math.hypot(q[0] - p[(i + 1) % 4][0], q[1] - p[(i + 1) % 4][1]));
    assert.ok(e[0] < e[1] && e[2] < e[3], "edge 0->1 short");
    const t = card(deg);
    assert.ok([0, 2].some((k) => p.every((q, i) => near(q, t[(i + k) % 4]))), "TL..BR up to 180 deg");
  }
});
test("homography maps the 4 points exactly", () => {
  const from = [[0, 0], [160, 0], [160, 256], [0, 256]], to = card(23);
  const H = homography(from, to);
  from.forEach(([x, y], i) => {
    const d = H[6] * x + H[7] * y + H[8];
    assert.ok(near([(H[0] * x + H[1] * y + H[2]) / d, (H[3] * x + H[4] * y + H[5]) / d], to[i]));
  });
});
test("centroid", () => assert.ok(near(centroid(card(40)), [300, 300])));

console.log(`${passed} tests passed`);
