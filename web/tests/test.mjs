// Node tests for the pure JS modules:  node web/tests/test.mjs
import assert from "node:assert/strict";
import { aspectDev, cardShapeError, centroid, homography, portraitOrder, sideRatio } from "../js/geometry.js";
import { readFileSync } from "node:fs";
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

// --- cardShapeError: a real card seen through a pinhole camera scores ~0
const project = (tiltDeg, W = 1200, H = 900, fov = 70) => {
  // 57x88 mm card on a table, camera tilted about the x axis, card at the image centre
  const f = (Math.hypot(W, H) / 2) / Math.tan((fov * Math.PI) / 360), t = (tiltDeg * Math.PI) / 180, dist = 400;
  return [[-28.5, -44], [28.5, -44], [28.5, 44], [-28.5, 44]].map(([x, y]) => {
    const yc = y * Math.cos(t), zc = dist + y * Math.sin(t);
    return [W / 2 + (f * x) / zc, H / 2 + (f * yc) / zc];
  });
};
test("cardShapeError: true card shapes score ~0 at any tilt", () => {
  for (const tilt of [0, 15, 30, 40]) assert.ok(cardShapeError(project(tilt), 1200, 900) < 0.02, `tilt ${tilt}`);
});
test("cardShapeError: a card squashed against the edge scores high", () => {
  const q = project(10);
  const squashed = q.map(([x, y]) => [Math.min(x, 610), y]);   // right half clamped to x=610
  assert.ok(cardShapeError(squashed, 1200, 900) > 0.15);
});
test("sideRatio / aspectDev", () => {
  assert.ok(Math.abs(sideRatio(card(30)) - 171 / 267) < 1e-9);
  assert.ok(aspectDev(card(30), [card(0), card(70)]) < 1e-9);                       // same proportions as the others
  const clipped = card(0).map(([x, y]) => [x, Math.min(y, 300)]);                   // bottom half squashed off
  assert.ok(aspectDev(clipped, [card(0), card(70)]) > 0.15);
  assert.ok(Math.abs(aspectDev(card(0), []) - Math.abs((171 / 267) / (57 / 88) - 1)) < 1e-9);  // no whole cards: real card ratio
});
test("cardShapeError matches src/setsolver/edge_gate.py card_shape_error", () => {
  for (const c of JSON.parse(readFileSync(new URL("./shape_cases.json", import.meta.url))))
    assert.ok(Math.abs(cardShapeError(c.q, c.w, c.h) - c.err) < 1e-4, `${cardShapeError(c.q, c.w, c.h)} vs ${c.err}`);
});

console.log(`${passed} tests passed`);
