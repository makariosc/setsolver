// Pure geometry shared by the pipeline (no DOM, no ONNX): testable in Node.

function signedArea(c) {
  let s = 0;
  for (let i = 0; i < 4; i++) s += c[i][0] * c[(i + 1) % 4][1] - c[i][1] * c[(i + 1) % 4][0];
  return s;
}
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** Corners -> clockwise (image coords), starting so edge 0->1 is a short edge (portrait TL,TR,BR,BL). */
export function portraitOrder(corners) {
  let c = corners.map((p) => [p[0], p[1]]);
  if (signedArea(c) < 0) c = c.reverse();
  const e = c.map((p, i) => dist(p, c[(i + 1) % 4]));
  const start = e[0] + e[2] <= e[1] + e[3] ? 0 : 1;
  return [0, 1, 2, 3].map((i) => c[(i + start) % 4]);
}

/** Homography mapping the 4 `from` points onto the 4 `to` points (3x3, row-major). */
export function homography(from, to) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = from[i], [u, v] = to[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  // Gaussian elimination with partial pivoting
  for (let col = 0; col < 8; col++) {
    let piv = col;
    for (let r = col + 1; r < 8; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]]; [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = 0; r < 8; r++) {
      if (r === col) continue;
      const f = A[r][col] / A[col][col];
      for (let k = col; k < 8; k++) A[r][k] -= f * A[col][k];
      b[r] -= f * b[col];
    }
  }
  const h = b.map((v, i) => v / A[i][i]);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

/**
 * How far a detected quad is from a real 57x88 mm card seen through a pinhole
 * camera (diagonal FOV ~70 deg, typical of phones): 0 = a perfect card shape
 * under some perspective. A homography from the card rectangle to the quad,
 * pre-multiplied by K^-1, has a rotation's first two columns (up to scale) when
 * the quad is a real card: score = their non-orthogonality + scale mismatch.
 * Used to trust readings of cards cut off by the photo edge only when their
 * (partly off-frame) corners still describe a plausible card.
 * Mirrors card_shape_error in src/setsolver/edge_gate.py.
 */
export function cardShapeError(corners, W, H, fovDeg = 70) {
  const f = (Math.hypot(W, H) / 2) / Math.tan((fovDeg * Math.PI) / 360);
  const cx = (W - 1) / 2, cy = (H - 1) / 2;
  let best = Infinity;
  for (const [a, b] of [[57, 88], [88, 57]]) {   // either side may be the long one
    const h = homography([[0, 0], [a, 0], [a, b], [0, b]], corners);
    const col = (j) => [(h[j] - cx * h[6 + j]) / f, (h[3 + j] - cy * h[6 + j]) / f, h[6 + j]];
    const c1 = col(0), c2 = col(1);
    const n1 = Math.hypot(...c1), n2 = Math.hypot(...c2);
    const dot = c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2];
    const err = Math.abs(dot) / (n1 * n2) + Math.abs(n1 - n2) / Math.max(n1, n2);
    if (Number.isFinite(err)) best = Math.min(best, err);
  }
  return best;
}

/** Short / long side of a quad (means of opposite sides). Mirrors setsolver/edge_gate.py side_ratio. */
export function sideRatio(c) {
  const e = c.map((p, i) => dist(p, c[(i + 1) % 4]));
  const a = (e[0] + e[2]) / 2, b = (e[1] + e[3]) / 2;
  return Math.min(a, b) / Math.max(a, b);
}

/** |sideRatio / reference - 1|: reference is the median of the photo's whole cards, else a real card's 57/88. */
export function aspectDev(corners, wholeCards) {
  const rs = wholeCards.map(sideRatio).sort((x, y) => x - y);
  const ref = rs.length ? (rs.length % 2 ? rs[(rs.length - 1) / 2] : (rs[rs.length / 2 - 1] + rs[rs.length / 2]) / 2) : 57 / 88;
  return Math.abs(sideRatio(corners) / ref - 1);
}

/** |sideRatio / 57:88 - 1|: only a (nearly) whole card has a real card's proportions. */
export const realCardDev = (corners) => Math.abs(sideRatio(corners) / (57 / 88) - 1);

export function centroid(corners) {
  return [corners.reduce((a, p) => a + p[0], 0) / 4, corners.reduce((a, p) => a + p[1], 0) / 4];
}
