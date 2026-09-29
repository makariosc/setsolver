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

export function centroid(corners) {
  return [corners.reduce((a, p) => a + p[0], 0) / 4, corners.reduce((a, p) => a + p[1], 0) / 4];
}
