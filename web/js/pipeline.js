// In-browser inference: card detector (YOLO pose, 4 corners per card) ->
// perspective-straightened 160x256 crops -> attribute classifier.
// Mirrors src/setsolver/crops.py so crops match what the classifier trained on.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { centroid, dist, homography, portraitOrder } from "./geometry.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// Multi-threaded WASM needs cross-origin isolation (see scripts/serve_web.py).
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

const MAX_SIDE = 2048;       // working resolution for detection + crops
const EDGE_MARGIN = 0.004;   // fraction of the short side: corners closer than this to the edge => cut off
// A card only counts toward sets if the detector is at least this sure it's a
// card AND the classifier is at least this sure of every attribute. Lower
// detections (down to meta.detector.conf) are still shown, marked rejected.
export const MIN_CONFIDENCE = 0.85;

let meta = null;
const sessions = {};
export const backends = {};

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // Content-Length is only a progress estimate: servers like GitHub Pages gzip the
  // models, and then it is the *compressed* size while the body arrives decoded.
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(Math.min(got / total, 0.99));
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf;
}

async function createSession(bytes) {
  // Prefer WebGPU; fall back to WASM (CPU) where WebGPU is missing or fails.
  const attempts = "gpu" in navigator ? [["webgpu", "wasm"], ["wasm"]] : [["wasm"]];
  let lastErr;
  for (const eps of attempts) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
      return [s, eps[0]];
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr;
}

/** Load both models. onProgress(fraction 0..1, label). */
export async function loadModels(onProgress) {
  meta = await (await fetch("models/meta.json")).json();
  const parts = [["detector", meta.detector.file], ["classifier", meta.classifier.file]];
  for (const [i, [name, file]] of parts.entries()) {
    const bytes = await fetchWithProgress(`models/${file}`, (f) => onProgress?.((i + f * 0.9) / parts.length, `Loading ${name}…`));
    onProgress?.((i + 0.95) / parts.length, `Starting ${name}…`);
    [sessions[name], backends[name]] = await createSession(bytes);
  }
  onProgress?.(1, "Ready");
  return meta;
}

/**
 * Straighten one card into a W x H portrait crop (RGBA ImageData).
 * `src` is the working canvas; outside the photo is filled black.
 */
export function warpCard(src, corners, W = 160, H = 256) {
  const c = portraitOrder(corners);
  const shrink = dist(c[0], c[1]) / W;
  // When shrinking a lot, let the browser downscale the card's neighbourhood
  // first (smoothed resampling) so stripes don't alias - the counterpart of
  // the Gaussian pre-blur in crops.py.
  const f = shrink > 1.5 ? 1 / shrink : 1;
  const xs = c.map((p) => p[0]), ys = c.map((p) => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)) - 2), y0 = Math.max(0, Math.floor(Math.min(...ys)) - 2);
  const x1 = Math.min(src.width, Math.ceil(Math.max(...xs)) + 3), y1 = Math.min(src.height, Math.ceil(Math.max(...ys)) + 3);
  const rw = Math.max(1, Math.round((x1 - x0) * f)), rh = Math.max(1, Math.round((y1 - y0) * f));
  const region = new OffscreenCanvas(rw, rh);
  const rctx = region.getContext("2d", { willReadFrequently: true });
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = "high";
  rctx.drawImage(src, x0, y0, x1 - x0, y1 - y0, 0, 0, rw, rh);
  const rd = rctx.getImageData(0, 0, rw, rh).data;
  const sx = rw / (x1 - x0), sy = rh / (y1 - y0);
  const srcPts = c.map((p) => [(p[0] - x0) * sx, (p[1] - y0) * sy]);
  // inverse map: crop pixel -> region pixel
  const Hm = homography([[0, 0], [W, 0], [W, H], [0, H]], srcPts);
  const out = new ImageData(W, H);
  const o = out.data;
  for (let v = 0; v < H; v++) {
    for (let u = 0; u < W; u++) {
      const d = Hm[6] * u + Hm[7] * v + Hm[8];
      const x = (Hm[0] * u + Hm[1] * v + Hm[2]) / d, y = (Hm[3] * u + Hm[4] * v + Hm[5]) / d;
      const i = (v * W + u) * 4;
      const xf = Math.floor(x), yf = Math.floor(y);
      if (xf < 0 || yf < 0 || xf + 1 >= rw || yf + 1 >= rh) { o[i + 3] = 255; continue; }  // black
      const ax = x - xf, ay = y - yf;
      const p00 = (yf * rw + xf) * 4, p10 = p00 + 4, p01 = p00 + rw * 4, p11 = p01 + 4;
      for (let ch = 0; ch < 3; ch++)
        o[i + ch] = (rd[p00 + ch] * (1 - ax) + rd[p10 + ch] * ax) * (1 - ay) + (rd[p01 + ch] * (1 - ax) + rd[p11 + ch] * ax) * ay;
      o[i + 3] = 255;
    }
  }
  return out;
}

// ---------- detector ----------

function letterbox(src, size) {
  const r = Math.min(size / src.width, size / src.height);
  const w = Math.round(src.width * r), h = Math.round(src.height * r);
  const px = Math.floor((size - w) / 2), py = Math.floor((size - h) / 2);
  const cv = new OffscreenCanvas(size, size);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, px, py, w, h);
  const d = ctx.getImageData(0, 0, size, size).data;
  const n = size * size, t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    t[i] = d[i * 4] / 255; t[n + i] = d[i * 4 + 1] / 255; t[2 * n + i] = d[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", t, [1, 3, size, size]), r, px, py };
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

/** Parse detector output into [{score, corners:[[x,y]x4]}] in letterboxed pixels. */
function parseDetections(out, conf, nk) {
  const [, a, b] = out.dims, d = out.data;
  const dets = [];
  if (b === 6 + nk * 3) {
    // end-to-end (NMS-free) export: [1, N, x1 y1 x2 y2 score cls k0x k0y k0v ...]
    for (let i = 0; i < a; i++) {
      const row = d.subarray(i * b, (i + 1) * b);
      if (row[4] < conf) continue;
      const corners = [];
      for (let k = 0; k < nk; k++) corners.push([row[6 + 3 * k], row[7 + 3 * k]]);
      dets.push({ score: row[4], box: [row[0], row[1], row[2], row[3]], corners });
    }
    return dets;
  }
  if (a === 5 + nk * 3) {
    // raw export: [1, 5 + 3k, anchors] with cx cy w h score k0x k0y k0v ..., needs NMS
    const n = b, cand = [];
    for (let j = 0; j < n; j++) {
      const s = d[4 * n + j];
      if (s < conf) continue;
      const cx = d[j], cy = d[n + j], w = d[2 * n + j], h = d[3 * n + j];
      const corners = [];
      for (let k = 0; k < nk; k++) corners.push([d[(5 + 3 * k) * n + j], d[(6 + 3 * k) * n + j]]);
      cand.push({ score: s, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], corners });
    }
    cand.sort((p, q) => q.score - p.score);
    for (const c of cand) if (dets.every((k) => iou(k.box, c.box) < 0.7)) dets.push(c);
    return dets;
  }
  throw new Error(`unexpected detector output shape [${out.dims}]`);
}

// ---------- full pipeline ----------

function toWorkingCanvas(bitmap) {
  const s = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const cv = document.createElement("canvas");
  cv.width = Math.round(bitmap.width * s);
  cv.height = Math.round(bitmap.height * s);
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, cv.width, cv.height);
  return cv;
}

/**
 * Detect, straighten and classify every card in a photo.
 * Returns { canvas, cards, timings, ... }; cards are in the canvas's coordinates.
 */
export async function analyze(bitmap, { conf = meta?.detector.conf ?? 0.5 } = {}) {
  const t = { start: performance.now() };
  const canvas = toWorkingCanvas(bitmap);
  const W = canvas.width, H = canvas.height;

  const size = meta.detector.imgsz, nk = meta.detector.keypoints;
  const lb = letterbox(canvas, size);
  t.prep = performance.now();
  const detOut = await sessions.detector.run({ [sessions.detector.inputNames[0]]: lb.tensor });
  t.detect = performance.now();
  const raw = parseDetections(detOut[sessions.detector.outputNames[0]], conf, nk);
  const margin = EDGE_MARGIN * Math.min(W, H);
  const cards = raw.map((det, i) => {
    const corners = det.corners.map(([x, y]) => [(x - lb.px) / lb.r, (y - lb.py) / lb.r]);
    const cutOff = corners.some(([x, y]) => x < margin || y < margin || x > W - 1 - margin || y > H - 1 - margin);
    return { id: i, score: det.score, corners, cutOff, detLow: det.score < MIN_CONFIDENCE };
  });
  // number cards top-to-bottom, left-to-right, so debug indices read naturally
  const rowH = 0.5 * Math.min(W, H) / Math.max(3, Math.sqrt(cards.length));
  cards.sort((a, b) => {
    const ca = centroid(a.corners), cb = centroid(b.corners);
    return Math.round(ca[1] / rowH) - Math.round(cb[1] / rowH) || ca[0] - cb[0];
  });
  cards.forEach((c, i) => (c.id = i));

  const [cw, ch] = [meta.classifier.input[1], meta.classifier.input[0]];
  for (const c of cards) c.crop = warpCard(canvas, c.corners, cw, ch);
  t.crop = performance.now();

  const todo = cards.filter((c) => !c.cutOff);
  if (todo.length) {
    const n = cw * ch, x = new Float32Array(todo.length * 3 * n);
    todo.forEach((c, b) => {
      const d = c.crop.data, off = b * 3 * n;
      for (let i = 0; i < n; i++) {
        x[off + i] = d[i * 4] / 255; x[off + n + i] = d[i * 4 + 1] / 255; x[off + 2 * n + i] = d[i * 4 + 2] / 255;
      }
    });
    const out = await sessions.classifier.run({
      [sessions.classifier.inputNames[0]]: new ort.Tensor("float32", x, [todo.length, 3, ch, cw]),
    });
    const p = out[sessions.classifier.outputNames[0]].data;  // [N, 4, 3] probabilities
    todo.forEach((c, b) => {
      c.idx = {}; c.p = {}; c.probs = {};
      meta.classifier.attrs.forEach(([name, values], a) => {
        const pr = Array.from(p.subarray((b * 4 + a) * 3, (b * 4 + a) * 3 + 3));
        const k = pr.indexOf(Math.max(...pr));
        c.idx[name] = k; c[name] = values[k]; c.p[name] = pr[k]; c.probs[name] = pr;
      });
      const least = Object.entries(c.p).sort((u, v) => u[1] - v[1])[0];
      c.minP = least[1]; c.leastSure = least[0];
      c.lowConf = c.minP < MIN_CONFIDENCE;
    });
  }
  t.classify = performance.now();
  for (const c of cards) {
    c.rejected = c.cutOff || c.detLow || !!c.lowConf;
    c.rejectReason = c.cutOff ? "cut off by the photo edge"
      : c.detLow ? `detection ${c.score.toFixed(2)} < ${MIN_CONFIDENCE}`
      : c.lowConf ? `${c.leastSure} ${c.minP.toFixed(2)} < ${MIN_CONFIDENCE}` : null;
  }
  return {
    canvas, cards, original: [bitmap.width, bitmap.height],
    timings: { prepare: t.prep - t.start, detect: t.detect - t.prep, crop: t.crop - t.detect, classify: t.classify - t.crop, total: t.classify - t.start },
    backends: { ...backends }, conf,
  };
}

export { centroid };
