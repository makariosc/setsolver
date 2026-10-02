import { analyze, centroid, loadModels, MIN_CONFIDENCE } from "./pipeline.js";
import { findSets } from "./sets.js";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const SET_COLORS = 8;  // --set-1 .. --set-8, assigned in fixed order

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode etc. */ } },
};

const state = {
  ready: false,
  result: null,      // from analyze()
  sets: [],
  revealed: false,
  shown: new Set(),  // indices of sets whose cards are highlighted
  hint: 0,           // cards revealed by the hint button (0..3)
  focus: null,       // card id highlighted from the debug panel
  debug: store.get("setsolver.debug", false),
  autoReveal: store.get("setsolver.autoReveal", false),
};

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
function sv(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
const cardName = (c) => `${c.number} ${c.color} ${c.shape}${c.number > 1 ? "s" : ""}, ${c.shading}`;
const pts = (corners) => corners.map((p) => p.join(",")).join(" ");
const shrinkTo = (corners, f) => { const c = centroid(corners); return corners.map(([x, y]) => [c[0] + (x - c[0]) * f, c[1] + (y - c[1]) * f]); };

// ---------------- model loading ----------------

async function init() {
  $("debugToggle").checked = state.debug;
  $("autoReveal").checked = state.autoReveal;
  try {
    await loadModels((f, label) => {
      $("loadingBar").style.width = `${Math.round(f * 100)}%`;
      $("loadingText").textContent = f < 1 ? `${label} ${Math.round(f * 100)}%` : "Ready";
    });
    state.ready = true;
    $("loading").classList.add("done");
    if (state.pendingFile) handleFile(state.pendingFile);
  } catch (e) {
    showError(`Couldn't load the models (${e.message}). Check that web/models/ contains the exported ONNX files.`);
    $("loadingText").textContent = "Models failed to load";
  }
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = !msg;
}

// ---------------- input ----------------

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  showError("");
  if (!state.ready) {
    state.pendingFile = file;
    $("capture").hidden = true; $("results").hidden = false; $("resultPanel").hidden = true;
    showBusy(true);
    return;
  }
  state.pendingFile = null;
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    showError("Couldn't read that image. Try a JPEG or PNG.");
    return;
  }
  $("capture").hidden = true;
  $("results").hidden = false;
  $("resultPanel").hidden = true;   // no stale controls while the new photo is analyzed
  $("debugPanel").hidden = true;
  showBusy(true);
  // let the spinner paint before the (blocking) work starts; the timeout covers
  // background tabs, where requestAnimationFrame never fires
  await new Promise((r) => { requestAnimationFrame(() => setTimeout(r, 0)); setTimeout(r, 100); });
  try {
    const result = await analyze(bitmap);
    result.fileName = file.name;
    state.result = result;
    state.sets = findSets(result.cards.filter((c) => c.idx && !c.rejected));
    state.revealed = state.autoReveal;
    state.shown = new Set();
    state.hint = 0;
    state.focus = null;
    $("resultPanel").hidden = false;
    render();
  } catch (e) {
    console.error(e);
    showError(`Something went wrong while analyzing the photo: ${e.message}`);
  } finally {
    showBusy(false);
    bitmap.close?.();
  }
}

function showBusy(on) { $("busy").hidden = !on; }

for (const id of ["takeInput", "uploadInput"]) {
  $(id).addEventListener("change", (e) => { handleFile(e.target.files[0]); e.target.value = ""; });
}
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) handleFile(item.getAsFile());
});
$("newPhotoBtn").addEventListener("click", () => {
  state.result = null;
  $("results").hidden = true;
  $("capture").hidden = false;
  $("stageInner").replaceChildren();
  showError("");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// ---------------- controls ----------------

$("revealBtn").addEventListener("click", () => { if (!state.result) return; state.revealed = true; renderResult(); });
$("autoReveal").addEventListener("change", (e) => {
  state.autoReveal = e.target.checked;
  store.set("setsolver.autoReveal", state.autoReveal);
  if (state.autoReveal && state.result) { state.revealed = true; renderResult(); }
});
$("debugToggle").addEventListener("change", (e) => {
  state.debug = e.target.checked;
  store.set("setsolver.debug", state.debug);
  if (!state.debug) state.focus = null;
  if (state.result) render();
});
$("downloadJson").addEventListener("click", () => {
  const r = state.result;
  if (!r) return;
  const data = {
    file: r.fileName, original_size: r.original, working_size: [r.canvas.width, r.canvas.height],
    timings_ms: r.timings, backends: r.backends, detector_conf: r.conf,
    cards: r.cards.map(({ crop, ...c }) => c),
    sets: state.sets,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  el("a", { href: url, download: `${(r.fileName || "photo").replace(/\.[^.]+$/, "")}-setsolver.json` }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// ---------------- rendering ----------------

function render() {
  const r = state.result;
  const inner = $("stageInner");
  if (!inner.contains(r.canvas)) inner.replaceChildren(r.canvas);
  renderResult();
  renderDebug();
}

function renderResult() {
  const r = state.result;
  const cards = r.cards;
  const classified = cards.filter((c) => c.idx && !c.rejected);  // the cards that count
  const unsure = cards.filter((c) => c.rejected);

  $("cardCount").textContent = cards.length === 0 ? "No cards found"
    : `${classified.length} card${classified.length === 1 ? "" : "s"} found`;
  const chips = [];
  if (unsure.length) chips.push(el("span", { class: "chip warn" }, `? ${unsure.length} not read confidently`));
  $("chips").replaceChildren(...chips);

  const enough = classified.length >= 3;
  $("spoiler").hidden = state.revealed || !enough;
  $("answer").hidden = !(state.revealed && enough) && enough;

  if (!enough) {
    $("answer").hidden = false;
    $("headline").textContent = cards.length === 0 ? "No cards found" : "Not enough cards";
    $("subline").textContent = cards.length === 0
      ? "Try again from directly above, with all the cards in the frame and in reasonable light."
      : unsure.length
      ? `Some cards couldn't be read with at least ${Math.round(MIN_CONFIDENCE * 100)}% confidence (marked “?”). Retake closer or in better light.`
      : "A set needs three cards. Make sure all the cards are in the photo.";
    $("setControls").replaceChildren();
    $("hintText").textContent = "";
    drawOverlay();
    return;
  }

  if (state.revealed) {
    const n = state.sets.length;
    $("headline").textContent = n === 0 ? "No set here" : `Yes: ${n} set${n === 1 ? "" : "s"}`;
    const notes = [];
    if (n === 0) notes.push(cards.length >= 12 ? "Deal three more cards." : "No three of these cards make a set.");
    if (cut.length) notes.push(`${cut.length} card${cut.length === 1 ? " is" : "s are"} cut off and not counted; retake the photo to include ${cut.length === 1 ? "it" : "them"}.`);
    if (unsure.length) notes.push(`${unsure.length} card${unsure.length === 1 ? " couldn't" : "s couldn't"} be read with at least ${Math.round(MIN_CONFIDENCE * 100)}% confidence and ${unsure.length === 1 ? "isn't" : "aren't"} counted (marked “?”). Retake closer or in better light.`);
    $("subline").textContent = notes.join(" ");
    renderSetControls();
  }
  drawOverlay();
}

function renderSetControls() {
  const box = $("setControls");
  const n = state.sets.length;
  if (!n) { box.replaceChildren(); $("hintText").textContent = ""; return; }
  const btns = [];
  btns.push(el("button", {
    type: "button", class: "btn", "aria-pressed": String(state.hint > 0 && state.shown.size === 0),
    onclick: () => { state.hint = Math.min(3, state.hint + 1); state.shown.clear(); renderResult(); },
  }, state.hint === 0 ? "Hint" : state.hint < 3 ? "More hint" : "Hint (full set)"));
  state.sets.forEach((s, i) => {
    const color = `var(--set-${(i % SET_COLORS) + 1})`;
    btns.push(el("button", {
      type: "button", class: "btn", "aria-pressed": String(state.shown.has(i)),
      onclick: () => { state.shown.has(i) ? state.shown.delete(i) : state.shown.add(i); state.hint = 0; renderResult(); },
    }, el("span", { class: "swatch", style: `background:${color}` }), `Set ${i + 1}`));
  });
  if (n > 1) btns.push(el("button", {
    type: "button", class: "btn",
    onclick: () => { state.shown = state.shown.size === n ? new Set() : new Set(state.sets.map((_, i) => i)); state.hint = 0; renderResult(); },
  }, state.shown.size === n ? "Hide all" : "Show all"));
  box.replaceChildren(...btns);
  $("hintText").textContent = state.hint === 0 ? "" : state.hint === 1
    ? "One card that's part of a set is highlighted."
    : state.hint === 2 ? "Two cards of the same set are highlighted: find the third." : "The whole set is highlighted.";
}

function drawOverlay() {
  const r = state.result;
  const W = r.canvas.width, H = r.canvas.height;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  const byId = new Map(r.cards.map((c) => [c.id, c]));

  // every detected card: a thin outline (not a spoiler), dashed if not counted
  for (const c of r.cards) {
    svg.append(sv("polygon", { points: pts(c.corners), class: "ov-halo" }));
    svg.append(sv("polygon", { points: pts(c.corners), class: `ov-card${c.rejected ? " ov-cut" : ""}` }));
  }

  // revealed sets: colored outlines, nested when a card belongs to several shown sets
  const layers = new Map();
  [...state.shown].sort((a, b) => a - b).forEach((si) => {
    const color = `var(--set-${(si % SET_COLORS) + 1})`;
    for (const id of state.sets[si].ids) {
      const layer = layers.get(id) || 0;
      layers.set(id, layer + 1);
      const poly = shrinkTo(byId.get(id).corners, 1 - 0.07 * layer);
      svg.append(sv("polygon", { points: pts(poly), class: "ov-set", stroke: "rgba(0,0,0,.55)", "stroke-width": 9 }));
      svg.append(sv("polygon", { points: pts(poly), class: "ov-set", stroke: color }));
    }
  });

  // progressive hint: 1, 2 or 3 cards of the first set
  if (state.hint > 0 && state.sets.length) {
    for (const id of state.sets[0].ids.slice(0, state.hint))
      svg.append(sv("polygon", { points: pts(byId.get(id).corners), class: "ov-hint" }));
  }

  // labels: "?" on hard-to-read cards; card numbers + scores in debug mode
  // labels are drawn in image pixels; size them relative to the photo so they stay readable when scaled down
  const fontPx = Math.max(14, Math.round(Math.min(W, H) / 24));
  for (const c of r.cards) {
    const [cx, cy] = centroid(c.corners);
    let text = "";
    if (state.debug) text = `#${c.id + 1}${c.rejected ? " ?" : ""}`;
    else if (c.rejected) text = "?";  // says nothing about sets, so not a spoiler
    if (!text) continue;
    const t = sv("text", { x: cx, y: cy, "text-anchor": "middle", "dominant-baseline": "middle", class: "ov-label", "font-size": fontPx });
    t.textContent = text;
    svg.append(t);
  }
  if (state.focus != null && byId.has(state.focus)) {
    svg.append(sv("polygon", { points: pts(byId.get(state.focus).corners), class: "ov-focus" }));
  }

  const inner = $("stageInner");
  inner.querySelector("svg")?.remove();
  inner.append(svg);
}

function renderDebug() {
  const panel = $("debugPanel");
  panel.hidden = !state.debug;
  if (!state.debug) return;
  const r = state.result;
  const t = r.timings;
  const stat = (k, v, d) => el("div", { class: "stat" }, el("div", { class: "k" }, k), el("div", { class: "v" }, v), d ? el("div", { class: "d" }, d) : null);
  $("debugStats").replaceChildren(
    stat("Total", `${t.total.toFixed(0)} ms`, `prepare ${t.prepare.toFixed(0)} · detect ${t.detect.toFixed(0)} · crop ${t.crop.toFixed(0)} · classify ${t.classify.toFixed(0)} ms`),
    stat("Backends", `${r.backends.detector} / ${r.backends.classifier}`, "detector / classifier"),
    stat("Image", `${r.original[0]}×${r.original[1]}`, `analyzed at ${r.canvas.width}×${r.canvas.height}`),
    stat("Detections", String(r.cards.length), `shown ≥ ${r.conf} · counted ≥ ${MIN_CONFIDENCE} · ${r.cards.filter((c) => c.rejected).length} rejected`),
  );

  const cards = $("debugCards");
  cards.replaceChildren(...r.cards.map((c) => {
    const cv = document.createElement("canvas");
    cv.width = c.crop.width; cv.height = c.crop.height;
    cv.getContext("2d").putImageData(c.crop, 0, 0);
    const kids = [cv, el("div", { class: "ct-head" }, `#${c.id + 1} · det ${c.score.toFixed(2)}`)];
    if (!c.idx) {
      kids.push(el("div", { class: "flag warn" }, "not classified"));
    } else {
      kids.push(el("div", { class: "ct-pred" }, cardName(c)));
      if (c.rejected) kids.push(el("div", { class: "flag warn" }, `not counted: ${c.rejectReason}`));
      if (c.atEdge) kids.push(el("div", { class: "flag" }, "at the photo edge"));
      const probs = el("div", { class: "probs" });
      for (const [name] of Object.entries(c.probs)) {
        const p = c.p[name];
        probs.append(el("div", { class: "prob" }, name,
          el("div", { class: `pb${p < MIN_CONFIDENCE ? " low" : ""}` }, el("div", { style: `width:${(p * 100).toFixed(0)}%` })),
          el("b", {}, p.toFixed(2))));
      }
      kids.push(probs);
    }
    return el("button", {
      type: "button", class: `card-tile${state.focus === c.id ? " active" : ""}`,
      "aria-label": `Card ${c.id + 1}: ${c.idx ? cardName(c) : "not read"}${c.rejected ? ", not counted" : ""}`,
      onclick: () => { state.focus = state.focus === c.id ? null : c.id; drawOverlay(); renderDebug(); },
    }, ...kids);
  }));

  $("debugSets").replaceChildren(state.sets.length
    ? el("ul", {}, ...state.sets.map((s, i) => el("li", {}, `Set ${i + 1}: cards ${s.ids.map((id) => `#${id + 1}`).join(", ")}`)))
    : "No sets among the counted cards.");
}

init();
