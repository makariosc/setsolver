# setsolver

Photographs of SET cards → detected cards → valid sets, using ML models trained on
procedurally generated, photorealistic synthetic photos with exact labels.

This repo currently contains the **synthetic data generator** (`src/setsolver/synth`).

## Quick start

```sh
uv run scripts/generate.py -n 48 --out data/synth/preview    # images + JSON labels
uv run scripts/preview.py data/synth/preview                 # label overlays + contact sheets
```

Useful flags: `--seed`, `--start`, `--workers`, `--width/--height`, and filters
`--only-backgrounds rug,wood`, `--only-rigs dim_lamp`, `--only-layouts grid`,
`--only-distractors envelope,book`, `--distractor-prob 1`. `--supersample 1`
renders ~3.5x faster than the default 2 and is plenty for detector training
(the model sees 640 px); keep 2 for scenes that feed classifier crops.
Image `i` is generated from seed `(seed, i)`, so any single image is reproducible.

## How a scene is made

Every scene is a real camera looking at one flat table (the plane z = 0, in mm):

1. **Deck style** (`cards.py`): sampled once per scene — ink palette (classic
   red/green/purple or orange/teal/purple), stripe pitch/width, stroke weight,
   symbol size, paper tint. Cards are drawn procedurally from vector shapes;
   the squiggle outline is traced from a photo of a real card.
2. **Layout** (`layout.py`): card positions/rotations on the table in mm.
3. **Camera** (`camera.py`): phone-like FOV, mostly near-top-down tilt, usually
   square to a grid. On the plane, projection is one homography
   `H = K[r1 r2 t]` shared by everything, so perspective is consistent.
   Framing is solved in image space (zoom + pan until the cards' bounding box
   hits a target occupancy), and the phone is usually turned to match the
   layout's shape. Modes (`SceneConfig`): tight, centered, fully visible
   (~83%); `loose_prob` 0.12 loosely framed with big margins; `clip_prob` 0.05
   cards cut off by the frame edge.
4. **Table surface** (`backgrounds/`): material maps (albedo, gloss) rendered in
   plane coordinates at the resolution the camera needs, then sampled per pixel.
5. **Cards** are warped through the same `H` (rendered at the on-screen
   resolution, 2x supersampled), with drop shadows and contact occlusion.
6. **Lighting** (`lighting.py`): eight rigs — neutral room LED, window
   daylight, overcast/diffuse, fluorescent, phone flash, warm room, dim warm
   lamp, mixed warm+cool — weighted so ~3/4 of scenes are white light. Soft
   photographer shadows and object shadows (cup, phone/book, bottle/arm,
   irregular blob, blinds/slats), plus Blinn-Phong glare. Shading is evaluated
   in linear RGB from each pixel's plane position, so cards and table share
   the same light.
7. **Occluders** (`occluders.py`, ~30% of scenes): fingers, coins, phones,
   pens, paper/sticky notes lying on or near the cards — they hide corners
   and edges, and paper is a white-rectangle hard negative.
8. **Distractors** (`distractors.py`, ~35% of scenes, 1–4 each): flat
   rectangles that are *not* SET cards — envelopes (front: address, stamp,
   postmark, barcode; back: flaps), books/magazines, e-readers/tablets/phones,
   playing cards (faces and backs), UNO/trading cards, ruled index cards,
   letters, receipts, business cards, photo prints, coasters. Textured with real
   glyphs, lit and shadowed like the cards, placed in free table space, across
   the frame edge, under the cards or occasionally on top. They carry no card
   labels, so the detector learns them as negatives. Tight shots with
   distractors are often framed a bit looser ("roomy") so they're in view.
9. **Camera pipeline** (`sensor.py`): defocus/motion blur, vignetting, auto
   exposure, noise, *partial* auto white balance, tone curve, chroma denoise,
   sharpening, JPEG.

## Labels (`labels/NNNNNN.json`)

Per card: `number`, `color`, `shape`, `shading`, `index` (0–80), `corners`
(image px, TL/TR/BR/BL of the card's portrait frame), `corner_visibility`
(per corner: `visible` / `occluded` / `outside`), `visible_fraction`
(occlusion + out-of-frame), `clipped` (a corner lies outside the image),
`px_per_mm`. Plus `distractors` (type, layer, image corners) and all scene parameters (camera K/R/C, framing mode, lights,
background, deck style, sensor settings) for debugging and slicing evaluations.

## Card detector (box + 4 corners)

```sh
# data: generate train/val, then convert to an Ultralytics pose dataset
uv run scripts/generate.py -n 10000 --seed 1 --out data/synth/train --width 960 --height 1280
uv run scripts/generate.py -n 1000  --seed 2 --out data/synth/val   --width 960 --height 1280
uv run scripts/export_yolo.py data/synth/train data/synth/val --out data/yolo/cards

# train (separate env so the generator's env is untouched)
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --group train
.venv-train/bin/python scripts/train_detector.py data/yolo/cards/data.yaml --model yolo26n-pose.pt --device 0
```

**Training dashboard** (`scripts/dashboard.py` + `dashboard.html`, stdlib only):
charts every run under `runs/` (losses, validation mAP, learning rate, per-epoch
table, validation prediction images) and shows live progress of the run in
flight from the newest `logs/*.log`, plus GPU stats. Refreshes every 10 s.

```sh
tmux new -d -s setsolver-dash ".venv-train/bin/python scripts/dashboard.py --host <tailscale-ip> --port 8765"
```

For live progress, tee the training output into `logs/` (e.g. `... | tee logs/train_<name>.log`).

**Predictions on real photos** (`/predictions` on the same server): run the
detector on a folder of photos and browse the overlays next to the originals,
with per-photo counts against known card counts.

```sh
.venv-train/bin/python scripts/predict_detector.py runs/pose/runs/detector/<run>/weights/best.pt \
    data/real/setchecker_samples --classifier runs/classify/<run>/weights/best.pt \
    --truth data/real/setchecker_counts.json --card-truth data/real/setchecker_cards.json \
    --out preds/<name> --device 0
```

With `--classifier`, each detected card is straightened, classified and
labeled on the overlay; a per-photo sheet of card crops with predictions and
their lowest attribute probability goes to `<out>/cards/`. Cards touching the
photo edge are marked "cut off" and not classified.

`export_yolo.py` orders each card's corners by its visible orientation (a card
looks the same rotated 180°), marks hidden corners as occluded-but-labeled
(trained) and out-of-frame corners as unlabeled, and drops cards less than 35%
visible. Because of that ordering, training disables flips and rotations.

**False positives on distractors** (`scripts/eval_distractors.py`): runs an
ONNX detector over scenes generated with `--distractor-prob 1` and counts, per
distractor type, how many the detector mistakes for a card.

`tests/test_geometry.py` guards the card-to-image mapping (no mirroring, corners
match labels): `uv run pytest`.

## Card classifier (number / color / shape / shading)

Detected cards are straightened into 160x256 portrait crops
(`src/setsolver/crops.py`, shared with inference) and classified by an
ImageNet-pretrained EfficientNet-B0 with four 3-way heads
(`src/setsolver/classifier.py`).

```sh
# crops from generated scenes (parallel; ~1000 scenes/s on 16 cores)
uv run scripts/extract_crops.py data/synth/train --out data/crops/train                 # corners jittered 1.5%
uv run scripts/extract_crops.py data/synth/val   --out data/crops/val --jitter 0        # clean evaluation
uv run scripts/extract_crops.py data/synth/val   --out data/crops/val_jitter

# train on the GPU box (results.csv / progress.json show up in the dashboard)
.venv-train/bin/python scripts/train_classifier.py --backbone efficientnet_b0 --epochs 15 --name effb0
```

The extractor skips cards cut off by the photo edge and cards less than 70%
visible: their labels can't be read from the crop (whole symbols missing). In
the app, cut-off cards are flagged for a retake instead of classified.
Out-of-frame areas are filled black (never edge-replicated, which smears
symbols into fake shapes). Training rotates crops by 180° (a card reads the
same upside down) but never mirrors them (a mirrored squiggle isn't a card).
Each run writes `slices.csv` (accuracy by lighting, background, card size,
occlusion...) and `val_errors.jpg` (misclassified examples).

## Browser app (`web/`)

Take or upload a photo; the detector and classifier run in the browser with
ONNX Runtime Web (WebGPU, falling back to WASM), so photos never leave the
device. Results are spoiler-safe: the app shows how many cards it found, and
only reveals whether there's a set (and which) on request, with a progressive
hint (1, 2, then 3 cards of a set). Debug mode shows every detection, the
straightened crop the classifier saw, per-attribute probabilities, timings and
backends. Cards touching the photo edge are flagged "cut off" and not counted.

```sh
# export the trained models (on the GPU box), checking ONNX against PyTorch on real photos
.venv-train/bin/python scripts/export_onnx.py \
    --detector runs/pose/runs/detector/<run>/weights/best.pt \
    --classifier runs/classify/<run>/weights/best.pt \
    --photos data/real/setchecker_samples --out web/models

# serve locally (adds the headers multi-threaded WASM needs); --host 0.0.0.0 for a phone on the LAN
uv run scripts/serve_web.py
node web/tests/test.mjs   # set-finding + geometry tests
```

**Model precision** (`scripts/quantize_models.py`, `scripts/compare_precision.py`,
`web/bench.html`): fp16 halves the download with identical answers; int8 is not
worth it (dynamic int8 breaks the classifier; static int8 changes a few cards
and is slower than fp32 on WebGPU). fp16 keeps GlobalAveragePool in fp32:
WebGPU accumulates in fp16 and the classifier then misreads nearly every card.

**Live site (GitHub Pages):** https://makariosc.github.io/setsolver/ — deployed by
`.github/workflows/pages.yml` on every push to `main` that touches `web/`. It
publishes the app plus the two models `web/models/meta.json` names (the fp16
pair, committed to the repo; other exports stay git-ignored). To ship a new
model: export it, run `scripts/quantize_models.py --install fp16`, commit
`web/models/{meta.json,detector_fp16.onnx,classifier_fp16.onnx}`, push.
Pages can't send the COOP/COEP headers, so the WASM fallback runs
single-threaded there; WebGPU (the normal path) is unaffected.

`web/js/pipeline.js` mirrors `src/setsolver/crops.py` (same corner ordering
and straightening), so browser crops match what the classifier was trained on.

## Extending

Components are plugins registered with a weight (`registry.py`); scenes sample
them by weight.

- **Background:** add a module to `src/setsolver/synth/backgrounds/` with a
  class decorated `@BACKGROUNDS.register("name", weight=1.0)` implementing
  `render(region: PlaneRegion, rng) -> SurfaceMaps`. It is auto-discovered.
  `region.grid_mm()` gives plane coordinates; `region.px(mm)` converts sizes.
- **Lighting rig:** a function `(rng, ctx) -> LightingSetup` decorated
  `@RIGS.register("name")` in `lighting.py`.
- **Layout:** a function `(rng) -> list[Placement]` decorated
  `@LAYOUTS.register("name")` in `layout.py`.
- **Distractor:** a function `(rng) -> Distractor` (size in mm + a texture
  renderer) decorated `@DISTRACTORS.register("name")` in `distractors.py`.
- **Occluder:** a function `(rng, ctx) -> Occluder` (a signed-distance shape
  on the table + color) decorated `@OCCLUDERS.register("name")` in `occluders.py`.

## Known gaps / next steps

- Image-texture backgrounds (Poly Haven / ambientCG / own photos) — the
  interface is ready; needs a loader module.
- Soft surfaces (bed, couch) are still flat; per-card tilt/curl not modeled.
- No table edge / second surface in frame.
- Out-of-frame corners are not supervised (Ultralytics requires in-image
  keypoints); clipped cards learn only their visible corners.
- No digital (app/screenshot) card styles.
# setsolver
