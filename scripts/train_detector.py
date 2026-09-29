"""Fine-tune an Ultralytics pose model to detect cards as box + 4 corners.

    uv run --group train scripts/train_detector.py data/yolo/cards/data.yaml --model yolo26n-pose.pt
    uv run --group train scripts/train_detector.py data.yaml --model yolo26s-pose.pt --epochs 80 --batch 32

Augmentation choices (see export_yolo.py for the corner ordering):
- fliplr=0, degrees=0: flips/rotations would permute corners inconsistently with
  the orientation-canonical ordering. The generator already varies rotation,
  perspective and mirroring-free layouts, so we lose nothing.
- Mild HSV/scale/translate on top of the generator's photometric variety.
- Mosaic stays on early (more cards per batch, partial cards at tile seams) and
  is switched off for the last epochs so the model finishes on whole photos.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps (default: auto)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the train set (smoke tests)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="runs/detector")
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=str(a.data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        fraction=a.fraction,
        project=a.project,
        name=a.name or Path(a.model).stem,
        # geometry: keep corner order consistent
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        # photometric (the generator already does most of this)
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        cos_lr=True,
        patience=20,
        plots=True,
    )


if __name__ == "__main__":
    main()
