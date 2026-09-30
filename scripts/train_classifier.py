"""Train the card attribute classifier on straightened crops.

    .venv-train/bin/python scripts/train_classifier.py --name effb0 \
        --train data/crops/train --val data/crops/val --val-jitter data/crops/val_jitter

Each of --train/--val/--val-jitter takes one crop folder or several,
comma-separated (e.g. --train data/crops/train,data/crops/train_v3).

Writes runs/classify/<name>/:
  results.csv     per-epoch losses/accuracies (charted by scripts/dashboard.py)
  progress.json   live in-epoch progress for the dashboard
  args.yaml       run settings
  weights/        best.pt (by clean-val card accuracy) and last.pt
  slices.csv      final accuracy broken down by lighting, background, card size, occlusion...
  val_errors.jpg  a grid of misclassified validation crops (true vs predicted)

Augmentation: 180-degree rotation (a card reads the same upside down), small
affine jitter (on top of the extractor's corner noise), mild brightness /
contrast / saturation and a very small hue shift (color is a class, so hue
stays nearly fixed), occasional blur. No mirroring: a mirrored squiggle is not
a real card.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import v2

from setsolver.classifier import ATTRS, MEAN, STD, CardClassifier
from setsolver.crops import add_glare

ROOT = Path(__file__).resolve().parent.parent


class Crops(Dataset):
    def __init__(self, roots: list[Path], train: bool, limit: int | None = None, glare_prob: float = 0.0):
        self.glare_prob = glare_prob
        self.rows = []
        for root in roots:
            rows = [json.loads(l) for l in (root / "index.jsonl").open()][:limit]
            for r in rows:
                r["_path"] = str(root / "images" / r["file"])
            self.rows += rows
        self.y = torch.tensor([[values.index(r[name]) for name, values in ATTRS] for r in self.rows])
        norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        if train:
            self.tf = v2.Compose([
                v2.RandomAffine(degrees=2, translate=(0.03, 0.02), scale=(0.95, 1.05)),
                v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.2))], p=0.2),
                *norm,
            ])
        else:
            self.tf = v2.Compose(norm)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = decode_jpeg(read_file(self.rows[i]["_path"]), mode=ImageReadMode.RGB)
        if self.train and torch.rand(()) < 0.5:
            img = img.flip(-1).flip(-2)  # 180-degree rotation (not a mirror)
        if self.train and torch.rand(()) < self.glare_prob:
            rng = np.random.default_rng(int(torch.randint(0, 2**62, ())))  # per-worker torch seed -> distinct patches
            img = torch.from_numpy(add_glare(img.permute(1, 2, 0).numpy(), rng)).permute(2, 0, 1).contiguous()
        return self.tf(img), self.y[i], i


def evaluate(model, loader, device) -> dict:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss = torch.zeros(len(ATTRS))
    correct = torch.zeros(len(ATTRS))
    preds, n = [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device)
            logits = model(x).float()
            for a in range(len(ATTRS)):
                loss[a] += ce(logits[:, a], y[:, a]).item()
            p = logits.argmax(-1)
            correct += (p == y).sum(0).cpu()
            preds.append(p.cpu())
            n += len(y)
    return {"loss": loss / n, "acc": correct / n, "pred": torch.cat(preds)}


def write_error_grid(ds: Crops, pred: torch.Tensor, path: Path, max_n: int = 64):
    wrong = (pred != ds.y).any(1).nonzero().flatten().tolist()
    if not wrong:
        return
    rng = np.random.default_rng(0)
    pick = rng.permutation(wrong)[:max_n]
    cells = []
    for i in pick:
        r = ds.rows[i]
        im = cv2.imread(r["_path"])
        cap = np.full((44, im.shape[1], 3), 255, np.uint8)
        t = " ".join(str(r[name]) for name, _ in ATTRS)
        p = " ".join(str(values[pred[i, a]]) if pred[i, a] != ds.y[i, a] else "·" for a, (name, values) in enumerate(ATTRS))
        cv2.putText(cap, t, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 110, 0), 1, cv2.LINE_AA)
        cv2.putText(cap, p, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(cap, f"{r['px_per_mm']:.1f}px/mm {r['rig']}", (3, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (110, 110, 110), 1, cv2.LINE_AA)
        cells.append(cv2.copyMakeBorder(np.vstack([im, cap]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    cols = 8
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    grid = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_slices(ds: Crops, pred: torch.Tensor, path: Path) -> list[list]:
    ok_card = (pred == ds.y).all(1).numpy()
    ok_attr = (pred == ds.y).numpy()
    groups = defaultdict(list)
    for i, r in enumerate(ds.rows):
        p = r["px_per_mm"]
        size = "<1.5 px/mm" if p < 1.5 else "1.5-2.5 px/mm" if p < 2.5 else "2.5-4 px/mm" if p < 4 else ">=4 px/mm"
        keys = [("all", "all"), ("rig", r["rig"]), ("background", r["background"]), ("palette", r["palette"]),
                ("card size", size), ("corners hidden", "yes" if r["corners_occluded"] else "no"),
                ("visible", "70-90%" if r["visible_fraction"] < 0.9 else ">=90%"),
                ("shading", r["shading"]), ("color", r["color"])]
        for k in keys:
            groups[k].append(i)
    rows = []
    for (dim, val), idx in sorted(groups.items(), key=lambda kv: (kv[0][0] != "all", kv[0])):
        idx = np.array(idx)
        rows.append([dim, val, len(idx), ok_card[idx].mean(), *ok_attr[idx].mean(0)])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice", "value", "n", "card_acc", *[f"{n}_acc" for n, _ in ATTRS]])
        for r in rows:
            w.writerow([r[0], r[1], r[2], *[f"{v:.4f}" for v in r[3:]]])
    return rows


def main():
    paths = lambda s: [Path(p) for p in s.split(",")]  # noqa: E731
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=paths, default=[ROOT / "data/crops/train"])
    ap.add_argument("--val", type=paths, default=[ROOT / "data/crops/val"])
    ap.add_argument("--val-jitter", type=paths, default=[ROOT / "data/crops/val_jitter"])
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--project", type=Path, default=ROOT / "runs/classify")
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N crops of each set (smoke tests)")
    ap.add_argument("--glare-prob", type=float, default=0.3,
                    help="fraction of training crops given a painted glare patch (crops.add_glare)")
    a = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = a.project / (a.name or a.backbone)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    (out / "args.yaml").write_text("".join(f"{k}: {v}\n" for k, v in {
        **{k: str(v) for k, v in vars(a).items()}, "model": a.backbone, "task": "classify"}.items()))

    train_ds = Crops(a.train, True, a.limit, glare_prob=a.glare_prob)
    val_ds, jit_ds = Crops(a.val, False, a.limit), Crops(a.val_jitter, False, a.limit)
    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(train_ds, a.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, a.batch * 2, **kw)
    jit_dl = DataLoader(jit_ds, a.batch * 2, **kw)
    print(f"train {len(train_ds)}  val {len(val_ds)}  val_jitter {len(jit_ds)}  -> {out}", flush=True)

    model = CardClassifier(a.backbone).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * len(train_dl)
    warm = len(train_dl)  # one epoch of linear warmup, then cosine to ~0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ce = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    names = [n for n, _ in ATTRS]
    header = ["epoch", "time", *[f"train/{n}_loss" for n in names], *[f"val/{n}_loss" for n in names],
              *[f"metrics/acc_{n}" for n in names], "metrics/acc_card", "metrics/acc_card_jitter", "lr/pg0"]
    results = out / "results.csv"
    with results.open("w", newline="") as f:
        csv.writer(f).writerow(header)

    t_start, best = time.time(), -1.0
    for epoch in range(1, a.epochs + 1):
        model.train()
        run_loss = torch.zeros(len(ATTRS), device=device)
        t_ep, seen = time.time(), 0
        for b, (x, y, _) in enumerate(train_dl, 1):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            losses = torch.stack([ce(logits[:, i], y[:, i]) for i in range(len(ATTRS))])
            opt.zero_grad(set_to_none=True)
            losses.sum().backward()
            opt.step()
            sched.step()
            run_loss += losses.detach()
            seen += len(x)
            if b % 20 == 0 or b == len(train_dl):
                rate = seen / (time.time() - t_ep)
                (out / "progress.json").write_text(json.dumps({
                    "epoch": epoch, "epochs": a.epochs, "batch": b, "batches": len(train_dl),
                    "rate": f"{rate:.0f} img/s", "updated": time.time(),
                    "eta_epoch": f"{(len(train_dl) - b) * a.batch / rate:.0f}s",
                }))
        train_loss = (run_loss / len(train_dl)).cpu()

        ev, evj = evaluate(model, val_dl, device), evaluate(model, jit_dl, device)
        card = (ev["pred"] == val_ds.y).all(1).float().mean().item()
        card_j = (evj["pred"] == jit_ds.y).all(1).float().mean().item()
        with results.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, round(time.time() - t_start, 1),
                                    *[f"{v:.5f}" for v in train_loss.tolist()], *[f"{v:.5f}" for v in ev["loss"].tolist()],
                                    *[f"{v:.5f}" for v in ev["acc"].tolist()], f"{card:.5f}", f"{card_j:.5f}",
                                    f"{sched.get_last_lr()[0]:.6g}"])
        ckpt = {"model": model.state_dict(), "backbone": a.backbone, "epoch": epoch, "card_acc": card,
                "attrs": [n for n, _ in ATTRS], "input_size": [256, 160], "mean": MEAN, "std": STD}
        torch.save(ckpt, out / "weights" / "last.pt")
        if card > best:
            best = card
            torch.save(ckpt, out / "weights" / "best.pt")
        accs = "  ".join(f"{n} {v:.4f}" for n, v in zip(names, ev["acc"].tolist()))
        print(f"epoch {epoch}/{a.epochs}  {time.time() - t_ep:.0f}s  train loss {train_loss.sum():.4f}  "
              f"val loss {ev['loss'].sum():.4f}  card {card:.4f} (jitter {card_j:.4f})  {accs}", flush=True)

    # final analysis with the best checkpoint
    model.load_state_dict(torch.load(out / "weights" / "best.pt", weights_only=False)["model"])
    ev = evaluate(model, val_dl, device)
    write_error_grid(val_ds, ev["pred"], out / "val_errors.jpg")
    rows = write_slices(val_ds, ev["pred"], out / "slices.csv")
    print("\nclean-val accuracy by slice (card / number color shape shading):")
    for r in rows:
        print(f"  {r[0]:>14} {str(r[1]):<20} n={r[2]:<6} {r[3]:.4f} / " + " ".join(f"{v:.4f}" for v in r[4:]))
    (out / "progress.json").unlink(missing_ok=True)
    print(f"done in {(time.time() - t_start) / 60:.1f} min; best clean card accuracy {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
