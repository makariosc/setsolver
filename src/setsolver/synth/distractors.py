"""Flat rectangular things lying on the table that are *not* SET cards:
envelopes, books, e-readers/tablets, playing cards and other game cards, index
cards, letters, receipts, business cards, photo prints, coasters.

They are hard negatives for the detector: they have no labels, so a detection
on one is a false positive. Unlike occluders (occluders.py), which are placed
on top of the cards to hide corners, distractors are placed where the camera
sees them: in free table space, at the frame edge, under the cards (cards
lying on a book/envelope) or occasionally on top of one.

Each distractor is a textured rectangle drawn through the same camera,
shadows and lighting as the cards (scene.py). Textures are drawn in sRGB with
OpenCV (so text is real glyphs) and converted to linear albedo.

To add one: a function `(rng) -> Distractor` decorated with
`@DISTRACTORS.register("name", weight=...)`.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .cards import rounded_rect_mask
from .color import jitter_hsv_srgb, srgb_to_linear
from .noise import fractal_noise
from .registry import Registry

DISTRACTORS: Registry = Registry("distractor")

Texture = tuple[np.ndarray, np.ndarray]  # (linear albedo HxWx3, alpha HxW)


@dataclass
class Distractor:
    name: str
    size_mm: tuple[float, float]  # (w, h) of the texture's frame
    render: Callable[[float, np.random.Generator], Texture]  # (px_per_mm, rng) -> texture
    gloss: float
    shininess: float
    lift_mm: float  # thickness, for the drop shadow
    info: dict = field(default_factory=dict)


# --- drawing helpers (sRGB uint8 canvas, mm coordinates) ---------------------------

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
SCRIPT_FONTS = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
INKS = [(0.08, 0.08, 0.09), (0.2, 0.2, 0.22), (0.1, 0.15, 0.45), (0.35, 0.35, 0.37)]


class Canvas:
    def __init__(self, w_mm: float, h_mm: float, ppm: float, rgb):
        self.s = ppm
        self.w_mm, self.h_mm = w_mm, h_mm
        self.W, self.H = max(4, int(round(w_mm * ppm))), max(4, int(round(h_mm * ppm)))
        self.img = np.empty((self.H, self.W, 3), np.uint8)
        self.img[:] = _u8(rgb)

    def px(self, x_mm, y_mm):
        return int(round(x_mm * self.s)), int(round(y_mm * self.s))

    def rect(self, x0, y0, x1, y1, rgb, thickness_mm: float | None = None):
        t = -1 if thickness_mm is None else max(1, int(round(thickness_mm * self.s)))
        cv2.rectangle(self.img, self.px(x0, y0), self.px(x1, y1), _u8(rgb), t, cv2.LINE_AA)

    def line(self, x0, y0, x1, y1, rgb, thickness_mm=0.3):
        cv2.line(self.img, self.px(x0, y0), self.px(x1, y1), _u8(rgb), max(1, int(round(thickness_mm * self.s))), cv2.LINE_AA)

    def circle(self, x, y, r_mm, rgb, thickness_mm: float | None = None):
        t = -1 if thickness_mm is None else max(1, int(round(thickness_mm * self.s)))
        cv2.circle(self.img, self.px(x, y), max(1, int(round(r_mm * self.s))), _u8(rgb), t, cv2.LINE_AA)

    def poly(self, pts_mm, rgb, closed=True, thickness_mm: float | None = None):
        pts = np.round(np.asarray(pts_mm) * self.s * 16).astype(np.int32).reshape(-1, 1, 2)
        if thickness_mm is None:
            cv2.fillPoly(self.img, [pts], _u8(rgb), cv2.LINE_AA, 4)
        else:
            cv2.polylines(self.img, [pts], closed, _u8(rgb), max(1, int(round(thickness_mm * self.s))), cv2.LINE_AA, 4)

    def text(self, s: str, x, y_baseline, cap_mm: float, rgb, font=cv2.FONT_HERSHEY_SIMPLEX, bold=1.0, max_w_mm=None):
        scale = cap_mm * self.s / 22.0  # Hershey cap height is ~22 px at scale 1
        th = max(1, int(round(scale * 1.6 * bold)))
        if max_w_mm is not None:  # shrink to fit
            (tw, _), _ = cv2.getTextSize(s, font, scale, th)
            if tw > max_w_mm * self.s:
                scale *= max_w_mm * self.s / tw
                th = max(1, int(round(scale * 1.6 * bold)))
        if scale * 22 < 1.2:  # sub-pixel glyphs: draw the line as a thin smear, as a camera would see it
            self.line(x, y_baseline - cap_mm / 2, x + len(s) * cap_mm * 0.6, y_baseline - cap_mm / 2, rgb, cap_mm * 0.35)
            return
        cv2.putText(self.img, s, self.px(x, y_baseline), font, scale, _u8(rgb), th, cv2.LINE_AA)

    def paragraph(self, rng, x0, y0, w_mm, h_mm, line_mm, rgb, font=None, cap_frac=0.62, ragged=True):
        """Lines of random words filling a box."""
        font = FONTS[rng.integers(2)] if font is None else font
        y = y0 + line_mm
        while y <= y0 + h_mm:
            n_chars = int(w_mm / (line_mm * cap_frac * 0.75))
            if ragged and rng.random() < 0.2:
                n_chars = int(n_chars * rng.uniform(0.3, 0.9))
            self.text(words(rng, n_chars), x0, y, line_mm * cap_frac, rgb, font, max_w_mm=w_mm)
            y += line_mm

    def finish(self, rng, alpha: np.ndarray | None = None, grain=0.02, drift=0.015) -> Texture:
        lin = srgb_to_linear(self.img.astype(np.float32) / 255)
        g = 1 + grain * (fractal_noise((self.H, self.W), 1.5, rng, octaves=2) - 0.5)
        d = 1 + drift * (fractal_noise((self.H, self.W), 25 * self.s, rng, octaves=2) - 0.5)
        lin = lin * (g * d)[..., None]
        if alpha is None:
            alpha = np.ones((self.H, self.W), np.float32)
        return np.clip(lin, 0, 1).astype(np.float32), alpha


def _u8(rgb):
    return tuple(int(round(255 * float(np.clip(c, 0, 1)))) for c in rgb)


def words(rng, n_chars: int, caps=0.15, digits=0.05) -> str:
    out, n = [], 0
    while n < n_chars:
        L = int(rng.integers(1, 10))
        w = "".join(rng.choice(list(string.ascii_lowercase), L))
        if rng.random() < caps:
            w = w.capitalize()
        if rng.random() < digits:
            w = str(rng.integers(1, 9999))
        out.append(w)
        n += len(w) + 1
    return " ".join(out)[:max(1, n_chars)]


def _color(rng, base, dh=0.02, ds=0.1, dv=0.06):
    return jitter_hsv_srgb(base, rng, dh=dh, ds=ds, dv=dv)


def _rand_color(rng, sat=(0.3, 0.9), val=(0.25, 0.95)):
    import colorsys
    return np.array(colorsys.hsv_to_rgb(rng.random(), rng.uniform(*sat), rng.uniform(*val)), np.float32)


def _ink(rng):
    return _color(rng, INKS[rng.integers(len(INKS))], dv=0.2)


def _image_block(c: Canvas, rng, x0, y0, x1, y1):
    """A 'photo' / illustration: smooth random colors plus a few shapes."""
    X0, Y0 = c.px(x0, y0)
    X1, Y1 = c.px(x1, y1)
    h, w = max(1, Y1 - Y0), max(1, X1 - X0)
    cols = np.stack([_rand_color(rng) for _ in range(3)])
    n = [fractal_noise((h, w), max(2, w / rng.uniform(1.5, 4)), rng, octaves=3) for _ in range(2)]
    img = cols[0] * (1 - n[0])[..., None] + cols[1] * n[0][..., None]
    img = img * (1 - 0.5 * n[1])[..., None] + cols[2] * (0.5 * n[1])[..., None]
    c.img[Y0:Y0 + h, X0:X0 + w] = (np.clip(img, 0, 1) * 255).astype(np.uint8)[: c.H - Y0, : c.W - X0]
    for _ in range(rng.integers(0, 4)):
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        r = rng.uniform(0.05, 0.25) * min(x1 - x0, y1 - y0)
        c.circle(cx, cy, r, _rand_color(rng))


def _barcode(c: Canvas, rng, x0, y0, w, h):
    x = x0
    while x < x0 + w:
        bw = rng.choice([0.25, 0.35, 0.5, 0.7])
        if rng.random() < 0.55:
            c.rect(x, y0, x + bw, y0 + h, (0.05, 0.05, 0.05))
        x += bw


def _edge_darken(tex: Texture, px: float, amount=0.12) -> Texture:
    alb, a = tex
    edge = 1 - cv2.erode(a, np.ones((3, 3), np.uint8), iterations=max(1, int(px)))
    return (alb * (1 - amount * np.clip(edge, 0, 1))[..., None]).astype(np.float32), a


def _rrect(c: Canvas, r_mm: float) -> np.ndarray:
    return rounded_rect_mask(c.W, c.H, r_mm * c.s)


# --- distractors -------------------------------------------------------------------

PAPER_WHITE = (0.95, 0.95, 0.93)


@DISTRACTORS.register("envelope", weight=2.0)
def envelope(rng):
    """Letter envelope, front (address, stamp, postmark, barcode, window) or back (flaps)."""
    w, h = [(220, 110), (241, 105), (165, 92), (162, 114), (184, 133), (152, 102)][rng.integers(6)]
    base = [PAPER_WHITE, (0.93, 0.91, 0.84), (0.72, 0.56, 0.40), (0.90, 0.80, 0.56),
            (0.80, 0.87, 0.95), (0.95, 0.82, 0.85), (0.75, 0.12, 0.15)][rng.choice(7, p=[0.4, 0.15, 0.15, 0.1, 0.07, 0.07, 0.06])]
    paper = _color(rng, base, dv=0.04)
    front = rng.random() < 0.65

    def render(s, rng):
        c = Canvas(w, h, s, paper)
        ink = _ink(rng)
        if front:
            if rng.random() < 0.8:  # recipient, centered-ish
                x, y = w * rng.uniform(0.3, 0.45), h * rng.uniform(0.45, 0.6)
                hand = rng.random() < 0.35
                font = SCRIPT_FONTS[rng.integers(2)] if hand else FONTS[rng.integers(4)]
                for i in range(int(rng.integers(3, 5))):
                    c.text(words(rng, int(rng.integers(10, 26)), caps=0.5, digits=0.2), x, y + i * 6.5, rng.uniform(3, 4.2),
                           ink, font, max_w_mm=w - x - 5)
            if rng.random() < 0.7:  # return address
                for i in range(3):
                    c.text(words(rng, int(rng.integers(12, 24)), caps=0.5, digits=0.2), 8, 10 + i * 4.2, 2.4, ink, max_w_mm=w * 0.45)
            if rng.random() < 0.6:  # stamp, top right
                sw, sh = rng.uniform(20, 26), rng.uniform(24, 30)
                x0, y0 = w - sw - rng.uniform(5, 9), rng.uniform(5, 9)
                c.rect(x0 - 1.2, y0 - 1.2, x0 + sw + 1.2, y0 + sh + 1.2, (0.97, 0.97, 0.95))
                _image_block(c, rng, x0, y0, x0 + sw, y0 + sh)
                for t in np.arange(x0 - 1.2, x0 + sw + 1.2, 2.0):  # perforations
                    c.circle(t, y0 - 1.2, 0.6, paper); c.circle(t, y0 + sh + 1.2, 0.6, paper)
                if rng.random() < 0.7:  # postmark over the stamp's edge
                    pc = (x0 - rng.uniform(0, 12), y0 + sh * 0.5)
                    c.circle(*pc, 11, (0.15, 0.15, 0.2), 0.5)
                    for k in range(5):
                        yy = pc[1] - 8 + k * 4
                        xs = np.linspace(pc[0] - 45, pc[0] - 12, 20)
                        c.poly(np.column_stack([xs, yy + 1.2 * np.sin(xs / 3)]), (0.15, 0.15, 0.2), closed=False, thickness_mm=0.5)
            elif rng.random() < 0.5:  # printed indicia
                c.rect(w - 32, 6, w - 7, 26, ink, 0.4)
                c.text("POSTAGE", w - 30, 14, 2.2, ink); c.text("PAID", w - 30, 20, 2.2, ink)
            if rng.random() < 0.3:  # address window
                c.rect(20, h * 0.45, 20 + 90, h * 0.45 + 32, np.asarray(paper) * 0.93)
            if rng.random() < 0.5:
                _barcode(c, rng, w * 0.25, h - 12, w * 0.5, 4.5)
            if rng.random() < 0.3:  # airmail / priority stripe
                col = _rand_color(rng, sat=(0.6, 0.9), val=(0.5, 0.9))
                c.rect(0, h - 5, w, h, col)
        else:  # back: flap edges meeting near the middle
            edge = np.asarray(paper) * 0.8
            tip = (w / 2, h * rng.uniform(0.5, 0.65))
            c.poly([(0, 0), tip, (w, 0)], edge, closed=False, thickness_mm=0.35)
            c.poly([(0, h), (w * 0.4, h * 0.45), (w * 0.6, h * 0.45), (w, h)], edge, closed=False, thickness_mm=0.35)
            if rng.random() < 0.3:
                c.circle(*tip, 7, _rand_color(rng, val=(0.4, 0.8)))  # seal / sticker
        return _edge_darken(c.finish(rng, rounded_rect_mask(c.W, c.H, 0.8 * s), grain=0.03), 0.4 * s, 0.08)

    return Distractor("envelope", (w, h), render, 0.03, 20, rng.uniform(0.5, 3), {"front": bool(front)})


@DISTRACTORS.register("book", weight=1.2)
def book(rng):
    """Book or magazine cover: color field, title, author, maybe artwork."""
    w = rng.uniform(125, 175)
    h = w * rng.uniform(1.3, 1.55)
    magazine = rng.random() < 0.25
    bg = _rand_color(rng, sat=(0.0, 0.8), val=(0.1, 0.95))
    light = float(np.mean(bg)) > 0.55
    fg = _color(rng, (0.08, 0.08, 0.08) if light else (0.96, 0.95, 0.9), dv=0.1)

    def render(s, rng):
        c = Canvas(w, h, s, bg)
        if magazine or rng.random() < 0.5:
            y0 = h * (0.25 if not magazine else 0.0)
            _image_block(c, rng, 0 if magazine else w * 0.1, y0, w if magazine else w * 0.9, h * (1.0 if magazine else 0.8))
        font = FONTS[rng.integers(4)]
        y = h * rng.uniform(0.12, 0.2)
        for _ in range(int(rng.integers(1, 3))):
            c.text(words(rng, int(rng.integers(5, 14)), caps=0.8).upper() if rng.random() < 0.5 else words(rng, 10, caps=0.8),
                   w * 0.08, y, rng.uniform(8, 14), fg, font, bold=rng.uniform(1, 2), max_w_mm=w * 0.84)
            y += rng.uniform(14, 20)
        c.text(words(rng, int(rng.integers(8, 18)), caps=1.0), w * 0.08, h * 0.93, rng.uniform(4, 6), fg, font, max_w_mm=w * 0.84)
        if not magazine:  # spine hinge
            c.rect(0, 0, rng.uniform(4, 8), h, np.asarray(bg) * 0.75)
        return _edge_darken(c.finish(rng, _rrect(c, rng.uniform(0.5, 2.5)), grain=0.015), 0.5 * s, 0.2)

    lift = rng.uniform(2, 6) if magazine else rng.uniform(15, 45)
    return Distractor("book", (w, h), render, rng.uniform(0.03, 0.25), rng.uniform(20, 200), lift, {"magazine": bool(magazine)})


@DISTRACTORS.register("ereader", weight=0.8)
def ereader(rng):
    """E-reader (e-ink page), tablet or phone lying screen-up."""
    kind = rng.choice(["ereader", "tablet", "phone"], p=[0.45, 0.35, 0.2])
    w, h = {"ereader": (rng.uniform(108, 128), rng.uniform(157, 175)), "tablet": (rng.uniform(160, 190), rng.uniform(240, 260)),
            "phone": (rng.uniform(70, 78), rng.uniform(145, 162))}[kind]
    body = _color(rng, [(0.08, 0.08, 0.09), (0.25, 0.25, 0.27), (0.85, 0.85, 0.86)][rng.choice(3, p=[0.6, 0.2, 0.2])])

    def render(s, rng):
        c = Canvas(w, h, s, body)
        if kind == "ereader":
            m = rng.uniform(4, 11)
            sx0, sy0, sx1, sy1 = m, rng.uniform(8, 14), w - m, h - rng.uniform(8, 22)
            page = _color(rng, (0.82, 0.82, 0.80), dv=0.05)
            c.rect(sx0, sy0, sx1, sy1, page)
            if rng.random() < 0.85:
                c.paragraph(rng, sx0 + 6, sy0 + 5, sx1 - sx0 - 12, sy1 - sy0 - 14, rng.uniform(4, 6), (0.18, 0.18, 0.18), font=cv2.FONT_HERSHEY_COMPLEX)
            else:  # cover / sleep screen
                _image_block(c, rng, sx0 + 4, sy0 + 4, sx1 - 4, sy1 - 4)
                c.img[:] = c.img.mean(2, keepdims=True).astype(np.uint8)
        else:
            m = rng.uniform(3, 9) if kind == "tablet" else rng.uniform(1.5, 4)
            sx0, sy0, sx1, sy1 = m, m, w - m, h - m
            mode = rng.choice(["off", "apps", "page"], p=[0.3, 0.35, 0.35])
            if mode == "off":
                c.rect(sx0, sy0, sx1, sy1, (0.03, 0.03, 0.035))
            elif mode == "apps":
                _image_block(c, rng, sx0, sy0, sx1, sy1)
                step = (sx1 - sx0) / 4.6
                for i in range(4):
                    for j in range(int((sy1 - sy0 - 20) / step)):
                        x, y = sx0 + step * (0.45 + i * 1.05), sy0 + 15 + step * j * 1.05
                        c.rect(x, y, x + step * 0.7, y + step * 0.7, _rand_color(rng, val=(0.5, 1)))
            else:
                c.rect(sx0, sy0, sx1, sy1, (0.97, 0.97, 0.97))
                c.rect(sx0, sy0, sx1, sy0 + 10, _rand_color(rng))
                _image_block(c, rng, sx0 + 4, sy0 + 14, sx1 - 4, sy0 + 14 + (sx1 - sx0) * 0.5)
                c.paragraph(rng, sx0 + 4, sy0 + 18 + (sx1 - sx0) * 0.5, sx1 - sx0 - 8, sy1 - sy0 - 30 - (sx1 - sx0) * 0.5,
                            rng.uniform(3, 4.5), (0.15, 0.15, 0.15))
        r = {"ereader": rng.uniform(4, 9), "tablet": rng.uniform(6, 12), "phone": rng.uniform(7, 11)}[kind]
        return c.finish(rng, _rrect(c, r), grain=0.005, drift=0.01)

    return Distractor("ereader", (w, h), render, rng.uniform(0.08, 0.3), rng.uniform(150, 600), rng.uniform(7, 10), {"kind": str(kind)})


# playing cards ------------------------------------------------------------------

def _pip(c: Canvas, suit: str, x, y, size, color, flip=False):
    sg = -1 if flip else 1
    if suit == "diamond":
        c.poly([(x, y - size * 0.62), (x + size * 0.42, y), (x, y + size * 0.62), (x - size * 0.42, y)], color)
        return
    r = size * 0.28
    if suit in ("heart", "spade"):
        d = sg if suit == "heart" else -sg
        c.circle(x - r * 0.95, y - d * r * 0.5, r, color)
        c.circle(x + r * 0.95, y - d * r * 0.5, r, color)
        c.poly([(x - size * 0.5, y - d * r * 0.25), (x + size * 0.5, y - d * r * 0.25), (x, y + d * size * 0.55)], color)
        if suit == "spade":
            c.poly([(x, y), (x - size * 0.22, y + sg * size * 0.55), (x + size * 0.22, y + sg * size * 0.55)], color)
    else:  # club
        c.circle(x, y - sg * r * 1.1, r, color)
        c.circle(x - r * 1.1, y + sg * r * 0.4, r, color)
        c.circle(x + r * 1.1, y + sg * r * 0.4, r, color)
        c.poly([(x, y), (x - size * 0.2, y + sg * size * 0.6), (x + size * 0.2, y + sg * size * 0.6)], color)


PIP_LAYOUT = {  # (x, y) in the pip area, 0..1
    1: [(0.5, 0.5)], 2: [(0.5, 0), (0.5, 1)], 3: [(0.5, 0), (0.5, 0.5), (0.5, 1)],
    4: [(0, 0), (1, 0), (0, 1), (1, 1)], 5: [(0, 0), (1, 0), (0.5, 0.5), (0, 1), (1, 1)],
    6: [(0, 0), (1, 0), (0, 0.5), (1, 0.5), (0, 1), (1, 1)],
    7: [(0, 0), (1, 0), (0.5, 0.25), (0, 0.5), (1, 0.5), (0, 1), (1, 1)],
    8: [(0, 0), (1, 0), (0.5, 0.25), (0, 0.5), (1, 0.5), (0.5, 0.75), (0, 1), (1, 1)],
    9: [(0, 0), (1, 0), (0, 1 / 3), (1, 1 / 3), (0.5, 0.5), (0, 2 / 3), (1, 2 / 3), (0, 1), (1, 1)],
    10: [(0, 0), (1, 0), (0.5, 1 / 6), (0, 1 / 3), (1, 1 / 3), (0, 2 / 3), (1, 2 / 3), (0.5, 5 / 6), (0, 1), (1, 1)],
}


@DISTRACTORS.register("playing_card", weight=1.5)
def playing_card(rng):
    """Poker/bridge card, face (pips, indices, court art) or back. Same size as a SET card."""
    w, h = (63.5, 88.9) if rng.random() < 0.7 else (57.0, 89.0)
    face = rng.random() < 0.7
    suit = ["heart", "diamond", "spade", "club"][rng.integers(4)]
    rank = int(rng.integers(1, 14))
    back_col = [(0.72, 0.1, 0.12), (0.12, 0.2, 0.55), (0.1, 0.4, 0.2), (0.1, 0.1, 0.1)][rng.choice(4, p=[0.45, 0.4, 0.08, 0.07])]

    def render(s, rng):
        c = Canvas(w, h, s, _color(rng, (0.96, 0.96, 0.94), dv=0.03))
        if face:
            col = (0.8, 0.08, 0.12) if suit in ("heart", "diamond") else (0.06, 0.06, 0.07)
            label = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(rank, str(rank))
            for flip in (False, True):
                x, y = (4.5, 10.5) if not flip else (w - 4.5, h - 10.5)
                if not flip:
                    c.text(label, 2.2, 10, 6, col, cv2.FONT_HERSHEY_DUPLEX, bold=1.3)
                _pip(c, suit, x if not flip else x, y + (5 if not flip else -5), 4.5, col, flip)
            if rank <= 10:
                x0, x1, y0, y1 = w * 0.3, w * 0.7, h * 0.2, h * 0.8
                size = 14 if rank > 1 else 22
                for u, v in PIP_LAYOUT[rank]:
                    _pip(c, suit, x0 + u * (x1 - x0), y0 + v * (y1 - y0), size * rng.uniform(0.9, 1.05), col, v > 0.5)
            else:  # court card: framed picture
                c.rect(w * 0.18, h * 0.14, w * 0.82, h * 0.86, (0.9, 0.75, 0.2))
                _image_block(c, rng, w * 0.19, h * 0.15, w * 0.81, h * 0.85)
                c.rect(w * 0.18, h * 0.14, w * 0.82, h * 0.86, (0.1, 0.1, 0.4), 0.6)
        else:
            bc = _color(rng, back_col)
            m = rng.uniform(2.5, 5)
            c.rect(m, m, w - m, h - m, bc)
            step = rng.uniform(2.5, 5)
            for t in np.arange(-h, w + h, step):
                c.line(t, m, t + (h - 2 * m), h - m, (0.96, 0.96, 0.94), 0.3)
                c.line(t, h - m, t + (h - 2 * m), m, (0.96, 0.96, 0.94), 0.3)
            # white margin over the lattice's overshoot
            c.rect(0, 0, m, h, (0.96, 0.96, 0.94)); c.rect(w - m, 0, w, h, (0.96, 0.96, 0.94))
            c.rect(0, 0, w, m, (0.96, 0.96, 0.94)); c.rect(0, h - m, w, h, (0.96, 0.96, 0.94))
        return _edge_darken(c.finish(rng, _rrect(c, rng.uniform(2.5, 4))), 0.3 * s, 0.15)

    return Distractor("playing_card", (w, h), render, rng.uniform(0.04, 0.12), rng.uniform(20, 90), rng.uniform(0.3, 1.2),
                      {"face": bool(face), "suit": suit if face else None, "rank": rank if face else None})


@DISTRACTORS.register("game_card", weight=1.0)
def game_card(rng):
    """Other card games at SET-card size: UNO-style (color field, white oval, big
    number) or trading-card style (border, title bar, art, text box)."""
    w, h = (57, 87) if rng.random() < 0.5 else (63, 88)
    style = rng.choice(["uno", "tcg"])

    def render(s, rng):
        if style == "uno":
            c = Canvas(w, h, s, (0.97, 0.97, 0.96))
            col = [(0.85, 0.1, 0.1), (0.98, 0.8, 0.05), (0.1, 0.6, 0.25), (0.05, 0.35, 0.75), (0.07, 0.07, 0.07)][rng.integers(5)]
            m = 4
            inner = np.zeros((c.H, c.W), np.float32)
            inner[int(m * s):c.H - int(m * s), int(m * s):c.W - int(m * s)] = rounded_rect_mask(c.W - 2 * int(m * s), c.H - 2 * int(m * s), 3 * s)
            c.img[inner > 0.5] = _u8(col)
            cv2.ellipse(c.img, c.px(w / 2, h / 2), (int(w * 0.4 * s), int(h * 0.26 * s)), -35, 0, 360, _u8((0.97, 0.97, 0.96)), -1, cv2.LINE_AA)
            label = str(rng.integers(0, 10)) if rng.random() < 0.8 else ["+2", "S", "R"][rng.integers(3)]
            c.text(label, w * 0.3, h * 0.64, 22, col, cv2.FONT_HERSHEY_TRIPLEX, bold=2, max_w_mm=w * 0.45)
            c.text(label, m + 2, m + 8, 5, (0.97, 0.97, 0.96), cv2.FONT_HERSHEY_TRIPLEX, bold=1.5)
        else:
            border = _color(rng, [(0.08, 0.08, 0.08), (0.95, 0.8, 0.2), (0.75, 0.75, 0.78), (0.9, 0.9, 0.88)][rng.integers(4)])
            c = Canvas(w, h, s, border)
            m = 3
            frame = _rand_color(rng, sat=(0.2, 0.7), val=(0.5, 0.9))
            c.rect(m, m, w - m, h - m, frame)
            c.rect(m + 1.5, m + 1.5, w - m - 1.5, m + 8, np.clip(frame + 0.2, 0, 1))
            c.text(words(rng, 14, caps=1), m + 3, m + 6.5, 3, (0.05, 0.05, 0.05), max_w_mm=w - 2 * m - 6)
            _image_block(c, rng, m + 2, m + 10, w - m - 2, h * 0.55)
            c.rect(m + 2, h * 0.58, w - m - 2, h - m - 2, (0.93, 0.91, 0.85))
            c.paragraph(rng, m + 4, h * 0.58, w - 2 * m - 8, h * 0.36, 3.2, (0.1, 0.1, 0.1))
        return _edge_darken(c.finish(rng, _rrect(c, rng.uniform(2.5, 4))), 0.3 * s, 0.15)

    return Distractor("game_card", (w, h), render, rng.uniform(0.04, 0.15), rng.uniform(20, 120), rng.uniform(0.3, 1.2), {"style": str(style)})


@DISTRACTORS.register("index_card", weight=1.0)
def index_card(rng):
    """Ruled index card / note card, blank or handwritten."""
    w, h = [(127, 76), (152, 102), (76, 127), (102, 152), (105, 74)][rng.integers(5)]
    ruled = rng.random() < 0.7
    written = rng.random() < 0.6

    def render(s, rng):
        c = Canvas(w, h, s, _color(rng, (0.96, 0.96, 0.94), dv=0.03))
        if ruled:
            top = rng.uniform(12, 18)
            c.line(0, top, w, top, (0.85, 0.35, 0.4), 0.35)
            for y in np.arange(top + 6.35, h - 2, 6.35):
                c.line(0, y, w, y, (0.55, 0.7, 0.9), 0.2)
        if written:
            ink = [(0.1, 0.15, 0.5), (0.08, 0.08, 0.08), (0.6, 0.1, 0.1)][rng.integers(3)]
            font = SCRIPT_FONTS[rng.integers(2)] if rng.random() < 0.7 else FONTS[0]
            y = rng.uniform(10, 16)
            for _ in range(int(rng.integers(1, 7))):
                c.text(words(rng, int(rng.integers(6, 30))), rng.uniform(4, 10), y, rng.uniform(3, 5), ink, font, max_w_mm=w - 14)
                y += 6.35
                if y > h - 3:
                    break
        return _edge_darken(c.finish(rng, _rrect(c, rng.uniform(0.3, 1.5))), 0.3 * s, 0.08)

    return Distractor("index_card", (w, h), render, 0.02, 20, rng.uniform(0.2, 0.8), {"ruled": bool(ruled)})


@DISTRACTORS.register("letter", weight=0.7)
def letter(rng):
    """Printed letter / flyer, flat or folded in thirds."""
    W, H = (216, 279) if rng.random() < 0.5 else (210, 297)
    folded = rng.random() < 0.5
    w, h = (W, H / 3) if folded else (W, H)

    def render(s, rng):
        c = Canvas(w, h, s, _color(rng, PAPER_WHITE, dv=0.03))
        ink = _ink(rng)
        y = 20 if not folded else rng.uniform(5, 20)
        if rng.random() < 0.5:
            c.circle(28, y + 4, 8, _rand_color(rng))
            c.text(words(rng, 14, caps=1), 40, y + 7, 5, ink, cv2.FONT_HERSHEY_DUPLEX, max_w_mm=w - 60)
            y += 22
        c.paragraph(rng, 25, y, w - 50, h - y - 15, rng.uniform(4.5, 6.5), ink)
        if not folded:  # crease lines of a letter that was folded for mailing
            for f in (H / 3, 2 * H / 3):
                if rng.random() < 0.6:
                    c.line(0, f, w, f, np.asarray(c.img[1, 1]) / 255 * 0.9, 0.4)
        return _edge_darken(c.finish(rng, grain=0.02), 0.3 * s, 0.05)

    return Distractor("letter", (w, h), render, 0.02, 15, rng.uniform(0.2, 3), {"folded": bool(folded)})


@DISTRACTORS.register("receipt", weight=0.6)
def receipt(rng):
    """Thermal-paper receipt: narrow, long, monospace lines, a total, a barcode."""
    w, h = rng.uniform(57, 80), rng.uniform(90, 260)

    def render(s, rng):
        c = Canvas(w, h, s, _color(rng, (0.96, 0.96, 0.95), dv=0.02))
        ink = (0.2, 0.2, 0.22)
        c.text(words(rng, 12, caps=1).upper(), w * 0.2, 12, 4, ink, cv2.FONT_HERSHEY_DUPLEX, max_w_mm=w * 0.6)
        y = 20
        while y < h - 25:
            if rng.random() < 0.12:
                for x in np.arange(4, w - 4, 2.5):
                    c.line(x, y - 1, x + 1.2, y - 1, ink, 0.25)
            else:
                c.text(words(rng, int(rng.integers(8, 18))).upper(), 4, y, 2.2, ink, max_w_mm=w * 0.6)
                c.text(f"{rng.uniform(0.5, 60):.2f}", w - 16, y, 2.2, ink)
            y += 4.2
        if rng.random() < 0.5:
            _barcode(c, rng, w * 0.15, h - 18, w * 0.7, 10)
        return c.finish(rng, grain=0.015)

    return Distractor("receipt", (w, h), render, 0.05, 30, rng.uniform(0.1, 2), {})


@DISTRACTORS.register("business_card", weight=0.8)
def business_card(rng):
    w, h = (89, 51) if rng.random() < 0.7 else (85, 55)
    dark = rng.random() < 0.25
    bg = _rand_color(rng, sat=(0.2, 0.8), val=(0.1, 0.35)) if dark else _color(rng, (0.95, 0.95, 0.93), dv=0.03)
    fg = (0.95, 0.95, 0.93) if dark else _ink(rng)
    accent = _rand_color(rng, sat=(0.5, 0.9), val=(0.4, 0.9))

    def render(s, rng):
        c = Canvas(w, h, s, bg)
        if rng.random() < 0.6:  # logo
            x, y, r = rng.uniform(8, 20), rng.uniform(8, 18), rng.uniform(4, 8)
            [lambda: c.circle(x, y, r, accent), lambda: c.rect(x - r, y - r, x + r, y + r, accent),
             lambda: c.poly([(x, y - r), (x + r, y + r), (x - r, y + r)], accent)][rng.integers(3)]()
        c.text(words(rng, int(rng.integers(8, 18)), caps=1), w * 0.35, h * 0.45, rng.uniform(3.5, 5), fg, FONTS[rng.integers(4)], max_w_mm=w * 0.6)
        c.text(words(rng, 14, caps=1), w * 0.35, h * 0.45 + 6, 2.2, fg, max_w_mm=w * 0.6)
        for i in range(int(rng.integers(2, 4))):
            c.text(words(rng, 22, digits=0.4), w * 0.35, h * 0.72 + i * 4, 1.8, fg, max_w_mm=w * 0.6)
        if rng.random() < 0.3:
            c.rect(0, h - 3, w, h, accent)
        return _edge_darken(c.finish(rng, _rrect(c, rng.choice([0.3, 0.3, 3.0]))), 0.3 * s, 0.1)

    return Distractor("business_card", (w, h), render, rng.uniform(0.02, 0.2), rng.uniform(20, 150), rng.uniform(0.3, 0.6), {"dark": bool(dark)})


@DISTRACTORS.register("photo_print", weight=0.5)
def photo_print(rng):
    """Glossy photo print, bordered or borderless (a polaroid now and then)."""
    kind = rng.choice(["4x6", "polaroid"], p=[0.75, 0.25])
    w, h = (152, 102) if kind == "4x6" else (88, 107)
    if kind == "4x6" and rng.random() < 0.5:
        w, h = h, w
    border = rng.uniform(0, 5) if kind == "4x6" and rng.random() < 0.4 else 0.0

    def render(s, rng):
        c = Canvas(w, h, s, (0.95, 0.95, 0.94))
        if kind == "polaroid":
            _image_block(c, rng, 6, 6, w - 6, w - 6)
        else:
            _image_block(c, rng, border, border, w - border, h - border)
        return c.finish(rng, grain=0.01)

    return Distractor("photo_print", (w, h), render, rng.uniform(0.2, 0.45), rng.uniform(200, 800), 0.3, {"kind": str(kind)})


@DISTRACTORS.register("coaster", weight=0.3)
def coaster(rng):
    """Square or round coaster: cork or printed."""
    d = rng.uniform(90, 108)
    round_ = rng.random() < 0.5
    cork = rng.random() < 0.6

    def render(s, rng):
        c = Canvas(d, d, s, (0.66, 0.5, 0.33) if cork else _rand_color(rng))
        if cork:
            n = fractal_noise((c.H, c.W), 0.8 * s, rng, octaves=3)
            c.img[:] = np.clip(c.img * (0.75 + 0.5 * n)[..., None], 0, 255).astype(np.uint8)
        else:
            _image_block(c, rng, 5, 5, d - 5, d - 5)
        a = np.zeros((c.H, c.W), np.float32)
        if round_:
            cv2.circle(a, (c.W // 2, c.H // 2), c.W // 2 - 1, 1.0, -1, cv2.LINE_AA)
        else:
            a = _rrect(c, rng.uniform(1, 8))
        return c.finish(rng, a)

    return Distractor("coaster", (d, d), render, 0.02, 10, rng.uniform(2, 5), {"round": bool(round_), "cork": bool(cork)})
