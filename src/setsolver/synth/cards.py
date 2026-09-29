"""Procedural SET card faces.

Cards are drawn in a portrait frame (x right, y down, millimeters): symbols have
their long axis horizontal and are stacked vertically. Stripes run perpendicular
to the symbol's long axis, as on real decks.

`DeckStyle` holds everything shared by one physical deck (ink colors, stripe
pitch, stroke weight, symbol size). Sample it once per scene; per-card variation
is limited to small print/wear jitter.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .color import jitter_hsv_srgb, srgb_to_linear
from .noise import fractal_noise

NUMBERS = (1, 2, 3)
COLORS = ("red", "green", "purple")
SHAPES = ("diamond", "oval", "squiggle")
SHADINGS = ("solid", "striped", "open")

CARD_W_MM = 57.0
CARD_H_MM = 89.0


@dataclass(frozen=True)
class Card:
    number: int
    color: str
    shape: str
    shading: str

    @property
    def index(self) -> int:
        """0..80, stable ordering over (number, color, shape, shading)."""
        return (
            NUMBERS.index(self.number) * 27
            + COLORS.index(self.color) * 9
            + SHAPES.index(self.shape) * 3
            + SHADINGS.index(self.shading)
        )

    def to_dict(self) -> dict:
        return asdict(self)


ALL_CARDS: tuple[Card, ...] = tuple(Card(*c) for c in itertools.product(NUMBERS, COLORS, SHAPES, SHADINGS))


# --- deck style -------------------------------------------------------------

# Ink palettes in sRGB 0-1. "classic" is the common red/green/purple print;
# "modern" is the orange/teal/purple print seen in some newer decks.
PALETTES = {
    "classic": {"red": (0.86, 0.10, 0.16), "green": (0.00, 0.60, 0.30), "purple": (0.36, 0.14, 0.55)},
    "modern": {"red": (0.98, 0.36, 0.20), "green": (0.00, 0.60, 0.52), "purple": (0.42, 0.22, 0.68)},
}
PALETTE_WEIGHTS = {"classic": 0.75, "modern": 0.25}


@dataclass
class DeckStyle:
    inks_srgb: dict[str, list[float]]
    palette: str
    paper_srgb: list[float]
    symbol_w_mm: float
    symbol_h_mm: float
    symbol_pitch_mm: float
    stroke_mm: float
    stripe_pitch_mm: float
    stripe_width_mm: float
    corner_radius_mm: float
    squiggle_variant: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "DeckStyle":
        names = list(PALETTES)
        palette = names[rng.choice(len(names), p=[PALETTE_WEIGHTS[n] for n in names])]
        inks = {
            c: jitter_hsv_srgb(PALETTES[palette][c], rng, dh=0.025, ds=0.12, dv=0.12).tolist() for c in COLORS
        }
        paper_v = rng.uniform(0.90, 0.97)
        paper = jitter_hsv_srgb((paper_v, paper_v, paper_v * rng.uniform(0.96, 1.0)), rng, dv=0.02).tolist()
        # Ranges measured from rectified photos of official cards: symbols ~40-43 mm
        # wide, outline ~1.0-1.2 mm, stripes ~1.1 mm apart and ~0.3 mm thick
        # (about 30% ink coverage), three symbols spanning ~70 mm.
        sym_w = rng.uniform(38, 43)
        return cls(
            inks_srgb=inks,
            palette=palette,
            paper_srgb=paper,
            symbol_w_mm=sym_w,
            symbol_h_mm=sym_w * rng.uniform(0.43, 0.52),
            symbol_pitch_mm=sym_w * rng.uniform(0.6, 0.66),
            stroke_mm=rng.uniform(0.85, 1.2),
            stripe_pitch_mm=rng.uniform(1.0, 1.2),
            stripe_width_mm=rng.uniform(0.25, 0.35),
            corner_radius_mm=rng.uniform(3.0, 4.5),
            squiggle_variant=int(rng.integers(0, 3)),
        )


# --- symbol outlines (unit coords: x in [-1, 1], y in [-0.5, 0.5], y down) ---


# Squiggle outline traced from a photo of a real card (perspective-rectified,
# averaged with its 180-degree rotation, Fourier-smoothed). Unit box, y down;
# the left lobe sits low and the right lobe high, as printed.
_SQUIGGLE_TRACED = np.array([
    (-1.0000, 0.1500),
    (-0.9954, 0.2482),
    (-0.9735, 0.3430),
    (-0.9321, 0.4222),
    (-0.8730, 0.4738),
    (-0.8020, 0.4909),
    (-0.7265, 0.4753),
    (-0.6519, 0.4370),
    (-0.5797, 0.3898),
    (-0.5077, 0.3468),
    (-0.4324, 0.3162),
    (-0.3522, 0.3013),
    (-0.2686, 0.3014),
    (-0.1853, 0.3144),
    (-0.1057, 0.3379),
    (-0.0309, 0.3690),
    (0.0412, 0.4039),
    (0.1142, 0.4380),
    (0.1911, 0.4666),
    (0.2723, 0.4867),
    (0.3562, 0.4976),
    (0.4401, 0.5000),
    (0.5221, 0.4940),
    (0.6016, 0.4781),
    (0.6779, 0.4495),
    (0.7499, 0.4054),
    (0.8151, 0.3461),
    (0.8709, 0.2747),
    (0.9157, 0.1963),
    (0.9497, 0.1149),
    (0.9747, 0.0314),
    (0.9919, -0.0563),
    (1.0000, -0.1500),
    (0.9954, -0.2482),
    (0.9735, -0.3430),
    (0.9321, -0.4222),
    (0.8730, -0.4738),
    (0.8020, -0.4909),
    (0.7265, -0.4753),
    (0.6519, -0.4370),
    (0.5797, -0.3898),
    (0.5077, -0.3468),
    (0.4324, -0.3162),
    (0.3522, -0.3013),
    (0.2686, -0.3014),
    (0.1853, -0.3144),
    (0.1057, -0.3379),
    (0.0309, -0.3690),
    (-0.0412, -0.4039),
    (-0.1142, -0.4380),
    (-0.1911, -0.4666),
    (-0.2723, -0.4867),
    (-0.3562, -0.4976),
    (-0.4401, -0.5000),
    (-0.5221, -0.4940),
    (-0.6016, -0.4781),
    (-0.6779, -0.4495),
    (-0.7499, -0.4054),
    (-0.8151, -0.3461),
    (-0.8709, -0.2747),
    (-0.9157, -0.1963),
    (-0.9497, -0.1149),
    (-0.9747, -0.0314),
    (-0.9919, 0.0563)
])


def _squiggle(variant: int, n: int = 256) -> np.ndarray:
    """Smoothly upsample the traced outline (periodic FFT interpolation).

    Variants low-pass it slightly differently, mimicking small differences
    between print runs."""
    keep = (12, 9, 7)[variant % 3]
    z = _SQUIGGLE_TRACED[:, 0] + 1j * _SQUIGGLE_TRACED[:, 1]
    F = np.fft.fft(z) / len(z)
    G = np.zeros(n, complex)
    G[: keep + 1] = F[: keep + 1]
    G[-keep:] = F[-keep:]
    out = np.fft.ifft(G) * n
    return np.stack([out.real, out.imag], 1)


def symbol_outline(shape: str, variant: int = 0) -> np.ndarray:
    if shape == "diamond":
        return np.array([(-1.0, 0.0), (0.0, -0.5), (1.0, 0.0), (0.0, 0.5)])
    if shape == "oval":  # stadium
        r = 0.5
        a = np.linspace(-np.pi / 2, np.pi / 2, 40)
        right = np.stack([1 - r + r * np.cos(a), r * np.sin(a)], 1)
        left = np.stack([-(1 - r) - r * np.cos(a), -r * np.sin(a)], 1)
        return np.concatenate([right, left])
    if shape == "squiggle":
        out = _squiggle(variant)
        # normalize to the unit box
        out -= (out.max(0) + out.min(0)) / 2
        out /= (out.max(0) - out.min(0)) / np.array([2.0, 1.0])
        return out
    raise ValueError(shape)


# --- rendering ---------------------------------------------------------------

_SHIFT = 4  # subpixel bits for cv2 drawing
_S = 1 << _SHIFT


def _fix(pts_px: np.ndarray) -> np.ndarray:
    return np.round(pts_px * _S).astype(np.int32).reshape(-1, 1, 2)


def rounded_rect_mask(w_px: int, h_px: int, r_px: float) -> np.ndarray:
    mask = np.zeros((h_px, w_px), np.uint8)
    r = max(1.0, r_px)
    a = np.linspace(0, np.pi / 2, 16)
    corners = [
        (w_px - r, h_px - r, 0),
        (r, h_px - r, np.pi / 2),
        (r, r, np.pi),
        (w_px - r, r, 3 * np.pi / 2),
    ]
    pts = np.concatenate([np.stack([cx + r * np.cos(a + o), cy + r * np.sin(a + o)], 1) for cx, cy, o in corners])
    cv2.fillPoly(mask, [_fix(pts - 0.5)], 255, cv2.LINE_AA, _SHIFT)
    return mask.astype(np.float32) / 255.0


def render_card(
    card: Card, style: DeckStyle, px_per_mm: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Render a card face. Returns (linear RGB albedo HxWx3, alpha HxW), portrait."""
    s = px_per_mm
    w_px, h_px = int(round(CARD_W_MM * s)), int(round(CARD_H_MM * s))

    # Ink coverage mask: 1 where ink is printed.
    ink = np.zeros((h_px, w_px), np.uint8)
    unit = symbol_outline(card.shape, style.squiggle_variant)
    cx = CARD_W_MM / 2 + rng.normal(0, 0.25)
    cy0 = CARD_H_MM / 2 + rng.normal(0, 0.25)
    offsets = (np.arange(card.number) - (card.number - 1) / 2) * style.symbol_pitch_mm
    stroke_px = max(1, int(round(style.stroke_mm * s)))
    misreg = rng.normal(0, 0.12, 2)  # stripe print misregistration, mm

    for dy in offsets:
        pts_mm = unit * np.array([style.symbol_w_mm / 2, style.symbol_h_mm]) + np.array([cx, cy0 + dy])
        pts = _fix(pts_mm * s)
        if card.shading == "solid":
            cv2.fillPoly(ink, [pts], 255, cv2.LINE_AA, _SHIFT)
        else:
            if card.shading == "striped":
                shape_mask = np.zeros_like(ink)
                cv2.fillPoly(shape_mask, [pts], 255, cv2.LINE_AA, _SHIFT)
                stripes = np.zeros_like(ink)
                x0 = cx - style.symbol_w_mm / 2 - 2 + misreg[0]
                sw = max(1, int(round(style.stripe_width_mm * s)))
                for x in np.arange(x0, cx + style.symbol_w_mm / 2 + 2, style.stripe_pitch_mm):
                    p1 = np.array([[x * s, (cy0 + dy - style.symbol_h_mm) * s]])
                    p2 = np.array([[x * s, (cy0 + dy + style.symbol_h_mm) * s]])
                    cv2.line(stripes, tuple(_fix(p1)[0, 0]), tuple(_fix(p2)[0, 0]), 255, sw, cv2.LINE_AA, _SHIFT)
                ink = np.maximum(ink, ((stripes.astype(np.uint16) * shape_mask) // 255).astype(np.uint8))
            cv2.polylines(ink, [pts], True, 255, stroke_px, cv2.LINE_AA, _SHIFT)

    cov = ink.astype(np.float32) / 255.0
    # Print imperfection: slightly uneven ink density.
    density = 0.98 + 0.02 * fractal_noise(cov.shape, 10 * s, rng, octaves=2)
    cov *= density

    paper = srgb_to_linear(style.paper_srgb)
    ink_lin = srgb_to_linear(style.inks_srgb[card.color]) * rng.uniform(0.95, 1.05)
    # Fine paper grain + low-frequency tint drift.
    grain = 1.0 + 0.02 * (fractal_noise(cov.shape, 1.5, rng, octaves=2) - 0.5)
    drift = 1.0 + 0.012 * (fractal_noise(cov.shape, 30 * s, rng, octaves=2) - 0.5)
    base = paper[None, None, :] * (grain * drift)[..., None]
    albedo = base * (1 - cov[..., None]) + ink_lin[None, None, :] * cov[..., None]

    # Occasional wear: faint smudges.
    if rng.random() < 0.3:
        smudge = fractal_noise(cov.shape, 15 * s, rng, octaves=3)
        albedo *= 1 - 0.04 * np.clip(smudge - 0.6, 0, 1)[..., None] / 0.4

    alpha = rounded_rect_mask(w_px, h_px, style.corner_radius_mm * s)
    # Slightly darker cut edge.
    edge = 1 - cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=max(1, int(0.3 * s)))
    albedo *= (1 - 0.15 * np.clip(edge, 0, 1))[..., None]
    return albedo.astype(np.float32), alpha
