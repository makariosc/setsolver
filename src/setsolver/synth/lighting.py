"""Scene lighting on the table plane.

A lighting rig is a set of lights (point or directional) plus ambient light,
sampled relative to the layout. Shading is evaluated per image pixel from its
table-plane position, so cards and table always share the same illumination.

Large soft shadows (the photographer, a cup, a phone) are plane-space masks
that block the key light (lights[0]). Card drop/contact shadows are handled in
scene.py because they depend on the card layout.

To add a rig: write a function `(rng, ctx) -> LightingSetup` decorated with
`@RIGS.register("name", weight=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .color import kelvin_to_linear_rgb
from .registry import Registry


@dataclass
class Light:
    kind: str  # "point" | "directional"
    color: np.ndarray  # linear RGB, max channel 1
    intensity: float  # irradiance at the layout center (relative units)
    position: np.ndarray | None = None  # point: world mm
    direction: np.ndarray | None = None  # directional: unit vector *toward* the light
    falloff_mm: float = 0.0  # directional only: exponential gradient length (window light)

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "color": self.color.tolist(), "intensity": self.intensity}
        if self.position is not None:
            d["position"] = self.position.tolist()
        if self.direction is not None:
            d["direction"] = self.direction.tolist()
        if self.falloff_mm:
            d["falloff_mm"] = self.falloff_mm
        return d


@dataclass
class LightingSetup:
    name: str
    lights: list[Light]
    ambient: np.ndarray  # linear RGB irradiance
    info: dict = field(default_factory=dict)

    @property
    def key(self) -> Light:
        return self.lights[0]

    def average_color(self) -> np.ndarray:
        """What an ideal auto white balance would neutralize."""
        total = self.ambient.copy()
        for l in self.lights:
            total += l.color * l.intensity
        return total / total.max()


@dataclass
class LightingContext:
    center: np.ndarray  # layout center, world mm (z = 0)
    extent_mm: float
    camera_pos: np.ndarray  # world mm


RIGS: Registry = Registry("lighting rig")


def _point(rng, ctx: LightingContext, height, spread, kelvin, intensity, tint=0.0) -> Light:
    ang = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(0, spread)
    pos = ctx.center + np.array([r * np.cos(ang), r * np.sin(ang), height])
    col = kelvin_to_linear_rgb(kelvin)
    col = col * np.array([1.0, 1.0 + tint, 1.0])
    return Light("point", (col / col.max()).astype(np.float32), intensity, position=pos)


def _directional(rng, elevation_deg, kelvin, intensity, falloff_mm=0.0) -> Light:
    az = rng.uniform(0, 2 * np.pi)
    el = np.radians(elevation_deg)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return Light("directional", kelvin_to_linear_rgb(kelvin), intensity, direction=d, falloff_mm=falloff_mm)


# Rig weights are tuned so roughly 3/4 of scenes are lit by white/neutral
# light (3800 K+) and 1/4 by warm light.


@RIGS.register("warm_room", weight=1.0)
def warm_room(rng, ctx):
    """Warm ceiling light (incandescent / warm LED) plus bounce light."""
    k = rng.uniform(2700, 3500)
    key = _point(rng, ctx, rng.uniform(1500, 2600), rng.uniform(0, 900), k, 1.0)
    amb = kelvin_to_linear_rgb(k) * rng.uniform(0.2, 0.5)
    return LightingSetup("warm_room", [key], amb, {"kelvin": k})


@RIGS.register("neutral_room", weight=2.0)
def neutral_room(rng, ctx):
    """Neutral/cool-white LED ceiling light plus bounce light."""
    k = rng.uniform(4000, 6000)
    key = _point(rng, ctx, rng.uniform(1500, 2800), rng.uniform(0, 1000), k, 1.0)
    amb = kelvin_to_linear_rgb(k) * rng.uniform(0.25, 0.6)
    return LightingSetup("neutral_room", [key], amb, {"kelvin": k})


@RIGS.register("window_daylight", weight=1.5)
def window_daylight(rng, ctx):
    """Daylight from a window: directional, cool, with a gradient across the table."""
    k = rng.uniform(5000, 7500)
    key = _directional(rng, rng.uniform(20, 60), k, 1.0, falloff_mm=rng.uniform(400, 1500))
    amb = kelvin_to_linear_rgb(rng.uniform(6500, 9000)) * rng.uniform(0.3, 0.7)
    return LightingSetup("window_daylight", [key], amb, {"kelvin": k})


@RIGS.register("overcast_diffuse", weight=1.2)
def overcast_diffuse(rng, ctx):
    """Soft, flat white light (overcast sky / big diffuse room light): mostly ambient."""
    k = rng.uniform(5200, 7000)
    key = _directional(rng, rng.uniform(40, 80), k, rng.uniform(0.2, 0.5), falloff_mm=rng.uniform(1500, 4000))
    amb = kelvin_to_linear_rgb(k) * rng.uniform(0.8, 1.3)
    return LightingSetup("overcast_diffuse", [key], amb, {"kelvin": k})


@RIGS.register("fluorescent_office", weight=1.0)
def fluorescent_office(rng, ctx):
    """Flat, bright, slightly green-tinted overhead light."""
    k = rng.uniform(3800, 5200)
    key = _point(rng, ctx, rng.uniform(1800, 2800), rng.uniform(0, 1200), k, 1.0, tint=rng.uniform(0.03, 0.12))
    amb = key.color * rng.uniform(0.5, 0.9)
    return LightingSetup("fluorescent_office", [key], amb, {"kelvin": k})


@RIGS.register("phone_flash", weight=0.5)
def phone_flash(rng, ctx):
    """Phone flash in a dim room: light at the camera, hot spot + glare in frame."""
    k = rng.uniform(5000, 6000)
    pos = ctx.camera_pos + np.append(rng.normal(0, 10, 2), 0.0)
    key = Light("point", kelvin_to_linear_rgb(k), 1.0, position=pos)
    amb = kelvin_to_linear_rgb(rng.uniform(2700, 4500)) * rng.uniform(0.02, 0.12)
    return LightingSetup("phone_flash", [key], amb, {"kelvin": k})


@RIGS.register("dim_lamp", weight=0.6)
def dim_lamp(rng, ctx):
    """A single warm lamp close by: strong falloff, dark surroundings."""
    k = rng.uniform(2200, 3000)
    key = _point(rng, ctx, rng.uniform(400, 1000), rng.uniform(300, 900), k, 1.0)
    amb = kelvin_to_linear_rgb(k) * rng.uniform(0.03, 0.15)
    return LightingSetup("dim_lamp", [key], amb, {"kelvin": k, "dim": True})


@RIGS.register("mixed_warm_cool", weight=0.8)
def mixed_warm_cool(rng, ctx):
    """Warm lamp + cool window: two differently colored lights."""
    warm = _point(rng, ctx, rng.uniform(800, 2000), rng.uniform(200, 1000), rng.uniform(2400, 3200), 1.0)
    cool = _directional(rng, rng.uniform(15, 50), rng.uniform(5500, 8000), rng.uniform(0.3, 1.0), rng.uniform(500, 1500))
    lights = [warm, cool] if rng.random() < 0.5 else [cool, warm]
    amb = kelvin_to_linear_rgb(5000) * rng.uniform(0.1, 0.3)
    return LightingSetup("mixed_warm_cool", lights, amb)


# --- shading -------------------------------------------------------------------


def light_vectors(light: Light, X: np.ndarray, Y: np.ndarray):
    """Unit direction toward the light (Lx, Ly, Lz) and unnormalized irradiance."""
    if light.kind == "point":
        dx, dy, dz = light.position[0] - X, light.position[1] - Y, light.position[2]
        d2 = dx * dx + dy * dy + dz * dz
        d = np.sqrt(d2)
        Lx, Ly, Lz = dx / d, dy / d, dz / d
        E = Lz / d2
    else:
        Lx, Ly, Lz = (np.full_like(X, v) for v in light.direction)
        E = np.full_like(X, light.direction[2])
        if light.falloff_mm:
            # brighter toward the window side
            s = X * light.direction[0] + Y * light.direction[1]
            E = E * np.exp(s / light.falloff_mm)
    return Lx, Ly, Lz, E.astype(np.float32)


def irradiance_at(light: Light, pt: np.ndarray) -> float:
    return float(light_vectors(light, np.array([pt[0]], np.float32), np.array([pt[1]], np.float32))[3][0])


def shade(
    albedo: np.ndarray,
    gloss: np.ndarray,
    shininess: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    setup: LightingSetup,
    camera_pos: np.ndarray,
    center: np.ndarray,
    key_visibility: np.ndarray,
    ambient_occlusion: np.ndarray,
    slope_x: np.ndarray | None = None,
    slope_y: np.ndarray | None = None,
) -> np.ndarray:
    """Linear radiance = diffuse (Lambert) + specular (normalized Blinn-Phong).

    slope_x/slope_y: optional fine surface slopes (plane x/y) that tilt the
    normal (-sx, -sy, 1) in the specular term only, e.g. a card's linen
    texture breaking a highlight into specks."""
    Vx, Vy, Vz = camera_pos[0] - X, camera_pos[1] - Y, np.float32(camera_pos[2])
    vn = np.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
    Vx, Vy, Vz = Vx / vn, Vy / vn, Vz / vn

    diffuse = setup.ambient[None, None, :] * ambient_occlusion[..., None]
    specular = np.zeros_like(albedo)
    for i, light in enumerate(setup.lights):
        Lx, Ly, Lz, E = light_vectors(light, X, Y)
        E = E * (light.intensity / max(irradiance_at(light, center), 1e-12))
        if i == 0:
            E = E * key_visibility
        diffuse = diffuse + (E[..., None] * light.color[None, None, :])
        Hx, Hy, Hz = Lx + Vx, Ly + Vy, Lz + Vz
        if slope_x is None:
            NdotH = Hz / np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
        else:
            NdotH = (Hz - Hx * slope_x - Hy * slope_y) / (
                np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz) * np.sqrt(1 + slope_x * slope_x + slope_y * slope_y))
        spec = gloss * (shininess + 8) / (8 * np.pi) * np.power(np.clip(NdotH, 0, 1), shininess) * E
        specular += spec[..., None] * light.color[None, None, :]
    return (albedo * diffuse + specular).astype(np.float32)


# --- large soft shadows (plane space) ------------------------------------------


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ellipse_occlusion(X, Y, cx, cy, a, b, angle, softness):
    """1 inside the ellipse, 0 outside, with a soft edge `softness` mm wide."""
    c, s = np.cos(angle), np.sin(angle)
    u = (X - cx) * c + (Y - cy) * s
    v = -(X - cx) * s + (Y - cy) * c
    r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    sdf = (r - 1) * min(a, b)
    return 1 - _smoothstep(-softness / 2, softness / 2, sdf)


def photographer_shadow(X, Y, rng, near_edge_pts: np.ndarray, toward: np.ndarray):
    """Head-and-shoulders shadow entering from the photographer's side of the frame.

    `near_edge_pts`: plane points along the image's bottom edge.
    `toward`: unit plane vector pointing from the photographer into the scene.
    """
    p = near_edge_pts[rng.integers(len(near_edge_pts))]
    body_w = rng.uniform(180, 320)
    softness = rng.uniform(30, 120)
    perp = np.arctan2(toward[1], toward[0]) + np.pi / 2
    body_c = p - toward * rng.uniform(0, 120)
    occ = ellipse_occlusion(X, Y, body_c[0], body_c[1], body_w, body_w * 0.5, perp, softness)
    head_c = body_c + toward * body_w * rng.uniform(0.55, 0.8)
    head_r = body_w * rng.uniform(0.3, 0.4)
    occ = np.maximum(occ, ellipse_occlusion(X, Y, head_c[0], head_c[1], head_r, head_r * 1.15, perp, softness))
    return occ


def rounded_rect_occlusion(X, Y, cx, cy, a, b, radius, angle, softness):
    """1 inside a rounded rectangle (half-sizes a, b), soft edge `softness` mm wide."""
    c, s = np.cos(angle), np.sin(angle)
    u = np.abs((X - cx) * c + (Y - cy) * s) - (a - radius)
    v = np.abs(-(X - cx) * s + (Y - cy) * c) - (b - radius)
    outside = np.sqrt(np.maximum(u, 0) ** 2 + np.maximum(v, 0) ** 2)
    sdf = outside + np.minimum(np.maximum(u, v), 0) - radius
    return 1 - _smoothstep(-softness / 2, softness / 2, sdf)


def blob_occlusion(X, Y, cx, cy, radius, rng, softness):
    """Irregular rounded blob: a circle whose radius wobbles with a few harmonics."""
    theta = np.arctan2(Y - cy, X - cx)
    r = np.ones_like(theta)
    for k in (2, 3, 4):
        r += rng.uniform(0, 0.25 / k * 2) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    sdf = np.hypot(X - cx, Y - cy) - radius * r
    return 1 - _smoothstep(-softness / 2, softness / 2, sdf)


def slats_occlusion(X, Y, cx, cy, angle, period, duty, extent, softness):
    """Parallel bars (window blinds, chair back, railing), fading out beyond `extent`."""
    c, s = np.cos(angle), np.sin(angle)
    u = (X - cx) * c + (Y - cy) * s
    v = -(X - cx) * s + (Y - cy) * c
    phase = np.abs(((u / period) % 1.0) - 0.5) * 2  # 0 at bar center, 1 at gap center
    bars = 1 - _smoothstep(duty - softness / period, duty + softness / period, phase)
    window = 1 - _smoothstep(extent * 0.8, extent * 1.2, np.maximum(np.abs(u), np.abs(v)))
    return bars * window


def object_shadow(X, Y, rng, center, extent):
    """A shadow from an off-frame object. Returns (occlusion, kind)."""
    c = center + rng.uniform(-0.6, 0.6, 2) * extent
    angle = rng.uniform(0, np.pi)
    softness = rng.uniform(5, 60)
    kind = str(rng.choice(["round", "rect", "long", "blob", "slats"], p=[0.22, 0.22, 0.2, 0.22, 0.14]))
    if kind == "round":  # cup, glass
        r = rng.uniform(35, 70)
        occ = ellipse_occlusion(X, Y, c[0], c[1], r * rng.uniform(1.0, 1.8), r, angle, softness)
    elif kind == "rect":  # phone, book, box
        a, b = rng.uniform(35, 120), rng.uniform(30, 80)
        occ = rounded_rect_occlusion(X, Y, c[0], c[1], a, b, rng.uniform(2, 15), angle, softness)
    elif kind == "long":  # bottle, arm, lamp stem
        a, b = rng.uniform(120, 350), rng.uniform(20, 45)
        occ = rounded_rect_occlusion(X, Y, c[0], c[1], a, b, b * 0.9, angle, softness)
    elif kind == "blob":  # plant, bag, hand
        occ = blob_occlusion(X, Y, c[0], c[1], rng.uniform(50, 150), rng, softness)
    else:
        period = rng.uniform(25, 90)
        occ = slats_occlusion(X, Y, c[0], c[1], angle, period, rng.uniform(0.3, 0.6), rng.uniform(0.4, 1.0) * extent, softness * 0.3)
    return occ, kind
