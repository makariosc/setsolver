"""Compose one synthetic photo: deck style -> layout -> camera -> table surface
-> distractors -> cards -> occluders -> shadows -> lighting -> camera pipeline,
plus exact labels.

Everything on the table (surface, cards, shadows) lives in the plane z = 0 and
is seen through ONE camera, so perspective, lighting, and shadows are shared
and consistent across the whole scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .backgrounds import BACKGROUNDS, PlaneRegion
from .camera import sample_camera
from .cards import ALL_CARDS, CARD_H_MM, CARD_W_MM, DeckStyle, render_card
from .distractors import DISTRACTORS, Distractor
from .layout import LAYOUTS, Placement
from .color import kelvin_to_linear_rgb
from .lighting import RIGS, Light, LightingContext, LightingSetup, object_shadow, photographer_shadow, shade
from .occluders import OCCLUDERS, Occluder, OccluderContext
from .sensor import develop
from .surface import CardFinish, render_finish

MAX_BG_PIXELS = 9e6  # cap on the table-surface canvas


@dataclass
class SceneConfig:
    width: int = 1200  # portrait size; swapped for landscape shots
    height: int = 1600
    supersample: int = 2
    landscape_prob: float = 0.35
    clip_prob: float = 0.05  # scenes where cards are cut off by the frame edge
    loose_prob: float = 0.12  # loosely framed scenes: cards small with lots of margin
    occluder_prob: float = 0.3  # scenes with objects lying on/near the cards (fingers, coins, paper...)
    distractor_prob: float = 0.35  # scenes with non-card rectangles in view (envelopes, books, playing cards...)
    roomy_prob: float = 0.5  # of tight shots with distractors: frame looser so they fit in view
    glossy_prob: float = 0.3  # scenes with glossy cards (glare that washes out ink, linen speckle)
    glare_light_prob: float = 0.75  # of glossy scenes: a lamp at the mirror angle so the glare lands on cards
    backgrounds: list[str] | None = None  # restrict to these names
    rigs: list[str] | None = None
    layouts: list[str] | None = None
    distractors: list[str] | None = None


def _tex_to_plane(center_xy: np.ndarray, angle: float, w_mm: float, h_mm: float, ppm: float) -> np.ndarray:
    """3x3 affine: texture pixel of a w x h mm rectangle -> table plane mm.

    The texture is y-down (image convention) while the table's y axis points
    away from a viewer looking down (z up), so the texture's y is flipped here;
    without the flip every card would be drawn mirrored (visible on squiggles)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, center_xy[0]], [s, c, center_xy[1]], [0, 0, 1]])
    center = np.array([[1, 0, -w_mm / 2], [0, -1, h_mm / 2], [0, 0, 1]])
    tex = np.array([[1 / ppm, 0, 0.5 / ppm], [0, 1 / ppm, 0.5 / ppm], [0, 0, 1]])
    return rot @ center @ tex


def _card_to_plane(p: Placement, ppm: float) -> np.ndarray:
    return _tex_to_plane(p.center, p.angle, CARD_W_MM, CARD_H_MM, ppm)


def _rect_corners(center_xy: np.ndarray, angle: float, w_mm: float, h_mm: float) -> np.ndarray:
    """Corners of a w x h rectangle on the table, same order/convention as Placement.corners()."""
    hw, hh = w_mm / 2, h_mm / 2
    local = np.array([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]])
    c, s = np.cos(angle), np.sin(angle)
    return local @ np.array([[c, s], [-s, c]]) + center_xy


def _magnification(corners_ss: np.ndarray, w_mm: float, h_mm: float) -> float:
    """Largest on-screen (supersampled) px per mm along the rectangle's edges."""
    edge_px = np.linalg.norm(np.diff(np.vstack([corners_ss, corners_ss[:1]]), axis=0), axis=1)
    return float(max(edge_px[0] / w_mm, edge_px[1] / h_mm, edge_px[2] / w_mm, edge_px[3] / h_mm))


def _translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _drop_offset(setup: LightingSetup, center_xy: np.ndarray, lift_mm: float) -> np.ndarray:
    """Plane offset of a card's shadow given its height above the table."""
    key = setup.key
    if key.kind == "point":
        rel = center_xy - key.position[:2]
        return lift_mm * rel / key.position[2]
    d = key.direction
    return -lift_mm * d[:2] / max(d[2], 0.1)


@dataclass
class _Canvases:
    """Per-pixel scene buffers that flat objects are composited into."""
    albedo: np.ndarray
    gloss: np.ndarray
    shininess: np.ndarray
    key_vis: np.ndarray
    ao: np.ndarray
    owner: np.ndarray
    slope_x: np.ndarray  # fine surface slopes (plane x/y), specular only
    slope_y: np.ndarray


def _draw_flat(tex_rgb, tex_a, A, corners_ss, mag, lift, H_ss, setup, center_xy, rng, buf: _Canvases, owner_id: int,
               obj_gloss: float, obj_shininess: float, min_blur_mm: float = 0.0, surface=None) -> bool:
    """Composite a flat textured object lying on the table (card or distractor):
    warp its texture through the camera, add a drop shadow (blocks the key
    light) and a contact/ambient-occlusion ring. `A` maps texture px -> plane mm.
    `surface` = (gloss map, slope_x, slope_y, angle) in texture space, for
    per-texel gloss and fine bumps (cards); otherwise gloss is uniform.
    Returns False if it is entirely out of frame."""
    H_img, W_img = buf.owner.shape
    tex = np.dstack([tex_rgb * tex_a[..., None], tex_a])  # premultiplied RGBA
    shadow_off = _drop_offset(setup, center_xy, lift)
    blur_mm = max(rng.uniform(0.3, 1.2), min_blur_mm)
    margin = int(np.ceil(mag * (4 + np.linalg.norm(shadow_off) + 3 * min_blur_mm))) + 2
    bx0 = max(0, int(np.floor(corners_ss[:, 0].min())) - margin)
    by0 = max(0, int(np.floor(corners_ss[:, 1].min())) - margin)
    bx1 = min(W_img, int(np.ceil(corners_ss[:, 0].max())) + margin)
    by1 = min(H_img, int(np.ceil(corners_ss[:, 1].max())) + margin)
    if bx1 <= bx0 or by1 <= by0:
        return False
    bw, bh = bx1 - bx0, by1 - by0
    M = _translate(-bx0, -by0) @ H_ss @ A
    warped = cv2.warpPerspective(tex, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    prem, a = warped[..., :3], warped[..., 3]

    Ms = _translate(-bx0, -by0) @ H_ss @ _translate(*shadow_off) @ A
    sh = cv2.warpPerspective(tex_a, Ms, (bw, bh), flags=cv2.INTER_LINEAR)
    sh = cv2.GaussianBlur(sh, (0, 0), max(0.5, mag * blur_mm))
    occ = cv2.GaussianBlur(a, (0, 0), max(0.5, mag * rng.uniform(1.0, 3.0)))
    sl = (slice(by0, by1), slice(bx0, bx1))
    buf.key_vis[sl] *= 1 - rng.uniform(0.6, 0.95) * sh
    buf.ao[sl] *= 1 - rng.uniform(0.2, 0.45) * occ
    # The object's own top surface is unshadowed by itself.
    buf.key_vis[sl] = buf.key_vis[sl] * (1 - a) + a
    buf.ao[sl] = buf.ao[sl] * (1 - a) + a

    buf.albedo[sl] = prem + buf.albedo[sl] * (1 - a[..., None])
    buf.shininess[sl] = obj_shininess * a + buf.shininess[sl] * (1 - a)
    if surface is None:
        buf.gloss[sl] = obj_gloss * a + buf.gloss[sl] * (1 - a)
        buf.slope_x[sl] *= 1 - a
        buf.slope_y[sl] *= 1 - a
    else:
        g_tex, sx_tex, sy_tex, angle = surface
        # texture +x -> plane (cos, sin); texture +y (down) -> plane (sin, -cos): see _tex_to_plane
        c, s_ = np.cos(angle), np.sin(angle)
        maps = np.dstack([g_tex * tex_a, (sx_tex * c + sy_tex * s_) * tex_a, (sx_tex * s_ - sy_tex * c) * tex_a])
        wm = cv2.warpPerspective(maps, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        buf.gloss[sl] = wm[..., 0] + buf.gloss[sl] * (1 - a)
        buf.slope_x[sl] = wm[..., 1] + buf.slope_x[sl] * (1 - a)
        buf.slope_y[sl] = wm[..., 2] + buf.slope_y[sl] * (1 - a)
    buf.owner[sl][a > 0.5] = owner_id
    return bool((a > 0.5).any())


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    c = poly.astype(np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for x, y in pts])


def _place_distractor(rng, d: Distractor, view: np.ndarray, placements: list[Placement]):
    """Pick a pose for a distractor so that it is (at least partly) in view.
    Returns (center, angle, layer) or None. Modes: free table space in view,
    straddling the frame edge, or under the cards. Thick things (books,
    tablets) never lie under cards; anything overlapping cards on top of them
    must leave most of the cards visible."""
    w, h = d.size_mm
    card_polys = [p.corners() for p in placements]
    u = (np.stack(np.meshgrid(np.linspace(-0.45, 0.45, 7), np.linspace(-0.45, 0.45, 7)), -1).reshape(-1, 2))
    vc = view.mean(0)
    thick = d.lift_mm > 5
    for _ in range(40):
        if rng.random() < 0.5:  # square to the tableau, as people put things down
            angle = placements[rng.integers(len(placements))].angle + rng.integers(4) * np.pi / 2 + np.radians(rng.normal(0, 4))
        else:
            angle = rng.uniform(0, 2 * np.pi)
        mode = rng.choice(["free", "edge", "under"], p=[0.5, 0.35, 0.15])
        if mode == "free":
            lo, hi = view.min(0), view.max(0)
            center = rng.uniform(lo, hi)
        elif mode == "edge":
            i = rng.integers(4)
            e = view[i] + (view[(i + 1) % 4] - view[i]) * rng.uniform(0.1, 0.9)
            out = (e - vc) / (np.linalg.norm(e - vc) + 1e-9)
            center = e + out * rng.uniform(-0.35, 0.2) * min(w, h)
        else:
            if thick:
                continue
            center = placements[rng.integers(len(placements))].center + rng.normal(0, 0.25, 2) * [w, h]
        c, s = np.cos(angle), np.sin(angle)
        pts = (u * [w, h]) @ np.array([[c, s], [-s, c]]) + center
        in_view = _inside(view, pts).mean()
        on_cards = np.any([_inside(q, pts) for q in card_polys], axis=0).mean()
        if in_view < 0.45:  # a sliver at the frame edge teaches little
            continue
        if on_cards < 0.02:
            return center, angle, "under" if rng.random() < 0.7 else "top"
        if mode == "under" and on_cards < 0.8:
            return center, angle, "under"
        if on_cards < 0.2 and rng.random() < 0.5:  # e.g. an envelope dropped over a card's edge
            return center, angle, "top"
    return None


def _corner_visibility(k: int, p: Placement, corner_radius_mm: float, cam_ss, owner: np.ndarray) -> list[str]:
    """Per corner: "visible", "occluded" (covered by another card or object) or
    "outside" (beyond the frame). Cards have rounded corners, so we probe a
    point just inside the rounded tip along the diagonal toward the center."""
    corners = p.corners()
    inset = 0.42 * corner_radius_mm + 1.0  # mm: rounded tip sits ~0.41 r from the virtual corner
    out = []
    for c in corners:
        d = (p.center - c) / np.linalg.norm(p.center - c)
        u, v = cam_ss.project_plane((c + d * inset)[None])[0]
        H, W = owner.shape
        if not (0 <= u < W and 0 <= v < H):
            out.append("outside")
        else:
            out.append("visible" if owner[int(v), int(u)] == k else "occluded")
    return out


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _draw_occluder(o: Occluder, rng, X, Y, cam_ss, setup, albedo, gloss, shininess, key_vis, ao, owner) -> bool:
    """Composite an occluder (in place) on top of everything drawn so far,
    with a drop shadow and contact occlusion. Returns False if off-frame."""
    center = (o.bounds[0] + o.bounds[1]) / 2
    shadow_off = _drop_offset(setup, center, o.lift_mm)
    penumbra = 0.5 + 0.25 * o.lift_mm  # mm
    pad = np.linalg.norm(shadow_off) + penumbra + 4
    lo, hi = o.bounds[0] - pad, o.bounds[1] + pad
    box = cam_ss.project_plane(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]))
    H, W = owner.shape
    x0, y0 = max(0, int(box[:, 0].min())), max(0, int(box[:, 1].min()))
    x1, y1 = min(W, int(np.ceil(box[:, 0].max())) + 1), min(H, int(np.ceil(box[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    sl = (slice(y0, y1), slice(x0, x1))
    Xs, Ys = X[sl], Y[sl]
    # pixel size in mm at this spot, for anti-aliasing the edge
    q = cam_ss.project_plane(np.array([center, center + [1.0, 0.0]]))
    px_mm = 1.0 / max(np.linalg.norm(q[1] - q[0]), 1e-6)
    d = o.sdf(Xs, Ys)
    a = np.clip(0.5 - d / px_mm, 0, 1).astype(np.float32)
    if not (a > 0.5).any():
        return False
    sh = 1 - _smoothstep(-penumbra / 2, penumbra / 2, o.sdf(Xs - shadow_off[0], Ys - shadow_off[1]))
    key_vis[sl] *= 1 - rng.uniform(0.6, 0.9) * sh * (1 - a)
    ao[sl] *= 1 - 0.3 * (1 - _smoothstep(0, 2.5, d)) * (1 - a)
    key_vis[sl] = key_vis[sl] * (1 - a) + a
    ao[sl] = ao[sl] * (1 - a) + a
    albedo[sl] = o.color(Xs, Ys) * a[..., None] + albedo[sl] * (1 - a[..., None])
    gloss[sl] = o.gloss * a + gloss[sl] * (1 - a)
    shininess[sl] = o.shininess * a + shininess[sl] * (1 - a)
    owner[sl][a > 0.5] = -2
    return True


def generate_scene(rng: np.random.Generator, cfg: SceneConfig = SceneConfig()) -> tuple[np.ndarray, dict]:
    ss = cfg.supersample
    w, h = (cfg.height, cfg.width) if rng.random() < cfg.landscape_prob else (cfg.width, cfg.height)

    # --- content
    style = DeckStyle.sample(rng)
    layout = LAYOUTS.sample(rng, cfg.layouts)
    placements: list[Placement] = layout.obj(rng)
    cards = [ALL_CARDS[i] for i in rng.choice(len(ALL_CARDS), len(placements), replace=False)]
    all_corners = np.concatenate([p.corners() for p in placements])
    lo, hi = all_corners.min(0), all_corners.max(0)
    center = (lo + hi) / 2
    extent = float(max(hi - lo))

    # --- camera
    # People usually shoot a tableau square-on; scattered cards have no preferred axis.
    azimuth = None
    if layout.name != "scatter" and rng.random() < 0.8:
        azimuth = rng.integers(4) * np.pi / 2 + np.radians(rng.normal(0, 5))
    # Framing. Most photos fill the frame with the cards, nicely centered and
    # fully visible; some are loosely framed with lots of margin (edge case);
    # a few crop cards at the frame edge.
    corners_all = np.stack([p.corners() for p in placements]).reshape(-1, 2)
    r = rng.random()
    if r < cfg.clip_prob:
        framing = "clipped"
        occupancy = rng.uniform(1.05, 1.4)
        offset = rng.normal(0, 0.08, 2)
    elif r < cfg.clip_prob + cfg.loose_prob:
        framing = "loose"
        occupancy = rng.uniform(0.4, 0.75)
        offset = rng.uniform(-1, 1, 2) * (1 - occupancy) / 2 * 0.8
    else:
        framing = "tight"
        occupancy = rng.uniform(0.82, 0.95)
        offset = np.clip(rng.normal(0, 0.015, 2), -0.03, 0.03)
    n_distractors = 0
    if rng.random() < cfg.distractor_prob:
        n_distractors = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.3, 0.15, 0.1]))
        if framing == "tight" and rng.random() < cfg.roomy_prob:
            framing = "roomy"  # people rarely crop out the envelope next to the cards
            occupancy *= rng.uniform(0.6, 0.85)
    if len(placements) <= 2 and framing != "clipped":
        occupancy = min(occupancy, rng.uniform(0.45, 0.85))  # close-ups of 1-2 cards don't fill the frame
    def frame(w, h, occupancy, offset):
        for _ in range(20):
            cam = sample_camera(rng, w, h, corners_all, occupancy, tuple(offset), azimuth)
            px = cam.project_plane(corners_all)
            margin = 0.01 * min(w, h)
            inside = (px[:, 0] >= margin) & (px[:, 0] <= w - margin) & (px[:, 1] >= margin) & (px[:, 1] <= h - margin)
            if framing == "clipped" or inside.all():
                return cam, occupancy
            occupancy *= 0.97  # tight shot touched the edge: back off slightly
            offset = offset * 0.5
        return cam, occupancy

    cam, used_occupancy = frame(w, h, occupancy, offset)
    # People usually turn the phone to match the tableau's shape.
    px = cam.project_plane(corners_all)
    aspect = np.ptp(px[:, 0]) / max(np.ptp(px[:, 1]), 1e-6)
    if ((aspect > 1.15 and w < h) or (aspect < 0.87 and w > h)) and rng.random() < 0.8:
        w, h = h, w
        cam, used_occupancy = frame(w, h, occupancy, offset)
    occupancy = used_occupancy
    cam_ss = cam.scaled(ss)
    H_ss = cam_ss.H
    X, Y = cam_ss.plane_grid()
    H_img, W_img = X.shape

    # --- lighting
    ctx = LightingContext(np.array([center[0], center[1], 0.0]), extent, cam.C)
    setup: LightingSetup = RIGS.sample(rng, cfg.rigs).obj(rng, ctx)

    # --- table surface, rendered in plane space then sampled per pixel
    bg_entry = BACKGROUNDS.sample(rng, cfg.backgrounds)
    bg = bg_entry.obj()
    x0, y0 = float(X.min()) - 2, float(Y.min()) - 2
    rw, rh = float(X.max()) - x0 + 4, float(Y.max()) - y0 + 4
    ppm_bg = min(cam_ss.px_per_mm_max(), np.sqrt(MAX_BG_PIXELS / (rw * rh)))
    region = PlaneRegion(x0, y0, rw, rh, ppm_bg)
    maps = bg.render(region, rng)
    mapx = ((X - x0) * ppm_bg - 0.5).astype(np.float32)
    mapy = ((Y - y0) * ppm_bg - 0.5).astype(np.float32)
    albedo = cv2.remap(maps.albedo, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gloss = cv2.remap(maps.gloss, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    shininess = np.full((H_img, W_img), maps.shininess, np.float32)

    key_vis = np.ones((H_img, W_img), np.float32)
    ao = np.ones((H_img, W_img), np.float32)
    owner = np.full((H_img, W_img), -1, np.int16)

    buf = _Canvases(albedo, gloss, shininess, key_vis, ao, owner,
                    np.zeros((H_img, W_img), np.float32), np.zeros((H_img, W_img), np.float32))
    finish = CardFinish.sample(rng, glossy=rng.random() < cfg.glossy_prob)
    card_gloss, card_shininess = finish.gloss, finish.shininess
    glare_light = None
    if finish.glossy and rng.random() < cfg.glare_light_prob:
        # A ceiling lamp whose mirror reflection (seen from the camera) lands on a
        # card: the broad glare that bleaches ink in real photos.
        target = placements[rng.integers(len(placements))].center + rng.normal(0, 20, 2)
        V = cam.C - np.array([target[0], target[1], 0.0])
        R = np.array([-V[0], -V[1], V[2]]) / np.linalg.norm(V)
        pos = np.array([target[0], target[1], 0.0]) + R * np.linalg.norm(V) * rng.uniform(0.8, 2.5)
        glare_light = Light("point", kelvin_to_linear_rgb(rng.uniform(3200, 6500)), float(rng.uniform(0.3, 1.2)), position=pos)
        setup.lights.append(glare_light)
    mm_to_px = []  # local magnification per card (SS px per mm)

    # --- distractors: non-card rectangles in view (hard negatives, unlabeled)
    view = cam.unproject_to_plane(np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    placed = []
    for _ in range(n_distractors):
        entry = DISTRACTORS.sample(rng, cfg.distractors)
        d: Distractor = entry.obj(rng)
        pose = _place_distractor(rng, d, view, placements)
        if pose is not None:
            placed.append((d, *pose))

    def draw_distractor(d: Distractor, center_xy, angle):
        dw, dh = d.size_mm
        corners_ss = cam_ss.project_plane(_rect_corners(center_xy, angle, dw, dh))
        mag = _magnification(corners_ss, dw, dh)
        ppm = float(min(np.clip(mag, 2, 16), np.sqrt(8e6 / (dw * dh))))
        tex_rgb, tex_a = d.render(ppm, rng)
        A = _tex_to_plane(center_xy, angle, dw, dh, ppm)
        return _draw_flat(tex_rgb, tex_a, A, corners_ss, mag, d.lift_mm, H_ss, setup, center_xy, rng, buf, -3,
                          d.gloss, d.shininess, min_blur_mm=0.25 * d.lift_mm)

    def distractor_label(d: Distractor, center_xy, angle, layer):
        corners = cam.project_plane(_rect_corners(center_xy, angle, *d.size_mm))
        return {"type": d.name, "layer": layer, "size_mm": [round(v, 1) for v in d.size_mm],
                "corners": corners.round(2).tolist(), **d.info}  # image px, for evaluating false positives

    distractors = []
    for d, center_xy, angle, layer in placed:
        if layer == "under" and draw_distractor(d, center_xy, angle):
            distractors.append(distractor_label(d, center_xy, angle, layer))

    # --- cards, bottom to top
    for k, (card, p) in enumerate(zip(cards, placements)):
        corners_ss = cam_ss.project_plane(p.corners())
        mag = _magnification(corners_ss, CARD_W_MM, CARD_H_MM)
        mm_to_px.append(mag)
        ppm = float(np.clip(mag, 3, 32))
        tex_rgb, tex_a, ink = render_card(card, style, ppm, rng, with_ink=True)
        g_tex, sx_tex, sy_tex = render_finish(finish, ink, ppm, rng)
        lift = rng.uniform(0.3, 1.5)  # mm; card thickness + curl
        _draw_flat(tex_rgb, tex_a, _card_to_plane(p, ppm), corners_ss, mag, lift, H_ss, setup, p.center, rng, buf, k,
                   card_gloss, card_shininess, surface=(g_tex, sx_tex, sy_tex, p.angle))

    for d, center_xy, angle, layer in placed:
        if layer == "top" and draw_distractor(d, center_xy, angle):
            distractors.append(distractor_label(d, center_xy, angle, layer))

    # Where the photographer is: plane points just past the image's bottom edge,
    # and "into the scene" = image-up direction on the table (stable even when
    # the camera is straight overhead).
    bottom = np.column_stack([np.linspace(0.15, 0.85, 8) * W_img, np.full(8, H_img * 1.02)])
    near_pts = cam_ss.unproject_to_plane(bottom)
    mid = cam_ss.unproject_to_plane(np.array([[W_img / 2, H_img * 0.75], [W_img / 2, H_img * 0.25]]))
    toward = (mid[1] - mid[0]) / (np.linalg.norm(mid[1] - mid[0]) + 1e-9)

    # --- objects lying on top of the cards
    occluders = []
    if rng.random() < cfg.occluder_prob:
        octx = OccluderContext(placements, toward, near_pts)
        for _ in range(int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))):
            entry = OCCLUDERS.sample(rng)
            o: Occluder = entry.obj(rng, octx)
            if _draw_occluder(o, rng, X, Y, cam_ss, setup, albedo, gloss, shininess, key_vis, ao, owner):
                occluders.append({"type": o.name, **o.info})

    buf.slope_x[owner == -2] = 0  # occluders on top of cards are smooth
    buf.slope_y[owner == -2] = 0

    # --- large soft shadows from things off-frame
    shadows = []
    if rng.random() < 0.35:
        occ = photographer_shadow(X, Y, rng, near_pts, toward)
        strength = rng.uniform(0.7, 1.0)
        key_vis *= 1 - strength * occ
        shadows.append({"type": "photographer", "strength": strength})
    for _ in range(int(rng.random() < 0.25) + int(rng.random() < 0.08)):
        occ, kind = object_shadow(X, Y, rng, center, extent)
        strength = rng.uniform(0.4, 0.95)
        key_vis *= 1 - strength * occ
        shadows.append({"type": "object", "shape": kind, "strength": strength})

    # --- light it
    radiance = shade(albedo, gloss, shininess, X, Y, setup, cam.C, np.array([center[0], center[1]]), key_vis, ao,
                     buf.slope_x, buf.slope_y)

    # --- camera pipeline
    depth = (cam.R[2, 0] * X + cam.R[2, 1] * Y + cam.t[2]).astype(np.float32)
    image, sensor_info = develop(radiance, depth, setup.average_color(), rng, ss, (w, h))

    # --- labels (output-resolution pixels)
    card_labels = []
    for k, (card, p) in enumerate(zip(cards, placements)):
        corners = cam.project_plane(p.corners())
        area_ss = _polygon_area(cam_ss.project_plane(p.corners()))
        visible = float((owner == k).sum()) / max(area_ss, 1.0)
        if visible <= 0.0:
            continue
        card_labels.append(
            {
                **card.to_dict(),
                "index": card.index,
                "corners": corners.round(2).tolist(),  # TL, TR, BR, BL of the card's portrait frame
                "corner_visibility": _corner_visibility(k, p, style.corner_radius_mm, cam_ss, owner),
                "visible_fraction": round(min(visible, 1.0), 4),
                "clipped": bool(((corners < 0) | (corners > [w, h])).any()),
                "px_per_mm": round(mm_to_px[k] / ss, 3),
            }
        )

    labels = {
        "image_size": [w, h],
        "cards": card_labels,
        "framing": {"mode": framing, "occupancy": round(float(occupancy), 3)},
        "layout": layout.name,
        "background": {"name": bg_entry.name, **maps.info},
        "lighting": {"rig": setup.name, "lights": [l.to_dict() for l in setup.lights], "ambient": setup.ambient.tolist(),
                     "glare_light": glare_light is not None, **setup.info},
        "card_finish": finish.to_dict(),
        "shadows": shadows,
        "occluders": occluders,
        "distractors": distractors,
        "camera": {
            "K": cam.K.tolist(),
            "R": cam.R.tolist(),
            "C_mm": cam.C.tolist(),
            "tilt_deg": cam.tilt_deg,
            "fov_diag_deg": cam.fov_diag_deg,
        },
        "deck_style": {k: v for k, v in vars(style).items() if k != "extra"},
        "sensor": sensor_info,
    }
    return image, labels
