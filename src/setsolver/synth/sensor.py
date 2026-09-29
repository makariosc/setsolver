"""Phone camera pipeline: linear scene radiance -> 8-bit JPEG-like sRGB image.

Order follows a real camera: optics (defocus / motion blur, vignetting) ->
exposure -> sensor noise -> white balance -> tone curve -> denoise/sharpen ->
JPEG. Runs at supersampled resolution until the downsample step.
"""

from __future__ import annotations

import cv2
import numpy as np

from .color import linear_to_srgb


def _luma(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def auto_exposure(radiance: np.ndarray, rng: np.random.Generator) -> float:
    lum = _luma(radiance[::4, ::4])
    geo_mean = float(np.exp(np.mean(np.log(lum + 1e-4))))
    p995 = float(np.percentile(lum, 99.5))
    e_mid = 0.18 / geo_mean
    e_highlight = rng.uniform(0.9, 1.5) / p995  # phones protect highlights, not perfectly
    return min(e_mid, e_highlight) * 2 ** rng.normal(0, 0.3)


def defocus(img: np.ndarray, depth: np.ndarray, rng: np.random.Generator, ss: int) -> tuple[np.ndarray, dict]:
    """Depth-dependent blur: blend between a sharp and a blurred copy."""
    focus = float(np.median(depth)) * rng.uniform(0.9, 1.1)
    coc = np.abs(1 / depth - 1 / focus) * focus  # relative circle of confusion
    # Phones have deep depth of field: usually everything is sharp.
    max_sigma = (rng.uniform(1.0, 4.0) if rng.random() < 0.25 else rng.uniform(0.0, 0.8)) * ss
    sigma_map = np.clip(coc * max_sigma * 4, 0, max_sigma)
    base_sigma = rng.uniform(0.2, 0.7) * ss  # lens softness everywhere
    sharp = cv2.GaussianBlur(img, (0, 0), base_sigma)
    if sigma_map.max() < 0.5:
        return sharp, {"base_sigma_px": base_sigma / ss, "max_defocus_px": 0.0}
    blurred = cv2.GaussianBlur(img, (0, 0), float(sigma_map.max()))
    t = (sigma_map / sigma_map.max())[..., None]
    return sharp * (1 - t) + blurred * t, {"base_sigma_px": base_sigma / ss, "max_defocus_px": float(sigma_map.max()) / ss}


def motion_blur(img: np.ndarray, rng: np.random.Generator, ss: int) -> tuple[np.ndarray, float]:
    length = rng.uniform(2, 10) * ss
    k = int(np.ceil(length)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel), length / ss


def develop(
    radiance: np.ndarray,
    depth: np.ndarray,
    light_color: np.ndarray,
    rng: np.random.Generator,
    ss: int,
    out_size: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Returns (uint8 BGR image at out_size (w, h), info)."""
    info: dict = {}
    img, info["defocus"] = defocus(radiance, depth, rng, ss)
    if rng.random() < 0.15:
        img, info["motion_blur_px"] = motion_blur(img, rng, ss)

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    vig = rng.uniform(0.0, 0.35)
    img = img * (1 - vig * r2)[..., None]
    info["vignette"] = vig

    exposure = auto_exposure(img, rng)
    img = img * exposure
    info["exposure"] = exposure

    img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)

    # Sensor noise: shot noise grows with exposure gain (dark scenes are noisier).
    gain = np.clip(exposure / 3.0, 0.3, 8.0)
    shot = rng.uniform(0.0003, 0.002) * gain
    read = rng.uniform(0.0005, 0.003) * np.sqrt(gain)
    sigma = np.sqrt(np.clip(img, 0, None) * shot + read**2)
    img = img + rng.standard_normal(img.shape, dtype=np.float32) * sigma
    info["noise"] = {"shot": float(shot), "read": float(read)}

    # Auto white balance: only partially removes the illuminant color.
    completeness = 0.5 + 0.5 * rng.beta(4, 1.5)  # usually good, sometimes lazy
    est = light_color * completeness + (1 - completeness) * np.ones(3, np.float32)
    est = est * np.exp(rng.normal(0, 0.03, 3))  # AWB estimation error
    gains = est[1] / est
    img = img * gains[None, None, :].astype(np.float32)
    info["awb"] = {"completeness": completeness, "gains": gains.tolist()}

    # Tone: soft highlight rolloff, sRGB encode, mild S-curve.
    img = np.clip(img, 0, None)
    knee = rng.uniform(0.75, 0.95)
    over = np.clip(img - knee, 0, None)
    img = np.where(img > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), img)
    out = linear_to_srgb(img)
    contrast = rng.uniform(0.0, 0.35)
    out = out * (1 - contrast) + contrast * (out * out * (3 - 2 * out))

    out8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1].copy()  # BGR

    # Phone ISP: chroma denoise + sharpening.
    y, cr, cb = cv2.split(cv2.cvtColor(out8, cv2.COLOR_BGR2YCrCb))
    s = rng.uniform(0.5, 2.0)
    out8 = cv2.cvtColor(cv2.merge([y, cv2.GaussianBlur(cr, (0, 0), s), cv2.GaussianBlur(cb, (0, 0), s)]), cv2.COLOR_YCrCb2BGR)
    amount = rng.uniform(0.0, 0.8)
    if amount > 0.05:
        blur = cv2.GaussianBlur(out8, (0, 0), rng.uniform(0.8, 1.6))
        out8 = cv2.addWeighted(out8, 1 + amount, blur, -amount, 0)

    q = int(rng.integers(60, 96))
    ok, buf = cv2.imencode(".jpg", out8, [cv2.IMWRITE_JPEG_QUALITY, q])
    out8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    info["jpeg_quality"] = q
    info["sharpen"] = amount
    return out8, info
