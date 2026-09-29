"""Pinhole camera looking at the table plane z = 0 (world units: millimeters).

World frame: x/y on the table, z up. Camera frame follows OpenCV: x right,
y down, z forward. For points on the plane the projection collapses to a
homography H = K [r1 r2 t] mapping (x_mm, y_mm, 1) -> image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 world->camera rotation
    C: np.ndarray  # camera center in world (mm)
    width: int
    height: int
    tilt_deg: float
    fov_diag_deg: float

    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.C

    @property
    def H(self) -> np.ndarray:
        """Plane (x, y) mm -> image pixels."""
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return H / H[2, 2]

    def scaled(self, s: float) -> "Camera":
        """Same camera rendering at s times the resolution (for supersampling)."""
        K = self.K.copy()
        K[:2] *= s
        # keep pixel-center convention consistent: (u + 0.5) * s - 0.5
        K[0, 2] += 0.5 * s - 0.5
        K[1, 2] += 0.5 * s - 0.5
        return Camera(K, self.R, self.C, int(self.width * s), int(self.height * s), self.tilt_deg, self.fov_diag_deg)

    def project_plane(self, pts_mm: np.ndarray) -> np.ndarray:
        p = np.column_stack([pts_mm, np.ones(len(pts_mm))]) @ self.H.T
        return p[:, :2] / p[:, 2:3]

    def unproject_to_plane(self, pts_px: np.ndarray) -> np.ndarray:
        Hi = np.linalg.inv(self.H)
        p = np.column_stack([pts_px, np.ones(len(pts_px))]) @ Hi.T
        return p[:, :2] / p[:, 2:3]

    def plane_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates (X, Y) in mm for every pixel center."""
        Hi = np.linalg.inv(self.H).astype(np.float64)
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        d = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
        X = (Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]) / d
        Y = (Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]) / d
        return X.astype(np.float32), Y.astype(np.float32)

    def px_per_mm_max(self) -> float:
        """Largest local magnification (image px per plane mm) over the frame."""
        corners = np.array([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]], float)
        best = 0.0
        for q in self.unproject_to_plane(corners):
            a = self.project_plane(np.array([q, q + [1, 0], q + [0, 1]]))
            best = max(best, np.linalg.norm(a[1] - a[0]), np.linalg.norm(a[2] - a[0]))
        return float(best)


def look_at(target: np.ndarray, azimuth: float, tilt: float, distance: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera at `distance` from `target`, tilted `tilt` rad from straight down.

    `azimuth` is the direction (in the table plane) the photographer stands
    *from* the target. The far side of the table appears at the top of the image.
    """
    back = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    C = target + distance * (np.sin(tilt) * back + np.array([0, 0, np.cos(tilt)]))
    f = (target - C) / np.linalg.norm(target - C)
    away = -back  # image "up" direction on the table
    up = away - (away @ f) * f
    up /= np.linalg.norm(up)
    y = -up  # image y points down
    x = np.cross(y, f)
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    R = np.stack([x, y, f])
    return R, C


def intrinsics(width: int, height: int, fov_diag_deg: float) -> np.ndarray:
    diag = np.hypot(width, height)
    f = (diag / 2) / np.tan(np.radians(fov_diag_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]], float)


def sample_camera(
    rng: np.random.Generator,
    width: int,
    height: int,
    points_mm: np.ndarray,
    occupancy: float,
    offset: tuple[float, float] = (0.0, 0.0),
    azimuth: float | None = None,
) -> Camera:
    """Sample a phone-like handheld camera and frame `points_mm` (the card corners).

    Pose (FOV, tilt, azimuth, roll) is sampled first; distance and aim are then
    solved iteratively so the points' image bounding box fills `occupancy` of the
    frame along its limiting axis, centered at `offset` (fractions of the frame
    size from the image center).

    Tilt is mostly near top-down (the typical "photo of the table" shot),
    with a tail toward oblique views.
    """
    fov = float(rng.choice([rng.uniform(64, 80), rng.uniform(45, 64)], p=[0.85, 0.15]))
    tilt = np.radians(min(abs(rng.normal(0, 14)), 42.0))
    if azimuth is None:
        azimuth = rng.uniform(0, 2 * np.pi)
    roll = np.radians(rng.normal(0, 4))
    K = intrinsics(width, height, fov)

    lo, hi = points_mm.min(0), points_mm.max(0)
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
    distance = K[0, 0] * float(max(hi - lo)) / (occupancy * min(width, height))
    desired = np.array([width * (0.5 + offset[0]), height * (0.5 + offset[1])])

    for _ in range(12):
        R, C = look_at(target, azimuth, tilt, distance, roll)
        cam = Camera(K, R, C, width, height, float(np.degrees(tilt)), fov)
        px = cam.project_plane(points_mm)
        occ = max(np.ptp(px[:, 0]) / width, np.ptp(px[:, 1]) / height)
        center = (px.min(0) + px.max(0)) / 2
        if abs(occ / occupancy - 1) < 0.005 and np.linalg.norm(center - desired) < 1.0:
            break
        distance *= occ / occupancy  # zoom: image size ~ 1 / distance
        # pan: shift so the plane point under the bbox center moves to `desired`
        pc, pd = cam.unproject_to_plane(np.array([center, desired]))
        target[:2] += pc - pd

    # Every pixel ray must hit the table in front of the camera.
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], float)
    rays = (np.linalg.inv(K) @ np.column_stack([corners, np.ones(4)]).T).T @ cam.R  # world-frame dirs
    assert (rays[:, 2] < -1e-3).all(), "camera sees the horizon"
    return cam
