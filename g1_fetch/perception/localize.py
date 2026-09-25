"""From detections + depth to 3D targets in the pelvis frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import Frames
from .camera import Frame
from .detector import Detection


@dataclass
class Plane:
    """n . p = d in the pelvis frame, with n pointing toward the robot (negative x)."""
    n: np.ndarray
    d: float
    inliers: np.ndarray            # Nx3 pelvis-frame points
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def distance_x(self) -> float:
        """Forward distance from the pelvis origin to the plane along +x (positive when ahead)."""
        return float(self.d / self.n[0]) if abs(self.n[0]) > 1e-6 else float("inf")

    @property
    def yaw_error(self) -> float:
        """Rotation about z that would make the robot face the plane squarely (radians, +ccw)."""
        # Squared up means heading == -n, so the required heading change is atan2(-n.y, -n.x) (+ = turn left).
        return float(np.arctan2(-self.n[1], -self.n[0]))

    @property
    def width(self) -> float:
        return float(self.y_max - self.y_min)

    def point_at(self, y: float, z: float) -> np.ndarray:
        x = (self.d - self.n[1] * y - self.n[2] * z) / self.n[0]
        return np.array([x, y, z])


def box_pixels(box, shrink: float = 0.0, step: int = 1):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 += w * shrink / 2
    x2 -= w * shrink / 2
    y1 += h * shrink / 2
    y2 -= h * shrink / 2
    us = np.arange(int(max(0, x1)), int(x2) + 1, step)
    vs = np.arange(int(max(0, y1)), int(y2) + 1, step)
    return np.meshgrid(us, vs)


def median_depth(frame: Frame, box, shrink: float = 0.4) -> float | None:
    """Median valid depth (m) in the central part of a box; None if too few valid pixels."""
    uu, vv = box_pixels(box, shrink)
    uu = np.clip(uu, 0, frame.depth.shape[1] - 1)
    vv = np.clip(vv, 0, frame.depth.shape[0] - 1)
    z = frame.depth[vv, uu]
    z = z[z > 0]
    if z.size < 10:
        return None
    return float(np.median(z))


def box_to_point(frame: Frame, det: Detection, frames: Frames, shrink: float = 0.4) -> np.ndarray | None:
    """3D point (pelvis frame) of the box centre using the box's median depth."""
    z = median_depth(frame, det.box, shrink)
    if z is None:
        return None
    u, v = det.center
    p_opt = frame.intrinsics.deproject(u, v, z)
    return frames.optical_to_pelvis(p_opt)


def box_cloud(frame: Frame, box, frames: Frames, step: int = 4, shrink: float = 0.0) -> np.ndarray:
    """Pelvis-frame points for valid depth pixels inside a box."""
    uu, vv = box_pixels(box, shrink, step)
    uu = np.clip(uu, 0, frame.depth.shape[1] - 1)
    vv = np.clip(vv, 0, frame.depth.shape[0] - 1)
    z = frame.depth[vv, uu]
    ok = z > 0
    if ok.sum() == 0:
        return np.zeros((0, 3))
    p_opt = frame.intrinsics.deproject(uu[ok], vv[ok], z[ok])
    return frames.optical_to_pelvis(p_opt)


def fit_plane_ransac(points: np.ndarray, thresh: float = 0.015, iters: int = 200, rng=None) -> tuple[np.ndarray, float, np.ndarray] | None:
    """RANSAC plane fit -> (n, d, inlier_mask) with n unit and n . p = d."""
    if points.shape[0] < 30:
        return None
    rng = rng or np.random.default_rng(0)
    best = None
    n_pts = points.shape[0]
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        a, b, c = points[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = float(n @ a)
        dist = np.abs(points @ n - d)
        inl = dist < thresh
        if best is None or inl.sum() > best[2].sum():
            best = (n, d, inl)
    if best is None:
        return None
    n, d, inl = best
    # refine with SVD on inliers
    P = points[inl]
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n)
    d = float(n @ c)
    inl = np.abs(points @ n - d) < thresh
    return n, d, inl


def door_plane(frame: Frame, box, frames: Frames, min_height: float = 0.05) -> Plane | None:
    """Vertical plane of the fridge front inside a box, in the pelvis frame.

    Points below `min_height` above the floor are dropped (floor), the plane is fitted by RANSAC,
    and rejected unless it is near-vertical and facing the robot.
    """
    pts = box_cloud(frame, box, frames, step=3)
    if pts.shape[0] == 0:
        return None
    pts = pts[pts[:, 2] + frames.pelvis_height_m > min_height]
    fit = fit_plane_ransac(pts)
    if fit is None:
        return None
    n, d, inl = fit
    if n[0] > 0:            # make the normal point toward the robot
        n, d = -n, -d
    if abs(n[2]) > 0.35:    # not vertical enough (tilt > ~20 deg)
        return None
    if inl.sum() < 0.4 * pts.shape[0]:
        return None
    P = pts[inl]
    return Plane(n, d, P, float(P[:, 1].min()), float(P[:, 1].max()), float(P[:, 2].min()), float(P[:, 2].max()))


def cavity_depth_jump(before: Frame, after: Frame, box) -> float | None:
    """Increase of median depth inside a box between two frames (door open -> cavity visible)."""
    a = median_depth(before, box, shrink=0.5)
    b = median_depth(after, box, shrink=0.5)
    if a is None or b is None:
        return None
    return float(b - a)


def handle_from_prior(plane: Plane, cfg, frames: Frames, door_width_measured: float | None = None) -> np.ndarray:
    """Handle bar centre from the door plane + fridge priors (pelvis frame)."""
    f = cfg.fridge
    width = door_width_measured or float(f.door_width_m)
    # the free (handle) edge is opposite the hinge
    if f.hinge_side == "left":
        edge_y = plane.y_max - width if plane.width > width * 1.5 else plane.y_min
        handle_y = edge_y + float(f.handle_inset_m)
    else:
        edge_y = plane.y_min + width if plane.width > width * 1.5 else plane.y_max
        handle_y = edge_y - float(f.handle_inset_m)
    lo, hi = f.handle_height_band
    h = float(np.clip(float(f.handle_grasp_height_m), lo, hi))
    z = frames.pelvis_z_from_floor(h)
    p = plane.point_at(handle_y, z)
    p[0] -= float(f.handle_protrusion_m)
    return p


def can_axis_from_box(frame: Frame, det: Detection, frames: Frames, cfg) -> np.ndarray | None:
    """Can axis point at grasp height: box centre depth is the front surface, so push back by the radius."""
    p = box_to_point(frame, det, frames, shrink=0.5)
    if p is None:
        return None
    p = p.copy()
    p[0] += float(cfg.fridge.can_diameter_m) / 2
    return p
