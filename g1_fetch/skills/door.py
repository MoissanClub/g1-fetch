"""Door geometry in the world (odometry) frame so it survives the robot moving.

Conventions: the robot faces the closed door along -normal_w. "right" as seen by the robot is
(-normal) x z. Opening rotates the door about the hinge toward the robot: clockwise (about +z)
for a left hinge, counter-clockwise for a right hinge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..frames import Pose2D
from ..perception.localize import Plane
from .common import rot_pelvis, rot_world, to_pelvis, to_world


def _rot_z(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


@dataclass
class DoorModel:
    hinge_w: np.ndarray        # hinge axis point in world (x, y, z at handle height)
    normal_w: np.ndarray       # closed-door outward normal in world (unit, horizontal, toward the robot)
    width: float               # hinge -> free edge
    handle_inset: float
    handle_protrusion: float
    hinge_side: str            # "left" | "right" as seen by a robot facing the closed door
    handle_z: float            # pelvis-frame z of the grasp point
    theta: float = 0.0         # current opening angle (rad, >= 0)

    @property
    def sgn(self) -> float:
        return -1.0 if self.hinge_side == "left" else 1.0

    @property
    def along_w(self) -> np.ndarray:
        """Unit vector along the closed door from the hinge to the free edge (world)."""
        n = self.normal_w
        right = np.array([-n[1], n[0], 0.0])        # (-n) x z
        return right if self.hinge_side == "left" else -right

    def edge_w(self, theta: float | None = None, r: float | None = None) -> np.ndarray:
        th = self.theta if theta is None else theta
        r = self.width if r is None else r
        return self.hinge_w + r * _rot_z(self.along_w, self.sgn * th)

    def handle_w(self, theta: float | None = None) -> np.ndarray:
        th = self.theta if theta is None else theta
        return self.edge_w(th, self.width - self.handle_inset) + self.handle_protrusion * self.door_normal_w(th)

    def door_normal_w(self, theta: float | None = None) -> np.ndarray:
        """Outer-face unit normal of the door at angle theta (toward the robot when closed)."""
        th = self.theta if theta is None else theta
        return _rot_z(self.normal_w, self.sgn * th)

    def tangent_w(self, theta: float | None = None) -> np.ndarray:
        """Unit direction the free edge moves when the door opens further."""
        th = self.theta if theta is None else theta
        d = self.edge_w(th + 1e-3) - self.edge_w(th - 1e-3)
        return d / np.linalg.norm(d)

    def theta_from_point(self, p_w: np.ndarray, on_handle: bool = True) -> float:
        """Opening angle implied by a point on the door (e.g. the hand holding the handle).

        With on_handle=True the handle's protrusion in front of the panel is compensated.
        """
        v = np.asarray(p_w, dtype=float) - self.hinge_w
        v[2] = 0.0
        a = self.along_w
        ang = self.sgn * math.atan2(a[0] * v[1] - a[1] * v[0], a[0] * v[0] + a[1] * v[1])
        if on_handle:
            ang -= math.atan2(self.handle_protrusion, self.width - self.handle_inset)
        return max(0.0, ang)

    def cavity_center_w(self) -> np.ndarray:
        """Centre of the opening on the closed-door plane (world)."""
        return self.hinge_w + 0.5 * self.width * self.along_w

    # ---- pelvis-frame views for the current robot pose ----
    def handle_pelvis(self, pose: Pose2D, theta: float | None = None) -> np.ndarray:
        return to_pelvis(pose, self.handle_w(theta))

    def edge_pelvis(self, pose: Pose2D, theta: float | None = None, r: float | None = None) -> np.ndarray:
        return to_pelvis(pose, self.edge_w(theta, r))

    def normal_pelvis(self, pose: Pose2D, theta: float | None = None) -> np.ndarray:
        return rot_pelvis(pose, self.door_normal_w(theta))

    def tangent_pelvis(self, pose: Pose2D, theta: float | None = None) -> np.ndarray:
        return rot_pelvis(pose, self.tangent_w(theta))

    def facing_yaw(self) -> float:
        """World yaw that faces the closed door squarely."""
        return math.atan2(-self.normal_w[1], -self.normal_w[0])

    def standoff_pose_w(self, standoff: float, lateral: float = 0.0, at: np.ndarray | None = None) -> tuple[float, float, float]:
        """World (x, y, yaw) of a pelvis pose `standoff` in front of a plane point (default cavity centre),
        shifted by `lateral` metres along the robot's +y (left)."""
        p = self.cavity_center_w() if at is None else at
        n = self.normal_w
        left = np.array([n[1], -n[0], 0.0])           # robot +y when facing the door = -((-n) x z)
        q = p + standoff * n + lateral * left
        return float(q[0]), float(q[1]), self.facing_yaw()


def door_from_plane(plane: Plane, pose: Pose2D, handle_pelvis: np.ndarray, cfg, width: float | None = None) -> DoorModel:
    """World-frame door model from the fitted plane and the chosen handle point (pelvis frame)."""
    f = cfg.fridge
    width = float(width or f.door_width_m)
    n_p = np.array([plane.n[0], plane.n[1], 0.0])
    n_p /= np.linalg.norm(n_p)
    normal_w = rot_world(pose, n_p)
    on_plane = np.array(handle_pelvis, dtype=float)
    on_plane += float(f.handle_protrusion_m) * (-n_p)      # back onto the plane (n points toward the robot)
    handle_w = to_world(pose, on_plane)
    right_w = np.array([-normal_w[1], normal_w[0], 0.0])
    along = right_w if f.hinge_side == "left" else -right_w
    hinge_w = handle_w - (width - float(f.handle_inset_m)) * along
    return DoorModel(hinge_w, normal_w, width, float(f.handle_inset_m), float(f.handle_protrusion_m),
                     str(f.hinge_side), float(handle_pelvis[2]))
