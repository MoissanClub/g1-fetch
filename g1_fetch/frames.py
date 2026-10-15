"""Coordinate frames.

pelvis  : URDF root of g1_body23, x forward, y left, z up. All manipulation targets live here.
torso   : pelvis + waist yaw (assumed 0 during manipulation).
d435    : camera body link from the URDF (x forward along the optical axis, pitched 47.6 deg down).
optical : RealSense convention, x right, y down, z forward. Depth values are along optical z.
world   : Unitree odometry frame anchored where the robot was powered on (rt/odommodestate).
floor   : heights above the floor = pelvis z + pelvis_height_m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Optical axes expressed in the camera link frame (columns): x_opt=-y_link, y_opt=-z_link, z_opt=x_link
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def make_T(xyz, rpy=(0.0, 0.0, 0.0), R: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy) if R is None else R
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_T(T: np.ndarray, p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return p @ T[:3, :3].T + T[:3, 3]


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def deproject(self, u, v, z):
        """Pixel(s) + depth (m, along optical z) -> optical-frame xyz."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        z = np.asarray(z, dtype=float)
        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=-1)

    def project(self, p_opt):
        p = np.asarray(p_opt, dtype=float)
        z = p[..., 2]
        return np.stack([p[..., 0] / z * self.fx + self.cx, p[..., 1] / z * self.fy + self.cy], axis=-1)

    @staticmethod
    def from_fov(width: int, height: int, hfov_deg: float = 69.0, vfov_deg: float = 42.0) -> "Intrinsics":
        fx = width / 2 / math.tan(math.radians(hfov_deg) / 2)
        fy = height / 2 / math.tan(math.radians(vfov_deg) / 2)
        return Intrinsics(fx, fy, width / 2, height / 2, width, height)

    @staticmethod
    def from_hfov(width: int, height: int, hfov_deg: float) -> "Intrinsics":
        """Square pixels: fy = fx derived from the horizontal FOV."""
        fx = width / 2 / math.tan(math.radians(hfov_deg) / 2)
        return Intrinsics(fx, fx, width / 2, height / 2, width, height)

    def bearing(self, u: float) -> float:
        """Horizontal angle of a pixel column (rad, +left in the body sense: left of centre is positive)."""
        return -math.atan((u - self.cx) / self.fx)

    def elevation(self, v: float) -> float:
        """Vertical angle of a pixel row (rad, +up)."""
        return -math.atan((v - self.cy) / self.fy)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float

    def world_to_body(self, wx: float, wy: float) -> tuple[float, float]:
        dx, dy = wx - self.x, wy - self.y
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        return c * dx - s * dy, s * dx + c * dy

    def body_to_world(self, bx: float, by: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * bx - s * by, self.y + s * bx + c * by


class Frames:
    """Static transforms from the config."""

    def __init__(self, cfg):
        cam = cfg.camera
        self.T_pelvis_torso = make_T(cam.pelvis_T_torso.xyz, cam.pelvis_T_torso.rpy)
        self.T_torso_link = make_T(cam.torso_T_link.xyz, cam.torso_T_link.rpy)
        self.T_link_optical = make_T((0, 0, 0), R=R_LINK_OPTICAL)
        self.T_pelvis_optical = self.T_pelvis_torso @ self.T_torso_link @ self.T_link_optical
        self.T_optical_pelvis = inv_T(self.T_pelvis_optical)
        self.camera_in_pelvis = self.T_pelvis_optical[:3, 3].copy()
        # floor offset: camera height measured on the robot minus camera z in the pelvis frame
        self.pelvis_height_m = float(cfg.robot.camera_height_m) - float(self.camera_in_pelvis[2])
        self.camera_pitch_down = float(cam.torso_T_link.rpy[1])
        # optional forward RGB camera on the head (cfg.cameras.head.mount, torso frame; x forward along its axis)
        head = cfg.get("cameras")
        head = None if head is None else head.get("head")
        self.has_head = head is not None and bool(head.get("enabled", False))
        if head is not None and head.get("mount") is not None:
            self.T_torso_head = make_T(head.mount.xyz, head.mount.rpy)
            self.T_pelvis_head_optical = self.T_pelvis_torso @ self.T_torso_head @ self.T_link_optical
            self.T_head_optical_pelvis = inv_T(self.T_pelvis_head_optical)
            self.head_in_pelvis = self.T_pelvis_head_optical[:3, 3].copy()
            self.head_pitch_down = float(head.mount.rpy[1])
            self.head_yaw = float(head.mount.rpy[2])

    def optical_to_pelvis(self, p):
        return apply_T(self.T_pelvis_optical, p)

    def pelvis_to_optical(self, p):
        return apply_T(self.T_optical_pelvis, p)

    def floor_height(self, p_pelvis) -> float:
        return float(np.asarray(p_pelvis)[2] + self.pelvis_height_m)

    def pelvis_z_from_floor(self, h: float) -> float:
        return h - self.pelvis_height_m

    def head_optical_to_pelvis(self, p):
        return apply_T(self.T_pelvis_head_optical, p)

    def head_ray_hit_plane_x(self, u: float, v: float, intr, x_plane: float) -> np.ndarray:
        """Pelvis-frame point where the head-camera ray through pixel (u, v) meets the vertical plane x = x_plane."""
        d_opt = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
        R = self.T_pelvis_head_optical[:3, :3]
        o = self.T_pelvis_head_optical[:3, 3]
        d = R @ d_opt
        t = (x_plane - o[0]) / d[0] if abs(d[0]) > 1e-6 else float("inf")
        return o + t * d

    def head_bearing_to_body(self, bearing: float) -> float:
        """Head-camera horizontal bearing (rad) -> yaw error in the body frame (mount yaw added)."""
        return bearing + self.head_yaw

    def view_ceiling(self, range_m: float, vfov_deg: float = 42.0) -> float:
        """Highest floor height visible at a given forward range from the camera (RGB by default)."""
        top = self.camera_pitch_down - math.radians(vfov_deg) / 2
        cam_h = float(self.camera_in_pelvis[2] + self.pelvis_height_m)
        return cam_h - range_m * math.tan(top)
