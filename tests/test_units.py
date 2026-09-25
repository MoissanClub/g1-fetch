import math

import numpy as np
import pytest

from g1_fetch.frames import Frames, Intrinsics, Pose2D, wrap_angle
from g1_fetch.perception.camera import Frame
from g1_fetch.perception.detector import Detection, Stabilizer
from g1_fetch.perception.localize import door_plane, fit_plane_ransac, handle_from_prior, median_depth
from g1_fetch.skills.door import DoorModel, door_from_plane


def test_optical_to_pelvis_points_down(cfg_default):
    fr = Frames(cfg_default)
    p = fr.optical_to_pelvis([0.0, 0.0, 1.0])            # 1 m along the optical axis
    assert p[0] > 0.6 and p[2] < fr.camera_in_pelvis[2] - 0.6   # forward and well below the camera
    assert abs(fr.pelvis_height_m - (cfg_default.robot.camera_height_m - fr.camera_in_pelvis[2])) < 1e-9
    # view ceiling: nothing above ~1 m is visible in RGB beyond 0.4 m
    assert fr.view_ceiling(0.4) < 1.0 and fr.view_ceiling(2.4) < 0.05


def test_intrinsics_roundtrip():
    intr = Intrinsics.from_fov(640, 480)
    p = intr.deproject(400.0, 100.0, 1.7)
    uv = intr.project(p)
    assert np.allclose(uv, [400.0, 100.0])


def test_pose2d_roundtrip():
    p = Pose2D(1.0, -2.0, 0.7)
    wx, wy = p.body_to_world(0.5, 0.2)
    bx, by = p.world_to_body(wx, wy)
    assert abs(bx - 0.5) < 1e-9 and abs(by - 0.2) < 1e-9
    assert abs(wrap_angle(3 * math.pi)) - math.pi < 1e-9


def _plane_frame(fr, x_plane=0.9, yaw=0.0, w=640, h=480):
    """Synthetic depth of a vertical plane at pelvis x = x_plane (rotated by yaw about z) plus a floor."""
    intr = Intrinsics.from_fov(w, h)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    d_opt = intr.deproject(us, vs, np.ones((h, w)))
    R = fr.T_pelvis_optical[:3, :3]
    o = fr.T_pelvis_optical[:3, 3]
    d = d_opt @ R.T
    n = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])       # toward the robot
    dist = -(n @ np.array([x_plane, 0.0, 0.0]))
    denom = d @ n
    t_plane = -(o @ n + dist) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    t_floor = (-fr.pelvis_height_m - o[2]) / d[..., 2]
    t = np.where((t_plane > 0), t_plane, np.inf)
    t = np.where((t_floor > 0) & (t_floor < t), t_floor, t)
    depth = np.where(np.isfinite(t), t * d_opt[..., 2], 0.0).astype(np.float32)
    color = np.zeros((h, w, 3), np.uint8)
    return Frame(color, depth, intr, 0.0)


def test_door_plane_fit_recovers_distance_and_yaw(cfg_default):
    fr = Frames(cfg_default)
    for yaw in (0.0, 0.2, -0.15):
        f = _plane_frame(fr, x_plane=0.9, yaw=yaw)
        plane = door_plane(f, (0, 0, 639, 479), fr)
        assert plane is not None
        assert abs(plane.distance_x - 0.9) < 0.02
        assert abs(plane.yaw_error - yaw) < 0.03, (plane.yaw_error, yaw)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(0)
    pts = np.column_stack([np.full(300, 0.8), rng.uniform(-0.3, 0.3, 300), rng.uniform(-0.5, 0.4, 300)])
    pts += rng.normal(0, 0.003, pts.shape)
    outl = rng.uniform(-1, 1, (60, 3))
    n, d, inl = fit_plane_ransac(np.vstack([pts, outl]))
    assert abs(abs(n[0]) - 1) < 0.02 and inl[:300].mean() > 0.95 and inl[300:].mean() < 0.2


def test_median_depth_ignores_invalid():
    depth = np.zeros((100, 100), np.float32)
    depth[40:60, 40:60] = 1.5
    depth[50, 50] = 0.0
    f = Frame(np.zeros((100, 100, 3), np.uint8), depth, Intrinsics.from_fov(100, 100), 0.0)
    assert abs(median_depth(f, (30, 30, 70, 70), shrink=0.5) - 1.5) < 1e-6
    assert median_depth(f, (0, 0, 10, 10)) is None


def test_stabilizer_needs_consecutive_frames():
    s = Stabilizer("fridge", 3)
    d = Detection("fridge", 0.9, (10, 10, 100, 100))
    assert s.update([d]) is None and s.update([d]) is None and s.update([d]) is not None
    assert s.update([]) is None
    assert s.update([d]) is None


def test_handle_prior_and_door_model(cfg_default):
    fr = Frames(cfg_default)
    f = _plane_frame(fr, x_plane=0.9)
    plane = door_plane(f, (0, 0, 639, 479), fr)
    # a 0.6 m door whose free edge is the right edge of a wide plane -> handle inset from the right edge
    p = handle_from_prior(plane, cfg_default, fr)
    assert abs(fr.floor_height(p) - cfg_default.fridge.handle_grasp_height_m) < 1e-6
    assert abs(p[0] - (0.9 - cfg_default.fridge.handle_protrusion_m)) < 0.02
    door = door_from_plane(plane, Pose2D(0, 0, 0), p, cfg_default)
    assert door.hinge_side == "left"
    assert abs(door.handle_w(0.0)[0] - p[0]) < 1e-6 and abs(door.handle_w(0.0)[1] - p[1]) < 1e-6
    # opening toward the robot moves the free edge to negative x
    assert door.edge_w(math.pi / 2)[0] < door.hinge_w[0] - 0.5
    assert abs(door.theta_from_point(door.handle_w(0.6)) - 0.6) < 1e-6


def test_door_model_right_hinge():
    d = DoorModel(np.array([1.0, -0.3, 0.2]), np.array([-1.0, 0.0, 0.0]), 0.6, 0.05, 0.04, "right", 0.2)
    assert np.allclose(d.along_w, [0, 1, 0])
    assert d.edge_w(math.pi / 2)[0] < 0.5 and d.edge_w(math.pi / 2)[1] == pytest.approx(-0.3)
    assert np.allclose(d.tangent_w(0.0), [-1, 0, 0])
