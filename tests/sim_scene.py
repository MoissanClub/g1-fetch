"""A tiny ray-cast world for dry runs: floor, fridge front + door panel + cavity, a can, a person.

It renders the depth image the D435i would see from the mock robot's odometry pose and produces
scripted detections from projected 3D boxes, so the whole task logic can run without hardware.
"""
from __future__ import annotations

import math

import numpy as np

from g1_fetch.frames import Pose2D, apply_T, inv_T, make_T
from g1_fetch.perception.detector import Detection
from g1_fetch.skills.common import to_world
from g1_fetch.skills.door import DoorModel


class Scene:
    def __init__(self, rb, fridge_x: float = 1.8, fridge_yc: float = 0.0, width: float = 0.6, height: float = 1.7,
                 hinge_side: str = "left", handle_h: float = 1.0, can_h: float = 0.95, person_xy=(-1.5, 0.0)):
        self.rb = rb
        self.fr = rb.frames
        self.width = width
        self.height = height
        self.hinge_side = hinge_side
        self.can_h = can_h
        self.person_xy = np.array(person_xy, dtype=float)
        n = np.array([-1.0, 0.0, 0.0])
        yh = fridge_yc + (width / 2 if hinge_side == "left" else -width / 2)
        hz = self.fr.pelvis_z_from_floor(handle_h)
        f = rb.cfg.fridge
        self.door = DoorModel(np.array([fridge_x, yh, hz]), n, width, float(f.handle_inset_m), float(f.handle_protrusion_m),
                              hinge_side, hz, 0.0)
        self.cavity_depth = 0.5
        # can standing on a shelf inside, centred laterally
        cc = self.door.cavity_center_w()
        self.can_w = np.array([fridge_x + 0.12, cc[1], self.fr.pelvis_z_from_floor(can_h)])
        self.can_r = float(f.can_diameter_m) / 2
        self.can_held = False
        self.hooked = False
        self.log: list[str] = []

    # ---- world model updates driven by the mock robot ----
    def _update(self):
        rb = self.rb
        pose = rb.loco.pose()
        door_hand = rb.cfg.task.door_hand
        can_hand = rb.cfg.task.can_hand
        hand_w = to_world(pose, rb.kin.fk(door_hand, rb.arms.measured(door_hand))[:3, 3])
        closed = rb.hands[door_hand].closed_fraction() or 0.0
        dh = hand_w - self.door.handle_w()
        if not self.hooked and closed > 0.5 and math.hypot(dh[0], dh[1]) < 0.06 and abs(dh[2]) < 0.18:
            self.hooked = True
            self.log.append("hooked")
        if self.hooked:
            if closed < 0.3:
                self.hooked = False
                self.log.append(f"released at {math.degrees(self.door.theta):.0f} deg")
            else:
                self.door.theta = self.door.theta_from_point(hand_w)
        # stage B / closing pushes: any hand within 4 cm of the free-edge region moves the door with it
        for side in (door_hand,):
            hw = to_world(pose, rb.kin.fk(side, rb.arms.measured(side))[:3, 3])
            v = hw - self.door.hinge_w
            r = math.hypot(v[0], v[1])
            if not self.hooked and 0.3 < r < self.door.width + 0.06 and abs(hw[2] - self.door.handle_z) < 0.25:
                th = self.door.theta_from_point(hw, on_handle=False)
                # the hand pushes the panel only when it is on the panel's line within tolerance
                if abs(th - self.door.theta) < 0.35:
                    self.door.theta = max(0.0, min(math.radians(110), th))
        # can follows the can hand once grasped near it
        chw = to_world(pose, rb.kin.fk(can_hand, rb.arms.measured(can_hand))[:3, 3])
        cclosed = rb.hands[can_hand].closed_fraction() or 0.0
        if not self.can_held and cclosed > 0.5 and np.linalg.norm(chw - self.can_w) < 0.08:
            self.can_held = True
            self.log.append("can grasped")
        if self.can_held:
            if cclosed < 0.3:
                self.can_held = False
                self.log.append("can released")
            else:
                self.can_w = chw.copy()

    # ---- rendering ----
    def camera_pose_w(self, pose: Pose2D) -> np.ndarray:
        T_w_pelvis = make_T([pose.x, pose.y, 0.0], [0.0, 0.0, pose.yaw])
        return T_w_pelvis @ self.fr.T_pelvis_optical

    def render(self, seq: int):
        self._update()
        rb = self.rb
        intr = rb.camera.intrinsics
        pose = rb.loco.pose()
        T_w_cam = self.camera_pose_w(pose)
        R, o = T_w_cam[:3, :3], T_w_cam[:3, 3]
        us, vs = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
        d_cam = intr.deproject(us, vs, np.ones_like(us, dtype=float))
        d_w = d_cam @ R.T
        best = np.full(us.shape, np.inf)
        # floor (world z = -pelvis_height)
        floor_z = -self.fr.pelvis_height_m
        tz = (floor_z - o[2]) / d_w[..., 2]
        best = np.where((d_w[..., 2] < 0) & (tz > 0), np.minimum(best, tz), best)
        # fridge front plane x = fx, excluding the door opening (cavity) when the door is open
        fx = self.door.hinge_w[0]
        yl, yr = self._span()
        t = (fx - o[0]) / d_w[..., 0]
        hit = o[None, None, :] + t[..., None] * d_w
        on_front = (t > 0) & (hit[..., 1] > yl - 0.02) & (hit[..., 1] < yr + 0.02) & (hit[..., 2] > floor_z) \
            & (hit[..., 2] < floor_z + self.height)
        in_open = on_front & (self.door.theta > 0.05)
        best = np.where(on_front & ~in_open, np.minimum(best, t), best)
        # cavity back wall behind the opening
        tb = (fx + self.cavity_depth - o[0]) / d_w[..., 0]
        best = np.where(in_open & (tb > 0), np.minimum(best, tb), best)
        # door panel: plane through the hinge with the door's normal, segment 0..width
        nrm = self.door.door_normal_w()
        along = self.door.edge_w(r=1.0) - self.door.hinge_w
        denom = d_w @ nrm
        tp = ((self.door.hinge_w - o) @ nrm) / np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        hp = o[None, None, :] + tp[..., None] * d_w
        r = (hp - self.door.hinge_w) @ along
        on_panel = (tp > 0) & (r > 0) & (r < self.door.width) & (hp[..., 2] > floor_z) & (hp[..., 2] < floor_z + self.height)
        best = np.where(on_panel, np.minimum(best, tp), best)
        # can: vertical cylinder approximated by its axis-aligned box
        if not self.can_held:
            best = self._box(best, o, d_w, self.can_w + [-self.can_r, -self.can_r, -0.06], self.can_w + [self.can_r, self.can_r, 0.06])
        # person: 0.4 x 0.3 x 1.7 box
        p = self.person_xy
        best = self._box(best, o, d_w, np.array([p[0] - 0.15, p[1] - 0.2, floor_z]), np.array([p[0] + 0.15, p[1] + 0.2, floor_z + 1.7]))
        depth = np.where(np.isfinite(best), best * d_cam[..., 2], 0.0).astype(np.float32)
        depth[(depth < 0.15) | (depth > 4.0)] = 0.0
        color = np.zeros((intr.height, intr.width, 3), np.uint8)
        color[..., 1] = np.clip(depth * 60, 0, 255).astype(np.uint8)
        return color, depth

    def _span(self):
        a = self.door.along_w
        h = self.door.hinge_w
        e = h + self.door.width * a
        return min(h[1], e[1]), max(h[1], e[1])

    @staticmethod
    def _box(best, o, d, lo, hi):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - o) / d
            t2 = (hi - o) / d
        tmin = np.max(np.minimum(t1, t2), axis=-1)
        tmax = np.min(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= tmin) & (tmax > 0)
        t = np.where(tmin > 0, tmin, tmax)
        return np.where(hit, np.minimum(best, t), best)

    # ---- detections from projected boxes ----
    def detect(self, seq: int) -> list[Detection]:
        rb = self.rb
        intr = rb.camera.intrinsics
        pose = rb.loco.pose()
        T_cam_w = inv_T(self.camera_pose_w(pose))
        floor_z = -self.fr.pelvis_height_m
        out = []
        fx = self.door.hinge_w[0]
        yl, yr = self._span()
        fridge_pts = np.array([[fx, y, z] for y in (yl, yr) for z in (floor_z, floor_z + self.height)])
        out += self._proj("fridge", fridge_pts, T_cam_w, intr, 0.85, min_frac=0.06)
        if self.door.theta < 0.05:
            hw = self.door.handle_w()
            out += self._proj("handle", np.array([hw + [0, -0.015, -0.15], hw + [0, 0.015, 0.15]]), T_cam_w, intr, 0.7,
                              min_frac=0.3, min_px=8)
        if self.door.theta > 0.6 and not self.can_held:
            c = self.can_w
            r = self.can_r
            out += self._proj("can", np.array([c + [-r, -r, -0.06], c + [r, r, 0.06]]), T_cam_w, intr, 0.6, min_frac=0.3, min_px=8)
        p = self.person_xy
        out += self._proj("person", np.array([[p[0] - 0.15, p[1] - 0.2, floor_z], [p[0] + 0.15, p[1] + 0.2, floor_z + 1.7]]),
                          T_cam_w, intr, 0.9, min_frac=0.06)
        return out

    @staticmethod
    def _proj(label, corners_w, T_cam_w, intr, conf, min_frac=0.5, max_dist=3.5, min_px=24):
        """Project a grid over the 3D box; the detection is the bounding box of the in-image samples.

        The visible fraction (in-image samples / all samples) gates the detection like a real detector
        that needs to see a reasonable part of the object.
        """
        lo, hi = np.min(corners_w, axis=0), np.max(corners_w, axis=0)
        axes = [np.linspace(lo[i], hi[i], 1 if hi[i] - lo[i] < 1e-6 else 8) for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        pc = apply_T(T_cam_w, pts)
        front = pc[:, 2] > 0.1
        if not front.any() or np.min(pc[front, 2]) > max_dist:
            return []
        uv = intr.project(pc[front])
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < intr.width) & (uv[:, 1] >= 0) & (uv[:, 1] < intr.height)
        frac = inside.sum() / len(pts)
        if inside.sum() < 2 or frac < min_frac:
            return []
        x1, y1 = uv[inside].min(axis=0)
        x2, y2 = uv[inside].max(axis=0)
        if (x2 - x1) < min_px or (y2 - y1) < min_px:
            return []
        return [Detection(label, conf, (float(x1), float(y1), float(x2), float(y2)), "sim")]


def attach(rb, **kw) -> Scene:
    """Wire a Scene into a dry-run robot's SyntheticCamera, StubDetector and MockHands."""
    scene = Scene(rb, **kw)
    rb.camera.producer = scene.render
    rb.detector.script = scene.detect
    orig_event = rb.log.event

    def _event(name, **data):
        orig_event(name, **data)
        if name == "handover_extended":
            # the person takes the can ~0.5 s later: pull the elbow of the can hand
            import threading

            side = rb.cfg.task.can_hand

            def pull():
                rb.arms.disturbance[side] = np.array([0.0, 0.0, 0.0, 0.15, 0.0])

            threading.Timer(0.5, pull).start()
        if name == "handover_released":
            rb.arms.disturbance[rb.cfg.task.can_hand] = np.zeros(5)

    rb.log.event = _event
    for hand in rb.hands.values():
        orig = hand.set

        def _set(q, _orig=orig):
            _orig(q)
            scene._update()

        hand.set = _set
    return scene
