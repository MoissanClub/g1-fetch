"""Shared skill helpers: perception polling, world<->pelvis conversion, Cartesian arm moves."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from ..frames import Pose2D
from ..perception.camera import Frame
from ..perception.detector import Detection, Stabilizer

log = logging.getLogger(__name__)


class SkillFailed(RuntimeError):
    pass


@dataclass
class Observation:
    frame: Frame
    dets: list[Detection]


def observe(rb) -> Observation:
    frame = rb.camera.read()
    dets = rb.detector.detect(frame.color)
    rb.log.frame(frame, dets)
    return Observation(frame, dets)


def look_for(rb, label: str, n_frames: int | None = None, max_frames: int = 12, min_conf: float = 0.0):
    """Poll until `label` is seen in n consecutive frames. Returns (det, frame) or (None, last_frame)."""
    n = n_frames or int(rb.cfg.detector.stable_frames)
    stab = Stabilizer(label, n)
    last = None
    for _ in range(max_frames):
        rb.check_abort()
        obs = observe(rb)
        last = obs.frame
        det = stab.update([d for d in obs.dets if d.conf >= min_conf])
        if det is not None:
            return det, obs.frame
    return None, last


def best(dets: list[Detection], label: str) -> Detection | None:
    c = [d for d in dets if d.label == label]
    return max(c, key=lambda d: d.conf) if c else None


# ---- world (odometry) <-> pelvis ----
def to_world(pose: Pose2D, p_pelvis) -> np.ndarray:
    p = np.asarray(p_pelvis, dtype=float)
    wx, wy = pose.body_to_world(float(p[0]), float(p[1]))
    return np.array([wx, wy, float(p[2])])


def to_pelvis(pose: Pose2D, p_world) -> np.ndarray:
    p = np.asarray(p_world, dtype=float)
    bx, by = pose.world_to_body(float(p[0]), float(p[1]))
    return np.array([bx, by, float(p[2])])


def rot_world(pose: Pose2D, v_pelvis) -> np.ndarray:
    v = np.asarray(v_pelvis, dtype=float)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def rot_pelvis(pose: Pose2D, v_world) -> np.ndarray:
    v = np.asarray(v_world, dtype=float)
    c, s = math.cos(-pose.yaw), math.sin(-pose.yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


# ---- arm moves ----
def solve_pose(rb, side: str, p_pelvis, x_dir, palm_dir=None, q_seed=None, dir_weight: float | None = None):
    """IK for a hand pose; returns (q5, IKResult)."""
    seed = rb.arms.cmd[side] if q_seed is None else q_seed
    if dir_weight is None:
        dir_weight = float(rb.cfg.arm.ik_dir_weight)
    other = "right" if side == "left" else "left"
    res = rb.kin.ik(side, p_pelvis, x_dir, q_seed=seed, q_other=rb.arms.cmd[other], dir_weight=dir_weight)
    q = res.q.copy()
    if palm_dir is not None:
        roll_in = float(getattr(rb.cfg.robot.palm_inward_wrist_roll, side))
        q[4] = rb.kin.wrist_roll_for_palm(side, q, palm_dir, roll_in)
    else:
        q[4] = seed[4]
    return q, res


def cart_move(rb, side: str, p_goal, x_dir, palm_dir=None, speed: float | None = None,
              tol: float | None = None, dir_weight: float | None = None, settle: bool = True) -> float:
    """Straight-line Cartesian move of one hand to p_goal (pelvis frame), streaming IK solutions.

    Returns the final position error (m). Raises SkillFailed if the goal is unreachable beyond `tol`.
    """
    cfg = rb.cfg.arm
    speed = float(cfg.cart_speed if speed is None else speed)
    tol = float(cfg.ik_max_pos_err if tol is None else tol)
    p_goal = np.asarray(p_goal, dtype=float)
    p_start = rb.kin.fk(side, rb.arms.cmd[side])[:3, 3]
    dist = float(np.linalg.norm(p_goal - p_start))
    steps = max(1, int(math.ceil(dist / (speed * float(cfg.cart_step_s)))))
    q_prev = rb.arms.cmd[side].copy()
    # check the goal first so we do not start a move we cannot finish
    q_goal, res_goal = solve_pose(rb, side, p_goal, x_dir, palm_dir, q_seed=q_prev, dir_weight=dir_weight)
    if res_goal.pos_err > tol:
        raise SkillFailed(f"{side} hand cannot reach {np.round(p_goal, 3)} (err {res_goal.pos_err:.3f} m)")
    for i in range(1, steps + 1):
        rb.check_abort()
        a = i / steps
        p = (1 - a) * p_start + a * p_goal
        q, res = solve_pose(rb, side, p, x_dir, palm_dir, q_seed=q_prev, dir_weight=dir_weight)
        if res.pos_err > 2 * tol:
            log.warning("cart_move: intermediate IK error %.3f m at step %d/%d", res.pos_err, i, steps)
        rb.arms.set_target(side, q)
        q_prev = q
        time.sleep(float(cfg.cart_step_s))
    if settle:
        rb.arms.wait_settled(side, tol=0.08, timeout=2.0)
    return float(res_goal.pos_err)


def ensure_reachable(rb, side: str, p_goal, x_dir, max_nudge: float = 0.12) -> np.ndarray:
    """If p_goal (pelvis frame) is beyond reach, walk forward by the shortfall (once) and return the
    goal re-expressed in the new pelvis frame. Raises SkillFailed if it is still unreachable."""
    p_goal = np.asarray(p_goal, dtype=float)
    res = rb.kin.ik(side, p_goal, x_dir, q_seed=rb.arms.cmd[side], dir_weight=float(rb.cfg.arm.ik_dir_weight))
    if res.ok:
        return p_goal
    reached = rb.kin.fk(side, res.q)[:3, 3]
    shortfall = float(p_goal[0] - reached[0]) + 0.02
    if shortfall <= 0 or shortfall > max_nudge:
        raise SkillFailed(f"{side} hand cannot reach {np.round(p_goal, 3)} (err {res.pos_err:.3f} m, shortfall {shortfall:.3f})")
    pose = rb.loco.pose()
    goal_w = to_world(pose, p_goal)
    ahead = to_world(pose, [shortfall, 0.0, 0.0])
    r = rb.cfg.loco.return_
    rb.log.event("nudge_forward", m=shortfall)
    rb.loco.drive_to(ahead[0], ahead[1], None, 0.015, 0.05, float(r.k_pos), float(r.k_yaw), timeout=15.0, abort=rb.abort_requested)
    p_new = to_pelvis(rb.loco.pose(), goal_w)
    res = rb.kin.ik(side, p_new, x_dir, q_seed=rb.arms.cmd[side], dir_weight=float(rb.cfg.arm.ik_dir_weight))
    if not res.ok:
        raise SkillFailed(f"{side} hand still cannot reach {np.round(p_new, 3)} after nudging (err {res.pos_err:.3f} m)")
    return p_new


def hand_position(rb, side: str) -> np.ndarray:
    """Commanded hand position (pelvis frame)."""
    return rb.kin.fk(side, rb.arms.cmd[side])[:3, 3]


def go_home(rb, side: str, duration: float = 2.0):
    q = np.array(getattr(rb.cfg.arm.home_q, side), dtype=float)
    rb.arms.move_to(side, q, duration, abort=rb.abort_requested)


def wait_user(rb, msg: str):
    """In step mode, wait for Enter; otherwise just log."""
    log.info("STEP: %s", msg)
    if rb.cfg.task.step_mode:
        input(f"[step] {msg}  -- press Enter to continue (Ctrl-C to abort) ")
