"""Phase 3: locate the handle, hook-grasp it, pull the door open while stepping back, then sweep it wider."""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from ..perception.localize import box_to_point, cavity_depth_jump, median_depth
from .common import SkillFailed, best, cart_move, ensure_reachable, go_home, hand_position, observe, to_world, wait_user
from .door import DoorModel, door_from_plane
from .door_moves import hook_edge_and_pull

log = logging.getLogger(__name__)


def locate_handle(rb, appr) -> tuple[np.ndarray, str]:
    """Handle grasp point in the pelvis frame: detector if it sees the handle, else plane + priors."""
    f = rb.cfg.fridge
    obs = observe(rb)
    det = best(obs.dets, "handle")
    if det is not None:
        p = box_to_point(obs.frame, det, rb.frames, shrink=0.5)
        if p is not None:
            off_plane = abs(float(appr.plane.n @ p - appr.plane.d))
            h = rb.frames.floor_height(p)
            lo, hi = f.handle_height_band
            if off_plane < 0.12 and lo - 0.1 <= h <= hi + 0.1:
                p = p.copy()
                p[2] = rb.frames.pelvis_z_from_floor(float(np.clip(h, lo, hi)))
                rb.log.event("handle_detected", p=p.tolist(), conf=det.conf)
                return p, "detector"
            log.info("handle detection rejected (off-plane %.2f m, height %.2f m)", off_plane, h)
    rb.log.event("handle_prior", p=appr.handle_prior.tolist())
    return appr.handle_prior, "prior"


def _palm_inward(side: str) -> np.ndarray:
    return np.array([0.0, 1.0, 0.0]) if side == "right" else np.array([0.0, -1.0, 0.0])


def grasp_handle(rb, door: DoorModel, side: str) -> None:
    f = rb.cfg.fridge
    hand = rb.hands[side]
    pose = rb.loco.pose()
    n = door.normal_pelvis(pose)                 # toward the robot
    x_dir = -n                                   # forearm into the door
    palm = _palm_inward(side)
    lateral = -palm * 0.015                      # bar sits against the palm, not the fingertips
    handle_p = door.handle_pelvis(pose)
    grasp = ensure_reachable(rb, side, handle_p + n * 0.005 + lateral, x_dir)
    pose = rb.loco.pose()
    n = door.normal_pelvis(pose)
    x_dir = -n
    pre = grasp + n * float(f.pregrasp_offset_m)
    hand.open()
    wait_user(rb, f"{side} hand to pre-grasp {np.round(pre, 3)}")
    cart_move(rb, side, pre, x_dir, palm)
    wait_user(rb, f"{side} hand to handle {np.round(grasp, 3)}")
    cart_move(rb, side, grasp, x_dir, palm, speed=0.06)
    hand.hook()
    closed = hand.closed_fraction()
    if closed is not None and closed > 0.97:
        raise SkillFailed(f"hook closed fully ({closed:.2f}); the handle was missed")
    rb.log.event("handle_grasped", closed=closed)


def pull_and_back(rb, door: DoorModel, side: str) -> float:
    """Hold the handle compliantly and walk backward; the door follows. Returns the opening angle."""
    f = rb.cfg.fridge
    arm = rb.arms
    arm.set_gain(side, float(rb.cfg.arm.kp_compliant))
    start = rb.loco.pose()
    back = float(f.pull_back_m)
    period = 1.0 / float(rb.cfg.loco.rate_hz)
    t0 = time.time()
    theta = 0.0
    try:
        while True:
            rb.check_abort()
            pose = rb.loco.pose()
            moved = math.hypot(pose.x - start.x, pose.y - start.y)
            hand_w = to_world(pose, rb.kin.fk(side, arm.measured(side))[:3, 3])
            theta = door.theta_from_point(hand_w)
            err = arm.tracking_error(side)
            if err > float(f.pull_max_tracking_err):
                log.warning("pull: arm tracking error %.2f rad, stopping", err)
                break
            if moved >= back or theta >= math.radians(float(f.pull_angle_deg)):
                break
            if time.time() - t0 > 15.0:
                raise SkillFailed("pull_and_back timeout")
            # step back; a small drift toward the hinge side keeps the hand near the arc
            vy = 0.03 if door.hinge_side == "left" else -0.03
            rb.loco.set_velocity(-float(f.pull_back_speed), vy, 0.0)
            time.sleep(period)
    finally:
        rb.loco.stop()
        arm.set_gain(side, float(rb.cfg.arm.kp))
    door.theta = theta
    rb.log.event("pull_done", theta_deg=math.degrees(theta), moved=moved)
    return theta


def release_and_retract(rb, door: DoorModel, side: str):
    hand = rb.hands[side]
    hand.open()
    pose = rb.loco.pose()
    p = hand_position(rb, side)
    n = door.normal_pelvis(pose)                 # outer normal at the current angle
    away = p + 0.08 * n
    try:
        cart_move(rb, side, away, x_dir=-n, palm_dir=_palm_inward(side), tol=0.05)
    except SkillFailed as e:
        log.warning("retract: %s", e)
    go_home(rb, side)


def verify_open(rb, appr, frame_before) -> bool:
    obs = observe(rb)
    jump = cavity_depth_jump(frame_before, obs.frame, appr.fridge_box)
    ok = jump is not None and jump > float(rb.cfg.fridge.cavity_depth_jump_m)
    v = rb.verifier.ask(obs.frame.color, "Is the refrigerator door open?")
    rb.log.event("verify_open", depth_jump=jump, vlm=v)
    if v is not None:
        return ok or v
    return ok


def open_door(rb, appr) -> DoorModel:
    side = rb.cfg.task.door_hand
    pose0 = rb.loco.pose()
    handle_p, src = locate_handle(rb, appr)
    width = appr.plane.width if appr.plane.width < 1.5 * float(rb.cfg.fridge.door_width_m) else None
    door = door_from_plane(appr.plane, pose0, handle_p, rb.cfg, width)
    rb.log.event("door_model", hinge=door.hinge_w.tolist(), normal=door.normal_w.tolist(), width=door.width, src=src)
    grasp_handle(rb, door, side)
    wait_user(rb, "pull the door while stepping back")
    pull_and_back(rb, door, side)
    release_and_retract(rb, door, side)
    if not verify_open(rb, appr, appr.frame):
        raise SkillFailed("door does not look open after the pull")
    f = rb.cfg.fridge
    if bool(f.stage_b_sweep) and door.theta < math.radians(float(f.open_angle_deg)) - 0.1:
        wait_user(rb, "hook the free edge and open wider")
        hook_edge_and_pull(rb, door, side, float(f.open_angle_deg))
    if door.theta < math.radians(float(f.min_open_deg)):
        raise SkillFailed(f"door only open {math.degrees(door.theta):.0f} deg (< {f.min_open_deg}); not safe to step in")
    return door
