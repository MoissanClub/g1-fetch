"""Whole-body door moves shared by opening and closing: stand in front of the free edge (outside the
swing wedge), put the hand on/behind the panel edge, then walk while the compliant arm holds contact."""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from ..frames import wrap_angle
from .common import SkillFailed, cart_move, go_home, to_world, wait_user
from .door import DoorModel

log = logging.getLogger(__name__)


def face_edge(rb, door: DoorModel, side: str, standoff: float, lateral: float | None = None, r_frac: float = 0.85):
    """Drive to `standoff` in front of the panel point at r_frac·width, squared to the panel's current angle."""
    r = rb.cfg.loco.return_
    th = door.theta
    outer = door.door_normal_w(th)
    p = door.edge_w(th, r_frac * door.width)
    yaw = math.atan2(-outer[1], -outer[0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    if lateral is None:
        lateral = -0.10 if side == "right" else 0.10        # edge in front of the working shoulder
    q = p + standoff * outer + lateral * left
    rb.log.event("face_edge", x=float(q[0]), y=float(q[1]), yaw=yaw, theta_deg=math.degrees(th))
    try:
        rb.loco.drive_to(float(q[0]), float(q[1]), yaw, 0.06, 0.08, float(r.k_pos), float(r.k_yaw), timeout=25.0,
                         abort=rb.abort_requested)
    except Exception as e:  # LocoError: accept a near miss, the arm moves are re-planned from the actual pose
        pose = rb.loco.pose()
        miss = math.hypot(pose.x - q[0], pose.y - q[1])
        rb.log.event("face_edge_incomplete", miss=miss, reason=str(e))
        if miss > 0.2:
            raise SkillFailed(f"could not reach the door-edge pose ({e}); {miss:.2f} m away")
        rb.loco.turn_by(wrap_angle(yaw - pose.yaw), timeout=10.0, abort=rb.abort_requested)


def walk_with_door(rb, door: DoorModel, side: str, direction: float, stop_deg: float, max_dist: float,
                   speed: float, on_handle: bool = False) -> float:
    """Walk forward (+1) or backward (-1) with the arm compliant; the door angle is read from the hand.

    Stops when the angle passes `stop_deg` (in the direction of travel), after `max_dist`, on a tracking-error
    spike (hand slipped or blocked), or on timeout. Returns the final angle.
    """
    f = rb.cfg.fridge
    arm = rb.arms
    arm.set_gain(side, float(rb.cfg.arm.kp_compliant))
    start = rb.loco.pose()
    period = 1.0 / float(rb.cfg.loco.rate_hz)
    t0 = time.time()
    theta = door.theta
    stop = math.radians(stop_deg)
    moved = 0.0
    err0 = arm.tracking_error(side)
    try:
        while True:
            rb.check_abort()
            pose = rb.loco.pose()
            moved = math.hypot(pose.x - start.x, pose.y - start.y)
            hand_w = to_world(pose, rb.kin.fk(side, arm.measured(side))[:3, 3])
            theta = door.theta_from_point(hand_w, on_handle=on_handle)
            err = arm.tracking_error(side)
            if err - err0 > float(f.pull_max_tracking_err):
                log.warning("walk_with_door: tracking error rose by %.2f rad, stopping", err - err0)
                break
            if (direction < 0 and theta >= stop) or (direction > 0 and theta <= stop):
                break
            if moved >= max_dist or time.time() - t0 > 20.0:
                break
            rb.loco.set_velocity(direction * speed, 0.0, 0.0)
            time.sleep(period)
    finally:
        rb.loco.stop()
        arm.set_gain(side, float(rb.cfg.arm.kp))
    door.theta = theta
    rb.log.event("walk_with_door_done", direction=direction, theta_deg=math.degrees(theta), moved=moved)
    return theta


def hook_edge_and_pull(rb, door: DoorModel, side: str, target_deg: float) -> float:
    """Open wider: re-hook the handle at its current (swung) position from a stance that faces the panel, and walk
    backward again. The handle is on the free edge, so this is the same hook as the first pull, just rotated."""
    f = rb.cfg.fridge
    hand = rb.hands[side]
    # stand in front of the handle along the panel's current normal (outside the swing wedge)
    r = rb.cfg.loco.return_
    th = door.theta
    outer = door.door_normal_w(th)
    hw = door.handle_w(th)
    yaw = math.atan2(-outer[1], -outer[0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    lateral = -float(rb.cfg.hand.get("hook_side_offset_m", 0.03)) if side == "right" else float(rb.cfg.hand.get("hook_side_offset_m", 0.03))
    q = hw + float(rb.cfg.loco.approach.standoff_m) * outer + lateral * left
    rb.log.event("face_handle", x=float(q[0]), y=float(q[1]), yaw=yaw, theta_deg=math.degrees(th))
    try:
        rb.loco.drive_to(float(q[0]), float(q[1]), yaw, 0.06, 0.08, float(r.k_pos), float(r.k_yaw), timeout=25.0, abort=rb.abort_requested)
    except Exception as e:
        pose = rb.loco.pose()
        if math.hypot(pose.x - q[0], pose.y - q[1]) > 0.2:
            raise SkillFailed(f"could not reach the second-pull pose ({e})")
        rb.loco.turn_by(wrap_angle(yaw - pose.yaw), timeout=10.0, abort=rb.abort_requested)
    pose = rb.loco.pose()
    n = door.normal_pelvis(pose, th)
    x_dir = -n
    palm = np.array([0.0, 1.0, 0.0]) if side == "right" else np.array([0.0, -1.0, 0.0])
    handle_p = door.handle_pelvis(pose, th)
    reach = float(rb.cfg.hand.get("finger_reach_m", 0.08))
    gap = float(f.handle_protrusion_m)
    scoop = float(rb.cfg.hand.get("scoop_m", 0.05))
    entry = handle_p + n * (reach - min(0.9 * gap, 0.035)) - palm * (float(rb.cfg.hand.get("hook_side_offset_m", 0.03)) + scoop)
    pre = entry + n * float(f.pregrasp_offset_m)
    target = float(np.mean(np.array(rb.cfg.hand.hook, dtype=float)[2:]))
    margin = float(rb.cfg.hand.get("grasp_margin", 0.08))
    hand.open()
    wait_user(rb, f"{side} hand to the swung handle {np.round(entry, 3)}")
    cart_move(rb, side, pre, x_dir, palm, tol=0.04)
    cart_move(rb, side, entry, x_dir, palm, tol=0.04, speed=0.06)
    rb.arms.set_gain(side, float(rb.cfg.arm.kp_compliant))
    try:
        cart_move(rb, side, entry + palm * (scoop + 0.01), x_dir, palm, speed=0.04, tol=0.05, settle=False)
        time.sleep(0.4)
    finally:
        rb.arms.set_gain(side, float(rb.cfg.arm.kp))
    hand.hook()
    closed = hand.closed_fraction()
    if closed is not None and closed > target - margin:
        hand.open()
        go_home(rb, side)
        raise SkillFailed(f"second pull: hook closed to {closed:.2f}, nothing between the fingers")
    rb.log.event("handle_grasped", closed=closed, commanded=target, second_pull=True)
    wait_user(rb, "walk backward holding the handle again")
    th = walk_with_door(rb, door, side, -1.0, target_deg, float(f.pull_back_m), float(f.pull_back_speed), on_handle=True)
    hand.open()
    p = rb.kin.fk(side, rb.arms.cmd[side])[:3, 3]
    pose = rb.loco.pose()
    try:
        cart_move(rb, side, p + 0.10 * door.normal_pelvis(pose, th), x_dir, palm, tol=0.05)
    except SkillFailed as e:
        log.warning("retract after edge pull: %s", e)
    go_home(rb, side)
    return th


def push_panel_and_walk(rb, door: DoorModel, side: str, target_deg: float, max_dist: float = 0.35) -> float:
    """Close: palm on the outer face near the free edge, walk forward until target_deg."""
    f = rb.cfg.fridge
    hand = rb.hands[side]
    hand.open()
    face_edge(rb, door, side, float(f.edge_standoff_m), r_frac=0.85)
    pose = rb.loco.pose()
    th = door.theta
    outer = door.normal_pelvis(pose, th)
    x_dir = -outer
    contact = door.edge_pelvis(pose, th, 0.85 * door.width) + 0.03 * outer
    palm = x_dir                                          # palm flat on the panel
    pre = contact + 0.10 * outer
    wait_user(rb, f"{side} palm on the door {np.round(contact, 3)}")
    cart_move(rb, side, pre, x_dir, palm, tol=0.04)
    cart_move(rb, side, contact, x_dir, palm, tol=0.04, speed=0.06)
    wait_user(rb, "walk forward pushing the door")
    th = walk_with_door(rb, door, side, +1.0, target_deg, max_dist, float(f.pull_back_speed))
    p = rb.kin.fk(side, rb.arms.cmd[side])[:3, 3]
    try:
        cart_move(rb, side, p - 0.08 * x_dir, x_dir, palm, tol=0.05)
    except SkillFailed as e:
        log.warning("retract after push: %s", e)
    go_home(rb, side)
    return th
