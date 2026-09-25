"""Whole-body door moves shared by opening and closing: stand in front of the free edge (outside the
swing wedge), put the hand on/behind the panel edge, then walk while the compliant arm holds contact."""
from __future__ import annotations

import logging
import math
import time

import numpy as np

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
    rb.loco.drive_to(float(q[0]), float(q[1]), yaw, 0.03, 0.04, float(r.k_pos), float(r.k_yaw), timeout=40.0,
                     abort=rb.abort_requested)


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
    try:
        while True:
            rb.check_abort()
            pose = rb.loco.pose()
            moved = math.hypot(pose.x - start.x, pose.y - start.y)
            hand_w = to_world(pose, rb.kin.fk(side, arm.measured(side))[:3, 3])
            theta = door.theta_from_point(hand_w, on_handle=on_handle)
            err = arm.tracking_error(side)
            if err > float(f.pull_max_tracking_err):
                log.warning("walk_with_door: tracking error %.2f rad, stopping", err)
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
    """Open wider: hook the free edge from the inner side, walk backward until target_deg."""
    f = rb.cfg.fridge
    hand = rb.hands[side]
    face_edge(rb, door, side, float(f.edge_standoff_m), r_frac=1.0)
    pose = rb.loco.pose()
    th = door.theta
    outer = door.normal_pelvis(pose, th)                 # toward the robot
    x_dir = -outer
    inward = door.edge_pelvis(pose, th, door.width - 0.04) - door.edge_pelvis(pose, th, door.width)
    inward /= np.linalg.norm(inward) + 1e-9              # along the panel toward the hinge
    hook_p = door.edge_pelvis(pose, th, door.width - 0.03) - 0.05 * outer   # 5 cm behind the panel, 3 cm in
    palm = inward                                         # palm toward the hinge; fingers curl around the edge
    pre = hook_p + 0.12 * outer
    hand.open()
    wait_user(rb, f"{side} hand behind the free edge {np.round(hook_p, 3)}")
    cart_move(rb, side, pre, x_dir, palm, tol=0.04)
    cart_move(rb, side, hook_p, x_dir, palm, tol=0.04, speed=0.06)
    hand.hook()
    wait_user(rb, "walk backward holding the edge")
    th = walk_with_door(rb, door, side, -1.0, target_deg, float(f.pull_back_m), float(f.pull_back_speed))
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
