"""Phase 4: step to the opening, find the can, grasp it with the free hand, tuck it against the chest."""
from __future__ import annotations

import logging

import numpy as np

from ..perception.localize import can_axis_from_box
from .common import SkillFailed, cart_move, ensure_reachable, look_for, to_pelvis, wait_user
from .door import DoorModel

log = logging.getLogger(__name__)


def go_to_reach_pose(rb, door: DoorModel):
    """Stand `reach_standoff_m` in front of the cavity centre, squared to the fridge."""
    f = rb.cfg.fridge
    r = rb.cfg.loco.return_
    x, y, yaw = door.standoff_pose_w(float(f.reach_standoff_m), lateral=float(f.reach_lateral_m))
    rb.log.event("reach_pose", x=x, y=y, yaw=yaw)
    rb.loco.drive_to(x, y, yaw, 0.03, 0.04, float(r.k_pos), float(r.k_yaw), timeout=30.0, abort=rb.abort_requested)


def locate_can(rb, door: DoorModel) -> tuple[np.ndarray, str]:
    det, frame = look_for(rb, "can", max_frames=10)
    if det is not None:
        p = can_axis_from_box(frame, det, rb.frames, rb.cfg)
        if p is not None:
            rb.log.event("can_detected", p=p.tolist(), conf=det.conf, height=rb.frames.floor_height(p))
            return p, "detector"
    f = rb.cfg.fridge
    pose = rb.loco.pose()
    c = to_pelvis(pose, door.cavity_center_w())
    p = np.array([c[0] + float(f.can_depth_in_cavity_m), c[1], rb.frames.pelvis_z_from_floor(float(f.can_shelf_height_m))])
    rb.log.event("can_prior", p=p.tolist())
    return p, "prior"


def grab_can(rb, p_can: np.ndarray) -> None:
    f = rb.cfg.fridge
    side = rb.cfg.task.can_hand
    hand = rb.hands[side]
    palm = np.array([0.0, -1.0, 0.0]) if side == "left" else np.array([0.0, 1.0, 0.0])   # palm toward the midline
    x_dir = np.array([1.0, 0.0, 0.0])
    radius = float(f.can_diameter_m) / 2
    hand_p = p_can - palm * (radius + 0.01)           # palm centre beside the can axis
    hand_p = ensure_reachable(rb, side, hand_p, x_dir)
    pre = hand_p - x_dir * float(f.pregrasp_offset_m)
    hand.open()
    wait_user(rb, f"{side} hand to can pre-grasp {np.round(pre, 3)}")
    cart_move(rb, side, pre, x_dir, palm)
    wait_user(rb, f"{side} hand to can {np.round(hand_p, 3)}")
    cart_move(rb, side, hand_p, x_dir, palm, speed=0.06)
    hand.grasp()
    closed = hand.closed_fraction()
    if closed is not None and closed > 0.97:
        hand.open()
        raise SkillFailed(f"grasp closed fully ({closed:.2f}); the can was missed")
    rb.log.event("can_grasped", closed=closed)
    lifted = hand_p + np.array([0.0, 0.0, 0.03])
    cart_move(rb, side, lifted, x_dir, palm, speed=0.05, tol=0.03)
    out = lifted - x_dir * float(f.retract_m)
    cart_move(rb, side, out, x_dir, palm, tol=0.04)
    carry = np.array(getattr(rb.cfg.arm.carry_q, side), dtype=float)
    rb.arms.move_to(side, carry, 2.0, abort=rb.abort_requested)
    rb.log.event("can_carried")


def fetch_can(rb, door: DoorModel):
    import math

    if door.theta < math.radians(float(rb.cfg.fridge.min_open_deg)):
        raise SkillFailed(f"door only open {math.degrees(door.theta):.0f} deg; not stepping into the swing")
    go_to_reach_pose(rb, door)
    p, src = locate_can(rb, door)
    grab_can(rb, p)
