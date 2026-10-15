"""Phase 4: step to the opening, find the can, grasp it with the free hand, tuck it against the chest."""
from __future__ import annotations

import logging

import numpy as np

from ..perception.localize import can_axis_from_box
from .common import SkillFailed, cart_move, ensure_reachable, look_for, to_pelvis, to_world, wait_user
from .door import DoorModel

log = logging.getLogger(__name__)


def go_to_reach_pose(rb, door: DoorModel):
    """Stand `reach_standoff_m` in front of the cavity centre, squared to the fridge."""
    f = rb.cfg.fridge
    r = rb.cfg.loco.return_
    # stand on the hinge-far side of the opening so the half-open panel is not in the body's way
    side_shift = float(f.get("reach_side_shift_m", 0.25))
    lateral = float(f.reach_lateral_m) + (side_shift if door.hinge_side == "right" else -side_shift)
    x, y, yaw = door.standoff_pose_w(float(f.reach_standoff_m), lateral=lateral)
    rb.log.event("reach_pose", x=x, y=y, yaw=yaw)
    rb.loco.drive_to(x, y, yaw, 0.03, 0.04, float(r.k_pos), float(r.k_yaw), timeout=30.0, abort=rb.abort_requested)


def locate_can(rb, door: DoorModel) -> tuple[np.ndarray, str]:
    """Can position in the pelvis frame: D435i detection, else walk in a step and look again, else a prior.

    Odometry drifts during the two pulls, so the open fridge may be further than the door model says; the prior
    is therefore placed relative to the robot (in front of the can hand) rather than at the modelled cavity.
    """
    f = rb.cfg.fridge
    r = rb.cfg.loco.return_
    for step in range(2):
        det, frame = look_for(rb, "can", max_frames=10)
        if det is not None:
            p = can_axis_from_box(frame, det, rb.frames, rb.cfg)
            if p is not None:
                rb.log.event("can_detected", p=p.tolist(), conf=det.conf, height=rb.frames.floor_height(p), step=step)
                return p, "detector"
        if step == 0:
            # nothing seen: the shelf may be above the D435i's view ceiling at this range; step in and look again
            pose = rb.loco.pose()
            ahead = to_world(pose, [float(f.get("can_search_step_m", 0.15)), 0.0, 0.0])
            rb.log.event("can_search_step", m=float(f.get("can_search_step_m", 0.15)))
            try:
                rb.loco.drive_to(ahead[0], ahead[1], None, 0.04, 0.05, float(r.k_pos), float(r.k_yaw), timeout=12.0, abort=rb.abort_requested)
            except Exception as e:  # LocoError
                log.warning("can search step failed: %s", e)
    side = rb.cfg.task.can_hand
    y = 0.10 if side == "left" else -0.10
    p = np.array([0.40, y, rb.frames.pelvis_z_from_floor(float(f.can_shelf_height_m))])
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
    hand_p = ensure_reachable(rb, side, hand_p, x_dir, max_nudge=float(f.get("can_max_nudge_m", 0.30)))
    pre = hand_p - x_dir * float(f.pregrasp_offset_m)
    hand.open()
    wait_user(rb, f"{side} hand to can pre-grasp {np.round(pre, 3)}")
    cart_move(rb, side, pre, x_dir, palm)
    wait_user(rb, f"{side} hand to can {np.round(hand_p, 3)}")
    cart_move(rb, side, hand_p, x_dir, palm, speed=0.06)
    hand.grasp()
    closed = hand.closed_fraction()
    target = float(np.mean(np.array(rb.cfg.hand.grasp, dtype=float)[2:]))
    if closed is not None and closed > target - float(rb.cfg.hand.get("grasp_margin", 0.08)):
        hand.open()
        raise SkillFailed(f"grasp closed to {closed:.2f} (commanded {target:.2f}); the can was missed")
    rb.log.event("can_grasped", closed=closed, commanded=target)
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
