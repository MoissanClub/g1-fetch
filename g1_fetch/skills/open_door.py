"""Phase 3: locate the handle, hook-grasp it, pull the door open while stepping back, then sweep it wider."""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from ..perception.localize import bar_from_depth, box_to_point, cavity_depth_jump, median_depth
from .common import SkillFailed, best, cart_move, ensure_reachable, go_home, hand_position, observe, to_world, wait_user
from .door import DoorModel, door_from_plane
from .door_moves import hook_edge_and_pull

log = logging.getLogger(__name__)


def locate_handle(rb, appr) -> tuple[np.ndarray, str]:
    """Handle grasp point in the pelvis frame: detector if it sees the handle, else plane + priors."""
    f = rb.cfg.fridge
    obs = observe(rb)
    bars = []
    for _ in range(3):                                  # three frames: take the median bar estimate
        o = observe(rb)
        b = bar_from_depth(o.frame, appr.plane, rb.frames, rb.cfg)
        if b is not None:
            bars.append(b)
    if bars:
        b = np.median(np.array(bars), axis=0)
        p = np.array(b[:3])
        rb.log.event("handle_bar_depth", p=p.tolist(), protrusion=float(b[3]), points=int(b[4]), frames=len(bars))
        return p, "depth-bar"
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
    p = np.array(appr.handle_prior, dtype=float)
    lo, hi = f.handle_height_band
    p[2] = rb.frames.pelvis_z_from_floor(float(np.clip(rb.frames.floor_height(p), lo, hi)))
    rb.log.event("handle_prior", p=p.tolist())
    return p, "prior"


def seam_height_from_head(rb, head_frame, plane, y_grip: float) -> float | None:
    """Height (above the floor) of the dark seam between the doors, read from the head camera at the stance.

    Looks for the darkest horizontal band in a column strip around the grip point; the row is turned into a
    height by intersecting the camera ray with the D435i door plane. Returns None when nothing convincing is found.
    """
    import cv2

    f = rb.cfg.fridge
    intr = head_frame.intrinsics
    # pixel column of the grip point: project a plane point at the prior height
    z_prior = rb.frames.pelvis_z_from_floor(float(f.seam_height_m))
    p = plane.point_at(y_grip, z_prior)
    uv = intr.project(rb.frames.T_head_optical_pelvis[:3, :3] @ p + rb.frames.T_head_optical_pelvis[:3, 3])
    u = int(np.clip(uv[0], 40, intr.width - 41))
    gray = cv2.cvtColor(head_frame.color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    strip = gray[:, u - 40:u + 40].mean(axis=1)
    strip = cv2.GaussianBlur(strip.reshape(-1, 1), (1, 9), 0).ravel()
    lo, hi = f.seam_height_band
    best_v, best_val = None, None
    for v in range(10, intr.height - 10):
        h = rb.frames.floor_height(rb.frames.head_ray_hit_plane_x(u, v, intr, plane.distance_x))
        if not (lo <= h <= hi):
            continue
        val = strip[v] - 0.5 * (strip[v - 8] + strip[v + 8])     # darker than its surroundings
        if best_val is None or val < best_val:
            best_v, best_val = v, val
    if best_v is None or best_val > -12.0:                       # not clearly darker than the panel
        return None
    return float(rb.frames.floor_height(rb.frames.head_ray_hit_plane_x(u, best_v, intr, plane.distance_x)))


def locate_top_edge(rb, appr) -> tuple[np.ndarray, str]:
    """Grip point on the lower door's top lip near the free edge (pelvis frame)."""
    f = rb.cfg.fridge
    plane = appr.plane
    if f.hinge_side == "right":
        y_grip = plane.y_max - float(f.grip_from_edge_m)
    else:
        y_grip = plane.y_min + float(f.grip_from_edge_m)
    h = None
    if appr.head_frame is not None:
        h = seam_height_from_head(rb, appr.head_frame, plane, y_grip)
    src = "head-seam" if h is not None else "prior"
    if h is None:
        h = float(f.seam_height_m)
    lo, hi = f.seam_height_band
    h = float(np.clip(h, lo, hi))
    p = plane.point_at(y_grip, rb.frames.pelvis_z_from_floor(h))
    rb.log.event("top_edge_grip", p=p.tolist(), height=h, src=src, edge_y=float(plane.y_max if f.hinge_side == "right" else plane.y_min))
    return p, src


def grasp_top_edge(rb, grip: np.ndarray, side: str) -> np.ndarray:
    """Hook the fingers over the lower door's top lip: come in from the front above the seam, lower into it, close.

    Retries at +-3 cm height if the fingers close on nothing. Returns the grip point actually used.
    """
    f = rb.cfg.fridge
    hand = rb.hands[side]
    x_dir = np.array([0.85, 0.0, 0.5])                 # forearm forward and up
    palm = np.array([0.0, 0.0, -1.0])                  # palm down: fingers curl over the lip
    target = float(np.mean(np.array(rb.cfg.hand.hook, dtype=float)[2:]))
    margin = float(rb.cfg.hand.get("grasp_margin", 0.08))
    for dz in (0.0, 0.03, -0.03):
        p = grip + np.array([0.0, 0.0, dz])
        p = ensure_reachable(rb, side, p + np.array([-0.02, 0.0, 0.06]), x_dir)   # the highest point we need
        p = p - np.array([-0.02, 0.0, 0.06])
        above = p + np.array([-0.02, 0.0, 0.06])
        pre = above + np.array([-0.10, 0.0, 0.0])
        hand.open()
        wait_user(rb, f"{side} hand above the seam {np.round(above, 3)} (dz {dz:+.2f})")
        cart_move(rb, side, pre, x_dir, palm, tol=0.04)
        cart_move(rb, side, above, x_dir, palm, tol=0.04, speed=0.08)
        cart_move(rb, side, p + np.array([0.0, 0.0, 0.0]), x_dir, palm, tol=0.04, speed=0.05)   # down into the groove
        hand.hook()
        closed = hand.closed_fraction()
        if closed is None or closed <= target - margin:
            rb.log.event("handle_grasped", closed=closed, commanded=target, grip="top_edge", dz=dz)
            return p
        log.info("top-edge hook closed to %.2f at dz %+.2f: nothing hooked, retrying", closed, dz)
        hand.open()
        cart_move(rb, side, above, x_dir, palm, tol=0.05, speed=0.08)
    raise SkillFailed("top-edge hook found nothing at the seam (tried -3..+3 cm)")


def _palm_inward(side: str) -> np.ndarray:
    return np.array([0.0, 1.0, 0.0]) if side == "right" else np.array([0.0, -1.0, 0.0])


def grasp_handle(rb, door: DoorModel, side: str) -> None:
    f = rb.cfg.fridge
    hand = rb.hands[side]
    pose = rb.loco.pose()
    n = door.normal_pelvis(pose)                 # toward the robot
    x_dir = -n                                   # forearm into the door
    palm = _palm_inward(side)
    # Plate/bar handle standing `handle_protrusion_m` off the door: the open fingers (reaching `finger_reach_m`
    # beyond the palm centre) slide into the gap behind the plate from the hinge-far side, then the hand scoops
    # toward the plate before the fingers curl. All targets are derived from the door model in the CURRENT pelvis
    # frame; after a forward nudge they are recomputed.
    reach = float(rb.cfg.hand.get("finger_reach_m", 0.08))
    gap = float(f.handle_protrusion_m)
    deep = min(0.9 * gap, 0.035)                 # how far the fingertips go behind the plate face
    side_off = float(rb.cfg.hand.get("hook_side_offset_m", 0.03))
    scoop = float(rb.cfg.hand.get("scoop_m", 0.05))

    def targets():
        pose = rb.loco.pose()
        n = door.normal_pelvis(pose)
        hp = door.handle_pelvis(pose)
        entry = hp + n * (reach - deep) - palm * (side_off + scoop)
        return n, -n, entry

    n, x_dir, entry = targets()
    ensure_reachable(rb, side, entry + x_dir * 0.02, x_dir)      # deepest point we may need (dx retry)
    n, x_dir, entry = targets()                                   # refresh after a possible nudge
    pre = entry + n * float(f.pregrasp_offset_m)
    target = float(np.mean(np.array(rb.cfg.hand.hook, dtype=float)[2:]))
    margin = float(rb.cfg.hand.get("grasp_margin", 0.08))
    hand.open()
    wait_user(rb, f"{side} hand to pre-grasp {np.round(pre, 3)}")
    cart_move(rb, side, pre, x_dir, palm)
    # Scoop: enter the gap behind the plate well to the hinge-far side of it, then slide toward the plate until it
    # sits against the palm, and only then close. +-2 cm errors in the plate estimate no longer matter.
    for attempt, dx in enumerate((0.0, 0.02)):
        e = entry + x_dir * dx
        wait_user(rb, f"{side} hand into the gap beside the plate {np.round(e, 3)}")
        cart_move(rb, side, e, x_dir, palm, speed=0.06)
        rb.arms.set_gain(side, float(rb.cfg.arm.kp_compliant))
        try:
            cart_move(rb, side, e + palm * (scoop + 0.01), x_dir, palm, speed=0.04, tol=0.05, settle=False)   # slide onto the plate
            time.sleep(0.4)
        finally:
            rb.arms.set_gain(side, float(rb.cfg.arm.kp))
        hand.hook()
        closed = hand.closed_fraction()
        if closed is None or closed <= target - margin:
            rb.log.event("handle_grasped", closed=closed, commanded=target, attempt=attempt, dx=dx)
            return
        log.info("hook closed to %.2f (commanded %.2f) after the scoop (dx %+.3f): nothing between the fingers", closed, target, dx)
        hand.open()
        cart_move(rb, side, e - x_dir * 0.06, x_dir, palm, speed=0.08, tol=0.05)
    raise SkillFailed(f"hook found nothing at the handle after the scoop attempts (last closure {closed:.2f} of {target:.2f})")


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
    err0 = arm.tracking_error(side)              # the hand is already pressed against the handle: use the increase
    try:
        while True:
            rb.check_abort()
            pose = rb.loco.pose()
            moved = math.hypot(pose.x - start.x, pose.y - start.y)
            hand_w = to_world(pose, rb.kin.fk(side, arm.measured(side))[:3, 3])
            theta = door.theta_from_point(hand_w)
            err = arm.tracking_error(side)
            if err - err0 > float(f.pull_max_tracking_err):
                log.warning("pull: arm tracking error rose by %.2f rad (from %.2f), stopping", err - err0, err0)
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
    top_edge = str(rb.cfg.fridge.get("grip", "bar")) == "top_edge"
    handle_p, src = locate_top_edge(rb, appr) if top_edge else locate_handle(rb, appr)
    half_fov = appr.plane.distance_x * math.tan(math.radians(85.0) / 2) - 0.05   # depth stream ~87 deg wide
    bounded = (-half_fov < appr.plane.y_min) and (appr.plane.y_max < half_fov)
    width = appr.plane.width if bounded and 0.4 < appr.plane.width < 1.5 * float(rb.cfg.fridge.door_width_m) else None
    rb.log.event("door_width", measured=appr.plane.width, bounded=bounded, used=width or float(rb.cfg.fridge.door_width_m))
    door = door_from_plane(appr.plane, pose0, handle_p, rb.cfg, width)
    rb.log.event("door_model", hinge=door.hinge_w.tolist(), normal=door.normal_w.tolist(), width=door.width, src=src)
    if top_edge:
        grasp_top_edge(rb, handle_p, side)
    else:
        grasp_handle(rb, door, side)
    wait_user(rb, "pull the door while stepping back")
    pull_and_back(rb, door, side)
    release_and_retract(rb, door, side)
    if not verify_open(rb, appr, appr.frame):
        raise SkillFailed("door does not look open after the pull")
    f = rb.cfg.fridge
    if bool(f.stage_b_sweep) and door.theta < math.radians(float(f.open_angle_deg)) - 0.1:
        wait_user(rb, "re-hook the handle and open wider")
        try:
            hook_edge_and_pull(rb, door, side, float(f.open_angle_deg))
        except SkillFailed as e:
            # the door IS open (verified above); a failed second pull must not make the phase re-grasp a moved handle
            log.warning("second pull failed (%s); continuing with the door at %.0f deg", e, math.degrees(door.theta))
            rb.log.event("second_pull_failed", reason=str(e), theta_deg=math.degrees(door.theta))
            rb.hands[side].open()
            go_home(rb, side)
    if door.theta < math.radians(float(f.min_open_deg)):
        raise SkillFailed(f"door only open {math.degrees(door.theta):.0f} deg (< {f.min_open_deg}); not safe to step in")
    return door
