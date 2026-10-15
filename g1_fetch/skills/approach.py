"""Phase 2: servo toward the fridge, square up to the door plane, put the handle beside the door hand."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from ..perception.camera import Frame
from ..perception.localize import Plane, bar_from_depth, box_to_point, door_plane, handle_from_prior
from .common import SkillFailed, bearing_of, best, has_head, observe

log = logging.getLogger(__name__)


@dataclass
class ApproachResult:
    plane: Plane
    frame: Frame
    fridge_box: tuple
    handle_prior: np.ndarray      # pelvis frame, from plane + priors
    head_frame: Frame | None = None   # one head-camera frame taken at the stance (seam height, debugging)


def _full_box(frame: Frame) -> tuple:
    h, w = frame.depth.shape
    return (w * 0.1, h * 0.2, w * 0.9, h * 0.95)


def handle_from_head(rb, head_obs, plane: Plane, fallback: np.ndarray):
    """Handle position (pelvis frame) from the level head camera, or None.

    Primary: the fridge box's free edge (left edge for a right hinge) projected onto the depth-fitted door plane,
    inset by `fridge.handle_inset_m`. Refinement: a `handle` detection that is a tall thin box, lies inside the
    grasp height band and agrees with the edge estimate within 0.15 m. Seams and trim (short/wide boxes) and the
    freezer handle (above the band) are rejected; an edge touching the image border is not trusted.
    """
    f = rb.cfg.fridge
    frame = head_obs.frame
    intr = frame.intrinsics
    margin = 12
    lo, hi = f.handle_height_band
    z = rb.frames.pelvis_z_from_floor(float(np.clip(float(f.handle_grasp_height_m), lo, hi)))
    edge_est = None
    fridge = best(head_obs.dets, "fridge")
    if fridge is not None:
        x1, y1, x2, y2 = fridge.box
        if f.hinge_side == "right" and x1 > margin:
            u_edge = x1
        elif f.hinge_side == "left" and x2 < intr.width - margin:
            u_edge = x2
        else:
            u_edge = None
        if u_edge is not None:
            pe = rb.frames.head_ray_hit_plane_x(u_edge, (y1 + y2) / 2, intr, plane.distance_x)
            if np.isfinite(pe).all() and abs(float(pe[1])) < 1.5:
                y_h = float(pe[1]) - float(f.handle_inset_m) if f.hinge_side == "right" else float(pe[1]) + float(f.handle_inset_m)
                edge_est = np.array([plane.distance_x - float(f.handle_protrusion_m), y_h, z])
    result = (edge_est, "head-edge") if edge_est is not None else None
    ref_y = float(edge_est[1]) if edge_est is not None else float(fallback[1])
    hmc = float(rb.cfg.loco.approach.get("handle_min_conf", 0.2))
    cands = []
    for d in head_obs.dets:
        if d.label != "handle" or d.conf < hmc or d.height < 2.5 * d.width:
            continue
        u, v = d.center
        hp = rb.frames.head_ray_hit_plane_x(u, v, intr, plane.distance_x)
        if not np.isfinite(hp).all():
            continue
        h_floor = rb.frames.floor_height(hp)
        if not (lo - 0.15 <= h_floor <= hi + 0.15) or abs(float(hp[1]) - ref_y) > 0.15:
            continue
        cands.append((abs(float(hp[1]) - ref_y), hp))
    if cands:
        hp = min(cands, key=lambda t: t[0])[1]
        result = (np.array([plane.distance_x - float(f.handle_protrusion_m), float(hp[1]), z]), "head")
    return result


def approach(rb) -> ApproachResult:
    """Servo to the manipulation stance.

    Far (> slow zone): heading from the head camera's fridge bearing when a head camera exists (level view, whole
    fridge visible), otherwise from the RealSense box; forward speed from the RealSense door-plane distance when the
    plane is already visible, else a constant cruise. Near: square to the plane, put the handle at the configured
    lateral target (handle seen by the head camera first, then the RealSense, then the edge prior).
    """
    a = rb.cfg.loco.approach
    period = 1.0 / float(rb.cfg.loco.rate_hz)
    lateral_target = -float(a.lateral_offset_m) if rb.cfg.task.door_hand == "right" else float(a.lateral_offset_m)
    standoff = float(a.standoff_m)
    cruise = float(a.get("cruise_vx", 0.2))
    min_conf = float(a.get("min_conf", 0.25))
    t0 = time.time()
    lost = 0
    last_plane: Plane | None = None
    last_box = None
    use_head = has_head(rb)
    try:
        while True:
            rb.check_abort()
            if time.time() - t0 > float(a.timeout_s):
                raise SkillFailed("approach timeout")
            obs = observe(rb)                                   # RealSense: depth, plane, near-range detections
            frame = obs.frame
            fridge = best([d for d in obs.dets if d.conf >= min_conf], "fridge")
            near = last_plane is not None and last_plane.distance_x < float(a.slow_zone_m)
            # the head camera only steers from afar; inside the slow zone everything comes from the D435i
            head_obs = observe(rb, "head") if (use_head and not near) else None
            head_fridge = best(head_obs.dets, "fridge") if head_obs else None
            # ---- geometry from the RealSense ----
            if fridge is not None:
                box = fridge.box
            elif near or last_plane is None:
                box = _full_box(frame)                          # the door fills the view when close
            else:
                box = last_box
            plane = door_plane(frame, box, rb.frames) if box is not None else None
            if plane is not None and plane.distance_x > float(a.slow_zone_m) + 1.0 and fridge is None and not near:
                plane = None                                    # a wall far away is not the fridge
            if plane is not None:
                last_plane = plane
                last_box = box
            seen = fridge is not None or head_fridge is not None or plane is not None
            if not seen:
                lost += 1
                if lost > 10:
                    raise SkillFailed("lost the fridge during approach")
                rb.loco.stop()
                time.sleep(period)
                continue
            lost = 0
            # ---- distance ----
            if plane is not None:
                dist = plane.distance_x
            else:
                p = box_to_point(frame, fridge, rb.frames) if fridge is not None else None
                dist = float(p[0]) if p is not None else None
            # ---- heading ----
            if plane is not None and dist is not None and dist < float(a.slow_zone_m):
                yaw_err = plane.yaw_error                       # square to the door
            elif head_fridge is not None:
                yaw_err = bearing_of(rb, head_fridge, head_obs.frame)
            elif fridge is not None:
                yaw_err = bearing_of(rb, fridge, frame)
            else:
                yaw_err = plane.yaw_error if plane is not None else 0.0
            # ---- lateral (near only) ----
            lat_err = 0.0
            handle_prior = None
            handle_src = None
            if plane is not None and dist < float(a.slow_zone_m):
                handle_prior = handle_from_prior(plane, rb.cfg, rb.frames)
                handle_src = "prior"
                hmc = float(a.get("handle_min_conf", 0.2))
                hd = best([d for d in obs.dets if d.conf >= hmc], "handle")
                if hd is not None:
                    hp = box_to_point(frame, hd, rb.frames, shrink=0.5)
                    if hp is not None and abs(float(hp[0]) - dist) < 0.25:
                        handle_prior, handle_src = hp, "depth-det"
                if str(rb.cfg.fridge.get("grip", "bar")) == "bar" and dist < 0.9:
                    bar = bar_from_depth(frame, plane, rb.frames, rb.cfg)
                    if bar is not None:
                        handle_prior, handle_src = bar[:3], "depth-bar"
                lat_err = float(handle_prior[1] - lateral_target)
            # ---- commands ----
            if dist is None:
                vx = cruise
            else:
                vx = float(a.k_x) * (dist - standoff)
                if dist < float(a.slow_zone_m):
                    vx = max(-0.1, min(0.15, vx))
            vy = float(a.k_y) * lat_err
            yaw = float(a.k_yaw) * yaw_err
            done = dist is not None and abs(dist - standoff) < float(a.tol_dist) and abs(yaw_err) < float(a.tol_yaw) \
                and abs(lat_err) < float(a.tol_lat) and plane is not None
            v_min = float(rb.cfg.loco.deadband_v) + 0.02
            w_min = float(rb.cfg.loco.deadband_w) + 0.02
            if dist is not None and abs(dist - standoff) >= float(a.tol_dist) and abs(vx) < v_min:
                vx = math.copysign(v_min, dist - standoff)
            if abs(lat_err) >= float(a.tol_lat) and abs(vy) < v_min:
                vy = math.copysign(v_min, lat_err)
            if abs(yaw_err) >= float(a.tol_yaw) and abs(yaw) < w_min:
                yaw = math.copysign(w_min, yaw_err)
            log.debug("approach dist=%s yaw_err=%.2f lat_err=%.2f (%s) -> v=(%.2f,%.2f,%.2f)",
                      None if dist is None else f"{dist:.2f}", yaw_err, lat_err, handle_src, vx, vy, yaw)
            if done:
                rb.loco.stop()
                time.sleep(0.5)
                obs = observe(rb)
                plane2 = door_plane(obs.frame, last_box, rb.frames) or plane
                if handle_src == "prior":
                    handle_prior = handle_from_prior(plane2, rb.cfg, rb.frames)
                rb.log.event("approach_done", dist=plane2.distance_x, yaw_err=plane2.yaw_error, plane_width=plane2.width,
                             plane_y=[plane2.y_min, plane2.y_max], handle_prior=handle_prior.tolist(), handle_src=handle_src)
                head_frame = observe(rb, "head").frame if use_head else None
                if head_frame is not None:
                    rb.log.frame(head_frame, [], force=True)
                return ApproachResult(plane2, obs.frame, tuple(last_box), handle_prior, head_frame)
            rb.loco.set_velocity(vx, vy, yaw)
            time.sleep(period)
    finally:
        rb.loco.stop()
