"""Phase 2: servo toward the fridge, square up to the door plane, put the handle beside the door hand."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from ..perception.camera import Frame
from ..perception.localize import Plane, box_to_point, door_plane, handle_from_prior
from .common import SkillFailed, best, observe

log = logging.getLogger(__name__)


@dataclass
class ApproachResult:
    plane: Plane
    frame: Frame
    fridge_box: tuple
    handle_prior: np.ndarray      # pelvis frame, from plane + priors


def _full_box(frame: Frame) -> tuple:
    h, w = frame.depth.shape
    return (w * 0.1, h * 0.2, w * 0.9, h * 0.95)


def approach(rb) -> ApproachResult:
    a = rb.cfg.loco.approach
    period = 1.0 / float(rb.cfg.loco.rate_hz)
    lateral_target = -float(a.lateral_offset_m) if rb.cfg.task.door_hand == "right" else float(a.lateral_offset_m)
    standoff = float(a.standoff_m)
    t0 = time.time()
    lost = 0
    last_plane: Plane | None = None
    last_box = None
    try:
        while True:
            rb.check_abort()
            if time.time() - t0 > float(a.timeout_s):
                raise SkillFailed("approach timeout")
            obs = observe(rb)
            frame = obs.frame
            fridge = best(obs.dets, "fridge")
            near = last_plane is not None and last_plane.distance_x < float(a.slow_zone_m)
            if fridge is None:
                lost += 1
                if near:
                    box = _full_box(frame)          # the door fills the view when close
                elif lost > 10:
                    raise SkillFailed("lost the fridge during approach")
                else:
                    rb.loco.stop()
                    time.sleep(period)
                    continue
            else:
                lost = 0
                box = fridge.box
            last_box = box
            plane = door_plane(frame, box, rb.frames)
            intr = frame.intrinsics
            if plane is not None:
                last_plane = plane
                dist = plane.distance_x
                yaw_err = plane.yaw_error
            else:
                p = box_to_point(frame, fridge, rb.frames) if fridge is not None else None
                if p is None:
                    rb.loco.stop()
                    time.sleep(period)
                    continue
                dist = float(p[0])
                cx = fridge.center[0]
                yaw_err = -math.atan((cx - intr.cx) / intr.fx)
            lat_err = 0.0
            handle_prior = None
            if plane is not None and dist < float(a.slow_zone_m):
                handle_prior = handle_from_prior(plane, rb.cfg, rb.frames)
                lat_err = float(handle_prior[1] - lateral_target)
            vx = float(a.k_x) * (dist - standoff)
            if dist < float(a.slow_zone_m):
                vx = max(-0.1, min(0.15, vx))
            vy = float(a.k_y) * lat_err
            yaw = float(a.k_yaw) * yaw_err
            done = abs(dist - standoff) < float(a.tol_dist) and abs(yaw_err) < float(a.tol_yaw) \
                and abs(lat_err) < float(a.tol_lat) and plane is not None
            # keep a crawl speed on any axis still outside tolerance so the deadband cannot stall us
            v_min = float(rb.cfg.loco.deadband_v) + 0.02
            w_min = float(rb.cfg.loco.deadband_w) + 0.02
            if abs(dist - standoff) >= float(a.tol_dist) and abs(vx) < v_min:
                vx = math.copysign(v_min, dist - standoff)
            if abs(lat_err) >= float(a.tol_lat) and abs(vy) < v_min:
                vy = math.copysign(v_min, lat_err)
            if abs(yaw_err) >= float(a.tol_yaw) and abs(yaw) < w_min:
                yaw = math.copysign(w_min, yaw_err)
            log.debug("approach dist=%.2f yaw_err=%.2f lat_err=%.2f -> v=(%.2f,%.2f,%.2f)", dist, yaw_err, lat_err, vx, vy, yaw)
            if done:
                rb.loco.stop()
                time.sleep(0.5)
                obs = observe(rb)
                plane2 = door_plane(obs.frame, last_box, rb.frames) or plane
                handle_prior = handle_from_prior(plane2, rb.cfg, rb.frames)
                rb.log.event("approach_done", dist=plane2.distance_x, yaw_err=plane2.yaw_error,
                             plane_width=plane2.width, handle_prior=handle_prior.tolist())
                return ApproachResult(plane2, obs.frame, tuple(last_box), handle_prior)
            rb.loco.set_velocity(vx, vy, yaw)
            time.sleep(period)
    finally:
        rb.loco.stop()
