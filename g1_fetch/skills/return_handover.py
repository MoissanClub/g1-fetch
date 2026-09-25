"""Phase 6: return to the start pose by odometry, find the person, hand over the can."""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from ..perception.localize import box_to_point
from .common import SkillFailed, best, cart_move, go_home, observe, wait_user
from .search import search

log = logging.getLogger(__name__)


def return_home(rb, home) -> None:
    r = rb.cfg.loco.return_
    rb.log.event("return_home", to=[home.x, home.y, home.yaw])
    rb.loco.drive_to(home.x, home.y, home.yaw, float(r.tol_pos), float(r.tol_yaw), float(r.k_pos), float(r.k_yaw),
                     timeout=float(r.timeout_s), abort=rb.abort_requested)


def find_person(rb) -> None:
    """Rotate until a person is seen, then close in to person_standoff_m."""
    det, frame = search(rb, "person")
    standoff = float(rb.cfg.loco.person_standoff_m)
    period = 1.0 / float(rb.cfg.loco.rate_hz)
    a = rb.cfg.loco.approach
    t0 = time.time()
    lost = 0
    try:
        while True:
            rb.check_abort()
            if time.time() - t0 > 30.0:
                raise SkillFailed("person approach timeout")
            obs = observe(rb)
            person = best(obs.dets, "person")
            if person is None:
                lost += 1
                if lost > 10:
                    raise SkillFailed("lost the person")
                rb.loco.stop()
                time.sleep(period)
                continue
            lost = 0
            intr = obs.frame.intrinsics
            yaw_err = -math.atan((person.center[0] - intr.cx) / intr.fx)
            p = box_to_point(obs.frame, person, rb.frames, shrink=0.6)
            dist = float(p[0]) if p is not None else standoff + 0.5
            if abs(dist - standoff) < 0.1 and abs(yaw_err) < 0.08:
                rb.loco.stop()
                rb.log.event("person_reached", dist=dist)
                return
            vx = max(-0.1, min(0.2, float(a.k_x) * (dist - standoff)))
            rb.loco.set_velocity(vx, 0.0, float(a.k_yaw) * yaw_err)
            time.sleep(period)
    finally:
        rb.loco.stop()


def handover(rb) -> None:
    t = rb.cfg.task
    side = rb.cfg.task.can_hand
    hand = rb.hands[side]
    z = rb.frames.pelvis_z_from_floor(float(rb.cfg.fridge.handover_height_m))
    y = 0.12 if side == "left" else -0.12
    target = np.array([0.32, y, z])
    palm = np.array([0.0, -1.0, 0.0]) if side == "left" else np.array([0.0, 1.0, 0.0])
    wait_user(rb, "extend the can toward the person")
    cart_move(rb, side, target, x_dir=[1.0, 0.0, 0.0], palm_dir=palm, tol=0.04)
    q_hold = rb.arms.cmd[side].copy()
    rb.log.event("handover_extended")
    t0 = time.time()
    above_since = None
    released = False
    while time.time() - t0 < float(t.handover_timeout_s):
        rb.check_abort()
        err = float(np.max(np.abs(rb.arms.measured(side) - q_hold)))
        pulled = err > float(t.pull_trigger_err_rad)
        if pulled:
            above_since = above_since or time.time()
            if time.time() - above_since >= float(t.pull_trigger_hold_s):
                v = rb.verifier.ask(rb.camera.read().color, "Is a person's hand holding the can?")
                if v is not False:
                    released = True
                    break
                above_since = None
        else:
            above_since = None
        time.sleep(0.05)
    if released:
        hand.open()
        rb.log.event("handover_released")
        time.sleep(1.0)
        go_home(rb, side)
    else:
        rb.log.event("handover_timeout")
        raise SkillFailed("nobody took the can; still holding it")
