"""Phase 1: rotate in place continuously, detecting every frame, until the fridge is seen; then confirm."""
from __future__ import annotations

import logging
import math
import time

from ..frames import wrap_angle
from ..perception.detector import Stabilizer
from .common import SkillFailed, look_for, observe, to_world

log = logging.getLogger(__name__)


def search(rb, label: str = "fridge", direction: float = 1.0):
    """Returns (detection, frame).

    Continuous rotation at `search.yaw_rate` with the detector running on every frame (motion blur at
    0.4 rad/s is negligible for a 640 px frame). A detection that persists for `detector.stable_frames`
    consecutive frames at >= `search.conf` stops the rotation; a stationary confirmation look must then see
    it again, otherwise the rotation resumes. Rotation is accounted from odometry yaw.
    """
    s = rb.cfg.loco.search
    conf = float(s.get("conf", 0.15))
    period = 1.0 / float(rb.cfg.loco.rate_hz)
    full = 2 * math.pi * float(s.max_turns)
    explore_rounds = 2 if float(s.explore_step_m) > 0 else 1
    for round_ in range(explore_rounds):
        turned = 0.0
        last_yaw = rb.loco.pose().yaw
        stab = Stabilizer(label, int(rb.cfg.detector.stable_frames))
        t_last_log = time.time()
        try:
            while turned <= full + 1e-6:
                rb.check_abort()
                obs = observe(rb)
                det = stab.update([d for d in obs.dets if d.conf >= conf])
                if det is not None:
                    rb.loco.stop()
                    time.sleep(float(s.settle_s))
                    confirmed, frame = look_for(rb, label, n_frames=int(s.get("confirm_frames", 2)),
                                                max_frames=8, min_conf=conf)
                    if confirmed is not None:
                        rb.log.event(f"{label}_found", conf=confirmed.conf, box=confirmed.box, turned=turned)
                        return confirmed, frame
                    log.info("search %s: candidate not confirmed when stationary; resuming", label)
                    stab.reset()
                yaw = rb.loco.pose().yaw
                turned += abs(wrap_angle(yaw - last_yaw))
                last_yaw = yaw
                rb.loco.set_velocity(0.0, 0.0, direction * float(s.yaw_rate))
                if time.time() - t_last_log > 2.0:
                    log.info("search %s: turned %.2f rad total", label, turned)
                    t_last_log = time.time()
                time.sleep(period)
        finally:
            rb.loco.stop()
        if round_ + 1 < explore_rounds:
            pose = rb.loco.pose()
            ahead = to_world(pose, [float(s.explore_step_m), 0.0, 0.0])
            rb.log.event("search_explore", to=[float(ahead[0]), float(ahead[1])])
            r = rb.cfg.loco.return_
            rb.loco.drive_to(ahead[0], ahead[1], None, float(r.tol_pos), float(r.tol_yaw), float(r.k_pos),
                             float(r.k_yaw), timeout=20.0, abort=rb.abort_requested)
    raise SkillFailed(f"{label} not found after {s.max_turns} rotation(s)")
