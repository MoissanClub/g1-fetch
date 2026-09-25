"""Phase 1: rotate in place (stop-and-look) until the fridge is detected."""
from __future__ import annotations

import logging
import math
import time

from .common import SkillFailed, look_for, to_world

log = logging.getLogger(__name__)


def search(rb, label: str = "fridge", direction: float = 1.0):
    """Returns (detection, frame). Stop-and-look sweeps avoid motion blur and command latency."""
    s = rb.cfg.loco.search
    explore_rounds = 2 if float(s.explore_step_m) > 0 else 1
    for round_ in range(explore_rounds):
        turned = 0.0
        full = 2 * math.pi * float(s.max_turns)
        while turned <= full + 1e-6:
            rb.check_abort()
            time.sleep(float(s.settle_s))
            det, frame = look_for(rb, label, max_frames=int(rb.cfg.detector.stable_frames) + 4)
            if det is not None:
                rb.log.event(f"{label}_found", conf=det.conf, box=det.box, turned=turned)
                return det, frame
            step = float(s.step_rad) * direction
            got = rb.loco.turn_by(step, abort=rb.abort_requested)
            turned += abs(got)
            log.info("search %s: turned %.2f rad total", label, turned)
        if round_ + 1 < explore_rounds:
            pose = rb.loco.pose()
            ahead = to_world(pose, [float(s.explore_step_m), 0.0, 0.0])
            rb.log.event("search_explore", to=[float(ahead[0]), float(ahead[1])])
            r = rb.cfg.loco.return_
            rb.loco.drive_to(ahead[0], ahead[1], None, float(r.tol_pos), float(r.tol_yaw), float(r.k_pos),
                             float(r.k_yaw), timeout=20.0, abort=rb.abort_requested)
    raise SkillFailed(f"{label} not found after {s.max_turns} rotation(s)")
