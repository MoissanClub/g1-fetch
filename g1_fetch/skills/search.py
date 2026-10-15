"""Phase 1: rotate in place continuously, detecting every frame, until the fridge is seen; then confirm."""
from __future__ import annotations

import logging
import math
import time

from ..frames import wrap_angle
from ..perception.detector import Stabilizer
from ..perception.localize import door_plane
from .common import SkillFailed, best, has_head, look_for, observe, to_world

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
    cam = "head" if has_head(rb) else "depth"      # the level head camera sees the whole fridge and a standing person
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
                obs = observe(rb, cam)
                det = stab.update([d for d in obs.dets if d.conf >= conf])
                if det is not None:
                    rb.loco.stop()
                    time.sleep(float(s.settle_s))
                    confirmed, frame = look_for(rb, label, n_frames=int(s.get("confirm_frames", 2)),
                                                max_frames=8, min_conf=conf, cam=cam)
                    if confirmed is not None and cam == "head" and label == "fridge":
                        # A box filling the head frame can be any white wall up close: the D435i must agree
                        # (fridge detection, or a vertical plane within 1.5 m). Far candidates (small box) pass.
                        frac = confirmed.area / float(frame.intrinsics.width * frame.intrinsics.height)
                        if frac > float(s.get("near_box_frac", 0.5)):
                            obs_d = observe(rb)
                            d_fridge = best([d for d in obs_d.dets if d.conf >= float(rb.cfg.loco.approach.get("min_conf", 0.25))], "fridge")
                            h, w = obs_d.frame.depth.shape
                            plane = door_plane(obs_d.frame, (0, 0, w - 1, h - 1), rb.frames)
                            agrees = d_fridge is not None or (plane is not None and plane.distance_x < 1.5 and plane.width > 0.4)
                            rb.log.event("search_crosscheck", head_conf=confirmed.conf, box_frac=frac,
                                         depth_fridge=None if d_fridge is None else d_fridge.conf,
                                         plane=None if plane is None else [plane.distance_x, plane.width], accepted=agrees)
                            if not agrees:
                                confirmed = None
                    if confirmed is not None:
                        rb.log.event(f"{label}_found", conf=confirmed.conf, box=confirmed.box, turned=turned, cam=cam)
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
