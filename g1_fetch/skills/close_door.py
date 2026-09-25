"""Phase 5: close the door with the door hand while the other hand keeps holding the can:
stand in front of the free edge (outside the swing), palm on the outer face, walk forward until shut."""
from __future__ import annotations

import logging
import math

from ..perception.localize import door_plane
from .approach import _full_box
from .common import SkillFailed, observe, wait_user
from .door import DoorModel
from .door_moves import push_panel_and_walk

log = logging.getLogger(__name__)


def verify_closed(rb, door: DoorModel) -> bool:
    """Drive to the closed-door standoff, fit the plane, expect it at the standoff distance and square."""
    a = rb.cfg.loco.approach
    r = rb.cfg.loco.return_
    x, y, yaw = door.standoff_pose_w(float(a.standoff_m) + 0.15)
    rb.loco.drive_to(x, y, yaw, 0.05, 0.05, float(r.k_pos), float(r.k_yaw), timeout=30.0, abort=rb.abort_requested)
    obs = observe(rb)
    plane = door_plane(obs.frame, _full_box(obs.frame), rb.frames)
    pose = rb.loco.pose()
    c = door.cavity_center_w()
    expected = float((pose.x - c[0]) * door.normal_w[0] + (pose.y - c[1]) * door.normal_w[1])   # pelvis to closed plane
    ok = plane is not None and abs(plane.distance_x - expected) < 0.08 and abs(plane.yaw_error) < 0.25
    v = rb.verifier.ask(obs.frame.color, "Is the refrigerator door closed?")
    rb.log.event("verify_closed", plane_dist=None if plane is None else plane.distance_x, expected=expected, vlm=v)
    return ok if v is None else (ok or v)


def close_door(rb, door: DoorModel) -> None:
    side = rb.cfg.task.door_hand
    for attempt in range(2):
        pushes = 0
        while door.theta > math.radians(4.0) and pushes < 6:
            wait_user(rb, f"push the door from {math.degrees(door.theta):.0f} deg")
            target = max(2.0, math.degrees(door.theta) - 30.0)
            th = push_panel_and_walk(rb, door, side, target, max_dist=0.35)
            pushes += 1
            rb.log.event("close_push_done", attempt=attempt, push=pushes, theta_deg=math.degrees(th))
        if verify_closed(rb, door):
            door.theta = 0.0
            rb.log.event("door_closed")
            return
        log.warning("door may not be closed (theta %.0f deg); trying again", math.degrees(door.theta))
        door.theta = max(door.theta, math.radians(10.0))
    raise SkillFailed("door still open after two closing attempts")
