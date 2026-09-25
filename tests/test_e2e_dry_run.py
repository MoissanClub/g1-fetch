"""Whole task on the mock robot inside the ray-cast scene. Exercises every phase and all sign conventions."""
import logging
import math

import numpy as np
import pytest

from g1_fetch.robot import build_robot
from g1_fetch.task import FetchDrinkTask
from tests.sim_scene import attach


@pytest.mark.timeout(240)
def test_full_task_dry_run(cfg, caplog):
    caplog.set_level(logging.INFO)
    rb = build_robot(cfg, dry_run=True, log_enabled=False)
    # fridge 1.8 m ahead but 100 degrees to the right of the initial heading; person 1.5 m behind the start
    scene = attach(rb, fridge_x=1.8, fridge_yc=0.0)
    from g1_fetch.frames import Pose2D

    rb.loco._pose = Pose2D(0.0, 0.0, math.radians(100))
    try:
        task = FetchDrinkTask(rb)
        ok = task.run()
    finally:
        rb.close()
    print("scene log:", scene.log)
    print("history:", task.state.history)
    assert ok, task.state.history
    assert "hooked" in scene.log and "can grasped" in scene.log and "can released" in scene.log
    assert scene.door.theta < 0.2, "door should be closed at the end"
    # the robot ended near the person
    p = rb.loco.pose()
    assert math.hypot(p.x - scene.person_xy[0], p.y - scene.person_xy[1]) < 1.3


def test_search_and_approach_only(cfg):
    rb = build_robot(cfg, dry_run=True, log_enabled=False)
    scene = attach(rb, fridge_x=1.5, fridge_yc=0.3)
    try:
        task = FetchDrinkTask(rb, stop_after="approach")
        ok = task.run()
        appr = task.state.appr
    finally:
        rb.close()
    assert ok, task.state.history
    assert abs(appr.plane.distance_x - cfg.loco.approach.standoff_m) < 0.05
    assert abs(appr.plane.yaw_error) < 0.08
    # the handle prior is on the right of the pelvis at the configured lateral offset
    assert abs(appr.handle_prior[1] + cfg.loco.approach.lateral_offset_m) < 0.06
    assert np.linalg.norm(appr.handle_prior[:2] - scene.door.handle_pelvis(rb.loco.pose())[:2]) < 0.08
