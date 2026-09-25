import numpy as np
import pytest

from g1_fetch.control.arm import ArmKinematics, MockArmStreamer
from g1_fetch.frames import Frames


@pytest.fixture(scope="module")
def kin():
    from g1_fetch.config import load_config

    return ArmKinematics(load_config())


def test_fk_matches_urdf_geometry(kin):
    T = kin.fk("right", np.zeros(5))
    assert np.allclose(T[:3, 3], [0.116 + kin.ee_offset, -0.149, 0.095], atol=0.005)
    assert np.allclose(T[:3, 0], [1, 0, 0], atol=1e-3)


def test_ik_roundtrip_on_reachable_targets(kin):
    rng = np.random.default_rng(3)
    lo, hi = kin.lower[kin.qidx["left"]], kin.upper[kin.qidx["left"]]
    worst = 0.0
    for _ in range(25):
        q = lo + rng.random(5) * (hi - lo)
        q[4] = 0.0
        T = kin.fk("left", q)
        r = kin.ik("left", T[:3, 3], T[:3, 0], q_seed=np.zeros(5))
        worst = max(worst, r.pos_err)
    assert worst < 0.01, worst


def test_ik_reports_unreachable(kin):
    r = kin.ik("right", [0.9, -0.2, 0.1])
    assert not r.ok and r.pos_err > 0.3


def test_handle_and_can_targets_reachable(kin, cfg_default):
    """The default standoff geometry must be inside the reach envelope."""
    fr = Frames(cfg_default)
    f = cfg_default.fridge
    a = cfg_default.loco.approach
    z = fr.pelvis_z_from_floor(f.handle_grasp_height_m)
    handle = [a.standoff_m - f.handle_protrusion_m, -a.lateral_offset_m, z]
    pre = [handle[0] - f.pregrasp_offset_m, handle[1] - 0.015, z]
    for p in (handle, pre):
        r = kin.ik("right", p, [1, 0, 0], dir_weight=cfg_default.arm.ik_dir_weight)
        assert r.ok, (p, r.pos_err)
    can = [f.reach_standoff_m + f.can_depth_in_cavity_m, 0.0, fr.pelvis_z_from_floor(f.can_shelf_height_m)]
    hand = [can[0], can[1] + f.can_diameter_m / 2 + 0.01, can[2]]
    r = kin.ik("left", hand, [1, 0, 0], dir_weight=cfg_default.arm.ik_dir_weight)
    assert r.ok, (hand, r.pos_err)


def test_wrist_roll_for_palm(kin):
    q = np.zeros(5)
    assert abs(kin.wrist_roll_for_palm("right", q, [0, 1, 0])) < 0.05       # inward already
    assert abs(abs(kin.wrist_roll_for_palm("right", q, [0, 0, 1])) - np.pi / 2) < 0.06   # palm up
    assert abs(kin.wrist_roll_for_palm("left", q, [0, -1, 0])) < 0.05


def test_streamer_clips_joint_velocity(cfg):
    s = MockArmStreamer(cfg)
    s.start()
    s.enable(0.05)
    s.set_target("left", np.array([1.0, 0, 0, 0, 0]))
    import time

    time.sleep(0.05)
    step = cfg.arm.max_joint_vel / cfg.arm.rate_hz
    assert s.cmd["left"][0] <= 1.0 + 1e-9 and s.cmd["left"][0] > 0
    assert s.tracking_error("left") < 1.5
    s.close()
    assert step > 0
