import pinocchio  # noqa: F401  -- must precede torch/ultralytics imports (libstdc++ ordering on the workstation)
import pytest

from g1_fetch.config import load_config

FAST = {
    "loco": {"max_vx": 0.6, "max_vy": 0.4, "max_yaw": 1.2, "rate_hz": 20,
             "search": {"settle_s": 0.05, "yaw_rate": 1.0, "step_rad": 0.6},
             "approach": {"k_x": 1.0, "k_y": 1.2, "k_yaw": 2.0, "timeout_s": 40},
             "return": {"k_pos": 1.2, "k_yaw": 2.5, "timeout_s": 40}},
    "arm": {"cart_speed": 1.0, "cart_step_s": 0.005, "weight_ramp_s": 0.1, "max_joint_vel": 20.0, "rate_hz": 200},
    "hand": {"close_time_s": 0.02},
    "fridge": {"pull_back_speed": 0.4},
    "task": {"handover_timeout_s": 3.0, "max_retries": 0},
    "detector": {"backend": "stub", "stable_frames": 2},
}


@pytest.fixture
def cfg():
    return load_config(overrides=FAST)


@pytest.fixture
def cfg_default():
    return load_config()
