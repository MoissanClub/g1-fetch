"""Robot-state RPC helpers: disable the built-in arm action service that competes for rt/arm_sdk."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

ARM_ACTION_SERVICE = "g1_arm_example"


def set_arm_action_service(enabled: bool, timeout_s: float = 3.0) -> int:
    """ServiceSwitch on the `robot_state` service (same API as the C++ RobotStateClient)."""
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

    rsc = RobotStateClient()
    rsc.SetTimeout(timeout_s)
    rsc.Init()
    code = rsc.ServiceSwitch(ARM_ACTION_SERVICE, bool(enabled))
    log.info("ServiceSwitch(%s, %s) -> %s", ARM_ACTION_SERVICE, enabled, code)
    return code


def service_list(timeout_s: float = 3.0):
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

    rsc = RobotStateClient()
    rsc.SetTimeout(timeout_s)
    rsc.Init()
    return rsc.ServiceList()
