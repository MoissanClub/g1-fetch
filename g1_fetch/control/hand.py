"""BrainCo Revo2 hands through the brainco_hand_service DDS bridge (six-finger MotorCmds_)."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

N_FINGERS = 6  # thumb, thumb-aux, index, middle, ring, pinky; 0 = open, 1 = closed


class HandBase:
    def __init__(self, cfg, side: str):
        self.cfg = cfg.hand
        self.side = side
        self.target = np.array(self.cfg.open, dtype=float)

    def set(self, q):
        q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        if q.shape != (N_FINGERS,):
            raise ValueError("hand target must have 6 entries")
        self.target = q
        self._publish()

    def open(self, wait: bool = True):
        self.set(self.cfg.open)
        if wait:
            time.sleep(float(self.cfg.close_time_s))

    def grasp(self, wait: bool = True):
        self.set(self.cfg.grasp)
        if wait:
            time.sleep(float(self.cfg.close_time_s))

    def hook(self, wait: bool = True):
        self.set(self.cfg.hook)
        if wait:
            time.sleep(float(self.cfg.close_time_s))

    def state(self) -> np.ndarray | None:
        return None

    def closed_fraction(self) -> float | None:
        """Mean finger closure (index..pinky) from feedback; None if no feedback."""
        s = self.state()
        return None if s is None else float(np.mean(s[2:]))

    def _publish(self):
        raise NotImplementedError

    def close(self):
        pass


class UnitreeBraincoHand(HandBase):
    def __init__(self, cfg, side: str):
        super().__init__(cfg, side)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        self.pub = ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
        self.pub.Init()
        self.sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
        self.sub.Init(self._on_state, 5)
        self.msg = MotorCmds_()
        self.msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(N_FINGERS)]
        for c in self.msg.cmds:
            c.q = 0.0
            c.dq = 1.0
        self._state = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"hand-{side}")
        self._thread.start()

    def _on_state(self, msg):
        with self._lock:
            self._state = np.array([msg.states[i].q for i in range(N_FINGERS)], dtype=float)

    def _publish(self):
        for i, c in enumerate(self.msg.cmds):
            c.q = float(self.target[i])
        self.pub.Write(self.msg)

    def _loop(self):
        period = 1.0 / float(self.cfg.rate_hz)
        while not self._stop:
            self._publish()
            time.sleep(period)

    def state(self):
        with self._lock:
            return None if self._state is None else self._state.copy()

    def close(self):
        self._stop = True


class MockHand(HandBase):
    def __init__(self, cfg, side: str):
        super().__init__(cfg, side)
        self.history: list[np.ndarray] = []

    def _publish(self):
        self.history.append(self.target.copy())

    def state(self):
        return self.target.copy()
