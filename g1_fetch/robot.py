"""Robot bundle: everything the skills need, built for the real robot or for a dry run."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Cfg, resolve_path
from .frames import Frames, Intrinsics

log = logging.getLogger(__name__)


class Aborted(RuntimeError):
    pass


class RunLog:
    """events.jsonl + periodic/keyframe images under <log_dir>/<timestamp>/."""

    def __init__(self, root: Path, save_every: int = 5, enabled: bool = True):
        self.dir = root / time.strftime("%Y%m%d-%H%M%S")
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._events = open(self.dir / "events.jsonl", "a")
        self.save_every = save_every
        self._n = 0
        self._lock = threading.Lock()

    def event(self, name: str, **data):
        rec = {"t": time.time(), "event": name, **_jsonable(data)}
        log.info("EVENT %s %s", name, json.dumps(_jsonable(data))[:300])
        if self.enabled:
            with self._lock:
                self._events.write(json.dumps(rec) + "\n")
                self._events.flush()

    def frame(self, frame, dets, force: bool = False):
        self._n += 1
        if not self.enabled or (not force and self._n % self.save_every):
            return
        try:
            import cv2

            img = frame.color.copy()
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d.box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            name = f"{self._n:06d}"
            cv2.imwrite(str(self.dir / f"{name}.jpg"), img)
            if force:
                np.save(self.dir / f"{name}_depth.npy", frame.depth)
        except Exception as e:  # logging must never break the task
            log.debug("frame log failed: %s", e)

    def close(self):
        if self.enabled:
            self._events.close()


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out


@dataclass
class Robot:
    cfg: Cfg
    frames: Frames
    camera: object
    detector: object
    verifier: object
    loco: object
    arms: object
    kin: object
    hands: dict
    log: RunLog
    dry_run: bool = False
    _abort: threading.Event = field(default_factory=threading.Event)

    def abort_requested(self) -> bool:
        return self._abort.is_set()

    def request_abort(self):
        self._abort.set()

    def check_abort(self):
        if self._abort.is_set():
            raise Aborted("abort requested")

    def safe_stop(self, release_arms: bool = False):
        """Stop walking; keep the arms where they are (holding the can if we have it)."""
        try:
            self.loco.stop()
        except Exception as e:
            log.warning("stop failed: %s", e)
        if release_arms:
            try:
                self.arms.disable()
            except Exception as e:
                log.warning("arm release failed: %s", e)

    def close(self):
        for obj in (self.camera, self.arms, self.loco, *self.hands.values(), self.log):
            try:
                obj.close()
            except Exception as e:
                log.debug("close failed: %s", e)


def build_robot(cfg: Cfg, dry_run: bool = False, camera_source: str | None = None, detector=None,
                log_enabled: bool = True) -> Robot:
    """Real robot (DDS + RealSense) or a fully mocked one for dry runs and tests."""
    import pinocchio  # noqa: F401  (before torch: libstdc++ ordering on the workstation)

    from .control.arm import ArmKinematics, MockArmStreamer
    from .control.hand import MockHand
    from .control.loco import MockLoco
    from .perception.camera import FolderCamera, SyntheticCamera
    from .perception.detector import StubDetector, build_detector
    from .perception.verifier import build_verifier

    frames = Frames(cfg)
    kin = ArmKinematics(cfg)
    runlog = RunLog(resolve_path(cfg, cfg.task.log_dir), enabled=log_enabled)
    if dry_run:
        camera = FolderCamera(camera_source) if camera_source else SyntheticCamera(
            Intrinsics.from_fov(int(cfg.camera.width), int(cfg.camera.height)))
        det = detector if detector is not None else (build_detector(cfg) if camera_source else StubDetector())
        rb = Robot(cfg, frames, camera, det, build_verifier(cfg), MockLoco(cfg), MockArmStreamer(cfg), kin,
                   {"left": MockHand(cfg, "left"), "right": MockHand(cfg, "right")}, runlog, True)
        rb.arms.start()
        return rb

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from .control.arm import UnitreeArmStreamer
    from .control.hand import UnitreeBraincoHand
    from .control.loco import UnitreeLoco
    from .perception.camera import make_camera

    ChannelFactoryInitialize(0, str(cfg.network_interface))
    if bool(cfg.task.get("disable_arm_service", True)):
        from .control.robot_state import set_arm_action_service

        try:
            set_arm_action_service(False)
        except Exception as e:  # the service may not exist on every firmware
            log.warning("could not switch off the arm action service: %s", e)
    camera = make_camera(cfg, camera_source)
    det = detector if detector is not None else build_detector(cfg)
    loco = UnitreeLoco(cfg)
    arms = UnitreeArmStreamer(cfg)
    hands = {"left": UnitreeBraincoHand(cfg, "left"), "right": UnitreeBraincoHand(cfg, "right")}
    rb = Robot(cfg, frames, camera, det, build_verifier(cfg), loco, arms, kin, hands, runlog, False)
    loco.start()
    arms.start()
    return rb
