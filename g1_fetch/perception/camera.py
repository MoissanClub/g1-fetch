"""RGB-D frame sources: RealSense on PC2, folder replay on the workstation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..frames import Intrinsics


@dataclass
class Frame:
    color: np.ndarray          # HxWx3 uint8 BGR
    depth: np.ndarray          # HxW float32 metres, 0 = invalid, aligned to color
    intrinsics: Intrinsics     # of the color stream
    t: float                   # capture time (time.time())
    seq: int = 0


class RealSenseCamera:
    """D435i colour + depth aligned to colour through pyrealsense2 (imported lazily)."""

    def __init__(self, cfg):
        import pyrealsense2 as rs  # noqa: F401  (PC2 only)

        self._rs = rs
        self.cfg = cfg
        self._seq = 0
        self.pipeline = rs.pipeline()
        conf = rs.config()
        w, h, fps = int(cfg.camera.width), int(cfg.camera.height), int(cfg.camera.fps)
        conf.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        conf.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        profile = self.pipeline.start(conf)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height)
        self.depth_max = float(cfg.camera.depth_max)
        self.depth_min = float(cfg.camera.depth_min)
        for _ in range(10):  # auto-exposure warm-up
            self.pipeline.wait_for_frames()

    def read(self, timeout_ms: int = 1000) -> Frame:
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms))
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0
        self._seq += 1
        return Frame(color.copy(), depth, self.intrinsics, time.time(), self._seq)

    def close(self):
        self.pipeline.stop()


class FolderCamera:
    """Replays frames saved by save_frame(): NAME_color.png, NAME_depth.npy, intrinsics.json."""

    def __init__(self, folder: str | Path, loop: bool = True):
        import cv2

        self.folder = Path(folder)
        self.loop = loop
        meta = json.loads((self.folder / "intrinsics.json").read_text())
        self.intrinsics = Intrinsics(**meta)
        self.names = sorted(p.name[: -len("_color.png")] for p in self.folder.glob("*_color.png"))
        if not self.names:
            raise FileNotFoundError(f"no *_color.png in {self.folder}")
        self._cv2 = cv2
        self._i = 0

    def read(self, timeout_ms: int = 0) -> Frame:
        if self._i >= len(self.names):
            if not self.loop:
                raise StopIteration
            self._i = 0
        name = self.names[self._i]
        color = self._cv2.imread(str(self.folder / f"{name}_color.png"), self._cv2.IMREAD_COLOR)
        depth = np.load(self.folder / f"{name}_depth.npy").astype(np.float32)
        self._i += 1
        return Frame(color, depth, self.intrinsics, time.time(), self._i)

    def close(self):
        pass


class SyntheticCamera:
    """Programmable frames for tests: set .next(color, depth) or a callable producing frames."""

    def __init__(self, intrinsics: Intrinsics, producer=None):
        self.intrinsics = intrinsics
        self.producer = producer
        self._seq = 0
        blank = np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8)
        self._frame = (blank, np.zeros((intrinsics.height, intrinsics.width), np.float32))

    def set(self, color: np.ndarray, depth: np.ndarray):
        self._frame = (color, depth)

    def read(self, timeout_ms: int = 0) -> Frame:
        self._seq += 1
        if self.producer is not None:
            color, depth = self.producer(self._seq)
        else:
            color, depth = self._frame
        return Frame(color, depth, self.intrinsics, time.time(), self._seq)

    def close(self):
        pass


def save_frame(frame: Frame, folder: str | Path, name: str):
    import cv2

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{name}_color.png"), frame.color)
    np.save(folder / f"{name}_depth.npy", frame.depth)
    meta = folder / "intrinsics.json"
    if not meta.exists():
        meta.write_text(json.dumps(vars(frame.intrinsics)))


def make_camera(cfg, source: str | None = None):
    """source: None -> RealSense; a folder path -> replay."""
    if source:
        return FolderCamera(source)
    return RealSenseCamera(cfg)
