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
    color: np.ndarray                 # HxWx3 uint8 BGR
    depth: np.ndarray | None          # HxW float32 metres, 0 = invalid, aligned to color; None for RGB-only cameras
    intrinsics: Intrinsics            # of the color stream
    t: float                          # capture time (time.time())
    seq: int = 0
    source: str = "depth"             # "depth" (RealSense) | "head" (forward RGB camera)

    @property
    def has_depth(self) -> bool:
        return self.depth is not None


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


class V4L2Camera:
    """Any UVC/V4L2 colour camera through OpenCV (YUYV or MJPG). A grab thread keeps only the newest frame so a
    slow consumer never reads a stale, buffered image. Intrinsics come from the configured horizontal FOV unless
    `fx/fy/cx/cy` are given (calibrated)."""

    def __init__(self, cam_cfg, source: str = "head"):
        import threading

        import cv2

        self.cfg = cam_cfg
        self.source = source
        self._cv2 = cv2
        dev = str(cam_cfg.device)
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {dev}")
        time.sleep(float(cam_cfg.get("settle_s", 1.0)))   # the SR300 re-enumerates if streamed too soon after open
        w, h, fps = int(cam_cfg.width), int(cam_cfg.height), int(cam_cfg.fps)
        fourcc = str(cam_cfg.get("fourcc", "YUYV"))
        # OpenCV's V4L2 backend reports the negotiated size only once the stream runs, so verify with a real frame.
        # Do NOT set CAP_PROP_BUFFERSIZE (it re-initialised this uvcvideo stream at 640x480); the grab thread
        # below drains the driver queue continuously and keeps only the newest frame.
        shape = None
        for attempt in range(2):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            for _ in range(10):
                ok, img = self.cap.read()
                if ok and img is not None:
                    shape = img.shape[:2]
                    break
            if shape == (h, w):
                break
            time.sleep(0.5)
        if shape != (h, w):
            self.cap.release()
            raise RuntimeError(f"camera {dev} streams {shape}, wanted {(h, w)} ({fourcc})")
        if cam_cfg.get("fx"):
            self.intrinsics = Intrinsics(float(cam_cfg.fx), float(cam_cfg.fy), float(cam_cfg.cx), float(cam_cfg.cy), w, h)
        else:
            self.intrinsics = Intrinsics.from_hfov(w, h, float(cam_cfg.hfov_deg))
        self.flip = bool(cam_cfg.get("flip", False))
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{source}")
        self._thread.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 3.0:
            time.sleep(0.02)
        if self._frame is None:
            raise RuntimeError(f"camera {dev}: no frames within 3 s")

    def _loop(self):
        while not self._stop:
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.flip:
                img = self._cv2.rotate(img, self._cv2.ROTATE_180)
            with self._lock:
                self._frame = img
                self._seq += 1

    def read(self, timeout_ms: int = 1000) -> Frame:
        with self._lock:
            img, seq = self._frame, self._seq
        return Frame(img, None, self.intrinsics, time.time(), seq, self.source)

    def close(self):
        self._stop = True
        try:
            self.cap.release()
        except Exception:
            pass


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
        dpath = self.folder / f"{name}_depth.npy"
        depth = np.load(dpath).astype(np.float32) if dpath.exists() else None
        self._i += 1
        return Frame(color, depth, self.intrinsics, time.time(), self._i, "depth" if depth is not None else "head")

    def close(self):
        pass


class SyntheticCamera:
    """Programmable frames for tests: set .next(color, depth) or a callable producing frames."""

    def __init__(self, intrinsics: Intrinsics, producer=None, source: str = "depth"):
        self.intrinsics = intrinsics
        self.producer = producer
        self.source = source
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
        return Frame(color, depth, self.intrinsics, time.time(), self._seq, self.source)

    def close(self):
        pass


def save_frame(frame: Frame, folder: str | Path, name: str):
    import cv2

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{name}_color.png"), frame.color)
    if frame.depth is not None:
        np.save(folder / f"{name}_depth.npy", frame.depth)
    meta = folder / "intrinsics.json"
    if not meta.exists():
        meta.write_text(json.dumps(vars(frame.intrinsics)))


def make_camera(cfg, source: str | None = None):
    """source: None -> RealSense; a folder path -> replay."""
    if source:
        return FolderCamera(source)
    return RealSenseCamera(cfg)


def make_head_camera(cfg, attempts: int = 3):
    """Forward RGB camera from cfg.cameras.head; None when disabled or when it cannot be opened.

    UVC cameras on this robot occasionally re-enumerate or stay busy for a second after a previous process
    released them, so the open is retried; the pipeline then runs RealSense-only rather than not at all.
    """
    import logging

    head = cfg.get("cameras")
    head = None if head is None else head.get("head")
    if head is None or not bool(head.get("enabled", False)):
        return None
    last = None
    for i in range(attempts):
        try:
            return V4L2Camera(head, "head")
        except Exception as e:  # noqa: BLE001
            last = e
            logging.getLogger(__name__).warning("head camera open attempt %d/%d failed: %s", i + 1, attempts, str(e)[:160])
            time.sleep(2.0)
    logging.getLogger(__name__).error("head camera unavailable (%s); continuing with the RealSense only", str(last)[:160])
    return None
