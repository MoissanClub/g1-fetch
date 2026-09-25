"""Object detectors. Labels are canonical: fridge | handle | can | person."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from ..config import resolve_path

CANONICAL = ("fridge", "handle", "can", "person")


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    source: str = ""
    mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class Detector(Protocol):
    def detect(self, color: np.ndarray) -> list[Detection]: ...


class UltralyticsDetector:
    """YOLOE (text prompts) or COCO YOLO through the ultralytics package (imported lazily)."""

    COCO_MAP = {"refrigerator": "fridge", "person": "person"}

    def __init__(self, model_path: str, classes: list[str] | None, label_map: dict[str, str],
                 conf: float = 0.25, imgsz: int = 640, device="cpu", name: str = "yolo"):
        from ultralytics import YOLO, YOLOE

        self.name = name
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.label_map = dict(label_map)
        is_yoloe = "yoloe" in str(model_path).lower()
        if is_yoloe:
            self.model = YOLOE(model_path)
            have = list(getattr(self.model, "names", {}).values()) if isinstance(getattr(self.model, "names", None), dict) else []
            if classes and str(model_path).endswith(".pt") and have != list(classes):
                # needs the MobileCLIP text encoder (downloaded once); baked checkpoints skip this
                self.model.set_classes(classes, self.model.get_text_pe(classes))
        else:
            self.model = YOLO(model_path)
            self.label_map.update(self.COCO_MAP)
        self.last_ms = 0.0

    def detect(self, color: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        res = self.model.predict(color, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        self.last_ms = (time.perf_counter() - t0) * 1e3
        out: list[Detection] = []
        if res.boxes is None:
            return out
        names = res.names
        masks = res.masks.data.cpu().numpy() if getattr(res, "masks", None) is not None else None
        for i, (xyxy, c, k) in enumerate(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                             res.boxes.cls.cpu().numpy().astype(int))):
            raw = names[int(k)]
            label = self.label_map.get(raw)
            if label is None:
                continue
            mask = masks[i] if masks is not None else None
            out.append(Detection(label, float(c), tuple(float(v) for v in xyxy), self.name, mask))
        return out


class MultiDetector:
    """Runs several detectors and concatenates their outputs."""

    def __init__(self, detectors: Iterable[Detector]):
        self.detectors = list(detectors)

    def detect(self, color: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        for d in self.detectors:
            out.extend(d.detect(color))
        return out

    @property
    def last_ms(self) -> float:
        return sum(getattr(d, "last_ms", 0.0) for d in self.detectors)


class StubDetector:
    """Scripted detections for tests: a callable (frame_seq -> list[Detection]) or a fixed list."""

    def __init__(self, script=None):
        self.script = script or []
        self.calls = 0
        self.last_ms = 0.0

    def detect(self, color: np.ndarray) -> list[Detection]:
        self.calls += 1
        if callable(self.script):
            return list(self.script(self.calls))
        return list(self.script)


class Stabilizer:
    """Keeps the best detection of a label only when it persisted for N consecutive frames."""

    def __init__(self, label: str, n: int = 3, max_jump_px: float = 80.0):
        self.label = label
        self.n = n
        self.max_jump = max_jump_px
        self.history: deque[Detection] = deque(maxlen=n)

    def update(self, dets: list[Detection]) -> Detection | None:
        cands = [d for d in dets if d.label == self.label]
        if not cands:
            self.history.clear()
            return None
        best = max(cands, key=lambda d: d.conf * (1.0 + 1e-6 * d.area))
        if self.history:
            px, py = self.history[-1].center
            cx, cy = best.center
            if abs(px - cx) > self.max_jump or abs(py - cy) > self.max_jump:
                self.history.clear()
        self.history.append(best)
        return best if len(self.history) >= self.n else None

    def reset(self):
        self.history.clear()


def build_detector(cfg):
    d = cfg.detector
    if d.backend == "stub":
        return StubDetector()
    label_map = d.label_map.to_dict() if hasattr(d.label_map, "to_dict") else dict(d.label_map)
    dets = []
    if d.backend in ("yoloe", "both"):
        dets.append(UltralyticsDetector(str(resolve_path(cfg, d.yoloe_model)), list(d.classes), label_map,
                                        d.conf, d.imgsz, d.device, name="yoloe"))
    if d.backend in ("coco", "both") or (d.backend == "yoloe" and d.get("coco_model")):
        coco = d.get("coco_model")
        if coco:
            dets.append(UltralyticsDetector(str(resolve_path(cfg, coco)), None, {}, d.conf, d.imgsz, d.device, name="coco"))
    return dets[0] if len(dets) == 1 else MultiDetector(dets)
