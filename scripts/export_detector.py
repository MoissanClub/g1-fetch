"""Prepare the detector checkpoints.

Workstation (CPU is fine):   python scripts/export_detector.py --bake
    downloads yoloe-26s-seg.pt + the MobileCLIP text encoder, bakes the class prompts from
    configs/default.yaml into models/yoloe-26s-seg-fridge.pt, and downloads yolo26n.pt.
PC2 (Jetson):                python scripts/export_detector.py --engine
    exports both checkpoints to TensorRT FP16 engines next to them (needs CUDA + tensorrt).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pinocchio  # noqa: F401,E402  (import order on the workstation)

from g1_fetch.config import load_config  # noqa: E402


def bake(cfg, scale: str):
    from ultralytics import YOLO, YOLOE

    os.chdir(ROOT / "models")
    classes = list(cfg.detector.classes)
    m = YOLOE(f"yoloe-26{scale}-seg.pt")
    m.set_classes(classes, m.get_text_pe(classes))
    out = f"yoloe-26{scale}-seg-fridge.pt"
    m.save(out)
    print("saved", ROOT / "models" / out, "classes", classes)
    YOLO("yolo26n.pt")
    print("yolo26n.pt ready")


def engine(cfg, imgsz: int):
    from ultralytics import YOLO, YOLOE

    for key in ("yoloe_model", "coco_model"):
        p = cfg.detector.get(key)
        if not p:
            continue
        p = ROOT / p
        if p.suffix != ".pt":
            p = p.with_suffix(".pt")
        if not p.exists():
            print("missing", p)
            continue
        model = YOLOE(str(p)) if "yoloe" in p.name else YOLO(str(p))
        out = model.export(format="engine", half=True, imgsz=imgsz, device=0)
        print("exported", out)
    print("now set detector.yoloe_model / coco_model to the .engine paths")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", action="store_true")
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--scale", default="s", choices=["n", "s", "m"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    cfg = load_config()
    if args.bake:
        bake(cfg, args.scale)
    if args.engine:
        engine(cfg, args.imgsz)
    if not (args.bake or args.engine):
        ap.print_help()
