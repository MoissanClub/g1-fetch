"""Command line: run the task, check the robot, capture frames, test the detector.

  python -m g1_fetch.cli run   [--dry-run [--sim]] [--step] [--from PHASE] [--until PHASE] [--camera FOLDER]
  python -m g1_fetch.cli check                       # PC2: SDK, FSM, odometry, camera, detector latency
  python -m g1_fetch.cli capture OUT_DIR [-n N]      # save RGB-D frames for offline tuning
  python -m g1_fetch.cli detect IMAGES_OR_FOLDER     # run the detector, print + save annotated images
  python -m g1_fetch.cli geometry                    # print camera view ceilings and reach numbers
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import load_config
from .task import PHASES, FetchDrinkTask


def _common(p: argparse.ArgumentParser):
    p.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. loco.max_vx=0.2")
    p.add_argument("-v", "--verbose", action="store_true")


def _overrides(items):
    import yaml

    out: dict = {}
    for item in items:
        key, _, val = item.partition("=")
        node = out
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def cmd_run(args):
    cfg = load_config(args.config, _overrides(args.set))
    if args.step:
        cfg.task.step_mode = True
    from .robot import build_robot

    rb = build_robot(cfg, dry_run=args.dry_run, camera_source=args.camera)
    if args.sim:
        import math

        from tests.sim_scene import attach
        from .frames import Pose2D

        attach(rb, fridge_x=1.8, fridge_yc=0.0)
        rb.loco._pose = Pose2D(0.0, 0.0, math.radians(100))
    try:
        task = FetchDrinkTask(rb, start_phase=args.from_phase, stop_after=args.until)
        ok = task.run()
    finally:
        rb.close()
    print("RESULT:", "success" if ok else "failed", "| history:", task.state.history)
    return 0 if ok else 1


def cmd_check(args):
    cfg = load_config(args.config, _overrides(args.set))
    import numpy as np

    print("== SDK / DDS ==")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, str(cfg.network_interface))
    from .control.loco import UnitreeLoco

    loco = UnitreeLoco(cfg)
    try:
        loco.start()
        print(f"fsm_id={loco.fsm_id()} odom={loco.pose()}")
    except Exception as e:
        print("LOCO PROBLEM:", e)
    print("== arm lowstate ==")
    from .control.arm import UnitreeArmStreamer

    arms = UnitreeArmStreamer(cfg)
    t0 = time.time()
    while not arms._ready() and time.time() - t0 < 3:
        time.sleep(0.1)
    print("mode_machine", arms.mode_machine, "left q", np.round(arms.measured("left"), 2) if arms._ready() else "n/a")
    print("== hands ==")
    from .control.hand import UnitreeBraincoHand

    for side in ("left", "right"):
        h = UnitreeBraincoHand(cfg, side, publish=False)   # read-only: no finger commands during check
        time.sleep(0.5)
        print(side, "state", None if h.state() is None else np.round(h.state(), 2))
        h.close()
    print("== camera ==")
    try:
        from .perception.camera import RealSenseCamera

        cam = RealSenseCamera(cfg)
        f = cam.read()
        valid = float((f.depth > 0).mean())
        print(f"color {f.color.shape} depth valid {valid:.0%} intrinsics {f.intrinsics}")
        print("== detector ==")
        from .perception.detector import build_detector

        det = build_detector(cfg)
        det.detect(f.color)
        ts = []
        for _ in range(10):
            f = cam.read()
            t0 = time.perf_counter()
            dets = det.detect(f.color)
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"detector {np.median(ts):.1f} ms median, last dets: {[(d.label, round(d.conf, 2)) for d in dets]}")
        cam.close()
    except Exception as e:
        print("CAMERA/DETECTOR PROBLEM:", e)
    return 0


def cmd_capture(args):
    cfg = load_config(args.config, _overrides(args.set))
    from .perception.camera import RealSenseCamera, save_frame

    cam = RealSenseCamera(cfg)
    out = Path(args.out)
    for i in range(args.n):
        f = cam.read()
        save_frame(f, out, f"{i:04d}")
        print("saved", i, "valid depth", f"{float((f.depth > 0).mean()):.0%}")
        time.sleep(args.interval)
    cam.close()
    return 0


def cmd_detect(args):
    cfg = load_config(args.config, _overrides(args.set))
    import cv2

    from .perception.detector import build_detector

    det = build_detector(cfg)
    paths = []
    for p in args.paths:
        p = Path(p)
        paths += sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")) if p.is_dir() else [p]
    out = Path(args.out) if args.out else None
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t0 = time.perf_counter()
        dets = det.detect(img)
        ms = (time.perf_counter() - t0) * 1e3
        print(f"{p.name}: {ms:.0f} ms  {[(d.label, round(d.conf, 2), [int(v) for v in d.box]) for d in dets]}")
        if out:
            out.mkdir(parents=True, exist_ok=True)
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d.box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(out / p.name), img)
    return 0


def cmd_geometry(args):
    cfg = load_config(args.config, _overrides(args.set))
    import pinocchio  # noqa: F401
    import numpy as np

    from .control.arm import ArmKinematics
    from .frames import Frames

    fr = Frames(cfg)
    print(f"camera height {cfg.robot.camera_height_m} m, pitch down {np.degrees(fr.camera_pitch_down):.1f} deg, pelvis height {fr.pelvis_height_m:.3f} m")
    print("range  RGB-top  depth-top   (highest floor height visible at that forward range)")
    for d in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5):
        print(f"{d:5.1f}  {fr.view_ceiling(d):6.2f}  {fr.view_ceiling(d, 58):8.2f}")
    kin = ArmKinematics(cfg)
    for side in ("left", "right"):
        T = kin.fk(side, np.zeros(5))
        print(f"{side} ee at q=0: {np.round(T[:3, 3], 3)}  (floor height {fr.floor_height(T[:3, 3]):.2f} m)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="g1_fetch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    _common(p)
    p.add_argument("--dry-run", action="store_true", help="mock robot, no DDS")
    p.add_argument("--step", action="store_true", help="wait for Enter before each motion")
    p.add_argument("--from", dest="from_phase", default="search", choices=PHASES)
    p.add_argument("--until", default=None, choices=PHASES)
    p.add_argument("--camera", default=None, help="replay folder instead of the RealSense")
    p.add_argument("--sim", action="store_true", help="with --dry-run: attach the synthetic fridge scene (tests/sim_scene.py)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("check")
    _common(p)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("capture")
    _common(p)
    p.add_argument("out")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--interval", type=float, default=0.5)
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("detect")
    _common(p)
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_detect)
    p = sub.add_parser("geometry")
    _common(p)
    p.set_defaults(fn=cmd_geometry)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
