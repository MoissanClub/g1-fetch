# 00 · Status and open questions (2026-09-25, end of the first session)

## Done
- Research and decisions written up (`01_research.md`, `02_decisions.md`), architecture (`03_architecture.md`),
  hardware bring-up checklist (`04_deploy_checklist.md`).
- Complete task implementation in `g1_fetch/`: perception (RealSense, YOLOE + YOLO26 COCO, RGB-D geometry, optional
  VLM verifier), control (LocoClient wrapper + odometry, arm_sdk streamer + pinocchio/CasADi IK for the 5-DoF arms,
  BrainCo hands), six skills, task runner with retries/step mode/logging, CLI (`run / check / capture / detect / geometry`).
- Host-side verification (CPU only, no GPU use):
  - unit tests: frames, deprojection, RANSAC door plane (distance + yaw), handle prior, door model both hinge sides,
    FK against the URDF, IK round-trips (< 1 cm on reachable targets), reach of the default handle/can targets,
    wrist-roll palm solver, streamer velocity clipping;
  - end-to-end dry run in a ray-cast scene (`tests/sim_scene.py`): all seven phases succeed, including hook-and-walk
    door opening to > 70°, can grasp, segmented push-to-close, odometry return, person servo and release-on-pull.
    With the default (realistic) speeds the whole task takes ~4.9 min of wall time in the mock (`data/runs/20260925-082050`).
  - detector pipeline smoke test on COCO images (fridge + person detected; ~50 ms/img on CPU).
- Detector checkpoints prepared in `models/` (prompts baked; TensorRT export runs on PC2).

## Session 2 (2026-09-26): PC2 bring-up, no robot motion
- PC2 environment reproducible via `scripts/pc2_setup.sh` + `requirements-pc2.txt` (lean CUDA/TensorRT apt set, `g1fetch`
  env cloned offline from `uni`, Jetson-index torch 2.11 with CUDA, TensorRT engines). `configs/pc2.yaml` carries the
  PC2-specific values (DDS interface `enP8p1s0`, GPU engines).
- Verified on PC2: unit + end-to-end dry-run tests pass; read-only robot check (DDS, odometry on `rt/odommodestate`,
  `mode_machine` 4, hands); camera capture through pyrealsense2; floor-plane calibration (camera 1.31 m, 48.7° down);
  detector benchmark 60–90 ms/frame on the GPU.
- Robot was left untouched (user put it in zero torque). Before arm tests it must be in Regular mode (R1 + X, FSM 500).
- Code fixes from the bring-up: odometry topic, `mode_machine` 4 accepted, read-only hand objects for `cli check`,
  measured camera geometry in the default config.

## Not done / not possible here
- Nothing has touched the robot. All motion parameters are geometric estimates.
- No training of any kind (user constraint on the workstation GPUs; no demonstrations exist anyway).
- LiDAR is not consumed. It is the documented fallback if the camera's down-tilt makes handle/can localization unreliable.

## Questions for the user (answers change the first hardware session)
1. **Camera view**: with the robot in its usual teleop stance in front of the fridge, can you see the handle and the
   can shelf in the head-camera stream? If yes, roughly at what distance? (The URDF says the camera looks 47.6° down;
   `python -m g1_fetch.cli geometry` prints what that implies.)
2. **Door swing during teleop**: how did you open the door: pull and step back, or pull from an angle? Did the robot
   stand square in front of the fridge or off to the handle side? This tunes `fridge.pull_back_m`,
   `edge_standoff_m` and the lateral offsets.
3. **Fridge numbers**: door width, handle inset/protrusion, handle bar height span, can shelf height. Defaults are
   0.60 / 0.05 / 0.04 m, 0.75–0.95 m band, 0.95 m.
4. **Home pose of the arms** that the built-in controller tolerates while walking with arm_sdk enabled; the default
   `arm.home_q` is an untested tucked pose.
5. **PC2 python stack**: which Python (system 3.10 vs conda) runs `xr_teleoperate` on PC2? The same interpreter should run
   this package so `unitree_sdk2py` and `pyrealsense2` are already available.

## Biggest risks, in order
1. Handle/can not visible to the RGB detector at manipulation range → geometric priors must carry the task.
2. Hook grasp on the real handle with a 5-DoF arm (no wrist pitch/yaw).
3. Door dynamics while walking with a compliant arm (slip, jam, magnetic seal on closing).
4. Loco controller behaviour with arms overridden (balance, foot placement close to the fridge).
