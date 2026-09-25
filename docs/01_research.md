# 01 · Research: options for an autonomous "fetch a drink from the fridge" on a G1 (23 DoF)

Date: 2026-09-25. Everything below is either (V) verified from local sources in this workspace,
(D) taken from vendor/third-party documentation fetched on this date, or (I) inferred/estimated.
Nothing here has yet been validated on the robot.

## 1. Platform facts that drive the design

### 1.1 Robot and compute (V)
- G1 Edu, **23 DoF**: 6/leg, 1 waist yaw, **5 per arm** (shoulder pitch/roll/yaw, elbow, wrist roll). No wrist pitch/yaw, no waist pitch/roll. Source: `g1-docs/pages/03_01_joint_motor_sequence.md`, `xr_teleoperate/assets/g1/g1_body23.urdf`.
- Arm reach ≈ 0.45 m, payload ≈ 2 kg (varies strongly with pose). Source: `00_about_G1.md`.
- Hands: BrainCo Revo2 (6 DoF) driven by `brainco_hand_service` over DDS topics `rt/brainco/{left,right}/cmd|state` (`MotorCmds_`/`MotorStates_`, fingers normalized 0 = open … 1 = closed, order thumb, thumb-aux, index, middle, ring, pinky). The bridge also accepts a one-entry gripper command (`cmds[0].q`) mapped to open/closed poses (`config/gripper.yaml`). Feedback is position/speed/current only; no tactile.
- PC2: Jetson Orin NX 16 GB, JetPack 6.2 (L4T 36.4.3), Ubuntu 22.04, IP 192.168.123.164. Unitree deploys no services on it.
- Head sensors: Intel RealSense D435i (RGB 69°×42°, depth 87°×58°) and Livox Mid-360 (360°×59°, DDS topic `rt/utlidar/cloud_livox_mid360` at 10 Hz, frame `livox_frame`).

### 1.2 The head camera looks 47.6° downward (V, needs on-robot confirmation)
`g1_body23.urdf` places `d435_link` at torso +(0.058, 0.018, 0.420) m with pitch **0.8308 rad = 47.6°** below horizontal. The Unitree depth-camera page shows the same geometry (optical axis 42.4° from vertical, 55° vertical FOV drawn). With straight legs the URDF puts the camera ≈1.26 m above the ground; in the controller's default stance (knees bent) ≈1.15–1.20 m (I).

Consequence, with h = 1.18 m camera height:

| range D from camera | RGB top-of-view height (26.6° below horizontal) | depth top-of-view height (18.6°) |
|---|---|---|
| 0.4 m | 0.98 m | 1.05 m |
| 0.6 m | 0.88 m | 0.98 m |
| 1.0 m | 0.68 m | 0.84 m |
| 1.5 m | 0.43 m | 0.68 m |
| 2.0 m | 0.18 m | 0.51 m |
| 2.4 m | floor | 0.37 m |

So the RGB stream **never sees anything above ~1.0 m** and beyond ~2.4 m sees only the floor. A typical fridge door handle (1.0–1.4 m) and a can on a chest-height shelf are visible only from very close, and only in the top rows of the image, if at all. Search and approach must therefore work from the lower half of the fridge; handle and can localization must lean on depth/LiDAR geometry and known priors, not on a detector seeing them from afar. The user's successful teleop with this RGB stream proves the workspace is visible from the position they used; the deployment checklist starts by measuring exactly that.

The Mid-360 is mounted inverted with 2.3° pitch, so its vertical FOV spans ≈ +5° above horizontal to −54° below: it does see the door front and handle region at 0.5–1.0 m and is the best geometric fallback (I).

### 1.3 Motion interfaces available without leaving the built-in controller (V)
- **LocoClient** RPC (`unitree_sdk2py.g1.loco`): `SetVelocity(vx, vy, omega, duration)` (command expires after `duration`, default 1 s), `Move`, `StopMove`, `SetFsmId/GetFsmId`, `GetFsmMode`, `SetStandHeight`, `SetSpeedMode`, `BalanceStand`. FSM 500 = normal walking controller; 4 = locked stand; 801/802 = walk-run.
- **arm_sdk** DDS topic `rt/arm_sdk` (`unitree_hg LowCmd_`): motors 12–28 controllable while the built-in loco controller keeps balance; `motor_cmd[29].q` is the blend weight 0…1. Works in locked stand and FSM 500/501. Python example: `unitree_sdk2_python/example/g1/high_level/g1_arm5_sdk_dds_example.py` (kp 60, kd 1.5, 50 Hz). The arm action service `g1_arm_example` must be off, otherwise it competes for the arms.
- **Odometry** DDS `rt/odommodestate` (500 Hz) / `rt/lf/odommodestate` (20 Hz), `unitree_go SportModeState_`: position (m, world frame anchored at power-on), velocity (body frame), rpy (rad), yaw_speed, quaternion.
- **LowState** `rt/lowstate` (`unitree_hg LowState_`): all joint q/dq/tau_est, `mode_machine` (1 = 23 DoF).
- **IK**: `xr_teleoperate/teleop/robot_control/robot_arm_ik.py::G1_23_ArmIK` (pinocchio + CasADi/IPOPT, locks legs+waist, end-effector frames 0.20 m past the wrist-roll joints, cost 50·pos + 0.5·rot + regularization). Teleop-proven on this robot.

### 1.4 Camera sharing on PC2 (V)
`realsense-webrtc` captures the D435i RGB through V4L2 at 1080p30 for teleop; a V4L2 node has one capture owner. The autonomous stack uses `pyrealsense2` for aligned RGB-D and must run **instead of** `teleimager-webrtc`, or stream its own annotated frames for monitoring.

## 2. Perception options for the Orin NX 16 GB

Budget assumption: perception ≤ ~40 ms per frame at 640-px input so a 10 Hz sense-act loop leaves headroom for depth processing and the SDK threads (I).

### 2.1 Closed-set real-time detectors (COCO)
- YOLO26n TensorRT FP16 **4.1 ms**/image on Orin NX 16 GB, INT8 3.5 ms (D, Ultralytics Jetson guide, batch 1, no pre/post-processing). COCO has `refrigerator`, `person`, `bottle`, `cup` but **no can and no handle**.
- RF-DETR-nano: 4–5× slower than YOLO26n and ~4× peak VRAM on Jetson (D, Labellerr edge benchmark) but finds more small/distant objects.
- Verdict: excellent for `refrigerator` (search/approach) and `person` (handover); insufficient for handle/can.

### 2.2 Open-vocabulary detectors
- **YOLOE-26 (Ultralytics)**: text prompts, visual prompts, or prompt-free (4,585-class vocabulary). LVIS zero-shot mAP50-95 24.7 (n) … 40.6 (x). Prompt embeddings are baked into the exported TensorRT engine (no re-prompting at runtime; ~89 % extra latency only with the full vocabulary, negligible for 4–6 classes). Text encoder (MobileCLIP, 254 MB) downloaded once on the workstation, not needed on the robot. Latency on Orin NX not published; the YOLO26 backbone numbers apply, so ≈5–10 ms FP16 for n/s (I).
- **NanoOWL (OWL-ViT + TensorRT)**: 95 FPS (ViT-B/32) / 25 FPS (ViT-B/16) on AGX Orin; NX not published, expect ~40 % of AGX (I). Free-form prompts at runtime; weaker localization precision than YOLO-style boxes; larger memory.
- **Grounding DINO / DINO-X / OWLv2 full**: too slow or not open for edge (I).
- Smoke test (V, this workstation, CPU): `yoloe-26s-seg-fridge.pt` + `yolo26n.pt` on 16 COCO val images: the one kitchen scene (000000000139) yields `fridge` 0.79 (YOLOE) / 0.56 (COCO) on a small distant fridge, `person` agrees across both models; ~45–80 ms per image on CPU. No fridge-interior / can images were available offline, so `handle`/`can` recall is unmeasured until robot captures exist.
- Verdict: YOLOE-26s with baked prompts `["refrigerator", "refrigerator door handle", "soda can", "person"]` is the primary detector; YOLO26n COCO stays as a cross-check for `refrigerator`/`person`. Zero-shot quality on the user's fridge is unknown and must be measured on captured frames; a few-shot fine-tune of YOLOE on ~50 auto-labelled robot frames is the cheap upgrade path (would run on the Jetson itself in minutes, not on the workstation GPUs).

### 2.3 Small VLMs as verifier / planner
- **Qwen3-VL-2B / 4B**: 2B on Orin Nano Super reaches ~0.9 queries/s with transformers for image + ~150 tokens (D, NVIDIA forum); Orin NX has roughly 2× the GPU → ~1–2 s per query (I). 4B AWQ served by vLLM is listed on Jetson AI Lab (no NX numbers).
- **Moondream 2 (1.9B)** fits in 2 GB, has `point()`/`detect()`/`query()` skills with single-token coordinates; JetPack 6 install documented. **Moondream 3 preview** (9B MoE, 2B active) needs ~18 GB in bf16, too large next to everything else on 16 GB shared memory (I).
- SmolVLM2 (256M–2.2B): fastest, weakest grounding.
- Verdict: a VLM is **not** on the control path (seconds of latency, ~2–5 GB memory). It is worth having as an optional phase verifier ("is the fridge door open?", "is the can in the hand?") through an OpenAI-compatible HTTP endpoint (vLLM or Moondream server on PC2). Ship the interface and a stub; enable only after the geometric checks are proven.

### 2.4 Depth and 3D
- D435i aligned depth is good to ±1–2 % at 0.5–2 m; median filtering inside detection boxes and plane RANSAC on the door front are sufficient for ±2 cm targets at 0.5 m (I).
- Mid-360 point cloud at 10 Hz (200k pts/s) through DDS `PointCloud2_` is available in `unitree_sdk2py.idl.sensor_msgs`; used as an optional door-plane/handle-protrusion source when the camera view ceiling bites.
- Monocular depth models (Depth Anything v2) add nothing when metric RGB-D is on board.

## 3. Policy options

### 3.1 End-to-end VLA
- **GR00T N1.7 (3B)** lists `UNITREE_G1` and `UNITREE_G1_SONIC` embodiments; zero-shot only for pretraining embodiments (29-DoF G1 with Unitree/Inspire hands); new embodiments need fine-tuning on LeRobot-format demos. Jetson deployment path targets JetPack 7.2 (Thor) with TensorRT export; Orin support listed but no latency numbers (D).
- **π0.5 / SmolVLA via LeRobot**: LeRobot now supports the G1 (29 and 23 DoF) with Holosoma / GR00T-WBC locomotion controllers, exoskeleton or joystick teleop, and π0.5 training recipes (D, updated March 2026). Requires demonstrations and GPU training.
- Edge runtimes (Jetson-PI, vla.cpp, reflex-vla) reach 5–10 Hz on Orin for pi0-class models with asynchronous chunking (D).
- **Constraints here**: no demonstrations exist, the workstation GPUs are off-limits for training, and 23-DoF + BrainCo is not a pretraining embodiment. A VLA is therefore not viable for v1. Recording demos through xr_teleoperate (`--record`) or LeRobot and fine-tuning elsewhere is the documented v2 path.

### 3.2 Sim-to-real RL for door opening
- NVIDIA **DoorMan** (2026): G1 opens spring-loaded doors from a single shaky RGB stream, 83 % success, trained in Isaac Lab with staged resets + GRPO; policy not released (D). Confirms feasibility on this hardware and two lessons reused below: move compliantly with the door, and plan for the handle leaving the view.
- Not reproducible here without simulation infrastructure and GPU training.

### 3.3 Structured autonomy (perception + scripted, closed-loop skills)
- Open-vocab detection + RGB-D geometry → 3D targets in the pelvis frame → IK-driven Cartesian primitives (pre-grasp, approach, close, arc pull, retract) with LocoClient servoing for search/approach/return and odometry for the way back.
- Matches what teleop did kinematically (same IK, same hand commands), needs no training data, runs in ~15 ms of GPU per frame, and every phase has a measurable success check.
- Costs: geometry priors (door width, handle offset, hinge side) live in a config and need one afternoon of on-robot tuning; the approach is specific to *a* fridge, not fridges in general.

## 4. Related work consulted
- Unitree docs (local mirror `g1-docs/`): sport services, arm control, joint order, odometry, depth camera, LiDAR, motion switcher, arm action service.
- xr_teleoperate (local): `G1_23_ArmController`, `G1_23_ArmIK`, BrainCo controller; teleop-proven parameters (kp 80/40 wrist in debug mode; arm_sdk weight ramp).
- Ultralytics Jetson guide and YOLOE docs; Labellerr 2026 edge benchmark (RF-DETR vs YOLO26); NanoOWL README; Jetson AI Lab model pages; Moondream local docs; NVIDIA forum Qwen3-VL-2B thread; Isaac-GR00T README; LeRobot Unitree G1 docs; Humanoids Daily on DoorMan; a G1 navigation write-up (deadband on SetVelocity, always stop in `finally`, FAST-LIO + ICP localization).

Sources (fetched 2026-09-25):
- https://docs.ultralytics.com/guides/nvidia-jetson/
- https://docs.ultralytics.com/models/yoloe/
- https://www.labellerr.com/blog/best-vision-model-for-edge-deployment/
- https://github.com/NVIDIA-AI-IOT/nanoowl
- https://docs.moondream.ai/running-locally/
- https://forums.developer.nvidia.com/t/performance-inquiry-optimizing-qwen3-vl-2b-inference-for-2-qps-target-on-orin-nano-super/359639
- https://www.jetson-ai-lab.com/models/qwen3-vl-4b/
- https://github.com/NVIDIA/Isaac-GR00T
- https://huggingface.co/docs/lerobot/unitree_g1
- https://www.humanoidsdaily.com/news/nvidia-s-doorman-teaches-humanoids-to-open-doors-faster-than-humans-can
- https://arxiv.org/html/2607.12659v3 (Jetson-PI)
- https://www.rawatprashant.com/blog/g1-web-control-robot-side.html
- https://github.com/jetsonhacks/jetson-orin-librealsense
