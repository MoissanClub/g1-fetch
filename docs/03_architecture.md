# 03 · Architecture

## 1. Process layout on PC2 (Jetson Orin NX)

One Python process (`python -m g1_fetch.cli run`) with three background threads plus the main task thread:

| thread | rate | does |
|---|---|---|
| arm streamer | 100 Hz | publishes `rt/arm_sdk` (`unitree_hg LowCmd_`) with per-step joint-velocity clipping, holds waist yaw, ramps the blend weight (motor 29) |
| hand streamer ×2 | 20 Hz | publishes `rt/brainco/{left,right}/cmd` (six-finger `MotorCmds_`) |
| DDS callbacks | as received | `rt/lowstate` (joint q / tau_est), `rt/lf/odommodestate` (pose), `rt/brainco/*/state` |
| task (main) | 10 Hz sense-act | RealSense aligned RGB-D → detector → geometry → skill logic → `LocoClient.SetVelocity` / arm targets / hand poses |

External processes: Unitree's built-in controller (FSM 500, "ai" mode) and `brainco_hand_server`. `teleimager-webrtc` must be stopped (single V4L2 owner). The optional VLM server (vLLM / Moondream) is a separate process reached over HTTP.

## 2. Frames and geometry (`g1_fetch/frames.py`)

- **pelvis**: URDF root, x forward, y left, z up. All arm targets live here. Height above the floor: `pelvis_height_m = camera_height_m − 0.474` (0.706 m with the default 1.18 m camera height).
- **camera**: `T_pelvis_optical = T_pelvis_torso · T_torso_d435 · R_link_optical`, from the URDF (`d435_joint`: xyz 0.058 0.018 0.420, pitch 0.831 rad). RealSense optical: x right, y down, z forward.
- **world**: Unitree odometry (`rt/odommodestate`), anchored at power-on. Only differences matter; a `Pose2D(x, y, yaw)` converts world ↔ pelvis for horizontal targets.
- **DoorModel** (`skills/door.py`): hinge point, closed-door outward normal and width in the world frame; gives handle / free-edge / normal / tangent for any opening angle and converts to the current pelvis frame. Built once from the fitted door plane and the chosen handle point.

The **view ceiling** (highest visible floor height at range D): RGB `1.18 − 0.50·D`, depth `1.18 − 0.34·D`. The design never expects to see anything above ~0.9 m unless within 0.5 m.

## 3. Perception (`g1_fetch/perception/`)

- `camera.py`: `RealSenseCamera` (aligned depth, metres, invalid = 0), `FolderCamera` (replay of `capture` output), `SyntheticCamera` (tests).
- `detector.py`: `UltralyticsDetector` for YOLOE (text prompts baked) and YOLO26 COCO; `MultiDetector` concatenates; `Stabilizer` requires N consecutive frames; canonical labels `fridge | handle | can | person`.
- `localize.py`: box median depth → 3D point; `door_plane()` = RANSAC vertical plane on box points above the floor, returns distance along x, yaw error, lateral/vertical extent; `handle_from_prior()` = free edge + inset/protrusion at the grasp height; `cavity_depth_jump()` = door-open evidence; `can_axis_from_box()`.
- `verifier.py`: optional yes/no VLM through an OpenAI-compatible endpoint; failures return `None` and never block.

## 4. Control (`g1_fetch/control/`)

- `loco.py`: `UnitreeLoco` wraps `LocoClient.SetVelocity(vx, vy, ω, 0.3 s)` with clipping and deadband, checks FSM ∈ {500, 501} at start, subscribes to odometry; `turn_by()` and `drive_to()` are blocking P-controllers on odometry with timeouts and abort hooks. `MockLoco` integrates commands kinematically.
- `arm.py`: `ArmKinematics` (pinocchio reduced model: legs + waist locked, ee frames 0.15 m past the wrist-roll joints). IK = CasADi/IPOPT on position (weight 100) + forearm direction (weight 0.5) with the four proximal joints, multi-seed, ~2 ms; wrist roll solved analytically for the palm direction. `ArmStreamer` holds/streams targets; `enable()/disable()` ramp the arm_sdk weight; `set_gain()` lowers kp for compliant phases; `tracking_error()` = max |measured − commanded|.
- `hand.py`: open / grasp / hook poses (six normalized finger values) and `closed_fraction()` from feedback to detect a missed grasp (fingers fully closed = nothing in the hand).
- `robot_state.py`: `ServiceSwitch("g1_arm_example", off)` so Unitree's arm-action service does not fight `rt/arm_sdk`.

## 5. Task state machine (`g1_fetch/task.py`, `g1_fetch/skills/`)

```
INIT ─ enable arm_sdk, hands open, arms home, record home pose
 │
SEARCH ─ stop-and-look rotation (0.45 rad steps) until `fridge` is stable for 3 frames; explore 1 m and retry once
 │
APPROACH ─ far: yaw on box centre, vx on depth; near (<1.2 m): yaw on door-plane normal, vy to put the
 │          handle at y = −0.15 (right hand), stop at 0.38 m (pelvis → door plane)
 │
OPEN_DOOR ─ locate handle (detector | plane+priors) → DoorModel (hinge, normal, width in the world frame)
 │          right hand: pre-grasp → grasp → hook; kp → 25; walk backward 0.08 m/s until 0.45 m or 60°
 │          (angle read from the hand position); release, retract along the door normal, home
 │          verify: cavity depth jump > 0.3 m (or VLM)
 │          stage B: face the free edge from 0.30 m, hook it from the inner side, walk backward to 95°
 │          refuse to continue below `min_open_deg` (70°)
 │
FETCH_CAN ─ drive to the reach pose (0.30 m in front of the cavity centre, facing the fridge)
 │          locate `can` (detector | prior at cavity centre, shelf height); nudge forward if just out of reach
 │          left hand: pre-grasp from the front → beside the can → grasp (closed_fraction check) → lift 3 cm →
 │          retract 0.2 m → carry pose
 │
CLOSE_DOOR ─ repeat: face the free edge (0.30 m, squared to the panel), right palm on the outer face, walk
 │           forward until the angle drops 30° or 0.35 m; until < 4°. Verify by plane fit from the standoff (+VLM);
 │           one retry
 │
RETURN ─ drive_to(home pose), then stop-and-look for `person`, servo to 0.8 m
 │
HANDOVER ─ extend the can at 1.0 m height; release when the arm tracking error exceeds 0.08 rad for 0.3 s
            (the user pulling); optional VLM confirmation; on timeout keep holding and report
```

Each phase is retried once (`task.max_retries`) except handover. `--step` waits for Enter before each motion. Any exception stops walking; arms keep their last targets (holding the can if we have it). The task log (`data/runs/<timestamp>/events.jsonl` + annotated JPEGs) records every decision.

## 6. Timing, rates, and latency budget (Orin NX)

| item | budget | basis |
|---|---|---|
| RealSense aligned RGB-D 640×480 | 5–8 ms CPU | librealsense align on the CPU |
| YOLOE-26s TensorRT FP16 | 6–10 ms | YOLO26n measured at 4.1 ms; s-scale ~2× |
| YOLO26n cross-check | 4 ms | Ultralytics Jetson table |
| plane RANSAC (≈ 6k points) | 5–10 ms | numpy |
| IK per Cartesian step | 2 ms | measured on the workstation, ~2–3× slower on the Orin CPU |
| total per 10 Hz tick | ≈ 40 ms | leaves > 50 % idle |

Velocity commands expire after 0.3 s, so a crashed process stops the robot within 0.3 s. Arm targets are clipped to 1.2 rad/s per joint; Cartesian moves stream at 50 Hz with 0.12 m/s default speed.

## 7. Safety

- The remote controller stays with the operator (built-in damp / stop).
- Loco: hard clips (0.3 / 0.2 m/s, 0.4 rad/s), deadband, expiry, `finally: stop()` in every skill.
- Arm: IK reject beyond 2 cm error, per-step velocity clip, compliant gains during contact phases, tracking-error watchdog during the pull.
- Hands: grasp verification by closure fraction; `hand.open()` in every failure path that could trap the door.
- `--step` mode and `--until PHASE` for incremental hardware bring-up.

## 8. What is deliberately not in v1

- No LiDAR consumer (the `rt/utlidar/cloud_livox_mid360` idl is available in the Python SDK; see decisions D5 for where it would slot in).
- No learned manipulation policy; no simulation beyond the ray-cast dry-run scene in `tests/sim_scene.py`.
- No obstacle avoidance during the return beyond odometry and a person detector; keep the path clear.
