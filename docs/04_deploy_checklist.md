# 04 · Deployment and hardware bring-up checklist (PC2, Jetson Orin NX)

Work through the sections in order. Every "MEASURE" item feeds a value in `configs/default.yaml`
(or a robot-specific override file passed with `--config`). Nothing moves the robot until section 5.

## 0. Before touching the robot
- [ ] Copy `g1-fridge-fetch/` and `xr_teleoperate/assets/g1/` (URDF only) to PC2, e.g. `~/junda/`.
- [ ] Read `docs/01_research.md §1.2` (camera looks 47.6° down) and `docs/02_decisions.md`.

## 1. PC2 software (`scripts/pc2_setup.sh` does the non-interactive parts)
- [ ] `unitree_sdk2_python` installed (`cyclonedds==0.10.2`), `python -c "import unitree_sdk2py"` works.
- [ ] `brainco_hand_service` running (already deployed) and `rt/brainco/left/state` publishing.
- [ ] librealsense with `pyrealsense2` for JetPack 6.2: either the JetsonHacks prebuilt kernel modules
      (https://github.com/jetsonhacks/jetson-orin-librealsense) or a source build with `-DFORCE_RSUSB_BACKEND=ON
      -DBUILD_PYTHON_BINDINGS=ON`. `realsense-viewer` or `rs-enumerate-devices` sees the D435i on USB 3.
- [ ] PyTorch / torchvision JetPack 6 wheels, `ultralytics`, TensorRT Python bindings present
      (`python -c "import tensorrt"`), `pinocchio` + `casadi` (conda-forge or pip `pin`), `opencv-python-headless`,
      `pyyaml`, `scipy`.
- [ ] Stop the teleop camera pipeline while running autonomy: `sudo systemctl stop teleimager-webrtc` (and `teleimager`).
- [ ] Detector engines: copy `models/yoloe-26s-seg-fridge.pt` and `models/yolo26n.pt`, then on PC2
      `python scripts/export_detector.py --engine` (TensorRT FP16, ~5 min). Set `detector.yoloe_model` / `coco_model`
      to the `.engine` files.

## 2. Static checks (robot powered, standing in damp/locked stand, no motion commands)
- [ ] `python -m g1_fetch.cli check` prints: fsm_id (expect 500 once the main controller is started), odometry pose,
      `mode_machine 1`, both hand states, camera intrinsics, detector latency.
      MEASURE: detector ms → should be < 20 ms for both models combined.
- [ ] `python -m g1_fetch.cli geometry` prints the view-ceiling table for the configured camera height.

## 3. Measurements (tape measure + a few captures)
- [ ] MEASURE `robot.camera_height_m`: lens centre above the floor in the normal standing pose.
- [ ] Stand the robot ~1.0 m from the fridge, square to it: `python -m g1_fetch.cli capture data/captures/fridge_1m -n 10`.
      Repeat at 0.6 m and at the teleop manipulation distance. Run `python -m g1_fetch.cli detect data/captures/fridge_1m --out /tmp/ann`
      and look at the annotated images. Is the fridge detected? Is the handle ever in view? Is the can shelf in view with the door open?
      This decides whether `handle`/`can` detections are usable or the geometric priors carry the task.
- [ ] MEASURE fridge: door width, hinge side, handle bar inset from the free edge, protrusion, height span, can shelf height, can diameter
      → `fridge.*`. Check `fridge.handle_grasp_height_m` is inside the bar's span and inside `[0.75, 0.95]`.
- [ ] MEASURE `robot.ee_offset_x`: distance from the wrist-roll joint axis to the palm centre of the BrainCo hand.
- [ ] MEASURE `robot.palm_inward_wrist_roll`: with arm_sdk enabled and the arm at home, jog the wrist roll until the palm faces the body midline; record the joint angle for each hand.
- [ ] Camera extrinsic sanity: with the right hand extended in view, compare the hand's pixel position with the projection of
      `kin.fk("right", measured_q)` through `Frames.T_pelvis_optical`. > 3 cm error → adjust `camera.torso_T_link` (pitch first).

## 4. Dry runs on PC2 (no robot motion)
- [ ] `python -m g1_fetch.cli run --dry-run --camera data/captures/fridge_1m --until approach` : real detector on replayed frames,
      mock loco/arms. Check the log for `approach_done` with a sane plane distance/yaw.
- [ ] `pytest tests -q` passes on PC2 (skips nothing that matters).

## 5. Robot on, incremental (operator holds the remote, robot in FSM 500 / "ai" mode, arms free)
Every run uses `--step` first; remove `--step` only once a phase has succeeded twice.
- [ ] Disable the arm action service once: `python -c "from g1_fetch.control.robot_state import set_arm_action_service as f; f(False)"`
      (after `ChannelFactoryInitialize`; the CLI `run` does this automatically when `task.disable_arm_service: true`).
- [ ] **Arms only**: `run --step --from open_door --until open_door` is too much for a first test; instead
      `python -m g1_fetch.cli run --step --until search` and confirm the arm_sdk ramp moves the arms to `arm.home_q` smoothly.
      MEASURE/tune `arm.home_q`, `arm.kp/kd` if the arms oscillate or sag.
- [ ] **Search**: `run --step --until search` with the fridge 1.5–2 m away. Expect rotation steps and `fridge_found`.
- [ ] **Approach**: `run --step --until approach`. Expect the robot to stop at `approach.standoff_m` squared to the door with the handle at y ≈ −0.15.
      Tune `approach.k_*`, `tol_*`, `slow_zone_m`. Watch for oscillation (lower gains) or shuffling (raise deadband).
- [ ] **Open door**: `run --step --until open_door`. Step through pre-grasp, grasp, hook. If the hook closes fully → adjust
      `fridge.handle_inset_m`/`handle_protrusion_m`/grasp height. Then the pull-and-back; tune `fridge.pull_back_m`, `pull_back_speed`,
      `arm.kp_compliant`. If the hand slips off, raise `hand.hook` closure or lower the retreat speed. Check `verify_open` in the log.
- [ ] **Fetch can**: `--until fetch_can`. Verify the reach pose does not collide with the open door (adjust `fridge.reach_lateral_m`).
      If `can` is never detected, tune `fridge.can_depth_in_cavity_m` / `can_shelf_height_m` priors; consider `fridge.stage_b_sweep: true`.
- [ ] **Close door**: `--until close_door`. Tune `fridge.close_standoff_m`, `close_lateral_m`. If hook_sweep cannot reach, the push alone
      may suffice with a wider approach; if the door bounces open, lower push speed.
- [ ] **Return + handover**: full run. Stand where the robot started; take the can firmly and pull; the hand should open within ~0.5 s.
      Tune `task.pull_trigger_err_rad` if it releases too early (walking vibration) or too late.

## 6. After the first successful run
- [ ] Save the tuned values into `configs/<robot-or-fridge>.yaml` and commit.
- [ ] Keep `data/runs/*` from successful and failed runs; they are the training set for a few-shot YOLOE fine-tune (on the Jetson) if
      detection was the weak link, and the evidence for deciding whether a learned policy is worth demonstrations (decisions D1).
- [ ] Optional: start a VLM server (`vllm serve Qwen/Qwen3-VL-2B-Instruct --max-model-len 4096` on PC2, or Moondream station) and set
      `verifier.enabled: true` to add the yes/no checks at phase boundaries.

## Known unknowns to resolve on the robot (ordered by risk)
1. Whether the RGB camera sees the handle and the can shelf at manipulation distance (view ceiling ≈ 0.9 m at 0.5 m range).
2. Door swing vs. robot body: the door cannot be pulled far while the robot stands in its swing wedge; the pull-and-back and the
   reach/close poses are geometric guesses to be tuned on the real fridge.
3. Hook grasp reliability with a 5-DoF arm (no wrist pitch/yaw) on the specific handle profile.
4. `LocoClient` velocity limits and turning behaviour in FSM 500 with the arms overridden.
5. RealSense on JetPack 6.2 (kernel modules / RSUSB backend).
