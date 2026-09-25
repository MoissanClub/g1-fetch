# 02 · Decisions

Each entry: decision, why, what it costs, when to revisit. See `01_research.md` for the evidence.

## D1 · v1 is structured autonomy, not a learned policy
Perception (open-vocabulary detector + RGB-D geometry, LiDAR optional) feeds scripted, closed-loop skills built on the same IK and DDS interfaces that made the teleop run work.
- Why: no demonstrations exist, the workstation GPUs are reserved (no training anywhere on the host), the 23-DoF + BrainCo embodiment is not covered by any pretrained VLA, and DoorMan-style sim-to-real RL is out of scope. Structured autonomy needs no data, fits the Orin NX budget with a wide margin, and every phase has a measurable pass/fail.
- Cost: the door/handle/can geometry priors are specific to the user's fridge; the first hardware session is tuning, not magic.
- Revisit: after ≥30 successful runs, record demonstrations (xr_teleoperate `--record` or LeRobot) and evaluate π0.5/GR00T fine-tuning on external compute for the manipulation phases only.

## D2 · Detector: YOLOE-26s with baked text prompts, YOLO26n COCO as cross-check
Classes: `refrigerator`, `refrigerator door handle`, `soda can`, `person`. Exported to TensorRT FP16 at 640 px on PC2.
- Why: ~5–10 ms/frame on Orin NX, boxes precise enough to seed depth lookups, prompts cover handle and can which COCO lacks. YOLO26n adds a second opinion on `refrigerator` and `person` for ≈4 ms.
- Cost: zero-shot accuracy on this specific fridge is unmeasured. The pipeline logs every frame so a few-shot YOLOE fine-tune (on the Jetson, minutes) can follow if precision is poor.
- Rejected: NanoOWL (slower on NX, larger, worse box precision), RF-DETR (4–5× slower, 4× memory), Grounding DINO family (not edge-real-time).

## D3 · VLM only as an optional phase verifier, never on the control path
Interface: OpenAI-compatible chat endpoint on PC2 (vLLM with Qwen3-VL-2B/4B-AWQ, or Moondream 2 server). Off by default.
- Why: 1–3 s per query and 2–5 GB memory; geometric checks (door-plane depth jump, hand closure state, wrist load) are faster and deterministic. The VLM adds robustness for ambiguous states (door ajar, wrong object in hand) once the basics work.

## D4 · Fridge search and approach use the fridge body, not the handle
The camera's 47.6° down-tilt means the handle and chest-height shelf are outside the RGB view beyond ~0.4 m. Search rotates in place in stop-and-look steps and accepts the fridge when the detector (both models agree or one is confident over 3 frames) sees its lower body; approach servos on the box center and on the depth-derived door plane; alignment squares the pelvis to the plane normal.
- Cost: needs the fridge within ~2.2 m of some point on the search circle; otherwise a one-step "explore" (walk 1 m forward, repeat) is attempted before aborting.

## D5 · Handle and can localization: detector first, geometry fallback, priors last
1. If the detector sees the handle/can at the standoff pose, use its box + median depth.
2. Else derive the handle from the door plane: free edge = lateral discontinuity of the plane at the hinge-opposite side, handle = edge + configured inset/protrusion, grasp height = configured height clamp to arm reach. The Mid-360 cloud (optional) confirms the protrusion.
3. Else use the configured prior (`fridge.handle_*`) and rely on the hook-grasp tolerance.
- Why: robust to the view-ceiling problem; priors are measured once with a tape measure.

## D6 · Door opening and closing are whole-body "hold and walk" moves, planned outside the swing wedge
Opening: (A) right hand hook-grasps the vertical handle from the standoff pose, the arm goes compliant (kp 25) and the
robot walks backward; the door follows the hand and the opening angle is read from the hand position (stops at 0.45 m
or 60°). (B) The robot re-positions 0.30 m in front of the free edge, squared to the panel at its current angle, hooks
the edge from the inner side and walks backward again to ≥ 95°. Closing: palm on the outer face near the free edge and
walk forward in 30° segments, re-facing the edge between segments, until the plane fit says the door is back.
Success checks: cavity depth jump > 0.3 m after (A), `min_open_deg` (70°) before stepping into the cavity front,
plane distance/yaw after closing.
- Why: a 0.44 m arm cannot swing a 0.6 m door from a fixed stance, and a robot standing 0.38 m from the door is inside
  the door's swing wedge for any angle above ~15°. Letting the legs provide the travel and the arm only the contact keeps
  every arm target within reach and every stance outside the wedge. DoorMan's lesson (move with the door) becomes low-gain
  tracking plus a tracking-error watchdog that stops the walk if the hand slips or the door jams.
- Cost: the exact hook/palm poses on the real handle and edge are geometric guesses to tune (`fridge.edge_standoff_m`,
  `hand.hook`, `arm.kp_compliant`).

## D7 · Whole-body control stays with the built-in loco controller (FSM 500 + arm_sdk)
No custom RL walking policy, no low-level `rt/lowcmd`.
- Why: teleop already used this mode with arms overriding via `rt/arm_sdk` weight 1.0; balance, stepping and arm blending are Unitree's problem. The arm action service must be disabled to avoid contention.

## D8 · Return by odometry, handover by person detection + wrist-load trigger
Home pose recorded at start from `rt/odommodestate`; return uses yaw-then-drive P control with a 0.15 m tolerance; then rotate to the detected `person`, approach to 0.8 m, extend the left arm, and release when the wrist-roll/elbow tracking error or `tau_est` rises above threshold for 0.3 s (the user pulling), with a VLM confirm if enabled and a hard timeout that keeps holding and asks for help.
- Why: no map, no SLAM; odometry drift over a 5 m round trip is a few centimetres of translation and a few degrees of yaw, which the person-servo absorbs.

## D9 · Loop rates and command expiry
Perception 10 Hz, loco commands re-sent every 100 ms with `duration = 0.3 s` (robot stops within 0.3 s if the process dies), arm targets streamed at 100 Hz with joint-velocity clipping, hand commands at 20 Hz. A deadband (|v| < 0.03 m/s, |ω| < 0.05 rad/s → 0) avoids shuffling.

## D10 · Host-side verification is CPU-only, target validation is on PC2
Unit tests (frames, IK, localization, state machine with mocks) run in the `g1fetch` conda env on this AMD64 workstation without GPUs. Detector quality, latency, camera extrinsics and every motion parameter are validated on the Jetson following `04_deploy_checklist.md`. No training job is run on this host (user constraint).

## D11 · Hand assignment (from the user's teleop): right hand opens the door, left hand takes the can
Configurable via `task.door_hand` / `task.can_hand`; the door hinge is assumed on the robot's left, handle on the right (`fridge.hinge_side: left`).
