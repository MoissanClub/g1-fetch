# 05 · Deployment learnings (what was wrong, what was assumed wrong, what to do instead)

Sessions: 2026-09-26 (PC2 bring-up) and 2026-10-02 (first motion runs). Written for the next person or agent
touching this robot. "Assumed" = something I took from docs or defaults without checking; "Did wrong" = my own error.

## A. Platform assumptions that were wrong

| # | Assumed | Reality | Consequence / fix |
|---|---|---|---|
| A1 | "JetPack 6.2 installed" means CUDA, cuDNN and TensorRT are present | PC2 had only the L4T 36.4.3 base (driver packages). No `/usr/local/cuda`, no `tensorrt` module | Detector ran at 2.5–5 s/frame on CPU. Lean fix: `cuda-libraries-12-6` (+nvtx, cupti), `libcudnn9-cuda-12`, TensorRT 10.3 runtime + `python3-libnvinfer`, `nvidia-l4t-dla-compiler` (1.8 GB) instead of the 6 GB `nvidia-jetpack` metapackage |
| A2 | DDS interface is `eth0` | It is `enP8p1s0` (192.168.123.164) | `configs/pc2.yaml` carries it; always pass `--config configs/pc2.yaml` |
| A3 | Low-rate odometry topic `rt/lf/odommodestate` exists (docs) | Only `rt/odommodestate` (~50 Hz) is published | `loco.odom_topic` config key |
| A4 | `mode_machine` = 1 for a 23-DoF G1 | It is 4 (`g1_23dof_rev_1_0`, per the Unitree URDF table) | accepted {1, 4} |
| A5 | Camera 1.18 m high, 47.6° down (URDF) | Floor-plane fit: 1.31 m, 48.7° down | Shoulder is 1.13 m above the floor; all height priors moved up; the camera still sees nothing above ~1.07 m beyond 0.5 m |
| A6 | Handle on the right edge, hinge left (my default, left unverified) | Handle on the LEFT edge, hinge RIGHT (door swings to the robot's right) | `fridge.hinge_side: right`; door model and tests follow the config |
| A7 | Manipulation stance: handle beside the right shoulder, 0.38 m away | User's teleop stance: handle centred in front of the body, ~0.45–0.47 m from the door | `approach.lateral_offset_m: 0`, `standoff_m: 0.45`; the arm reach there is marginal (0.44 m) → `ensure_reachable()` nudges forward |
| A8 | The built-in controller executes any small velocity | It ignores speeds below ~0.08 m/s; at the 0.05 m/s "crawl" the robot stood still until the approach timed out | `deadband_v/w: 0.08`, minimum crawl 0.10 m/s, tolerances 0.04 m |
| A9 | `SetVelocity` returns a reply quickly | In the task process the reply timed out (SDK error 3104) every call, blocking the loop 5 s per call; standalone the same RPC takes 2 ms | Root cause was **B2**, not the RPC. The no-reply variant (`_CallNoReply`) is kept because it cannot block the control loop |
| A10 | RPC right after `LocoClient.Init()` works | First requests race DDS discovery with the sport service → 3104 | 1 s settle + retries with a 1.5 s timeout in `UnitreeLoco.start()` |
| A11 | PyPI/conda-forge torch for aarch64 can use the Orin GPU | Both are CPU or SBSA/CUDA-13 builds; PyPI's `torch 2.11.0+cu130` even installs and then reports "driver too old" | Only NVIDIA's Jetson index (`jp6/cu126`), installed with `--no-deps --index-url` (no PyPI fallback); plus `nvidia-cudss-cu12` for `libcudss.so.0` |
| A12 | A TensorRT engine built once is good until TensorRT changes | After a reboot the first inference failed once (`cuTensor permutate execute failed`, CUDA stream-capture error) with a "different models of devices" warning; later calls work | Detector is built and warmed up **before** the RealSense opens, with one rebuild-and-retry; rebuilding the engines on the current boot clears the warning |
| A13 | The pip `pyrealsense2` wheel needs kernel patches on JetPack 6 | `pyrealsense2==2.58.4` works as is (RSUSB) once the V4L2 owner (`teleimager-realsense-webrtc.service`) is stopped | the run scripts stop/start that service |
| A14 | PC2's LAN address is stable | DHCP lease on a USB Wi-Fi dongle; it changed mid-session (…226 → …183) and a MacBook on the robot's own AP confused the picture | find it by MAC `94:ba:06:f8:16:0d` (`ip neigh` after a ping sweep) |
| A15 | Blind "explore 1 m forward" is a reasonable search fallback | In an office it is a blind walk into chairs | disabled (`explore_step_m: 0`); run 3 was interrupted just before it |

## B. Things I did wrong

| # | What I did | Why it was wrong | What to do |
|---|---|---|---|
| B1 | Measured detector latency standalone (51 ms) and assumed the loop would run at 10 Hz | In the full process the same detector took 330–460 ms and the plane fit 3.5 s; the loop ran at 0.2 Hz, so each 0.5 s velocity command expired long before the next one (jerky, tiny turns; "it barely moves") | **Benchmark the whole stack** (`cli loopbench`) before any motion run |
| B2 | Subscribed to `rt/lowstate` (500 Hz) with a Python callback that built numpy arrays, plus a 100 Hz arm streamer, 2×20 Hz hands, 50 Hz odometry | cyclonedds-python deserializes every message in Python under the GIL (~1.5 ms each → most of a core at 500 Hz). Perception, RANSAC and even the RPC reply thread starved. The robot's balance does not need any of this; I never needed 500 Hz | Poll the newest lowstate sample at the 50 Hz stream rate (no handler); all other callbacks only store the message; parse on demand. Arm stream 50 Hz, hands 10 Hz |
| B3 | Stop-and-look search (turn 0.6 rad, settle 0.8 s, 7 frames) | ~8 s per step, 90 s per sweep, and the fridge beyond 1.5 m is a thin strip at the top of the tilted image scoring 0.09 (< 0.25 threshold) → it rotated past the fridge twice | Continuous rotation at 0.4 rad/s with detection every frame, threshold 0.15 + 3 consecutive frames + a stationary confirmation; found the fridge in 5.7 s |
| B4 | RANSAC on every valid pixel of a full-frame box (34k points, 200 iterations) | Fine on the workstation (7 ms), disastrous on a GIL-contended Orin | Cap at ~2000 points, 60 iterations (`box_cloud(max_points=…)`) |
| B5 | Approach retry started from "no plane known" and declared the fridge lost after 10 frames when the door filled the view | The detector often does not fire on a wall of white; the plane fit does | On a miss at close range (or on a retry), fit the whole frame first and only then declare "lost" |
| B6 | `check` opened the camera before building the TensorRT detector | Combined with A12 this produced the first-run CUDA failure | Detector first, then camera, always |
| B7 | Allowed PyPI as `--extra-index-url` while installing torch from the Jetson index | pip resolved the same version number from PyPI (wrong build) | Jetson index only, `--no-deps` |
| B8 | Started the `nvidia-jetpack` metapackage install, then killed `apt-get` mid-way to go lean | dpkg kept unpacking the whole set; needed `dpkg --configure -a`, `apt-mark auto/manual`, `autoremove --purge` to end up lean | Decide the package list first; never kill apt during unpack |
| B9 | `pkill -f "<pattern>"` where the pattern also matched my own SSH command line | Killed my own shell (exit 144), and once the patch that followed never ran | Use `pkill -f` only with patterns that cannot match the caller, or kill by PID |
| B10 | `sudo -v` in one shell, expecting the credential to carry into a `nohup` background shell | Different session; the background job stalled on a password prompt | `echo <pw> \| sudo -S …` inside the same background command (or configure sudoers) |
| B11 | Deleted a directory of downloaded models because I thought they were in the wrong place | They were the only copies; re-downloaded 250 MB | Look before deleting (`ls`), always |
| B12 | Assumed `--step` mode (press Enter between motions) would be usable | Over a non-interactive SSH session it cannot be | Use `--until PHASE` for scope, and SIGINT (`pkill -INT`) for a clean stop that runs `safe_stop()`; SIGTERM from `timeout` skips the `finally` block |
| B13 | Wrote a tilt analysis predicting the fridge is invisible beyond ~2.4 m, then still let the first search assume a 0.25 threshold | The analysis was right; I did not carry it into the search parameters | When the geometry says a target will be weak, lower the acceptance threshold and demand persistence instead |

## C. Verified good (keep)
- Lean GPU stack + Jetson-index torch 2.11: detector 51 ms standalone, 60–90 ms on camera frames once the GIL load is gone.
- RealSense through pip `pyrealsense2`; aligned RGB-D 640×480 at 30 fps; floor-plane self-calibration of camera height/tilt.
- Continuous search: fridge found in 5.7 s on run 4.
- Approach reached 0.47 m, squared within 0.03 rad, using the full-frame plane fit at close range.
- arm_sdk enable/home pose works in FSM 500 (arms moved to `arm.home_q` and were handed back on exit).
- `cli check` (read-only), `cli loopbench` (zero-velocity), `cli capture`, run logs with annotated frames in `data/runs/`.

## D. Still unverified (next session)
1. Lateral servo to the user's stance (handle centred, 0.45 m) and the right-hand hook grasp at that reach.
2. Pull-and-back door opening; `fridge.pull_back_m`, `arm.kp_compliant`; door dynamics.
3. Loop rate under walking load. After B2/B4 `cli loopbench` (standing, full stack) gives camera 12 ms, detectors 92 ms, plane 65 ms → ~6 Hz; the 0.5 s command expiry has margin, but verify while walking.
4. Whether a 25 W / MAXN power mode is worth it (15 W, 4 cores online today).
5. Engine rebuild on the current boot to clear the "different models of devices" warning.
