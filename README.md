# g1-fridge-fetch

Autonomous "fetch a drink from the fridge" for a Unitree G1 Edu (23 DoF, BrainCo Revo2 hands), running
entirely on PC2 (Jetson Orin NX 16 GB): turn until the fridge is seen, walk up and square to the door,
open it with the right hand, take a can with the left, close the door, walk back, hand the can over.

Status (2026-09-26): research, design, code, host-side tests and the PC2 environment are done; the robot has only been
read from, **no motion has been commanded yet**.
Start with `docs/04_deploy_checklist.md` when the robot is available.

## Layout

```
docs/01_research.md         options considered (detectors, VLMs, VLAs, RL), platform facts, camera geometry
docs/02_decisions.md        what was chosen and why (D1..D11)
docs/03_architecture.md     frames, modules, state machine, timing, safety
docs/04_deploy_checklist.md PC2 setup, measurements, incremental hardware bring-up
configs/default.yaml        every tunable, with MEASURE markers for values to check on the robot
g1_fetch/                   the package (perception / control / skills / task / cli)
models/                     detector checkpoints (see models/README.md)
scripts/                    pc2_setup.sh, export_detector.py
tests/                      unit tests + an end-to-end dry run in a small ray-cast scene
data/                       captured frames and run logs (git-ignored content)
```

## Quick start (workstation, CPU only)

```bash
conda activate g1fetch                          # see below for how it was made
cd ~/junda/g1-fridge-fetch
python -m pytest tests -q                       # ~1-2 min, includes the dry run
python -m g1_fetch.cli geometry                 # camera view-ceiling table and arm reach
python -m g1_fetch.cli run --dry-run --sim              # whole task on the mock robot in the synthetic scene (~2 min)
python -m g1_fetch.cli detect data/samples --out /tmp/ann --set detector.device=cpu
```

`g1fetch` on the workstation is created by `scripts/host_setup.sh` (clone of `uni` + `requirements-host.txt`, CPU torch).
Import `pinocchio` before `torch` on this machine (libstdc++ ordering); `tests/conftest.py` does that.

## On the robot (PC2)

```bash
bash scripts/pc2_setup.sh                       # lean CUDA/TensorRT apt set, g1fetch env, Jetson torch, engines (see the script header)
conda activate g1fetch
sudo systemctl stop teleimager-realsense-webrtc # the teleop stream owns the camera; start it again when done
python -m g1_fetch.cli check   --config configs/pc2.yaml   # SDK, FSM, odometry, hands, camera, detector latency (read-only)
python -m g1_fetch.cli capture data/captures/fridge_1m -n 10 --config configs/pc2.yaml
python -m g1_fetch.cli run --step --until approach --config configs/pc2.yaml   # then open_door, fetch_can, close_door, full
```

PC2's LAN address is a DHCP lease on its USB Wi-Fi dongle (MAC `94:ba:06:f8:16:0d`); find it with
`ip neigh | grep 94:ba:06:f8:16:0d` after a ping sweep. The DDS interface to the mainboard is `enP8p1s0`.

The robot must be in the main controller (FSM 500, "ai" mode) with the operator holding the remote.
`--step` waits for Enter before each motion. Logs land in `data/runs/<timestamp>/`.

## Design in one paragraph

No demonstrations exist and the workstation GPUs are off-limits for training, so v1 is structured autonomy:
YOLOE-26s (open-vocabulary, prompts baked, ~5-10 ms on the Orin) plus a COCO YOLO26n cross-check find the
fridge, handle, can and person; RealSense depth gives the door plane and 3D targets in the pelvis frame; a
world-frame door model (hinge, normal, width) turns the handle, free edge and cavity into targets at any opening
angle; the built-in loco controller walks (velocity RPC with 0.3 s expiry) while `rt/arm_sdk` streams
IK solutions (pinocchio + CasADi, position + forearm direction, wrist roll for the palm); BrainCo hands
use open / hook / grasp poses with closure feedback. A VLM verifier is optional and off the control path.
The single biggest physical constraint is the head camera's 47.6° down-tilt: above ~0.9 m nothing is visible
beyond 0.5 m, so search and approach use the fridge body and handle/can localization falls back to geometry
and measured priors. See `docs/02_decisions.md`.
