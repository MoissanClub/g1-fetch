# 07 · Notes on VISTA (MIT, 2026) and what it means for the fridge-fetch robot

Source: https://vista-research.github.io/ , arXiv:2610.02200 (tech report, 2026-10-01; blog 2026-08-05),
code https://github.com/joshhhhhan/VISTA (MIT license). Authors: Qiushi Han, Keya Hu, Linlu Qiu, Cathy Wu, Kaiming He (MIT).
Read on 2026-10-04. Numbers below are quoted from the report; interpretations are marked (I).

## 1. What it is
A thin "visual harness" around an unmodified multimodal model (Claude Opus 5.0 via Claude Code, GPT-5.6 Sol via
Codex CLI) for interactive environments. Three components:
1. **Visual observations**: the environment is shown as rendered PNGs (64×64 grids upscaled 8× to 512 px), not as text.
2. **Lossless visual memory**: every returned frame, including animation frames, is archived at full resolution and
   indexed by (turn, frame); it never leaves the model's reach even when the context window is reset.
3. **Visual inspection tools**: `inspect(question, views)` returns enlarged crops of any region of any past frame
   (spatial and temporal re-attention); `read_pixels(question, views)` returns a grid of RGB values for a region.
   Plus `play(action)`, `history`, and two note files: `GUIDE.md` (persistent knowledge of the game) and
   `WORKING.md` (current-level scratchpad). On context overflow the agent writes a compact checkpoint and resumes
   in a fresh context with the notes, the action history and the current frame.

## 2. Results
- ARC-AGI-3 (25 public games, 183 levels, scored by Relative Human Action Efficiency): Opus 5.0 goes from 40.68 to
  **100.00 RHAE**, all levels, with **57.4 % fewer actions than first-time humans** (7,302 vs 17,135). GPT-5.6 Sol: 98.27.
  Frontier systems without a harness scored below 1 % when the benchmark launched (March 2026).
- Ablation (GPT-5.6 Sol, 200K context): text grids 13.33 → PNG 47.32 → raised action limits 51.66 → continuous
  conversation 65.82 → note files 70.05 → **visual memory + inspect 94.10** → + read_pixels 99.00.
- Images vs text grids: same RHAE (99.0 vs 98.2) but **58 % fewer tokens** (30.7 M vs 71.9 M per game).
- Context size: 100K costs 0.4 RHAE and halves tokens; 780K costs 4× tokens for no gain. Image scale: 8× (512 px) optimal.
- 3D renderings of the same games: 84.12 RHAE with ~3× the actions (20,640) — still works, much less efficient.
- Open-weight GLM-5.3 Flash: 13.33 → 66.93 with VISTA.
- Transfers with minimal adaptation: GameWorld SR 38 → 72 %, AI GameStore 58 → 127 (human median = 100),
  BabyVision 41.0 → 63.2 %.
- Cost: ~30 M tokens per ARC game; wall time not reported.

## 3. The authors' conclusions
- Capability was already in the models; the harness elicits it. "Whatever falls outside the window gets compressed
  or lost" in ordinary VLM use; decoupling encoding from retrieval fixes that.
- Free-form language reasoning beats program synthesis for rule discovery (a jumping mechanic needs ~4,000 lines of
  Python but "a page of notes"); "much of the reasoning that leads to a good action can be carried out in a fuzzy manner".
- Caveats they state: public games may be in training data (the private set is the real test); no cross-game learning;
  future work explicitly names embodied tasks and robotic control.

## 4. Critical reading (I)
- The environments are turn-based, deterministic, fully observable per frame, with small discrete action sets and
  free retries. None of that holds for a walking robot: continuous actions, noisy partial observations, irreversible
  outcomes, and the world does not wait ~minutes for a decision.
- 30 M tokens per game at "xhigh" effort is a research budget, not a control loop budget.
- The largest single ablation gain is **memory + inspect (+24 RHAE)**, not model choice or prompting. That is the
  transferable lesson: give the model lossless access to its own history and a zoom tool.
- The 3D result (84 vs 99) is the honest preview of the physical world: perspective, occlusion and clutter cost
  3× the interaction. Geometry tools (the robotic analogue of `read_pixels`) will matter more, not less.

## 5. What to apply to the G1 fridge-fetch stack
Two-tier split, which is already how the project is shaped: a fast reactive loop (detector + depth + servo at ~6 Hz)
and a slow deliberative layer. VISTA tells us how to build the slow layer well.

1. **Lossless frame archive with inspect/zoom.** `data/runs/<ts>/` already keeps every 5th annotated JPEG and the
   events. Keep every frame (640×480 JPEG ≈ 30 KB × 6 Hz ≈ 180 KB/s, trivial) plus depth on events, indexed by tick
   and phase, and expose `inspect(tick, region, zoom)` and `read_depth(tick, region)` to a VLM. The handle at the top
   rows of the image, the 0.09-confidence fridge strip, the door angle after a pull: all are questions a 13× zoom on an
   earlier frame answers. My manual re-scoring of run frames at conf 0.05 was exactly this, done by hand.
2. **Skills as the action vocabulary.** The VLM should not emit velocities; it should call `play(skill, params)` over
   our phases (search, approach, grasp_handle, pull_and_back, …) and read back the event log, like VISTA's discrete
   `play`. The reactive loop stays as it is. Decisions then cost seconds per phase, not per tick, which the token
   budget and the robot's patience both allow.
3. **GUIDE.md / WORKING.md, literally.** `docs/05_deploy_learnings.md` and `configs/pc2.yaml` are a GUIDE.md written
   by hand ("handle on the left edge", "controller ignores < 0.08 m/s", "camera sees nothing above 1.07 m at 0.45 m").
   Let the agent maintain a robot-and-fridge GUIDE.md across runs and a WORKING.md per attempt (door model, odometry
   home pose, what was tried). Checkpoint it the way VISTA does so a crash or context reset resumes mid-task.
4. **Verify with history, not a single frame.** The VLM verifier currently gets one frame and a yes/no question.
   Give it before/after frames and the zoom tool ("compare tick 120 and tick 410 at the door edge; is the door open
   wider?"), which is what the ablation says makes the difference.
5. **Measure relative human efficiency.** The user's teleop run is the human baseline. Report wall time, number of
   skill invocations and corrective actions per phase against it; it is a better progress metric than pass/fail.
6. **Images over text, at a sane resolution.** Feed the VLM 512–640 px frames, not detection lists; the detections
   are a hint, the pixels are the evidence. Don't go above 640 px per frame for the deliberative layer.
7. **Fuzzy reasoning where it is cheap, code where it must be exact.** Rule/affordance discovery ("this door needs
   ~0.4 m of pull before the gap shows") belongs to the model's notes; the arc kinematics and safety limits stay in code.
8. **Exploration has a price in the real world.** VISTA explores freely; a robot must explore with reversible,
   bounded primitives (the compliant pull with a tracking-error watchdog is one). Simulation (`tests/sim_scene.py`)
   is where free exploration belongs.

What not to copy: per-tick model calls, program-synthesis of the whole task, or expecting 100 % from a harness alone
on real hardware. The 3D ablation is the realistic expectation, and geometry sensors are our `read_pixels`.
