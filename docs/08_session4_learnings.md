# 08 · Session 4 learnings (2026-10-16, home fridge, two cameras)

Scope of the session: first attempts at the complete task on the user's home fridge with the SR300 added as a
level top camera. Eleven runs, each one phase further than the last. Numbers below are from the run logs in
`data/runs/` on PC2; the best run is `20261016-014645`.

## 1. Outcome in one table

| phase | status at end of session | evidence |
|---|---|---|
| search | reliable, 2–8 s | head camera finds the fridge; a white wall filling the frame is rejected by the D435i cross-check |
| approach | reliable, 4–9 s | stance 0.42 ± 0.06 m, squared within 0.03 rad |
| handle localization | reliable from depth | plate found as points 5–6 cm proud of the door plane, 300–400 points, 3-frame median |
| hook grasp | works, ~50 % first try | closure 0.62–0.75 of a 0.90 command when hooked; misses retried in place |
| first pull | works every time once hooked | 0.45 m retreat, door to 36–47°, depth jump 0.8–1.15 m |
| second pull | worked once | re-hooked the swung handle, door to 67° |
| reach-in | failed every time | stance 0.2 m too far (odometry drift), no can detected, prior out of reach |
| close door, return, handover | never reached | — |

## 2. What the session established (facts, not guesses)
- The home fridge has a white flat plate handle on the left edge of the lower door, standing 5–6 cm proud of the
  door (depth measurement), with a finger gap behind it; hinge on the right; door ≈ 0.74 m wide. RGB detectors never
  fire on it (white on white); depth finds it immediately.
- The hook that works: hand centre 1.5 cm to the hinge-far side of the plate, fingertips ~2.5 cm into the gap behind
  it, close to the hook pose. Entering 6.5 cm beside the plate and sliding onto it at normal stiffness is tolerant to
  ±2 cm localization error; sliding under compliant gains stalled short of the plate and missed four times.
- A real grasp shows in the finger feedback: the fingers stop at 0.62–0.75 when the plate is between them and reach
  the commanded 0.90 when it is not. This single number is the most reliable sensor in the whole pipeline.
- The built-in controller ignores creep speeds up to ~0.10 m/s; 0.14 m/s moves. The robot cannot be positioned to
  better than ~5 cm by velocity commands, so every tolerance below that stalls.
- Odometry drifts noticeably over two pull-and-back walks (~1 m total with the door's reaction force): the modelled
  cavity was 0.2 m off when the robot went to reach in.
- The SR300 drops off USB under load (several reconnects per hour); opening it immediately after a reconnect fails.
- Unitree TTS works for status; English is speaker id 1.

## 3. Mistakes I made, and the rule each one taught
1. **Reading absence into blind frames.** I declared "no handle" from frames in which the handle was above the
   D435i's view ceiling and outside the head camera's crop, switched the grasp to a seam hook, and lost a run.
   Rule: before concluding something is not there, check the camera could have seen it; otherwise ask one precise
   question.
2. **Trusting a detector label over geometry.** The head camera's "handle" boxes were the seam between the doors.
   Rule: for white-on-white parts, depth protrusion beats RGB; when two sources disagree by more than their error,
   prefer the geometric one and log both.
3. **Grasp verification against the wrong reference.** A hook that closed to its command was counted as a grasp
   and the robot "pulled" air. Rule: compare closure to the commanded value, not to 1.0.
4. **Absolute thresholds on a signal with a baseline.** The pull watchdog tripped on the pre-existing tracking error of
   a hand pressed against the plate. Rule: watch the change from the baseline at move start.
5. **Retrying a phase after its goal was already reached.** A failed stage B re-ran the grasp on a handle that had
   swung away. Rule: phases record sub-goals; a verified sub-goal is never redone.
6. **Planning a stance from a model built a metre of walking earlier.** Rule: re-measure from live depth right before
   any reach; odometry is for getting roughly there.
7. **Compliant gains for a motion that must complete.** The scoop slide needs the plate to stop it, so it needs
   stiffness; compliance belongs to the pull, where the door must lead.
8. **Tolerances tighter than the actuator.** 1.5 cm drive tolerances and 0.05–0.10 m/s creep speeds produced
   timeouts and "stuck in front of the door". Rule: tolerance ≥ what one minimum-speed burst moves.
9. **`pkill -f` with a pattern that matched my own shell**, three times in one day. Rule: kill by PID, excluding
   `$$` and `$PPID`.
10. **Camera open order and USB flakiness treated as software bugs.** Rule: open cameras once, settle, retry, fall
    back; and check the cable.

## 4. Design insights worth keeping
- Two cameras with a clear split (level RGB for search and bearings at range, tilted RGB-D for everything metric
  within 1.2 m) is the right architecture for this head geometry. The cross-check between them removed a whole
  class of false positives.
- "Hold and walk" door opening is robust: the legs provide the travel, the compliant arm only keeps contact, the
  door angle is read from the hand. Two pulls got 67° on a stiff home fridge.
- Closure feedback + in-place retries turned a 50 % grasp into a reliable phase; the same pattern (verify from the
  sensor that would show failure, retry locally, then re-localize) should wrap every contact step.
- Spoken status (phase names, warnings, errors) was immediately useful to the person standing next to the robot;
  it also exposed how often the loop was silently stuck.
- The whole loop ran at ~3–6 Hz with both detectors; adequate because commands expire in 0.5 s, but the second
  camera's detector pass is the largest single cost.

## 5. Open problems for the next session, in priority order
1. **Reach-in stance from live depth.** After the door is open, fit the cabinet/opening geometry from the D435i
   (back wall plane and the rim) and servo to a measured 0.30 m, ignoring the odometry-based door model for
   position. Alternatively re-detect the fridge's lower front with the detector and reuse the approach servo.
2. **Can detection inside the cavity.** The shelf is above the D435i's view ceiling until ~0.3 m; use the head camera
   to find the can and intersect its ray with the back-wall plane (depth) or with the shelf height, then confirm
   with the D435i once close.
3. **Visual servo of the hand for the final 10 cm.** The D435i sees the hand and the plate at the stance; close the
   loop on the measured hand-to-plate offset instead of relying on the scoop.
4. **Closing the door and the return/handover phases** have never run on hardware; they will have their own
   surprises (door swing geometry, person detection range).
5. **Simulator fidelity.** The dry-run scene reproduces the first pull but not the second; keep it for regression of
   the logic, do not expect it to predict door physics.
6. **Hardware:** SR300 cable/hub; consider a global-shutter USB camera as the head camera.

## 6. Numbers to carry forward (home fridge, this robot)
| item | value |
|---|---|
| stance distance (pelvis → door plane) | 0.42 m (0.38–0.48 accepted) |
| handle plate protrusion (depth) | 0.05–0.06 m |
| hook offset beside the plate | 0.015 m; scoop entry 0.065 m |
| fingertip depth behind the plate face | ~0.025 m |
| hook closure when hooked / when empty | 0.62–0.75 / 0.90 |
| pull: retreat, speed, result | 0.45 m, 0.12 m/s, 36–47° first pull, 67° after the second |
| controller minimum effective speed | ~0.14 m/s |
| head camera roll (measured from frames) | −15° |
| loop rate with two detectors | ~3–6 Hz |
