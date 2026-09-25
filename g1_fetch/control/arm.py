"""Arm kinematics (pinocchio + CasADi) and the rt/arm_sdk streamer for the 23-DoF G1.

Joint conventions
-----------------
Per arm, 5 joints in motor order: shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll.
Motor indices: left 15..19, right 22..26, waist yaw 12, weight 29.
The end-effector frame `{L,R}_ee` is the wrist-roll joint frame translated by `robot.ee_offset_x`
along its x axis (forearm axis). Its origin and x direction do not depend on the wrist-roll angle,
so IK solves position (3) + forearm direction (2) with the four proximal joints and the wrist roll is
set analytically for the palm direction.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..config import resolve_path
from ..frames import rot_x

log = logging.getLogger(__name__)

SIDES = ("left", "right")
MOTOR_IDX = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}
WAIST_YAW = 12
WEIGHT_IDX = 29
JOINT_NAMES = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
             "left_elbow_joint", "left_wrist_roll_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint"],
}
LOCKED_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
]


@dataclass
class IKResult:
    q: np.ndarray            # 5 joints of the requested arm
    pos_err: float
    dir_err: float           # angle between achieved and requested forearm direction (rad)
    ok: bool


class ArmKinematics:
    """FK/IK for both arms in the pelvis frame (legs and waist locked at zero)."""

    def __init__(self, cfg):
        import pinocchio as pin  # must be imported before torch on the workstation (libstdc++)

        self.pin = pin
        self.ee_offset = float(cfg.robot.ee_offset_x)
        self.max_pos_err = float(cfg.arm.ik_max_pos_err)
        urdf = resolve_path(cfg, cfg.robot.urdf)
        full = pin.buildModelFromUrdf(str(urdf))
        lock_ids = [full.getJointId(n) for n in LOCKED_JOINTS]
        self.model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        self.data = self.model.createData()
        for side, tag in (("left", "L_ee"), ("right", "R_ee")):
            jid = self.model.getJointId(JOINT_NAMES[side][-1])
            self.model.addFrame(pin.Frame(tag, jid, pin.SE3(np.eye(3), np.array([self.ee_offset, 0.0, 0.0])),
                                          pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        self.fid = {"left": self.model.getFrameId("L_ee"), "right": self.model.getFrameId("R_ee")}
        # map arm joints -> configuration indices of the reduced model
        self.qidx = {s: [self.model.joints[self.model.getJointId(n)].idx_q for n in JOINT_NAMES[s]] for s in SIDES}
        self.lower = self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit
        self._build_ik()

    # ---- full configuration helpers ----
    def full_q(self, q_left, q_right) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[self.qidx["left"]] = q_left
        q[self.qidx["right"]] = q_right
        return q

    def fk(self, side: str, q5) -> np.ndarray:
        """4x4 pose of the ee frame in the pelvis frame for one arm (other arm at zero)."""
        q = np.zeros(self.model.nq)
        q[self.qidx[side]] = q5
        self.pin.framesForwardKinematics(self.model, self.data, q)
        M = self.data.oMf[self.fid[side]]
        T = np.eye(4)
        T[:3, :3] = M.rotation
        T[:3, 3] = M.translation
        return T

    def wrist_frame_rotation(self, side: str, q5) -> np.ndarray:
        return self.fk(side, q5)[:3, :3]

    # ---- IK ----
    def _build_ik(self):
        import casadi
        import pinocchio.casadi as cpin

        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        cq = casadi.SX.sym("q", self.model.nq)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, cq)
        self._fk_fun = {}
        for side in SIDES:
            M = self.cdata.oMf[self.fid[side]]
            self._fk_fun[side] = casadi.Function(f"fk_{side}", [cq], [M.translation, M.rotation[:, 0]])
        self._solvers = {}
        for side in SIDES:
            opti = casadi.Opti()
            var = opti.variable(self.model.nq)
            q0 = opti.parameter(self.model.nq)
            p_t = opti.parameter(3)
            x_t = opti.parameter(3)
            w_dir = opti.parameter(1)
            pos, xdir = self._fk_fun[side](var)
            cost = 100.0 * casadi.sumsqr(pos - p_t) + w_dir * casadi.sumsqr(xdir - x_t) \
                + 0.01 * casadi.sumsqr(var - q0) + 0.001 * casadi.sumsqr(var)
            # keep the other arm and this wrist roll exactly at q0 (they do not affect the objective)
            other = "right" if side == "left" else "left"
            for i in self.qidx[other] + [self.qidx[side][-1]]:
                opti.subject_to(var[i] == q0[i])
            opti.subject_to(opti.bounded(self.lower, var, self.upper))
            opti.minimize(cost)
            opti.solver("ipopt", {"expand": True, "print_time": False, "ipopt.sb": "yes", "ipopt.print_level": 0,
                                  "ipopt.max_iter": 100, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-5})
            self._solvers[side] = (opti, var, q0, p_t, x_t, w_dir)

    # extra seeds tried when the first solve is not within 5 mm
    EXTRA_SEEDS = (np.zeros(5), np.array([-0.5, 0.3, 0.0, 1.0, 0.0]), np.array([-1.2, 0.5, 0.3, 0.5, 0.0]),
                   np.array([0.3, 0.8, -0.5, 1.5, 0.0]), np.array([0.0, 1.4, 0.0, 1.5, 0.0]),
                   np.array([-0.9, 1.0, 0.8, 1.2, 0.0]), np.array([-1.6, 0.4, 1.5, 2.0, 0.0]))

    def ik(self, side: str, p_target, x_dir=None, q_seed=None, q_other=None, dir_weight: float = 1.0) -> IKResult:
        """Solve one arm for a target ee position and (optionally) forearm direction, both in the pelvis frame.

        Several seeds are tried until the position error is below 5 mm; the best solution is returned.
        """
        opti, var, q0, p_t, x_t, w_dir = self._solvers[side]
        other = "right" if side == "left" else "left"
        p_target = np.asarray(p_target, dtype=float)
        x_dir = np.array([1.0, 0.0, 0.0]) if x_dir is None else np.asarray(x_dir, dtype=float)
        x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-9)
        mirror = np.array([1.0, -1.0, 1.0, 1.0, 1.0]) if side == "right" else np.ones(5)
        seeds = ([np.asarray(q_seed, dtype=float)] if q_seed is not None else []) + [s * mirror for s in self.EXTRA_SEEDS]
        best: IKResult | None = None
        for sd in seeds:
            seed = np.zeros(self.model.nq)
            seed[self.qidx[side]] = sd
            if q_other is not None:
                seed[self.qidx[other]] = q_other
            opti.set_value(q0, seed)
            opti.set_value(p_t, p_target)
            opti.set_value(x_t, x_dir)
            opti.set_value(w_dir, dir_weight)
            opti.set_initial(var, seed)
            try:
                sol = opti.solve()
                q = np.array(sol.value(var)).ravel()
            except Exception:  # IPOPT infeasible/max-iter: take the best iterate
                q = np.array(opti.debug.value(var)).ravel()
            q5 = q[self.qidx[side]]
            T = self.fk(side, q5)
            pos_err = float(np.linalg.norm(T[:3, 3] - p_target))
            dir_err = float(math.acos(float(np.clip(T[:3, 0] @ x_dir, -1.0, 1.0))))
            res = IKResult(q5, pos_err, dir_err, pos_err <= self.max_pos_err)
            if best is None or res.pos_err < best.pos_err:
                best = res
            if res.pos_err < 0.005:
                break
        return best

    def wrist_roll_for_palm(self, side: str, q5, palm_dir_world, roll_inward: float = 0.0) -> float:
        """Wrist-roll angle so the palm normal best aligns with a world direction.

        Convention: at wrist roll = `roll_inward` (config robot.palm_inward_wrist_roll, measured on the
        robot) the palm faces the body midline, i.e. +y for the right hand and -y for the left, in the
        wrist frame. The palm normal at roll t is R_x(t - roll_inward) applied to that vector.
        """
        inward = np.array([0.0, -1.0, 0.0]) if side == "left" else np.array([0.0, 1.0, 0.0])
        q = np.array(q5, dtype=float)
        q[-1] = 0.0
        R0 = self.wrist_frame_rotation(side, q)          # wrist frame at roll 0
        d = np.asarray(palm_dir_world, dtype=float)
        d = d / (np.linalg.norm(d) + 1e-9)
        d_local = R0.T @ d
        lo = float(self.lower[self.qidx[side][-1]])
        hi = float(self.upper[self.qidx[side][-1]])
        best, best_val = 0.0, -2.0
        for theta in np.linspace(lo, hi, 121):
            val = float(d_local @ (rot_x(theta - roll_inward) @ inward))
            if val > best_val:
                best, best_val = theta, val
        return float(best)


class ArmStreamer:
    """Streams joint targets to rt/arm_sdk at a fixed rate with per-step velocity clipping.

    The built-in controller keeps balance; motors 15..19 / 22..26 follow our targets once the weight
    (motor 29 q) has been ramped to 1. Waist yaw is held at its measured value.
    """

    def __init__(self, cfg):
        self.cfg = cfg.arm
        self.rate = float(self.cfg.rate_hz)
        self.dt = 1.0 / self.rate
        self.max_step = float(self.cfg.max_joint_vel) * self.dt
        self.target = {s: None for s in SIDES}     # commanded joint targets (5,) per side
        self.cmd = {s: None for s in SIDES}        # last streamed (clipped) command
        self.weight = 0.0
        self.kp = {s: float(self.cfg.kp) for s in SIDES}
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None

    # ---- to be provided by subclasses ----
    def measured(self, side: str) -> np.ndarray:
        raise NotImplementedError

    def _write(self):
        raise NotImplementedError

    def _ready(self) -> bool:
        return True

    # ---- public API ----
    def start(self):
        t0 = time.time()
        while not self._ready():
            if time.time() - t0 > 5.0:
                raise RuntimeError("no rt/lowstate; is the robot on and the network interface right?")
            time.sleep(0.05)
        for s in SIDES:
            self.target[s] = self.measured(s).copy()
            self.cmd[s] = self.measured(s).copy()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="arm-streamer")
        self._thread.start()

    def enable(self, ramp_s: float | None = None):
        """Ramp the arm_sdk weight to 1 while holding the measured pose."""
        ramp = float(self.cfg.weight_ramp_s if ramp_s is None else ramp_s)
        with self._lock:
            for s in SIDES:
                self.target[s] = self.measured(s).copy()
                self.cmd[s] = self.measured(s).copy()
        n = max(1, int(ramp * self.rate))
        for i in range(n):
            with self._lock:
                self.weight = (i + 1) / n
            time.sleep(self.dt)

    def disable(self, ramp_s: float | None = None):
        ramp = float(self.cfg.weight_ramp_s if ramp_s is None else ramp_s)
        n = max(1, int(ramp * self.rate))
        for i in range(n):
            with self._lock:
                self.weight = 1.0 - (i + 1) / n
            time.sleep(self.dt)

    def set_target(self, side: str, q5):
        with self._lock:
            self.target[side] = np.asarray(q5, dtype=float).copy()

    def set_gain(self, side: str, kp: float):
        with self._lock:
            self.kp[side] = float(kp)

    def tracking_error(self, side: str) -> float:
        return float(np.max(np.abs(self.measured(side) - self.cmd[side])))

    def move_to(self, side: str, q5, duration: float, abort=None):
        """Linear joint interpolation from the current command to q5 over `duration` seconds."""
        q_from = self.cmd[side].copy()
        q_to = np.asarray(q5, dtype=float)
        n = max(1, int(duration * self.rate))
        for i in range(n):
            if abort and abort():
                return
            a = (i + 1) / n
            self.set_target(side, (1 - a) * q_from + a * q_to)
            time.sleep(self.dt)

    def wait_settled(self, side: str, tol: float = 0.05, timeout: float = 3.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if np.max(np.abs(self.measured(side) - self.target[side])) < tol:
                return True
            time.sleep(0.02)
        return False

    def _loop(self):
        next_t = time.time()
        while not self._stop:
            with self._lock:
                for s in SIDES:
                    if self.target[s] is None:
                        continue
                    delta = self.target[s] - self.cmd[s]
                    scale = np.max(np.abs(delta)) / self.max_step
                    self.cmd[s] = self.cmd[s] + (delta / scale if scale > 1.0 else delta)
                self._write()
            next_t += self.dt
            time.sleep(max(0.0, next_t - time.time()))

    def close(self):
        self._stop = True


class UnitreeArmStreamer(ArmStreamer):
    def __init__(self, cfg):
        super().__init__(cfg)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)
        self._q = None
        self._tau = None
        self._waist = 0.0
        self._state_lock = threading.Lock()
        self.mode_machine = None

    def _on_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(35)], dtype=float)
        tau = np.array([msg.motor_state[i].tau_est for i in range(35)], dtype=float)
        with self._state_lock:
            self._q = q
            self._tau = tau
            self.mode_machine = int(msg.mode_machine)

    def _ready(self) -> bool:
        with self._state_lock:
            if self._q is None:
                return False
            if self.mode_machine not in (1, 4):   # 1 = g1_23dof, 4 = g1_23dof_rev_1_0 (Unitree URDF table)
                log.warning("mode_machine=%s (expected 1 or 4 for a 23-DoF G1)", self.mode_machine)
            self._waist = float(self._q[WAIST_YAW])
            return True

    def measured(self, side: str) -> np.ndarray:
        with self._state_lock:
            return self._q[MOTOR_IDX[side]].copy()

    def measured_tau(self, side: str) -> np.ndarray:
        with self._state_lock:
            return self._tau[MOTOR_IDX[side]].copy()

    def _write(self):
        m = self.msg
        m.motor_cmd[WEIGHT_IDX].q = float(self.weight)
        # hold the waist yaw so it does not go limp under arm_sdk authority
        w = m.motor_cmd[WAIST_YAW]
        w.q, w.dq, w.tau, w.kp, w.kd = self._waist, 0.0, 0.0, float(self.cfg.kp), float(self.cfg.kd)
        for s in SIDES:
            for j, idx in enumerate(MOTOR_IDX[s]):
                c = m.motor_cmd[idx]
                c.q = float(self.cmd[s][j])
                c.dq = 0.0
                c.tau = 0.0
                if j == 4:
                    c.kp, c.kd = float(self.cfg.kp_wrist), float(self.cfg.kd_wrist)
                else:
                    c.kp, c.kd = self.kp[s], float(self.cfg.kd)
        m.crc = self.crc.Crc(m)
        self.pub.Write(m)


class MockArmStreamer(ArmStreamer):
    """Perfect tracking with a first-order lag, for dry runs and tests."""

    def __init__(self, cfg, q_init=None):
        super().__init__(cfg)
        self._q = {s: np.zeros(5) if q_init is None else np.asarray(q_init[s], dtype=float) for s in SIDES}
        self.disturbance = {s: np.zeros(5) for s in SIDES}   # external load, e.g. a person pulling
        self.writes = 0

    def measured(self, side: str) -> np.ndarray:
        return self._q[side] + self.disturbance[side]

    def _write(self):
        for s in SIDES:
            if self.cmd[s] is not None:
                self._q[s] = 0.7 * self._q[s] + 0.3 * self.cmd[s]
        self.writes += 1


def hand_orientation(forward, palm_dir):
    """Convenience: unit forearm direction and palm direction as numpy arrays."""
    f = np.asarray(forward, dtype=float)
    p = np.asarray(palm_dir, dtype=float)
    return f / np.linalg.norm(f), p / np.linalg.norm(p)
