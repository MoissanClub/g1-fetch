"""Locomotion through the built-in controller (LocoClient RPC) plus odometry (rt/lf/odommodestate)."""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

from ..frames import Pose2D, wrap_angle

log = logging.getLogger(__name__)


class LocoError(RuntimeError):
    pass


@dataclass
class VelCmd:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0


class LocoBase:
    """Shared clipping/deadband logic; subclasses implement _send and pose()."""

    def __init__(self, cfg):
        self.cfg = cfg.loco
        self.last = VelCmd()

    def _clip(self, vx: float, vy: float, yaw: float) -> VelCmd:
        c = self.cfg
        vx = max(-c.max_vx, min(c.max_vx, vx))
        vy = max(-c.max_vy, min(c.max_vy, vy))
        yaw = max(-c.max_yaw, min(c.max_yaw, yaw))
        if abs(vx) < c.deadband_v:
            vx = 0.0
        if abs(vy) < c.deadband_v:
            vy = 0.0
        if abs(yaw) < c.deadband_w:
            yaw = 0.0
        return VelCmd(vx, vy, yaw)

    def set_velocity(self, vx: float, vy: float = 0.0, yaw: float = 0.0):
        cmd = self._clip(vx, vy, yaw)
        self.last = cmd
        self._send(cmd)

    def stop(self):
        self.last = VelCmd()
        self._send(VelCmd())

    def _send(self, cmd: VelCmd):
        raise NotImplementedError

    def pose(self) -> Pose2D:
        raise NotImplementedError

    def start(self):
        pass

    def close(self):
        self.stop()

    # ---- blocking helpers used by the skills ----
    def turn_by(self, angle: float, rate: float | None = None, timeout: float = 20.0, abort=None) -> float:
        """Rotate in place by `angle` (rad, +ccw) using odometry yaw. Returns the achieved rotation."""
        rate = rate or self.cfg.search.yaw_rate
        start = self.pose().yaw
        target = start + angle
        t0 = time.time()
        period = 1.0 / self.cfg.rate_hz
        while True:
            if abort and abort():
                self.stop()
                raise LocoError("aborted")
            err = wrap_angle(target - self.pose().yaw)
            if abs(err) < 0.06:
                break
            w = max(-rate, min(rate, 1.5 * err))
            w = math.copysign(max(abs(w), self.cfg.deadband_w + 0.02), w)
            self.set_velocity(0.0, 0.0, w)
            if time.time() - t0 > timeout:
                self.stop()
                raise LocoError("turn_by timeout")
            time.sleep(period)
        self.stop()
        return wrap_angle(self.pose().yaw - start)

    def drive_to(self, wx: float, wy: float, wyaw: float | None, tol_pos: float, tol_yaw: float,
                 k_pos: float, k_yaw: float, timeout: float = 60.0, abort=None, obstacle=None):
        """Drive to a world (odometry-frame) pose. First face the goal, then translate, then face `wyaw`."""
        t0 = time.time()
        period = 1.0 / self.cfg.rate_hz
        while True:
            if abort and abort():
                self.stop()
                raise LocoError("aborted")
            if time.time() - t0 > timeout:
                self.stop()
                raise LocoError("drive_to timeout")
            p = self.pose()
            bx, by = p.world_to_body(wx, wy)
            dist = math.hypot(bx, by)
            if dist < tol_pos:
                break
            heading = math.atan2(by, bx)
            if abs(heading) > 0.35 and dist > 0.3:
                w = max(-self.cfg.max_yaw, min(self.cfg.max_yaw, k_yaw * heading))
                self.set_velocity(0.0, 0.0, w)
            else:
                if obstacle and obstacle():
                    self.stop()
                    raise LocoError("obstacle ahead")
                vx = k_pos * bx
                vy = k_pos * by
                # never let the deadband swallow the last centimetres: keep a minimum crawl speed
                v_min = self.cfg.deadband_v + 0.02
                norm = math.hypot(vx, vy)
                if norm < v_min:
                    vx, vy = vx / max(norm, 1e-9) * v_min, vy / max(norm, 1e-9) * v_min
                w = k_yaw * heading if dist > 0.3 else 0.0
                self.set_velocity(vx, vy, w)
            time.sleep(period)
        self.stop()
        if wyaw is not None:
            self.turn_by(wrap_angle(wyaw - self.pose().yaw), timeout=timeout / 2, abort=abort)


class UnitreeLoco(LocoBase):
    """Real robot. Requires ChannelFactoryInitialize() to have been called."""

    def __init__(self, cfg):
        super().__init__(cfg)
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        self.client = LocoClient()
        self.client.SetTimeout(5.0)
        self.client.Init()
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._pose_t = 0.0
        self._lock = threading.Lock()
        topic = str(cfg.loco.get("odom_topic", "rt/odommodestate"))
        self._sub = ChannelSubscriber(topic, SportModeState_)
        self._sub.Init(self._on_odom, 10)
        self._required = set(int(v) for v in cfg.loco.required_fsm_ids)

    def _on_odom(self, msg):
        with self._lock:
            self._pose = Pose2D(float(msg.position[0]), float(msg.position[1]), float(msg.imu_state.rpy[2]))
            self._pose_t = time.time()

    def start(self):
        t0 = time.time()
        while self._pose_t == 0.0:
            if time.time() - t0 > 5.0:
                raise LocoError("no odometry received (State Estimator service >= 1.0.0.1 required; topic loco.odom_topic)")
            time.sleep(0.05)
        code, fsm = self.client.GetFsmId()
        if code != 0:
            raise LocoError(f"GetFsmId failed: {code}")
        if int(fsm) not in self._required:
            raise LocoError(f"robot FSM id is {fsm}; expected one of {sorted(self._required)} (main controller running)")
        log.info("loco ready, fsm_id=%s pose=%s", fsm, self._pose)

    def _send(self, cmd: VelCmd):
        code = self.client.SetVelocity(cmd.vx, cmd.vy, cmd.yaw, float(self.cfg.cmd_duration_s))
        if code != 0:
            log.warning("SetVelocity returned %s", code)

    def stop(self):
        self.last = VelCmd()
        for _ in range(2):
            self.client.SetVelocity(0.0, 0.0, 0.0, 0.5)

    def pose(self) -> Pose2D:
        with self._lock:
            if time.time() - self._pose_t > 0.5:
                log.warning("odometry stale (%.2fs)", time.time() - self._pose_t)
            return Pose2D(self._pose.x, self._pose.y, self._pose.yaw)

    def fsm_id(self) -> int:
        code, fsm = self.client.GetFsmId()
        return int(fsm) if code == 0 else -1


class MockLoco(LocoBase):
    """Kinematic integrator for dry runs and tests; time advances with the wall clock."""

    def __init__(self, cfg, pose: Pose2D | None = None):
        super().__init__(cfg)
        self._pose = pose or Pose2D(0.0, 0.0, 0.0)
        self._t = time.time()
        self.commands: list[VelCmd] = []

    def _integrate(self):
        now = time.time()
        dt = now - self._t
        self._t = now
        c = self.last
        p = self._pose
        wx, wy = p.body_to_world(c.vx * dt, c.vy * dt)
        self._pose = Pose2D(wx, wy, wrap_angle(p.yaw + c.yaw * dt))

    def _send(self, cmd: VelCmd):
        self._integrate()
        self.commands.append(cmd)

    def pose(self) -> Pose2D:
        self._integrate()
        return Pose2D(self._pose.x, self._pose.y, self._pose.yaw)
