"""The fetch-a-drink task: phases, retries, and safe termination."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .robot import Aborted, Robot
from .skills import approach, close_door, grab_can, open_door, return_handover, search
from .skills.common import SkillFailed, go_home, wait_user

log = logging.getLogger(__name__)

PHASES = ("search", "approach", "open_door", "fetch_can", "close_door", "return", "handover")


@dataclass
class TaskState:
    home: object = None
    fridge_det: object = None
    appr: object = None
    door: object = None
    phase: str = ""
    history: list = field(default_factory=list)


class FetchDrinkTask:
    def __init__(self, rb: Robot, start_phase: str = "search", stop_after: str | None = None):
        self.rb = rb
        self.state = TaskState()
        self.start_phase = start_phase
        self.stop_after = stop_after

    # ---- phases ----
    def _search(self):
        self.state.fridge_det, _ = search.search(self.rb, "fridge")

    def _approach(self):
        self.state.appr = approach.approach(self.rb)

    def _open_door(self):
        self.state.door = open_door.open_door(self.rb, self.state.appr)

    def _fetch_can(self):
        grab_can.fetch_can(self.rb, self.state.door)

    def _close_door(self):
        close_door.close_door(self.rb, self.state.door)

    def _return(self):
        return_handover.return_home(self.rb, self.state.home)
        return_handover.find_person(self.rb)

    def _handover(self):
        return_handover.handover(self.rb)

    def _prepare(self):
        rb = self.rb
        rb.log.event("task_start", dry_run=rb.dry_run, cfg=rb.cfg.to_dict())
        self.state.home = rb.loco.pose()
        rb.arms.enable()
        for side in ("left", "right"):
            rb.hands[side].open(wait=False)
            go_home(rb, side, duration=2.5)
        wait_user(rb, f"arms at home, pose {self.state.home}; start the task")

    def run(self) -> bool:
        rb = self.rb
        retries = int(rb.cfg.task.max_retries)
        fns = {"search": self._search, "approach": self._approach, "open_door": self._open_door,
               "fetch_can": self._fetch_can, "close_door": self._close_door, "return": self._return,
               "handover": self._handover}
        ok = False
        try:
            self._prepare()
            started = False
            for phase in PHASES:
                if phase == self.start_phase:
                    started = True
                if not started:
                    continue
                self.state.phase = phase
                self._run_phase(phase, fns[phase], retries)
                if self.stop_after == phase:
                    break
            ok = True
            rb.log.event("task_done")
        except Aborted:
            rb.log.event("task_aborted", phase=self.state.phase)
        except SkillFailed as e:
            rb.log.event("task_failed", phase=self.state.phase, reason=str(e))
            log.error("task failed in %s: %s", self.state.phase, e)
        except KeyboardInterrupt:
            rb.log.event("task_interrupted", phase=self.state.phase)
        finally:
            rb.safe_stop(release_arms=False)
        return ok

    def _run_phase(self, name: str, fn, retries: int):
        rb = self.rb
        for attempt in range(retries + 1):
            t0 = time.time()
            rb.log.event("phase_start", phase=name, attempt=attempt)
            try:
                fn()
                rb.log.event("phase_done", phase=name, seconds=time.time() - t0)
                self.state.history.append((name, attempt, "ok"))
                return
            except SkillFailed as e:
                rb.loco.stop()
                rb.log.event("phase_failed", phase=name, attempt=attempt, reason=str(e))
                self.state.history.append((name, attempt, str(e)))
                if attempt >= retries or name in ("handover",):
                    raise
                log.warning("retrying %s after: %s", name, e)
                time.sleep(1.0)
