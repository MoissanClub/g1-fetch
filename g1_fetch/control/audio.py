"""Spoken status through the G1's speaker (Unitree AudioClient TTS) plus a log-to-speech bridge.

Design: one worker thread drains a small queue; high-priority phrases (task/phase status) are kept in order,
low-priority ones (log warnings/errors, selected events) are coalesced so that only the newest waits. Everything
is rate limited and de-duplicated; speech never blocks the control loop and never fails the task.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time

log = logging.getLogger(__name__)

# EVENT names worth hearing, with short phrasing (None = say the event name)
EVENT_PHRASES = {
    "fridge_found": "fridge found",
    "approach_done": "at the door",
    "handle_bar_depth": "handle located",
    "handle_detected": "handle located",
    "handle_prior": "using handle prior",
    "nudge_forward": "stepping closer",
    "handle_grasped": "handle grasped",
    "pull_done": "door pulled",
    "verify_open": None,
    "walk_with_door_done": "door moved",
    "can_detected": "can located",
    "can_prior": "using can prior",
    "can_grasped": "can grasped",
    "can_carried": "can secured",
    "door_closed": "door closed",
    "person_found": "person found",
    "person_reached": "at the person",
    "handover_released": "released",
    "handover_timeout": "nobody took the can",
    "phase_failed": "phase failed",
    "search_crosscheck": None,
}
_SKIP_EVENTS = {"task_start", "phase_start", "phase_done", "task_done", "task_failed", "task_aborted",
                "task_interrupted", "door_model", "face_edge", "return_home", "reach_pose", "top_edge_grip",
                "verify_closed", "close_push_done", "handover_extended", "search_explore", "sweep_done",
                "hook_sweep_done"}


def _shorten(text: str, limit: int = 70) -> str:
    text = re.sub(r"[\[\]{}()\"']", " ", text)
    text = re.sub(r"/\S+", "", text)                                  # paths
    text = re.sub(r"(-?\d+\.\d\d)\d+", r"\1", text)                   # 2 decimals
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


class _SpeakerBase:
    def __init__(self, min_interval_s: float = 2.0, dedup_s: float = 10.0):
        self.min_interval = min_interval_s
        self.dedup_s = dedup_s
        self.said: list[str] = []
        self._hi: queue.Queue = queue.Queue()
        self._lo: str | None = None
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="tts")
        self._thread.start()

    def say(self, text: str, priority: str = "high"):
        text = _shorten(text)
        if not text:
            return
        now = time.time()
        if now - self._last.get(text, 0.0) < self.dedup_s:
            return
        self._last[text] = now
        if priority == "high":
            self._hi.put(text)
        else:
            with self._lock:
                self._lo = text                                        # newest low-priority phrase wins

    def _next(self) -> str | None:
        try:
            return self._hi.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            t, self._lo = self._lo, None
        return t

    def _loop(self):
        last_t = 0.0
        while not self._stop:
            text = self._next()
            if text is None:
                time.sleep(0.05)
                continue
            wait = self.min_interval - (time.time() - last_t)
            if wait > 0:
                time.sleep(wait)
            try:
                self._emit(text)
            except Exception as e:  # noqa: BLE001
                log.debug("tts emit failed: %s", e)
            self.said.append(text)
            last_t = time.time()

    def _emit(self, text: str):
        raise NotImplementedError

    def close(self):
        self._stop = True


class MockSpeaker(_SpeakerBase):
    def __init__(self):
        super().__init__(min_interval_s=0.0)

    def _emit(self, text: str):
        log.info("SAY: %s", text)


class UnitreeSpeaker(_SpeakerBase):
    """TTS via the `voice` RPC service."""

    def __init__(self, cfg):
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

        self.client = AudioClient()
        self.client.SetTimeout(3.0)
        self.client.Init()
        self.speaker_id = int(cfg.audio.get("speaker_id", 0))
        vol = cfg.audio.get("volume")
        if vol is not None:
            try:
                self.client.SetVolume(int(vol))
            except Exception as e:  # noqa: BLE001
                log.warning("SetVolume failed: %s", e)
        super().__init__(min_interval_s=float(cfg.audio.get("min_interval_s", 2.0)),
                         dedup_s=float(cfg.audio.get("dedup_s", 10.0)))

    def _emit(self, text: str):
        log.info("SAY: %s", text)
        code = self.client.TtsMaker(text, self.speaker_id)
        if code not in (0, None):
            log.warning("TtsMaker returned %s for %r", code, text)


class TtsLogHandler(logging.Handler):
    """Forwards WARNING/ERROR log records and selected EVENT lines to the speaker (low priority)."""

    def __init__(self, speaker, min_level: int = logging.WARNING):
        super().__init__(level=logging.INFO)
        self.speaker = speaker
        self.min_level = min_level

    def emit(self, record: logging.LogRecord):
        try:
            if record.name.startswith("g1_fetch.control.audio"):
                return
            msg = record.getMessage()
            if record.levelno >= self.min_level:
                head = "error" if record.levelno >= logging.ERROR else "warning"
                self.speaker.say(f"{head}: {msg}", priority="low")
                return
            if msg.startswith("EVENT "):
                name = msg.split()[1]
                if name in _SKIP_EVENTS:
                    return
                phrase = EVENT_PHRASES.get(name, name.replace("_", " "))
                if phrase is None:
                    return
                if name == "phase_failed":
                    m = re.search(r'"reason": "([^"]*)"', msg)
                    phrase = "failed: " + (m.group(1) if m else "")
                self.speaker.say(phrase, priority="low")
        except Exception:  # noqa: BLE001
            pass


def build_speaker(cfg, dry_run: bool):
    audio = cfg.get("audio")
    enabled = True if audio is None else bool(audio.get("enabled", True))
    if dry_run or not enabled:
        return MockSpeaker()
    try:
        return UnitreeSpeaker(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("audio client unavailable (%s); continuing silently", e)
        return MockSpeaker()


def attach_log_to_speech(speaker, cfg) -> TtsLogHandler | None:
    audio = cfg.get("audio")
    if audio is not None and not bool(audio.get("speak_logs", True)):
        return None
    handler = TtsLogHandler(speaker)
    logging.getLogger("g1_fetch").addHandler(handler)
    return handler
