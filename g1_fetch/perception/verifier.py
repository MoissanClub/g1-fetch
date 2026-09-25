"""Optional VLM yes/no verifier over an OpenAI-compatible chat endpoint (vLLM, Moondream server...)."""
from __future__ import annotations

import base64
import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Verifier(Protocol):
    def ask(self, color: np.ndarray, question: str) -> bool | None: ...


class NullVerifier:
    def ask(self, color, question) -> bool | None:
        return None


class StubVerifier:
    def __init__(self, answers: dict[str, bool] | None = None, default: bool | None = True):
        self.answers = answers or {}
        self.default = default
        self.asked: list[str] = []

    def ask(self, color, question) -> bool | None:
        self.asked.append(question)
        for key, val in self.answers.items():
            if key in question:
                return val
        return self.default


class HttpVLMVerifier:
    def __init__(self, url: str, model: str, timeout_s: float = 8.0):
        import requests  # noqa: F401

        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout_s

    def ask(self, color: np.ndarray, question: str) -> bool | None:
        import cv2
        import requests

        ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": self.model,
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": question + " Answer with exactly one word: yes or no."},
                ],
            }],
        }
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:  # network/model failure must never stop the task
            log.warning("verifier failed: %s", e)
            return None
        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False
        return None


def build_verifier(cfg) -> Verifier:
    v = cfg.verifier
    if not v.enabled:
        return NullVerifier()
    return HttpVLMVerifier(v.url, v.model, float(v.timeout_s))
