"""YAML config loaded into attribute-accessible namespaces."""
from __future__ import annotations

import copy
import keyword
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Cfg(SimpleNamespace):
    """Nested namespace with dict-style helpers."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return _to_plain(self)


def _key(k: str) -> str:
    """YAML keys that are Python keywords (e.g. `return`) get a trailing underscore."""
    return k + "_" if keyword.iskeyword(k) else k


def _to_cfg(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg(**{_key(k): _to_cfg(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_cfg(v) for v in obj]
    return obj


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Cfg):
        return {k.rstrip("_") if keyword.iskeyword(k.rstrip("_")) else k: _to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    """Load configs/default.yaml, then an optional file, then dict overrides."""
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        data = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != (PROJECT_ROOT / "configs" / "default.yaml").resolve():
        with open(path) as f:
            data = _merge(data, yaml.safe_load(f) or {})
    if overrides:
        data = _merge(data, overrides)
    cfg = _to_cfg(data)
    cfg.project_root = str(PROJECT_ROOT)
    return cfg


def resolve_path(cfg: Cfg, p: str) -> Path:
    """Paths in the config are relative to the project root unless absolute."""
    path = Path(p)
    return path if path.is_absolute() else Path(cfg.project_root) / path
