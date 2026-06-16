from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_api_keys(path: str | Path = "configs/api_keys.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"API keys file not found at {p}. Copy configs/api_keys.yaml.example "
            "to configs/api_keys.yaml and fill in your keys."
        )
    return load_yaml(p)


def apply_cli_overrides(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    out = dict(cfg)
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out
