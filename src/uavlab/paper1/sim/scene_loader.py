from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from uavlab.common.config import load_resolved_config


def load_scene_yaml(path: str) -> Dict[str, Any]:
    """Load scene with ``extends`` merged (e.g. ``base.yaml`` bounds, GCS, no-fly)."""

    p = Path(path)
    data = load_resolved_config(p)
    if not isinstance(data, dict):
        raise ValueError(f"Scene YAML must be a dict: {path}")
    return data

