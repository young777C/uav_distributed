#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sweep_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("sweep_mod", root / "scripts" / "sweep.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_resolve_slow_interval_steps_override(tmp_path: Path):
    sweep = _load_sweep_module()
    cfg = tmp_path / "combined.yaml"
    cfg.write_text(
        "extends: []\nexperiment:\n  sweep:\n    slow_interval_steps: 200\n",
        encoding="utf-8",
    )
    assert sweep._resolve_slow_interval_steps(combined_cfg_path=cfg, plan_default=40) == 200


def test_resolve_slow_interval_steps_plan_default(tmp_path: Path):
    sweep = _load_sweep_module()
    cfg = tmp_path / "combined.yaml"
    cfg.write_text("extends: []\nexperiment:\n  id: test\n", encoding="utf-8")
    assert sweep._resolve_slow_interval_steps(combined_cfg_path=cfg, plan_default=40) == 40
