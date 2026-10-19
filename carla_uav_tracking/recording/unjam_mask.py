"""Training-time frame mask for unjam relocations (scene.kinematics.unjam, 2026-10-09; read-only, offline).

`_unjam_step` may teleport a stuck ambient car / blocking distractor to a far spawn point. When that car was inside the
camera frame it vanishes between two frames — a label/appearance discontinuity no tracker should be trained on. Each
relocation is logged in the episode h5 attr `kin_unjam` (json: events[{tick, kind, actor, in_view}]).

Tick → frame: the recorder's warm-up calls step_target 90 times before frame 0, and the first recorded frame that shows
a relocation made at unjam tick T is T - 90 (verified on smoke3: every distractor relocation's position jump in
distractors/positions lands exactly on that frame). Events with T - 90 < 0 happened during warm-up (nothing recorded).

    from recording.unjam_mask import unjam_keep_mask
    keep = unjam_keep_mask("episode_000000.h5")          # bool[n_frames], False = drop (±1 s around in-view relocations)
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import h5py
import numpy as np

TICK_TO_FRAME = 90


@contextmanager
def _opened(h5):
    """accept a path or an already open h5py.File"""
    if isinstance(h5, h5py.File):
        yield h5
    else:
        with h5py.File(h5, "r") as f:
            yield f


def unjam_events(h5) -> list[dict]:
    """[{frame, kind, actor, in_view}] from attr kin_unjam ([] for episodes recorded without scene.kinematics)."""
    with _opened(h5) as f:
        raw = f.attrs.get("kin_unjam")
    if raw is None:
        return []
    evs = (json.loads(raw) or {}).get("events", [])
    return [{**{k: v for k, v in e.items() if k != "tick"}, "frame": int(e["tick"]) - TICK_TO_FRAME} for e in evs]


def unjam_keep_mask(h5, n_frames: int | None = None, pad_s: float = 1.0, in_view_only: bool = True,
                    fps: int = 10) -> np.ndarray:
    """bool[n_frames]: False for frames within ±pad_s of a relocation (only in-view ones by default)."""
    with _opened(h5) as f:
        n = int(n_frames) if n_frames is not None else len(np.asarray(f["target/tspeed"]))
        evs = unjam_events(f)
    keep = np.ones(n, dtype=bool)
    pad = int(round(pad_s * fps))
    for e in evs:
        if in_view_only and not e.get("in_view"):
            continue
        a, b = max(0, e["frame"] - pad), min(n, e["frame"] + pad + 1)     # inclusive ±pad (codex 10-11 R1-4)
        if b > a:
            keep[a:b] = False
    return keep
