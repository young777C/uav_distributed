"""Build per-frame candidate boxes + cand_feats for the target-id head (§5, Option A).

At rollout time the target-id head needs a candidate SET (target + distractors as
image boxes) to argmax over. Training used GT-projected boxes (dataset_stage2.py:273
via acot_probe.projection); we reproduce that ONLINE from live CARLA GT positions +
the current camera pose.

H0 honesty (guide §5): the boxes are UNLABELED and SHUFFLED every frame — they give
the policy a candidate set (positions), never target identity. The tid head must
still pick the referred car via language. `true_idx` (the shuffled slot of the true
target) is returned for the SCORER only, never fed to the policy. Only IN-FRAME
vehicles become candidates (a detector sees only what's in frame); if the target is
off-screen, `true_idx == -1` and that frame is a "target-absent" (re-acquisition)
frame for scoring.

Swap this module for a real detector to remove all GT from the control path (Option B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acot_probe.projection import Candidate, project_point
from acot_probe.backbones import pool_box


@dataclass
class CandidateSet:
    feats: np.ndarray          # (N, C) pooled VLM features, shuffled order
    cands: list[Candidate]     # length N, same shuffled order (carries is_target/idx)
    true_idx: int              # slot of the true target in this set, or -1 if target off-screen
    mask: np.ndarray           # (N,) bool, all True (no padding for a single frame)


def _pseudo_box(depth, W, fov_deg, vehicle_size_m, box_min_px, box_max_px):
    import math
    f = W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    size = f * vehicle_size_m / max(depth, 0.1)
    return float(np.clip(size, box_min_px, box_max_px))


def build_candidates(grid_last: np.ndarray, tpos, dstates, campose,
                     image_cfg: dict, rng: np.random.Generator) -> CandidateSet | None:
    """grid_last: (Gh,Gw,C) VLM grid. tpos: (3,) target world xyz.
    dstates: (D,6) distractor [x,y,z,vx,vy,vz]. campose: (cx,cy,cz,pitch,yaw).
    Returns None if NO vehicle is in frame (nothing to select)."""
    W, H, fov = int(image_cfg["width"]), int(image_cfg["height"]), float(image_cfg["fov_deg"])
    vsz = float(image_cfg["vehicle_size_m"])
    bmin, bmax = float(image_cfg["box_min_px"]), float(image_cfg["box_max_px"])

    def _mk(world_xyz, is_target, idx):
        pr = project_point(world_xyz, campose, W, H, fov)
        if pr is None:
            return None
        u, v, depth = pr
        if not (0 <= u < W and 0 <= v < H):
            return None
        s = _pseudo_box(depth, W, fov, vsz, bmin, bmax)
        return Candidate(u, v, s, s, depth, is_target, idx)

    cands = []
    tgt = _mk((float(tpos[0]), float(tpos[1]), float(tpos[2])), True, -1)
    if tgt is not None:
        cands.append(tgt)
    dstates = np.asarray(dstates)
    for d in range(dstates.shape[0]):
        c = _mk((dstates[d, 0], dstates[d, 1], dstates[d, 2]), False, d)
        if c is not None:
            cands.append(c)

    if not cands:
        return None                                   # nothing in frame

    # Shuffle so the true target is not at a fixed slot (metric-honesty, §5 / [[tid-metric-artifact]]).
    order = rng.permutation(len(cands))
    cands = [cands[o] for o in order]
    true_idx = next((i for i, c in enumerate(cands) if c.is_target), -1)

    frac_boxes = [c.frac_box(W, H) for c in cands]
    feats = np.stack([pool_box(grid_last, fb) for fb in frac_boxes]).astype(np.float32)
    mask = np.ones(len(cands), dtype=bool)
    return CandidateSet(feats=feats, cands=cands, true_idx=true_idx, mask=mask)
