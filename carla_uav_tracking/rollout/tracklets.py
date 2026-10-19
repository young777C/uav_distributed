"""GT-free tracklet IDs for the candidate boxes (1004 · 'deployable v2').

Replaces the simulator's persistent actor index (`Candidate.idx`) as the identity key used by PGIF, the commitment
hysteresis, the re-ID gallery and WHERE. A SORT/DeepSORT-style assigner run at every S2 tick on what a detector
would output: box centre + size and an appearance embedding (crop-DINOv2) — never `is_target` / `idx`.

  • predict each live track with an image-space constant-velocity model (velocity per tick, EMA);
  • cost(track, box) = (1 − w_app)·d/gate + w_app·(1 − cos(app_track, feat)), gate = max(gate_px, gate_box·box size);
    pairs outside the motion gate or below `min_cos` are forbidden;
  • Hungarian assignment (scipy; greedy fallback); matched → update state; unmatched track → age += 1, deleted after
    `max_age` missed ticks; unmatched box → NEW id.
So a vehicle that stays unseen for more than `max_age` ticks (occlusion / out of frame) comes back with a new id, and
look-alikes that cross can swap ids — the failure modes a real tracker has and the simulator ids do not.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:          # pragma: no cover - scipy missing → greedy
    _lsa = None

_BIG = 1e6


class TrackletAssigner:
    def __init__(self, max_age: int = 5, gate_px: float = 100.0, gate_box: float = 3.0, w_app: float = 0.5,
                 min_cos: float = 0.3, vel_beta: float = 0.5, app_alpha: float = 0.3):
        self.max_age, self.gate_px, self.gate_box = int(max_age), float(gate_px), float(gate_box)
        self.w_app, self.min_cos = float(w_app), float(min_cos)
        self.vel_beta, self.app_alpha = float(vel_beta), float(app_alpha)
        self.reset()

    def reset(self):
        self.tracks: dict[int, dict] = {}      # id -> {pos(2,), vel(2,), app(C,)|None, age, hits}
        self.next_id = 0
        self.n_new = self.n_deleted = 0

    def _cost(self, uv, sizes, feats):
        ids = list(self.tracks)
        C = np.full((len(ids), len(uv)), _BIG)
        for a, tid in enumerate(ids):
            t = self.tracks[tid]
            pred = t["pos"] + t["vel"] * (t["age"] + 1)            # vel is per tick; coast over missed ticks
            d = np.linalg.norm(uv - pred[None], axis=1)
            gate = np.maximum(self.gate_px, self.gate_box * sizes)
            m = d / gate
            if feats is not None and t["app"] is not None:
                cos = feats @ t["app"]
                app = 1.0 - cos
                ok = (m <= 1.0) & (cos >= self.min_cos)
            else:
                app = np.zeros(len(uv)); ok = m <= 1.0
            C[a] = np.where(ok, (1 - self.w_app) * m + self.w_app * app, _BIG)
        return ids, C

    def update(self, uv, sizes, feats=None) -> list[int]:
        """uv (n,2) box centres px, sizes (n,) box side px, feats (n,C) L2-normalised or None → n tracklet ids."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2); sizes = np.asarray(sizes, np.float64).reshape(-1)
        feats = None if feats is None else np.asarray(feats, np.float64)
        n = len(uv); out = [-1] * n
        if self.tracks and n:
            ids, C = self._cost(uv, sizes, feats)
            if _lsa is not None:
                rows, cols = _lsa(C)
                pairs = [(r, c) for r, c in zip(rows, cols) if C[r, c] < _BIG]
            else:                                                  # greedy by ascending cost
                pairs, used_r, used_c = [], set(), set()
                for flat in np.argsort(C, axis=None):
                    r, c = divmod(int(flat), C.shape[1])
                    if C[r, c] >= _BIG:
                        break
                    if r not in used_r and c not in used_c:
                        pairs.append((r, c)); used_r.add(r); used_c.add(c)
            for r, c in pairs:
                tid = ids[r]; t = self.tracks[tid]
                step_v = (uv[c] - t["pos"]) / (t["age"] + 1)
                t["vel"] = (1 - self.vel_beta) * t["vel"] + self.vel_beta * step_v if t["hits"] > 1 else step_v
                t["pos"] = uv[c].copy(); t["age"] = 0; t["hits"] += 1
                if feats is not None:
                    a = feats[c] if t["app"] is None else (1 - self.app_alpha) * t["app"] + self.app_alpha * feats[c]
                    t["app"] = a / (np.linalg.norm(a) + 1e-8)
                out[c] = tid
        matched = {tid for tid in out if tid >= 0}
        for tid in list(self.tracks):                              # age / delete unmatched tracks
            if tid not in matched:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]; self.n_deleted += 1
        for c in range(n):                                         # births
            if out[c] < 0:
                tid = self.next_id; self.next_id += 1; self.n_new += 1
                self.tracks[tid] = {"pos": uv[c].copy(), "vel": np.zeros(2), "age": 0, "hits": 1,
                                    "app": None if feats is None else feats[c] / (np.linalg.norm(feats[c]) + 1e-8)}
                out[c] = tid
        return out
