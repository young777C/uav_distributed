"""Offline tuning of the GT-free tracker (rollout/tracklets.py) on DEV replay logs (<ASD_LOG>.trk.jsonl with uv/sz/f/gt).
The detections (box centre, size, crop-DINOv2 feature) are replayed open-loop through TrackletAssigner for every
parameter setting; association is scored against the simulator actor index (perception tuning on dev only):
  IDF1      = IDTP / N, IDTP = max co-occurrence bipartite matching between GT actors and tracker ids (per episode);
  tgt_IDF1  = share of the target's observations carrying the tracker id matched to the target;
  tgt_ids   = distinct tracker ids given to the target per episode.
  python trk_tune.py runs/asd/iter/v2_pgif_trk_p1.trk.jsonl [...] [--top 10]"""
import argparse
import collections
import itertools
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracklets import TrackletAssigner


def load(paths):
    E = collections.defaultdict(list)
    for p in paths:
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:                   # truncated / NUL line (e.g. host reboot mid-write) → skip
                continue
            if "f" in r:
                E[(p, r["ep"])].append(r)
    return [sorted(v, key=lambda r: r["k"]) for v in E.values()]


def score(eps, **kw):
    idtp = n = t_ok = t_n = 0; tgt_ids = []
    for seq in eps:
        tr = TrackletAssigner(**kw); pairs = []
        for r in seq:
            ids = tr.update(r["uv"], r["sz"], np.asarray(r["f"], np.float64))
            pairs += list(zip(r["gt"], ids))
        gts = sorted({g for g, _ in pairs}); tids = sorted({t for _, t in pairs})
        gi = {g: i for i, g in enumerate(gts)}; ti = {t: j for j, t in enumerate(tids)}
        M = np.zeros((len(gts), len(tids)))
        for g, t in pairs:
            M[gi[g], ti[t]] += 1
        r_, c_ = linear_sum_assignment(-M)
        idtp += M[r_, c_].sum(); n += len(pairs)
        if -1 in gi:
            row = gi[-1]; match = dict(zip(r_, c_)).get(row)
            t_n += M[row].sum(); t_ok += M[row, match] if match is not None else 0
            tgt_ids.append(int((M[row] > 0).sum()))
    return {"IDF1": idtp / max(n, 1), "tgt_IDF1": t_ok / max(t_n, 1), "tgt_ids": float(np.mean(tgt_ids)) if tgt_ids else 0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs", nargs="+"); ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    eps = load(a.logs)
    print(f"{len(eps)} episodes, {sum(len(s) for s in eps)} ticks")
    base = score(eps); print("default", {k: round(v, 3) for k, v in base.items()})
    grid = dict(max_age=[3, 5, 8, 12], w_app=[0.3, 0.5, 0.7, 0.85], gate_px=[60, 100, 150],
                gate_box=[2.0, 3.0, 4.0], min_cos=[0.2, 0.4, 0.6])
    res = []
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals)); s = score(eps, **kw); res.append((s["tgt_IDF1"], s["IDF1"], kw, s))
    res.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for t, i, kw, s in res[:a.top]:
        print({k: round(v, 3) for k, v in s.items()}, kw)


if __name__ == "__main__":
    main()
