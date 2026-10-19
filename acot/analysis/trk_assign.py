"""Greedy vs Hungarian assignment of the deploy-v2 tracker (2026-10-06), open-loop on recorded candidate streams
(<ASD_LOG>.trk.jsonl: per S2 tick the boxes uv / sz, crop-DINOv2 features f and the simulator index gt, −1 = target).
The tracker is pure perception — its ids depend only on these inputs — so the replay is exact for a fixed stream; it
does not include the closed-loop effect of different ids on the flight.
  IDF1 / tgt_IDF1 / tgt_ids   as in carla_uav_tracking/rollout/trk_tune.py (the script the tracker was tuned with);
  sw_tgt     the target's tracker id changes between two of its observations;
  steal_tgt  an id passes between the target and another vehicle (target ↔ distractor identity swap).
  python -m acot.analysis.trk_assign [log prefix ...]      (default: the final-model logs zo* / aa_legacy* / v2h*)"""
import collections
import glob
import json
import os
import sys

import numpy as np

from acot import ROOT

CFG = dict(max_age=8, gate_px=150.0, gate_box=2.0, w_app=0.85, min_cos=0.2, vel_beta=0.5, app_alpha=0.3)


def episodes(prefixes):
    E = collections.defaultdict(list)
    for p in prefixes:
        for f in sorted(glob.glob(os.path.join(ROOT, "runs", "asd", "iter", f"{p}*.trk.jsonl"))):
            for line in open(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "f" in r:
                    E[(f, r["ep"])].append(r)
    return [sorted(v, key=lambda r: r["k"]) for v in E.values()]


def replay(seq, lsa):
    import rollout.tracklets as T
    T._lsa = lsa
    a = T.TrackletAssigner(**CFG)
    return [a.update(r["uv"], r["sz"], np.asarray(r["f"], np.float64)) for r in seq]


def events(seq, ids):
    s, owner, prev = collections.Counter(), {}, {}
    for r, tid in zip(seq, ids):
        cur = dict(zip(r["gt"], tid))
        for g, i in cur.items():
            if g in prev and prev[g] != i:
                s["sw_all"] += 1
                s["sw_tgt"] += g == -1
            if i in owner and owner[i] != g:
                s["steal_all"] += 1
                s["steal_tgt"] += g == -1 or owner[i] == -1
            owner[i] = g
        s["tgt_obs"] += -1 in cur
        prev.update(cur)
    return s


def idf1(seqs, all_ids):
    from scipy.optimize import linear_sum_assignment
    idtp = n = t_ok = t_n = 0
    tgt_ids = []
    for seq, ids in zip(seqs, all_ids):
        pairs = [p for r, i in zip(seq, ids) for p in zip(r["gt"], i)]
        gts, tids = sorted({g for g, _ in pairs}), sorted({t for _, t in pairs})
        gi, ti = {g: k for k, g in enumerate(gts)}, {t: k for k, t in enumerate(tids)}
        M = np.zeros((len(gts), len(tids)))
        for g, t in pairs:
            M[gi[g], ti[t]] += 1
        r_, c_ = linear_sum_assignment(-M)
        idtp += M[r_, c_].sum()
        n += len(pairs)
        if -1 in gi:
            row = gi[-1]
            m = dict(zip(r_, c_)).get(row)
            t_n += M[row].sum()
            t_ok += M[row, m] if m is not None else 0
            tgt_ids.append(int((M[row] > 0).sum()))
    return {"IDF1": idtp / max(n, 1), "tgt_IDF1": t_ok / max(t_n, 1), "tgt_ids": float(np.mean(tgt_ids))}


def main(argv):
    sys.path.insert(0, os.path.join(ROOT, "carla_uav_tracking"))
    from scipy.optimize import linear_sum_assignment
    seqs = episodes(argv or ["zo", "aa_legacy", "v2h"])
    out = {}
    for name, lsa in (("greedy", None), ("hungarian", linear_sum_assignment)):
        ids = [replay(s, lsa) for s in seqs]
        ev = sum((events(s, i) for s, i in zip(seqs, ids)), collections.Counter())
        out[name] = dict(idf1(seqs, ids), **{k: int(v) for k, v in ev.items()}, per_ep=[
            events(s, i)["sw_tgt"] + events(s, i)["steal_tgt"] for s, i in zip(seqs, ids)])
    d = np.array(out["greedy"]["per_ep"]) - np.array(out["hungarian"]["per_ep"])
    print(f"{len(seqs)} episodes, {sum(len(s) for s in seqs)} ticks")
    for k in ("IDF1", "tgt_IDF1", "tgt_ids", "tgt_obs", "sw_tgt", "steal_tgt", "sw_all", "steal_all"):
        g, h = out["greedy"][k], out["hungarian"][k]
        print(f"{k:10s} greedy {g:9.3f}  hungarian {h:9.3f}" if isinstance(g, float) else
              f"{k:10s} greedy {g:9d}  hungarian {h:9d}")
    print(f"target events per episode (greedy − hungarian): mean {d.mean():+.2f}; greedy worse in {(d > 0).sum()}, "
          f"Hungarian worse in {(d < 0).sum()}, equal in {(d == 0).sum()}")


if __name__ == "__main__":
    main(sys.argv[1:])
