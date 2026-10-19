"""Downstream effect of the attribute opening (1004 opening fix, F): final deployable model with OPENING=attr
(zoa_pgif_trk_{a,b,c}, dev 91000–19 in three seed blocks) vs the same model with the grounding-head opening (H-A,
v2h_pgif_trk), paired on common seeds: opening lock (closed loop), committed correct-lock ratio, longest correct lock,
longest wrong lock; paired seed-bootstrap 95% CI + sign-flip permutation p.
  python opening_ds.py [arm stems ...]   (default: the v1 arm zoa_pgif_trk_{a,b,c}; v3: zo3_pgif_trk_a zo3_pgif_trk_b)"""
import sys

import numpy as np

import iter_gate as g
import opening_cl as oc

ARM, REF = ["zoa_pgif_trk_a", "zoa_pgif_trk_b", "zoa_pgif_trk_c"], "v2h_pgif_trk"


def opening(stems):
    T = {}
    for s in stems:
        T.update(oc.load(oc.pairs_for(s)))
    out = {}
    for s, R in T.items():
        P = [r for r in R if r.get("true") is not None and r["true"] >= 0]
        rl = next((r for r in P if r["t"] >= 12), None)
        if rl is not None:
            out[s] = oc.out_id(rl) == rl["aidx"][rl["true"]]
    return out


def main():
    global ARM
    ARM = sys.argv[1:] or ARM
    A = {}
    for s in ARM:
        A.update(g.load(s) or {})
    B = g.load(REF)
    S = sorted(set(A) & set(B)); n = len(S)
    oa, ob = opening(ARM), opening([REF])
    so = [s for s in S if s in oa and s in ob]
    print(f"paired seeds n={n} (attr arm has {len(A)}, H-A has {len(B)})")
    print(f"  opening lock (t≥12): attr {sum(oa[s] for s in so)}/{len(so)} | head {sum(ob[s] for s in so)}/{len(so)}")
    rng = np.random.default_rng(20261003)
    for m in ("correct", "longC", "longW", "correct_s"):
        a = np.array([g.METRICS[m](A[s]) for s in S], float); b = np.array([g.METRICS[m](B[s]) for s in S], float)
        d = a - b; d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        bs = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
        perm = (np.abs((d[None] * rng.choice([-1, 1], (20000, len(d)))).mean(1)) >= abs(d.mean())).mean()
        print(f"  {g.LABEL[m]:18s} attr {np.nanmean(a):.3f} | head {np.nanmean(b):.3f} | Δ {d.mean():+.3f} "
              f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  p={perm:.3f}")
    print("  per-seed (seed: opening attr/head, correct attr/head):")
    print("   " + " ".join(f"{s % 1000}:{'✓' if oa.get(s) else '✗'}{'✓' if ob.get(s) else '✗'}"
                          f"({g.METRICS['correct'](A[s]):.2f}/{g.METRICS['correct'](B[s]):.2f})" for s in S))


if __name__ == "__main__":
    main()
