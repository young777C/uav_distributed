"""ASD PoC paired statistics: for each (base, arm) pair over common scored seeds → Δ track_correct, Δ max_wrong
(paired bootstrap 95% CI), latch@{2,5}s, SR with Wilson CI + exact McNemar.
  python asd_stats.py base.json:arm.json [...] --json out.json      (stems relative to eval_out/, no .json)
"""
import argparse
import json

import numpy as np

from stats_significance import load, wilson, mcnemar_exact, boot_diff


def per_seed(e):
    return {"correct": 1 - e["mis_follow_inst"], "max_wrong": e["max_wrong_run_s"],
            "l2": float(e["max_wrong_run_s"] < 2.0), "l5": float(e["max_wrong_run_s"] < 5.0),
            "sr": float(bool(e["success"])), "idsw": e["id_switches"]}


def compare(base, arm):
    A, B = load(base), load(arm)
    seeds = sorted(set(A) & set(B))
    a = [per_seed(A[s]) for s in seeds]; b = [per_seed(B[s]) for s in seeds]
    res = {"base": base, "arm": arm, "n": len(seeds)}
    for k in ("correct", "max_wrong", "l2", "l5", "idsw"):
        x = np.array([r[k] for r in b]); y = np.array([r[k] for r in a])
        m, lo, hi = boot_diff(x, y)
        res[k] = {"base": float(y.mean()), "arm": float(x.mean()), "delta": float(m), "ci": [float(lo), float(hi)]}
    sa = np.array([r["sr"] for r in a]); sb = np.array([r["sr"] for r in b])
    res["sr"] = {"base": wilson(int(sa.sum()), len(sa)), "arm": wilson(int(sb.sum()), len(sb)),
                 "mcnemar_p": mcnemar_exact(int(((sb == 1) & (sa == 0)).sum()), int(((sb == 0) & (sa == 1)).sum()))}
    return res


def fmt(r):
    c, w = r["correct"], r["max_wrong"]
    return (f"{r['arm']:34s} vs {r['base']:30s} n={r['n']:2d} | correct {c['base']:.3f}→{c['arm']:.3f} "
            f"Δ{c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] | max_wrong {w['base']:.2f}→{w['arm']:.2f}s "
            f"Δ{w['delta']:+.2f} [{w['ci'][0]:+.2f},{w['ci'][1]:+.2f}] | latch@5 {r['l5']['base']:.2f}→{r['l5']['arm']:.2f} "
            f"| SR {r['sr']['base'][0]:.3f}→{r['sr']['arm'][0]:.3f} (McN p={r['sr']['mcnemar_p']:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pairs", nargs="+"); ap.add_argument("--json")
    a = ap.parse_args()
    out = []
    for p in a.pairs:
        b, x = p.split(":")
        r = compare(b, x); out.append(r); print(fmt(r))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
