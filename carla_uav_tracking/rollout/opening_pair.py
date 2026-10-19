"""Paired closed-loop opening comparison on the same held-out seeds (1004 opening fix): tag A vs tag B, per-seed opening
lock after the window (first S2 tick with the target present at t ≥ 12) and at the first tick; Wilson CIs, exact
McNemar.   python opening_pair.py attr:1,2,3,4,9 head:5,6,7"""
import math
import os
import sys

import opening_cl as o

D = "/nvidia/hque/code/cyhe/uav-acot-track/runs/opening_dump"


def per_seed(tag, ks):
    T = {}
    for k in ks:
        j = f"eval_out/open_val_{tag}_{k}.json"; j = j if os.path.exists(j) else j + ".partial"
        if os.path.exists(j):
            T.update(o.load([(f"{D}/val/{tag}_{k}.jsonl", j)]))
    out = {}
    for s, R in T.items():
        P = [r for r in R if r.get("true") is not None and r["true"] >= 0]
        if not P:
            continue
        rl = next((r for r in P if r["t"] >= 12), None)
        out[s] = (o.out_id(P[0]) == P[0]["aidx"][P[0]["true"]],
                  None if rl is None else o.out_id(rl) == rl["aidx"][rl["true"]])
    return out


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    (ta, ka), (tb, kb) = [(a.split(":")[0], a.split(":")[1].split(",")) for a in sys.argv[1:3]]
    A, B = per_seed(ta, ka), per_seed(tb, kb)
    for i, lab in ((1, "after window (t≥12)"), (0, "first tick")):
        S = [s for s in A if s in B and A[s][i] is not None and B[s][i] is not None]
        a = sum(A[s][i] for s in S); b_ = sum(B[s][i] for s in S)
        only_a = sum(A[s][i] and not B[s][i] for s in S); only_b = sum(B[s][i] and not A[s][i] for s in S)
        la, ha = o.wilson(a, len(S)); lb, hb = o.wilson(b_, len(S))
        print(f"{lab:20s} n={len(S)} | {ta} {a}/{len(S)} = {a / len(S):.1%} [{la:.1%}, {ha:.1%}] | {tb} {b_}/{len(S)} = "
              f"{b_ / len(S):.1%} [{lb:.1%}, {hb:.1%}] | discordant {only_a}:{only_b}  McNemar p={mcnemar(only_a, only_b):.2g}")


if __name__ == "__main__":
    main()
