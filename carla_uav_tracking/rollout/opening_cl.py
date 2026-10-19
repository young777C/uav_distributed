"""Closed-loop OPENING correct-lock rate (1004 opening fix) from the policy's ASD tick logs (<name>_p*.jsonl) aligned to
the scorer jsons (paper_it_<name>_p*.json / open_val_*.json): per episode, the identity output at
  first  = the first S2 tick on which the target is among the candidates (what reacq_factors calls 'opening correct')
  locked = the first S2 tick at/after step LOCK (end of the opening window, default 12)
is compared with the target's id (GT used only to score). Wilson 95% CIs.
  python opening_cl.py <log.jsonl>:<scorer.json> [...]   |   python opening_cl.py --stem zo_attr_trk"""
import argparse
import collections
import glob
import json
import math
import os

from reacq_factors import out_id

LG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "asd")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pairs_for(stem):
    out = []
    for jl in sorted(glob.glob(os.path.join(LG, "iter", f"{stem}_p[0-9].jsonl"))):
        k = jl.split("_p")[-1].split(".")[0]
        js = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_out", f"paper_it_{stem}_p{k}.json*")))
        if js:
            out.append((jl, js[-1]))
    return out


def load(pairs):
    eps = {}
    for jl, js in pairs:
        try:
            S = [e for e in json.load(open(js))["episodes"] if not e.get("skipped")]
        except Exception:
            continue
        E = collections.defaultdict(list)
        for line in open(jl):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            E[r["ep"]].append(r)
        for ep, e in zip(sorted(E), S):
            eps.setdefault(e["episode_seed"], sorted(E[ep], key=lambda r: r["t"]))
    return eps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pairs", nargs="*"); ap.add_argument("--stem", nargs="*", default=[])
    ap.add_argument("--lock", type=int, default=12)
    a = ap.parse_args()
    groups = [(s, pairs_for(s)) for s in a.stem] + [(p, [tuple(p.split(":"))]) for p in a.pairs]
    for name, pr in groups:
        T = load(pr); first, locked = [], []
        for s, R in sorted(T.items()):
            P = [r for r in R if r.get("true") is not None and r["true"] >= 0]
            if not P:
                continue
            r0 = P[0]; first.append(out_id(r0) == r0["aidx"][r0["true"]])
            rl = next((r for r in P if r["t"] >= a.lock), None)
            if rl is not None:
                locked.append(out_id(rl) == rl["aidx"][rl["true"]])
        for lab, v in (("first tick", first), (f"t≥{a.lock} (after window)", locked)):
            lo, hi = wilson(sum(v), len(v))
            print(f"{name:28s} {lab:22s} {sum(v)}/{len(v)} = {sum(v) / max(1, len(v)):.1%}  [95% CI {lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main()
