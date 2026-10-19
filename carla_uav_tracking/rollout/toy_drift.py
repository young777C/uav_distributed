"""Toy model for Proposition 1 (confident drift): self-updating EMA memory over symmetric look-alikes.

Checks numerically that (a) a wrong lock is (near-)absorbing under self-update, (b) the memory's own
confidence has AUROC ~0.5 for "is the current lock wrong" once the memory has settled, whereas a
statistic built from a drift-immune frozen anchor separates them; and (c) ASD restores the target.
  python toy_drift.py  → prints a table + writes eval_out/toy_drift.json
"""
import json
import os

import numpy as np

from asd import ASD
from drift_analysis import auroc


def sim(seed, T=300, d=64, n=4, gamma=0.3, sigma=0.12, gap=0.35, p_occl=0.02, use_asd=False):
    rng = np.random.default_rng(seed)
    t0 = rng.normal(size=d); t0 /= np.linalg.norm(t0)
    mus = []
    for i in range(n):                                  # look-alikes: equal cosine gap to the target, symmetric
        if i == 0:
            mus.append(t0); continue
        r = rng.normal(size=d); r -= (r @ t0) * t0; r /= np.linalg.norm(r)
        v = (1 - gap) * t0 + np.sqrt(1 - (1 - gap) ** 2) * r; mus.append(v / np.linalg.norm(v))
    mus = np.array(mus)
    m = mus[0].copy(); asd = ASD(mode="rule") if use_asd else None
    recs = []
    for t in range(T):
        x = mus + sigma * rng.normal(size=mus.shape); x /= np.linalg.norm(x, axis=1, keepdims=True)
        s = x @ m
        pick = int(s.argmax())
        if rng.random() < p_occl:                       # occlusion / bad view: memory grabs a random look-alike
            pick = int(rng.integers(1, n))
        srt = np.sort(s)[::-1]; conf = 1 / (1 + np.exp(-(srt[0] - srt[1]) * 10))
        if asd is not None:
            o = asd.step(x, list(range(n)), pick, conf, lang=None, true_slot=0, t=t)
            if o["fire"]:
                pick = o["slot"]; m = x[pick].copy()
        anchor_cos = float(x[pick] @ mus[0])            # drift-immune reference (frozen initial appearance)
        recs.append((t, int(pick != 0), conf, 1 - anchor_cos))
        m = (1 - gamma) * m + gamma * x[pick]; m /= np.linalg.norm(m)
    return recs


def main():
    out = {}
    for name, ua in (("no_asd", False), ("asd", True)):
        R = [r for s in range(200) for r in sim(s, use_asd=ua)]
        arr = np.array(R)
        settled = arr[arr[:, 0] >= 50]
        y = settled[:, 1].astype(int)
        res = {"wrong_frac": float(y.mean()),
               "auroc_conf": auroc(-settled[:, 2], y) if 0 < y.mean() < 1 else None,
               "auroc_anchor": auroc(settled[:, 3], y) if 0 < y.mean() < 1 else None}
        # absorption: once wrong for >=5 ticks, prob of ever returning
        ret = tot = 0
        for s in range(200):
            seq = [r[1] for r in R[s * 300:(s + 1) * 300]]
            for t in range(5, 295):
                if all(seq[t - 5:t]) and (t == 5 or not seq[t - 6]):
                    tot += 1; ret += int(not all(seq[t:]))
        res["return_after_5wrong"] = ret / tot if tot else None
        out[name] = res
        print(name, json.dumps(res))
    os.makedirs("eval_out", exist_ok=True)
    json.dump(out, open("eval_out/toy_drift.json", "w"), indent=1)


if __name__ == "__main__":
    main()
