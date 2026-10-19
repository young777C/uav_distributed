"""Open-loop replay ablations for PGIF on observe-mode ASD logs (GT used ONLY for scoring).

Variants (all GT-free at decision time; inputs = memory pick, memory confidence, tracklet ids):
  mem              : the memory's raw pick (baseline)
  pgif_rankq       : PGIF (sticky posterior, rank-normalised confidence → error rate)       [the method]
  hmm_constq       : same sticky posterior, constant error rate (no confidence)            [ablation: confidence]
  longest_resid    : visible tracklet with the largest decayed residence (pick-count)      [ablation: heuristic]
  last_stable      : last tracklet held >= K consecutive ticks, if visible, else pick      [ablation: heuristic]
ID-noise robustness: the tracklet-id stream fed to the variants is corrupted with per-tick
  fragmentation (an actor gets a fresh id) and switches (two visible actors swap ids) at rate p.
  python pgif_replay.py <observe.jsonl ...> [--json out.json]
"""
import argparse
import collections
import json

import numpy as np

from asd import ASD


def episodes(paths):
    E = collections.defaultdict(list)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            E[(p, r["ep"])].append(r)
    return list(E.values())


def corrupt(seq, p, rng):
    """Return per-tick noisy tracklet ids (lists aligned with r['aidx'])."""
    lab, nxt, out = {}, 10_000, []
    for r in seq:
        ids = []
        for a in r["aidx"]:
            if a not in lab:
                lab[a] = a
            if p > 0 and rng.random() < p:               # fragmentation: fresh id
                lab[a] = nxt; nxt += 1
            ids.append(lab[a])
        if p > 0 and len(ids) >= 2 and rng.random() < p:  # identity switch between two visible actors
            i, j = rng.choice(len(ids), 2, replace=False)
            ai, aj = r["aidx"][i], r["aidx"][j]
            lab[ai], lab[aj] = lab[aj], lab[ai]; ids[i], ids[j] = ids[j], ids[i]
        out.append(ids)
    return out


def run_variant(name, seq, ids_seq, K=5, decay=0.98):
    picks = []
    if name == "mem":
        return [r["mem"] for r in seq]
    if name.startswith("tempered") or name.startswith("residA"):
        lam = float(name.split("@")[1]) if "@" in name else 0.98
        a = ASD(mode="filter", filter_kind="tempered" if name.startswith("tempered") else "resid", resid_decay=lam)
        for r, ids in zip(seq, ids_seq):
            picks.append(a.step(None, ids, r["mem"], r["conf"], None, None, r["t"])["slot"])
        return picks
    if name in ("pgif_rankq", "hmm_constq"):
        a = ASD(mode="filter", q_lo=0.2 if name == "pgif_rankq" else 0.45,
                q_hi=0.7 if name == "pgif_rankq" else 0.45)
        for r, ids in zip(seq, ids_seq):
            picks.append(a.step(None, ids, r["mem"], r["conf"], None, None, r["t"])["slot"])
        return picks
    resid = collections.defaultdict(float); run_id, run_len, stable = None, 0, None
    for r, ids in zip(seq, ids_seq):
        act = ids[r["mem"]]
        for k in resid:
            resid[k] *= decay
        resid[act] += 1
        run_len = run_len + 1 if act == run_id else 1; run_id = act
        if run_len >= K:
            stable = act
        if name == "longest_resid":
            picks.append(max(range(len(ids)), key=lambda s: resid.get(ids[s], 0.0)))
        else:  # last_stable
            picks.append(ids.index(stable) if stable in ids else r["mem"])
    return picks


def score(eps, name, p=0.0, seed=0):
    rng = np.random.default_rng(seed); n = c = 0; runs = []
    for seq in eps:
        ids_seq = corrupt(seq, p, rng)
        picks = run_variant(name, seq, ids_seq)
        cur = 0
        for r, s in zip(seq, picks):
            if r["true"] is None or r["true"] < 0:
                continue
            n += 1; ok = int(s == r["true"]); c += ok
            cur = 0 if ok else cur + 1
            if not ok:
                runs.append(cur)
    longest = []
    for seq in eps:                                       # longest wrong streak per episode (ticks)
        pass
    return c / max(n, 1), n


def longest_wrong(eps, name, p=0.0, seed=0):
    rng = np.random.default_rng(seed); out = []
    for seq in eps:
        picks = run_variant(name, seq, corrupt(seq, p, rng))
        cur = best = 0
        for r, s in zip(seq, picks):
            if r["true"] is None or r["true"] < 0:
                continue
            cur = cur + 1 if s != r["true"] else 0; best = max(best, cur)
        out.append(best)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("logs", nargs="+"); ap.add_argument("--json")
    a = ap.parse_args()
    eps = episodes(a.logs)
    V = ["mem", "tempered@0.98", "residA@0.98", "pgif_rankq", "hmm_constq"]
    res = {"variants": {}, "id_noise": {}}
    print(f"episodes={len(eps)}")
    for v in V:
        acc, n = score(eps, v); lw = longest_wrong(eps, v)
        res["variants"][v] = {"acc": acc, "n": n, "longest_wrong_ticks": lw}
        print(f"  {v:14s} acc={acc:.3f}  mean longest-wrong={lw:.1f} ticks  (n={n})")
    print("  id-noise robustness (acc, avg over 5 seeds):")
    NV = ["mem", "tempered@0.98", "residA@0.98", "pgif_rankq"]
    for p in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
        row = {v: float(np.mean([score(eps, v, p, s)[0] for s in range(5)])) for v in NV}
        res["id_noise"][str(p)] = row
        print(f"   p={p:<5} " + "  ".join(f"{v}={row[v]:.3f}" for v in NV))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
