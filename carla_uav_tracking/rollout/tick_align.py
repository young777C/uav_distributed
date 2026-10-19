"""Align ASD tick logs (runs/asd/*.jsonl, keyed by policy-reset index `ep`) to scorer episode seeds.
Skipped episodes (e.g. 'target stationary') never reset the policy, so logged episodes map IN ORDER onto the
NON-skipped seeds of the eval json. Verified by step counts (max logged t <= scorer steps, near-equal)."""
import json, os

EO = os.path.join(os.path.dirname(__file__), "eval_out")
LG = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "asd")


def aligned(stem, jsonl):
    """→ {seed: [tick records]} for non-skipped seeds."""
    d = json.load(open(os.path.join(EO, stem + ".json")))
    seeds = [e["episode_seed"] for e in d["episodes"] if not e.get("skipped")]
    E = {}
    for line in open(os.path.join(LG, jsonl + ".jsonl")):
        r = json.loads(line); E.setdefault(r["ep"], []).append(r)
    eps = sorted(E)
    assert len(eps) <= len(seeds), (stem, len(eps), len(seeds))
    return {s: E[e] for e, s in zip(eps, seeds)}


def runs(S, field):
    """Longest continuous CORRECT and WRONG run (s) over target-visible ticks (1 tick = 6 frames = 0.6 s); an
    out-of-view tick ends both runs. Final pick = `field` (PGIF output 'anc') or the raw memory pick 'mem'."""
    bc = bw = 0.0; cc = cw = 0; t0c = t0w = 0; ok_n = n = 0
    for r in S:
        if r["true"] is None or r["true"] < 0:
            cc = cw = 0; continue
        pick = r[field] if r.get(field) is not None else r["mem"]
        ok = pick == r["true"]; n += 1; ok_n += ok
        if ok:
            t0c = r["t"] if cc == 0 else t0c; cc += 1; cw = 0; bc = max(bc, (r["t"] - t0c) / 10 + 0.6)
        else:
            t0w = r["t"] if cw == 0 else t0w; cw += 1; cc = 0; bw = max(bw, (r["t"] - t0w) / 10 + 0.6)
    return bc, bw, (ok_n / n if n else float("nan"))
