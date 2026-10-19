"""What decides correct re-acquisition? (dev, Town05, seeds 91000–19; tick logs aligned to scorer seeds per part)
(1) Episode level: share of the variance of the committed correct-lock ratio explained by opening correctness (R²), and
    the decomposition  mean = P(open ok)·C_ok + P(open wrong)·C_wrong.
(2) Event level: every RE-APPEARANCE of the target among the candidates after ≥1 tick of absence (occluded / out of
    frame). Outcome over the first 5 ticks after re-appearance: RE-ACQUIRED (output = target in ≥3), WRONG (output =
    a distractor in ≥3), else MIXED. Factors: opening correct, pre-loss lock correct (output = target at the last tick
    before the absence), absence length, target tracker-id changed across the absence (tracker arms), look-alikes.
GT is used only for scoring.   python reacq_factors.py"""
import collections
import glob
import json
import os

import numpy as np

import iter_gate as g

LG = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "asd")
ARMS = [("V0 旧设定(理想 ID,不考虑遮挡)", "pgif", ["deploy_all/tahrelm_pgif.jsonl"], "eval_out/paper_dep_tahrelm_pgif.json"),
        ("B  PGIF·理想 ID", "v2_pgif_sim", None, None), ("H-B PGIF+不可见·理想 ID", "v2h_pgif_sim", None, None),
        ("A′ PGIF·调后跟踪器", "v2t_pgif_trk", None, None), ("H-A PGIF+不可见·调后跟踪器", "v2h_pgif_trk", None, None)]


def aligned(stem, logs, json_path):
    """seed -> sorted tick records."""
    pairs = []
    if logs:
        pairs = [(os.path.join(LG, logs[0]), json_path)]
    else:
        for jl in sorted(glob.glob(os.path.join(LG, "iter", f"{stem}_p[0-9].jsonl"))):
            k = jl.split("_p")[-1].split(".")[0]
            js = sorted(glob.glob(f"eval_out/paper_it_{stem}_p{k}.json*"))
            if js:
                pairs.append((jl, js[-1]))
    out = {}
    for jl, js in pairs:
        try:
            eps = [e for e in json.load(open(js))["episodes"] if not e.get("skipped")]
        except Exception:
            continue
        E = collections.defaultdict(list)
        for line in open(jl):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            E[r["ep"]].append(r)
        for ep, e in zip(sorted(E), eps):
            out.setdefault(e["episode_seed"], sorted(E[ep], key=lambda r: r["t"]))
    return out


def out_id(r):
    """identity output this tick: actor/tracklet id of PGIF's pick (None = absent / no pick)."""
    s = r.get("anc") if "anc" in r else r.get("mem")
    if s is None or s < 0 or s >= len(r["aidx"]):
        return None
    return r["aidx"][s]


def events(R):
    ev = []; i = 0
    present = [r.get("true") is not None and r["true"] >= 0 for r in R]
    while i < len(R):
        if present[i]:
            i += 1; continue
        j = i
        while j < len(R) and not present[j]:
            j += 1
        if i > 0 and j < len(R):                               # absence [i, j) bounded by presence on both sides
            pre = R[i - 1]; post = R[j:j + 5]
            tgt_pre = pre["aidx"][pre["true"]]; tgt_post = R[j]["aidx"][R[j]["true"]]
            ok = sum(out_id(r) is not None and r["true"] >= 0 and out_id(r) == r["aidx"][r["true"]] for r in post)
            wrong = sum(out_id(r) is not None and (r["true"] is None or r["true"] < 0 or out_id(r) != r["aidx"][r["true"]]) for r in post)
            ev.append({"len": j - i, "pre_ok": out_id(pre) == tgt_pre, "id_changed": tgt_pre != tgt_post,
                       "res": "re-acq" if ok >= 3 else ("wrong" if wrong >= 3 else "mixed"), "n_cand": len(R[j]["aidx"])})
        i = j
    return ev


def main():
    for lab, stem, logs, jpath in ARMS:
        A = g.load(stem) if jpath is None else {e["episode_seed"]: e for e in json.load(open(jpath))["episodes"] if not e.get("skipped")}
        T = aligned(stem, logs, jpath)
        S = [s for s in A if s in T and T[s]]
        op = {}
        for s in S:
            r0 = next((r for r in T[s] if r.get("true") is not None and r["true"] >= 0), None)
            op[s] = r0 is not None and out_id(r0) == r0["aidx"][r0["true"]]
        y = np.array([1 - A[s]["mis_follow_sustained"] for s in S]); x = np.array([op[s] for s in S], float)
        r2 = np.corrcoef(x, y)[0, 1] ** 2 if x.std() > 0 else float("nan")
        p = x.mean(); c_ok = y[x == 1].mean(); c_bad = y[x == 0].mean() if (x == 0).any() else float("nan")
        print(f"\n## {lab}  (n={len(S)})")
        print(f"  episode: R²(opening → correct ratio) = {r2:.2f} | P(open ok) {p:.2f} · C_ok {c_ok:.2f} + P(wrong) {1-p:.2f} · C_wrong {c_bad:.2f} = {y.mean():.3f}")
        print(f"           if opening were always right (C_ok kept): {c_ok:.2f}  (+{c_ok - y.mean():.2f})")
        EV = [(s, e) for s in S for e in events(T[s])]
        if not EV:
            continue
        def rate(sel):
            sub = [e for s, e in EV if sel(s, e)]
            if not sub:
                return "—"
            c = collections.Counter(e["res"] for e in sub)
            return f"n={len(sub):3d} 找回 {c['re-acq']/len(sub):.0%} 错锁 {c['wrong']/len(sub):.0%}"
        print(f"  re-appearance events: {len(EV)}")
        print(f"    all                         : {rate(lambda s, e: True)}")
        print(f"    opening right / wrong       : {rate(lambda s, e: op[s])}  ||  {rate(lambda s, e: not op[s])}")
        print(f"    pre-loss lock right / wrong : {rate(lambda s, e: e['pre_ok'])}  ||  {rate(lambda s, e: not e['pre_ok'])}")
        print(f"    open right & pre-loss right : {rate(lambda s, e: op[s] and e['pre_ok'])}  ||  open wrong & pre-loss right: {rate(lambda s, e: (not op[s]) and e['pre_ok'])}")
        print(f"    pre-loss right, absence ≤8 / >8 ticks : {rate(lambda s, e: e['pre_ok'] and e['len'] <= 8)}  ||  {rate(lambda s, e: e['pre_ok'] and e['len'] > 8)}")
        if "trk" in stem:
            print(f"    pre-loss right, target id kept / changed : {rate(lambda s, e: e['pre_ok'] and not e['id_changed'])}  ||  {rate(lambda s, e: e['pre_ok'] and e['id_changed'])}")
        print(f"    pre-loss right, ≤4 / ≥5 candidates at re-appearance : {rate(lambda s, e: e['pre_ok'] and e['n_cand'] <= 4)}  ||  {rate(lambda s, e: e['pre_ok'] and e['n_cand'] >= 5)}")


if __name__ == "__main__":
    main()
