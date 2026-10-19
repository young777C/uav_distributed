"""1003 mechanism diagnostics from the per-tick ASD logs (runs/asd/iter/<arm>[_p<k>].jsonl), aligned to scorer seeds
(skipped episodes never reset the policy → logged episodes map in order onto the NON-skipped seeds of the same part).
Per arm: tick accuracy of every source and of the final output, opening correctness (first logged tick), and the
committed correct-lock ratio stratified by opening correctness. Fusion arms (V3 family): error correlation between
TAHRelM and DAM4SAM picks (pre-registered H3c: measured on V3-ensemble = no feedback), mis-gates (a source that was
RIGHT gets rewritten to a WRONG posterior identity). V1: AUROC of the SAM2 predicted IoU for "DAM4SAM pick is right".
  python iter_mech.py [--json eval_out/iter_mech.json]"""
import argparse
import glob
import json
import os
import re

import numpy as np

EO = os.path.join(os.path.dirname(__file__), "eval_out")
LG = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "asd", "iter")
ARMS = ["v1_t05", "v1nog_t05", "v2_t05", "v3_t05", "v3v2_t05", "v3ens_t05", "v3com_t05", "v3gt_t05",
        "v3_t03", "v3_t10", "pgif_t03"]


def _json_of(path):
    for f in (path, path + ".partial"):
        if os.path.exists(f):
            return json.load(open(f))["episodes"]
    return []


def aligned(arm):
    """→ {seed: (scorer episode, [tick records])}."""
    out = {}
    logs = sorted(glob.glob(os.path.join(LG, f"{arm}_p[0-9].jsonl"))) or [os.path.join(LG, f"{arm}.jsonl")]
    for jl in logs:
        if not os.path.exists(jl):
            continue
        m = re.search(r"_p(\d)\.jsonl$", jl)
        eps = _json_of(os.path.join(EO, f"paper_it_{arm}_p{m.group(1)}.json" if m else f"paper_it_{arm}.json"))
        scored = [e for e in eps if not e.get("skipped")]
        E = {}
        for line in open(jl):
            r = json.loads(line)
            E.setdefault(r["ep"], []).append(r)
        for k, e in zip(sorted(E), scored):
            out.setdefault(int(e["episode_seed"]), (e, E[k]))
    return out


def auroc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def arm_stats(arm):
    A = aligned(arm)
    if not A:
        return None
    vis = lambda R: [r for r in R if r.get("true") is not None and r["true"] >= 0]
    fused = any("picks" in r for _, R in A.values() for r in R)
    out_key = "map" if fused else "anc"
    res = {"arm": arm, "n_ep": len(A)}
    T = [r for _, R in A.values() for r in vis(R)]
    res["ticks"] = len(T)
    res["acc_out"] = float(np.mean([r[out_key] == r["true"] for r in T]))
    if fused:
        for s in ("tah", "dam", "lang"):
            P = [r for r in T if r["picks"].get(s) is not None and r["picks"][s] >= 0]
            res[f"acc_{s}"] = float(np.mean([r["picks"][s] == r["true"] for r in P])) if P else float("nan")
            res[f"present_{s}"] = len(P) / max(len(T), 1)
        B = [r for r in T if r["picks"].get("tah") is not None and r["picks"].get("dam") is not None]
        et = np.array([r["picks"]["tah"] != r["true"] for r in B], float)
        ed = np.array([r["picks"]["dam"] != r["true"] for r in B], float)
        res["rho_err"] = float(np.corrcoef(et, ed)[0, 1]) if len(B) > 2 and et.std() > 0 and ed.std() > 0 else float("nan")
        res["p_dam_wrong_given_tah_wrong"] = float(ed[et == 1].mean()) if (et == 1).any() else float("nan")
        res["both_wrong_same_id"] = float(np.mean([r["picks"]["tah"] == r["picks"]["dam"] != r["true"] for r in B]))
        res["tah_eq_dam"] = float(np.mean([r["picks"]["tah"] == r["picks"]["dam"] for r in B]))
        for s in ("tah", "dam"):   # gate events on a source = posterior ≠ its pick → its memory is rewritten
            G = [r for r in T if r.get("gate", {}).get(s) and r["picks"].get(s) is not None]
            res[f"gates_{s}"] = len(G)
            res[f"misgate_{s}"] = float(np.mean([r["picks"][s] == r["true"] for r in G])) if G else float("nan")
    else:
        res["acc_mem"] = float(np.mean([r["mem"] == r["true"] for r in T]))
        res["fires"] = int(sum(bool(r.get("fire")) or r["mem"] != r["anc"] for r in T))
        F = [r for r in T if r["mem"] != r["anc"]]
        res["misgate_mem"] = float(np.mean([r["mem"] == r["true"] for r in F])) if F else float("nan")
        if arm.startswith("v1"):
            res["auroc_iou"] = auroc([r["conf"] for r in T], [r["mem"] == r["true"] for r in T])
    # opening correctness (first logged visible tick) and committed correct-lock ratio by stratum
    op, cr = [], []
    for e, R in A.values():
        V = vis(R)
        if not V:
            continue
        op.append(V[0][out_key] == V[0]["true"]); cr.append(1 - e["mis_follow_sustained"])
    op, cr = np.array(op, bool), np.array(cr, float)
    res["open_ok"] = f"{int(op.sum())}/{len(op)}"
    res["correct_open_ok"] = float(cr[op].mean()) if op.any() else float("nan")
    res["correct_open_wrong"] = float(cr[~op].mean()) if (~op).any() else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    R = [x for x in (arm_stats(arm) for arm in ARMS) if x]
    keys = ["n_ep", "ticks", "acc_out", "acc_mem", "acc_tah", "acc_dam", "acc_lang", "present_dam", "rho_err",
            "p_dam_wrong_given_tah_wrong", "both_wrong_same_id", "tah_eq_dam", "gates_tah", "misgate_tah", "gates_dam",
            "misgate_dam", "fires", "misgate_mem", "auroc_iou", "open_ok", "correct_open_ok", "correct_open_wrong"]
    print("| 指标 | " + " | ".join(r["arm"] for r in R) + " |\n|---|" + "---|" * len(R))
    for k in keys:
        cells = []
        for r in R:
            v = r.get(k, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"| {k} | " + " | ".join(cells) + " |")
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
