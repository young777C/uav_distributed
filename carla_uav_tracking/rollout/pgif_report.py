"""PGIF PoC report: paired closed-loop tables + filter diagnostics + cold-start stratification.
  python pgif_report.py --json eval_out/pgif_report.json
Arms missing on disk are skipped (reported as pending)."""
import argparse
import json
import os

import numpy as np

from asd_stats import compare, fmt

ROOT = os.path.dirname(__file__)
EO = os.path.join(ROOT, "eval_out")
LOGS = os.path.join(ROOT, "..", "..", "runs", "asd")

TAH_BASE = "paper_asd_tah_observe"          # committed + language cold-start, no correction (this session's code)
TAH_ARMS = ["paper_asd_tah_tempered", "paper_asd_tah_resid", "paper_asd_tah_tempered_nogate", "paper_asd_tah_bayes",
            "paper_asd_tah_anchor_p3", "paper_ext_tahrelm_deploy", "paper_ext_tahrelm_oracledrift",
            "paper_ext_tahrelm_gt"]
REID_BASE = "paper_asd_reid_observe"
REID_ARMS = ["paper_asd_reid_tempered", "reid_deploy"]
JSONL = {"paper_asd_tah_tempered": "tah_tempered_91000", "paper_asd_tah_resid": "tah_resid_91000",
         "paper_asd_tah_tempered_nogate": "tah_tempered_nogate_91000", "paper_asd_tah_bayes": "tah_bayes_91000",
         "paper_asd_reid_tempered": "reid_tempered_91000", "paper_asd_tah_observe": "tah_observe_91000",
         "paper_asd_reid_observe": "reid_observe_91000"}


def have(stem):
    return os.path.exists(os.path.join(EO, stem + ".json"))


def jsonl_diag(stem):
    p = os.path.join(LOGS, JSONL.get(stem, "") + ".jsonl")
    if stem not in JSONL or not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p)]
    eps = sorted(set(r["ep"] for r in R)); e0 = eps[0]
    lab = [r for r in R if r["true"] is not None and r["true"] >= 0]
    out = {"ticks": len(R), "labelled": len(lab),
           "mem_acc": float(np.mean([r["mem"] == r["true"] for r in lab])) if lab else None,
           "out_acc": float(np.mean([r["anc"] == r["true"] for r in lab if r["anc"] is not None])) if lab else None,
           "gate_rate": float(np.mean([r["fire"] for r in R])) if R else None}
    fires = [r for r in lab if r["fire"]]
    if fires:                                          # gate precision: override moved the pick ONTO the true target
        out["gate_to_true"] = float(np.mean([r["anc"] == r["true"] for r in fires]))
        out["gate_from_true"] = float(np.mean([r["mem"] == r["true"] for r in fires]))   # harmful overrides
    cold = {}
    for e in eps:                                      # cold-start lock correct? (first labelled tick)
        L = [r for r in lab if r["ep"] == e]
        if L:
            cold[91000 + (e - e0)] = bool(L[0]["mem"] == L[0]["true"])
    out["cold_start_correct"] = cold
    return out


def stratified(base, arm, cold):
    """track_correct by cold-start stratum (seed→bool from the ARM's own log)."""
    from stats_significance import load
    A, B = load(base), load(arm); res = {}
    for name, keep in (("cold_right", True), ("cold_wrong", False)):
        ss = [s for s in sorted(set(A) & set(B)) if cold.get(s) is keep]
        if ss:
            res[name] = {"n": len(ss), "base": float(np.mean([1 - A[s]["mis_follow_inst"] for s in ss])),
                         "arm": float(np.mean([1 - B[s]["mis_follow_inst"] for s in ss]))}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    out = {"tah": [], "reid": [], "diag": {}, "strat": {}, "pending": []}
    for base, arms, key in ((TAH_BASE, TAH_ARMS, "tah"), (REID_BASE, REID_ARMS, "reid")):
        if not have(base):
            out["pending"].append(base); print(f"[pending base] {base}"); continue
        print(f"=== {key.upper()} (base = {base}) ===")
        for arm in arms:
            if not have(arm):
                out["pending"].append(arm); print(f"  [pending] {arm}"); continue
            r = compare(base, arm); out[key].append(r); print("  " + fmt(r))
            d = jsonl_diag(arm)
            if d:
                out["diag"][arm] = d
                print(f"      diag: mem_acc={d['mem_acc']:.3f} out_acc={d['out_acc']:.3f} gate_rate={d['gate_rate']:.3f} "
                      f"gate→true={d.get('gate_to_true', float('nan')):.3f} gate-from-true={d.get('gate_from_true', float('nan')):.3f}")
                st = stratified(base, arm, d["cold_start_correct"]); out["strat"][arm] = st
                for k, v in st.items():
                    print(f"      {k}: n={v['n']} correct {v['base']:.3f}→{v['arm']:.3f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
