"""§5.5 WHICH-module comparison — ALL models DEPLOYABLE (GT-free, online closed-loop CARLA), same seeds 91000-19,
three objective metrics from the scorer (frame level): correct-lock ratio, longest continuous correct lock
(max_correct_run_s, committed identity, out-of-frame ends the run), longest wrong lock (max_wrong_run_s).
Per-arm mean with seed-bootstrap 95% CI; paired PGIF − arm with 95% paired-bootstrap CI. Missing arms → "pending".
  python deploy_compare.py --json eval_out/deploy_compare.json"""
import argparse
import json
import os

import numpy as np

EO = os.path.join(os.path.dirname(__file__), "eval_out")
ARMS = [  # stem suffix, label, group
    ("xattn", "xattn(单帧语言)", "language"),
    ("l4", "l4(reid+置信门)", "ours"),
    ("l5", "l5(+时序EMA)", "ours"),
    ("borrow1", "borrow#1(+运动共识)", "ours"),
    ("tahdag", "TAH+DAgger", "ours"),
    ("tahrelm", "TAHRelM 无纠偏", "ours"),
    ("ocsort", "OC-SORT", "external"),
    ("deepsort", "DeepSORT", "external"),
    ("dam4sam", "DAM4SAM", "external"),
    ("tahrelm_pgif", "TAHRelM+PGIF(ours)", "pgif"),
]
def _strict(e):
    """per-frame pick == target over target-present frames; ABSTENTION (no pick) counts as NOT correct."""
    nd, bn = e.get("n_dist") or {}, e.get("mis_by_N") or {}
    t = sum(nd.values())
    return 1 - sum(bn[k] * nd[k] for k in nd if k in bn) / t if t else float("nan")


# correct-lock ratio (2026-10-02 口径修正): COMMITTED identity == target over target-present frames — the same notion of
# "lock" as longC / longW; an abstaining tracker (DAM4SAM: empty mask / nothing within its gate) holds its last commitment.
# The former definition 1 − mis_follow_inst counted abstention frames as CORRECT (DAM4SAM abstains ~60% → 0.786 vs 0.367);
# it and the strict one (abstention = wrong) are kept as correct_l / correct_s for the appendix.
EXTRA = [("correct_l", lambda e: 1 - e["mis_follow_inst"]), ("correct_s", _strict)]
METRICS = [("correct", "正确锁定比例(承诺)", lambda e: 1 - e["mis_follow_sustained"]),
           ("longC", "最长连续正确锁定(s)", lambda e: e["max_correct_run_s"]),
           ("longW", "最长锁错(s)", lambda e: e["max_wrong_run_s"])]
RNG = np.random.default_rng(20261002)


def load(stem):
    p = os.path.join(EO, f"paper_dep_{stem}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {int(e["episode_seed"]): e for e in d["episodes"] if e.get("mis_follow_inst") is not None}


def ci_mean(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; b = x[RNG.integers(0, len(x), (20000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    D = {s: load(s) for s, _, _ in ARMS}
    P = D["tahrelm_pgif"]
    out = {"arms": [], "pending": [s for s, _, _ in ARMS if D[s] is None]}
    for s, lab, grp in ARMS:
        A = D[s]
        if A is None:
            out["arms"].append({"stem": s, "label": lab, "group": grp, "pending": True}); continue
        row = {"stem": s, "label": lab, "group": grp, "n": len(A)}
        for k, _, f in METRICS:
            v = [f(e) for e in A.values()]
            row[k] = ci_mean(v); row[k + "_med"] = float(np.median(v))
        for k, f in EXTRA:
            row[k] = ci_mean([f(e) for e in A.values()])
        if s != "tahrelm_pgif":
            S = sorted(set(A) & set(P))
            for k, _, f in METRICS:
                d = np.array([f(P[x]) - f(A[x]) for x in S])
                b = d[RNG.integers(0, len(d), (20000, len(d)))].mean(1)
                row["pgif_minus_" + k] = [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        out["arms"].append(row)
    for r in out["arms"]:
        if r.get("pending"):
            print(f"{r['label']:22s} pending"); continue
        msg = f"{r['label']:22s} n={r['n']} " + " | ".join(
            f"{k} {r[k][0]:.3f}" if k == "correct" else f"{k} {r[k][0]:.1f}s(med {r[k + '_med']:.1f})" for k, _, _ in METRICS)
        if "pgif_minus_correct" in r:
            msg += " || PGIF−arm: " + "  ".join(f"{k} {r['pgif_minus_' + k][0]:+.2f}[{r['pgif_minus_' + k][1]:+.2f},{r['pgif_minus_' + k][2]:+.2f}]"
                                                for k, _, _ in METRICS)
        print(msg)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
