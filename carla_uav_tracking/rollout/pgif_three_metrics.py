"""Paper evaluation with the THREE objective metrics (SR is reported only as a supplementary reference):
  correct  = correct-lock ratio (scorer, per frame, target-present frames)
  longC    = longest continuous correct lock, s (scorer max_correct_run_s when present, else tick logs via tick_align)
  longW    = longest wrong lock, s (scorer max_wrong_run_s, committed identity)
Paired over common seeds; 95% paired bootstrap CI of the mean difference.
  python pgif_three_metrics.py --json eval_out/pgif_three_metrics.json"""
import argparse, json
import numpy as np
from stats_significance import load
from tick_align import aligned, runs

ARMS = {  # stem: (label, tick log, final-pick field)
    "paper_asd_tah_observe": ("TAHRelM 无纠偏 (T05)", "tah_observe_91000", "mem"),
    "paper_asd_tah_tempered": ("TAHRelM+PGIF (T05)", "tah_tempered_91000", "anc"),
    "paper_asd_tah_resid": ("等权驻留", "tah_resid_91000", "anc"),
    "paper_asd_tah_tempered_nogate": ("PGIF 去门控", "tah_tempered_nogate_91000", "anc"),
    "paper_asd_tah_bayes": ("未回火贝叶斯", "tah_bayes_91000", "anc"),
    "paper_asd_tah_anchor_p3": ("冻结外观锚重锚", None, None),
    "paper_ext_tahrelm_deploy": ("低置信语言重锚", None, None),
    "paper_ext_tahrelm_oracledrift": ("真值漂移检测上界", None, None),
    "paper_ext_tahrelm_gt": ("oracle 记忆上界", None, None),
    "paper_asd_tah_observe_t10": ("TAHRelM 无纠偏 (T10)", "tah_observe_t10", "mem"),
    "paper_asd_tah_tempered_t10": ("TAHRelM+PGIF (T10)", "tah_tempered_t10", "anc"),
    "paper_asd_reid_observe": ("reid 无纠偏", "reid_observe_91000", "mem"),
    "paper_asd_reid_tempered": ("reid+PGIF", "reid_tempered_91000", "anc"),
    "paper_asd_tah_gt_observe": ("TAHRelM gt-seeded 重跑", "tah_gt_observe_91000", "mem"),
    "paper_asd_tah_gt_pgif": ("TAHRelM+PGIF gt-seeded", "tah_gt_pgif_91000", "anc"),
}
RNG = np.random.default_rng(20261002)


def per_seed(stem):
    A = load(stem); lab, jl, fld = ARMS[stem]
    T = aligned(stem, jl) if jl else {}
    out = {}
    for s, e in A.items():
        lc = e.get("max_correct_run_s")
        if lc is None and s in T:
            lc = runs(T[s], fld)[0]
        out[s] = {"correct": 1 - e["mis_follow_inst"], "longW": e["max_wrong_run_s"], "longC": lc,
                  "sr": float(bool(e["success"]))}
    return out


def boot(d, B=20000):
    d = np.asarray(d, float); idx = RNG.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1); return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(stem):
    import os
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "eval_out", stem + ".json")):
        return None
    P = per_seed(stem); r = {"label": ARMS[stem][0], "n": len(P)}
    for k in ("correct", "longC", "longW", "sr"):
        v = [p[k] for p in P.values() if p[k] is not None]
        r[k] = float(np.mean(v)) if v else None
        r[k + "_med"] = float(np.median(v)) if v else None
    return r


def paired(base, arm):
    A, B = per_seed(base), per_seed(arm); S = sorted(set(A) & set(B)); r = {"base": base, "arm": arm, "n": len(S)}
    for k in ("correct", "longC", "longW"):
        ss = [s for s in S if A[s][k] is not None and B[s][k] is not None]
        r[k] = boot([B[s][k] - A[s][k] for s in ss]) if ss else None
    return r


def pooled(pairs):
    res = {}
    for k in ("correct", "longC", "longW"):
        arrs = []
        for b, a in pairs:
            A, B = per_seed(b), per_seed(a)
            arrs.append(np.array([B[s][k] - A[s][k] for s in sorted(set(A) & set(B))
                                  if A[s][k] is not None and B[s][k] is not None]))
        m = np.concatenate(arrs).mean()
        bs = [np.concatenate([x[RNG.integers(0, len(x), len(x))] for x in arrs]).mean() for _ in range(20000)]
        res[k] = [float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res["n"] = int(sum(len(x) for x in arrs)); return res


def cold_start(stem, base):
    """Stratify by whether the arm's own first labelled tick locked the target (correctly aligned seeds)."""
    lab, jl, fld = ARMS[stem]; T = aligned(stem, jl); A, B = per_seed(base), per_seed(stem); res = {}
    cold = {}
    for s, S in T.items():
        L = [r for r in S if r["true"] is not None and r["true"] >= 0]
        if L:
            cold[s] = (L[0][fld] if L[0].get(fld) is not None else L[0]["mem"]) == L[0]["true"]
    for name, keep in (("cold_right", True), ("cold_wrong", False)):
        ss = [s for s in sorted(set(A) & set(B)) if cold.get(s) is keep]
        if ss:
            res[name] = {"n": len(ss), "base_correct": float(np.mean([A[s]["correct"] for s in ss])),
                         "arm_correct": float(np.mean([B[s]["correct"] for s in ss])),
                         "d_correct": boot([B[s]["correct"] - A[s]["correct"] for s in ss])}
    return res


def f(t):
    return "—" if t is None else f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    out = {"summary": {s: summary(s) for s in ARMS}}
    out["summary"] = {k: v for k, v in out["summary"].items() if v is not None}
    for s, r in out["summary"].items():
        print(f"{r['label']:22s} n={r['n']:2d} correct={r['correct']:.3f} longC={r['longC'] if r['longC'] is None else round(r['longC'],1)}s"
              f"(med {r['longC_med'] if r['longC_med'] is None else round(r['longC_med'],1)}) longW={r['longW']:.1f}s(med {r['longW_med']:.1f}) [SR {r['sr']:.3f}]")
    PAIRS = [("paper_asd_tah_observe", "paper_asd_tah_tempered"), ("paper_asd_tah_observe", "paper_ext_tahrelm_deploy"),
             ("paper_asd_tah_observe", "paper_asd_tah_anchor_p3"), ("paper_asd_tah_observe", "paper_asd_tah_bayes"),
             ("paper_asd_tah_observe", "paper_asd_tah_resid"), ("paper_asd_tah_observe", "paper_asd_tah_tempered_nogate"),
             ("paper_asd_tah_bayes", "paper_asd_tah_tempered"), ("paper_asd_tah_resid", "paper_asd_tah_tempered"),
             ("paper_asd_tah_tempered_nogate", "paper_asd_tah_tempered"),
             ("paper_asd_tah_observe", "paper_ext_tahrelm_oracledrift"), ("paper_asd_tah_observe", "paper_ext_tahrelm_gt"),
             ("paper_asd_tah_observe_t10", "paper_asd_tah_tempered_t10"), ("paper_asd_reid_observe", "paper_asd_reid_tempered"),
             ("paper_asd_tah_gt_observe", "paper_asd_tah_gt_pgif"), ("paper_ext_tahrelm_gt", "paper_asd_tah_gt_observe")]
    PAIRS = [(b, x) for b, x in PAIRS if b in out["summary"] and x in out["summary"]]
    out["paired"] = [paired(b, x) for b, x in PAIRS]
    print("\npaired Δ (arm − base), mean [95% CI]:")
    for r in out["paired"]:
        print(f"  {ARMS[r['arm']][0]:16s} vs {ARMS[r['base']][0]:20s} n={r['n']} Δcorrect {f(r['correct'])}  ΔlongC {f(r['longC'])}  ΔlongW {f(r['longW'])}")
    out["pooled"] = pooled([("paper_asd_tah_observe", "paper_asd_tah_tempered"), ("paper_asd_tah_observe_t10", "paper_asd_tah_tempered_t10"),
                            ("paper_asd_reid_observe", "paper_asd_reid_tempered")])
    print("\npooled 3 settings:", {k: (v if k == "n" else [round(x, 3) for x in v]) for k, v in out["pooled"].items()})
    out["cold_start"] = {s: cold_start(s, b) for b, s in [("paper_asd_tah_observe", "paper_asd_tah_tempered"),
                                                         ("paper_asd_tah_observe_t10", "paper_asd_tah_tempered_t10"),
                                                         ("paper_asd_reid_observe", "paper_asd_reid_tempered")]}
    print("\ncold-start strata (correctly aligned):")
    for s, v in out["cold_start"].items():
        print(" ", ARMS[s][0], {k: (x["n"], round(x["base_correct"], 3), round(x["arm_correct"], 3), f(x["d_correct"])) for k, x in v.items()})
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
