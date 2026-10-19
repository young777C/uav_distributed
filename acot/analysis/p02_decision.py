"""P0-2 identity retrain gate (acot_note/1012_p02_identity_retrain.md §3, fixed 2026-10-12 before any closed-loop run):
dev-B 40 seeds, protocol @3, r = 2 (per-seed mean of repeat 0 and repeat 1), seed-paired difference new − existing Ours
(ours_final + conf_tau 0.05 with runs/tah_relm_dag.pt: opt1_screen_devB3 tau005 = repeat 0, opt2a_screen_devB3
tau005_r1 = repeat 1). Same rule as the DINOv3 stage 2 (acot/analysis/dinov3_decision.py): PASS iff
① pc_correct Δ ≥ +0.05 and its 95 % bootstrap CI lower bound > 0, ② TR90 Δ CI lower bound > −0.02,
③ p95 of t_identity_ms < 100 ms. PASS → the new weights replace the paper's Ours (old weights kept as a row).
  python -m acot.analysis.p02_decision --stage a
Stage B (1012 §6.4, fixed 2026-10-14 before collecting the @3 DAgger): stage b = the same gate for ours_fast_b
(p02_fast_b_devB3 fb_r0 / fb_r1) against the existing Ours; b_vs_a (secondary, reported only) = fast_b against the
stage-A model (p02_fast_a_devB3 fa_r0 / fa_r1). Both also report the paired difference of the committed-WRONG frame
fraction (pc_wrong_frac), the question stage B asks (stage A's gain came from fewer uncommitted frames)."""
import argparse
import json
import os

import numpy as np

from acot import ROOT
from acot.analysis import metrics as M
from acot.analysis import stats as S
from acot.analysis.results import load
from acot.config.resolve import resolve_experiment

E = "acot/configs/experiments/"
REF = ("existing", [("opt1_screen_devB3", "tau005"), ("opt2a_screen_devB3", "tau005_r1")])
STAGES = {"a": ("fast_a", [("p02_fast_a_devB3", "fa_r0"), ("p02_fast_a_devB3", "fa_r1")]),
          # secondary (declared before running): commit gate matched to the paper weights' block rate
          "a_cal": ("fast_a_cal", [("p02_fast_a_devB3", "fa_cal_r0"), ("p02_fast_a_devB3", "fa_cal_r1")]),
          "b": ("fast_b", [("p02_fast_b_devB3", "fb_r0"), ("p02_fast_b_devB3", "fb_r1")]),
          "b_vs_a": ("fast_b", [("p02_fast_b_devB3", "fb_r0"), ("p02_fast_b_devB3", "fb_r1")])}
REFS = {"b_vs_a": ("fast_a", [("p02_fast_a_devB3", "fa_r0"), ("p02_fast_a_devB3", "fa_r1")])}     # default: REF
WRONG = ("b", "b_vs_a")          # stages that also report pc_wrong_frac (added for stage B; a / a_cal outputs unchanged)
METRICS = ("pc_correct", "tr90", "auc_sr", "pc_longW")


def decide(stage):
    cache = {}

    def rows(exp, arm):
        if exp not in cache:
            cache[exp] = load(resolve_experiment(os.path.join(ROOT, E + exp + ".yaml")))
        return cache[exp][arm]

    def one(m, e):
        if m == "pc_wrong_frac":                          # committed-wrong frames / all frames the scorer counted
            n = e["pc_correct_frames"] + e["pc_wrong_frames"] + e["pc_none_frames"]
            return e["pc_wrong_frames"] / n if n else None
        return M.value(m, e)

    def vals(exp, arm, m):
        return {s: one(m, r["episode"]) for s, r in rows(exp, arm).items()
                if r["status"] == "complete" and r["episode"] and one(m, r["episode"]) is not None}

    def mean_r2(arms, m):
        a, b = (vals(e, x, m) for e, x in arms)
        return {s: (a[s] + b[s]) / 2 for s in set(a) & set(b)}

    (ln, new), (lr, ref) = STAGES[stage], REFS.get(stage, REF)
    out = {"stage": stage, "new": [ln, new], "ref": [lr, ref], "metrics": {}, "timing": {}}
    for m in METRICS + (("pc_wrong_frac",) if stage in WRONG else ()):
        x, y = mean_r2(new, m), mean_r2(ref, m)
        seeds = sorted(set(x) & set(y))
        d = np.array([x[s] - y[s] for s in seeds])
        mu, lo, hi = S.bootstrap_ci(d, S.rng_for("p02_decision", stage, m))
        p = S.sign_flip_p(d, S.rng_for("p02_decision", stage, m, "p"))
        out["metrics"][m] = {"n": len(seeds), ln: float(np.mean([x[s] for s in seeds])),
                             lr: float(np.mean([y[s] for s in seeds])), "diff": mu, "ci95": [lo, hi], "p": p}
    for e, arm in new + ref:
        t = [k["t_identity_ms"] for r in rows(e, arm).values() if r["status"] == "complete"
             for k in r["ticks"] if "t_identity_ms" in k]
        out["timing"][arm] = {"mean_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95)), "n_ticks": len(t)}
    pc, tr = out["metrics"]["pc_correct"], out["metrics"]["tr90"]
    out["criteria"] = {"pc_correct": bool(pc["diff"] >= 0.05 and pc["ci95"][0] > 0),
                       "tr90_noninferior": bool(tr["ci95"][0] > -0.02),
                       "identity_time_p95_lt_100ms": bool(max(out["timing"][a]["p95_ms"] for _, a in new) < 100.0)}
    out["pass"] = all(out["criteria"].values())
    if stage in REFS:                                     # not a gate: the pre-registered gate is against the existing Ours
        out["note"] = "secondary analysis (1012 §6.4) — reported only; criteria / pass are informational"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    a = ap.parse_args()
    out = decide(a.stage)
    p = os.path.join(ROOT, "runs", "p02_retrain", f"decision_{a.stage}.json")
    json.dump(out, open(p, "w"), indent=1)
    (ln, _), (lr, _) = STAGES[a.stage], REFS.get(a.stage, REF)
    for m, r in out["metrics"].items():
        print(f"{m:10s} n={r['n']}  {lr} {r[lr]:.3f}  {ln} {r[ln]:.3f}  Δ {r['diff']:+.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  p={r['p']:.3f}")
    for arm, t in out["timing"].items():
        print(f"t_identity {arm:14s} mean {t['mean_ms']:.1f} ms  p95 {t['p95_ms']:.1f}")
    print("criteria", out["criteria"], "→ P0-2 stage", a.stage, "PASS" if out["pass"] else "FAIL", f"→ {p}")


if __name__ == "__main__":
    main()
