"""1006 DINOv3 backbone decision — two-stage rule fixed 2026-10-06 23:17 before the repeat-1 results (1006_t0_log.md).
Both stages: dev-B 40 seeds, r = 2 (per-seed mean of repeat 0 and repeat 1), seed-paired difference new − reference;
PASS iff  ① pc_correct Δ ≥ +0.05 and its 95 % bootstrap CI lower bound > 0,  ② TR90 Δ CI lower bound > −0.02,
③ identity-module time ≪ the 0.6 s decision period (operationalised: p95 of t_identity_ms < 100 ms).
  stage 1  ours_v3off − ours_v2off   (data-matched offline-only, protocol @2: t0_dinov3_devB + t0_dinov3_r1_devB)
           PASS → invest in a DINOv3 DAgger round + full-recipe retrain (stage 2); FAIL → DINOv3 = ablation row only
  stage 2  ours_v3full − existing weights   (protocol @3: t0_dinov3_stage2_devB3 v3full_r0 / v3full_r1 vs
           t0_pilot_deploy_devB3 aa_r0a / aa_r1a) PASS → replace the main model (DINOv2 kept as an ablation row)
  python -m acot.analysis.dinov3_decision --stage 1|2"""
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
STAGES = {  # (label, [(experiment, arm) repeat 0, repeat 1]) new, reference
    1: (("v3off", [("t0_dinov3_devB", "ours_v3off"), ("t0_dinov3_r1_devB", "ours_v3off_r1")]),
        ("v2off", [("t0_dinov3_devB", "ours_v2off"), ("t0_dinov3_r1_devB", "ours_v2off_r1")])),
    2: (("v3full", [("t0_dinov3_stage2_devB3", "v3full_r0"), ("t0_dinov3_stage2_devB3", "v3full_r1")]),
        ("existing", [("t0_pilot_deploy_devB3", "aa_r0a"), ("t0_pilot_deploy_devB3", "aa_r1a")])),
}
METRICS = ("pc_correct", "tr90", "auc_sr", "pc_longW")


def decide(stage, cache=None):
    cache = {} if cache is None else cache

    def rows(exp, arm):
        if exp not in cache:
            cache[exp] = load(resolve_experiment(os.path.join(ROOT, E + exp + ".yaml")))
        return cache[exp][arm]

    def vals(exp, arm, m):
        return {s: M.value(m, r["episode"]) for s, r in rows(exp, arm).items()
                if r["status"] == "complete" and r["episode"] and M.value(m, r["episode"]) is not None}

    def mean_r2(arms, m):
        a, b = (vals(e, x, m) for e, x in arms)
        return {s: (a[s] + b[s]) / 2 for s in set(a) & set(b)}

    (ln, new), (lr, ref) = STAGES[stage]
    out = {"stage": stage, "new": [ln, new], "ref": [lr, ref], "metrics": {}, "timing": {}}
    for m in METRICS:
        x, y = mean_r2(new, m), mean_r2(ref, m)
        seeds = sorted(set(x) & set(y))
        d = np.array([x[s] - y[s] for s in seeds])
        mu, lo, hi = S.bootstrap_ci(d, S.rng_for("dinov3_decision", stage, m))
        p = S.sign_flip_p(d, S.rng_for("dinov3_decision", stage, m, "p"))
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    out = decide(a.stage)
    p = os.path.join(ROOT, "runs", "acot_logs", "t0", f"dinov3_stage{a.stage}.json")
    json.dump(out, open(p, "w"), indent=1)
    (ln, _), (lr, _) = STAGES[a.stage]
    for m, r in out["metrics"].items():
        print(f"{m:10s} n={r['n']}  {lr} {r[lr]:.3f}  {ln} {r[ln]:.3f}  Δ {r['diff']:+.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  p={r['p']:.3f}")
    for arm, t in out["timing"].items():
        print(f"t_identity {arm:14s} mean {t['mean_ms']:.1f} ms  p95 {t['p95_ms']:.1f}")
    print("criteria", out["criteria"], "→ STAGE", a.stage, "PASS" if out["pass"] else "FAIL", f"→ {p}")


if __name__ == "__main__":
    main()
