"""1006 §7.3: freeze the number of repeats r from the dev-B pilots (primary metric pc_correct).
  V₂     per-seed paired difference (left − right) of the r = 2 repeat means, sample variance; 80% one-sided UPPER
         confidence bound (chi-square); computed per comparison, the largest per experiment type is used
  V_run  A/A: difference of two runs of the same arm under DIFFERENT repeat ids (aa_r0a − aa_r1a, aa_r0b − aa_r1b,
         pooled) = 2σ²_run; 80% one-sided LOWER bound. Same-repeat pairs (aa_r0a − aa_r0b, aa_r1a − aa_r1b) are
         reported as the simulator-side share (CARLA non-determinism under identical policy RNG)
  V(r) = V₂ − V_run/2 + V_run/r;  ρ = V_run / 0.38² (≤ 1)
  SD_freeze(r) = max( √V(r), 0.38·√(1 − ρ + ρ/r), √(V_run/r) )
  MDE(r) = (z_{1−α/2} + z_{0.8}) · SD_freeze(r) / √n, n = 80; table 1 family A α = 0.025, table 2 α = 0.010
  r* = the smallest r ∈ {2, 3, 4} with MDE ≤ 0.10 for both; none → the §7.4 downgrade.
  python -m acot.analysis.freeze_r [--deploy <exp.yaml> --mech <exp.yaml> --out <json>]   (defaults: the @2 pilots;
  protocol @3 pilots: t0_pilot_deploy_devB3 / t0_pilot_mech_devB3 → freeze_r_p3.json)"""
import argparse
import json
import math
import os

import numpy as np
from scipy import stats as SS

from acot import ROOT
from acot.analysis import metrics as M
from acot.analysis.results import load
from acot.config.resolve import resolve_experiment

DEPLOY = "acot/configs/experiments/t0_pilot_deploy_devB.yaml"
MECH = "acot/configs/experiments/t0_pilot_mech_devB.yaml"
METRIC = "pc_correct"
N_TEST, TARGET = 80, 0.10
ALPHA = {"table1_famA": 0.025, "table2": 0.010}


def vals(rows):
    return {s: M.value(METRIC, r["episode"]) for s, r in rows.items()
            if r["status"] == "complete" and r["episode"] and M.value(METRIC, r["episode"]) is not None}


def var_ci(d, conf=0.80):
    d = np.asarray(d, float)
    n = len(d)
    v = float(d.var(ddof=1))
    ucb = (n - 1) * v / SS.chi2.ppf(1 - conf, n - 1)
    lcb = (n - 1) * v / SS.chi2.ppf(conf, n - 1)
    return v, lcb, ucb, n


def paired(a, b):
    S = sorted(set(a) & set(b))
    return np.array([a[s] - b[s] for s in S]), S


def mean2(x, y):
    return {s: (x[s] + y[s]) / 2 for s in set(x) & set(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default=DEPLOY)
    ap.add_argument("--mech", default=MECH)
    ap.add_argument("--out", default="runs/acot_logs/t0/freeze_r.json")
    args = ap.parse_args()
    Dd = load(resolve_experiment(os.path.join(ROOT, args.deploy)))
    Dm = load(resolve_experiment(os.path.join(ROOT, args.mech)))
    v = {k: vals(r) for k, r in {**Dd, **Dm}.items()}
    out = {"metric": METRIC, "n_test": N_TEST, "experiments": {"deploy": args.deploy, "mech": args.mech}}
    # A/A
    dr = np.concatenate([paired(v["aa_r0a"], v["aa_r1a"])[0], paired(v["aa_r0b"], v["aa_r1b"])[0]])
    ds = np.concatenate([paired(v["aa_r0a"], v["aa_r0b"])[0], paired(v["aa_r1a"], v["aa_r1b"])[0]])
    vrun, vrun_lcb, vrun_ucb, nrun = var_ci(dr)
    vsim, _, _, nsim = var_ci(ds)
    out["A/A"] = {"V_run": vrun, "V_run_lcb80": vrun_lcb, "n_pairs": nrun, "same_repeat_var": vsim, "n_same": nsim}
    ours = mean2(v["aa_r0a"], v["aa_r1a"])
    comps = {"deploy": [("ours", ours, "w/o PGIF", mean2(v["nopgif_r0"], v["nopgif_r1"])),
                        ("ours", ours, "w/o opening", mean2(v["noopen_r0"], v["noopen_r1"]))],
             "mech": [("full", mean2(v["full_sh_r0"], v["full_sh_r1"]), "w/o PGIF",
                       mean2(v["nopgif_sh_r0"], v["nopgif_sh_r1"]))]}
    z = {k: SS.norm.ppf(1 - a / 2) + SS.norm.ppf(0.8) for k, a in ALPHA.items()}
    rho = min(1.0, vrun_lcb / 0.38 ** 2)
    out["rho"] = rho
    for typ, cs in comps.items():
        V2s = []
        rep = []
        for la, a, lb, b in cs:
            d, S = paired(a, b)
            v2, _, v2u, n = var_ci(d)
            V2s.append(v2u)
            rep.append({"cmp": f"{la} − {lb}", "n": n, "mean": float(d.mean()), "V2": v2, "V2_ucb80": v2u})
        V2 = max(V2s)
        fr = {}
        for r in (2, 3, 4):
            Vr = max(V2 - vrun_lcb / 2 + vrun_lcb / r, 0.0)
            sd = max(math.sqrt(Vr), 0.38 * math.sqrt(1 - rho + rho / r), math.sqrt(vrun_lcb / r))
            fr[r] = {"SD_freeze": sd, **{f"MDE_{k}": z[k] * sd / math.sqrt(N_TEST) for k in z}}
        out[typ] = {"comparisons": rep, "V2_used": V2, "by_r": fr}
    fam = {"table1_famA": "deploy", "table2": "mech"}
    r_star = next((r for r in (2, 3, 4) if all(out[fam[k]]["by_r"][r][f"MDE_{k}"] <= TARGET for k in fam)), None)
    out["r_star"] = r_star
    out["decision"] = (f"freeze r = {r_star}" if r_star else
                       "no r ≤ 4 reaches MDE ≤ 0.10 → §7.4: H-A1 and H-P1–P5 become secondary (effect + CI only)")
    p = os.path.join(ROOT, args.out)
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
