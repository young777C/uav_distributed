"""1006 §8 confirmatory analysis (G11). An analysis spec (acot/configs/analysis/*.yaml) names the experiments, the
LOGICAL arms (each = its r repeats, one physical arm per repeat), the pre-registered families and the decision rule:

  unit          scene seed; per seed the mean over the logical arm's repeats (a seed counts only if every repeat of
                every arm of the table completed — the table's intersection); mechanism tables also drop every seed
                whose shared opening fell back in ANY arm (1006 §3.4 rule 5)
  assertions    every physical arm runs the expected protocol (name@version); non-diagnostic arms use no GT channel;
                every complete seed has a gt_access record without violations — any failure aborts the analysis
  test          left − right per seed; mean, 95% paired bootstrap CI (10,000; stratified by town), two-sided sign-flip
                permutation p (100,000; flips within town); Holm within a family
  decision      primary (pc_correct) Holm p < 0.05 AND mean > 0 AND the TR90 non-inferiority gate
                (95% CI lower bound of TR90(left − right) > −δ, δ = 0.02)
  descriptive   secondary / task metrics: effect + CI only; strata by the reference-opening correctness (if given)
  python -m acot.analysis.confirm acot/configs/analysis/<spec>.yaml [--allow-test]   → <out>.{md,json}"""
import argparse
import json
import os

import numpy as np

from acot import ROOT
from acot.analysis import metrics as M
from acot.analysis import stats as St
from acot.analysis.results import load
from acot.config.load import read_yaml
from acot.config.resolve import resolve_experiment

N_BOOT, N_PERM = 10000, 100000


class AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- statistics
def stratified_boot_ci(d, strata, rng, b=N_BOOT):
    d, strata = np.asarray(d, float), np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    tot = len(d)
    means = np.zeros(b)
    for g in groups:
        means += d[g][rng.integers(0, len(g), (b, len(g)))].sum(1)
    means /= tot
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signflip_p(d, rng, b=N_PERM):
    """two-sided; sign flips are per seed, hence automatically within town (stratified)."""
    d = np.asarray(d, float)
    obs = abs(d.sum())
    hits, chunk = 0, 10000
    for i in range(0, b, chunk):
        n = min(chunk, b - i)
        hits += int((np.abs((d[None] * rng.choice([-1.0, 1.0], (n, len(d)))).sum(1)) >= obs - 1e-12).sum())
    return hits / b


def compare(left, right, seeds, towns, metric, rng):
    d = np.array([left[s][metric] - right[s][metric] for s in seeds], float)
    m, lo, hi = stratified_boot_ci(d, [towns[s] for s in seeds], rng)
    return {"metric": metric, "n": len(seeds), "mean": m, "ci": [lo, hi], "p": signflip_p(d, rng)}


def decide(results, delta, alpha=0.05):
    """Holm on the primary p within the family; decision per §8.4."""
    ps = [r["primary"]["p"] for r in results]
    adj = St.holm(ps)
    for r, a in zip(results, adj):
        r["primary"]["p_holm"] = a
        tr = r.get("safety")
        r["safety_ok"] = tr is None or tr["ci"][0] > -delta
        r["confirmed"] = bool(a < alpha and r["primary"]["mean"] > 0 and r["safety_ok"])
    return results


# ------------------------------------------------------------------------------------------------------- loading
def seed_values(row, metrics):
    e = row["episode"] or {}
    return {m: M.value(m, e) for m in metrics}


def collect(spec, allow_test):
    """→ {logical arm: {seed: {metric: mean over repeats}}}, {seed: town}, {seed: fallback?}, audit notes."""
    metrics = [spec["primary"], spec["safety"]["metric"]] + list(spec.get("secondary", []))
    exps = {}
    for key, path in spec["experiments"].items():
        R = resolve_experiment(os.path.join(ROOT, path), allow_test=allow_test)
        exps[key] = (R, load(R))
    want = spec["protocol"]
    per, towns, fallback = {}, {}, {}
    for la, phys in spec["arms"].items():
        reps = []
        for ref in phys:
            ek, arm = ref.split(":")
            R, D = exps[ek]
            A = R["arms"][arm]
            got = f"{A['protocol']['name']}@{A['protocol']['version']}"
            if got != want:
                raise AnalysisError(f"{ref}: protocol {got}, the analysis requires {want}")
            if A["uses_gt"] and A["role"] != "diagnostic":
                raise AnalysisError(f"{ref}: non-diagnostic arm uses GT channels {A['uses_gt']}")
            rows = {}
            for s, r in D[arm].items():
                towns[s] = next((x["town"] for x in R["seeds"] if x["seed"] == s), "?")
                if r["status"] != "complete":
                    continue
                g = r.get("gt_access")
                if g is None or g.get("violations"):
                    raise AnalysisError(f"{ref} seed {s}: gt_access record missing or with violations")
                if any(t.get("open_fallback") for t in r["ticks"]):
                    fallback[s] = True
                rows[s] = seed_values(r, metrics)
            reps.append(rows)
        common = set.intersection(*[set(r) for r in reps]) if reps else set()
        per[la] = {s: {m: float(np.mean([r[s][m] for r in reps])) if all(r[s][m] is not None for r in reps)
                       else float("nan") for m in metrics} for s in common}
    return per, towns, fallback


def run(spec_path, allow_test=False):
    spec = read_yaml(spec_path)
    per, towns, fallback = collect(spec, allow_test)
    out = {"spec": spec_path, "tables": []}
    delta = float(spec["safety"]["delta"])
    for tab in spec["tables"]:
        arms = sorted({a for c in tab["comparisons"] for a in c})
        seeds = sorted(set.intersection(*[set(per[a]) for a in arms]))
        dropped_fb = []
        if tab.get("mechanism"):
            dropped_fb = [s for s in seeds if fallback.get(s)]
            seeds = [s for s in seeds if not fallback.get(s)]
        rng = St.rng_for(spec_path, tab["name"])
        res = []
        for left, right in tab["comparisons"]:
            r = {"left": left, "right": right,
                 "primary": compare(per[left], per[right], seeds, towns, spec["primary"], rng),
                 "secondary": [compare(per[left], per[right], seeds, towns, m, rng) for m in spec.get("secondary", [])]}
            r["safety"] = compare(per[left], per[right], seeds, towns, spec["safety"]["metric"], rng)
            res.append(r)
        if tab.get("confirmatory", True):
            decide(res, delta)
        strata = None
        if spec.get("strata"):
            ref = json.load(open(os.path.join(ROOT, spec["strata"])))
            strata = {}
            for a in arms:
                for k in (True, False):
                    ss = [s for s in seeds if ref.get(str(s)) is k]
                    strata.setdefault(a, {})[str(k)] = float(np.nanmean([per[a][s][spec["primary"]] for s in ss])) \
                        if ss else None
        out["tables"].append({"name": tab["name"], "n": len(seeds), "seeds": seeds, "dropped_fallback": dropped_fb,
                              "confirmatory": tab.get("confirmatory", True), "results": res, "strata": strata})
    return out


def markdown(o):
    L = [f"# 1006 confirmatory analysis — {o['spec']}", ""]
    for t in o["tables"]:
        L += [f"## {t['name']} (n = {t['n']}; fallback dropped {len(t['dropped_fallback'])})", "",
              "| left − right | primary Δ [95% CI] | p | Holm p | TR90 Δ [95% CI] | non-inferior | confirmed |",
              "|---|---|---|---|---|---|---|"]
        for r in t["results"]:
            p, s = r["primary"], r["safety"]
            L.append(f"| {r['left']} − {r['right']} | {p['mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] | "
                     f"{p['p']:.4f} | {p.get('p_holm', float('nan')):.4f} | {s['mean']:+.3f} [{s['ci'][0]:+.3f}, "
                     f"{s['ci'][1]:+.3f}] | {r.get('safety_ok', '—')} | {r.get('confirmed', '—')} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    o = run(a.spec, a.allow_test)
    out = a.out or os.path.splitext(a.spec)[0].replace("configs/analysis", "analysis_out")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(o, open(out + ".json", "w"), indent=1, default=float)
    open(out + ".md", "w").write(markdown(o))
    print(markdown(o))


if __name__ == "__main__":
    main()
