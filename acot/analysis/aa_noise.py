"""A/A noise floor (2026-10-04): three replicates of the SAME configuration (final model, sim_proxy_v2, dev 91000–19) —
acot re-run (experiment aa_noise), simultaneous legacy re-run (aa_legacy_{a,b,c}) and the historical legacy run
(zoa_pgif_trk, 10-03). Reports per-replicate means, pairwise paired differences, the run-to-run vs between-seed variance
decomposition (ICC), the minimal detectable effect of a paired single-run comparison and the seeds needed per effect.
  python -m acot.analysis.aa_noise [--md acot_note/1004_aa_noise.md]"""
import argparse
import json
import math
import os

import numpy as np

from acot import CONFIGS, ROOT
from acot.analysis import legacy_import as L
from acot.analysis import metrics as M
from acot.analysis import stats as S
from acot.analysis.results import load
from acot.config.resolve import resolve_experiment

METRICS = ("correct", "correct_s", "longC", "longW")
Z = 1.96 + 0.84                     # two-sided α = 0.05, power 0.8


def replicates():
    R = resolve_experiment(os.path.join(CONFIGS, "experiments", "aa_noise.yaml"))
    rows = load(R)["ours"]
    acot = {s: r["episode"] for s, r in rows.items() if r["status"] == "complete"}
    acot_open = {s: v for s, v in ((s, M.opening_lock(r["asd"])) for s, r in rows.items() if r["status"] == "complete")
                 if v is not None}
    out = {"acot": (acot, acot_open)}
    for name, stems in (("legacy_new", ["aa_legacy_a", "aa_legacy_b", "aa_legacy_c"]),
                        ("legacy_hist", ["zoa_pgif_trk_a", "zoa_pgif_trk_b", "zoa_pgif_trk_c"])):
        E, T = {}, {}
        for st in stems:
            E.update(L.episodes(st)[0])
            T.update(L.ticks(L.tick_pairs(st))[0])
        out[name] = (E, {s: v for s, v in ((s, M.opening_lock(R_)) for s, R_ in T.items()) if v is not None})
    skipped = sorted(s for s, r in rows.items() if r["status"] == "skipped")
    return out, skipped


def analyze():
    reps, skipped = replicates()
    names = list(reps)
    common = sorted(set.intersection(*[set(reps[n][0]) for n in names]))
    res = {"n_common": len(common), "seeds": common, "acot_skipped": skipped, "replicates": {}, "pairs": [],
           "noise": {}, "opening": {}}
    for n in names:
        E = reps[n][0]
        res["replicates"][n] = {m: S.bootstrap_ci([M.value(m, E[s]) for s in common], S.rng_for("aa", n, m))
                                for m in METRICS}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for m in METRICS:
                d = np.array([M.value(m, reps[a][0][s]) - M.value(m, reps[b][0][s]) for s in common], float)
                d = d[np.isfinite(d)]
                rng = S.rng_for("aa", a, b, m)
                mu, lo, hi = S.bootstrap_ci(d, rng)
                res["pairs"].append({"a": a, "b": b, "metric": m, "n": len(d), "delta": [mu, lo, hi],
                                     "p": S.sign_flip_p(d, rng), "sd_diff": float(d.std(ddof=1)),
                                     "mean_abs_diff": float(np.abs(d).mean())})
    for m in METRICS:
        X = np.array([[M.value(m, reps[n][0][s]) for n in names] for s in common], float)   # seeds × replicates
        X = X[np.all(np.isfinite(X), 1)]
        k = X.shape[1]
        within = float(X.var(axis=1, ddof=1).mean())                 # run-to-run variance (pooled over seeds)
        between = float(max(X.mean(1).var(ddof=1) - within / k, 0.0))  # seed-to-seed variance of the true seed means
        icc = between / (between + within) if between + within > 0 else float("nan")
        sd_d = math.sqrt(2 * within)                                  # SD of a paired single-run difference
        res["noise"][m] = {"run_sd": math.sqrt(within), "seed_sd": math.sqrt(between), "icc": icc,
                           "paired_sd": sd_d, "mde_n20": Z * sd_d / math.sqrt(20),
                           "n_for": {str(e): math.ceil((Z * sd_d / e) ** 2) for e in
                                     ((0.05, 0.10, 0.15) if m.startswith("correct") else (2.0, 4.0, 6.0))}}
    ops = [n for n in names]
    agree = [s for s in common if all(s in reps[n][1] for n in ops)]
    res["opening"] = {"n": len(agree), **{n: sum(reps[n][1][s] for s in agree) for n in ops},
                      "all_three_agree": sum(len({reps[n][1][s] for n in ops}) == 1 for s in agree)}
    res["per_seed"] = {s: {n: round(M.value("correct", reps[n][0][s]), 3) for n in names} for s in common}
    return res


def markdown(r):
    lab = {"acot": "acot 重跑", "legacy_new": "旧入口重跑", "legacy_hist": "旧入口历史(10-03)"}
    out = [f"# 1004 · A/A 噪声对照(同一配置三次重复)", "",
           f"最终模型(sim_proxy_v2@1,dev 91000–91019),三份重复的共同种子 n = {r['n_common']}"
           f"(acot 跳过 {r['acot_skipped']})。", "", "## 1. 每份重复的均值 [95% bootstrap CI]", "",
           "| 重复 | " + " | ".join(M.label(m) for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for n, v in r["replicates"].items():
        out.append(f"| {lab[n]} | " + " | ".join(f"{v[m][0]:.3f} [{v[m][1]:.3f}, {v[m][2]:.3f}]" for m in METRICS) + " |")
    out += ["", "## 2. 两两配对差(同一种子)", "", "| 对比 | 指标 | Δ [95% CI] | p | 逐种子差的 SD | 平均绝对差 |",
            "|---|---|---|---|---|---|"]
    for p in r["pairs"]:
        out.append(f"| {lab[p['a']]} − {lab[p['b']]} | {M.label(p['metric'])} | {p['delta'][0]:+.3f} "
                   f"[{p['delta'][1]:+.3f}, {p['delta'][2]:+.3f}] | {p['p']:.3f} | {p['sd_diff']:.3f} | "
                   f"{p['mean_abs_diff']:.3f} |")
    out += ["", "## 3. 噪声分解与检出能力", "",
            "| 指标 | 运行间 SD | 种子间 SD | ICC(种子决定的比例) | 单次配对差 SD | n=20 时最小可检出效应 | 检出所需种子数 |",
            "|---|---|---|---|---|---|---|"]
    for m, v in r["noise"].items():
        need = ", ".join(f"{e}: {n}" for e, n in v["n_for"].items())
        out.append(f"| {M.label(m)} | {v['run_sd']:.3f} | {v['seed_sd']:.3f} | {v['icc']:.2f} | {v['paired_sd']:.3f} | "
                   f"{v['mde_n20']:.3f} | {need} |")
    o = r["opening"]
    out += ["", f"## 4. 开局锁定", "", f"三份重复都有开局记录的种子 {o['n']} 个：" +
            ", ".join(f"{lab[n]} {o[n]}/{o['n']}" for n in ("acot", "legacy_new", "legacy_hist")) +
            f";三份完全一致的种子 {o['all_three_agree']}/{o['n']}。", "",
            "## 5. 逐种子承诺正确率", "", "| 种子 | " + " | ".join(lab[n] for n in r["replicates"]) + " |",
            "|---|" + "---|" * len(r["replicates"])]
    for s, v in r["per_seed"].items():
        out.append(f"| {s} | " + " | ".join(f"{v[n]:.3f}" for n in r["replicates"]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    r = analyze()
    md = markdown(r)
    print(md)
    if a.md:
        open(os.path.join(ROOT, a.md), "w").write(md + "\n")
        json.dump(r, open(os.path.join(ROOT, a.md[:-3] + ".json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
