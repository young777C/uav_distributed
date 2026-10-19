"""`acot analyze <experiment.yaml>` (framework §12): completeness (fail closed), per-arm metrics with seed-bootstrap CIs,
opening lock, GT-fence status, and the experiment's pre-declared comparisons (paired bootstrap Δ, sign-flip p, Holm
within a family; diagnostic arms only in family `diagnostic`) → <experiment>/report/report.{md,json}."""
import json
import os

import numpy as np

from acot.analysis import metrics as M
from acot.analysis import stats as St
from acot.analysis.results import completeness, exp_dir, load
from acot.config.resolve import resolve_experiment

METRICS = ("correct", "correct_l", "correct_s", "longC", "longW")


def opening(arm, rows, lock):
    out = {}
    for s, r in rows.items():
        if r["status"] != "complete":
            continue
        v = M.opening_lock(r["asd"], lock) if r["asd"] else M.opening_lock_ticks(r["ticks"], lock)
        if v is not None:
            out[s] = v
    return out


def check_families(R):
    """a diagnostic arm (uses declared GT channels) may only appear in comparison family `diagnostic` (§7.3 P0)."""
    for c in R["comparisons"]:
        fam = c["family"]
        for arm in [c["a"]] + (c["b"] if isinstance(c["b"], list) else [c["b"]]):
            if R["arms"][arm]["role"] == "diagnostic" and fam != "diagnostic":
                raise ValueError(f"comparison family {fam!r} contains the diagnostic arm {arm!r}")


def analyze(path, allow_test=False, force=False):
    R = resolve_experiment(path, allow_test=allow_test)
    check_families(R)
    data = load(R)
    ok, probs = completeness(R, data)
    rep = {"experiment": R["name"], "protocol": R["protocol_ref"], "completeness_ok": ok, "problems": probs,
           "arms": {}, "comparisons": []}
    lines = [f"# {R['name']} — protocol {R['protocol_ref']} (sim-proxy perception: "
             f"{', '.join(next(iter(R['arms'].values()))['protocol']['privileged_perception'])})", ""]
    if not ok and not force:
        lines += ["**INCOMPLETE — no statistics reported (fail closed).**", "", "| arm | seed | status | reason |",
                  "|---|---|---|---|"] + [f"| {p['arm']} | {p['seed']} | {p['status']} | {p['reason']} |" for p in probs]
        return rep, "\n".join(lines)
    pairing = R["completeness"].get("pairing", "strict")
    lines += ["| arm | role | n | " + " | ".join(M.label(m) for m in METRICS) + " | opening lock | GT fence |",
              "|---|---|---|" + "---|" * len(METRICS) + "---|---|"]
    for arm, rows in data.items():
        a = R["arms"][arm]
        done = {s: r for s, r in rows.items() if r["status"] == "complete"}
        lock = a["protocol"]["sim"]["s2_period"] * (a["protocol"]["frontend"]["opening"]["window"] - 1)
        op = opening(arm, done, lock)
        row = {"role": a["role"], "n": len(done), "uses_gt": a["uses_gt"], "config_hash": a["config_hash"]}
        cells = []
        for m in METRICS:
            v = [M.value(m, r["episode"]) for r in done.values()]
            mu, lo, hi = St.bootstrap_ci(v, St.rng_for(R["name"], arm, m)) if v else (float("nan"),) * 3
            row[m] = [mu, lo, hi]
            cells.append(f"{mu:.3f} [{lo:.3f}, {hi:.3f}]")
        k, n = sum(op.values()), len(op)
        lo, hi = St.wilson(k, n)
        row["opening_lock"] = [k, n, lo, hi]
        viol = sum(bool(r["gt_access"] and (r["gt_access"]["violations"] or r["gt_access"]["unknown_sites"]))
                   for r in done.values())
        fence = "audited, 0 violations" if all(r["gt_access"] for r in done.values()) and not viol else \
            ("not audited" if not any(r["gt_access"] for r in done.values()) else f"{viol} seed(s) with violations")
        row["gt_fence"] = fence
        rep["arms"][arm] = row
        lines.append(f"| {arm} | {a['role']} | {len(done)} | " + " | ".join(cells) +
                     f" | {k}/{n} [{lo:.2f}, {hi:.2f}] | {fence} |")
    for c in R["comparisons"]:
        fam, a0, bs = c["family"], c["a"], c["b"] if isinstance(c["b"], list) else [c["b"]]
        res = []
        for b in bs:
            A = {s: r for s, r in data[a0].items() if r["status"] == "complete"}
            B = {s: r for s, r in data[b].items() if r["status"] == "complete"}
            if pairing == "strict" and set(A) != set(B):
                raise ValueError(f"{a0} vs {b}: complete seed sets differ (pairing strict)")
            common = sorted(set(A) & set(B))
            for m in c["metrics"]:
                d = np.array([M.value(m, A[s]["episode"]) - M.value(m, B[s]["episode"]) for s in common], float)
                d = d[np.isfinite(d)]
                rng = St.rng_for(R["name"], fam, a0, b, m)
                mu, lo, hi = St.bootstrap_ci(d, rng)
                p = St.sign_flip_p(d, rng)
                res.append({"family": fam, "a": a0, "b": b, "metric": m, "n": len(d), "delta": [mu, lo, hi], "p": p})
        adj = St.holm([r["p"] for r in res])
        for r, q in zip(res, adj):
            r["p_holm"] = q
        rep["comparisons"] += res
    if rep["comparisons"]:
        lines += ["", "| family | a − b | metric | n | Δ [95% CI] | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['family']} | {r['a']} − {r['b']} | {M.label(r['metric'])} | {r['n']} | {r['delta'][0]:+.3f} "
                  f"[{r['delta'][1]:+.3f}, {r['delta'][2]:+.3f}] | {r['p']:.3f} | {r['p_holm']:.3f} |"
                  for r in rep["comparisons"]]
    return rep, "\n".join(lines)


def cmd(a):
    rep, md = analyze(a.experiment, allow_test=a.allow_test, force=a.force)
    R = resolve_experiment(a.experiment, allow_test=a.allow_test)
    out = os.path.join(exp_dir(R), "report")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "report.md"), "w").write(md + "\n")
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print(md)
    return 0 if rep["completeness_ok"] else 1
