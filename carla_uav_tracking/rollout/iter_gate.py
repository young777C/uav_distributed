"""1003 iteration gate check — pre-registered hypotheses (1003_v1_v2_design.md / 1003_v3_design.md), deployable only,
three objective metrics, paired seed-bootstrap 95% CI on the seeds common to arm and reference.
Arms are read from eval_out/paper_it_<name>.json, else merged from the watchdog parts paper_it_<name>_p*.json[.partial]
(or a single .partial) → works on running arms too (n is printed; verdicts on partial n are provisional).
  python iter_gate.py [--json eval_out/iter_gate.json]"""
import argparse
import glob
import json
import os
import re

import numpy as np

EO = os.path.join(os.path.dirname(__file__), "eval_out")
def _strict(e):
    """per-frame pick == target over target-present frames; ABSTENTION (no pick) counts as NOT correct."""
    nd, bn = e.get("n_dist") or {}, e.get("mis_by_N") or {}
    t = sum(nd.values())
    return 1 - sum(bn[k] * nd[k] for k in nd if k in bn) / t if t else float("nan")


# correct-lock ratio, three definitions (they differ ONLY for trackers that abstain, e.g. DAM4SAM loses its mask):
#   correct   = committed identity == target (abstention holds the last commitment) — SAME notion of lock as longC /
#               longW → primary; correct_l = 1 − mis_follow_inst (abstention counted as CORRECT; the metric used up to
#               1003 and in the pre-registration); correct_s = strict (abstention counted as WRONG).
METRICS = {"correct": lambda e: 1 - e["mis_follow_sustained"],
           "correct_l": lambda e: 1 - e["mis_follow_inst"],
           "correct_s": _strict,
           "longC": lambda e: e["max_correct_run_s"],
           "longW": lambda e: e["max_wrong_run_s"]}
LABEL = {"correct": "正确锁定比例(承诺)", "correct_l": "正确比例(宽松·弃权算对)", "correct_s": "正确比例(严格·弃权算错)",
         "longC": "最长连续正确(s)", "longW": "最长锁错(s)"}
REF = {"dam4sam": ["paper_dep_dam4sam.json"], "pgif": ["paper_dep_tahrelm_pgif.json"],
       "tahrelm": ["paper_dep_tahrelm.json"], "dam4sam_t10": ["paper_it_dam4sam_t10.json"],
       "tahrelm_t03": ["paper_it_tahrelm_t03.json"]}
# (id, arm, reference, metric, kind)  kind: noninf = Δ lower > −0.05 · less = Δ upper < 0 · greater = Δ lower > 0 · desc
TESTS = [
    ("V0 vs DAM4SAM", "pgif", "dam4sam", "correct", "greater"),
    ("V0 vs DAM4SAM", "pgif", "dam4sam", "longC", "greater"),
    ("V0 vs DAM4SAM", "pgif", "dam4sam", "longW", "less"),
    ("H1a 非劣", "v1_t05", "dam4sam", "correct", "noninf"),
    ("H1b-all", "v1_t05", "dam4sam", "longW", "less"),
    ("H1c(探索)", "v1_t05", "dam4sam", "longC", "desc"),
    ("V1 vs commit-only", "v1_t05", "v1nog_t05", "longW", "less"),
    ("V1 vs commit-only", "v1_t05", "v1nog_t05", "correct", "desc"),
    ("H2b", "v2_t05", "pgif", "longW", "less"),
    ("H2c 非劣", "v2_t05", "pgif", "correct", "noninf"),
    ("H3a 非劣", "v3_t05", "dam4sam", "correct", "noninf"),
    ("H3a 锁错", "v3_t05", "dam4sam", "longW", "less"),
    ("H3a 连续正确", "v3_t05", "dam4sam", "longC", "desc"),
    ("H3b", "v3_t05", "pgif", "correct", "greater"),
    ("V3 > ensemble", "v3_t05", "v3ens_t05", "correct", "greater"),
    ("V3 > ensemble", "v3_t05", "v3ens_t05", "longW", "less"),
    ("V3 > commit-only", "v3_t05", "v3com_t05", "correct", "greater"),
    ("V3 > commit-only", "v3_t05", "v3com_t05", "longW", "less"),
    ("V3 vs gateTAH-only", "v3_t05", "v3gt_t05", "longW", "desc"),
    ("H3a 非劣(V3+V2)", "v3v2_t05", "dam4sam", "correct", "noninf"),
    ("H3a 锁错(V3+V2)", "v3v2_t05", "dam4sam", "longW", "less"),
    ("超越 DAM4SAM", "v3_t05", "dam4sam", "correct", "greater"),
    ("超越 DAM4SAM(V3+V2)", "v3v2_t05", "dam4sam", "correct", "greater"),
    ("泛化 T03 正确", "v3_t03", "dam4sam_t03", "correct", "noninf"),
    ("泛化 T03 锁错", "v3_t03", "dam4sam_t03", "longW", "less"),
    ("泛化 T10 正确", "v3_t10", "dam4sam_t10", "correct", "noninf"),
    ("泛化 T10 锁错", "v3_t10", "dam4sam_t10", "longW", "less"),
    ("V0 泛化 T03", "pgif_t03", "tahrelm_t03", "correct", "desc"),
]
RNG = np.random.default_rng(20261003)


def _eps(path):
    return json.load(open(path))["episodes"] if os.path.exists(path) else []


def load(name):
    """seed → episode; first occurrence wins; skipped / unscored episodes dropped."""
    if name == "dam4sam_t03":   # pre-reboot part (91000–06) + post-reboot remainder (91007–19)
        files = [os.path.join(EO, "paper_it_dam4sam_t03_part1.json")] + _parts("dam4sam_t03b")
    elif name in REF:
        files = [os.path.join(EO, f) for f in REF[name]]
    else:
        files = _parts(name)
    out = {}
    for f in files:
        for e in _eps(f):
            if e.get("mis_follow_inst") is None or e.get("skipped"):
                continue
            out.setdefault(int(e["episode_seed"]), e)
    return out or None


def _parts(name):
    full = os.path.join(EO, f"paper_it_{name}.json")
    if os.path.exists(full):
        return [full]
    def part_no(p):
        m = re.search(r"_p(\d+)\.json", p)
        assert m is not None, p
        return m.group(1)
    parts = sorted(glob.glob(os.path.join(EO, f"paper_it_{name}_p[0-9]*.json*")), key=lambda p: int(part_no(p)))
    keep = {}
    for p in parts:   # prefer the final .json of a part over its .partial
        k = part_no(p)
        if k not in keep or not p.endswith(".partial"):
            keep[k] = p
    if keep:
        return [keep[k] for k in sorted(keep, key=int)]
    return [full + ".partial"]


def ci_mean(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]   # strict ratio is NaN for an episode without scored frames
    b = x[RNG.integers(0, len(x), (20000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(kind, lo, hi):
    if kind == "noninf":
        return "✓" if lo > -0.05 else "✗"
    if kind == "less":
        return "✓" if hi < 0 else "✗"
    if kind == "greater":
        return "✓" if lo > 0 else "✗"
    return "—"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    names = sorted({t[1] for t in TESTS} | {t[2] for t in TESTS})
    D = {n: load(n) for n in names}
    res = {"arms": {}, "tests": []}
    print("| 臂 | n | " + " | ".join(LABEL.values()) + " |\n|---|---|" + "---|" * len(LABEL))
    for n in names:
        A = D[n]
        if A is None:
            continue
        m = {k: ci_mean([f(e) for e in A.values()]) for k, f in METRICS.items()}
        res["arms"][n] = {"n": len(A), **{k: list(v) for k, v in m.items()}}
        print(f"| {n} | {len(A)} | " + " | ".join(f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for v in m.values()) + " |")
    print("\n| 假设 | 臂 − 参照 | 指标 | n 配对 | Δ [95% CI] | 判定 |\n|---|---|---|---|---|---|")
    tests = []
    for t in TESTS:   # every correct-ratio hypothesis is reported under all three definitions
        tests += [(t[0], t[1], t[2], k, t[4]) for k in ("correct", "correct_l", "correct_s")] if t[3] == "correct" else [t]
    for tid, arm, ref, k, kind in tests:
        A, B = D[arm], D[ref]
        if A is None or B is None:
            print(f"| {tid} | {arm} − {ref} | {LABEL[k]} | — | pending | — |"); continue
        S = sorted(set(A) & set(B))
        if len(S) < 3:
            print(f"| {tid} | {arm} − {ref} | {LABEL[k]} | {len(S)} | pending | — |"); continue
        d = np.array([METRICS[k](A[s]) - METRICS[k](B[s]) for s in S])
        mu, lo, hi = ci_mean(d)
        v = verdict(kind, lo, hi) + ("(暂)" if len(S) < 19 else "")
        print(f"| {tid} | {arm} − {ref} | {LABEL[k]} | {len(S)} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] | {v} |")
        res["tests"].append({"id": tid, "arm": arm, "ref": ref, "metric": k, "kind": kind, "n": len(S),
                             "delta": [mu, lo, hi], "verdict": v})
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
