"""DAM4SAM abstention diagnosis (runs/asd/iter/damdiag_g<gate>_p<k>.jsonl, written with DAM4SAM_DIAG=1).
Per association gate: share of target-present S2 ticks by outcome (pick / gate = nearest candidate beyond the gate /
empty = SAM2 mask empty / no_init), pick accuracy, and for gate-abstentions whether the nearest candidate WAS the target
and how far the predicted centre was from it (→ how many abstentions a wider gate could turn into correct picks).
Three metrics (committed / lenient / strict + longC / longW) per gate with paired deltas vs gate 160 and vs the original
DAM4SAM deployable run (paper_dep_dam4sam, replication check).
  python dam_diag.py [--json eval_out/dam_diag.json]"""
import argparse
import glob
import json
import os

import numpy as np

import iter_gate as g

LG = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "asd", "iter")
GATES = [160, 240, 320]


def ticks(gate):
    out = []
    for f in sorted(glob.glob(os.path.join(LG, f"damdiag_g{gate}_p[0-9].jsonl"))):
        out += [json.loads(line) for line in open(f)]
    return out


def tick_stats(T):
    P = [r for r in T if r["true"] >= 0]                     # target in frame (as a candidate)
    if not P:
        return None
    n = len(P); res: dict = {"ticks_present": n}
    for k in ("pick", "gate", "empty", "no_init"):
        res[f"share_{k}"] = sum(r["reason"] == k for r in P) / n
    pk = [r for r in P if r["reason"] == "pick"]
    res["pick_acc"] = float(np.mean([r["nearest_is_true"] for r in pk])) if pk else float("nan")
    gt = [r for r in P if r["reason"] == "gate"]
    if gt:
        res["gate_nearest_is_true"] = float(np.mean([r["nearest_is_true"] for r in gt]))
        dt = np.array([r["d_true"] for r in gt if r.get("d_true") is not None], float)
        res["gate_d_true_median"] = float(np.median(dt))
        for G in (240, 320, 480):
            res[f"gate_recoverable_at_{G}"] = float(np.mean([r["nearest_is_true"] and r["d_min"] < G for r in gt]))
    for k in ("pick", "gate", "empty"):
        I = [r["iou"] for r in P if r["reason"] == k and r["k"] > 0]
        res[f"iou_med_{k}"] = float(np.median(I)) if I else float("nan")
    # empty-mask dynamics: does SAM2 re-acquire after declaring the target absent? and is it size-related?
    # (empty ticks carry no geometry → proxy size = box area at the most recent successful pick in the episode)
    runs, never_back, eps_with_empty = [], 0, 0
    by_size = {"<300": [0, 0], "300-600": [0, 0], ">=600": [0, 0]}
    E = {}
    for r in T:
        E.setdefault(r["ep"], []).append(r)
    for R in E.values():
        R = sorted(R, key=lambda r: r["k"]); cur = 0; last_area = None; seen_empty = False; back = False
        for r in R:
            if r["true"] < 0:
                continue
            if r["reason"] == "empty":
                cur += 1; seen_empty = True
                if last_area is not None:
                    b = "<300" if last_area < 300 else ("300-600" if last_area < 600 else ">=600"); by_size[b][0] += 1; by_size[b][1] += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
                if r["reason"] == "pick":
                    back = back or seen_empty
                    last_area = r["box_wh"][0] * r["box_wh"][1]
                    b = "<300" if last_area < 300 else ("300-600" if last_area < 600 else ">=600"); by_size[b][1] += 1
        if cur:
            runs.append(cur)
        if seen_empty:
            eps_with_empty += 1; never_back += not back
    res["empty_run_mean_ticks"] = float(np.mean(runs)) if runs else float("nan")
    res["eps_with_empty"] = eps_with_empty
    res["eps_never_reacquire_after_empty"] = never_back
    for b, (e, t) in by_size.items():
        res[f"empty_rate_lastbox_{b}px2"] = e / t if t else float("nan")
    A = [r for r in T if r["true"] < 0]
    res["absent_ticks"] = len(A)
    res["absent_pick_rate"] = float(np.mean([r["reason"] == "pick" for r in A])) if A else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    out = {"ticks": {}, "metrics": {}, "paired": []}
    print("## 弃权原因(目标在候选中的 S2 tick)\n")
    rows: dict = {G: tick_stats(ticks(G)) for G in GATES}
    keys = sorted({k for r in rows.values() if r for k in r}, key=lambda k: (not k.startswith("share"), k))
    print("| 指标 | " + " | ".join(f"gate {G}px" for G in GATES) + " |\n|---|" + "---|" * len(GATES))
    for k in keys:
        print(f"| {k} | " + " | ".join(
            (f"{rows[G][k]:.3f}" if isinstance(rows[G].get(k), float) else str(rows[G].get(k, ""))) if rows[G] else "—"
            for G in GATES) + " |")
    out["ticks"] = {str(G): rows[G] for G in GATES}
    print("\n## 三项指标(Town05,deployable)\n")
    D = {"dep(原 05:06 运行)": g.load("dam4sam")}
    D.update({f"g{G}": g.load(f"damdiag_g{G}") for G in GATES})
    print("| 臂 | n | " + " | ".join(g.LABEL.values()) + " |\n|---|---|" + "---|" * len(g.LABEL))
    for name, A in D.items():
        if A is None:
            print(f"| {name} | pending |"); continue
        m = {k: g.ci_mean([f(e) for e in A.values()]) for k, f in g.METRICS.items()}
        out["metrics"][name] = {"n": len(A), **{k: list(v) for k, v in m.items()}}
        print(f"| {name} | {len(A)} | " + " | ".join(f"{v[0]:.3f} [{v[1]:.2f}, {v[2]:.2f}]" for v in m.values()) + " |")
    print("\n## 配对差(同种子)\n\n| 比较 | 指标 | n | Δ [95% CI] |\n|---|---|---|---|")
    pairs = [(f"g{G}", "g160") for G in (240, 320)] + [("g160", "dep(原 05:06 运行)")]
    for x, y in pairs:
        A, B = D.get(x), D.get(y)
        if A is None or B is None:
            continue
        S = sorted(set(A) & set(B))
        for k, f in g.METRICS.items():
            d = np.array([f(A[s]) - f(B[s]) for s in S], float); d = d[np.isfinite(d)]
            if len(d) < 3:
                continue
            mu, lo, hi = g.ci_mean(d)
            out["paired"].append({"a": x, "b": y, "metric": k, "n": len(d), "delta": [mu, lo, hi]})
            print(f"| {x} − {y} | {g.LABEL[k]} | {len(d)} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
