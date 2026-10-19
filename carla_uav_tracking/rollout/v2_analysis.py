"""1004 deployable-v2 dev comparison (Town05, seeds 91000–19): GT-free tracklet ids + occlusion-aware candidates.
Arms: A v2_pgif_trk (PGIF, default tracker) · B v2_pgif_sim (PGIF, simulator ids) · C v2_tah_trk (no PGIF, default
tracker) · A' v2t_pgif_trk / C' v2t_tah_trk (tuned tracker). Metrics: committed correct-lock ratio over in-frustum frames
(legacy) and over ray-cast VISIBLE frames, strict visible ratio, abstention on visible frames, longest correct/wrong
lock. Paired seed-bootstrap 95% CI (iter_gate.ci_mean) for the pre-stated contrasts.
  python v2_analysis.py [--json eval_out/v2_analysis.json]"""
import argparse
import json

import numpy as np

import iter_gate as g

ARMS = [("A", "v2_pgif_trk", "PGIF · 跟踪器(默认)"), ("B", "v2_pgif_sim", "PGIF · 理想 ID"),
        ("C", "v2_tah_trk", "无 PGIF · 跟踪器(默认)"), ("A'", "v2t_pgif_trk", "PGIF · 跟踪器(调后)"),
        ("C'", "v2t_tah_trk", "无 PGIF · 跟踪器(调后)"), ("V0", "pgif", "旧 V0(理想 ID,无遮挡感知,10-02 批次)")]
M = {"correct": lambda e: 1 - e["mis_follow_sustained"],
     "correct_vis": lambda e: e.get("correct_committed_vis"),
     "strict_vis": lambda e: e.get("correct_strict_vis"),
     "abstain_vis": lambda e: e.get("abstain_rate_vis"),
     "occl_frac": lambda e: (e.get("occluded_frames") / max(e["track_frames"], 1)) if e.get("occluded_frames") is not None else None,
     "longC": lambda e: e["max_correct_run_s"], "longW": lambda e: e["max_wrong_run_s"]}
LAB = {"correct": "正确比例(视锥内·承诺)", "correct_vis": "正确比例(可见帧·承诺)", "strict_vis": "严格(可见帧)",
       "abstain_vis": "弃权率(可见帧)", "occl_frac": "遮挡帧占比", "longC": "最长连续正确(s)", "longW": "最长锁错(s)"}
CONTRASTS = [("A", "C", "PGIF 增益 · 默认跟踪器"), ("A'", "C'", "PGIF 增益 · 调后跟踪器"),
             ("A", "B", "跟踪器 vs 理想 ID(PGIF)"), ("A'", "B", "调后跟踪器 vs 理想 ID(PGIF)"),
             ("A'", "A", "调参效果(PGIF)"), ("C'", "C", "调参效果(无 PGIF)")]


def vals(A, f):
    return [x for x in (f(e) for e in A.values()) if x is not None and np.isfinite(x)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    D = {k: g.load(stem) for k, stem, _ in ARMS}
    out = {"arms": {}, "contrasts": []}
    print("| 臂 | n | " + " | ".join(LAB.values()) + " |\n|---|---|" + "---|" * len(LAB))
    for k, stem, lab in ARMS:
        A = D[k]
        if A is None:
            print(f"| {k} {lab} | pending |"); continue
        row = {}
        cells = []
        for m, f in M.items():
            v = vals(A, f)
            if not v:
                cells.append("—"); continue
            mu, lo, hi = g.ci_mean(v); row[m] = [mu, lo, hi]
            cells.append(f"{mu:.3f}" if m not in ("longC", "longW") else f"{mu:.1f}")
        out["arms"][k] = {"stem": stem, "n": len(A), **row}
        print(f"| {k} {lab} | {len(A)} | " + " | ".join(cells) + " |")
    print("\n| 对比 | 指标 | n | Δ [95% CI] |\n|---|---|---|---|")
    for x, y, lab in CONTRASTS:
        X, Y = D[x], D[y]
        if X is None or Y is None:
            print(f"| {x} − {y} {lab} | — | — | pending |"); continue
        S = sorted(set(X) & set(Y))
        for m in ("correct_vis", "correct", "strict_vis", "longC", "longW"):
            d = [M[m](X[s]) - M[m](Y[s]) for s in S if M[m](X[s]) is not None and M[m](Y[s]) is not None]
            d = np.array(d, float); d = d[np.isfinite(d)]
            if len(d) < 3:
                continue
            mu, lo, hi = g.ci_mean(d)
            out["contrasts"].append({"a": x, "b": y, "label": lab, "metric": m, "n": len(d), "delta": [mu, lo, hi]})
            print(f"| {x} − {y} {lab} | {LAB[m]} | {len(d)} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
