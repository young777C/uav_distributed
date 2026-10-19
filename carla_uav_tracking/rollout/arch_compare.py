"""fig_architecture_legacy_pgif.png (old, formerly fig_architecture.png) → fig_architecture_v1.png (new): performance along the evolution path, same dev seeds
91000–19, same metric definitions (iter_gate.METRICS; committed = primary), opening lock from tick logs.
  python arch_compare.py"""
import os

import numpy as np

import iter_gate as g
import opening_cl as oc

HERE = os.path.dirname(os.path.abspath(__file__)); LG = os.path.join(HERE, "..", "..", "runs", "asd")
STEPS = [  # (label, scorer arm(s), tick-log pairs)
    ("S0 旧图·无 PGIF(理想 ID,不滤遮挡,语言头开局)", ["tahrelm"], None),
    ("S1 旧图模型 TAHRelM+PGIF(同上设定)", ["pgif"], [(os.path.join(LG, "deploy_all", "tahrelm_pgif.jsonl"),
                                                   os.path.join(HERE, "eval_out", "paper_dep_tahrelm_pgif.json"))]),
    ("S2 + 遮挡感知候选(ray-cast)", ["v2_pgif_sim"], "v2_pgif_sim"),
    ("S3 + 调后跟踪器 ID(GT-free)", ["v2t_pgif_trk"], "v2t_pgif_trk"),
    ("S4 + PGIF absent-hold = H-A", ["v2h_pgif_trk"], "v2h_pgif_trk"),
    ("S5 + 属性开局 v1 + 高清开局帧 = 新图", ["zoa_pgif_trk_a", "zoa_pgif_trk_b", "zoa_pgif_trk_c"], ["zoa_pgif_trk_a", "zoa_pgif_trk_b", "zoa_pgif_trk_c"]),
]
M = [("correct", "承诺正确", 3), ("correct_l", "宽松正确", 3), ("correct_s", "严格正确", 3), ("longC", "最长正确 s", 1),
     ("longW", "最长锁错 s", 1)]


def opening(spec):
    if spec is None:
        return None
    T = {}
    if isinstance(spec, str):
        T = oc.load(oc.pairs_for(spec))
    elif isinstance(spec[0], tuple):
        T = oc.load(spec)
    else:
        for s in spec:
            T.update(oc.load(oc.pairs_for(s)))
    out = {}
    for s, R in T.items():
        P = [r for r in R if r.get("true") is not None and r["true"] >= 0]
        rl = next((r for r in P if r["t"] >= 12), None)
        if rl is not None:
            out[s] = oc.out_id(rl) == rl["aidx"][rl["true"]]
    return out


def main():
    data = []
    for lab, arms, logs in STEPS:
        E = {}
        for a in arms:
            E.update(g.load(a) or {})
        data.append((lab, E, opening(logs)))
    common = sorted(set.intersection(*[set(E) for _, E, _ in data]))
    print(f"common seeds n={len(common)}: {[s % 1000 for s in common]}\n")
    hdr = f"{'':44s} {'开局锁对':>8s} " + " ".join(f"{h:>9s}" for _, h, _ in M) + f" {'集长(步)':>8s}"
    print(hdr)
    for lab, E, op in data:
        o = "—" if op is None else f"{sum(op.get(s, False) for s in common)}/{sum(s in op for s in common)}"
        vals = " ".join(f"{np.nanmean([g.METRICS[m](E[s]) for s in common]):9.{p}f}" for m, _, p in M)
        print(f"{lab:44s} {o:>8s} {vals} {np.mean([E[s]['steps'] for s in common]):8.0f}")
    rng = np.random.default_rng(20261003)
    print("\npaired Δ (95% seed-bootstrap CI; sign-flip p):")
    for (la, A, _), (lb, B, _) in [(data[i], data[i + 1]) for i in range(1, len(data) - 1)] + [(data[1], data[-1])]:
        parts = []
        for m, h, p in (M[0], M[3], M[4]):
            d = np.array([g.METRICS[m](B[s]) - g.METRICS[m](A[s]) for s in common], float)
            bs = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
            pv = (np.abs((d[None] * rng.choice([-1, 1], (20000, len(d)))).mean(1)) >= abs(d.mean())).mean()
            parts.append(f"{h} {d.mean():+.{p}f} [{np.percentile(bs, 2.5):+.{p}f}, {np.percentile(bs, 97.5):+.{p}f}] p={pv:.3f}")
        print(f"  {la[:2]} → {lb[:2]}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
