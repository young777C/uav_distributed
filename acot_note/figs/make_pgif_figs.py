"""PGIF main figure with the THREE paper metrics (correct-lock ratio, longest continuous correct lock, longest wrong
lock); paired n=19, TAHRelM Town05. Source: eval_out/pgif_three_metrics.json (carla_uav_tracking/rollout/
pgif_three_metrics.py). SR is NOT plotted (threshold-dependent; supplementary only).
  python acot_note/figs/make_pgif_figs.py → fig_pgif_main.{png,pdf}"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EO = os.path.join(HERE, "..", "..", "carla_uav_tracking", "rollout", "eval_out")
J = json.load(open(os.path.join(EO, "pgif_three_metrics.json")))
S = J["summary"]; PB = {(r["base"], r["arm"]): r for r in J["paired"]}
BASE = "paper_asd_tah_observe"
ORDER = [("paper_ext_tahrelm_deploy", "language\nre-anchor"), ("paper_asd_tah_anchor_p3", "frozen-anchor\nre-anchor"),
         ("paper_asd_tah_bayes", "untempered\nBayes"), ("paper_asd_tah_resid", "equal-weight\nresidence"),
         ("paper_asd_tah_tempered_nogate", "PGIF w/o\nmemory gate"), ("paper_asd_tah_tempered", "PGIF\n(ours)"),
         ("paper_ext_tahrelm_oracledrift", "oracle drift\n(upper bound)"), ("paper_ext_tahrelm_gt", "oracle memory\n(upper bound)")]
PANELS = [("correct", "correct-lock ratio (↑)", 1.0), ("longC", "longest continuous correct lock, s (↑)", None),
          ("longW", "longest wrong lock, s (↓)", None)]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (k, title, ymax) in zip(axes, PANELS):
    xs, vals, errs, cols = [], [], [[], []], []
    for i, (stem, _) in enumerate(ORDER):
        v = S[stem][k]
        col = "#C0392B" if stem == "paper_asd_tah_tempered" else ("#999999" if "oracle" in stem or "gt" in stem else "#4c72b0")
        if v is None:
            ax.text(i, 1.0, "n/a", ha="center", fontsize=7, color="#888888")
            continue
        p = PB.get((BASE, stem)); lo = hi = 0.0
        if p and p.get(k):
            d, l, h = p[k]; lo, hi = d - l, h - d
        xs.append(i); vals.append(v); errs[0].append(lo); errs[1].append(hi); cols.append(col)
    ax.bar(xs, vals, color=cols, yerr=errs, capsize=3)
    for x, v in zip(xs, vals):
        ax.text(x + 0.18, v, f"{v:.2f}" if k == "correct" else f"{v:.1f}", fontsize=6.8, va="bottom")
    ax.axhline(S[BASE][k], ls="--", c="k", lw=1, label=f"no correction ({S[BASE][k]:.2f})" if k == "correct" else f"no correction ({S[BASE][k]:.1f}s)")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([n for _, n in ORDER], fontsize=6.6, rotation=30, ha="right")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.set_ylim(0, ymax if ymax else None)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
fig.suptitle("Deployable (GT-free) TAHRelM, Town05, paired n=19 — error bars: 95% paired-bootstrap CI of Δ vs no-correction "
             "(n/a: longest-correct not logged for that arm)", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_pgif_main.{ext}"), dpi=170 if ext == "png" else None, bbox_inches="tight")
print("wrote fig_pgif_main")
