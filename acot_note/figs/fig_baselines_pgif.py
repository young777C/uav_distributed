"""WHICH-module comparison (907_baseline_compare_906.md, Figure 1, 2026-09-06 real data) + PGIF.
Same four metrics / same 19 scored seeds (91000-19 minus unscored 91004) / same definitions as make_baseline_figs.py.
HONEST SPLIT: the 09-06 arms are gt-seeded (oracle memory: the identity memory is refreshed from the GT target crop
every frame), PGIF runs DEPLOYABLE (GT-free committed memory). They are NOT the same setting; the deployable group
has its own no-correction reference. Numbers for the added arms are recomputed from eval_out/*.json with the same
formulas (checked: l5 0.738/3.18/0.105/9.32 and TAH+DAgger 0.716/3.11/0.158/8.26 reproduce the 09-06 figure).
  python acot_note/figs/fig_baselines_pgif.py  → fig_baselines_pgif.{png,pdf}"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EO = os.path.join(HERE, "..", "..", "carla_uav_tracking", "rollout", "eval_out")
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5})


TM = json.load(open(os.path.join(EO, "pgif_three_metrics.json")))["summary"]


def metrics(stem):   # (correct-lock ratio, longest wrong lock s, longest continuous correct lock s)
    r = TM[stem]
    return (r["correct"], r["longW"], r["longC"])


# 09-06 figure (verbatim from make_baseline_figs.py) — gt-seeded / oracle memory
# (correct, longest wrong mean s, longest correct = not logged in 09-06 runs)
rows = [("xattn", "#8C8C8C", (0.251, 9.44, None)),
        ("l4", "#7BA3D0", (0.671, 3.78, None)),
        ("l5", "#4C72B0", (0.738, 3.18, None)),
        ("borrow#1", "#2F5597", (0.810, 3.07, None)),
        ("TAH+DAgger", "#8172B3", (0.716, 3.11, None)),
        ("OC-SORT", "#E8A87C", (0.755, 2.69, None)),
        ("DeepSORT", "#C56A2E", (0.750, 3.20, None)),
        ("DAM4SAM", "#55A868", (0.941, 16.32, None))]
# 2026-10-02 supplementary gt-seeded runs (same oracle-memory setting, new scorer with frame-level longest-correct)
for stem, lab, col in (("paper_asd_tah_gt_observe", "TAHRelM (gt, rerun)", "#B39DDB"),
                       ("paper_asd_tah_gt_pgif", "TAHRelM+PGIF (gt)", "#E57373")):
    if stem in TM:
        rows.append((lab, col, metrics(stem)))
# deployable (GT-free) group — recomputed from this session's runs
dep = [("TAHRelM (no corr.)", "#BBBBBB", metrics("paper_asd_tah_observe")),
       ("TAHRelM+PGIF (ours)", "#C0392B", metrics("paper_asd_tah_tempered"))]
allrows = rows + dep
names = [r[0] for r in allrows]; cols = [r[1] for r in allrows]
vals = np.array([[np.nan if v is None else v for v in r[2]] for r in allrows], float)
x = np.concatenate([np.arange(len(rows)), len(rows) + 0.8 + np.arange(len(dep))])
PG = len(allrows) - 1
HL = [i for i, r in enumerate(allrows) if "PGIF" in r[0]]

panels = [("Correct-lock ratio  (↑)", 0, "%.2f"), ("Longest continuous correct lock, mean s  (↑)", 2, "%.1f"),
          ("Longest wrong lock, mean s  (↓)", 1, "%.1f")]
fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.9))
for ax, (title, j, fmt) in zip(axes.flat, panels):
    edges = ["#111111" if i in HL else "white" for i in range(len(allrows))]
    lws = [1.8 if i in HL else 0.8 for i in range(len(allrows))]
    v = np.nan_to_num(vals[:, j]); bars = ax.bar(x, v, color=cols, width=0.74, edgecolor=edges, linewidth=lws)
    ax.axvline(len(rows) - 0.1, color="#444444", ls=":", lw=1.1)
    top = np.nanmax(vals[:, j])
    ax.text((len(rows) - 1) / 2, top * 1.30, "gt-seeded (oracle memory): 09-06 arms + 10-02 supplementary", ha="center", fontsize=8,
            color="#555555")
    ax.text(x[-2:].mean(), top * 1.30, "deployable (GT-free)", ha="center", fontsize=8, color="#C0392B", fontweight="bold")
    ax.set_ylim(0, top * 1.42)
    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.8, rotation=22, ha="right", rotation_mode="anchor")
    for i in (5, 6, 7):
        ax.get_xticklabels()[i].set_style("italic"); ax.get_xticklabels()[i].set_color("#7a5a30")
    for i in HL:
        ax.get_xticklabels()[i].set_fontweight("bold"); ax.get_xticklabels()[i].set_color("#C0392B")
    for b, v in zip(bars, vals[:, j]):
        ax.text(b.get_x() + b.get_width() / 2, 0 if np.isnan(v) else v, "n/a" if np.isnan(v) else fmt % v,
                ha="center", va="bottom", fontsize=7.0, color="#888888" if np.isnan(v) else "black")
fig.suptitle("WHICH-module comparison, three objective metrics (19 same seeds) — 09-06 baselines (gt-seeded) + PGIF (deployable, GT-free)",
             fontsize=11, y=1.02)
fig.text(0.5, -0.06,
         "Left of the dotted line: 09-06 real data, oracle memory (refreshed from the GT target crop every frame) — upper-bound setting; "
         "longest continuous correct lock was not logged in those runs (n/a; scorer field max_correct_run_s added 2026-10-02).\n"
         "Right: deployable, no GT at decision time. PGIF: correct 0.272→0.613 (Δ+0.341 [+0.18,+0.51]), longest correct 5.9→30.3s "
         "(Δ+24.4s [+12.0,+37.7]); longest wrong 15.7→10.8s (n.s.) — the remaining wrong-lock tail comes from wrong cold starts.",
         ha="center", fontsize=7.6)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_baselines_pgif.{ext}"), bbox_inches="tight", dpi=170 if ext == "png" else None)
print("wrote fig_baselines_pgif", [(r[0], r[2]) for r in dep])
