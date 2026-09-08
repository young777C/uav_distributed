"""Data-driven regeneration of the 3 paper figures (baseline / scenario / stability) DIRECTLY from the
eval_out JSONs — so adding an arm (e.g. TAHRelM from A1) needs NO hand-edited numbers: just run this.
Auto-skips arms whose json is absent (paired on the common scored seed set of present arms). Overwrites
the doc-referenced PNGs. Honest: DAM4SAM excluded from the density panel (its per-bucket accounting is
inconsistent with its overall correct-frac — noted in the caption)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "carla_uav_tracking", "rollout", "eval_out"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

# label, file-stem, color, category (lang/heur/learn/ext)
REG = [
    ("xattn",      "paper_l1_xattn",           "#8C8C8C", "lang"),
    ("l4",         "paper_l4_dino_conf",       "#7BA3D0", "heur"),
    ("l5",         "paper_l5_dino_conf_tavg",  "#4C72B0", "heur"),
    ("borrow#1",   "paper_ext_reidmotion03",   "#2F5597", "heur"),
    ("TAH+DAgger", "paper_ext_tahdag_gt",      "#8172B3", "learn"),
    ("TAHRelM",    "paper_ext_tahrelm_gt",     "#B34767", "learn"),
    ("OC-SORT",    "paper_ext_assoc_motion",   "#E8A87C", "ext"),
    ("DeepSORT",   "paper_ext_assoc_deepsort", "#C56A2E", "ext"),
    ("DAM4SAM",    "paper_ext_dam4sam",        "#55A868", "ext"),
]


def load(stem):
    p = os.path.join(ROOT, stem + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {int(e["episode_seed"]): e for e in d["episodes"] if "mis_follow_inst" in e}


ARMS = [(lab, e, col, cat) for lab, stem, col, cat in REG for e in [load(stem)] if e]
COMMON = sorted(set.intersection(*[set(e) for _, e, _, _ in ARMS]))
print(f"[regen] {len(COMMON)} paired seeds; arms present: {[a[0] for a in ARMS]}")


def m_correct(e, S): return float(np.mean([1 - e[s]["mis_follow_inst"] for s in S]))
def m_mwr(e, S):     return float(np.mean([e[s]["max_wrong_run_s"] for s in S]))
def m_sr(e, S):      return float(np.mean([1.0 if e[s]["success"] else 0.0 for s in S]))
def m_idsw(e, S):    return float(np.mean([e[s]["id_switches"] for s in S]))
def m_latch(e, S, T): return float(np.mean([1.0 if e[s]["max_wrong_run_s"] < T else 0.0 for s in S]))
def m_q(e, S):
    att = np.array([e[s]["reacquire_attempts"] for s in S], float)
    qs = np.array([e[s]["reacquire_success_rate"] for s in S])
    return float((qs * att).sum() / att.sum()) if att.sum() > 0 else float("nan")


def track_by_N(e, S, n):
    num = den = 0.0
    for s in S:
        c = e[s].get("n_dist", {}).get(n, 0)
        if c and n in e[s].get("mis_by_N", {}):
            num += (1 - e[s]["mis_by_N"][n]) * c; den += c
    return (num / den) if den else float("nan")


# ---------- Figure 1: baseline 4-panel bars ----------
labs = [a[0] for a in ARMS]; cols = [a[2] for a in ARMS]; cats = [a[3] for a in ARMS]
panels = [("Correct-tracking fraction  (↑ better)", lambda e: m_correct(e, COMMON), "%.2f"),
          ("Longest wrong-track, mean s  (↓ better)", lambda e: m_mwr(e, COMMON), "%.1f"),
          ("Success rate — latch  (↑ better)", lambda e: m_sr(e, COMMON), "%.3f"),
          ("ID switches  (↓ better)", lambda e: m_idsw(e, COMMON), "%.1f")]
fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8))
x = np.arange(len(ARMS))
for ax, (title, fn, fmt) in zip(axes.flat, panels):
    vals = [fn(a[1]) for a in ARMS]
    edges = ["#111" if c == "learn" else "white" for c in cats]
    lws = [1.6 if c == "learn" else 0.8 for c in cats]
    bars = ax.bar(x, vals, color=cols, width=0.74, edgecolor=edges, linewidth=lws)
    ax.set_title(title); ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8, rotation=22, ha="right", rotation_mode="anchor")
    for i, c in enumerate(cats):
        if c == "learn": ax.get_xticklabels()[i].set_fontweight("bold")
        if c == "ext": ax.get_xticklabels()[i].set_style("italic"); ax.get_xticklabels()[i].set_color("#7a5a30")
    ax.margins(y=0.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha="center", va="bottom", fontsize=7.6)
fig.suptitle("WHICH-module comparison — ours (blue=heuristic, purple/pink=learned) vs external (italic) — same seeds, gt-seeded",
             fontsize=10, y=0.99)
fig.text(0.5, 0.005, "Learned modules (bold, black edge): TAH+DAgger (B1) and TAHRelM (A1, +learned motion-consensus). "
         "Bars auto-computed from eval_out JSONs (n=%d paired seeds). DAM4SAM's 0.94 correct is stickiness (16s wrong-windows)." % len(COMMON),
         ha="center", fontsize=7.2, wrap=True)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(os.path.join(HERE, "fig_external_baselines.png"), bbox_inches="tight"); plt.close(fig)
print("[regen] wrote fig_external_baselines.png")

# ---------- Figure 2: scenario — track_correct vs distractor density ----------
Ns = ["2", "3", "5"]
scen = [a for a in ARMS if a[0] in ("DeepSORT", "borrow#1", "l5", "TAH+DAgger", "TAHRelM")]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for lab, e, col, cat in scen:
    ys = [track_by_N(e, COMMON, n) for n in Ns]
    hi = cat == "learn"
    ax.plot([2, 3, 5], ys, "-o", color=col, lw=3.0 if hi else 2.0, ms=9 if hi else 7,
            markeredgecolor="#111" if hi else "white", markeredgewidth=1.6 if hi else 1.0,
            zorder=5 if hi else 3, label=f"{lab}{' (LEARNED)' if hi else ''}")
    ax.text(5.08, ys[-1], f"{ys[-1]:.3f}", va="center", fontsize=8.5, fontweight="bold" if hi else "normal", color=col)
ax.set_xticks([2, 3, 5]); ax.set_xlabel("in-frame distractor count  N  (look-alike density →)")
ax.set_ylabel("Correct-tracking fraction  (↑ better)"); ax.set_xlim(1.8, 5.6)
ax.set_title("Robustness to look-alike density — learned appearance wins where it matters", fontsize=10.5)
ax.legend(fontsize=8.3, loc="lower left", framealpha=0.9)
fig.text(0.5, 0.005, "Frame-weighted track_correct by # in-frame distractors (n=%d seeds). Motion-based association degrades as "
         "look-alikes densify (similar cars move similarly); the learned appearance modules stay robust. DAM4SAM/OC-SORT/xattn omitted "
         "for clarity (DAM4SAM by-bucket accounting inconsistent)." % len(COMMON), ha="center", fontsize=7.2, wrap=True)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(HERE, "fig_scenario_density.png"), bbox_inches="tight"); plt.close(fig)
print("[regen] wrote fig_scenario_density.png")

# ---------- Figure 3: stability plane (persistence vs recovery, color=SR) ----------
fig, ax = plt.subplots(figsize=(8.2, 5.6))
stab = [a for a in ARMS if a[0] != "xattn"]  # xattn is an off-scale outlier (persist ~9.4)
for lab, e, col, cat in stab:
    px, py, sr = m_mwr(e, COMMON), m_q(e, COMMON), m_sr(e, COMMON)
    hi = cat == "learn"
    ax.scatter([px], [py], s=180 if hi else 90, c=[col], edgecolors="#111" if hi else "white",
               linewidths=1.8 if hi else 1.0, zorder=5 if hi else 3)
    ax.annotate(f"{lab}\nSR {sr:.3f}", (px, py), (px + 0.06, py + 0.006), fontsize=8.3,
                fontweight="bold" if hi else "normal", color=col)
ax.set_xlabel("error persistence  max_wrong_run_s (mean, s)  ← lower better")
ax.set_ylabel("recovery rate  q_reacq  ↑ better")
ax.set_title("Closed-loop stability plane — good corner = fast settling + high recovery (upper-left)", fontsize=10)
ax.axvspan(ax.get_xlim()[0], 3.1, ymin=0.0, ymax=1.0, color="#e8f3ec", alpha=0.35, zorder=0)
fig.text(0.5, 0.005, "Learned modules (bold, black edge). Auto-computed (n=%d seeds). borrow#1 sits in the good corner; "
         "A1 (TAHRelM) targets recovery-rate — see if it moves up vs TAH+DAgger. xattn omitted (off-scale, persist ~9.4s)." % len(COMMON),
         ha="center", fontsize=7.2, wrap=True)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(HERE, "fig_stability_plane.png"), bbox_inches="tight"); plt.close(fig)
print("[regen] wrote fig_stability_plane.png")
