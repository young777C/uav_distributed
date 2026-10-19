"""PGIF (Posterior-Gated Identity Filtering) — architecture figure.
(a) closed-loop system: PGIF wraps an arbitrary self-updating identity memory (TAHRelM / reid gallery) between the
    memory's raw pick and (i) the memory's own update and (ii) the control commitment.
(b) one REAL closed-loop episode (runs/asd/tah_tempered_91000.jsonl, seed 91008, TAHRelM): the memory's raw pick
    keeps flipping to look-alikes; the tempered, confidence-weighted tracklet evidence keeps the target; gate fires.
Grounded in carla_uav_tracking/rollout/asd.py::_filter_step (tempered) and policy.py::_asd_gate_memory.
  python acot_note/figs/fig_pgif_architecture.py  → fig_pgif_architecture.{png,pdf,svg}"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "carla_uav_tracking", "rollout"))
from asd import ASD  # noqa: E402

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7.2})
C_IN, C_FRZ, C_MEM, C_MEME = "#EEF1F5", "#CBD8E8", "#E7D9F2", "#6A3D9A"
C_PG, C_PGE = "#FDE3DF", "#C0392B"
C_WHERE, C_HOW, C_ENV = "#D9EAD3", "#FCE5CD", "#F3F3F3"
INK, MUT, RED, GRN = "#1a2230", "#6b7280", "#C0392B", "#2E7D32"

fig = plt.figure(figsize=(13.2, 8.4))
ax = fig.add_axes((0.0, 0.34, 1.0, 0.66)); ax.set_xlim(0, 100); ax.set_ylim(24, 100); ax.axis("off")


def box(x, y, w, h, title, sub="", fc=C_FRZ, ec="#8fa3bd", lw=1.0, fs=7.6, bold=False, ls="-", tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.4",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=fs, color=tcol,
                fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x + w / 2, y + h - 2.0 - fs * 0.5, sub, ha="center", va="top", fontsize=6.3, color=MUT,
                zorder=3, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=tcol,
                fontweight="bold" if bold else "normal", zorder=3)


def arrow(x1, y1, x2, y2, col="#5b6675", lw=1.2, ls="-", rad=0.0, label=None, lx=None, ly=None, lc=None, fs=6.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=col, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2, label,
                fontsize=fs, color=lc or col, ha="center", va="center", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


# ---------------- title ----------------
ax.text(1, 99.5, "PGIF — Posterior-Gated Identity Filtering for closed-loop referring re-acquisition", fontsize=12.5,
        fontweight="bold", color=INK, va="top")
ax.text(1, 95.6, "(a) A training-free wrapper between an arbitrary self-updating identity memory and both (i) its own update and "
                 "(ii) the control commitment. Needs only: pick, confidence (any scale), update hook + persistent tracklet ids.",
        fontsize=7.4, color=MUT, va="top")

# ---------------- inputs ----------------
box(1, 70, 13, 11, "UAV RGB + language", "frame t · “track the yellow\nNissan Patrol …”", fc=C_IN, ec="#b8c0cc")
box(1, 52, 13, 14, "CandidateSet", "boxes: target + look-alikes\npersistent tracklet ids k\n(ideal association in sim)",
    fc=C_IN, ec="#b8c0cc")

# frozen encoders
box(18, 70, 14, 11, "Frozen encoders", "Qwen3-VL (L24) · crop-DINOv2\nlanguage logits ℓ · feats F", fc=C_FRZ)
arrow(14, 75.5, 18, 75.5); arrow(14, 59, 18, 72, rad=-0.15)

# identity memory (any)
box(36, 60, 18, 21, "Identity memory  M", "any self-updating memory:\nTAHRelM (learned, 4,740 p)\nor reid DINOv2 gallery (heuristic)\n"
    "cold start: language pick\n→ raw pick s$^M_t$, confidence c$_t$", fc=C_MEM, ec=C_MEME, lw=1.4, bold=True)
arrow(32, 75.5, 36, 75.5)

# crossed-out raw self-update loop
arrow(40, 81.4, 50, 81.4, col="#9aa3ad", ls=(0, (3, 2)), rad=-1.1)
ax.text(39.5, 86.5, "✗ raw self-update M.update(s$^M_t$)\nconfirmation-bias loop (removed)", fontsize=6.2, color="#8b939c",
        ha="right", va="center", linespacing=1.3)
ax.plot([43.6, 46.4], [85.0, 87.0], color=RED, lw=1.6, zorder=5); ax.plot([43.6, 46.4], [87.0, 85.0], color=RED, lw=1.6, zorder=5)

# PGIF block
ax.add_patch(FancyBboxPatch((58, 47), 29, 45, boxstyle="round,pad=0.4,rounding_size=1.8", fc="#FFF7F5", ec=C_PGE,
                            lw=1.8, zorder=1))
ax.text(72.5, 91.4, "PGIF  (no trained parameters; λ = 0.98, W = 30 frozen)", fontsize=7.8, fontweight="bold",
        color=C_PGE, ha="center", va="top")
box(60, 76, 25, 11.5, "① scale-free reliability", "ρ$_t$ = rank(c$_t$ ; last W confidences) ∈ [0,1]\nw$_t$ = 0.5 + ρ$_t$",
    fc=C_PG, ec=C_PGE, fs=7.2)
box(60, 62.5, 25, 11.5, "② tempered tracklet evidence", "L$_k$ ← λ·L$_k$   for all seen tracklets k\n"
    "L$_{s^M_t}$ ← L$_{s^M_t}$ + w$_t$     (τ ≈ 1/(1−λ) ≈ 50 ticks)", fc=C_PG, ec=C_PGE, fs=7.2)
box(60, 49, 25, 11.5, "③ posterior gate", "ŝ$_t$ = argmax$_{k\\ visible}$ L$_k$\nif ŝ$_t$ ≠ s$^M_t$ → override", fc=C_PG,
    ec=C_PGE, fs=7.2)
arrow(72.5, 76, 72.5, 74.4, col=C_PGE); arrow(72.5, 62.5, 72.5, 60.9, col=C_PGE)
arrow(54, 74, 60, 81, col=C_MEME, label="c$_t$", lx=56.5, ly=78.8, lc=C_MEME)
arrow(54, 68, 60, 68, col=C_MEME, label="s$^M_t$", lx=57, ly=69.6, lc=C_MEME)

# gated memory update (feedback to memory)
arrow(60, 52, 46, 60, col=RED, lw=1.8, rad=-0.3, label="posterior-gated update:\nM.update(crop of ŝ$_t$)",
      lx=49, ly=51.5, lc=RED, fs=6.8)

# commitment → WHERE → HOW → env
box(90, 71, 9, 12, "commit ŝ$_t$", "K-frame\nhysteresis", fc="#FFFFFF", ec=C_PGE, lw=1.2, fs=7.2)
arrow(85, 55, 91, 71, col=C_PGE, rad=0.25)
box(90, 49, 9, 17, "WHERE", "committed actor\nworld pos → CV\nextrapolate\n→ z$_{ex}$", fc=C_WHERE, ec="#93c47d", fs=7.2)
arrow(94.5, 71, 94.5, 66.4)
box(74, 31, 23, 11, "HOW: DiT (frozen, 32 layers)", "flow-matching action chunk\n(z$_{ex}$, z$_{im}$, vlm_ctx, proprio) → (dx,dy,dz,dyaw)",
    fc=C_HOW, ec="#f6b26b", fs=7.2)
arrow(94.5, 49, 92, 42.4)
box(38, 31, 28, 11, "CARLA closed loop", "UAV moves → next frame t+1\n(identity choice changes what is observed)",
    fc=C_ENV, ec="#b7b7b7", fs=7.2)
arrow(74, 36.5, 66, 36.5)
arrow(38, 36.5, 7.5, 52, rad=-0.25, col="#7d8794", label="t ← t+1", lx=16, ly=40)

# key-message strip
ax.text(1, 27.5, "Why it works (PoC, TAHRelM, n=19 paired):  untempered Bayes (independent-error assumption) is captured by the "
                 "drifting memory (correct 0.074 < 0.272 no-correction);\n"
                 "removing the gate or the confidence weight loses the gain;  full PGIF: correct-lock 0.272 → 0.613 [Δ+0.341, CI +0.18…+0.51], "
                 "longest continuous correct lock 5.9 → 30.3 s [Δ+24.4 s, CI +12.0…+37.7].",
        fontsize=6.9, color=INK, va="top")

# ---------------- (b) real episode trace ----------------
LOG = os.path.join(REPO, "runs", "asd", "tah_tempered_91000.jsonl")
EP = 10
R = [json.loads(l) for l in open(LOG)]
S = [r for r in R if r["ep"] == EP]
from tick_align import aligned  # noqa: E402  (log ep index → scorer seed; skipped episodes have no reset)
SEED = [sd for sd, recs in aligned("paper_asd_tah_tempered", "tah_tempered_91000").items() if recs[0]["ep"] == EP][0]
a = ASD(mode="filter", filter_kind="tempered", resid_decay=0.98)
ticks, Lt, Ld, mem_ok, out_ok, fires, dist_id = [], [], [], [], [], [], None
for r in S:
    o = a.step(None, r["aidx"], r["mem"], r["conf"], None, r["true"], r["t"])
    ids = r["aidx"]
    if r["true"] is None or r["true"] < 0:
        continue
    act = ids[r["mem"]]
    if act != -1 and dist_id is None:
        dist_id = act
    ticks.append(r["t"] / 10.0)  # seconds (10 Hz steps)
    Lt.append(a.belief.get(-1, 0.0))
    others = {k: v for k, v in a.belief.items() if k != -1}
    Ld.append(max(others.values()) if others else 0.0)
    mem_ok.append(int(r["mem"] == r["true"])); out_ok.append(int(o["slot"] == r["true"])); fires.append(o["fire"])
ticks = np.array(ticks)

axb = fig.add_axes((0.055, 0.065, 0.63, 0.25))
axb.plot(ticks, Lt, color=GRN, lw=1.6, label="evidence L of the true target")
axb.plot(ticks, Ld, color=RED, lw=1.2, ls="--", label="max evidence over look-alikes")
for t, ok in zip(ticks, mem_ok):
    if not ok:
        axb.axvspan(t - 0.3, t + 0.3, color="#f4b6b0", alpha=0.45, lw=0)
for t, f, o in zip(ticks, fires, out_ok):
    if f:
        axb.plot([t], [-1.6], marker="^", color=C_PGE if o else "#888888", ms=3.2, ls="none")
axb.set_xlabel("time in episode (s)", fontsize=7); axb.set_ylabel("tempered evidence", fontsize=7)
axb.set_ylim(-3, max(max(Lt), max(Ld)) * 1.12); axb.tick_params(labelsize=6.5)
axb.set_title(f"(b) real closed-loop episode (TAHRelM + PGIF, seed {SEED}):  shaded = memory's raw pick is a look-alike "
              f"({100 * (1 - np.mean(mem_ok)):.0f}% of ticks);  ▲ = gate override;  PGIF output correct "
              f"{100 * np.mean(out_ok):.0f}% of ticks", fontsize=7.2, loc="left")
axb.legend(fontsize=6.5, loc="upper left", frameon=False)

# ---------------- (c) inset: what each part buys ----------------
axc = fig.add_axes((0.735, 0.065, 0.25, 0.25))
names = ["no\ncorrection", "untempered\nBayes", "equal-weight\nresidence", "PGIF w/o\ngate", "PGIF", "oracle\ndrift"]
vals = [0.272, 0.074, 0.376, 0.396, 0.613, 0.721]
cols = ["#bbbbbb", "#4c72b0", "#4c72b0", "#4c72b0", C_PGE, "#999999"]
axc.bar(range(len(vals)), vals, color=cols)
for i, v in enumerate(vals):
    axc.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.2)
axc.set_xticks(range(len(vals))); axc.set_xticklabels(names, fontsize=5.8)
axc.set_ylim(0, 0.85); axc.set_ylabel("track-correct fraction", fontsize=6.8); axc.tick_params(axis="y", labelsize=6.3)
axc.set_title("(c) ablations, TAHRelM Town05, n=19 paired", fontsize=7.2, loc="left")
for s in ("top", "right"):
    axc.spines[s].set_visible(False); axb.spines[s].set_visible(False)

for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_pgif_architecture.{ext}"), dpi=200 if ext == "png" else None)
print("wrote fig_pgif_architecture.{png,pdf,svg}; episode stats:", np.mean(mem_ok), np.mean(out_ok), int(np.sum(fires)))
