"""Generate the external-baseline + reanchor comparison figures for baseline_compare_906.md.
Small multiples (metrics differ in scale — never one axis / dual axis). Colorblind-safe.
Honest: DAM4SAM's high correct-frac AND its 16s catastrophe, assoc_deepsort's high SR, all shown as-is.
All numbers are the real 19ep/17ep same-seed paired results (continuous_metrics.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "axes.titlesize": 10.5})

# ---- Figure A: external WHICH comparison (19ep same-seed same-config, only WHICH differs) ----
methods = ["xattn\n(lang)", "reid+conf\n(l4)", "reid+temporal\n(l5, ours)", "assoc\nmotion", "assoc\ndeepsort", "DAM4SAM\n(SOTA)"]
# category colors (colorblind-safe muted): language=gray, ours=blue, assoc=orange, dam4sam=green
cols = ["#8C8C8C", "#4C72B0", "#2F5597", "#DD8452", "#C56A2E", "#55A868"]
correct = [0.251, 0.671, 0.738, 0.755, 0.750, 0.941]
mwr     = [9.44, 3.78, 3.18, 2.69, 3.20, 16.32]   # longest wrong-track (mean, s)
sr      = [0.000, 0.105, 0.105, 0.053, 0.211, 0.105]
idsw    = [13.7, 14.1, 9.32, 13.4, 11.9, 0.79]
panels = [("Correct-tracking fraction  (↑ better)", correct, "%.2f"),
          ("Longest wrong-track, mean s  (↓ better)", mwr, "%.1f"),
          ("Success rate — latch  (↑ better)", sr, "%.3f"),
          ("ID switches  (↓ better)", idsw, "%.1f")]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
x = np.arange(len(methods))
for ax, (title, vals, fmt) in zip(axes.flat, panels):
    bars = ax.bar(x, vals, color=cols, width=0.72, edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.margins(y=0.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, fmt % v, ha="center", va="bottom", fontsize=8)
fig.suptitle("External WHICH-module comparison — same seeds (19ep), same closed loop, only the identity mechanism differs",
             fontsize=11, y=0.99)
fig.text(0.5, 0.005,
         "Honest read: our reid/temporal is NOT best — assoc-deepsort has higher SR, DAM4SAM higher correct-frac + near-zero id-switches (but 16s sticky wrong-windows). "
         "ALL appearance/association methods (0.67–0.94) ≫ single-frame language xattn (0.25) → the WHICH bottleneck & 'appearance/temporal association is the lever' is independently confirmed.",
         ha="center", fontsize=7.3, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig("acot_note/figs/fig_external_baselines.png", bbox_inches="tight")
print("wrote fig_external_baselines.png")

# ---- Figure B: reanchor (paper2 甲, 17ep same-seed paired) ----
arms = ["reid_deploy\n(GT-free, no re-anchor)", "+ language\nre-anchor", "+ oracle\nre-anchor (bound)"]
acol = ["#8C8C8C", "#DD8452", "#55A868"]
cfrac = [0.363, 0.225, 0.741]
mwr_med = [11.2, 12.6, 3.6]
fig2, ax2 = plt.subplots(1, 2, figsize=(8.4, 4.0))
for ax, (title, vals, fmt, ref) in zip(
        ax2, [("Correct-tracking fraction  (↑ better)", cfrac, "%.3f", None),
              ("Longest wrong-track, median s  (↓ better)", mwr_med, "%.1f", None)]):
    bars = ax.bar(np.arange(3), vals, color=acol, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_title(title); ax.set_xticks(np.arange(3)); ax.set_xticklabels(arms, fontsize=8); ax.margins(y=0.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, fmt % v, ha="center", va="bottom", fontsize=8.5)
fig2.suptitle("Deployable GT-free re-ID + re-anchor (paper2 Plan-A, 17ep paired) — language re-anchor is FALSIFIED", fontsize=10.5)
fig2.text(0.5, 0.01, "Language re-anchor does NOT help (0.363→0.225, worse); oracle re-anchor recovers it (→0.741) → the re-anchor MECHANISM has headroom, "
                     "but single-frame language is an unreliable arbiter among look-alikes (same ceiling).", ha="center", fontsize=7.3, wrap=True)
fig2.tight_layout(rect=[0, 0.05, 1, 0.94])
fig2.savefig("acot_note/figs/fig_reanchor.png", bbox_inches="tight")
print("wrote fig_reanchor.png")

# ---- Figure C: reanchor rescue (temporal language re-anchor, 17ep paired) — 甲 falsified even w/ rescue ----
rarms = ["reid_deploy\n(no re-anchor)", "single-frame\nlang re-anchor", "temporal\nlang re-anchor", "oracle\nre-anchor (bound)"]
rcol = ["#4C72B0", "#DD8452", "#C56A2E", "#55A868"]   # deploy=blue baseline, lang=orange, oracle=green
rcf = [0.363, 0.225, 0.291, 0.741]
rmw = [11.2, 12.6, 9.3, 3.6]
fig3, ax3 = plt.subplots(1, 2, figsize=(9.0, 4.1))
for ax, (title, vals, fmt) in zip(ax3, [("Correct-tracking fraction  (↑ better)", rcf, "%.3f"),
                                        ("Longest wrong-track, median s  (↓ better)", rmw, "%.1f")]):
    bars = ax.bar(np.arange(4), vals, color=rcol, width=0.66, edgecolor="white", linewidth=0.8)
    ax.set_title(title); ax.set_xticks(np.arange(4)); ax.set_xticklabels(rarms, fontsize=7.6); ax.margins(y=0.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, fmt % v, ha="center", va="bottom", fontsize=8.5)
# reference line = no-reanchor baseline (the bar temporal-lang must beat but doesn't)
ax3[0].axhline(0.363, color="#4C72B0", ls="--", lw=1, alpha=0.6)
fig3.suptitle("Temporal-EMA language re-anchor RESCUE (17ep paired) — Plan-A thoroughly falsified", fontsize=10.5)
fig3.text(0.5, 0.01, "Temporal EMA PARTIALLY de-noises the language vote (single-frame 0.225 → temporal 0.291, confirms 'single-frame too noisy') "
                     "BUT stays below the no-re-anchor baseline (0.363, dashed) → language is too weak an arbiter even temporally-integrated. "
                     "Oracle 0.741 = the re-anchor MECHANISM has headroom → the lever is a STRONGER signal (motion-consensus), not language.",
          ha="center", fontsize=7.0, wrap=True)
fig3.tight_layout(rect=(0, 0.06, 1, 0.94))
fig3.savefig("acot_note/figs/fig_reanchor_rescue.png", bbox_inches="tight")
print("wrote fig_reanchor_rescue.png")
