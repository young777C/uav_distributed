"""[LEGACY - marked obsolete 2026-10-04] formerly fig_architecture_v1_tahrelm.py (TAHRelM-only, 2026-09-07; no PGIF, ideal
sim ids, no occlusion filter, language-head opening). Current final model: fig_architecture_v1.py / .png.

Original: ACoT-UAV-Track model architecture (submission-grade schematic) — nature-figure Python backend.
Core conclusion: a single tiny (4,740-param) learnable temporal-association head (TAHRelM) resolves the
language-specified target among look-alikes on top of a fully FROZEN VLM/DINOv2/IAR/DiT stack; WHICH is
the only learned axis, WHERE/HOW are frozen. Archetype: schematic-led composite (dataflow), not a data plot.
Grounded in train/tah.py::TAHRelM and carla_uav_tracking/rollout/policy.py (tid_head=tah)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7.2,
})

# palette
C_IN   = "#EEF1F5"   # inputs
C_FRZ  = "#CBD8E8"   # frozen backbone (blue-gray)
C_FRZ2 = "#B9C7DB"
C_TAH  = "#E7D9F2"   # TAHRelM fill (learned → warm/purple)
C_TAHE = "#6A3D9A"   # TAHRelM edge (highlight)
C_WHERE= "#D9EAD3"   # WHERE path (green)
C_HOW  = "#FCE5CD"   # HOW / DiT (orange)
C_OUT  = "#F4CCCC"
INK, MUT = "#1a2230", "#6b7280"

fig, ax = plt.subplots(figsize=(12.4, 6.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, sub="", fc=C_FRZ, ec="#8fa3bd", lw=1.0, tcol=INK, fs=7.6, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
    if sub:                                   # proportional placement so small boxes don't overflow
        ty_ = y + h - min(2.8, h * 0.24)
        ax.text(x + w/2, ty_, title, ha="center", va="top", fontsize=fs, color=tcol,
                fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x + w/2, ty_ - min(4.6, h * 0.40), sub, ha="center", va="top", fontsize=6.3,
                color=MUT, zorder=3, linespacing=1.25)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=fs, color=tcol,
                fontweight="bold" if bold else "normal", zorder=3)


def arrow(x1, y1, x2, y2, col="#5b6675", lw=1.2, ls="-", rad=0.0, z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                 color=col, lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))


# ---- title ----
ax.text(1, 98.5, "ACoT-UAV-Track — deployable tracking policy", fontsize=12.5, fontweight="bold", color=INK, va="top")
ax.text(1, 94.3, "Only TAHRelM (4,740 params) is trainable; the VLM / DINOv2 / IAR / DiT backbone is FROZEN.",
        fontsize=7.6, color=MUT, va="top")

# ---- INPUTS (left column) ----
ax.text(9, 90, "per-frame inputs", fontsize=7, color=MUT, ha="center", style="italic")
box(1, 78, 16, 8.5, "UAV RGB 512×512", "onboard camera", fc=C_IN, ec="#b8c0cc", fs=7.2)
box(1, 67.5, 16, 8.5, "Language", "“track the yellow\nNissan …” (neutral)", fc=C_IN, ec="#b8c0cc", fs=7.2)
box(1, 55.5, 16, 9.5, "CandidateSet", "target + distractor\nboxes · stable actor idx", fc=C_IN, ec="#b8c0cc", fs=7.2)
box(1, 45, 16, 8.5, "Proprio 5D", "cam_z, pitch, uav_vel×3", fc=C_IN, ec="#b8c0cc", fs=7.2)

# ---- FROZEN BACKBONE ----
box(22, 76, 20, 11, "Qwen3-VL-4B", "frozen · RGB+lang →\nvlm_ctx + layer feats", fc=C_FRZ, ec="#8fa3bd", bold=True)
box(22, 62, 20, 10, "IAR", "frozen · layers → z_im\n(+ occ / maneuver)", fc=C_FRZ, ec="#8fa3bd", bold=True)
box(22, 44.5, 20, 12.5, "crop-DINOv2", "frozen · ViT-S/14 · per\ncandidate crop → 384-d\ninstance appearance", fc=C_FRZ2, ec="#8fa3bd", bold=True)

# ---- TAHRelM (hero, learned) ----
tx, ty, tw, th = 46.5, 40, 28.5, 33
ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0.6,rounding_size=2.2",
             fc=C_TAH, ec=C_TAHE, lw=2.4, zorder=2))
ax.text(tx + tw/2, ty + th - 2.2, "★ TAHRelM  (learnable, 4,740 params)", ha="center", va="top",
        fontsize=8.4, fontweight="bold", color=C_TAHE, zorder=3)
ax.text(tx + tw/2, ty + th - 6.4, "WHICH — vehicle-invariant temporal association", ha="center", va="top",
        fontsize=6.6, color="#5b458f", style="italic", zorder=3)
# internal: 7 relative features
box(tx+1.5, ty+15.5, tw-3, 9.0,
    "7 vehicle-invariant relative features",
    "cos(feat, mem.app) · relp_x · relp_y · depth\n· rank · ‖relp‖ · msoft=exp(−‖relp‖/gate)",
    fc="#F3ECFA", ec="#b79fd6", fs=6.9)
# MLP + heads
box(tx+1.5, ty+7.8, 12.5, 6.2, "MLP 7·64·64·1", "→ per-cand logits", fc="#F3ECFA", ec="#b79fd6", fs=6.7)
box(tx+15.2, ty+7.8, tw-16.7, 6.2, "argmax → pred_slot", "conf=σ(top1−top2)", fc="#F3ECFA", ec="#b79fd6", fs=6.7)
# recurrent memory
box(tx+1.5, ty+1.0, tw-3, 6.2, "recurrent memory  {app(384), pos, vel}",
    "learned-gate appearance EMA · learned-decay velocity EMA", fc="#EADFF6", ec="#b79fd6", fs=6.5)

# ---- WHERE ----
box(46.5, 24, 28.5, 12, "WHERE — committed target → z_ex",
    "believed-target world pos → CV extrapolation\n→ project to camera (/scale)   [frozen, no EAR in track mode]",
    fc=C_WHERE, ec="#7fae6f", bold=False, fs=7.2)

# ---- DiT (HOW) ----
box(79, 46, 18.5, 16, "DiT", "frozen · 32-layer\nflow-matching action head\n{z_ex, z_im, vlm_ctx, proprio}",
    fc=C_HOW, ec="#e0a45c", bold=True)
ax.text(88.2, 63.5, "HOW", fontsize=7.4, color="#b5761f", ha="center", style="italic", fontweight="bold")

# ---- OUTPUT + CARLA loop ----
box(79, 27, 18.5, 12.5, "action (dx,dy,dz,dyaw)", "+ search_mode  →  drone.step\n→  CARLA env → new frame", fc=C_OUT, ec="#d08b8b", fs=7.2)

# ================= ARROWS =================
# inputs → backbone
arrow(17, 82, 22, 82)                        # RGB → Qwen
arrow(17, 71.5, 22, 80)                      # lang → Qwen
arrow(17, 82, 22, 51, rad=-0.12)             # RGB → DINOv2 (crops)
arrow(17, 60, 22, 50, rad=-0.05)             # candidates → DINOv2
# backbone → TAHRelM
arrow(42, 50.5, 46.5, 58, col=C_TAHE, lw=1.6)          # DINOv2 appearance → TAHRelM
ax.text(42.6, 55.0, "384-d ×N\n+ box pos", fontsize=5.8, color=C_TAHE, ha="center", va="center", linespacing=1.1)
# internal flow
arrow(tx+tw/2, ty+15.3, tx+tw/2, ty+14.1, col=C_TAHE, lw=1.1)   # features → MLP
arrow(tx+14.0, ty+10.9, tx+15.2, ty+10.9, col=C_TAHE, lw=1.1)   # MLP → argmax
arrow(tx+tw/2, ty+7.3, tx+tw/2, ty+15.4, col=C_TAHE, lw=0.9, rad=-0.42, z=1)  # memory recurrence ↑ into features
# WHICH → WHERE (committed identity selects the tracked target)
arrow(tx+tw/2, ty+0.6, 60.7, 36.2, col="#5b458f", lw=1.5)
ax.text(64.0, 38.0, "committed identity", fontsize=6.0, color="#5b458f", ha="left", va="center")
# WHERE → DiT (z_ex)
arrow(75, 30.5, 79, 47, col="#5f8a4f", lw=1.4, rad=0.10); ax.text(76.4, 40.5, "z_ex", fontsize=6.4, color="#5f8a4f")
# Qwen → IAR (layer features)
arrow(31, 76, 31, 72.3, col="#8fa3bd", lw=1.1); ax.text(33.4, 74.2, "layers", fontsize=5.9, color=MUT)
# frozen context bus over the top: vlm_ctx (Qwen) + z_im (IAR) → DiT top  (clears TAHRelM)
ax.plot([42, 45.5], [81, 79.5], color="#9aa7b8", lw=0.9, ls=(0, (3, 2)), zorder=1)   # vlm_ctx feeder
ax.plot([42, 45.5], [68, 79.5], color="#9aa7b8", lw=0.9, ls=(0, (3, 2)), zorder=1)   # z_im feeder
arrow(45.5, 79.5, 84, 63, col="#9aa7b8", lw=1.0, rad=0.20, ls=(0, (3, 2)))
ax.text(61, 82.4, "vlm_ctx · z_im", fontsize=6.1, color="#8a94a3", ha="center", style="italic")
# proprio (S1 fast stream, per frame) → DiT bottom, routed along the base
ax.plot([16, 16], [45, 20.5], color="#9aa7b8", lw=0.9, ls=(0, (3, 2)), zorder=1)
ax.plot([16, 82], [20.5, 20.5], color="#9aa7b8", lw=0.9, ls=(0, (3, 2)), zorder=1)
arrow(82, 20.5, 82, 45.8, col="#9aa7b8", lw=0.9, ls=(0, (3, 2)))
ax.text(44, 22.2, "proprio  (S1 fast stream, every frame)", fontsize=6.1, color="#8a94a3", ha="center", style="italic")
# DiT → action
arrow(88.2, 46, 88.2, 39.6, col="#c0552b", lw=1.5)
# closed loop: action → back to inputs
arrow(88.2, 27, 88.2, 12, col="#b06a6a", lw=1.3)
ax.add_patch(FancyArrowPatch((88.2, 12), (9, 12), arrowstyle="-|>", mutation_scale=11,
             color="#b06a6a", lw=1.3, connectionstyle="arc3,rad=0.0", zorder=1))
arrow(9, 12, 9, 44, col="#b06a6a", lw=1.3)
ax.text(48, 10.2, "closed-loop rollout  (CARLA, s2-period control)", fontsize=6.6, color="#b06a6a", ha="center", style="italic")

# ---- legend (frozen vs trainable) — top-right, clear of subtitle ----
lx, ly = 79, 93.4
ax.add_patch(FancyBboxPatch((lx, ly), 3.0, 2.6, boxstyle="round,pad=0.2,rounding_size=0.6", fc=C_FRZ, ec="#8fa3bd", lw=1.0))
ax.text(lx+3.6, ly+1.3, "frozen", fontsize=6.7, va="center", color=INK)
ax.add_patch(FancyBboxPatch((lx+13, ly), 3.0, 2.6, boxstyle="round,pad=0.2,rounding_size=0.6", fc=C_TAH, ec=C_TAHE, lw=1.8))
ax.text(lx+16.6, ly+1.3, "trainable (TAHRelM)", fontsize=6.7, va="center", color=INK)

ax.text(50, 50, "LEGACY", fontsize=110, color="#d03b3b", alpha=0.10, ha="center", va="center", rotation=20,
        fontweight="bold", zorder=10)
ax.text(50, 2.5, "LEGACY - obsolete (TAHRelM-only, ideal sim ids, no occlusion filter). Current: fig_architecture_v1.png",
        fontsize=8, color="#0b0b0b", ha="center", va="bottom", fontweight="bold", zorder=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fde3d7", ec="#d03b3b", lw=1.0))
fig.savefig("acot_note/figs/fig_architecture_legacy_tahrelm.png", dpi=600, bbox_inches="tight")
fig.savefig("acot_note/figs/fig_architecture_legacy_tahrelm.pdf", bbox_inches="tight")
fig.savefig("acot_note/figs/fig_architecture_legacy_tahrelm.svg", bbox_inches="tight")
print("wrote fig_architecture_legacy_tahrelm.{png,pdf,svg}")
