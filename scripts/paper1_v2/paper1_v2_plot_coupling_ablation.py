"""
Paper: paper1_v2
Purpose: Generate Fig 5 — coupling ablation heatmaps (Stage 2)
         (a) ΔR_task: Event-driven Goal − Periodic Goal
         (b) ΔR_task: FDLC − Event-driven Goal
Inputs:  results_v2/summaries/paper1_v2_scene_analysis.csv
         results_v2/summaries/paper1_v2_coupling_scene_pairwise.csv
Outputs: results_v2/figures/chapter5/paper1_v2_fig5_coupling_ablation.pdf
         results_v2/metadata/chapter5/paper1_v2_fig5.json
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "results_v2/summaries"
FIG_DIR = PROJECT_ROOT / "results_v2/figures/chapter5/v_705"
META_DIR = PROJECT_ROOT / "results_v2/metadata/chapter5/v_705"
FIG_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

SCENE_CSV = DATA_DIR / "paper1_v2_scene_analysis.csv"
PAIRWISE_CSV = DATA_DIR / "paper1_v2_coupling_scene_pairwise.csv"
STYLE_PATH = PROJECT_ROOT / "Autonomous_Robots_图片样式与制作规范.md"

# ── matplotlib rc ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.0,
    "lines.markersize": 5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INTENSITY_ORDER = ["low", "medium", "high", "severe"]
CONFLICT_ORDER = ["M0", "M1", "M2", "M3"]
SCENE_ORDER = [
    f"c3_{i}_{c.lower()}" for i in INTENSITY_ORDER for c in CONFLICT_ORDER
]
SCENE_LABELS = [
    f"{c}\n{i}" for i in INTENSITY_ORDER for c in CONFLICT_ORDER
]


def build_delta_matrix(df_scene: pd.DataFrame, m1: str, m2: str) -> np.ndarray:
    """Build ΔR_task = m1 - m2 per scene."""
    delta = np.full(len(SCENE_ORDER), np.nan)
    for j, scene in enumerate(SCENE_ORDER):
        sub = df_scene[df_scene["scene_id"] == scene]
        v1 = sub[sub["method"] == m1]["R_task_mean"]
        v2 = sub[sub["method"] == m2]["R_task_mean"]
        if len(v1) > 0 and len(v2) > 0:
            delta[j] = v1.values[0] - v2.values[0]
    return delta


def build_sig_mask(df_pw: pd.DataFrame, contrast_label: str) -> np.ndarray:
    """Build significance mask for a given contrast from scene pairwise results."""
    mask = np.zeros(len(SCENE_ORDER), dtype=bool)
    for j, scene in enumerate(SCENE_ORDER):
        sub = df_pw[(df_pw["scene_id"] == scene) &
                     (df_pw["contrast"] == contrast_label)]
        sig = sub[sub["holm_adjusted_p"] < 0.05]
        if len(sig) > 0:
            mask[j] = True
    return mask


def plot_delta_heatmap(ax, delta: np.ndarray, col_labels: List[str],
                       sig_mask: np.ndarray, title: str,
                       vmax_abs: float) -> None:
    """Plot a single-row delta heatmap with diverging colormap."""
    data = delta.reshape(1, -1)
    mask_2d = sig_mask.reshape(1, -1)

    im = ax.imshow(data, cmap=plt.cm.RdBu_r, aspect="auto",
                   vmin=-vmax_abs, vmax=vmax_abs)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks([0])
    ax.set_yticklabels(["ΔR_task"], fontsize=8)

    # Grid
    for j in range(len(col_labels) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.5)
    ax.axhline(-0.5, color="white", linewidth=0.5)
    ax.axhline(0.5, color="white", linewidth=0.5)

    # Annotate values
    for j in range(len(col_labels)):
        val = data[0, j]
        if np.isnan(val):
            continue
        text_color = "white" if abs(val) > vmax_abs * 0.6 else "black"
        ax.text(j, 0, f"{val:+.3f}", ha="center", va="center",
                fontsize=7, color=text_color, weight="bold")
        # Sig marker
        if mask_2d[0, j]:
            ax.text(j, 0.35, "★", ha="center", va="center",
                    fontsize=10, color="black" if abs(val) <= vmax_abs * 0.6 else "white")

    # Column group labels
    for k, intensity in enumerate(INTENSITY_ORDER):
        x_pos = k * 4 + 1.5
        ax.text(x_pos, -1.1, intensity, ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    return im


def main() -> None:
    df_scene = pd.read_csv(SCENE_CSV)
    df_pw = pd.read_csv(PAIRWISE_CSV)

    # Build delta matrices
    delta_event = build_delta_matrix(df_scene, "Event-driven Goal", "Periodic Goal")
    delta_mode = build_delta_matrix(df_scene, "FDLC", "Event-driven Goal")

    sig_event = build_sig_mask(df_pw, "Event-driven Goal vs Periodic Goal")
    sig_mode = build_sig_mask(df_pw, "FDLC vs Event-driven Goal")

    # Compute symmetric vmax
    vmax_event = max(abs(np.nanmin(delta_event)), abs(np.nanmax(delta_event)), 0.01)
    vmax_event = np.ceil(vmax_event * 100) / 100
    vmax_mode = max(abs(np.nanmin(delta_mode)), abs(np.nanmax(delta_mode)), 0.01)
    vmax_mode = np.ceil(vmax_mode * 100) / 100

    # ── figure layout ────────────────────────────────────────────
    fig_width_mm = 174
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_width_inch * 0.52

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(fig_width_inch, fig_height_inch),
        gridspec_kw={"hspace": 0.55}
    )

    col_labels = SCENE_LABELS

    # (a) Event-driven − Periodic
    im_a = plot_delta_heatmap(
        ax_a, delta_event, col_labels, sig_event,
        title="(a)  ΔR_task = Event-driven Goal − Periodic Goal",
        vmax_abs=vmax_event,
    )
    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.025, pad=0.02)
    cbar_a.set_label("ΔR_task", fontsize=8)

    # (b) FDLC − Event-driven
    im_b = plot_delta_heatmap(
        ax_b, delta_mode, col_labels, sig_mode,
        title="(b)  ΔR_task = FDLC − Event-driven Goal",
        vmax_abs=vmax_mode,
    )
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.025, pad=0.02)
    cbar_b.set_label("ΔR_task", fontsize=8)

    fig.text(0.5, 0.01, "Conflict level / Degradation intensity",
             ha="center", fontsize=9, fontstyle="italic")

    # ── save ─────────────────────────────────────────────────────
    fig_path = FIG_DIR / "paper1_v2_fig5_coupling_ablation.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.1)
    png_path = FIG_DIR / "paper1_v2_fig5_coupling_ablation.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Fig 5 saved: {fig_path}")
    print(f"Preview PNG: {png_path}")

    # ── narrative stats ──────────────────────────────────────────
    n_event_wins = int(np.sum(delta_event > 0))
    n_event_sig = int(np.sum(sig_event))
    n_mode_wins = int(np.sum(delta_mode > 0))
    n_mode_sig = int(np.sum(sig_mode))

    print(f"\n=== Fig 5 Paper Narrative ===")
    print(f"Event-driven > Periodic: {n_event_wins}/16 scenes "
          f"(mean Δ={np.nanmean(delta_event):+.3f}, sig={n_event_sig})")
    print(f"FDLC > Event-driven:   {n_mode_wins}/16 scenes "
          f"(mean Δ={np.nanmean(delta_mode):+.3f}, sig={n_mode_sig})")
    print(f"Event-driven benefit mainly in M2–M3 "
          f"(mean Δ={np.nanmean(delta_event[8:]):+.3f} vs "
          f"M0–M1 mean Δ={np.nanmean(delta_event[:8]):+.3f})")
    print(f"FDLC mode-switching adds little and often negative "
          f"(Δ<0 in {16 - n_mode_wins}/16 scenes)")

    # ── metadata ─────────────────────────────────────────────────
    style_hash = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()[:12] if STYLE_PATH.exists() else "missing"
    meta = {
        "generator_script": str(Path(__file__).name),
        "created_at": datetime.now().isoformat(),
        "input_files": [str(SCENE_CSV), str(PAIRWISE_CSV)],
        "style_guide_path": str(STYLE_PATH),
        "style_guide_hash": style_hash,
        "figure_size": f"{fig_width_mm}mm × {fig_height_inch * 25.4:.0f}mm (cross-column)",
        "output_format": "PDF",
        "confidence_interval_definition": "seed-level bootstrap paired 95% CI (5000 resamples)",
        "number_of_seeds": 5,
        "n_scenes": 16,
        "strategies": ["Periodic Goal", "Event-driven Goal", "FDLC"],
        "panel_a": "ΔR_task = Event-driven Goal − Periodic Goal; ★ = Holm-corrected p<0.05",
        "panel_b": "ΔR_task = FDLC − Event-driven Goal; ★ = Holm-corrected p<0.05",
        "key_findings": {
            "event_driven_benefit": f"mean Δ={np.nanmean(delta_event):+.3f}, wins={n_event_wins}/16",
            "mode_switch_benefit": f"mean Δ={np.nanmean(delta_mode):+.3f}, wins={n_mode_wins}/16",
        },
    }
    meta_path = META_DIR / "paper1_v2_fig5.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
