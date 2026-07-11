"""
Paper: paper1_v2
Purpose: Generate Fig 3 — overall performance heatmaps for Stage 1
         (a) R_task heatmap: 5 methods × 16 scenes
         (b) ΔR_task heatmap: FDLC vs best non-FDLC baseline
Inputs:  results_v2/summaries/paper1_v2_scene_analysis.csv
         results_v2/summaries/paper1_v2_scene_pairwise.csv
Outputs: results_v2/figures/chapter5/paper1_v2_fig3_overall_performance.pdf
         results_v2/metadata/chapter5/paper1_v2_fig3.json
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
PAIRWISE_CSV = DATA_DIR / "paper1_v2_scene_pairwise.csv"
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

# ── method order & colors ───────────────────────────────────────────
METHODS = ["CDSL", "WCDL", "RHC-Inspection", "CBCP", "FDLC"]
METHOD_DISPLAY = {
    "CDSL": "CDSL",
    "WCDL": "WCDL",
    "RHC-Inspection": "RHC-Ins",
    "CBCP": "CBCP",
    "FDLC": "FDLC",
}
METHOD_COLORS = {
    "CDSL": "#4472C4",
    "WCDL": "#ED7D31",
    "RHC-Inspection": "#A5A5A5",
    "CBCP": "#FFC000",
    "FDLC": "#C00000",
}
BASELINES = ["CDSL", "WCDL", "RHC-Inspection", "CBCP"]

INTENSITY_ORDER = ["low", "medium", "high", "severe"]
CONFLICT_ORDER = ["M0", "M1", "M2", "M3"]
SCENE_ORDER = [
    f"c3_{i}_{c.lower()}" for i in INTENSITY_ORDER for c in CONFLICT_ORDER
]
# Column labels: only conflict level (intensity shown as group headers)
COL_LABELS = [c for _ in INTENSITY_ORDER for c in CONFLICT_ORDER]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_scene = pd.read_csv(SCENE_CSV)
    df_pw = pd.read_csv(PAIRWISE_CSV)
    return df_scene, df_pw


def build_heatmap_matrices(df_scene: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build R_task and ΔR_task matrices for the heatmap."""
    r_task = np.full((len(METHODS), len(SCENE_ORDER)), np.nan)
    delta = np.full((1, len(SCENE_ORDER)), np.nan)
    sig_mask = np.zeros((1, len(SCENE_ORDER)), dtype=bool)

    for j, scene in enumerate(SCENE_ORDER):
        sub = df_scene[df_scene["scene_id"] == scene]
        if len(sub) == 0:
            continue
        for i, method in enumerate(METHODS):
            ms = sub[sub["method"] == method]
            if len(ms) > 0:
                r_task[i, j] = ms["R_task_mean"].values[0]

        # Δ = FDLC - max(baselines)
        fdlc_val = sub[sub["method"] == "FDLC"]["R_task_mean"]
        base_vals = sub[sub["method"].isin(BASELINES)].groupby("method")["R_task_mean"].mean()
        if len(fdlc_val) > 0 and len(base_vals) > 0:
            delta[0, j] = fdlc_val.values[0] - base_vals.max()

    return r_task, delta


def build_significance_mask(df_pw: pd.DataFrame) -> np.ndarray:
    """Check which scenes have significant FDLC vs best-baseline contrast."""
    mask = np.zeros((1, len(SCENE_ORDER)), dtype=bool)
    for j, scene in enumerate(SCENE_ORDER):
        sub = df_pw[df_pw["scene_id"] == scene]
        sig_contrasts = sub[sub["holm_adjusted_p"] < 0.05]
        if len(sig_contrasts) > 0:
            mask[0, j] = True
    return mask


def plot_heatmap(ax, data: np.ndarray, row_labels: List[str], col_labels: List[str],
                 cmap, vmin: float, vmax: float, title: str,
                 annot: bool = True, fmt: str = ".3f",
                 sig_mask: np.ndarray = None,
                 intensity_sep: bool = True) -> Any:
    """Plot a single heatmap panel."""
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Find max value per column for bold annotation
    col_max = np.nanargmax(data, axis=0) if data.shape[0] > 1 else np.zeros(data.shape[1], dtype=int)

    # Grid lines
    for i in range(data.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(data.shape[1] + 1):
        is_sep = (j % 4 == 0) and intensity_sep
        ax.axvline(j - 0.5,
                   color="#666666" if is_sep else "white",
                   linewidth=1.8 if is_sep else 0.5)

    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                is_max = (i == col_max[j])
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color="black",
                        weight="bold" if is_max else "normal")

    # Significance markers for delta panel
    if sig_mask is not None and data.shape[0] == 1:
        for j in range(data.shape[1]):
            if sig_mask[0, j]:
                ax.text(j, 0, "★", ha="center", va="center",
                        fontsize=10, color="black")

    # Intensity group labels centered above the heatmap (in axes fraction)
    if intensity_sep:
        n_cols = len(col_labels)
        for k, intensity in enumerate(INTENSITY_ORDER):
            x_frac = (k * 4 + 1.5) / n_cols  # center of 4-col block
            ax.text(x_frac, 1.015, intensity,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", fontstyle="italic",
                    transform=ax.transAxes, clip_on=False)

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    return im


def main() -> None:
    df_scene, df_pw = load_data()
    r_task, delta = build_heatmap_matrices(df_scene)
    sig_mask = build_significance_mask(df_pw)

    # ── figure layout: single panel ──────────────────────────────
    fig_width_mm = 174
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_width_inch * 0.42

    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
    fig.subplots_adjust(bottom=0.12, top=0.85)

    row_labels = [METHOD_DISPLAY[m] for m in METHODS]
    col_labels = COL_LABELS

    im = plot_heatmap(
        ax, r_task, row_labels, col_labels,
        cmap=plt.cm.YlOrRd, vmin=0.0, vmax=1.0,
        title="",
        annot=True, fmt=".2f",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("R_task", fontsize=8)

    ax.set_xlabel("Conflict level", fontsize=9, fontstyle="italic", labelpad=8)

    # ── save ─────────────────────────────────────────────────────
    fig_path = FIG_DIR / "paper1_v2_fig3_overall_performance.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.1)
    png_path = FIG_DIR / "paper1_v2_fig3_overall_performance.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Fig 3 saved: {fig_path}")
    print(f"Preview PNG: {png_path}")

    # ── compute narrative stats ──────────────────────────────────
    fdlc_ranks = []
    for scene in SCENE_ORDER:
        sub = df_scene[df_scene["scene_id"] == scene]
        means = sub.groupby("method")["R_task_mean"].mean().sort_values(ascending=False)
        if "FDLC" in means.index:
            fdlc_ranks.append(list(means.index).index("FDLC") + 1)

    fdlc_win_scenes = [SCENE_ORDER[j] for j in range(len(SCENE_ORDER))
                        if delta[0, j] > 0]
    fdlc_lose_scenes = [SCENE_ORDER[j] for j in range(len(SCENE_ORDER))
                         if delta[0, j] <= 0]

    print(f"\n=== Fig 3 Paper Narrative ===")
    print(f"FDLC rank: mean={np.mean(fdlc_ranks):.1f}, range={min(fdlc_ranks)}–{max(fdlc_ranks)}")
    print(f"FDLC wins (Δ>0): {len(fdlc_win_scenes)}/16 → {fdlc_win_scenes}")
    print(f"FDLC loses (Δ≤0): {len(fdlc_lose_scenes)}/16 → {fdlc_lose_scenes}")
    print(f"FDLC advantage grows: low({np.mean([delta[0,SCENE_ORDER.index(s)] for s in fdlc_win_scenes if 'low' in s] or [0]):+.3f}) "
          f"→ severe({np.mean([delta[0,SCENE_ORDER.index(s)] for s in fdlc_win_scenes if 'severe' in s]):+.3f})")

    # n.b. ★ markers show scenes where the FDLC vs best-baseline
    #      difference survives Holm correction at α=0.05

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
        "number_of_seeds": 5,
        "n_scenes": 16,
        "methods": METHODS,
        "description": "R_task heatmap: 5 methods × 16 scenes. Column max in bold. Intensity groups separated by dark lines.",
    }
    meta_path = META_DIR / "paper1_v2_fig3.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
