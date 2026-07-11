"""
Paper: paper1_v2
Purpose: Generate Fig S1 — R_cov vs R_task scatter decomposition
         4 representative scenes, all methods, seed-level scatter
Inputs:  results_v2/summaries/paper1_v2_seed_analysis.csv
Outputs: results_v2/figures/chapter5/v_705/706/paper1_v2_figS1_coverage_delivery_decomposition.pdf
"""
from __future__ import annotations

import hashlib, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "results_v2/figures/chapter5/v_705/706"
META_DIR = PROJECT_ROOT / "results_v2/metadata/chapter5/v_705/706"
FIG_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

SEED_CSV = PROJECT_ROOT / "results_v2/summaries/paper1_v2_seed_analysis.csv"
STYLE_PATH = PROJECT_ROOT / "Autonomous_Robots_图片样式与制作规范.md"

# ---- Matplotlib RC ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.0, "lines.markersize": 4,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

REP_SCENES = {
    "c3_low_m0": "low-M0",
    "c3_medium_m2": "medium-M2",
    "c3_high_m3": "high-M3",
    "c3_severe_m3": "severe-M3",
}

# Method colors and markers (consistent with FIGURE_INDEX)
METHOD_STYLE = {
    "CDSL":           {"color": "#4472C4", "marker": "s", "label": "CDSL"},
    "WCDL":           {"color": "#ED7D31", "marker": "^", "label": "WCDL"},
    "RHC-Inspection": {"color": "#A5A5A5", "marker": "D", "label": "RHC-Ins"},
    "CBCP":           {"color": "#FFC000", "marker": "d", "label": "CBCP"},
    "FDLC":           {"color": "#C00000", "marker": "o", "label": "FDLC"},
}

METHOD_ORDER = ["CDSL", "WCDL", "RHC-Inspection", "CBCP", "FDLC"]
ALPHA_BASELINE = 0.55
ALPHA_FDLC = 0.95
S_FDLC = 36        # larger marker for FDLC
S_BASELINE = 22


def main() -> None:
    df = pd.read_csv(SEED_CSV)
    df = df[df["method"].isin(METHOD_ORDER)]

    # ── 1×4 horizontal subplots, compressed height ──
    fig_width_mm = 174
    fig_height_mm = 52
    fig, axes = plt.subplots(
        1, 4,
        figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4),
        sharex=False, sharey=True,
    )

    # ── Unified axis limits ──
    all_rcov = df["R_cov_mean"].dropna()
    all_rtask = df["R_task_mean"].dropna()
    margin = 0.04
    x_lo = max(0, all_rcov.min() - margin)
    x_hi = min(1.0, all_rcov.max() + margin)
    y_lo = max(0, all_rtask.min() - margin)
    y_hi = min(1.0, all_rtask.max() + margin)
    id_lo = max(x_lo, y_lo)
    id_hi = min(x_hi, y_hi)

    for idx, (scene_id, scene_label) in enumerate(REP_SCENES.items()):
        ax = axes[idx]
        scene_df = df[df["scene_id"] == scene_id]

        # Baselines: all seed-level points
        for method in METHOD_ORDER[:-1]:  # CDSL through CBCP
            sub = scene_df[scene_df["method"] == method]
            if len(sub) == 0:
                continue
            style = METHOD_STYLE[method]
            ax.scatter(
                sub["R_cov_mean"].values,
                sub["R_task_mean"].values,
                s=S_BASELINE,
                c=style["color"],
                marker=style["marker"],
                alpha=ALPHA_BASELINE,
                edgecolors="none",
                label=style["label"],
                zorder=2,
            )

        # FDLC: only the seed with maximum R_task
        sub_fdlc = scene_df[scene_df["method"] == "FDLC"]
        if len(sub_fdlc) > 0:
            max_idx = sub_fdlc["R_task_mean"].idxmax()
            best = sub_fdlc.loc[max_idx]
            style = METHOD_STYLE["FDLC"]
            ax.scatter(
                [best["R_cov_mean"]],
                [best["R_task_mean"]],
                s=S_FDLC,
                c=style["color"],
                marker=style["marker"],
                alpha=ALPHA_FDLC,
                edgecolors="black",
                linewidths=0.5,
                label=style["label"],
                zorder=5,
            )

        # Identity line y = x
        ax.plot(
            [id_lo, id_hi], [id_lo, id_hi],
            color="gray", linestyle=":", linewidth=0.7, alpha=0.6, zorder=1,
        )

        ax.set_title(f"({chr(97 + idx)}) {scene_label}", fontweight="bold", pad=4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linewidth=0.4)
        if idx == 0:
            ax.set_ylabel("R_task")

    # ── Single shared x-axis label, centered ──
    fig.text(0.5, 0.01, "R_cov", ha="center", va="bottom", fontsize=8)

    # ── Single legend above ──
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        ncol=5, loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=7,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    fig.tight_layout(pad=0.6, rect=(0.0, 0.04, 1.0, 0.93))

    # ── Save ──
    pdf_path = FIG_DIR / "paper1_v2_figS1_coverage_delivery_decomposition.pdf"
    png_path = FIG_DIR / "paper1_v2_figS1_coverage_delivery_decomposition.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Fig S1 saved: {pdf_path}")

    # ── Metadata ──
    style_hash = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()[:12] if STYLE_PATH.exists() else "missing"
    meta = {
        "generator_script": str(Path(__file__).name),
        "created_at": datetime.now().isoformat(),
        "input_file": str(SEED_CSV),
        "style_guide_hash": style_hash,
        "figure_size": f"{fig_width_mm}mm × {fig_height_mm}mm",
        "output_format": "PDF",
        "description": "R_cov vs R_task scatter for 5 methods across 4 representative C3 scenes; "
                       "seed-level observations; identity line (y=x); FDLC highlighted with larger marker.",
    }
    (META_DIR / "paper1_v2_figS1.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
