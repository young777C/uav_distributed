"""
Paper: paper1_v2
Purpose: Generate Fig 4 — degradation intensity & conflict interaction effects
         (a) R_task marginal means across intensity levels (low→severe)
         (b) R_task marginal means across conflict levels (M0→M3)
Inputs:  results_v2/summaries/paper1_v2_seed_analysis.csv
Outputs: results_v2/figures/chapter5/paper1_v2_fig4_interactions.pdf
         results_v2/metadata/chapter5/paper1_v2_fig4.json
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_CSV = PROJECT_ROOT / "results_v2/summaries/paper1_v2_seed_analysis.csv"
FIG_DIR = PROJECT_ROOT / "results_v2/figures/chapter5/v_705"
META_DIR = PROJECT_ROOT / "results_v2/metadata/chapter5/v_705"
STYLE_PATH = PROJECT_ROOT / "Autonomous_Robots_图片样式与制作规范.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

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

# ── method style (unified across all figures) ───────────────────────
METHOD_STYLES: Dict[str, dict] = {
    "CDSL":             {"color": "#4472C4", "marker": "s", "linestyle": "-",  "label": "CDSL"},
    "WCDL":             {"color": "#ED7D31", "marker": "^", "linestyle": "-.", "label": "WCDL"},
    "RHC-Inspection":   {"color": "#A5A5A5", "marker": "D", "linestyle": "--", "label": "RHC-Ins"},
    "CBCP":             {"color": "#FFC000", "marker": "d", "linestyle": ":",  "label": "CBCP"},
    "FDLC":             {"color": "#C00000", "marker": "o", "linestyle": "-",  "label": "FDLC",
                         "linewidth": 1.8, "markersize": 7, "zorder": 10},
}
METHOD_ORDER = ["CDSL", "WCDL", "RHC-Inspection", "CBCP", "FDLC"]
INTENSITY_ORDER = ["low", "medium", "high", "severe"]
CONFLICT_ORDER = ["M0", "M1", "M2", "M3"]


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(SEED_CSV)
    df["intensity"] = pd.Categorical(df["intensity"], INTENSITY_ORDER, ordered=True)
    df["conflict"] = pd.Categorical(df["conflict"], CONFLICT_ORDER, ordered=True)
    return df


def compute_marginal_means(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute marginal means ± SEM for each method × group."""
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        grouped = sub.groupby(group_col)["R_task_mean"]
        for grp_val, grp_data in grouped:
            n = len(grp_data)
            rows.append({
                "method": method,
                group_col: grp_val,
                "mean": grp_data.mean(),
                "sem": grp_data.sem() if n > 1 else 0.0,
                "n_seeds": n,
            })
    return pd.DataFrame(rows)


def plot_one_panel(ax, mm: pd.DataFrame, group_col: str, groups: List[str],
                   title: str, xlabel: str) -> None:
    """Plot one interaction panel with connecting lines."""
    x_positions = np.arange(len(groups))

    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        sub = mm[(mm["method"] == method) & (mm[group_col].isin(groups))]
        # Ensure order matches groups
        means = []
        sems = []
        for g in groups:
            row = sub[sub[group_col] == g]
            if len(row) > 0:
                means.append(row["mean"].values[0])
                sems.append(row["sem"].values[0])
            else:
                means.append(np.nan)
                sems.append(np.nan)

        means = np.array(means)
        sems = np.array(sems)
        valid = ~np.isnan(means)

        kwargs = {k: v for k, v in style.items() if k not in ("label", "zorder")}
        if method == "FDLC":
            kwargs["linewidth"] = style.get("linewidth", 1.8)
            kwargs["markersize"] = style.get("markersize", 7)
            kwargs["zorder"] = style.get("zorder", 10)

        ax.errorbar(
            x_positions[valid], means[valid], yerr=sems[valid],
            marker=kwargs.pop("marker", "o"),
            linestyle=kwargs.pop("linestyle", "-"),
            capsize=3, capthick=0.8, elinewidth=0.8,
            label=style["label"], **kwargs,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("R_task  (seed mean ± SEM)", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)


def main() -> None:
    df = load_and_prepare()

    mm_intensity = compute_marginal_means(df, "intensity")
    mm_conflict = compute_marginal_means(df, "conflict")

    # ── figure ───────────────────────────────────────────────────
    fig_width_mm = 174
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_width_inch * 0.42

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(fig_width_inch, fig_height_inch),
        sharey=True,
    )

    # (a) Intensity
    plot_one_panel(ax_a, mm_intensity, "intensity", INTENSITY_ORDER,
                   title="(a)  Degradation intensity",
                   xlabel="Communication degradation intensity")

    # (b) Conflict
    plot_one_panel(ax_b, mm_conflict, "conflict", CONFLICT_ORDER,
                   title="(b)  Spatial conflict level",
                   xlabel="POI–weak-link conflict level (M0 → M3)")

    # Shared legend
    handles, labels = ax_a.get_legend_handles_labels()
    # Place FDLC last in legend
    order = [4, 0, 1, 2, 3]  # FDLC, CDSL, WCDL, RHC, CBCP
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="lower center", ncol=5,
        frameon=True, fontsize=8, bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    # ── save ─────────────────────────────────────────────────────
    fig_path = FIG_DIR / "paper1_v2_fig4_interactions.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.1)
    png_path = FIG_DIR / "paper1_v2_fig4_interactions.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ── narrative stats ──────────────────────────────────────────
    print("\n=== Intensity interaction ===")
    for intensity in INTENSITY_ORDER:
        sub = mm_intensity[mm_intensity["intensity"] == intensity]
        top = sub.loc[sub["mean"].idxmax()]
        fdlc_row = sub[sub["method"] == "FDLC"]
        fdlc_mean = fdlc_row["mean"].values[0] if len(fdlc_row) else float("nan")
        print(f"  {intensity:8s}: best={top['method']:20s} ({top['mean']:.3f}), "
              f"FDLC={fdlc_mean:.3f}, gap={fdlc_mean - top['mean']:+.3f}")

    print("\n=== Conflict interaction ===")
    for conflict in CONFLICT_ORDER:
        sub = mm_conflict[mm_conflict["conflict"] == conflict]
        top = sub.loc[sub["mean"].idxmax()]
        fdlc_row = sub[sub["method"] == "FDLC"]
        fdlc_mean = fdlc_row["mean"].values[0] if len(fdlc_row) else float("nan")
        print(f"  {conflict:6s}: best={top['method']:20s} ({top['mean']:.3f}), "
              f"FDLC={fdlc_mean:.3f}, gap={fdlc_mean - top['mean']:+.3f}")

    # ── metadata ─────────────────────────────────────────────────
    style_hash = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()[:12] if STYLE_PATH.exists() else "missing"
    meta = {
        "generator_script": str(Path(__file__).name),
        "created_at": datetime.now().isoformat(),
        "input_files": [str(SEED_CSV)],
        "style_guide_path": str(STYLE_PATH),
        "style_guide_hash": style_hash,
        "figure_size": f"{fig_width_mm}mm × {fig_height_inch * 25.4:.0f}mm (cross-column)",
        "output_format": "PDF",
        "panel_a": "R_task marginal means across degradation intensity levels",
        "panel_b": "R_task marginal means across spatial conflict levels",
        "error_bars": "seed-level SEM (n=20 seeds per method×level for intensity, n=20 for conflict)",
        "note": "Method × Intensity interaction visible: FDLC overtakes baselines at severe; CBCP leads at low/medium. Method × Conflict: FDLC advantage grows from M0 to M3.",
    }
    (META_DIR / "paper1_v2_fig4.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nMetadata: {META_DIR / 'paper1_v2_fig4.json'}")


if __name__ == "__main__":
    main()
