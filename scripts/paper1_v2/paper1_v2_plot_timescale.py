"""
Paper: paper1_v2
Purpose: Generate Fig 6 — timescale sensitivity (FDLC only)
         2×2 subplots: 4 representative scenes, R_task vs rho for FDLC
         Seed-level means with 95% CI error bars; vertical dashed line at rho=40.
Inputs:  Stage 3 raw metrics.jsonl files (FDLC-d2 + stage1 default)
Outputs: results_v2/figures/chapter5/v_705/paper1_v2_fig6_timescale.pdf
"""
from __future__ import annotations

import json, hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "results_v2/figures/chapter5/v_705"
META_DIR = PROJECT_ROOT / "results_v2/metadata/chapter5/v_705"
FIG_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# FDLC-d2 data for rho ∈ {20, 80, 100}; stage1 baseline for rho = 40
FOLC_D2 = PROJECT_ROOT / "results_v2/stage3_fdlc_d2/stage3_fdlc_d2_20260705"
FOLC_S1 = PROJECT_ROOT / "results_v2/stage1_fdlc_p0/stage1_fdlc_p0_v4_d2_20260704"
STYLE_PATH = PROJECT_ROOT / "Autonomous_Robots_图片样式与制作规范.md"

# ---- Matplotlib RC: matches Autonomous_Robots style guide & FIGURE_INDEX ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.0, "lines.markersize": 5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SCENES = ["c3_medium_m2", "c3_high_m2", "c3_high_m3", "c3_severe_m3"]
SCENE_LABELS = ["medium–M2", "high–M2", "high–M3", "severe–M3"]
RHOS = [20, 40, 80, 100]

# t-critical for seed-level 95% CI with 5 seeds: df = 4
T_CRIT = stats.t.ppf(0.975, df=4)  # ≈ 2.776


def _per_seed_mean(jsonl_path):
    """Return the per-seed mean R_task over all episodes in a metrics.jsonl."""
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f]
    return np.mean([l.get("R_task", l.get("effective_ratio", 0)) for l in lines])


def get_rt(root, scene, suffix, rho):
    """Return (mean across seeds, 95%-CI half-width) for a scene/suffix/rho combo."""
    d = root / f"{scene}__{suffix}" / f"Ts{rho}"
    seed_means = []
    for sd in sorted(d.glob("seed*")):
        mf = sd / "metrics.jsonl"
        if mf.exists():
            seed_means.append(_per_seed_mean(mf))
    n_seeds = len(seed_means)
    if n_seeds < 2:
        return None, None
    mean = np.mean(seed_means)
    sem = np.std(seed_means, ddof=1) / np.sqrt(n_seeds)
    ci_half = T_CRIT * sem  # 95% CI half-width (t-dist, df = n_seeds-1)
    return mean, ci_half


def main():
    fig, axes = plt.subplots(2, 2, figsize=(174 / 25.4, 120 / 25.4))

    # First pass: collect all FDLC values to compute unified y-axis range
    all_means = []
    for scene in SCENES:
        for rho in RHOS:
            root = FOLC_S1 if rho == 40 else FOLC_D2
            m, _ = get_rt(root, scene, "struct_full_dual_loop_distributed", rho)
            if m is not None:
                all_means.append(m)

    y_min = max(-0.02, np.min(all_means) - 0.05) if all_means else -0.02
    y_max = min(1.02, np.max(all_means) + 0.08) if all_means else 1.02

    for idx, (scene, label) in enumerate(zip(SCENES, SCENE_LABELS)):
        ax = axes[idx // 2][idx % 2]

        f_means, f_cis = [], []
        for rho in RHOS:
            # FDLC: ρ=40 from stage1, others from stage3 FDLC-d2
            root = FOLC_S1 if rho == 40 else FOLC_D2
            m, ci = get_rt(root, scene, "struct_full_dual_loop_distributed", rho)
            f_means.append(m)
            f_cis.append(ci if ci else 0)

        # FDLC — bold solid with circle marker
        ax.errorbar(RHOS, f_means, yerr=f_cis,
                    marker="o", color="#C00000", linewidth=1.5,
                    capsize=3, capthick=0.8, label="FDLC")

        # Vertical dashed line at default ρ = 40
        ax.axvline(x=40, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(f"({chr(97 + idx)}) {label}", fontweight="bold")
        ax.set_ylabel("R_task" if idx % 2 == 0 else "")
        ax.set_xlabel(r"$\rho$" if idx >= 2 else "")
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(RHOS)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.5)
    fig_path = FIG_DIR / "paper1_v2_fig6_timescale.pdf"
    fig.savefig(fig_path)
    fig.savefig(FIG_DIR / "paper1_v2_fig6_timescale.png", dpi=300)
    plt.close()
    print(f"Fig 6 saved: {fig_path}")

    style_hash = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()[:12] if STYLE_PATH.exists() else "missing"
    meta = {
        "generator_script": str(Path(__file__).name),
        "created_at": datetime.now().isoformat(),
        "input_files": [str(FOLC_D2), str(FOLC_S1)],
        "style_guide_hash": style_hash,
        "figure_size": "174mm × 150mm (cross-column)",
        "output_format": "PDF",
        "confidence_interval_definition": "seed-level 95% CI (t-dist, df=4, 5 seeds)",
        "description": "R_task vs rho for FDLC across 4 representative scenes; "
                       "vertical dashed line at default rho=40; unified y-axis across all subplots.",
    }
    (META_DIR / "paper1_v2_fig6.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Metadata: {META_DIR / 'paper1_v2_fig6.json'}")


if __name__ == "__main__":
    main()
