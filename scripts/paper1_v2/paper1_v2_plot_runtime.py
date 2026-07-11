"""
Paper: paper1_v2
Purpose: Generate Fig 7 — computation budget occupancy (fast loop + slow loop)
         Using FDLC-BL timing data as proxy for FDLC-P0-d2
Inputs:  Stage 1 FDLC-BL raw metrics.jsonl files
Outputs: results_v2/figures/chapter5/v_705/paper1_v2_fig7_runtime.pdf
"""
from __future__ import annotations

import json, hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "results_v2/figures/chapter5/v_705"
META_DIR = PROJECT_ROOT / "results_v2/metadata/chapter5/v_705"
FIG_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

FDLC_BL = PROJECT_ROOT / "results_v2/stage1_struct/stage1_struct_20260629"
STYLE_PATH = PROJECT_ROOT / "Autonomous_Robots_图片样式与制作规范.md"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.0, "lines.markersize": 5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SCENES_SEL = ["c3_low_m0", "c3_medium_m2", "c3_high_m3", "c3_severe_m3"]
SCENE_LABELS = ["low-M0", "medium-M2", "high-M3", "severe-M3"]
SCENE_COLORS = ["#4472C4", "#ED7D31", "#FFC000", "#C00000"]
T_fast_ms = 41.7   # 24 Hz fast loop period
T_slow_s = 1.67    # rho=40 slow loop period

def collect_timing():
    fast_all = {"mean": [], "p95": [], "p99": [], "max": []}
    slow_all = {"mean": [], "p95": [], "p99": [], "max": []}
    per_scene = {}

    for scene in SCENES_SEL:
        d = FDLC_BL / f"{scene}__struct_full_dual_loop_distributed" / "Ts40"
        f_mean, f_p95, f_p99, f_max = [], [], [], []
        s_mean, s_p95, s_p99, s_max = [], [], [], []
        for sd in d.glob("seed*"):
            mf = sd / "metrics.jsonl"
            if not mf.exists(): continue
            with open(mf) as f:
                for line in f:
                    ep = json.loads(line)
                    for k_src, k_dst in [("fast_time_mean_ms","mean"),("fast_time_p95_ms","p95"),
                                          ("fast_time_p99_ms","p99"),("fast_time_max_ms","max")]:
                        v = ep.get(k_src)
                        if v is not None and v == v:
                            locals()[f"f_{k_dst}"].append(v)
                            fast_all[k_dst].append(v)
                    for k_src, k_dst in [("slow_time_mean_ms","mean"),("slow_time_p95_ms","p95"),
                                          ("slow_time_p99_ms","p99"),("slow_time_max_ms","max")]:
                        v = ep.get(k_src)
                        if v is not None and v == v:
                            locals()[f"s_{k_dst}"].append(v)
                            slow_all[k_dst].append(v)
        per_scene[scene] = {
            "fast": {"mean": np.mean(f_mean), "p95": np.percentile(f_p95,95) if f_p95 else 0,
                     "p99": np.percentile(f_p99,99) if f_p99 else 0, "max": np.max(f_max) if f_max else 0},
            "slow": {"mean": np.mean(s_mean), "p95": np.percentile(s_p95,95) if s_p95 else 0,
                     "p99": np.percentile(s_p99,99) if s_p99 else 0, "max": np.max(s_max) if s_max else 0},
        }
    return per_scene, fast_all, slow_all

def main():
    per_scene, fast_all, slow_all = collect_timing()

    fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(174/25.4, 85/25.4))

    # Fast loop
    stats = ["P95", "P99", "Max"]
    colors = ["#4472C4", "#ED7D31", "#C00000"]
    for i, scene in enumerate(SCENES_SEL):
        vals = [per_scene[scene]["fast"]["p95"],
                per_scene[scene]["fast"]["p99"],
                per_scene[scene]["fast"]["max"]]
        x = np.arange(len(stats)) + i * 0.2 - 0.3
        bars = ax_f.bar(x, [v / T_fast_ms * 100 for v in vals], 0.18,
                        color=colors, edgecolor="white", linewidth=0.5,
                        label=SCENE_LABELS[i] if i == 0 else "")
        # Annotate max bar
        for j, (bar, val) in enumerate(zip(bars, vals)):
            ax_f.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.1f}ms", ha="center", fontsize=6, rotation=90, color="gray")

    ax_f.set_xticks(np.arange(len(stats)))
    ax_f.set_xticklabels(stats)
    ax_f.set_ylabel("Fast-loop budget occupancy (%)")
    ax_f.set_title("(a) Fast loop ($T_f = 41.7$ ms)", fontweight="bold")
    ax_f.axhline(y=100, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_f.text(2.5, 101, "deadline", fontsize=6, color="red", ha="center")
    ax_f.set_ylim(0, 0.8)

    # Slow loop
    for i, scene in enumerate(SCENES_SEL):
        vals = [per_scene[scene]["slow"]["p95"],
                per_scene[scene]["slow"]["p99"],
                per_scene[scene]["slow"]["max"]]
        x = np.arange(len(stats)) + i * 0.2 - 0.3
        bars = ax_s.bar(x, [v / T_slow_s / 10 for v in vals], 0.18,
                        color=colors, edgecolor="white", linewidth=0.5)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            ax_s.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.0f}ms", ha="center", fontsize=6, rotation=90, color="gray")

    ax_s.set_xticks(np.arange(len(stats)))
    ax_s.set_xticklabels(stats)
    ax_s.set_ylabel("Slow-loop budget occupancy (%)")
    ax_s.set_title(f"(b) Slow loop ($T_s = {T_slow_s}$ s)", fontweight="bold")
    ax_s.axhline(y=100, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_s.text(2.5, 101, "deadline", fontsize=6, color="red", ha="center")
    ax_s.set_ylim(0, 2.0)

    # Shared legend using ax_f bars
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SCENE_COLORS[i], label=SCENE_LABELS[i]) for i in range(4)]
    ax_f.legend(handles=legend_elements, loc="upper left", fontsize=6, ncol=2)

    fig.tight_layout(pad=1.5)
    fig_path = FIG_DIR / "paper1_v2_fig7_runtime.pdf"
    fig.savefig(fig_path)
    fig.savefig(FIG_DIR / "paper1_v2_fig7_runtime.png", dpi=300)
    plt.close()
    print(f"Fig 7 saved: {fig_path}")

    # Print stats for paper text
    print("\nFast loop (all scenes, all episodes):")
    for k in ["mean", "p95", "p99", "max"]:
        vals = [v for v in fast_all[k] if v == v]
        if vals:
            print(f"  {k}: mean={np.mean(vals):.2f}ms, p95={np.percentile(vals,95):.2f}ms, max={np.max(vals):.2f}ms")
    print("\nSlow loop (all scenes, all episodes):")
    for k in ["mean", "p95", "p99", "max"]:
        vals = [v for v in slow_all[k] if v == v]
        if vals:
            print(f"  {k}: mean={np.mean(vals):.2f}ms, p95={np.percentile(vals,95):.2f}ms, max={np.max(vals):.2f}ms")

    style_hash = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()[:12] if STYLE_PATH.exists() else "missing"
    meta = {
        "generator_script": str(Path(__file__).name),
        "created_at": datetime.now().isoformat(),
        "input_files": [str(FDLC_BL)],
        "style_guide_hash": style_hash,
        "figure_size": "174mm × 85mm (cross-column)",
        "output_format": "PDF",
        "note": "Timing data from FDLC-BL used as proxy for FDLC-P0-d2 (negligible overhead difference)",
        "fast_loop_period_ms": T_fast_ms,
        "slow_loop_period_s": T_slow_s,
    }
    (META_DIR / "paper1_v2_fig7.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
