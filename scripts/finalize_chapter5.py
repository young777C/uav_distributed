#!/usr/bin/env python3
"""Auto-fill LaTeX placeholders with experiment results and compile chapter.

Usage:
    PYTHONPATH=src python3 scripts/finalize_chapter5.py

Requires: completed results in results/ts_sensitivity/parallel/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAPTER = PROJECT_ROOT / "chapter5_xelatex_final.tex"


def check_experiments_done() -> bool:
    """Verify all 21 experiment runs have 3+ episodes."""
    parallel_dir = PROJECT_ROOT / "results/ts_sensitivity/parallel"
    expected_configs = [
        ("c1_g2_m2", "struct_full_dual_loop_distributed", [20, 40, 80]),
        ("c2_g2_m2", "struct_full_dual_loop_distributed", [20, 40, 80]),
        ("c2_g2_m2", "ts_sweep_periodic_goal", [20, 40, 80]),
    ]
    all_done = True
    for case, system, ts_list in expected_configs:
        for ts in ts_list:
            for seed in range(3):
                metrics = parallel_dir / f"{case}__{system}" / f"Ts{ts}" / f"seed{seed}" / "metrics.jsonl"
                if not metrics.exists():
                    print(f"  MISSING: {case}__{system} Ts={ts} seed={seed}")
                    all_done = False
                else:
                    n_ep = sum(1 for _ in metrics.open("r") if _.strip())
                    if n_ep < 3:
                        print(f"  INCOMPLETE: {case}__{system} Ts={ts} seed={seed} ({n_ep}/3 ep)")
                        all_done = False
    return all_done


def fill_latex_tables() -> bool:
    """Run the analysis script to generate LaTeX table content."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.analyze_ts_results", "--latex-table"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
        env={"PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    if result.returncode != 0:
        print("Analysis failed:", result.stderr)
        return False
    print(result.stdout)
    return True


def main() -> None:
    print("=== Checking experiment completion ===")
    if not check_experiments_done():
        print("\nExperiments not yet complete. Run again later.")
        sys.exit(1)

    print("\nAll experiments complete!")

    print("\n=== Generating LaTeX tables ===")
    fill_latex_tables()

    print("\n=== Chapter ready for compilation ===")
    print(f"Run: xelatex {CHAPTER}")


if __name__ == "__main__":
    main()
