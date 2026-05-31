#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt

from uavlab.common.config import load_resolved_config


Point2D = Tuple[float, float]


def _load_metrics_row(metrics_path: Path, episode: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("episode", -1)) == int(episode):
                rows.append(obj)
    if not rows:
        raise ValueError(f"No row found for episode={episode} in {metrics_path}")
    # If duplicates exist, use the last one.
    return rows[-1]


def _as_points(x: Any) -> List[Point2D]:
    pts = list(x or [])
    out: List[Point2D] = []
    for p in pts:
        out.append((float(p[0]), float(p[1])))
    return out


def _as_circles(x: Any) -> List[Tuple[float, float, float]]:
    cs = list(x or [])
    out: List[Tuple[float, float, float]] = []
    for c in cs:
        out.append((float(c[0]), float(c[1]), float(c[2])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot POI outcomes (effective / covered-only / unvisited).")
    ap.add_argument("--run_dir", type=str, required=True, help="A sweep leaf dir: .../Ts*/seed*/")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: <run_dir>/poi_outcomes_ep*.png)")
    ap.add_argument("--show_obstacles", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found under: {run_dir}")

    # prefer resolved_config.json emitted by runner
    resolved_path = run_dir / "resolved_config.json"
    if not resolved_path.exists():
        raise FileNotFoundError(f"resolved_config.json not found under: {run_dir}")
    cfg = json.loads(resolved_path.read_text(encoding="utf-8"))
    scene_file = str(cfg.get("scene_file", ""))
    if not scene_file:
        raise ValueError("scene_file missing from resolved_config.json")
    # Scenes often use `extends: ./base.yaml`; load_resolved_config resolves it.
    scene = load_resolved_config(scene_file)

    row = _load_metrics_row(metrics_path, episode=int(args.episode))
    covered_ids = row.get("covered_ids")
    effective_ids = row.get("effective_ids")
    if covered_ids is None or effective_ids is None:
        raise ValueError(
            "metrics.jsonl missing covered_ids/effective_ids. "
            "Run with runner module `uavlab.paper1.runner.run_vis`."
        )

    covered: Set[int] = set(int(i) for i in covered_ids)
    effective: Set[int] = set(int(i) for i in effective_ids)

    poi_list = _as_points(scene.get("poi_list"))
    gcs = tuple(scene.get("gcs_ne") or scene.get("start_ne") or (0.0, 0.0))
    gcs_ne: Point2D = (float(gcs[0]), float(gcs[1]))

    eff_xy = [poi_list[i] for i in range(len(poi_list)) if i in effective]
    cov_only_xy = [poi_list[i] for i in range(len(poi_list)) if (i in covered and i not in effective)]
    unvisited_xy = [poi_list[i] for i in range(len(poi_list)) if i not in covered]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=160)
    ax.set_title(f"POI outcomes (ep={args.episode})\n{run_dir.name}")

    if unvisited_xy:
        ax.scatter([p[0] for p in unvisited_xy], [p[1] for p in unvisited_xy], s=28, c="#9aa0a6", label="unvisited")
    if cov_only_xy:
        ax.scatter([p[0] for p in cov_only_xy], [p[1] for p in cov_only_xy], s=36, c="#fbbc04", label="visited, not returned")
    if eff_xy:
        ax.scatter([p[0] for p in eff_xy], [p[1] for p in eff_xy], s=40, c="#34a853", label="visited & returned")

    ax.scatter([gcs_ne[0]], [gcs_ne[1]], s=90, c="#1a73e8", marker="*", label="GCS/home")

    if "dist_to_home_m" in row:
        ax.text(
            0.01,
            0.01,
            f"dist_to_home={float(row['dist_to_home_m']):.1f} m",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    if args.show_obstacles:
        for cx, cy, r in _as_circles(scene.get("nofly_circles")):
            circ = plt.Circle((cx, cy), r, fill=False, lw=1.0, ec="#ea4335", alpha=0.7)
            ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("N")
    ax.set_ylabel("E")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    out = Path(args.out).resolve() if args.out.strip() else (run_dir / f"poi_outcomes_ep{int(args.episode)}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(str(out))


if __name__ == "__main__":
    main()

