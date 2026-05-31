#!/usr/bin/env python3
"""
Compare Paper1 struct-axis diag.jsonl logs (goal_id / planned_sequence / POI timing).

Usage:
  PYTHONPATH=src python3 scripts/analyze_struct_goal_trace.py \\
    --root runs/debug/struct_goal_trace/c1_g2_m2 \\
    --episode 0
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_diag(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _find_run_dirs(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(root.glob("*__struct_*")):
        if (p / "Ts40/seed0/diag.jsonl").is_file():
            key = p.name.split("__struct_", 1)[-1]
            out[key] = p / "Ts40/seed0"
    return out


def _bootstrap_seq(rows: Iterable[Dict[str, Any]], episode: int) -> Tuple[int, ...]:
    for r in rows:
        if r.get("event") == "slow_bootstrap" and int(r.get("episode", -1)) == episode:
            return tuple(int(x) for x in (r.get("planned_sequence") or []))
    return tuple()


def _replan_events(rows: Iterable[Dict[str, Any]], episode: int) -> List[Dict[str, Any]]:
    return [
        r
        for r in rows
        if int(r.get("episode", -1)) == episode and str(r.get("event")) == "slow_replan"
    ]


def _spatial_legs(rows: Iterable[Dict[str, Any]], episode: int) -> List[Dict[str, Any]]:
    return [
        r
        for r in rows
        if int(r.get("episode", -1)) == episode and str(r.get("event")) == "spatial_complete"
    ]


def _print_sequence_alignment(runs: Dict[str, Path], episode: int) -> None:
    print("\n=== planned_sequence alignment ===")
    boot: Dict[str, Tuple[int, ...]] = {}
    for name, d in runs.items():
        rows = _load_diag(d / "diag.jsonl")
        boot[name] = _bootstrap_seq(rows, episode)
        head = list(boot[name][:12])
        print(f"  {name}: t=0 len={len(boot[name])} head={head}")

    names = list(boot.keys())
    if len(names) >= 2:
        ref = boot[names[0]]
        for name in names[1:]:
            same = boot[name] == ref
            print(f"  {names[0]} vs {name} @t=0: {'SAME' if same else 'DIFF'}")
            if not same and ref and boot[name]:
                for i, (a, b) in enumerate(zip(ref[:16], boot[name][:16])):
                    if a != b:
                        print(f"    first mismatch index {i}: {a} vs {b}")
                        break

    print("\n  replan count (sequence or goal change):")
    for name, d in runs.items():
        n = len(_replan_events(_load_diag(d / "diag.jsonl"), episode))
        print(f"    {name}: {n}")


def _print_spatial_stats(runs: Dict[str, Path], episode: int) -> None:
    print("\n=== spatial_complete (leg_wall_s = t - goal_assign_t) ===")
    for name, d in runs.items():
        legs = _spatial_legs(_load_diag(d / "diag.jsonl"), episode)
        vals = [float(r["leg_wall_s"]) for r in legs if r.get("leg_wall_s") is not None]
        if not vals:
            print(f"  {name}: no leg_wall_s samples")
            continue
        vals.sort()
        mid = vals[len(vals) // 2]
        print(
            f"  {name}: n={len(vals)}  median_leg={mid:.1f}s  "
            f"min={vals[0]:.1f}s  max={vals[-1]:.1f}s"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Parent dir with c1_g2_m2__struct_* run folders")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    runs = _find_run_dirs(root)
    if not runs:
        raise SystemExit(f"No *__struct_*/Ts40/seed0/diag.jsonl under {root}")

    print(f"root={root}  episode={args.episode}  runs={list(runs.keys())}")
    _print_sequence_alignment(runs, int(args.episode))
    _print_spatial_stats(runs, int(args.episode))


if __name__ == "__main__":
    main()
