#!/usr/bin/env python3
"""Summarize coupling-axis diag.jsonl: replan reasons, goal oscillation, mode ratios."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_diag_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("diag.jsonl"))


def _goal_oscillation_count(goals: List[Any]) -> int:
    flips = 0
    for i in range(2, len(goals)):
        if goals[i] == goals[i - 2] and goals[i] != goals[i - 1]:
            flips += 1
    return flips


def analyze_leaf(diag_path: Path) -> Dict[str, Any]:
    events = [json.loads(l) for l in diag_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    replans = [e for e in events if e.get("event") == "slow_replan"]
    reasons = Counter(str(e.get("replan_reason", "unknown")) for e in replans)
    goals = [e.get("goal_after") for e in replans if e.get("goal_after") is not None]
    switches = sum(1 for e in events if e.get("event") == "goal_switch")
    return {
        "diag_path": str(diag_path),
        "replan_count": len(replans),
        "unique_goals": len(set(goals)),
        "goal_oscillation": _goal_oscillation_count(goals),
        "goal_switch_events": switches,
        "replan_reasons": dict(reasons),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep_root", type=str, required=True)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    root = Path(args.sweep_root).resolve()
    by_exp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for dp in _iter_diag_files(root):
        try:
            rel = dp.relative_to(root)
            exp_name = rel.parts[0] if len(rel.parts) >= 1 else dp.parent.name
        except ValueError:
            exp_name = dp.parent.name
        by_exp[exp_name].append(analyze_leaf(dp))

    summary: Dict[str, Any] = {}
    for exp, rows in sorted(by_exp.items()):
        n = max(1, len(rows))
        summary[exp] = {
            "leaves": len(rows),
            "replan_count_mean": sum(r["replan_count"] for r in rows) / n,
            "goal_oscillation_mean": sum(r["goal_oscillation"] for r in rows) / n,
            "unique_goals_mean": sum(r["unique_goals"] for r in rows) / n,
            "replan_reasons_merged": dict(
                Counter(k for r in rows for k in r.get("replan_reasons", {}))
            ),
            "samples": rows,
        }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json.strip():
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
