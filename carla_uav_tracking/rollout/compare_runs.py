"""Paired closed-loop comparison of two rollout JSONs (design §8 statistics).

Given two run_rollout outputs on the SAME seed set, pair episodes by seed and report,
for a panel of metrics, the paired mean(A), mean(B), Δ(B−A) and a per-episode
bootstrap 95% CI. Used for all hypotheses:
  H0 language:  A=M0 with-lang, B=M1 no-lang   → watch mis_follow_sustained ↑
  H1 EAR:       A=M0 full,      B=M3 w/o-EAR    → watch track_seconds ↓ / mis-follow
  H2 IAR:       A=M0 full,      B=M4 w/o-IAR    → watch track_seconds ↓
A metric's effect is significant iff its Δ 95% CI excludes 0.

    python carla_uav_tracking/rollout/compare_runs.py A.json B.json --label-a M0 --label-b M3
"""

from __future__ import annotations

import argparse
import json

import numpy as np

PANEL = ["mis_follow_sustained", "mis_follow_inst", "track_seconds",
         "id_switches", "centering_rate", "reacquire_lock_wrong_rate"]


def _scored_by_seed(path):
    d = json.load(open(path))
    return {int(r["episode_seed"]): r for r in d["episodes"] if not r.get("skipped")}


def _boot_ci(diffs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    m = diffs[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a"); ap.add_argument("run_b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    a = ap.parse_args()

    A, B = _scored_by_seed(a.run_a), _scored_by_seed(a.run_b)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        print("[compare] no paired scored episodes (need same seeds in both runs)")
        return 1

    sr_a = np.mean([bool(A[s]["success"]) for s in seeds])
    sr_b = np.mean([bool(B[s]["success"]) for s in seeds])
    print(f"[compare] paired episodes: {len(seeds)}  (seeds {seeds[0]}..{seeds[-1]})")
    print(f"[compare] {'metric':<26} {a.label_a:>10} {a.label_b:>10} {'Δ(B−A)':>10}   95% CI       sig")
    print(f"[compare] {'SR (success rate)':<26} {sr_a:>10.3f} {sr_b:>10.3f} {sr_b-sr_a:>+10.3f}   {'—':>14}")
    for m in PANEL:
        pairs = [(A[s].get(m), B[s].get(m)) for s in seeds]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        va = np.array([x for x, _ in pairs], float)
        vb = np.array([y for _, y in pairs], float)
        diff = vb - va
        lo, hi = _boot_ci(diff)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"[compare] {m:<26} {va.mean():>10.4f} {vb.mean():>10.4f} {diff.mean():>+10.4f}   "
              f"[{lo:+.3f},{hi:+.3f}]  {sig}")
    print("[compare] (* = Δ 95% CI excludes 0 → significant effect at this MVP scale)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
