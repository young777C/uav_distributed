#!/usr/bin/env python3
"""Calibrate communication degradation tiers using the actual Paper1 link model."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config


CASES = [
    "c1_g2_m2_low",
    "c1_g2_m2_mid",
    "c1_g2_m2_high",
    "c1_g2_m2_severe",
    "c2_g2_m2_low",
    "c2_g2_m2_mid",
    "c2_g2_m2_high",
    "c2_g2_m2_severe",
]

MULTIPLIERS = {
    "low": 0.75,
    "mid": 1.0,
    "high": 1.25,
    "severe": 1.5,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    case_root = ROOT / "configs/experiments/paper1/cases/comm_degradation_scan"

    for case_name in CASES:
        case_path = case_root / f"{case_name}.yaml"
        cfg = apply_experiment_presets(load_resolved_config(case_path))
        scene_path = str(cfg["scene_file"])
        scene_raw = load_scene_yaml(scene_path)
        scene_geom = load_scene_config(scene_path)
        sim_cfg = from_resolved_config(cfg, scene_raw)
        env = build_env(sim_cfg, scene_geom)

        xs = np.linspace(sim_cfg.n_min, sim_cfg.n_max, args.grid_size)
        ys = np.linspace(sim_cfg.e_min, sim_cfg.e_max, args.grid_size)
        losses: list[float] = []
        delays: list[float] = []
        bandwidths: list[float] = []
        blackhole_samples = 0

        for seed in args.seeds:
            env.reset(seed=int(seed))
            for x in xs:
                for y in ys:
                    env.pos_ne = (float(x), float(y))
                    blackhole_samples += int(env.in_blackhole(env.pos_ne))
                    link = env.observe_link_state()
                    losses.append(float(link.loss_p))
                    delays.append(float(link.delay_s))
                    bandwidths.append(float(link.bandwidth_bps))

        comm_case = "C1" if case_name.startswith("c1_") else "C2"
        intensity = case_name.rsplit("_", 1)[-1]
        comm = dict(cfg.get("comm") or {})
        rows.append(
            {
                "comm_case": comm_case,
                "intensity": intensity,
                "multiplier": MULTIPLIERS[intensity],
                "samples": len(losses),
                "mean_loss_p": float(np.mean(losses)),
                "std_loss_p": float(np.std(losses, ddof=1)),
                "mean_delay_s": float(np.mean(delays)),
                "std_delay_s": float(np.std(delays, ddof=1)),
                "mean_bandwidth_bps": float(np.mean(bandwidths)),
                "std_bandwidth_bps": float(np.std(bandwidths, ddof=1)),
                "data_degraded_ratio_loss_gt_0_2": float(np.mean(np.asarray(losses) > 0.20)),
                "weak_link_ratio_loss_ge_0_5": float(np.mean(np.asarray(losses) >= 0.50)),
                "blackhole_sample_ratio": float(blackhole_samples / max(1, len(losses))),
                "distance_loss_min": comm.get("distance_loss_min", ""),
                "distance_loss_max": comm.get("distance_loss_max", ""),
                "base_loss": comm.get("base_loss", ""),
                "blackhole_extra_loss": comm.get("blackhole_extra_loss", ""),
                "loss_jitter_sigma": comm.get("loss_jitter_sigma", ""),
                "delay_mean_config_s": comm.get("delay_mean_s", ""),
                "delay_jitter_config_s": comm.get("delay_jitter_s", ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)
    for row in rows:
        print(
            f"{row['comm_case']} {row['intensity']}: "
            f"loss={row['mean_loss_p']:.4f}, delay={row['mean_delay_s']:.4f}s, "
            f"bw={row['mean_bandwidth_bps'] / 1e6:.3f}Mbps, "
            f"weak={row['weak_link_ratio_loss_ge_0_5']:.4f}"
        )


if __name__ == "__main__":
    main()
