"""Batch generate multiple tracking episodes on a single CARLA server.

Usage:
    python scripts/generate_batch.py --num-episodes 100 --config config/scenarios/urban_hard.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import carla
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scene.scene_manager import SceneManager
from recording.recorder import EpisodeRecorder
from recording.postprocess import PostProcessor


def main():
    parser = argparse.ArgumentParser(description="Batch generate tracking episodes")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output", type=str, default="/data",
                        help="Output dir; container /data == host carla_data mount")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; per-episode seed = seed + episode_id")
    parser.add_argument("--max-size-gb", type=float, default=45.0,
                        help="Stop when total data exceeds this")
    parser.add_argument("--postprocess", action="store_true", default=True)
    args = parser.parse_args()

    # Load config
    cfg = _load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_size_gb * 1024**3)
    town_list = cfg.get("environment", {}).get("towns", ["Town01"])

    # Connect (generous timeout: first map load compiles shaders)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    summaries = []
    postproc = PostProcessor(output_dir, fov=float(cfg.get("output", {}).get("fov", 90.0)))

    # Load the scenario's first town up front (skip if the server is already on
    # it — important for crash-resume so we don't reload every restart).
    first_town = town_list[0] if town_list else "Town10HD"
    cur_map = client.get_world().get_map().name.split("/")[-1]
    if cur_map != first_town:
        try:
            print(f"Loading initial map: {first_town}")
            client.load_world(first_town)
            time.sleep(3.0)
        except RuntimeError:
            print("  Initial map load failed; using current map")

    progress_file = output_dir / ".attempted"
    for ep_id in range(args.start_id, args.start_id + args.num_episodes):
        # Record attempt so a supervisor can resume past a crashing episode.
        progress_file.write_text(str(ep_id))
        # Map switch every 10 episodes (with crash guard)
        if ep_id % 10 == 0 and len(town_list) > 1:
            town = town_list[(ep_id // 10) % len(town_list)]
            print(f"[{ep_id}] Loading map: {town}")
            try:
                client.load_world(town)
                time.sleep(3.0)
            except RuntimeError:
                print(f"  Map load failed, continuing with current map")

        world = client.get_world()

        # Update scenario-specific town override
        if "towns" not in cfg.get("environment", {}):
            cfg["environment"]["towns"] = town_list

        # Per-episode seed for reproducibility (P1-b)
        cfg.setdefault("environment", {})["seed"] = args.seed + ep_id

        scene = SceneManager(world, cfg, client)
        recorder = EpisodeRecorder(
            scene=scene, world=world, output_dir=output_dir,
            episode_id=ep_id, fps=cfg.get("environment", {}).get("fps", 10),
            resolution=tuple(cfg.get("output", {}).get("rgb_resolution", [336, 336])),
        )

        print(f"[{ep_id}] Starting...", end=" ", flush=True)
        summary = recorder.run()
        summaries.append(summary)

        n_steps = summary["steps"]
        dur = summary["duration_seconds"]
        if summary.get("skipped"):
            print(f"SKIPPED — {summary.get('reason', 'unknown')}")
            # Remove the empty H5 file
            ep_file = output_dir / f"episode_{ep_id:06d}.h5"
            if ep_file.exists():
                ep_file.unlink()
            continue
        print(f"{n_steps} steps in {dur:.1f}s ({n_steps/max(dur,0.001):.1f} fps)")

        # Check size limit
        total_size = sum(
            f.stat().st_size for f in output_dir.glob("episode_*.h5")
        )
        print(f"  Total data: {total_size/1024**3:.2f} GB / {args.max_size_gb:.1f} GB")
        if total_size >= max_bytes:
            print(f"  Reached {args.max_size_gb} GB limit. Stopping.")
            break

        # Post-process
        if args.postprocess:
            ep_path = output_dir / f"episode_{ep_id:06d}.h5"
            if ep_path.exists():
                postproc.process_episode(ep_path)

    # Save batch summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    total_steps = sum(s["steps"] for s in summaries)
    total_dur = sum(s["duration_seconds"] for s in summaries)
    print(f"\nBatch complete. {len(summaries)} episodes, {total_steps} steps, "
          f"{total_dur/60:.1f} min")
    print(f"Summary: {summary_path}")


def _load_config(config_path: str | None) -> dict:
    base_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if config_path:
        with open(config_path) as f:
            override = yaml.safe_load(f)
        _deep_merge(cfg, override)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    main()
