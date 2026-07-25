"""Generate a single tracking episode — for debugging and testing.

Usage:
    python scripts/generate_single.py --config config/scenarios/urban_hard.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import carla
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(config_path: str) -> dict:
    """Load scenario config, merged with defaults."""
    with open(Path(__file__).resolve().parent.parent / "config" / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            override = yaml.safe_load(f)
        # Simple merge
        _deep_merge(cfg, override)

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def main():
    parser = argparse.ArgumentParser(description="Generate one tracking episode")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to scenario YAML config")
    parser.add_argument("--host", type=str, default="localhost",
                        help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA server port")
    parser.add_argument("--output", type=str, default="./data/raw",
                        help="Output directory")
    parser.add_argument("--episode-id", type=int, default=0,
                        help="Episode ID for output filename")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max episode steps")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to CARLA
    print(f"Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(cfg.get("environment", {}).get("carla", {}).get("timeout", 20.0))

    world = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")

    # Setup scene
    from scene.scene_manager import SceneManager
    from recording.recorder import EpisodeRecorder

    scene = SceneManager(world, cfg, client)
    recorder = EpisodeRecorder(
        scene=scene,
        world=world,
        output_dir=output_dir,
        episode_id=args.episode_id,
        fps=cfg.get("environment", {}).get("fps", 10),
    )

    # Run
    print(f"Starting episode {args.episode_id}...")
    summary = recorder.run(max_steps=args.max_steps)
    if summary.get("skipped"):
        print(f"Skipped: {summary.get('reason', 'unknown')}")
    else:
        print(f"Done. {summary['steps']} steps in {summary['duration_seconds']:.1f}s "
              f"({summary['fps_actual']:.1f} fps)")
        print(f"Output: {summary.get('output_file', 'N/A')}")


if __name__ == "__main__":
    main()
