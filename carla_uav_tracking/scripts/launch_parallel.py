"""Launch parallel CARLA servers for multi-GPU data generation.

Usage:
    python scripts/launch_parallel.py --num-servers 4 --episodes-per-server 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CARLA_ROOT = os.environ.get("CARLA_ROOT", "/opt/carla")
CARLA_SERVER = f"{CARLA_ROOT}/CarlaUE4.sh"


def main():
    parser = argparse.ArgumentParser(description="Parallel data generation launcher")
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument("--episodes-per-server", type=int, default=50)
    parser.add_argument("--base-port", type=int, default=2000)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="./data/raw")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)

    processes = []
    output_dir = Path(args.output)

    for i in range(args.num_servers):
        port = args.base_port + i * 2
        gpu = gpu_ids[i % num_gpus]
        server_output = output_dir / f"server_{i:02d}"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["DISPLAY"] = ""  # headless

        # Launch CARLA server
        cmd = [
            CARLA_SERVER,
            f"-carla-rpc-port={port}",
            "-quality-level=Low",
            "-benchmark",
            "-fps=20",
            "-nosound",
            "-RenderOffScreen",
        ]
        print(f"[Server {i}] GPU={gpu}, Port={port}, Map={None}  # will be set per-episode")
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=open(server_output / "stdout.log", "w"),
            stderr=open(server_output / "stderr.log", "w"),
        )
        processes.append((i, proc, port, gpu))

    # Wait for servers to be ready
    print("\nWaiting for servers to start...")
    time.sleep(15)

    # Launch workers (one per server)
    worker_procs = []
    script_dir = Path(__file__).resolve().parent

    for i, server_proc, port, gpu in processes:
        worker_output = output_dir / f"worker_{i:02d}"
        worker_output.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            sys.executable,
            str(script_dir / "generate_batch.py"),
            f"--num-episodes={args.episodes_per_server}",
            f"--start-id={i * args.episodes_per_server}",
            f"--port={port}",
            f"--output={worker_output}",
        ]
        if args.config:
            cmd.append(f"--config={args.config}")

        print(f"[Worker {i}] GPU={gpu}, Port={port}")
        proc = subprocess.Popen(cmd, env=env)
        worker_procs.append((i, proc))

    # Wait for all workers
    print(f"\n{len(worker_procs)} workers running. Waiting for completion...")
    for i, proc in worker_procs:
        ret = proc.wait()
        print(f"[Worker {i}] Exit code: {ret}")

    # Cleanup CARLA servers
    print("\nShutting down CARLA servers...")
    for i, proc, port, gpu in processes:
        proc.terminate()
        proc.wait(timeout=10)

    # Merge summary
    print("\nAll workers complete.")
    total_frames = 0
    for i in range(args.num_servers):
        batch_dir = output_dir / f"worker_{i:02d}"
        for ep_file in batch_dir.glob("episode_*.h5"):
            import h5py
            with h5py.File(ep_file, "r") as f:
                total_frames += f["rgb"].shape[0]
    print(f"Total frames generated: {total_frames:,}")


if __name__ == "__main__":
    main()
