"""Detect whether CARLA data generation has FINISHED — metadata only.

SAFETY: never opens any .h5 (no h5py, no read) so it cannot lock / disturb the
generator writing the files. Uses only directory-entry stat (name/size/mtime).

"Ready" = at least one .h5 exists AND no .h5 has been modified in the last
`--stable-min` minutes (generator idle) AND no obviously-growing partial file.

Exit code 0 = ready, 1 = not ready. Prints a one-line JSON status.

    python -m train.check_data_ready --dir /nvidia/hque/data/carla_data/mvp --stable-min 20
"""

from __future__ import annotations

import argparse
import json
import os
import time


def scan(dir_path):
    """Return (count, total_bytes, newest_mtime) using stat only (no file open)."""
    count, total, newest = 0, 0, 0.0
    try:
        with os.scandir(dir_path) as it:
            for e in it:
                if not e.name.endswith(".h5"):
                    continue
                st = e.stat()                     # inode metadata only; does not open
                count += 1
                total += st.st_size
                newest = max(newest, st.st_mtime)
    except FileNotFoundError:
        pass
    return count, total, newest


def status(dir_path, stable_min):
    now = time.time()
    count, total, newest = scan(dir_path)
    idle_s = (now - newest) if newest else 0.0
    ready = count > 0 and idle_s >= stable_min * 60
    return {
        "ready": ready,
        "episodes": count,
        "total_gb": round(total / 1e9, 2),
        "idle_min": round(idle_s / 60, 1),
        "stable_min": stable_min,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/nvidia/hque/data/carla_data/mvp")
    ap.add_argument("--stable-min", type=float, default=20.0)
    a = ap.parse_args()
    s = status(a.dir, a.stable_min)
    print(json.dumps(s))
    raise SystemExit(0 if s["ready"] else 1)


if __name__ == "__main__":
    main()
