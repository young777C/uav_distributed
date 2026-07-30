"""A1c③ step-1 probe: measure per-town OVERHEAD-occluder density.

For a top-down/oblique UAV camera, in-frame structural occlusion comes mostly from
structures ABOVE the road: bridge decks, overpasses, tunnel roofs, sign gantries,
tree canopy. This probe casts a vertical ray up from each driving waypoint and reports
what fraction of the road network sits under an overhead structure (and its clearance).

That fraction ranks towns for the scene-redesign town mix, and validates the
up-raycast detector we will reuse to route the target through occluders.

Usage (inside container, against a server on <port>):
    python scripts/probe_occluders.py 2100 Town01,Town02,Town03,Town04,Town05,Town06,Town07,Town10HD
"""

from __future__ import annotations

import sys

import carla
import numpy as np


def probe(world, sample_step_m: float = 5.0, max_samples: int = 2000):
    m = world.get_map()
    wps = [w for w in m.generate_waypoints(sample_step_m)
           if w.lane_type == carla.LaneType.Driving]
    step = max(1, len(wps) // max_samples)
    wps = wps[::step]
    overhead = 0
    clearances = []
    for w in wps:
        loc = w.transform.location
        start = carla.Location(loc.x, loc.y, loc.z + 0.5)
        end = carla.Location(loc.x, loc.y, loc.z + 45.0)
        hits = world.cast_ray(start, end)
        above = [h.location.z - loc.z for h in hits if 2.0 < (h.location.z - loc.z) < 45.0]
        if above:
            overhead += 1
            clearances.append(min(above))
    n = len(wps)
    return {
        "n": n,
        "overhead_pct": 100.0 * overhead / max(n, 1),
        "median_clearance": float(np.median(clearances)) if clearances else 0.0,
        "low_clearance_pct": 100.0 * sum(1 for c in clearances if c < 10.0) / max(n, 1),
    }


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 2100
    towns = (sys.argv[2].split(",") if len(sys.argv) > 2 else
             ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"])
    client = carla.Client("localhost", port)
    client.set_timeout(120.0)
    print(f"{'town':10s} {'wps':>6s} {'overhead%':>9s} {'med_clr':>8s} {'low_clr%':>9s}")
    rows = []
    for town in towns:
        world = client.load_world(town)
        for _ in range(10):
            world.wait_for_tick()
        r = probe(world)
        rows.append((town, r))
        print(f"{town:10s} {r['n']:6d} {r['overhead_pct']:8.1f}% {r['median_clearance']:7.1f}m "
              f"{r['low_clearance_pct']:8.1f}%")
    rows.sort(key=lambda kv: kv[1]["overhead_pct"], reverse=True)
    print("\nRANK (by overhead occluder density):")
    for town, r in rows:
        print(f"  {town:10s} {r['overhead_pct']:5.1f}%  (clearance {r['median_clearance']:.0f}m)")


if __name__ == "__main__":
    main()
