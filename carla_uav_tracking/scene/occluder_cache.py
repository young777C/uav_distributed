"""A1c③: per-town cache of "occluder-leading" spawn anchors.

For a top-down/oblique UAV camera, in-frame structural occlusion comes from structures
ABOVE the road: bridge decks, overpasses, tunnel roofs, sign gantries, tree canopy. A
map spawn point is "occluder-leading" if, driving forward along its lane for up to
`look_ahead_m`, the road passes under such an overhead structure within a
UAV-reachable clearance band [min_clearance_m, max_clearance_m]. Spawning the target
there makes it drive into a TRANSIENT in-frame occlusion while staying on a valid,
collision-safe map spawn point (we reuse the map's curated spawn points, not arbitrary
waypoints, so the target/distractor spawn stays robust).

Caches are per-town JSON under cache_dir so the (raycast-heavy) build runs once.
"""

from __future__ import annotations

import json
from pathlib import Path

import carla


def _overhead(world: carla.World, loc: carla.Location, up_ray_m: float,
              min_clearance_m: float, max_clearance_m: float) -> bool:
    """True if an overhead structure sits within the clearance band above `loc`."""
    start = carla.Location(loc.x, loc.y, loc.z + 0.5)
    end = carla.Location(loc.x, loc.y, loc.z + up_ray_m)
    for h in world.cast_ray(start, end):
        dz = h.location.z - loc.z
        if min_clearance_m < dz < max_clearance_m:
            return True
    return False


def build_anchors(world: carla.World, look_ahead_m: float = 60.0, step_m: float = 4.0,
                  up_ray_m: float = 45.0, min_clearance_m: float = 2.0,
                  max_clearance_m: float = 22.0, min_enter_m: float = 12.0,
                  max_cover_span_m: float = 26.0) -> list[dict]:
    """Return map spawn points that yield a TRANSIENT overhead crossing: the lane is
    clear for the first `min_enter_m`, then passes under an occluder for a SHORT span
    (≤ `max_cover_span_m`), then clears again. This drives the target under a
    bridge/overpass and OUT the other side (re-emergence) — rejecting tunnels and long
    elevated-highway cover, which would occlude the target permanently and truncate."""
    m = world.get_map()
    out: list[dict] = []
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location, project_to_road=True,
                            lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        # sample overhead coverage along the forward path
        samples: list[tuple[float, bool]] = []
        cur = wp
        dist = 0.0
        while dist <= look_ahead_m:
            samples.append((dist, _overhead(world, cur.transform.location, up_ray_m,
                                            min_clearance_m, max_clearance_m)))
            nxt = cur.next(step_m)
            if not nxt:
                break
            cur = nxt[0]
            dist += step_m
        # must start clear (drive INTO the occluder, not spawn under it)
        if any(c for d, c in samples if d < min_enter_m):
            continue
        enter = next((d for d, c in samples if c), None)
        if enter is None:
            continue                                   # never enters an occluder
        exit_ = next((d for d, c in samples if d > enter and not c), None)
        if exit_ is None:
            continue                                   # never clears → tunnel / long deck
        span = exit_ - enter
        if span > max_cover_span_m:
            continue                                   # long cover (elevated highway) → skip
        t = sp
        out.append({"x": t.location.x, "y": t.location.y, "z": t.location.z,
                    "pitch": t.rotation.pitch, "yaw": t.rotation.yaw,
                    "roll": t.rotation.roll, "enter_m": enter, "span_m": span})
    return out


def load_or_build(world: carla.World, town: str, cache_dir: str,
                  params: dict | None = None, rebuild: bool = False) -> list[dict]:
    """Load the town's occluder-anchor cache, building (and saving) it if absent."""
    path = Path(cache_dir) / f"{town}_occluders.json"
    if path.exists() and not rebuild:
        return json.loads(path.read_text())
    data = build_anchors(world, **(params or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def anchors_to_transforms(data: list[dict]) -> list[carla.Transform]:
    return [carla.Transform(
        carla.Location(a["x"], a["y"], a["z"]),
        carla.Rotation(pitch=a["pitch"], yaw=a["yaw"], roll=a["roll"])) for a in data]
