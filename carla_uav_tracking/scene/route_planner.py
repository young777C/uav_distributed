"""1026 stress_v2: per-town lane graph with overhead-occluder coverage + a turn / occluder-seeking route planner.

stress_v1 occlusion came from occluder_cache.py: map spawn points whose first 60 m pass under an overhead structure.
In practice those were almost all 4 m spans (lamp-post arms / gantries, ~0.4 s) and only ever in the first 3-5 s of
an episode. stress_v2 instead plans the WHOLE route of the scene convoy (target + look-alikes + distractors):

  graph   map.get_topology() lane segments, sampled every `step_m`; each sample gets an upward ray-cast: covered if a
          non-vehicle surface sits `min_clear`..`max_clear` m above the road (bridge deck, overpass, tunnel roof,
          canopy, ...). Cached per town as <cache_dir>/<town>_routegraph_v<CACHE_VER>.json (pure data, no carla).
  plan    Monte-Carlo walks over the graph (junction choices biased to LEFT / RIGHT) from a start pose; each walk is
          scored on the covered runs it drives through (span in [min_span, max_span] m, entered ≥ min_enter m in) and
          on the number of ≥ turn_deg junction turns; the best walk is returned as sparse TM.set_path locations (one
          after every junction + every `pt_every_m` m) plus the planned events for the episode metadata.

TM.set_path follows the locations road-by-road (at each junction it takes the branch reaching the next location's
road; a location is consumed once the vehicle's buffer passes within ~5.5 m), so the convoy may use any lane of the
planned roads. Everything here is deterministic given the rng.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CACHE_VER = 1
# surfaces that are not occluders (or are the road itself / traffic) for the upward coverage ray
_NOT_OCCLUDER = {"Roads", "Sidewalks", "Ground", "Terrain", "RoadLines", "Water", "Sky", "NONE", "None", "Any",
                 "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrians", "Rider", "Train",
                 "Vehicles", "Dynamic"}


def _yaw_diff(a: float, b: float) -> float:
    """signed b − a in degrees, wrapped to (−180, 180]"""
    d = (b - a + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


# ----------------------------------------------------------------------------------------------------------------------
# graph build (needs carla) / cache
# ----------------------------------------------------------------------------------------------------------------------
def _cover_label(world, loc, up_ray_m: float, min_clear: float, max_clear: float) -> str:
    """label of the first occluder surface min_clear..max_clear m above loc ('' = open sky)"""
    import carla
    a = carla.Location(loc.x, loc.y, loc.z + 0.5)
    b = carla.Location(loc.x, loc.y, loc.z + up_ray_m)
    try:
        hits = world.cast_ray(a, b)
    except RuntimeError:
        return ""
    for h in hits:
        dz = h.location.z - loc.z
        lab = str(h.label).split(".")[-1]
        if lab in _NOT_OCCLUDER:
            continue
        if min_clear < dz < max_clear:
            return lab
    return ""


def build_graph(world, step_m: float = 2.0, up_ray_m: float = 45.0, min_clear: float = 2.0,
                max_clear: float = 22.0) -> dict:
    """lane-segment graph of the loaded town: edges = topology segments as sampled points [x, y, z, yaw, label],
    succ = edge indices whose first point is within 0.6 m of this edge's last point."""
    m = world.get_map()
    edges = []
    for a, b in m.get_topology():
        seq = [a]
        try:
            seq += list(a.next_until_lane_end(step_m))
        except RuntimeError:
            pass
        if seq[-1].transform.location.distance(b.transform.location) > 0.3:
            seq.append(b)
        pts = []
        for w in seq:
            t = w.transform
            pts.append([round(t.location.x, 2), round(t.location.y, 2), round(t.location.z, 2),
                        round(t.rotation.yaw, 1), _cover_label(world, t.location, up_ray_m, min_clear, max_clear)])
        length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
        edges.append({"jn": bool(a.is_junction), "road": int(a.road_id), "lane": int(a.lane_id),
                      "len": round(length, 2), "pts": pts})
    grid = {}
    for i, e in enumerate(edges):
        x, y = e["pts"][0][:2]
        grid.setdefault((int(math.floor(x)), int(math.floor(y))), []).append(i)
    for e in edges:
        x, y = e["pts"][-1][:2]
        cx, cy = int(math.floor(x)), int(math.floor(y))
        succ = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    p = edges[j]["pts"][0]
                    if math.hypot(p[0] - x, p[1] - y) < 0.6 and abs(_yaw_diff(e["pts"][-1][3], p[3])) < 100.0:
                        succ.append(j)
        e["succ"] = sorted(set(succ))
    return {"ver": CACHE_VER, "step_m": step_m, "up_ray_m": up_ray_m, "min_clear": min_clear,
            "max_clear": max_clear, "edges": edges}


def load_or_build_graph(world, town: str, cache_dir: str, params: dict | None = None, rebuild: bool = False) -> dict:
    p = Path(cache_dir) / f"{town}_routegraph_v{CACHE_VER}.json"
    if p.exists() and not rebuild:
        g = json.loads(p.read_text())
        if g.get("ver") == CACHE_VER:
            return g
    g = build_graph(world, **(params or {}))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(g))
    tmp.replace(p)
    return g


# ----------------------------------------------------------------------------------------------------------------------
# planning (pure python)
# ----------------------------------------------------------------------------------------------------------------------
def _locate(graph: dict, xyz, yaw: float, max_d: float = 6.0):
    """(edge, point index) of the graph point nearest to xyz whose heading is within 60° of yaw (else None)"""
    best, bd = None, max_d
    for i, e in enumerate(graph["edges"]):
        for k, p in enumerate(e["pts"]):
            d = math.hypot(p[0] - xyz[0], p[1] - xyz[1])
            if d < bd and abs(p[2] - xyz[2]) < 4.0 and abs(_yaw_diff(p[3], yaw)) < 60.0:
                best, bd = (i, k), d
    return best


def _turn_of(graph: dict, i: int) -> float:
    """heading change (deg, + = left in CARLA's left-handed yaw → we only use |·|) across junction edge i"""
    pts = graph["edges"][i]["pts"]
    return _yaw_diff(pts[0][3], pts[-1][3])


def _walk(graph: dict, start, length_m: float, rng, turn_w: float, revisit_w: float):
    """one random walk: list of (edge, k0) pieces covering ≥ length_m (or until a dead end)"""
    edges = graph["edges"]
    i, k = start
    pieces, dist, seen = [(i, k)], 0.0, {i}
    pts = edges[i]["pts"]
    dist += sum(math.hypot(pts[j + 1][0] - pts[j][0], pts[j + 1][1] - pts[j][1]) for j in range(k, len(pts) - 1))
    while dist < length_m:
        succ = edges[i]["succ"]
        if not succ:
            break
        w = []
        for j in succ:
            x = 1.0
            if edges[j]["jn"] and abs(_turn_of(graph, j)) >= 45.0:
                x *= turn_w
            if j in seen:
                x *= revisit_w
            w.append(x)
        r, acc = rng.random() * sum(w), 0.0
        for j, x in zip(succ, w):
            acc += x
            if r <= acc:
                break
        i = j
        seen.add(i)
        pieces.append((i, 0))
        dist += edges[i]["len"]
    return pieces, dist


def _flatten(graph: dict, pieces):
    """→ samples [(s, x, y, z, yaw, label, edge)] along the walk (duplicate joint points dropped)"""
    out, s = [], 0.0
    for n, (i, k0) in enumerate(pieces):
        pts = graph["edges"][i]["pts"][k0:]
        if n > 0 and out:
            pts = pts[1:]
        for p in pts:
            if out:
                s += math.hypot(p[0] - out[-1][1], p[1] - out[-1][2])
            out.append((s, p[0], p[1], p[2], p[3], p[4], i))
    return out


def occluded_runs(samples, merge_gap_m: float = 4.0):
    """covered runs along the samples: [(enter_s, span_m, label)], gaps ≤ merge_gap_m merged"""
    runs, cur = [], None
    for s, *_r in samples:
        lab = _r[4]
        if lab:
            if cur is not None and s - cur[1] <= merge_gap_m:
                cur[1] = s
                cur[2][lab] = cur[2].get(lab, 0) + 1
            else:
                if cur is not None:
                    runs.append(cur)
                cur = [s, s, {lab: 1}]
    if cur is not None:
        runs.append(cur)
    return [(a, b - a, max(c, key=c.get)) for a, b, c in runs]


def _score(graph, pieces, samples, cfg):
    runs = occluded_runs(samples)
    good = [r for r in runs if cfg["min_span_m"] <= r[1] <= cfg["max_span_m"] and r[0] >= cfg["min_enter_m"]]
    too_long = [r for r in runs if r[1] > cfg["max_span_m"]]
    turns = [i for i, _ in pieces if graph["edges"][i]["jn"] and abs(_turn_of(graph, i)) >= cfg["turn_deg"]]
    edges = [i for i, _ in pieces]
    repeats = len(edges) - len(set(edges))
    sc = (cfg["w_occ"] * min(len(good), cfg["max_occ"]) + cfg["w_turn"] * min(len(turns), cfg["max_turns"])
          - 5.0 * len(too_long) - 2.0 * repeats)
    return sc, good, too_long, turns


DEFAULTS = {"length_m": 1300.0, "n_walks": 300, "turn_w": 4.0, "revisit_w": 0.05, "min_span_m": 12.0,
            "max_span_m": 60.0, "min_enter_m": 25.0, "turn_deg": 45.0, "w_occ": 3.0, "w_turn": 1.0, "max_occ": 3,
            "max_turns": 8, "pt_every_m": 40.0}


def plan_route(graph: dict, xyz, yaw: float, rng, cfg: dict | None = None) -> dict | None:
    """best of n_walks random walks from the pose (None if the pose is not on the graph)."""
    c = dict(DEFAULTS, **(cfg or {}))
    start = _locate(graph, xyz, yaw)
    if start is None:
        return None
    best = None
    for _ in range(int(c["n_walks"])):
        pieces, _d = _walk(graph, start, c["length_m"], rng, c["turn_w"], c["revisit_w"])
        samples = _flatten(graph, pieces)
        sc, good, too_long, turns = _score(graph, pieces, samples, c)
        if best is None or sc > best[0]:
            best = (sc, pieces, samples, good, too_long, turns)
    sc, pieces, samples, good, too_long, turns = best
    return {"score": sc, "samples": samples, "pieces": pieces, "length_m": samples[-1][0] if samples else 0.0,
            "occluders": [{"enter_m": round(a, 1), "span_m": round(b, 1), "label": l} for a, b, l in good],
            "too_long": [{"enter_m": round(a, 1), "span_m": round(b, 1), "label": l} for a, b, l in too_long],
            "turns": [{"edge": i, "deg": round(_turn_of(graph, i), 1)} for i in turns],
            "points": path_points(graph, samples, c["pt_every_m"])}


def path_points(graph: dict, samples, every_m: float):
    """sparse set_path locations [(s, x, y, z)]: the first sample ≥ 6 m past every junction edge + every every_m m"""
    out, last_s, prev_jn = [], 0.0, False
    pending_after = None
    for s, x, y, z, yaw, lab, i in samples:
        jn = graph["edges"][i]["jn"]
        if prev_jn and not jn:
            pending_after = s + 6.0
        if pending_after is not None and s >= pending_after:
            out.append((s, x, y, z))
            last_s, pending_after = s, None
        elif s - last_s >= every_m and not jn:
            out.append((s, x, y, z))
            last_s = s
        prev_jn = jn
    return out


def points_for(plan: dict, xyz, max_d: float = 8.0, ahead_m: float = 5.0):
    """the planned locations still ahead of a vehicle at xyz: drop every location at or behind its nearest sample
    (a location behind the vehicle would never be consumed and would stall TM's path import). Off-route → all."""
    best, bd = None, max_d
    for s, x, y, z, *_ in plan["samples"]:
        d = math.hypot(x - xyz[0], y - xyz[1])
        if d < bd:
            best, bd = s, d
    if best is None:
        return [p for p in plan["points"]]
    return [p for p in plan["points"] if p[0] > best + ahead_m]
