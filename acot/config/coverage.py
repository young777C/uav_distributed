"""Legacy-surface coverage (framework §6.2): every row of acot/tools/legacy_surface.py:collect() must be mapped in
acot/config/legacy_map.yaml and the map must hold no stale entries. Used by `acot lint` and the tests."""
import fnmatch
import os

import yaml

from acot.tools.legacy_surface import collect

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_map.yaml")
SECTION = {"carla_uav_tracking/rollout/policy_server.py": "policy_server",
           "carla_uav_tracking/rollout/run_rollout.py": "run_rollout",
           "carla_uav_tracking/rollout/dam4sam_service.py": "dam4sam_service"}


def load_map():
    return yaml.safe_load(open(MAP_PATH))


def problems(surface=None, M=None):
    """→ {check name: [offending keys]} (empty dict = full coverage)."""
    s = surface or collect()
    M = M or load_map()
    out = {}
    out["unclassified_top_level"] = [e for e, st in s["top_level"] if st == "UNCLASSIFIED"]
    out["unparsable_files"] = [f for f, _ in s["broken"]]
    out["unmapped_flags"] = [r["key"] for r in s["argparse"]
                             if r["category"] not in M["categories"] and r["file"] not in M["files"]
                             and (SECTION.get(r["file"]) is None or r["flag"] not in M[SECTION[r["file"]]])]
    out["unmapped_argv_scripts"] = [r["key"] for r in s["argv"]
                                    if r["category"] not in M["categories"] and r["key"] not in M["files"]]
    out["unmapped_constructor_fields"] = [r["key"] for r in s["signatures"]
                                          if r["field"] not in (M["constructors"].get(r["key"].split(":")[0]) or {})]
    out["unmapped_env"] = sorted(set(s["env"]) - set(M["env"])) + \
        sorted({r["key"] for r in s["env_dynamic"]} - set(M["env_dynamic"]))
    out["unmapped_wire"] = sorted({r["key"] for r in s["wire"]} - set(M["wire"]))
    out["unmapped_constants"] = sorted({r["key"] for r in s["constants"]} - set(M["constants"]))
    out["unmapped_launchers"] = [r["key"] for r in s["launchers"]
                                 if not any(fnmatch.fnmatch(r["key"], g) for g in M["launchers"])]
    files = {r["file"] for r in s["argparse"]} | {r["key"] for r in s["argv"]}
    flags = {(SECTION.get(r["file"]), r["flag"]) for r in s["argparse"]}
    sig = {r["key"] for r in s["signatures"]}
    stale = [f for f in M["files"] if f not in files]
    stale += [f"{sec}:{fl}" for sec in ("policy_server", "run_rollout", "dam4sam_service") for fl in M[sec]
              if (sec, fl) not in flags]
    stale += [f"{c}:{f}" for c, fields in M["constructors"].items() for f in fields if f"{c}:{f}" not in sig]
    stale += [v for v in M["env"] if v not in s["env"]]
    stale += [w for w in M["wire"] if w not in {r["key"] for r in s["wire"]}]
    stale += [c for c in M["constants"] if c not in {r["key"] for r in s["constants"]}]
    stale += [g for g in M["launchers"] if not any(fnmatch.fnmatch(r["key"], g) for r in s["launchers"])]
    out["stale_map_entries"] = stale
    return {k: v for k, v in out.items() if v}
