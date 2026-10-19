"""Every row of the legacy configuration surface must be mapped in acot/config/legacy_map.yaml, and the map must not keep
entries for rows that no longer exist (framework §6.2)."""
import copy

from acot.config import coverage
from acot.tools.legacy_surface import collect


def test_full_coverage():
    assert coverage.problems() == {}


def test_detects_unmapped_and_stale():
    s, M = collect(), coverage.load_map()
    M2 = copy.deepcopy(M)
    del M2["policy_server"]["--conf-tau"]                  # a flag nobody maps
    M2["env"]["NO_SUCH_VAR"] = {"class": "infra"}          # a stale entry
    p = coverage.problems(s, M2)
    assert p["unmapped_flags"] == ["carla_uav_tracking/rollout/policy_server.py:--conf-tau"]
    assert "NO_SUCH_VAR" in p["stale_map_entries"]
    s2 = copy.deepcopy(s)
    s2["top_level"].append(("new_dir/", "UNCLASSIFIED"))
    assert coverage.problems(s2, M)["unclassified_top_level"] == ["new_dir/"]
