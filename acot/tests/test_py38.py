"""The CARLA-side client runs under cyh-carla's python 3.8 (framework §13.3): its sources and the legacy rollout modules
it imports must parse with the 3.8 grammar (the preflight additionally imports them inside the container)."""
import ast
import glob
import os

from acot import ROOT

FILES = sorted(glob.glob(os.path.join(ROOT, "carla_uav_tracking", "acot_client", "*.py"))) + [
    os.path.join(ROOT, "carla_uav_tracking", "rollout", f) for f in
    ("env.py", "run_rollout.py", "remote_policy.py", "bridge.py", "metrics.py")] + \
    sorted(glob.glob(os.path.join(ROOT, "carla_uav_tracking", "scene", "*.py")))


def test_carla_side_sources_parse_as_python38():
    assert len(FILES) > 6
    for f in FILES:
        ast.parse(open(f).read(), filename=f, feature_version=(3, 8))
