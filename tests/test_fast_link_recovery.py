"""Fast-loop mobile Srec recovery waypoint selection."""

from __future__ import annotations

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.loops.fast.guidance import pick_link_recovery_waypoint
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config


def test_pick_link_recovery_prefers_higher_link_toward_gcs():
    scene_path = "configs/scenes/g1_uniform.yaml"
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    cfg = {"scene_file": scene_path, "env": {"step_hz": 5}}
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    _ = env.reset(seed=0)
    # Far from GCS in a poor-link zone (edge of map).
    env.pos_ne = (50.0, 50.0)
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    params = FastLoopParams.from_contract(contract)
    wp = pick_link_recovery_waypoint(
        env=env,
        contract=contract,
        params=params,
        fallback_ne=(50.0, 50.0),
    )
    dist_home_wp = float((wp[0] - env.cfg.gcs_ne[0]) ** 2 + (wp[1] - env.cfg.gcs_ne[1]) ** 2) ** 0.5
    dist_home_pos = float((50.0 - env.cfg.gcs_ne[0]) ** 2 + (50.0 - env.cfg.gcs_ne[1]) ** 2) ** 0.5
    assert dist_home_wp < dist_home_pos
