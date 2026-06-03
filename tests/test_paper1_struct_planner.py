from __future__ import annotations

from uavlab.common.config import _deep_merge, load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.struct_profile import struct_axis_profile_defaults
from uavlab.paper1.loops.fast.fsm import select_comm_mode
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import (
    SlowLoopParams,
    build_candidate_window,
)
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.paper1.types import FastCommState
from uavlab.scene.loader import load_scene_config


def _build_c2_env(seed: int = 0):
    base = load_resolved_config("configs/base.yaml")
    case = load_resolved_config("configs/experiments/paper1/cases/c2_g2_m1.yaml")
    cfg = apply_experiment_presets(_deep_merge(base, case))
    scene_path = str(cfg.get("scene_file", "configs/scenes/g2_cluster_m1.yaml"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    env.reset(seed=seed)
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))
    return env, contract


def _feasible_poi_count(env, contract: Paper1ContractConfig) -> int:
    prof = contract.struct_profile
    params = SlowLoopParams(
        dual_link=contract.dual_link,
        energy_plan_margin_frac=float(prof.energy_plan_margin_frac),
        beta_ret=float(prof.comm_ret_objective_weight),
    )
    win = build_candidate_window(
        env=env,
        params=params,
        profile=prof,
        window_k=50,
        prefetch_k_multiplier=3,
        comm_objective=bool(contract.comm_objective),
        comm_path_quality_mask=False,
        apply_energy_per_node_mask=bool(
            contract.energy_hard_constraint or contract.energy_budget_constraint
        ),
        path_samples=9,
        degrade_level=0,
    )
    return sum(1 for i in range(len(win.poi_ids)) if win.m_i[i + 1])


def test_struct_profile_defaults_monotonicity():
    cdsl = struct_axis_profile_defaults("cdsl")
    wcdl = struct_axis_profile_defaults("wcdl")
    fdlc = struct_axis_profile_defaults("fdlc")
    assert cdsl.control_cmd_min_ratio >= wcdl.control_cmd_min_ratio >= fdlc.control_cmd_min_ratio
    assert cdsl.control_loss_exposure_max <= wcdl.control_loss_exposure_max <= fdlc.control_loss_exposure_max
    assert cdsl.energy_plan_margin_frac >= wcdl.energy_plan_margin_frac >= fdlc.energy_plan_margin_frac
    assert cdsl.candidate_degrade_max_level <= wcdl.candidate_degrade_max_level <= fdlc.candidate_degrade_max_level


def test_feasible_set_monotonicity_c2():
    env, _ = _build_c2_env(seed=0)
    n_cdsl = _feasible_poi_count(
        env,
        Paper1ContractConfig.from_cfg(
            {"paper1_loops": {"semantics": {"structure": "cdsl"}}},
            waypoint_delta_max_m=5.0,
        ),
    )
    n_wcdl = _feasible_poi_count(
        env,
        Paper1ContractConfig.from_cfg(
            {"paper1_loops": {"semantics": {"structure": "wcdl"}}},
            waypoint_delta_max_m=5.0,
        ),
    )
    n_fdlc = _feasible_poi_count(
        env,
        Paper1ContractConfig.from_cfg(
            {"paper1_loops": {"semantics": {"structure": "fdlc"}}},
            waypoint_delta_max_m=5.0,
        ),
    )
    assert n_cdsl <= n_wcdl <= n_fdlc


def test_fsm_transit_weak_link_stays_ins():
    class _Env:
        comm_mode = FastCommState.INS
        covered = set()
        remaining_energy = 1.0
        pos_ne = (0.0, 0.0)
        cfg = type("C", (), {"gcs_ne": (0.0, 0.0)})()

        def in_nofly(self, _pos):
            return False

    obs = type(
        "O",
        (),
        {"comm_mode": FastCommState.INS, "backlog_bits": 1000.0, "link_loss_p": 0.35},
    )()
    params = FastLoopParams(enable_recovery_mode=True, enable_safety_mode=False)
    contract = Paper1ContractConfig.from_cfg(
        {"comm": {"data_chunk_bits": 1_000_000}, "paper1_loops": {"semantics": {"structure": "cdsl"}}},
        waypoint_delta_max_m=5.0,
    )
    mode = select_comm_mode(
        env=_Env(),
        obs=obs,
        params=params,
        goal_id=3,
        use_comm_in_fast=True,
        use_energy_in_fast=False,
        mode_switching_allowed=True,
        dt_s=1.0,
        return_policy=contract.return_policy,
        dual_link=contract.dual_link,
        link_bandwidth_bps=1_000_000.0,
        link_delay_s=0.05,
    )
    assert mode == FastCommState.INS


def test_fsm_post_cover_uses_rec_on_weak_link():
    class _Env:
        comm_mode = FastCommState.INS
        covered = {3}
        remaining_energy = 1.0
        pos_ne = (0.0, 0.0)
        cfg = type("C", (), {"gcs_ne": (0.0, 0.0)})()

        def in_nofly(self, _pos):
            return False

    obs = type(
        "O",
        (),
        {"comm_mode": FastCommState.INS, "backlog_bits": 1000.0, "link_loss_p": 0.35},
    )()
    params = FastLoopParams(enable_recovery_mode=True, enable_safety_mode=False, link_loss_recover=0.20)
    contract = Paper1ContractConfig.from_cfg(
        {"comm": {"data_chunk_bits": 1_000_000}, "paper1_loops": {"semantics": {"structure": "cdsl"}}},
        waypoint_delta_max_m=5.0,
    )
    mode = select_comm_mode(
        env=_Env(),
        obs=obs,
        params=params,
        goal_id=3,
        use_comm_in_fast=True,
        use_energy_in_fast=False,
        mode_switching_allowed=True,
        dt_s=1.0,
        return_policy=contract.return_policy,
        dual_link=contract.dual_link,
        link_bandwidth_bps=1_000_000.0,
        link_delay_s=0.05,
    )
    assert mode == FastCommState.REC
