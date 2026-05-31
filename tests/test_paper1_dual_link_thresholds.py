from __future__ import annotations

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds, dual_link_thresholds_from_comm
from uavlab.paper1.comm.thresholds import control_link_ok, data_path_feasible_at, data_return_ok
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.comm.thresholds import link_proxy_at as link_state_at
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config
from uavlab.related_models.comm_link_state import LinkState


def test_dual_link_defaults_match_doc():
    th = Paper1DualLinkThresholds()
    assert th.control_max_loss_p == 0.05
    assert th.control_max_delay_s == 1.0
    assert th.data_max_loss_p == 0.20
    assert th.data_max_return_time_s == 10.0
    assert th.data_weak_loss_p == 0.50
    rp = th.to_return_params()
    assert rp.max_loss_p_for_return == 0.20
    assert rp.max_return_time_s == 10.0


def test_dual_link_from_comm_yaml():
    th = dual_link_thresholds_from_comm(
        {
            "control_link": {"max_loss_p": 0.04},
            "data_link": {"max_loss_p": 0.18, "max_return_time_s": 8.0, "weak_loss_p": 0.45},
        }
    )
    assert th.control_max_loss_p == 0.04
    assert th.data_max_loss_p == 0.18
    assert th.data_max_return_time_s == 8.0
    assert th.data_weak_loss_p == 0.45


def test_control_vs_data_feasibility():
    th = Paper1DualLinkThresholds()
    ok_ctrl = LinkState(loss_p=0.04, delay_s=0.5, bandwidth_bps=1e6)
    weak_data = LinkState(loss_p=0.15, delay_s=0.5, bandwidth_bps=1e6)
    assert control_link_ok(ok_ctrl, th)
    assert data_return_ok(weak_data, key_bits=8000.0, th=th)
    assert not control_link_ok(weak_data, th)


def test_data_return_jitter_gate():
    th = Paper1DualLinkThresholds()
    ok = LinkState(loss_p=0.10, delay_s=0.5, bandwidth_bps=1e6, jitter_s=0.5)
    bad_jit = LinkState(loss_p=0.10, delay_s=0.5, bandwidth_bps=1e6, jitter_s=1.5)
    assert data_return_ok(ok, key_bits=8000.0, th=th, check_jitter=True)
    assert not data_return_ok(bad_jit, key_bits=8000.0, th=th, check_jitter=True)
    assert data_return_ok(bad_jit, key_bits=8000.0, th=th, check_jitter=False)


def test_structure_use_jitter_in_slow_comm():
    from uavlab.paper1.contracts.contract_config import Paper1ContractConfig

    base = apply_experiment_presets(load_resolved_config("configs/base.yaml"))
    for struct, expect_jit in (("cdsl", False), ("wcdl", True), ("fdlc", True)):
        cfg = dict(base)
        cfg.setdefault("experiment", {}).setdefault("paper1", {})["struct"] = struct
        cfg.setdefault("paper1_loops", {}).setdefault("semantics", {})["structure"] = struct
        c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
        assert c.use_jitter_in_slow_comm is expect_jit, struct
        assert c.structure == struct


def test_contract_applies_dual_link_to_fsm_and_slow_triggers():
    cfg = apply_experiment_presets(load_resolved_config("configs/base.yaml"))
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    thr = dict(c.fast_loop.get("fsm_thresholds") or {})
    sl = dict(c.slow_loop_triggers or {})
    assert thr["link_loss_recover"] == 0.20
    assert thr["link_loss_safe"] == 0.05
    assert sl["link_drop_loss_p"] == 0.50
    assert c.dual_link.data_max_loss_p == 0.20


def test_data_path_feasible_near_gcs():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/sanity/c1_g1_comm_aware_decision.yaml")
    )
    scene_path = str(cfg.get("scene_file"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    env.reset(seed=0)
    th = sim_cfg.dual_link
    gcs = (float(sim_cfg.gcs_ne[0]), float(sim_cfg.gcs_ne[1]))
    assert data_path_feasible_at(env, gcs, sim_cfg.key_bits_per_poi, th)


def test_link_proxy_at_matches_loss_proxy():
    cfg = apply_experiment_presets(load_resolved_config("configs/base.yaml"))
    scene_path = str(cfg.get("scene_file"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    env.reset(seed=0)
    p = (100.0, 100.0)
    from uavlab.paper1.comm.link_proxy import loss_proxy_at

    ls = link_state_at(env, p)
    assert abs(float(ls.loss_p) - loss_proxy_at(env, p)) < 1e-9
