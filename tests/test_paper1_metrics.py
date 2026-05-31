from __future__ import annotations

import math

import pytest

from uavlab.paper1.metrics.link_recovery import LinkRecoveryRecorder
from uavlab.paper1.metrics.paper_metrics import (
    compute_mean_key_return_delay_s,
    compute_paper_episode_metrics,
)
from uavlab.paper1.types import FastCommState


def test_mean_key_return_delay_s():
    from uavlab.paper1.metrics.paper_metrics import compute_mean_key_return_delay_s

    t = compute_mean_key_return_delay_s(
        returned={0, 1},
        poi_cov_time_s={0: 10.0, 1: 20.0},
        poi_return_time_s={0: 15.0, 1: 30.0},
    )
    assert t == pytest.approx(7.5, rel=0, abs=1e-6)


def test_paper_metrics_T_ret_s():
    m = compute_paper_episode_metrics(
        n_pois=4,
        covered={0, 1, 2},
        returned={0, 1},
        poi_cov_time_s={0: 0.0, 1: 10.0, 2: 5.0},
        poi_return_time_s={0: 2.0, 1: 20.0},
        nofly_dwell_s=0.0,
    )
    assert m["T_ret_s"] == pytest.approx(6.0, rel=0, abs=1e-6)


def test_mean_key_return_delay_s_eq_61():
    m = compute_paper_episode_metrics(
        n_pois=4,
        covered={0, 1, 2},
        returned={0, 1},
        poi_cov_time_s={0: 10.0, 1: 20.0, 2: 5.0},
        poi_return_time_s={0: 15.0, 1: 28.0},
        nofly_dwell_s=0.0,
    )
    # (5 + 8) / (2 + eps) ≈ 6.5
    assert abs(m["T_ret_s"] - 6.5) < 1e-6
    assert m["mean_key_return_delay_s"] == m["T_ret_s"]


def test_mean_key_return_delay_nan_when_no_returns():
    m = compute_paper_episode_metrics(
        n_pois=4,
        covered={0, 1},
        returned=set(),
        poi_cov_time_s={0: 1.0, 1: 2.0},
        poi_return_time_s={},
        nofly_dwell_s=0.0,
    )
    assert math.isnan(m["T_ret_s"])


def test_paper_metrics_r_fail_given_cov():
    m = compute_paper_episode_metrics(
        n_pois=4,
        covered={0, 1, 2},
        returned={0, 1},
        nofly_dwell_s=1.5,
        link_recovery_latencies_s=[0.2, 0.4],
    )
    assert m["R_cov"] == 0.75
    assert m["R_task"] == 0.5
    assert abs(m["R_fail_given_cov"] - (1.0 / 3.0)) < 1e-9
    assert m["R_fail_cov"] == m["R_fail_given_cov"]
    assert math.isnan(m["T_ret_s"])
    assert m["T_nf_s"] == 1.5
    assert m["coverage_ratio"] == m["R_cov"]
    assert m["effective_ratio"] == m["R_task"]
    assert abs(m["R_task_check_product"] - m["R_task"]) < 1e-9


def test_mean_key_return_delay_s_eq_61():
    t_ret = compute_mean_key_return_delay_s(
        returned={0, 1},
        poi_cov_time_s={0: 10.0, 1: 20.0, 2: 5.0},
        poi_return_time_s={0: 15.0, 1: 28.0},
    )
    # (5 + 8) / (2 + eps) ≈ 6.5
    assert abs(t_ret - 6.5) < 1e-6

    m = compute_paper_episode_metrics(
        n_pois=4,
        covered={0, 1, 2},
        returned={0, 1},
        poi_cov_time_s={0: 10.0, 1: 20.0},
        poi_return_time_s={0: 15.0, 1: 28.0},
        nofly_dwell_s=0.0,
    )
    assert abs(m["T_ret_s"] - 6.5) < 1e-6

    empty = compute_paper_episode_metrics(
        n_pois=4,
        covered={0},
        returned=set(),
        poi_cov_time_s={0: 1.0},
        poi_return_time_s={},
        nofly_dwell_s=0.0,
    )
    assert math.isnan(empty["T_ret_s"])


def test_link_recovery_recorder_latency():
    rec = LinkRecoveryRecorder(
        link_drop_loss_p=0.8,
        link_loss_recover=0.5,
        dt=0.1,
        stable_steps_required=2,
    )
    rec.observe(step=0, loss_p=0.7, comm_mode=FastCommState.INS)
    rec.observe(step=1, loss_p=0.85, comm_mode=FastCommState.INS)  # degrade edge
    rec.observe(step=2, loss_p=0.9, comm_mode=FastCommState.REC)  # fast response
    rec.observe(step=3, loss_p=0.55, comm_mode=FastCommState.INS)
    rec.observe(step=4, loss_p=0.4, comm_mode=FastCommState.INS)
    rec.observe(step=5, loss_p=0.3, comm_mode=FastCommState.INS)  # stable x2 → close
    assert rec.latencies_s() == [0.4]


def test_env_returned_subset_of_covered():
    from uavlab.common.config import load_resolved_config
    from uavlab.experiments.presets import apply_experiment_presets
    from uavlab.paper1.sim.config import from_resolved_config
    from uavlab.paper1.sim.env import build_env
    from uavlab.paper1.sim.scene_loader import load_scene_yaml
    from uavlab.scene.loader import load_scene_config

    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/sanity/c1_g1_comm_aware_decision.yaml")
    )
    scene_path = str(cfg.get("scene_file", "configs/scenes/g1_uniform.yaml"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    _ = env.reset(seed=0)
    env.covered = {0, 1}
    env.returned = {0}
    env.poi_cov_time_s = {0: 1.0, 1: 5.0}
    env.poi_return_time_s = {0: 3.0}
    env._sync_effective_from_returned()
    m = env.metrics()
    assert m["T_ret_s"] == pytest.approx(2.0, rel=0, abs=1e-6)
    assert m["returned_count"] == 1
    assert m["failed_after_cov_count"] == 1
    assert abs(m["R_fail_given_cov"] - 0.5) < 1e-6
