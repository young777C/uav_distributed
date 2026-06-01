"""Tests for per-POI FCFS incremental return (ReturnQueue + Paper1Env.progress_key_return)."""

from __future__ import annotations

from uavlab.paper1.sim.return_queue import ReturnQueue
from uavlab.related_models.comm_link_state import LinkState
from uavlab.related_models.key_data_return import ReturnDecisionParams, effective_bandwidth_bps


def _good_link() -> LinkState:
    return LinkState(loss_p=0.05, delay_s=0.1, bandwidth_bps=1_000_000.0)


def test_single_poi_completes_over_multiple_steps() -> None:
    q = ReturnQueue()
    q.enqueue(poi_id=0, bits=8000.0, covered_time_s=0.0)
    params = ReturnDecisionParams(max_loss_p_for_return=0.2, max_return_time_s=60.0)
    link = _good_link()
    dt = 0.2
    done = False
    for step in range(500):
        now = step * dt
        r = q.progress_step(link=link, dt_s=dt, now_s=now, params=params)
        if r.completed_ids:
            done = True
            break
    assert done
    assert q.pending_bits == 0.0
    assert q.pending_count == 0


def test_multi_poi_fcfs_not_bulk_or_nothing() -> None:
    """Three POIs: partial progress on first before second completes (not bulk sum gate)."""
    q = ReturnQueue()
    chunk = 4_000_000.0
    for pid in (0, 1, 2):
        q.enqueue(poi_id=pid, bits=chunk, covered_time_s=float(pid))
    params = ReturnDecisionParams(max_loss_p_for_return=0.2, max_return_time_s=2000.0)
    link = _good_link()
    dt = 0.2
    budget_per_step = effective_bandwidth_bps(link) * dt
    assert budget_per_step > 0

    completed_order: list[int] = []
    for step in range(5000):
        now = step * dt
        r = q.progress_step(link=link, dt_s=dt, now_s=now, params=params)
        for pid in r.completed_ids:
            completed_order.append(int(pid))
        if len(completed_order) >= 3:
            break

    assert completed_order == [0, 1, 2]
    assert q.pending_bits == 0.0


def test_high_loss_no_transmit_no_false_complete() -> None:
    q = ReturnQueue()
    q.enqueue(poi_id=5, bits=1_000_000.0, covered_time_s=0.0)
    bad = LinkState(loss_p=0.9, delay_s=0.1, bandwidth_bps=1_000_000.0)
    params = ReturnDecisionParams(max_loss_p_for_return=0.2, max_return_time_s=100.0)
    r = q.progress_step(link=bad, dt_s=0.2, now_s=1.0, params=params)
    assert r.completed_ids == ()
    assert r.transmitted_bits == 0.0
    assert q.pending_bits == 1_000_000.0


def test_no_expire_before_first_tx_despite_long_wait() -> None:
    """Covered long ago but no transmission yet — must not expire (stall-only policy)."""
    q = ReturnQueue()
    q.enqueue(poi_id=7, bits=1_000_000.0, covered_time_s=0.0)
    params = ReturnDecisionParams(max_loss_p_for_return=0.2, max_return_time_s=1.0)
    bad = LinkState(loss_p=0.9, delay_s=0.1, bandwidth_bps=1_000_000.0)
    r = q.progress_step(link=bad, dt_s=0.2, now_s=100.0, params=params)
    assert r.expired_ids == ()
    assert q.pending_bits == 1_000_000.0


def test_stall_after_first_tx_marks_expired() -> None:
    q = ReturnQueue()
    q.enqueue(poi_id=7, bits=1_000_000.0, covered_time_s=0.0)
    params = ReturnDecisionParams(max_loss_p_for_return=0.2, max_return_time_s=1.0)
    link = _good_link()
    # First step: start TX but do not finish 1 Mbit.
    q.progress_step(link=link, dt_s=0.2, now_s=0.0, params=params)
    assert q._buffers[7].first_tx_time_s is not None
    # Stall > 1 s with no further progress (bad link → no TX this step).
    bad = LinkState(loss_p=0.9, delay_s=0.1, bandwidth_bps=1_000_000.0)
    r = q.progress_step(link=bad, dt_s=0.2, now_s=2.5, params=params)
    assert 7 in r.expired_ids
    assert 7 not in r.completed_ids
    assert q.pending_count == 0


def test_env_progress_key_return_integration() -> None:
    from uavlab.common.config import load_resolved_config
    from uavlab.experiments.presets import apply_experiment_presets
    from uavlab.paper1.comm.dual_link import dual_link_thresholds_from_comm
    from uavlab.paper1.sim.config import from_resolved_config
    from uavlab.paper1.sim.env import build_env
    from uavlab.paper1.sim.scene_loader import load_scene_yaml
    from uavlab.scene.loader import load_scene_config

    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/cases/c2_g2_m0.yaml")
    )
    cfg = {**cfg, "comm": {**(cfg.get("comm") or {}), "data_chunk_bits": 8000.0}}
    scene_raw = load_scene_yaml(cfg["scene_file"])
    sim = from_resolved_config(cfg, scene_raw)
    env = build_env(sim, load_scene_config(cfg["scene_file"]))
    th = dual_link_thresholds_from_comm(dict(cfg.get("comm") or {}))
    ret_params = th.to_return_params()
    dt = 1.0 / sim.step_hz

    env.reset(seed=0)
    env.return_queue.enqueue(poi_id=0, bits=8000.0, covered_time_s=0.0)
    env.covered.add(0)
    env._sync_backlog_from_queue()

    for _ in range(200):
        env.progress_key_return(dt_s=dt, params=ret_params)
        if 0 in env.returned:
            break
    assert 0 in env.returned
    assert 0 in env.effective
    assert env.backlog_bits == 0.0
