"""P1 / P1.5 backlog-aware return policy (struct axis)."""

from __future__ import annotations

from uavlab.paper1.comm.dual_link import dual_link_thresholds_from_comm
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import (
    CompletionStatus,
    FastObservation,
    FastToSlowPacket,
    SlowObservation,
)
from uavlab.paper1.contracts.return_policy import ReturnPolicyConfig, return_policy_from_cfg
from uavlab.paper1.loops.fast.fsm import expected_tx_gain_bits, select_comm_mode
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.event_classifier import classify_slow_events
from uavlab.paper1.loops.slow.slow_loop import SlowLoop
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.sim.return_queue import ReturnQueue
from uavlab.paper1.types import FastCommState


def test_return_policy_from_cfg_fdlc_defaults():
    rp = return_policy_from_cfg(
        {"comm": {"data_chunk_bits": 4_000_000}, "paper1_loops": {}},
        struct_key="fdlc",
    )
    assert rp.enable_backlog_gates is True
    assert rp.enable_upload_stuck_recovery is True
    assert rp.backlog_soft_bits == 8_000_000.0
    assert rp.backlog_hard_bits == 20_000_000.0
    assert rp.backlog_soft_poi_count == 2
    assert rp.backlog_hard_poi_count == 5
    assert rp.upload_stuck_s == 30.0
    assert rp.upload_stuck_min_pending_count == 2
    assert rp.upload_stuck_requires_soft_backlog is True


def test_return_policy_cdsl_no_gates():
    rp = return_policy_from_cfg(
        {"comm": {"data_chunk_bits": 4_000_000}, "paper1_loops": {}},
        struct_key="cdsl",
    )
    assert rp.enable_backlog_gates is False
    assert rp.mission_phase(99_000_000.0, pending_count=99) == "explore"


def test_mission_phase_thresholds():
    rp = ReturnPolicyConfig(
        backlog_soft_bits=10.0,
        backlog_hard_bits=20.0,
        min_tx_gain_bits=1.0,
        enable_backlog_gates=True,
        backlog_soft_poi_count=2,
        backlog_hard_poi_count=4,
    )
    assert rp.mission_phase(5.0, pending_count=1) == "explore"
    assert rp.mission_phase(5.0, pending_count=2) == "balance"
    assert rp.mission_phase(15.0, pending_count=2) == "balance"
    assert rp.mission_phase(5.0, pending_count=4) == "return"
    assert rp.mission_phase(25.0, pending_count=1) == "return"


def test_fsm_return_phase_with_backlog_uses_tx_not_ins():
    class _Env:
        comm_mode = FastCommState.INS
        covered: set[int] = set()
        remaining_energy = 1.0
        pos_ne = (0.0, 0.0)
        cfg = type("C", (), {"gcs_ne": (0.0, 0.0)})()

        def in_nofly(self, _pos):
            return False

    obs = FastObservation(
        step=1,
        pos_ne=(0.0, 0.0),
        comm_mode=FastCommState.INS,
        backlog_bits=4_000_000.0,
        link_loss_p=0.05,
    )
    rp = ReturnPolicyConfig(
        backlog_soft_bits=8e6,
        backlog_hard_bits=20e6,
        min_tx_gain_bits=0.0,
        enable_backlog_gates=True,
    )
    dual = dual_link_thresholds_from_comm({})
    mode = select_comm_mode(
        env=_Env(),
        obs=obs,
        params=FastLoopParams(enable_recovery_mode=True),
        goal_id=None,
        use_comm_in_fast=True,
        use_energy_in_fast=False,
        mode_switching_allowed=True,
        dt_s=1.0,
        return_policy=rp,
        dual_link=dual,
        link_bandwidth_bps=2_000_000.0,
        link_delay_s=0.05,
        return_phase=True,
    )
    assert mode == FastCommState.TX


def _slow_env_with_queue(*, stall_s: float, elapsed_s: float, goal_id: int = 5):
    q = ReturnQueue()
    q.enqueue(poi_id=goal_id, bits=1_000_000.0, covered_time_s=0.0)
    hz = 10

    class _Env:
        backlog_bits = 1_000_000.0
        covered = {goal_id}
        effective = set()
        returned = set()
        pois = {goal_id: type("P", (), {"pos_ne": (1.0, 1.0)})()}
        pos_ne = (0.0, 0.0)
        return_queue = q
        cfg = type(
            "C",
            (),
            {
                "gcs_ne": (0.0, 0.0),
                "step_hz": hz,
                "episode_steps": 1000,
                "return_reserve_s": 0.0,
                "energy_per_meter": 1.0,
                "energy_safe_margin": 0.0,
            },
        )()
        remaining_energy = 999.0
        t = int(round(elapsed_s * hz))

    return _Env()


def test_upload_stuck_recovery_waits_for_stall():
    from uavlab.common.config import load_config

    cfg = load_config("configs/base.yaml")
    pl = dict(cfg["paper1_loops"])
    pl["return_policy"] = {
        "enable_backlog_gates": True,
        "enable_upload_stuck_recovery": True,
        "upload_stuck_s": 30.0,
        "upload_stuck_min_pending_count": 1,
        "upload_stuck_requires_soft_backlog": False,
    }
    pl["semantics"] = {**dict(pl.get("semantics") or {}), "structure": "fdlc"}
    cfg = dict(cfg)
    cfg["paper1_loops"] = pl
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)

    env = _slow_env_with_queue(stall_s=30.0, elapsed_s=5.0)
    slow = SlowLoop(
        env=env,
        params=SlowLoopParams(lambda_q=1.0, mu_dist=1.0, beta_ret=1.0, dual_link=contract.dual_link),
        contract=contract,
        slow_interval_steps=9999,
    )
    slow._goal_id = 5
    pkt = FastToSlowPacket(step=env.t, goal_id=5, goal_ne=(1.0, 1.0))
    plan = slow.step(SlowObservation(step=env.t, fast_to_slow=pkt))
    assert plan.goal_id == 5


def test_upload_stuck_recovery_blocked_with_single_pending():
    from uavlab.common.config import load_config

    cfg = load_config("configs/base.yaml")
    pl = dict(cfg["paper1_loops"])
    pl["return_policy"] = {
        "enable_backlog_gates": True,
        "enable_upload_stuck_recovery": True,
        "upload_stuck_s": 5.0,
        "upload_stuck_min_pending_count": 2,
        "upload_stuck_requires_soft_backlog": True,
        "backlog_soft_bits": 2_000_000.0,
    }
    pl["semantics"] = {**dict(pl.get("semantics") or {}), "structure": "fdlc"}
    cfg = dict(cfg)
    cfg["paper1_loops"] = pl
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)

    env = _slow_env_with_queue(stall_s=5.0, elapsed_s=10.0)
    slow = SlowLoop(
        env=env,
        params=SlowLoopParams(lambda_q=1.0, mu_dist=1.0, beta_ret=1.0, dual_link=contract.dual_link),
        contract=contract,
        slow_interval_steps=9999,
    )
    slow._goal_id = 5
    pkt = FastToSlowPacket(step=env.t, goal_id=5, goal_ne=(1.0, 1.0))
    plan = slow.step(SlowObservation(step=env.t, fast_to_slow=pkt))
    assert plan.goal_id == 5


def test_return_policy_upload_stuck_s_scales_with_chunk():
    rp1 = return_policy_from_cfg({"comm": {"data_chunk_bits": 1_000_000}, "paper1_loops": {}}, struct_key="fdlc")
    rp4 = return_policy_from_cfg({"comm": {"data_chunk_bits": 4_000_000}, "paper1_loops": {}}, struct_key="fdlc")
    assert rp1.upload_stuck_s == 60.0
    assert rp4.upload_stuck_s == 30.0


def test_upload_stuck_recovery_after_stall_clears_goal():
    from uavlab.common.config import load_config

    cfg = load_config("configs/base.yaml")
    pl = dict(cfg["paper1_loops"])
    pl["return_policy"] = {
        "enable_backlog_gates": True,
        "enable_upload_stuck_recovery": True,
        "upload_stuck_s": 30.0,
        "upload_stuck_min_pending_count": 1,
        "upload_stuck_requires_soft_backlog": False,
    }
    pl["semantics"] = {**dict(pl.get("semantics") or {}), "structure": "fdlc"}
    cfg = dict(cfg)
    cfg["paper1_loops"] = pl
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)

    env = _slow_env_with_queue(stall_s=30.0, elapsed_s=31.0)
    slow = SlowLoop(
        env=env,
        params=SlowLoopParams(lambda_q=1.0, mu_dist=1.0, beta_ret=1.0, dual_link=contract.dual_link),
        contract=contract,
        slow_interval_steps=9999,
    )
    slow._goal_id = 5
    pkt = FastToSlowPacket(step=env.t, goal_id=5, goal_ne=(1.0, 1.0))
    plan = slow.step(SlowObservation(step=env.t, fast_to_slow=pkt))
    assert plan.goal_id is None
    classified = classify_slow_events(
        contract=contract,
        obs=SlowObservation(step=env.t, fast_to_slow=pkt),
        ev_flags=dict(contract.slow_loop_triggers.get("event_triggers") or {}),
        post_cover_upload_stuck=False,
        goal_edge=False,
    )
    assert classified.has_warning is False


def test_return_queue_poi_stall_before_first_tx():
    q = ReturnQueue()
    q.enqueue(poi_id=1, bits=1000.0, covered_time_s=0.0)
    assert not q.is_poi_return_stalled(1, now_s=10.0, stall_s=30.0)
    assert q.is_poi_return_stalled(1, now_s=31.0, stall_s=30.0)


def test_fsm_balance_phase_transit_uses_tx_with_backlog():
    class _Env:
        comm_mode = FastCommState.INS
        covered: set[int] = set()
        remaining_energy = 1.0
        pos_ne = (0.0, 0.0)
        cfg = type("C", (), {"gcs_ne": (0.0, 0.0)})()
        return_queue = type("Q", (), {"pending_count": 2})()

        def in_nofly(self, _pos):
            return False

    obs = FastObservation(
        step=1,
        pos_ne=(0.0, 0.0),
        comm_mode=FastCommState.INS,
        backlog_bits=8_000_000.0,
        link_loss_p=0.05,
    )
    rp = ReturnPolicyConfig(
        backlog_soft_bits=8e6,
        backlog_hard_bits=20e6,
        min_tx_gain_bits=0.0,
        enable_backlog_gates=True,
        backlog_soft_poi_count=2,
    )
    dual = dual_link_thresholds_from_comm({})
    mode = select_comm_mode(
        env=_Env(),
        obs=obs,
        params=FastLoopParams(enable_recovery_mode=True),
        goal_id=99,
        use_comm_in_fast=True,
        use_energy_in_fast=False,
        mode_switching_allowed=True,
        dt_s=1.0,
        return_policy=rp,
        dual_link=dual,
        link_bandwidth_bps=2_000_000.0,
        link_delay_s=0.05,
    )
    assert mode == FastCommState.TX


def test_expected_tx_gain_zero_on_high_loss():
    dual = dual_link_thresholds_from_comm({"data_max_loss_p": 0.2})
    gain = expected_tx_gain_bits(
        link_loss_p=0.9,
        link_bandwidth_bps=1e6,
        link_delay_s=0.05,
        dt_s=1.0,
        dual_link=dual,
    )
    assert gain == 0.0
