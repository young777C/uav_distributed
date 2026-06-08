from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastObservation, SlowObservation, SlowPlan
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper1.loops.fast.fast_loop import FastLoop
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config
from uavlab.paper1.metrics.link_recovery import link_recovery_recorder_from_contract
from uavlab.paper1.types import FastCommState
from uavlab.paper1.runner.return_scheduling import goal_spatial_complete, should_attempt_key_return
from uavlab.paper1.runner.run_context import format_paper1_episode_line, print_paper1_run_header
from uavlab.related_models.key_data_return import ReturnDecisionParams


def main() -> None:
    p = argparse.ArgumentParser(description="Paper1 lightweight runner (task-level).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slow_interval_steps", type=int, default=40)
    # For compatibility with scripts/sweep.py across runners; optional for Paper1Lite.
    p.add_argument("--comm_mode", type=str, default="")
    p.add_argument("--run_dir", type=str, default="")
    p.add_argument("--metrics_jsonl", type=str, default="")
    args = p.parse_args()

    cfg = apply_experiment_presets(load_resolved_config(args.config))
    scene_path = str(cfg.get("scene_file", "configs/scene/default.yaml"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)

    dt = 1.0 / float(max(1, sim_cfg.step_hz))
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))
    fast_params = FastLoopParams.from_contract(contract)
    dual_link = contract.dual_link
    ret_params = dual_link.to_return_params()
    slow_params = SlowLoopParams(
        t_obs_s=float(sim_cfg.poi_dwell_s),
        t_safe_s=0.0,
        dual_link=dual_link,
    )
    slow = SlowLoop(env=env, params=slow_params, contract=contract, slow_interval_steps=int(args.slow_interval_steps))
    fast = FastLoop(env=env, params=fast_params, contract=contract)
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    link_rec = link_recovery_recorder_from_contract(contract, dt=float(dt))

    run_dir: Optional[Path] = None
    if args.run_dir.strip():
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    metrics_f = None
    if args.metrics_jsonl.strip():
        mp = Path(args.metrics_jsonl)
        mp.parent.mkdir(parents=True, exist_ok=True)
        metrics_f = mp.open("w", encoding="utf-8")

    exp = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else {}
    exp_id = exp.get("id")
    p1 = exp.get("paper1") if isinstance(exp.get("paper1"), dict) else {}
    env_case = exp.get("env_case") if isinstance(exp.get("env_case"), dict) else {}

    run_ctx = print_paper1_run_header(
        cfg,
        contract=contract,
        experiment_id=exp_id,
        slow_interval_steps=int(args.slow_interval_steps),
    )

    for ep in range(int(args.episodes)):
        _ = env.reset(seed=int(args.seed) + ep)
        slow.reset()
        fast.reset()
        link_rec.reset()
        if args.comm_mode.strip():
            try:
                env.comm_mode = FastCommState(str(args.comm_mode))
            except Exception:
                env.comm_mode = str(args.comm_mode)

        # Main-loop order (parallel-friendly):
        # 1) Bootstrap slow once at env.t==0 so the first fast step has a real goal.
        # 2) Each iteration: fast uses the plan from the *end of the previous* iteration, then physics
        #    and progress_key_return, then build a fast→slow packet from the *post-step* env so
        #    CompletionStatus.spatial_complete reflects dwell/coverage updated by step_fast.
        exec_plan = SlowPlan(goal_id=None, goal_ne=(float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])))
        link0 = env.observe_link_state()
        pkt0 = coupling.build_fast_to_slow_packet(
            contract=contract,
            env=env,
            step=int(env.t),
            link_loss_p=float(link0.loss_p),
            link_delay_s=float(link0.delay_s),
            link_bandwidth_bps=float(link0.bandwidth_bps),
            control_link_lost=fast.control_link_lost,
            control_link_lost_duration_s=fast.control_link_lost_duration_s,
            plan=exec_plan,
        )
        exec_plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=pkt0))

        while not env.done():
            link = env.observe_link_state()
            cmd = fast.step(
                FastObservation(
                    step=int(env.t),
                    pos_ne=env.pos_ne,
                    comm_mode=env.comm_mode,
                    backlog_bits=float(env.backlog_bits),
                    link_loss_p=float(link.loss_p),
                ),
                exec_plan,
                dt=float(dt),
            )
            env.comm_mode = cmd.next_comm_mode
            env.step_fast(
                target_ne=cmd.target_ne,
                vel_ne_cmd=cmd.vel_ne_cmd,
                dt=float(dt),
                approach_goal_ne=cmd.approach_goal_ne,
            )
            link_rec.observe(step=int(env.t), loss_p=float(link.loss_p), comm_mode=env.comm_mode)

            if should_attempt_key_return(
                backlog_bits=float(env.backlog_bits),
                comm_mode=env.comm_mode,
                enable_fast_mode_switch=bool(contract.enable_fast_mode_switch),
                spatial_complete=goal_spatial_complete(env=env, goal_id=exec_plan.goal_id),
                return_phase=exec_plan.goal_id is None,
                fast_upload_mode=str(env.cfg.fast_upload_mode),
                fixed_send_ratio=float(env.cfg.fixed_send_ratio),
                step=int(env.t),
            ):
                env.progress_key_return(dt_s=float(dt), params=ret_params)

            pkt = coupling.build_fast_to_slow_packet(
                contract=contract,
                env=env,
                step=int(env.t),
                link_loss_p=float(link.loss_p),
                link_delay_s=float(link.delay_s),
                link_bandwidth_bps=float(link.bandwidth_bps),
                control_link_lost=fast.control_link_lost,
                control_link_lost_duration_s=fast.control_link_lost_duration_s,
                plan=exec_plan,
            )
            exec_plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=pkt))

        row: Dict[str, Any] = {
            "experiment_id": exp_id,
            "paper1": p1,
            "env_case": env_case,
            "episode": ep,
            **env.metrics(link_recovery_latencies_s=link_rec.latencies_s()),
        }
        print(format_paper1_episode_line(ctx=run_ctx, episode=ep, metrics=row), flush=True)
        if metrics_f is not None:
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_f.flush()

    if metrics_f is not None:
        metrics_f.close()


if __name__ == "__main__":
    main()

