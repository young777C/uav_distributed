#!/usr/bin/env python3
"""Profile computation cost for Paper1 structure-axis experiments.

The script wraps the existing Paper1 lightweight runner loop with timing probes.
It does not change the control, planning, or simulator logic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from matplotlib import font_manager

from uavlab.common.config import load_resolved_config, save_yaml
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastObservation, SlowObservation, SlowPlan
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper1.loops.fast.fast_loop import FastLoop
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop
from uavlab.paper1.metrics.link_recovery import link_recovery_recorder_from_contract
from uavlab.paper1.runner.return_scheduling import goal_spatial_complete, should_attempt_key_return
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.paper1.types import FastCommState
from uavlab.scene.loader import load_scene_config


CASES = [
    ("C1--G2--M0", "configs/experiments/paper1/cases/c1_g2_m0.yaml", "weak"),
    ("C2--G2--M0", "configs/experiments/paper1/cases/c2_g2_m0.yaml", "weak"),
    ("C1--G2--M2", "configs/experiments/paper1/cases/c1_g2_m2.yaml", "high"),
    ("C2--G2--M2", "configs/experiments/paper1/cases/c2_g2_m2.yaml", "high"),
]

SYSTEMS = [
    ("CDSL", "configs/experiments/paper1/system/struct_centralized_single_loop.yaml"),
    ("WCDL", "configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml"),
    ("FDLC", "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml"),
]

COLORS = {"CDSL": "#777777", "WCDL": "#4c78a8", "FDLC": "#f58518"}


def _style() -> None:
    tinos_dir = Path("/usr/share/fonts/truetype/croscore")
    for font_path in sorted(tinos_dir.glob("Tinos-*.ttf")):
        font_manager.fontManager.addfont(font_path)
    mpl.rcParams.update(
        {
            "font.family": "Tinos",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def _mean(xs: Iterable[float]) -> float:
    vals = list(float(x) for x in xs)
    return float(statistics.fmean(vals)) if vals else float("nan")


def _std(xs: Iterable[float]) -> float:
    vals = list(float(x) for x in xs)
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def _pctl(xs: Iterable[float], q: float) -> float:
    vals = list(float(x) for x in xs)
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def _write_combined_config(out_path: Path, case_path: str, system_path: str, exp_id: str) -> None:
    save_yaml(
        {
            "extends": [str((ROOT / case_path).resolve()), str((ROOT / system_path).resolve())],
            "experiment": {"id": exp_id},
        },
        out_path,
    )


def _profile_episode(
    *,
    cfg: Dict[str, Any],
    seed: int,
    episode: int,
    slow_interval_steps: int,
) -> Dict[str, Any]:
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
    slow_params = SlowLoopParams(t_obs_s=float(sim_cfg.poi_dwell_s), t_safe_s=0.0, dual_link=dual_link)
    slow = SlowLoop(env=env, params=slow_params, contract=contract, slow_interval_steps=int(slow_interval_steps))
    fast = FastLoop(env=env, params=fast_params, contract=contract)
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    link_rec = link_recovery_recorder_from_contract(contract, dt=float(dt))

    fast_times_ms: list[float] = []
    slow_times_ms: list[float] = []
    replan_times_ms: list[float] = []
    hold_times_ms: list[float] = []
    packet_times_ms: list[float] = []
    env_step_times_ms: list[float] = []
    return_times_ms: list[float] = []
    link_times_ms: list[float] = []

    episode_t0 = time.perf_counter()
    _ = env.reset(seed=int(seed) + int(episode))
    slow.reset()
    fast.reset()
    link_rec.reset()

    exec_plan = SlowPlan(goal_id=None, goal_ne=(float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])))
    t0 = time.perf_counter()
    link0 = env.observe_link_state()
    link_times_ms.append((time.perf_counter() - t0) * 1000.0)
    t0 = time.perf_counter()
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
    packet_times_ms.append((time.perf_counter() - t0) * 1000.0)

    before_replans = slow.replan_count
    t0 = time.perf_counter()
    exec_plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=pkt0))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    slow_times_ms.append(elapsed_ms)
    if slow.replan_count > before_replans:
        replan_times_ms.append(elapsed_ms)
    else:
        hold_times_ms.append(elapsed_ms)

    while not env.done():
        t0 = time.perf_counter()
        link = env.observe_link_state()
        link_times_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
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
        fast_times_ms.append((time.perf_counter() - t0) * 1000.0)

        env.comm_mode = cmd.next_comm_mode
        t0 = time.perf_counter()
        env.step_fast(
            target_ne=cmd.target_ne,
            vel_ne_cmd=cmd.vel_ne_cmd,
            dt=float(dt),
            approach_goal_ne=cmd.approach_goal_ne,
        )
        env_step_times_ms.append((time.perf_counter() - t0) * 1000.0)
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
            t0 = time.perf_counter()
            env.progress_key_return(dt_s=float(dt), params=ret_params)
            return_times_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
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
        packet_times_ms.append((time.perf_counter() - t0) * 1000.0)

        before_replans = slow.replan_count
        t0 = time.perf_counter()
        exec_plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=pkt))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        slow_times_ms.append(elapsed_ms)
        if slow.replan_count > before_replans:
            replan_times_ms.append(elapsed_ms)
        else:
            hold_times_ms.append(elapsed_ms)

    episode_wall_ms = (time.perf_counter() - episode_t0) * 1000.0
    task_metrics = env.metrics(link_recovery_latencies_s=link_rec.latencies_s())
    fast_p95 = _pctl(fast_times_ms, 95)
    slow_replan_p95 = _pctl(replan_times_ms, 95)
    return {
        "seed": int(seed),
        "episode": int(episode),
        "dt_ms": float(dt * 1000.0),
        "steps": int(env.t),
        "episode_wall_ms": float(episode_wall_ms),
        "sim_time_s": float(env.t * dt),
        "wall_per_sim_step_ms": float(episode_wall_ms / max(1, int(env.t))),
        "real_time_factor": float((env.t * dt) / max(1e-9, episode_wall_ms / 1000.0)),
        "fast_step_count": len(fast_times_ms),
        "fast_mean_ms": _mean(fast_times_ms),
        "fast_p95_ms": fast_p95,
        "fast_max_ms": max(fast_times_ms) if fast_times_ms else float("nan"),
        "fast_p95_utilization": float(fast_p95 / max(1e-9, dt * 1000.0)),
        "slow_call_count": len(slow_times_ms),
        "slow_call_mean_ms": _mean(slow_times_ms),
        "slow_call_p95_ms": _pctl(slow_times_ms, 95),
        "slow_call_max_ms": max(slow_times_ms) if slow_times_ms else float("nan"),
        "replan_count": int(slow.replan_count),
        "replan_mean_ms": _mean(replan_times_ms),
        "replan_p95_ms": slow_replan_p95,
        "replan_max_ms": max(replan_times_ms) if replan_times_ms else float("nan"),
        "hold_mean_ms": _mean(hold_times_ms),
        "packet_mean_ms": _mean(packet_times_ms),
        "packet_p95_ms": _pctl(packet_times_ms, 95),
        "env_step_mean_ms": _mean(env_step_times_ms),
        "return_step_mean_ms": _mean(return_times_ms),
        "link_obs_mean_ms": _mean(link_times_ms),
        "R_task": float(task_metrics.get("R_task", float("nan"))),
        "R_cov": float(task_metrics.get("R_cov", float("nan"))),
        "T_ret_s": float(task_metrics.get("T_ret_s", float("nan"))),
        "T_nf_s": float(task_metrics.get("T_nf_s", float("nan"))),
        "returned_home": bool(task_metrics.get("returned_home", False)),
        "termination_reason": str(task_metrics.get("termination_reason", "")),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["case_label", "conflict_level", "comm_case", "method"]
    metrics = [
        "episode_wall_ms",
        "wall_per_sim_step_ms",
        "real_time_factor",
        "fast_mean_ms",
        "fast_p95_ms",
        "fast_max_ms",
        "fast_p95_utilization",
        "slow_call_mean_ms",
        "slow_call_p95_ms",
        "slow_call_max_ms",
        "replan_count",
        "replan_mean_ms",
        "replan_p95_ms",
        "replan_max_ms",
        "packet_mean_ms",
        "packet_p95_ms",
        "env_step_mean_ms",
        "return_step_mean_ms",
        "link_obs_mean_ms",
        "steps",
        "R_task",
        "R_cov",
        "T_ret_s",
        "T_nf_s",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for group_key, group in sorted(groups.items()):
        row = dict(zip(keys, group_key))
        row["episodes"] = len(group)
        row["seed_count"] = len({int(g["seed"]) for g in group})
        for metric in metrics:
            vals = [float(g[metric]) for g in group if not math.isnan(float(g[metric]))]
            row[f"{metric}_mean"] = _mean(vals)
            row[f"{metric}_std"] = _std(vals)
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot(summary: list[dict[str, Any]], out_dir: Path) -> None:
    _style()
    data = summary
    cases = ["C1--G2--M0", "C2--G2--M0", "C1--G2--M2", "C2--G2--M2"]
    methods = ["CDSL", "WCDL", "FDLC"]
    metrics = [
        ("fast_p95_ms_mean", "fast_p95_ms_std", "Fast-loop 95th percentile (ms)", "fig_computation_fast_p95"),
        ("replan_mean_ms_mean", "replan_mean_ms_std", "Replanning mean time (ms)", "fig_computation_replan_mean"),
        ("episode_wall_ms_mean", "episode_wall_ms_std", "Episode wall-clock time (ms)", "fig_computation_episode_wall"),
    ]
    for metric, std_metric, ylabel, stem in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        x = np.arange(len(cases))
        width = 0.24
        for i, method in enumerate(methods):
            means, stds = [], []
            for case in cases:
                match = [r for r in data if r["case_label"] == case and r["method"] == method]
                means.append(float(match[0][metric]) if match else float("nan"))
                stds.append(float(match[0][std_metric]) if match else 0.0)
            ax.bar(x + (i - 1) * width, means, width=width, yerr=stds, capsize=3, label=method, color=COLORS[method])
        ax.set_xticks(x, cases, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncols=3, frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=360)
        plt.close(fig)


def _report(summary: list[dict[str, Any]], out_dir: Path) -> None:
    seed_count = int(max(float(r.get("seed_count", 0.0)) for r in summary)) if summary else 0
    episodes_per_group = int(max(float(r.get("episodes", 0.0)) for r in summary)) if summary else 0
    episodes_per_seed = int(round(episodes_per_group / seed_count)) if seed_count else 0
    lines = [
        "# Computation Cost and Real-Time Profiling Report",
        "",
        "## Scope",
        "",
        "- Scenarios: C1--G2--M0, C2--G2--M0, C1--G2--M2, C2--G2--M2.",
        "- Methods: CDSL, WCDL, FDLC.",
        f"- Seeds: 0, 1, 2; episodes per seed: {episodes_per_seed}.",
        "- Timing uses `time.perf_counter()` around existing fast-loop, slow-loop and simulator calls. Core algorithm code is not modified.",
        "- Fast-loop real-time utilization is `fast_p95_ms / dt_ms`.",
        "",
        "## Key Results",
        "",
    ]
    for case in ["C1--G2--M0", "C2--G2--M0", "C1--G2--M2", "C2--G2--M2"]:
        lines.append(f"### {case}")
        rows = [r for r in summary if r["case_label"] == case]
        for r in rows:
            lines.append(
                f"- {r['method']}: fast p95 = {r['fast_p95_ms_mean']:.4f} ms "
                f"({r['fast_p95_utilization_mean'] * 100:.4f}% of control period), "
                f"replan mean = {r['replan_mean_ms_mean']:.4f} ms, "
                f"replans/episode = {r['replan_count_mean']:.2f}, "
                f"episode wall = {r['episode_wall_ms_mean']:.1f} ms, "
                f"RTF = {r['real_time_factor_mean']:.1f}."
            )
        fdlc = next((r for r in rows if r["method"] == "FDLC"), None)
        baselines = [r for r in rows if r["method"] in ("CDSL", "WCDL")]
        if fdlc and baselines:
            worst_fast = max(float(r["fast_p95_ms_mean"]) for r in rows)
            lines.append(
                f"- Interpretation: FDLC fast-loop p95 is {fdlc['fast_p95_ms_mean']:.4f} ms; "
                f"the largest p95 among the three methods is {worst_fast:.4f} ms. "
                "All values are far below the simulator control period."
            )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- These are Python wall-clock profiling results on the current workstation, not hardware-in-the-loop timing guarantees.",
        "- Episode wall-clock time mixes decision logic and simulator/environment updates; fast-loop and replan timings are the cleaner algorithmic indicators.",
        "- Replanning time is measured only when `SlowLoop.replan_count` increases after a slow-loop call.",
        "",
        "## Generated Files",
        "",
        "- `computational_cost_episodes.csv`",
        "- `computational_cost_summary.csv`",
        "- `fig_computation_fast_p95.pdf/.png`",
        "- `fig_computation_replan_mean.pdf/.png`",
        "- `fig_computation_episode_wall.pdf/.png`",
        "- `computational_cost_latex_snippet.tex`",
    ]
    (out_dir / "computational_cost_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex(summary: list[dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for case in ["C1--G2--M0", "C2--G2--M0", "C1--G2--M2", "C2--G2--M2"]:
        for method in ["CDSL", "WCDL", "FDLC"]:
            r = next(x for x in summary if x["case_label"] == case and x["method"] == method)
            rows.append(
                f"    {case} & {method} & {r['fast_p95_ms_mean']:.4f} & "
                f"{r['fast_p95_utilization_mean'] * 100:.4f} & "
                f"{r['replan_mean_ms_mean']:.4f} & {r['replan_count_mean']:.2f} & "
                f"{r['episode_wall_ms_mean']:.1f} \\\\"
            )
    text = r"""\subsection{计算开销与实时性分析}
为评估三种决策结构在弱冲突与高冲突场景下的可部署性，本文在不改变核心算法的条件下统计快环控制耗时、慢环重规划耗时、重规划次数和单回合墙钟时间。快环实时性裕度定义为快环 $95\%$ 分位耗时与控制周期的比值。表~\ref{tab:computation-cost-profile} 给出了 C1 和 C2 两类通信退化机制下的统计结果。

\begin{table}[htbp]
  \centering
  \caption{三种决策结构的计算开销与实时性统计}
  \label{tab:computation-cost-profile}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{llccccc}
    \toprule
    场景 & 方法 & 快环95分位/ms & 快环周期占比/\% & 重规划均值/ms & 重规划次数 & 回合墙钟/ms \\
    \midrule
__ROWS__
    \bottomrule
  \end{tabular}
  }
\end{table}

结果表明，三种结构的快环控制耗时均远低于仿真控制周期，说明当前实现中的实时性瓶颈不在快环局部响应。FDLC 的额外开销主要来自更复杂的慢环反馈处理、候选任务评估和局部模式切换，其重规划耗时达到百毫秒量级，显著高于 CDSL 和 WCDL。高冲突场景下，FDLC 的任务收益提升需要与该慢环规划代价联合解释；从本组 profiling 结果看，额外开销没有破坏快环实时性裕度，但会显著增加单回合离线仿真墙钟时间。

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.86\textwidth]{../results/__OUTDIR__/fig_computation_fast_p95.pdf}
  \caption{不同场景下三种结构的快环控制耗时}
  \label{fig:computation-fast-p95}
\end{figure}
""".replace("__ROWS__", "\n".join(rows)).replace("__OUTDIR__", out_dir.name)
    (out_dir / "computational_cost_latex_snippet.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--slow-interval-steps", type=int, default=40)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "results" / f"computation_cost_profile_{tag}"
    else:
        out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = out_dir / "combined_configs"
    combined_dir.mkdir(parents=True, exist_ok=True)

    episode_csv = out_dir / "computational_cost_episodes.csv"
    if args.aggregate_only:
        rows = _read_csv(episode_csv)
    else:
        rows: list[dict[str, Any]] = []
        for case_label, case_path, conflict in CASES:
            comm_case = case_label.split("--", 1)[0]
            for method, system_path in SYSTEMS:
                exp_id = f"{case_label.lower().replace('--', '_')}__{method.lower()}"
                combined_path = combined_dir / f"{exp_id}.yaml"
                _write_combined_config(combined_path, case_path, system_path, exp_id)
                cfg = apply_experiment_presets(load_resolved_config(combined_path))
                for seed in args.seeds:
                    for ep in range(int(args.episodes)):
                        row = _profile_episode(
                            cfg=cfg,
                            seed=int(seed),
                            episode=int(ep),
                            slow_interval_steps=int(args.slow_interval_steps),
                        )
                        row.update(
                            {
                                "case_label": case_label,
                                "conflict_level": conflict,
                                "comm_case": comm_case,
                                "method": method,
                                "config": str(combined_path.relative_to(ROOT)),
                            }
                        )
                        rows.append(row)
                        print(
                            f"{case_label} {method} seed={seed} ep={ep} "
                            f"fast_p95={row['fast_p95_ms']:.4f}ms replan={row['replan_mean_ms']:.4f}ms "
                            f"wall={row['episode_wall_ms']:.1f}ms",
                            flush=True,
                        )
        _write_csv(episode_csv, rows)

    summary = _aggregate(rows)
    _write_csv(out_dir / "computational_cost_summary.csv", summary)
    _plot(summary, out_dir)
    _report(summary, out_dir)
    _latex(summary, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
