#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from uavlab.common.config import load_resolved_config, load_yaml, save_yaml
from uavlab.experiments.presets import apply_experiment_presets





def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _run_one(
    *,
    project_root: Path,
    runner_module: str,
    config_path: str,
    episodes: int,
    seed: int,
    slow_interval_steps: int,
    comm_mode: str,
    run_dir: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Avoid mixing results when re-running the same tag/seed directory.
    mp = run_dir / "metrics.jsonl"
    if mp.exists():
        try:
            mp.unlink()
        except Exception:
            pass

    # Persist run metadata (best-effort)
    (run_dir / "git_commit.txt").write_text(_git_commit(project_root) + "\n", encoding="utf-8")

    cmd: List[str] = [
        "python3",
        "-m",
        runner_module,
        "--config",
        config_path,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--slow_interval_steps",
        str(slow_interval_steps),
        "--run_dir",
        str(run_dir),
        "--metrics_jsonl",
        str(mp),
    ]
    if comm_mode:
        cmd += ["--comm_mode", comm_mode]

    (run_dir / "command.json").write_text(
        json.dumps({"cmd": cmd, "cwd": str(project_root)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    subprocess.check_call(cmd, cwd=str(project_root), env=env)


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _expand_paths(project_root: Path, specs: Iterable[str]) -> List[str]:
    """
    Expand a list of file specs into concrete file paths.

    Each spec can be:
    - a file path (relative to repo root or absolute)
    - a glob string (relative to repo root or absolute), e.g. "configs/experiments/paper1/system/*.yaml"
    """

    out: List[str] = []
    for s in specs:
        p = Path(str(s))
        if not p.is_absolute():
            p = (project_root / p).resolve()
        # glob pattern if it contains wildcard characters
        if any(ch in str(p) for ch in ["*", "?", "["]):
            out += [str(x) for x in sorted(p.parent.glob(p.name)) if x.is_file()]
        else:
            out.append(str(p))
    # de-dup while preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _resolve_slow_interval_steps(
    *,
    combined_cfg_path: Path,
    plan_default: int,
) -> int:
    """Use ``experiment.sweep.slow_interval_steps`` from preset/system YAML when set."""
    try:
        cfg = apply_experiment_presets(load_resolved_config(combined_cfg_path))
        exp = dict(cfg.get("experiment") or {})
        sweep = dict(exp.get("sweep") or {})
        raw = sweep.get("slow_interval_steps")
        if raw is not None:
            return int(raw)
    except Exception:
        pass
    return int(plan_default)


def _write_axis_combined_config(
    *,
    out_path: Path,
    case_path: str,
    system_path: str,
    experiment_id: str,
) -> None:
    """
    Create a tiny config that composes:
    - env case (base + comm profile + scene_file + env_case metadata)
    - system axis (paper1 struct/modeling/coupling selection)

    Order matters: later items override earlier ones, so system overrides case on shared keys.
    """

    payload: Dict[str, Any] = {
        "extends": [str(Path(case_path).resolve()), str(Path(system_path).resolve())],
        "experiment": {"id": experiment_id},
    }
    save_yaml(payload, out_path)


def _safe_mean(xs: List[float]) -> float:
    return float(statistics.fmean(xs)) if xs else float("nan")


def _safe_stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.stdev(xs))


def _metric_val(obj: Dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        if k not in obj or obj[k] is None:
            continue
        try:
            v = float(obj[k])
        except (TypeError, ValueError):
            continue
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None


def _write_summary_csv(*, root_out: Path) -> Path:
    """
    Aggregate per-run metrics into a single CSV at sweep root.

    Expected layout:
      <root_out>/<case>__<system>/Ts<k>/seed<s>/metrics.jsonl
    """

    rows: List[Dict[str, Any]] = []

    for metrics_path in root_out.rglob("metrics.jsonl"):
        if not metrics_path.is_file():
            continue
        try:
            rel = metrics_path.relative_to(root_out)
        except Exception:
            continue
        parts = rel.parts
        if len(parts) < 4:
            # Not the standard layout; skip.
            continue

        exp_name = str(parts[0])
        ts_part = str(parts[1])
        seed_part = str(parts[2])
        ts = int(ts_part[2:]) if ts_part.startswith("Ts") else None
        seed = int(seed_part[4:]) if seed_part.startswith("seed") else None

        # parse exp_name as case__system
        case_name = exp_name.split("__", 1)[0] if "__" in exp_name else exp_name
        system_name = exp_name.split("__", 1)[1] if "__" in exp_name else ""

        covs: List[float] = []
        effs: List[float] = []
        r_covs: List[float] = []
        r_tasks: List[float] = []
        r_fail_covs: List[float] = []
        t_nf: List[float] = []
        t_ret: List[float] = []
        link_lat: List[float] = []
        link_events: List[float] = []
        ens: List[float] = []
        steps: List[float] = []
        dhome: List[float] = []
        rhome: List[float] = []
        term_energy: List[float] = []
        term_return: List[float] = []
        term_collision: List[float] = []
        term_oob: List[float] = []
        term_time: List[float] = []
        term_unknown: List[float] = []
        goal_switches: List[float] = []
        replans: List[float] = []

        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                v_cov = _metric_val(obj, "R_cov", "coverage_ratio")
                if v_cov is not None:
                    covs.append(v_cov)
                    r_covs.append(v_cov)
                v_task = _metric_val(obj, "R_task", "effective_ratio")
                if v_task is not None:
                    effs.append(v_task)
                    r_tasks.append(v_task)
                v_fail = _metric_val(obj, "R_fail_given_cov", "R_fail_cov")
                if v_fail is not None:
                    r_fail_covs.append(v_fail)
                v_nf = _metric_val(obj, "T_nf_s", "nofly_dwell_s")
                if v_nf is not None:
                    t_nf.append(v_nf)
                v_tret = _metric_val(obj, "T_ret_s", "mean_key_return_delay_s")
                if v_tret is not None:
                    t_ret.append(v_tret)
                v_lat = _metric_val(obj, "link_recovery_latency_s")
                if v_lat is not None:
                    link_lat.append(v_lat)
                v_ev = _metric_val(obj, "link_recovery_event_count")
                if v_ev is not None:
                    link_events.append(v_ev)
                if "remaining_energy" in obj:
                    ens.append(float(obj["remaining_energy"]))
                if "steps" in obj:
                    steps.append(float(obj["steps"]))
                if "dist_to_home_m" in obj:
                    dhome.append(float(obj["dist_to_home_m"]))
                if "returned_home" in obj:
                    try:
                        rhome.append(1.0 if bool(obj["returned_home"]) else 0.0)
                    except Exception:
                        pass
                reason = str(obj.get("termination_reason", "") or "").strip().lower()
                term_energy.append(1.0 if reason == "energy_depleted" else 0.0)
                term_return.append(1.0 if reason == "returned_home" else 0.0)
                term_collision.append(1.0 if reason == "collision" else 0.0)
                term_oob.append(1.0 if reason == "oob" else 0.0)
                term_time.append(1.0 if reason == "time_limit" else 0.0)
                term_unknown.append(1.0 if (reason not in {"energy_depleted", "returned_home", "collision", "oob", "time_limit"}) else 0.0)
                v_gs = _metric_val(obj, "goal_switch_count")
                if v_gs is not None:
                    goal_switches.append(v_gs)
                v_rp = _metric_val(obj, "replan_count")
                if v_rp is not None:
                    replans.append(v_rp)

        rows.append(
            {
                "exp_name": exp_name,
                "Ts": ts if ts is not None else "",
                "seed": seed if seed is not None else "",
                "episodes": int(len(covs) or len(effs) or len(ens) or 0),
                "coverage_mean": _safe_mean(covs),
                "coverage_std": _safe_stdev(covs),
                "effective_mean": _safe_mean(effs),
                "effective_std": _safe_stdev(effs),
                "R_cov_mean": _safe_mean(r_covs),
                "R_cov_std": _safe_stdev(r_covs),
                "R_task_mean": _safe_mean(r_tasks),
                "R_task_std": _safe_stdev(r_tasks),
                "R_fail_given_cov_mean": _safe_mean(r_fail_covs),
                "R_fail_given_cov_std": _safe_stdev(r_fail_covs),
                "T_nf_s_mean": _safe_mean(t_nf),
                "T_nf_s_std": _safe_stdev(t_nf),
                "T_ret_s_mean": _safe_mean(t_ret),
                "T_ret_s_std": _safe_stdev(t_ret),
                "link_recovery_latency_s_mean": _safe_mean(link_lat),
                "link_recovery_latency_s_std": _safe_stdev(link_lat),
                "link_recovery_event_count_mean": _safe_mean(link_events),
                "remaining_energy_mean": _safe_mean(ens),
                "remaining_energy_std": _safe_stdev(ens),
                "steps_mean": _safe_mean(steps),
                "dist_to_home_mean_m": _safe_mean(dhome),
                "dist_to_home_std_m": _safe_stdev(dhome),
                # returned_home_rate: fraction of episodes whose FINAL state is within return_radius
                "returned_home_rate": _safe_mean(rhome),
                # termination_reason rates: fraction of episodes ending due to each reason
                "energy_rate": _safe_mean(term_energy),
                "term_return_rate": _safe_mean(term_return),
                "collision_rate": _safe_mean(term_collision),
                "oob_rate": _safe_mean(term_oob),
                "time_limit_rate": _safe_mean(term_time),
                "unknown_rate": _safe_mean(term_unknown),
                "goal_switch_count_mean": _safe_mean(goal_switches),
                "replan_count_mean": _safe_mean(replans),
            }
        )

    rows.sort(key=lambda r: (str(r.get("exp_name", "")), str(r.get("Ts", "")), str(r.get("seed", ""))))

    out_csv = root_out / "summary.csv"
    fieldnames = [
        "exp_name",
        "Ts",
        "seed",
        "episodes",
        "coverage_mean",
        "coverage_std",
        "effective_mean",
        "effective_std",
        "R_cov_mean",
        "R_cov_std",
        "R_task_mean",
        "R_task_std",
        "R_fail_given_cov_mean",
        "R_fail_given_cov_std",
        "T_nf_s_mean",
        "T_nf_s_std",
        "T_ret_s_mean",
        "T_ret_s_std",
        "link_recovery_latency_s_mean",
        "link_recovery_latency_s_std",
        "link_recovery_event_count_mean",
        "remaining_energy_mean",
        "remaining_energy_std",
        "steps_mean",
        "dist_to_home_mean_m",
        "dist_to_home_std_m",
        "returned_home_rate",
        "energy_rate",
        "term_return_rate",
        "collision_rate",
        "oob_rate",
        "time_limit_rate",
        "unknown_rate",
        "goal_switch_count_mean",
        "replan_count_mean",
        # "metrics_path",
    ]

    def fmt4(v: Any) -> Any:
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return ""
            return f"{v:.4f}"
        return v

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt4(r.get(k)) for k in fieldnames})

    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Expand and run a sweep plan.")
    p.add_argument("--plan", type=str, required=True, help="Sweep plan YAML under configs/sweeps/.")
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag; default is timestamp. Output under runs_root/<tag>/...",
    )
    p.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only write summary.csv for an existing sweep tag; do not run experiments.",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = (project_root / plan_path).resolve()

    plan: Dict[str, Any] = load_yaml(plan_path)
    axis_grid: Dict[str, Any] = dict(plan.get("axis_grid") or {})
    if not axis_grid:
        raise ValueError("Sweep plan must include `axis_grid` (dict).")

    episodes = int(plan.get("episodes", 5))
    seeds = list(plan.get("seeds", [42]))
    slow_list = list(plan.get("slow_interval_steps_list", [40]))
    comm_mode = str(plan.get("comm_mode", "") or "")
    runs_root = str(plan.get("runs_root", "runs/sweeps"))
    runner_module = str(plan.get("runner_module", "uavlab.paper1.runner.run") or "")

    tag = args.tag.strip() or _now_tag()
    root_out = (project_root / runs_root / tag).resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    save_yaml(plan, root_out / "plan.yaml")

    if bool(args.aggregate_only):
        _ = _write_summary_csv(root_out=root_out)
        return

    systems = _expand_paths(project_root, _as_list(axis_grid.get("systems")))
    cases = _expand_paths(project_root, _as_list(axis_grid.get("cases")))
    if not systems:
        raise ValueError("axis_grid.systems must be non-empty (paths or globs).")
    if not cases:
        raise ValueError("axis_grid.cases must be non-empty (paths or globs).")

    for case_path in cases:
        case_name = Path(case_path).stem
        for system_path in systems:
            system_name = Path(system_path).stem
            exp_name = f"{case_name}__{system_name}"
            for slow_interval_steps_plan in slow_list:
                for seed in seeds:
                    probe_cfg = root_out / exp_name / ".probe_combined.yaml"
                    _write_axis_combined_config(
                        out_path=probe_cfg,
                        case_path=case_path,
                        system_path=system_path,
                        experiment_id=exp_name,
                    )
                    slow_interval_steps = _resolve_slow_interval_steps(
                        combined_cfg_path=probe_cfg,
                        plan_default=int(slow_interval_steps_plan),
                    )
                    run_dir = root_out / exp_name / f"Ts{slow_interval_steps}" / f"seed{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    combined_cfg = run_dir / "combined_config.yaml"
                    _write_axis_combined_config(
                        out_path=combined_cfg,
                        case_path=case_path,
                        system_path=system_path,
                        experiment_id=exp_name,
                    )
                    _run_one(
                        project_root=project_root,
                        runner_module=runner_module,
                        config_path=str(combined_cfg),
                        episodes=episodes,
                        seed=int(seed),
                        slow_interval_steps=int(slow_interval_steps),
                        comm_mode=comm_mode,
                        run_dir=run_dir,
                    )

    # Aggregate for convenience (best-effort).
    _ = _write_summary_csv(root_out=root_out)


if __name__ == "__main__":
    main()

