"""Terminal labels for Paper1 runners (scene + paper1 axis tags)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig


def paper1_run_context_label(
    cfg: Mapping[str, Any],
    *,
    contract: Optional[Paper1ContractConfig] = None,
) -> str:
    """
    Compact tag for sweep logs, e.g.
    ``scene=g2_cluster_m2 model=fdlc/comm_energy_aware_decision/full_coupling``.
    """

    exp = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else {}
    p1 = exp.get("paper1") if isinstance(exp.get("paper1"), dict) else {}
    env_case = exp.get("env_case") if isinstance(exp.get("env_case"), dict) else {}

    scene_path = str(cfg.get("scene_file", "") or "").strip()
    scene = Path(scene_path).stem if scene_path else "unknown_scene"

    case_parts = [
        str(env_case.get("comm_case", "") or "").strip(),
        str(env_case.get("task_case", "") or "").strip(),
        str(env_case.get("conflict_level", "") or "").strip(),
    ]
    case_tag = "_".join(p for p in case_parts if p)

    struct = str(p1.get("struct") or (contract.structure if contract is not None else "") or "").strip()
    modeling = str(p1.get("modeling") or "").strip()
    coupling = str(
        p1.get("coupling") or (contract.coupling_mode if contract is not None else "") or ""
    ).strip()
    model_parts = [x for x in (struct, modeling, coupling) if x]
    model_tag = "/".join(model_parts) if model_parts else "unknown_model"

    if case_tag:
        return f"scene={scene} case={case_tag} model={model_tag}"
    return f"scene={scene} model={model_tag}"


def print_paper1_run_header(
    cfg: Mapping[str, Any],
    *,
    contract: Optional[Paper1ContractConfig] = None,
    experiment_id: Any = None,
    slow_interval_steps: Optional[int] = None,
) -> str:
    """Print once per runner invocation; return the context label."""

    ctx = paper1_run_context_label(cfg, contract=contract)
    extra = []
    if experiment_id is not None and str(experiment_id).strip():
        extra.append(f"exp={experiment_id}")
    if slow_interval_steps is not None:
        extra.append(f"Ts={int(slow_interval_steps)}")
    suffix = (" " + " ".join(extra)) if extra else ""
    print(f"[Paper1Lite] run {ctx}{suffix}", flush=True)
    return ctx


def format_paper1_episode_line(
    *,
    ctx: str,
    episode: int,
    metrics: Mapping[str, Any],
) -> str:
    return (
        f"[Paper1Lite] {ctx} ep={int(episode)} "
        f"R_cov={float(metrics['R_cov']):.3f} "
        f"R_task={float(metrics['R_task']):.3f} "
        f"R_fail|cov={float(metrics['R_fail_given_cov']):.3f} "
        f"T_nf={float(metrics['T_nf_s']):.2f}s"
    )
