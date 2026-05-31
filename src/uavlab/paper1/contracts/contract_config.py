from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Literal, TypedDict, cast

from uavlab.common.config import _deep_merge
from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds, dual_link_thresholds_from_comm
from uavlab.paper1.contracts.struct_profile import StructAxisProfile, struct_axis_profile_from_mapping



def normalize_paper1_coupling_mode(raw: str) -> str:
    """
    Map deprecated coupling axis tokens to paper §4.2.2 three profiles.

    - ``no_feedback`` → ``periodic_goal`` (no fast→slow informational feedback).
    - ``no_fast_switching`` → ``event_driven_goal`` (event-driven slow goals, fixed fast execution).
    ``no_replan`` is kept for ``SlowLoop`` (blocks event replan) when used with ``advance_on_poi_done``.
    """

    s = str(raw or "").strip().lower()
    if s == "no_feedback":
        return "periodic_goal"
    if s == "no_fast_switching":
        return "event_driven_goal"
    return s

# `normalize_paper1_struct`函数的作用是将历史（legacy）结构标识符（如 "b1_centralized_single_loop"、"b3_decouple_dual_loop"、"full_architecture"）映射为论文中使用的标准结构代号，如 "cdsl"（Centralized Distributed Single Loop）、"wcdl"（Weakly Coupled Dual Loop）、"fdlc"（Fully Distributed Loop Coupling）。如果输入不是历史标识符，则返回原始字符串的规范化小写形式。

def normalize_paper1_struct(raw: Any) -> str:
    """Map legacy struct keys to paper symbols ``cdsl`` / ``wcdl`` / ``fdlc``."""

    s = str(raw or "").strip().lower()
    legacy = {
        "b1_centralized_single_loop": "cdsl",
        "b3_decouple_dual_loop": "wcdl",
        "full_architecture": "fdlc",
    }
    return legacy.get(s, s)


# §4 structure comparison: fixed CommEnergy + coupling label ``full_coupling``;
# shared hybrid slow replan + full fast→slow feedback; only fast-loop capability
# differs (see 快慢环决策要点与仓库实现细节.md §2.2).
_STRUCTURE_FORCE_KEYS = frozenset(
    {
        "enable_fast_mode_switch",
        "use_comm_in_fast",
        "use_energy_in_fast",
    }
)

_STRUCTURE_SEMANTICS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cdsl": {
        "structure": "cdsl",
        "enable_fast_mode_switch": False,
        "use_comm_in_fast": False,
        "use_energy_in_fast": False,
    },
    "wcdl": {
        "structure": "wcdl",
        "enable_fast_mode_switch": True,
        "use_comm_in_fast": False,
        "use_energy_in_fast": False,
    },
    "fdlc": {
        "structure": "fdlc",
        "enable_fast_mode_switch": True,
        "use_comm_in_fast": True,
        "use_energy_in_fast": True,
    },
}


def _resolve_paper1_structure_key(*, cfg: Dict[str, Any], sem: Dict[str, Any]) -> str:
    if sem.get("structure") is not None and str(sem.get("structure")).strip():
        return normalize_paper1_struct(sem.get("structure"))
    exp = cfg.get("experiment")
    if isinstance(exp, dict):
        p1 = exp.get("paper1")
        if isinstance(p1, dict) and p1.get("struct") is not None:
            return normalize_paper1_struct(p1.get("struct"))
    return "fdlc"


def _apply_structure_semantics_defaults(*, cfg: Dict[str, Any], sem: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing ``semantics`` keys from ``experiment.paper1.struct`` profile (explicit sem wins)."""

    out = dict(sem)
    key = _resolve_paper1_structure_key(cfg=cfg, sem=out)
    profile = _STRUCTURE_SEMANTICS_DEFAULTS.get(key, {})
    # CDSL/WCDL: enforce fast-loop profile. FDLC: fill missing only so modeling ablation can toggle fast flags.
    force_struct_keys = key in ("cdsl", "wcdl")
    for k, v in profile.items():
        if force_struct_keys and k in _STRUCTURE_FORCE_KEYS:
            out[k] = v
        elif k not in out:
            out[k] = v
    return out


def normalize_paper1_slow_policy(raw: str) -> str:
    """
    Map legacy YAML tokens to current names.

    - ``open_loop_sequence`` → ``advance_on_poi_done`` (goal refresh only after POI effective).
    - ``rolling`` → ``periodic_or_event_replan`` (timer/event-driven refresh; see ``slow_loop`` triggers).
    """

    s = str(raw or "").strip().lower()
    if s == "open_loop_sequence":
        return "advance_on_poi_done"
    if s == "rolling":
        return "periodic_or_event_replan"
    return s


def parse_slow_event_triggers(ev_raw: Dict[str, Any]) -> SlowEventTriggersCfg:
    """
    Parse ``slow_loop.event_triggers`` with decoupled spatial vs effective goal edges.

    - ``on_goal_spatial_complete``: dwell/coverage done (``env.covered``), may switch POI before key return.
    - ``on_goal_effective_complete``: key returned (``env.returned``); legacy default via ``on_goal_completed``.
    """

    legacy_goal = bool(ev_raw.get("on_goal_completed", True))
    has_spatial_key = "on_goal_spatial_complete" in ev_raw
    has_effective_key = "on_goal_effective_complete" in ev_raw
    return cast(
        SlowEventTriggersCfg,
        {
            "on_safety_event": bool(ev_raw.get("on_safety_event", True)),
            "on_energy_low": bool(ev_raw.get("on_energy_low", True)),
            "on_link_drop": bool(ev_raw.get("on_link_drop", True)),
            "on_control_link_lost": bool(ev_raw.get("on_control_link_lost", True)),
            "on_goal_completed": legacy_goal,
            "on_goal_spatial_complete": bool(ev_raw.get("on_goal_spatial_complete", False))
            if has_spatial_key
            else False,
            "on_goal_effective_complete": bool(ev_raw.get("on_goal_effective_complete", legacy_goal))
            if has_effective_key
            else legacy_goal,
        },
    )


def _slow_loop_pos_int(value: Any, *, default: int, minimum: int = 1) -> int:
    """Parse a strictly positive int from YAML; reject bool (``bool`` subclasses ``int``)."""

    if isinstance(value, bool):
        return int(default)
    try:
        v = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(max(int(minimum), v))


class FastToSlowCfg(TypedDict, total=False):
    send_completion: bool
    send_link_stats: Literal["full", "none"]
    send_backlog: bool
    send_mode: bool
    send_safety_events: bool


class FastLoopFSMThresholdsCfg(TypedDict, total=False):
    link_loss_safe: float
    link_loss_recover: float
    safe_hover_steps: int


class FastLoopCfg(TypedDict, total=False):
    enable_fsm: bool
    enable_back_mode: bool
    enable_safety_mode: bool
    enable_recovery_mode: bool
    fsm_thresholds: FastLoopFSMThresholdsCfg


class FastGuidanceCfg(TypedDict, total=False):
    enable_obstacle_filter: bool
    enable_smooth_cost: bool
    link_cost_mode: Literal["current", "candidate_proxy"]
    n_candidates: int
    candidate_radius_m: float
    terminal_homing_radius_m: float
    w_dist: float
    w_link: float
    w_smooth: float
    w_goal: float


class FastTrackerPIDCfg(TypedDict, total=False):
    kp: float
    ki: float
    kd: float


class FastTrackerCfg(TypedDict, total=False):
    type: Literal["pure_pursuit", "pid"]
    pid_gains: FastTrackerPIDCfg


class SlowEventTriggersCfg(TypedDict, total=False):
    on_safety_event: bool
    on_energy_low: bool
    on_link_drop: bool
    on_control_link_lost: bool
    # Legacy alias: when ``on_goal_spatial_complete`` / ``on_goal_effective_complete`` are omitted,
    # maps to ``on_goal_effective_complete`` only (historical ``goal_edge`` semantics).
    on_goal_completed: bool
    on_goal_spatial_complete: bool
    on_goal_effective_complete: bool


class SlowLoopTriggerCfg(TypedDict, total=False):
    replan_trigger_policy: Literal["periodic", "event", "hybrid"]
    # When ``replan_trigger_policy`` is periodic/hybrid, gates the *timer* part: ``repeat`` every
    # ``slow_interval_steps``; ``init_only`` only at t=0 (replaces removed ``replan_policy: once``).
    periodic_replan_scope: Literal["repeat", "init_only"]
    event_triggers: SlowEventTriggersCfg
    min_replan_interval_steps: int
    link_drop_loss_p: float
    # NOTE: solver knobs live here to keep switchboard compact; they are consumed by the slow-loop optimizer.
    window_k: int
    prefetch_k_multiplier: int
    horizon_h: int
    path_samples: int
    # Wall-clock cap: after spatial cover, if still in Srec / backlog pending this long,
    # periodic goal_lock may switch to the next POI even without effective completion.
    goal_lock_stuck_s: float
    replan_cooldown_s: float
    suppress_link_interrupt_in_back: bool
    event_levels: Dict[str, str]


class SlowModelingCfg(TypedDict, total=False):
    """
    Slow-loop modeling / ablation knobs (``paper1_loops.modeling``, merged with legacy ``paper1_switchboard``).

    Symmetric mental model (recommended in YAML):
    - Communication: ``comm_objective`` (soft) + ``comm_constraint`` (hard mask M^Q).
    - Energy: ``energy_objective`` (soft on/off) + ``energy_constraint`` (hard level).
      Optional ``energy_objective_scale`` (float, default 1.0) sets the MILP soft penalty when objective is on.

    ``energy_constraint`` (optional string) overrides the granular energy booleans when set:
    - ``none``: no energy mask and no tour budget (ablation: ignore energy in the slow MILP).
    - ``return_home``: per-POI return-home feasibility M^E only (no CP-SAT tour inequality).
    - ``full``: M^E + tour energy budget (paper inequality).

    Legacy keys (still supported):
    - use_comm_term: if explicit comm_* / comm_constraint omitted, defaults both comm toggles
    - use_energy_mask: if ``energy_budget_constraint`` / ``energy_constraint`` omitted, defaults budget flag
    - comm_path_quality_mask: same meaning as ``comm_constraint``; if both are set, ``comm_path_quality_mask`` wins
    - energy_budget_constraint + energy_hard_constraint: fine-grained form of ``energy_constraint``
      (when ``full``, ``energy_hard_constraint`` is redundant but implied).
    - energy_objective_weight: legacy float when ``energy_objective`` is omitted; if ``energy_objective`` is set, use
      ``energy_objective_scale`` (or this key as magnitude when scale is omitted and objective is true).
    """

    # legacy
    use_comm_term: bool
    use_energy_mask: bool
    energy_hard_constraint: bool

    # explicit comm
    comm_objective: bool
    comm_path_quality_mask: bool
    comm_constraint: bool

    # explicit energy
    energy_budget_constraint: bool
    energy_objective: bool
    energy_objective_scale: float
    energy_objective_weight: float
    energy_constraint: str


@dataclass(frozen=True)
class Paper1ContractConfig:
    """
    Paper1 loop + modeling contract.

    Primary YAML block is ``paper1_loops`` (merged over legacy ``paper1_switchboard``).
    Optional ``paper1_loops.semantics`` documents structure/coupling ablations and
    toggles info use in fast/slow (see 快慢环决策要点与仓库实现细节.md).
    """

    slow_policy: str
    slow_sequence_horizon: int
    slow_backend: str
    waypoint_delta_max_m: float
    allow_waypoint_delta: bool
    allow_mode_switching: bool
    coupling_mode: str
    fast_to_slow: FastToSlowCfg
    fast_loop: FastLoopCfg
    fast_guidance: FastGuidanceCfg
    fast_tracker: FastTrackerCfg
    slow_loop_triggers: SlowLoopTriggerCfg
    modeling_mode: str
    # Legacy mirrors (derived for backward compatibility / logging)
    use_comm_term: bool
    use_energy_mask: bool
    # Explicit slow-loop modeling switches
    comm_objective: bool
    comm_path_quality_mask: bool
    energy_budget_constraint: bool
    energy_objective_weight: float
    # Minimal physical feasibility (per-POI return-home) when full budget modeling is off;
    # implied true when energy_budget_constraint is true (redundant to set both in YAML).
    energy_hard_constraint: bool
    # --- semantics (paper1_loops.semantics) ---
    structure: str
    coupling_mechanism: str
    use_comm_in_slow: bool
    use_energy_in_slow: bool
    use_comm_in_fast: bool
    use_energy_in_fast: bool
    enable_event_feedback: bool
    enable_fast_mode_switch: bool
    enable_goal_lock: bool
    dual_link: Paper1DualLinkThresholds
    struct_profile: StructAxisProfile

    @staticmethod
    def from_cfg(cfg: Dict[str, Any], *, waypoint_delta_max_m: float) -> "Paper1ContractConfig":
        dual_link = dual_link_thresholds_from_comm(cfg.get("comm"))
        sb0 = cfg.get("paper1_switchboard")
        sb0 = deepcopy(sb0) if isinstance(sb0, dict) else {}
        pl = cfg.get("paper1_loops")
        pl = deepcopy(pl) if isinstance(pl, dict) else {}
        sem_raw = pl.pop("semantics", None)
        sem: Dict[str, Any] = deepcopy(sem_raw) if isinstance(sem_raw, dict) else {}
        sem = _apply_structure_semantics_defaults(cfg=cfg, sem=sem)
        sb: Dict[str, Any] = _deep_merge(sb0, pl)

        sl = dict(cfg.get("slow_loop") or {})

        slow_policy = str(sb.get("slow_policy", "") or "").strip().lower()
        if not slow_policy:
            slow_policy = str(sl.get("policy", "") or "").strip().lower()
        if not slow_policy:
            slow_policy = "periodic_or_event_replan"
        else:
            slow_policy = normalize_paper1_slow_policy(slow_policy)

        slow_sequence_horizon = int(sb.get("slow_sequence_horizon", sl.get("sequence_horizon", 999)))
        slow_backend = str(sb.get("slow_backend", sl.get("backend", "heuristic"))).strip().lower()
        if slow_backend in ("onboard_greedy", "onboard", "greedy_onboard"):
            raise ValueError(
                "slow_backend 'onboard_greedy' was removed (pure-onboard baseline dropped per paper §4). "
                "Use paper1_loops.slow_backend: heuristic. "
                "See configs/experiments/paper1/README.md and docs/experiments_baseline_guide.md."
            )

        allow_waypoint_delta = bool(sb.get("allow_waypoint_delta", True))
        allow_mode_switching = bool(sb.get("allow_mode_switching", True))

        coupling_mode = normalize_paper1_coupling_mode(str(sb.get("coupling_mode", "full_coupling")).strip())
        f2s_raw = sb.get("fast_to_slow")
        f2s_raw = f2s_raw if isinstance(f2s_raw, dict) else {}
        send_link_stats = str(f2s_raw.get("send_link_stats", "full")).strip().lower()
        if send_link_stats not in ("full", "none"):
            send_link_stats = "full"
        fast_to_slow: FastToSlowCfg = cast(
            FastToSlowCfg,
            {
                "send_completion": bool(f2s_raw.get("send_completion", True)),
                "send_link_stats": cast(Literal["full", "none"], send_link_stats),
                "send_backlog": bool(f2s_raw.get("send_backlog", True)),
                "send_mode": bool(f2s_raw.get("send_mode", True)),
                "send_safety_events": bool(f2s_raw.get("send_safety_events", True)),
            },
        )

        fl_raw = sb.get("fast_loop")
        fl_raw = fl_raw if isinstance(fl_raw, dict) else {}
        thr_raw = fl_raw.get("fsm_thresholds")
        thr_raw = thr_raw if isinstance(thr_raw, dict) else {}
        fast_loop: FastLoopCfg = cast(
            FastLoopCfg,
            {
                # Back-compat: if enable_fsm absent, fall back to allow_mode_switching.
                "enable_fsm": bool(fl_raw.get("enable_fsm", allow_mode_switching)),
                "enable_back_mode": bool(fl_raw.get("enable_back_mode", True)),
                "enable_safety_mode": bool(fl_raw.get("enable_safety_mode", True)),
                "enable_recovery_mode": bool(fl_raw.get("enable_recovery_mode", True)),
                "fsm_thresholds": cast(
                    FastLoopFSMThresholdsCfg,
                    {
                        "link_loss_safe": float(
                            thr_raw.get("link_loss_safe", dual_link.control_max_loss_p)
                        ),
                        "link_loss_recover": float(
                            thr_raw.get("link_loss_recover", dual_link.data_max_loss_p)
                        ),
                        "safe_hover_steps": int(thr_raw.get("safe_hover_steps", 10)),
                    },
                ),
            },
        )

        fg_raw = sb.get("fast_guidance")
        fg_raw = fg_raw if isinstance(fg_raw, dict) else {}
        link_cost_mode = str(fg_raw.get("link_cost_mode", "current")).strip().lower()
        if link_cost_mode not in ("current", "candidate_proxy"):
            link_cost_mode = "current"
        fast_guidance: FastGuidanceCfg = cast(
            FastGuidanceCfg,
            {
                "enable_obstacle_filter": bool(fg_raw.get("enable_obstacle_filter", True)),
                "enable_smooth_cost": bool(fg_raw.get("enable_smooth_cost", True)),
                "link_cost_mode": cast(Literal["current", "candidate_proxy"], link_cost_mode),
                "n_candidates": int(fg_raw.get("n_candidates", 8)),
                "candidate_radius_m": float(fg_raw.get("candidate_radius_m", 6.0)),
                "terminal_homing_radius_m": float(fg_raw.get("terminal_homing_radius_m", 0.0)),
                "w_dist": float(fg_raw.get("w_dist", 1.0)),
                "w_link": float(fg_raw.get("w_link", 2.0)),
                "w_smooth": float(fg_raw.get("w_smooth", 0.25)),
                "w_goal": float(fg_raw.get("w_goal", 0.5)),
            },
        )

        ft_raw = sb.get("fast_tracker")
        ft_raw = ft_raw if isinstance(ft_raw, dict) else {}
        ft_type = str(ft_raw.get("type", "pure_pursuit")).strip().lower()
        if ft_type not in ("pure_pursuit", "pid"):
            ft_type = "pure_pursuit"
        pid_raw = ft_raw.get("pid_gains")
        pid_raw = pid_raw if isinstance(pid_raw, dict) else {}
        fast_tracker: FastTrackerCfg = cast(
            FastTrackerCfg,
            {
                "type": cast(Literal["pure_pursuit", "pid"], ft_type),
                "pid_gains": cast(
                    FastTrackerPIDCfg,
                    {
                        "kp": float(pid_raw.get("kp", 1.0)),
                        "ki": float(pid_raw.get("ki", 0.0)),
                        "kd": float(pid_raw.get("kd", 0.0)),
                    },
                ),
            },
        )

        st_raw = sb.get("slow_loop")
        st_raw = st_raw if isinstance(st_raw, dict) else {}
        replan_trigger_policy = str(st_raw.get("replan_trigger_policy", "periodic")).strip().lower()
        coupling_mechanism_resolved = str(sem.get("coupling_mechanism") or "").strip().lower()
        if not coupling_mechanism_resolved:
            coupling_mechanism_resolved = str(sb.get("coupling_mode", "full_coupling")).strip().lower()
        if coupling_mechanism_resolved == "periodic_goal" and "replan_trigger_policy" not in st_raw:
            replan_trigger_policy = "periodic"
        elif coupling_mechanism_resolved == "event_driven_goal" and "replan_trigger_policy" not in st_raw:
            replan_trigger_policy = "event"
        if replan_trigger_policy not in ("periodic", "event", "hybrid"):
            replan_trigger_policy = "periodic"

        pr_scope_raw = st_raw.get("periodic_replan_scope", sl.get("periodic_replan_scope"))
        if pr_scope_raw is None or str(pr_scope_raw).strip() == "":
            legacy_rp = str(sl.get("replan_policy", "") or "").strip().lower()
            periodic_replan_scope: Literal["repeat", "init_only"] = (
                "init_only" if legacy_rp in ("once", "one", "single") else "repeat"
            )
        else:
            _ps = str(pr_scope_raw).strip().lower()
            periodic_replan_scope = cast(
                Literal["repeat", "init_only"], _ps if _ps == "init_only" else "repeat"
            )

        ev_raw = st_raw.get("event_triggers")
        ev_raw = ev_raw if isinstance(ev_raw, dict) else {}
        slow_loop_triggers: SlowLoopTriggerCfg = cast(
            SlowLoopTriggerCfg,
            {
                "replan_trigger_policy": cast(Literal["periodic", "event", "hybrid"], replan_trigger_policy),
                "periodic_replan_scope": periodic_replan_scope,
                "event_triggers": parse_slow_event_triggers(ev_raw),
                "min_replan_interval_steps": int(st_raw.get("min_replan_interval_steps", 5)),
                "link_drop_loss_p": float(
                    st_raw.get("link_drop_loss_p", dual_link.data_weak_loss_p)
                ),
                "window_k": int(st_raw.get("window_k", 12)),
                "prefetch_k_multiplier": _slow_loop_pos_int(
                    st_raw.get("prefetch_k_multiplier", 3), default=3, minimum=1
                ),
                "horizon_h": int(st_raw.get("horizon_h", 4)),
                "path_samples": int(st_raw.get("path_samples", 9)),
                "goal_lock_stuck_s": float(st_raw.get("goal_lock_stuck_s", 90.0)),
                **(
                    {"replan_cooldown_s": float(st_raw["replan_cooldown_s"])}
                    if "replan_cooldown_s" in st_raw
                    else {}
                ),
                **(
                    {"suppress_link_interrupt_in_back": bool(st_raw["suppress_link_interrupt_in_back"])}
                    if "suppress_link_interrupt_in_back" in st_raw
                    else {}
                ),
                **(
                    {"event_levels": {str(k): str(v) for k, v in dict(st_raw["event_levels"]).items()}}
                    if isinstance(st_raw.get("event_levels"), dict)
                    else {}
                ),
            },
        )

        modeling_mode = str(sb.get("modeling_mode", "completion_comm_energy")).strip()
        modeling = sb.get("modeling")
        modeling = modeling if isinstance(modeling, dict) else {}

        use_comm_term_legacy = bool(modeling.get("use_comm_term", True))
        use_energy_mask_legacy = bool(modeling.get("use_energy_mask", True))

        comm_objective = bool(modeling.get("comm_objective", use_comm_term_legacy))
        if "comm_path_quality_mask" in modeling:
            comm_path_quality_mask = bool(modeling.get("comm_path_quality_mask"))
        elif "comm_constraint" in modeling:
            comm_path_quality_mask = bool(modeling.get("comm_constraint"))
        else:
            comm_path_quality_mask = bool(use_comm_term_legacy)

        if "energy_objective" in modeling:
            if not bool(modeling.get("energy_objective")):
                energy_objective_weight = 0.0
            elif "energy_objective_scale" in modeling:
                energy_objective_weight = float(modeling.get("energy_objective_scale"))
            elif "energy_objective_weight" in modeling:
                energy_objective_weight = float(modeling.get("energy_objective_weight"))
            else:
                energy_objective_weight = 1.0
        else:
            energy_objective_weight = float(modeling.get("energy_objective_weight", 0.0))

        energy_budget_constraint: bool
        energy_hard_constraint: bool
        _ec_raw = modeling.get("energy_constraint")
        if _ec_raw is not None and str(_ec_raw).strip():
            ec = str(_ec_raw).strip().lower()
            if ec in ("none", "off", "no", "false", "0"):
                energy_budget_constraint = False
                energy_hard_constraint = False
            elif ec in ("return_home", "minimal", "hard_only"):
                energy_budget_constraint = False
                energy_hard_constraint = True
            elif ec in ("full", "paper", "budget", "tour"):
                energy_budget_constraint = True
                energy_hard_constraint = True
            else:
                raise ValueError(
                    "paper1_loops.modeling.energy_constraint must be one of "
                    f"none | return_home | full (got {ec!r})"
                )
        else:
            energy_hard_constraint = bool(modeling.get("energy_hard_constraint", True))
            if "energy_budget_constraint" in modeling:
                energy_budget_constraint = bool(modeling.get("energy_budget_constraint"))
            elif "energy_feasibility_mask" in modeling and "energy_tour_budget_constraint" in modeling:
                # Back-compat: previously split; both must be true to match old "full energy modeling" meaning.
                energy_budget_constraint = bool(modeling.get("energy_feasibility_mask")) and bool(
                    modeling.get("energy_tour_budget_constraint")
                )
            else:
                energy_budget_constraint = bool(use_energy_mask_legacy)
            if energy_budget_constraint:
                energy_hard_constraint = True

        if "use_comm_in_slow" in sem:
            if not bool(sem.get("use_comm_in_slow")):
                comm_objective = False
                comm_path_quality_mask = False
        if "use_energy_in_slow" in sem:
            if not bool(sem.get("use_energy_in_slow")):
                energy_budget_constraint = False
                energy_hard_constraint = False
                energy_objective_weight = 0.0

        # Legacy mirrors for older code paths / configs
        use_comm_term = bool(comm_objective or comm_path_quality_mask)
        use_energy_mask = bool(
            energy_budget_constraint or energy_hard_constraint or (energy_objective_weight > 0.0)
        )

        structure = _resolve_paper1_structure_key(cfg=cfg, sem=sem)
        use_comm_in_slow = bool(sem.get("use_comm_in_slow", use_comm_term))
        use_energy_in_slow = bool(sem.get("use_energy_in_slow", use_energy_mask))
        use_comm_in_fast = bool(sem.get("use_comm_in_fast", True))
        use_energy_in_fast = bool(sem.get("use_energy_in_fast", True))
        enable_goal_lock = bool(sem.get("enable_goal_lock", True))

        cm_eff = coupling_mechanism_resolved
        if "enable_event_feedback" in sem:
            enable_event_feedback = bool(sem.get("enable_event_feedback"))
        elif cm_eff == "periodic_goal":
            enable_event_feedback = False
        elif cm_eff == "event_driven_goal":
            enable_event_feedback = True
        elif cm_eff == "full_coupling":
            enable_event_feedback = str(coupling_mode).strip().lower() not in ("periodic_goal", "no_replan", "no_feedback")
        else:
            enable_event_feedback = str(coupling_mode).strip().lower() not in ("periodic_goal", "no_replan", "no_feedback")

        if "enable_fast_mode_switch" in sem:
            enable_fast_mode_switch = bool(sem.get("enable_fast_mode_switch"))
        elif cm_eff == "periodic_goal":
            enable_fast_mode_switch = False
        elif cm_eff == "event_driven_goal":
            enable_fast_mode_switch = False
        elif cm_eff == "full_coupling":
            enable_fast_mode_switch = bool(allow_mode_switching) and str(coupling_mode).strip().lower() not in (
                "event_driven_goal",
                "periodic_goal",
                "no_fast_switching",
            )
        else:
            enable_fast_mode_switch = bool(allow_mode_switching) and str(coupling_mode).strip().lower() not in (
                "event_driven_goal",
                "periodic_goal",
                "no_fast_switching",
            )

        if str(coupling_mode).strip().lower() == "periodic_goal":
            enable_event_feedback = False

        struct_prof_raw = modeling.get("struct_profile")
        struct_prof_raw = struct_prof_raw if isinstance(struct_prof_raw, dict) else {}
        struct_profile = struct_axis_profile_from_mapping(structure, struct_prof_raw)

        # Structure-specific event trigger defaults (explicit ``slow_loop.event_triggers`` wins).
        st_ev_explicit = st_raw.get("event_triggers")
        has_explicit_ev = isinstance(st_ev_explicit, dict) and len(st_ev_explicit) > 0
        if not has_explicit_ev and structure in ("cdsl", "wcdl"):
            ev_merged = {**dict(slow_loop_triggers.get("event_triggers") or {})}
            ev_merged["on_link_drop"] = False
            ev_merged["on_control_link_lost"] = False
            slow_loop_triggers = cast(
                SlowLoopTriggerCfg,
                {
                    **dict(slow_loop_triggers),
                    "event_triggers": parse_slow_event_triggers(ev_merged),
                },
            )

        return Paper1ContractConfig(
            slow_policy=slow_policy,
            slow_sequence_horizon=slow_sequence_horizon,
            slow_backend=slow_backend,
            waypoint_delta_max_m=float(waypoint_delta_max_m),
            allow_waypoint_delta=allow_waypoint_delta,
            allow_mode_switching=allow_mode_switching,
            coupling_mode=coupling_mode,
            fast_to_slow=fast_to_slow,
            fast_loop=fast_loop,
            fast_guidance=fast_guidance,
            fast_tracker=fast_tracker,
            slow_loop_triggers=slow_loop_triggers,
            modeling_mode=modeling_mode,
            use_comm_term=use_comm_term,
            use_energy_mask=use_energy_mask,
            comm_objective=comm_objective,
            comm_path_quality_mask=comm_path_quality_mask,
            energy_budget_constraint=energy_budget_constraint,
            energy_objective_weight=float(energy_objective_weight),
            energy_hard_constraint=energy_hard_constraint,
            structure=structure,
            coupling_mechanism=cm_eff,
            use_comm_in_slow=use_comm_in_slow,
            use_energy_in_slow=use_energy_in_slow,
            use_comm_in_fast=use_comm_in_fast,
            use_energy_in_fast=use_energy_in_fast,
            enable_event_feedback=enable_event_feedback,
            enable_fast_mode_switch=enable_fast_mode_switch,
            enable_goal_lock=enable_goal_lock,
            dual_link=dual_link,
            struct_profile=struct_profile,
        )

