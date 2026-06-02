from __future__ import annotations

from typing import Any, Dict, List

from uavlab.common.config import ConfigDict, _deep_merge  # noqa: SLF001

def normalize_paper1_struct(raw: Any) -> str:
    """Map legacy struct keys to paper symbols ``cdsl`` / ``wcdl`` / ``fdlc``."""

    s = str(raw or "").strip().lower()
    legacy = {
        "b1_centralized_single_loop": "cdsl",
        "b3_decouple_dual_loop": "wcdl",
        "full_architecture": "fdlc",
    }
    return legacy.get(s, s)


def normalize_paper1_modeling(raw: Any) -> str:
    """Map legacy modeling keys to §4 information-utilization ablation names."""

    s = str(raw or "").strip().lower()
    legacy = {
        "task_comm": "comm_aware_decision",
        "task+comm": "comm_aware_decision",
        "completion_comm": "comm_aware_decision",
        "comm_only": "comm_aware_decision",
        "comm_aware": "comm_aware_decision",
        "task_energy": "energy_aware_decision",
        "task+energy": "energy_aware_decision",
        "completion_energy": "energy_aware_decision",
        "energy_only": "energy_aware_decision",
        "energy_aware": "energy_aware_decision",
        "full_model": "comm_energy_aware_decision",
        "task+comm+energy": "comm_energy_aware_decision",
        "completion_comm_energy": "comm_energy_aware_decision",
        "comm_energy": "comm_energy_aware_decision",
        "comm_energy_aware": "comm_energy_aware_decision",
    }
    return legacy.get(s, s)


# =============================================================================
# Paper-1 §4 / §6.3.1 架构轴（控制变量：建模固定为 comm_energy_aware_decision）
# Keys: cdsl (CDSL), wcdl (WCDL), fdlc (FDLC).
#
# Operational contract (2026-05): struct-axis slow-loop profiles default from
# ``Paper1ContractConfig.struct_profile`` (CDSL ⊂ WCDL ⊂ FDLC feasible sets).
# ``apply_experiment_presets``: preset layers fill *unset* fields; experiment YAML wins on conflict.
# =============================================================================
_ARCH_STRUCTURE_SLOW_SHARED: ConfigDict = {
    "paper1_loops": {
        "slow_policy": "periodic_or_event_replan",
        "slow_backend": "heuristic",
        "modeling": {
            "use_struct_comm_profile": True,
        },
        "slow_loop": {
            "periodic_replan_scope": "repeat",
            "event_triggers": {
                "on_goal_spatial_complete": True,
                "on_goal_effective_complete": True,
            },
        },
        "semantics": {
            "enable_goal_lock": True,
        },
    },
}

ARCH_VARIANT_PRESETS: Dict[str, ConfigDict] = {
    "cdsl": _deep_merge(
        _ARCH_STRUCTURE_SLOW_SHARED,
        {
            "env": {"fast_upload_mode": "fixed", "fixed_send_ratio": 0.7},
            "comm": {"waypoint_delta_max": 0.0},
            "paper1_loops": {
                "allow_waypoint_delta": False,
                "allow_mode_switching": False,
                "return_policy": {
                    "enable_backlog_gates": False,
                    "enable_upload_stuck_recovery": False,
                },
                "semantics": {
                    "structure": "cdsl",
                    "enable_fast_mode_switch": False,
                    "use_comm_in_fast": False,
                    "use_energy_in_fast": False,
                },
            },
            "experiment": {
                "paper1": {"struct": "cdsl"},
                "paper_ref": "paper1 §4 CDSL (center-dominant single-loop)",
                "note": (
                    "CDSL: strictest slow feasible set + periodic_goal coupling; "
                    "fast loop = track + fixed upload only (no link/energy FSM, no waypoint delta)."
                ),
            },
        },
    ),
    "wcdl": _deep_merge(
        _ARCH_STRUCTURE_SLOW_SHARED,
        {
            "env": {"fast_upload_mode": "fixed", "fixed_send_ratio": 0.7},
            "paper1_loops": {
                "allow_waypoint_delta": True,
                "allow_mode_switching": True,
                "return_policy": {
                    "enable_backlog_gates": False,
                    "enable_upload_stuck_recovery": False,
                },
                "fast_to_slow": {
                    "send_backlog": False,
                    "send_mode": False,
                },
                "semantics": {
                    "structure": "wcdl",
                    "enable_fast_mode_switch": True,
                    "use_comm_in_fast": False,
                    "use_energy_in_fast": False,
                },
            },
            "experiment": {
                "paper1": {"struct": "wcdl"},
                "paper_ref": "paper1 §4 WCDL (weakly coupled dual-loop)",
                "note": (
                    "WCDL: medium slow constraints; fast = waypoint delta + fixed upload "
                    "(no link-adaptive Stx/Srec transit); no slow replan on link-drop events."
                ),
            },
        },
    ),
    "fdlc": _deep_merge(
        _ARCH_STRUCTURE_SLOW_SHARED,
        {
            "env": {"fast_upload_mode": "policy"},
            "paper1_loops": {
                "allow_waypoint_delta": True,
                "allow_mode_switching": True,
                "return_policy": {
                    "enable_backlog_gates": True,
                    "enable_upload_stuck_recovery": True,
                    "backlog_soft_poi_count": 2,
                    "backlog_hard_poi_count": 5,
                },
                "fast_to_slow": {
                    "send_backlog": True,
                    "send_mode": True,
                },
                "semantics": {
                    "structure": "fdlc",
                    "enable_fast_mode_switch": True,
                    "use_comm_in_fast": True,
                    "use_energy_in_fast": True,
                },
            },
            "experiment": {
                "paper1": {"struct": "fdlc"},
                "paper_ref": "paper1 §4 FDLC (feedback-driven dual-loop collaboration)",
                "note": (
                    "FDLC: widest slow feasible set + full fast comm+energy FSM "
                    "+ waypoint delta + policy-driven upload + graded event feedback."
                ),
            },
        },
    ),
}

# =============================================================================
# Paper-1 §4 信息利用消融（控制变量：struct=fdlc, coupling=full_coupling）
# Slow + fast loops gated via ``paper1_loops.semantics`` and ``modeling``.
# =============================================================================
MODEL_VARIANT_PRESETS: Dict[str, ConfigDict] = {
    "comm_aware_decision": {
        "slow_loop": {"comm_lout_weight": 1.0, "enforce_energy_hard_constraint": False},
        "paper1_loops": {
            "semantics": {
                "use_comm_in_slow": True,
                "use_energy_in_slow": False,
                "use_comm_in_fast": True,
                "use_energy_in_fast": False,
            },
            "modeling_mode": "comm_aware_decision",
            "modeling": {
                "comm_objective": True,
                "comm_constraint": True,
                "energy_objective": False,
                "energy_constraint": "none",
            },
        },
        "experiment": {
            "paper1": {"modeling": "comm_aware_decision"},
            "paper_ref": "paper1 §4 Communication-aware decision",
            "note": (
                "Slow: comm suitability + control-link mask. Fast: Stx/Srec from link. "
                "No energy budget mask or low-battery RTH."
            ),
        },
    },
    "energy_aware_decision": {
        "slow_loop": {"comm_lout_weight": 0.0, "enforce_energy_hard_constraint": True},
        "paper1_loops": {
            "semantics": {
                "use_comm_in_slow": False,
                "use_energy_in_slow": True,
                "use_comm_in_fast": False,
                "use_energy_in_fast": True,
            },
            "modeling_mode": "energy_aware_decision",
            "modeling": {
                "comm_objective": False,
                "comm_constraint": False,
                "energy_objective": False,
                "energy_constraint": "full",
            },
        },
        "experiment": {
            "paper1": {"modeling": "energy_aware_decision"},
            "paper_ref": "paper1 §4 Energy-aware decision",
            "note": (
                "Slow: energy reachability + tour budget. Fast: low-battery RTH only. "
                "No comm mask/objective or link-adaptive backhaul (FSM: fixed Stx when covered)."
            ),
        },
    },
    "comm_energy_aware_decision": {
        "slow_loop": {"comm_lout_weight": 1.0, "enforce_energy_hard_constraint": True},
        "paper1_loops": {
            "semantics": {
                "use_comm_in_slow": True,
                "use_energy_in_slow": True,
                "use_comm_in_fast": True,
                "use_energy_in_fast": True,
            },
            "modeling_mode": "comm_energy_aware_decision",
            "modeling": {
                "comm_objective": True,
                "comm_constraint": True,
                "energy_objective": False,
                "energy_constraint": "full",
            },
        },
        "experiment": {
            "paper1": {"modeling": "comm_energy_aware_decision"},
            "paper_ref": "paper1 §4 Communication & Energy-aware decision",
            "note": (
                "Joint: slow comm+energy masks/objective; fast link-adaptive backhaul + energy RTH."
            ),
        },
    },
}

# =============================================================================
# Paper-1 §4.2.2 耦合机制三档（控制变量：struct=fdlc, modeling=comm_energy_aware_decision）
# coupling_v2: periodic Ts=200 + fixed 0.2; event/full hybrid + P0 event grading.
# =============================================================================
_COUPLING_P0_SLOW_LOOP: ConfigDict = {
    "replan_trigger_policy": "hybrid",
    "periodic_replan_scope": "repeat",
    "replan_cooldown_s": 15.0,
    "suppress_link_interrupt_in_back": True,
    "event_levels": {
        "on_safety_event": "critical",
        "on_energy_low": "critical",
        "on_link_drop": "warning",
        "on_control_link_lost": "warning",
        "on_post_cover_upload_stuck": "warning",
    },
    "event_triggers": {
        "on_safety_event": True,
        "on_energy_low": True,
        "on_link_drop": True,
        "on_control_link_lost": True,
        "on_goal_spatial_complete": True,
        "on_goal_effective_complete": True,
    },
}

COUPLING_VARIANT_PRESETS: Dict[str, ConfigDict] = {
    "periodic_goal": {
        "env": {"fast_upload_mode": "fixed", "fixed_send_ratio": 0.2},
        "paper1_loops": {
            "coupling_mode": "periodic_goal",
            "allow_mode_switching": False,
            "semantics": {
                "coupling_mechanism": "periodic_goal",
                "enable_event_feedback": False,
                "enable_fast_mode_switch": False,
                "enable_goal_lock": False,
            },
            "slow_loop": {"replan_trigger_policy": "periodic", "periodic_replan_scope": "repeat"},
            "fast_to_slow": {
                "send_completion": False,
                "send_link_stats": "none",
                "send_backlog": False,
                "send_mode": False,
                "send_safety_events": False,
            },
        },
        "experiment": {
            "paper1": {"coupling": "periodic_goal"},
            "sweep": {"slow_interval_steps": 200},
            "paper_ref": "paper1 §4.2.2 PeriodicGoal",
            "note": "coupling_v2: low-frequency periodic replan (Ts=200), fixed upload 0.2, no f2s.",
        },
    },
    "event_driven_goal": {
        "env": {"fast_upload_mode": "fixed", "fixed_send_ratio": 0.2},
        "paper1_loops": {
            "coupling_mode": "event_driven_goal",
            "allow_mode_switching": False,
            "semantics": {
                "coupling_mechanism": "event_driven_goal",
                "enable_event_feedback": True,
                "enable_fast_mode_switch": False,
                "enable_goal_lock": True,
            },
            "slow_loop": dict(_COUPLING_P0_SLOW_LOOP),
            "fast_to_slow": {
                "send_completion": True,
                "send_link_stats": "full",
                "send_backlog": False,
                "send_mode": False,
                "send_safety_events": True,
            },
        },
        "experiment": {
            "paper1": {"coupling": "event_driven_goal"},
            "paper_ref": "paper1 §4.2.2 EventDrivenGoal",
            "note": "coupling_v2: hybrid+P0 slow loop, fixed upload 0.2, partial f2s (no backlog/mode).",
        },
    },
    "full_coupling": {
        "env": {"fast_upload_mode": "policy"},
        "paper1_loops": {
            "coupling_mode": "full_coupling",
            "allow_mode_switching": True,
            "semantics": {
                "coupling_mechanism": "full_coupling",
                "enable_event_feedback": True,
                "enable_fast_mode_switch": True,
                "enable_goal_lock": True,
            },
            "slow_loop": dict(_COUPLING_P0_SLOW_LOOP),
            "fast_to_slow": {
                "send_completion": True,
                "send_link_stats": "full",
                "send_backlog": True,
                "send_mode": True,
                "send_safety_events": True,
            },
        },
        "experiment": {
            "paper1": {"coupling": "full_coupling"},
            "paper_ref": "paper1 §4.2.2 FullCoupling",
            "note": "coupling_v2: hybrid+P0 slow loop, policy FSM, full f2s.",
        },
    },
}

# Appendix / engineering coupling — not in ``paper1_axis_grid_coupling`` (§4.2.2 主三档).
COUPLING_APPENDIX_PRESETS: Dict[str, ConfigDict] = {
    "no_feedback": {
        "env": {"fast_upload_mode": "policy"},
        "paper1_loops": {
            "coupling_mode": "no_feedback",
            "allow_mode_switching": True,
            "semantics": {
                "coupling_mechanism": "no_feedback",
                "enable_event_feedback": False,
                "enable_fast_mode_switch": True,
            },
            "slow_loop": {"replan_trigger_policy": "event", "periodic_replan_scope": "repeat"},
            "fast_to_slow": {
                "send_completion": False,
                "send_link_stats": "none",
                "send_backlog": False,
                "send_mode": False,
                "send_safety_events": False,
            },
        },
        "experiment": {
            "paper1": {"coupling": "no_feedback"},
            "paper_ref": "appendix engineering",
            "note": "Legacy: event replan + fast mode switch, no fast→slow feedback. Not §4.2.2.",
        },
    },
    "no_fast_switching": {
        "env": {"fast_upload_mode": "fixed", "fixed_send_ratio": 0.5},
        "paper1_loops": {
            "coupling_mode": "no_fast_switching",
            "allow_mode_switching": False,
            "semantics": {
                "coupling_mechanism": "no_fast_switching",
                "enable_event_feedback": True,
                "enable_fast_mode_switch": False,
            },
            "slow_loop": {"replan_trigger_policy": "event", "periodic_replan_scope": "repeat"},
            "fast_to_slow": {
                "send_completion": True,
                "send_link_stats": "full",
                "send_backlog": False,
                "send_mode": False,
                "send_safety_events": True,
            },
        },
        "experiment": {
            "paper1": {"coupling": "no_fast_switching"},
            "paper_ref": "appendix engineering",
            "note": "Legacy: event-driven slow + fixed fast upload. Not §4.2.2.",
        },
    },
    "no_replan": {
        "slow_loop": {
            "replan_observation_mode": "full",
            "policy": "advance_on_poi_done",
            "periodic_replan_scope": "init_only",
        },
        "env": {"fast_upload_mode": "policy"},
        "paper1_loops": {
            "coupling_mode": "no_replan",
            "slow_policy": "advance_on_poi_done",
            "allow_mode_switching": True,
            "semantics": {
                "enable_event_feedback": True,
                "enable_fast_mode_switch": True,
            },
            "slow_loop": {"periodic_replan_scope": "init_only"},
        },
        "experiment": {
            "paper1": {"coupling": "no_replan"},
            "paper_ref": "appendix engineering",
            "note": "SlowLoop blocks event replan; advance_on_poi_done only. Not a §4.2.2 profile.",
        },
    },
}

# =============================================================================
# Paper-2 / 附录：工程消融（不要与论文一 paper1 预设轴混用）
# =============================================================================
LEGACY_ABLATION_PRESETS: Dict[str, ConfigDict] = {
    "ablation_no_link_adapt": {
        "env": {"fast_upload_mode": "fixed", "fixed_send_ratio": 0.5},
        "experiment": {
            "paper_ref": "model_design §13.3 Ablation-A",
            "note": "Upload does not adapt to measured link (fixed send_ratio).",
        },
    },
    "ablation_b_no_poi_drop": {
        "slow_loop": {"enforce_energy_hard_constraint": False},
        "experiment": {
            "paper_ref": "model_design §13.3 Ablation-B",
            "note": "All POIs must be visited; do not drop POIs for energy infeasibility.",
        },
    },
}

_COUPLING_REQUIRES_FULL_ARCH_MODEL = frozenset(
    {
        "periodic_goal",
        "event_driven_goal",
        "no_feedback",
        "no_fast_switching",
        "no_replan",
    }
)
_FULL_STRUCT = "fdlc"
_FULL_MODELING = "comm_energy_aware_decision"
_FULL_COUPLING = "full_coupling"

def _norm_key(x: Any) -> str:
    return str(x).strip()


def _merge_preset_defaults(*layers: ConfigDict) -> ConfigDict:
    """Stack preset layers (earlier → lower priority; later layers override on conflict)."""
    out: ConfigDict = {}
    for layer in layers:
        if layer:
            out = _deep_merge(out, layer)
    return out


def apply_experiment_presets(cfg: ConfigDict) -> ConfigDict:
    """
    论文一 paper1（struct × modeling + 可选 coupling）+ 论文二 paper2.ablation 预设合并入口。

    Merge order (low → high priority): ARCH → MODEL → COUPLING → (varying axis re-layer)
    → **experiment YAML (``cfg``)**. Presets only supply defaults for keys not declared in YAML.
    """
    exp = cfg.get("experiment")
    if not isinstance(exp, dict):
        return cfg

    flat_deprecated = [
        k
        for k in ("arch_variant", "model_variant", "coupling_variant", "legacy_ablation")
        if exp.get(k) is not None
    ]
    if flat_deprecated:
        raise ValueError(
            "Deprecated experiment keys: "
            + ", ".join(f"experiment.{k}" for k in flat_deprecated)
            + ". Use `experiment.paper1.{struct,modeling,coupling}` and `experiment.paper2.ablation`."
        )

    if exp.get("planner_variant") is not None:
        raise ValueError(
            "`experiment.planner_variant` is deprecated. "
            "Use `experiment.paper1.struct` + `experiment.paper1.modeling` (+ optional `paper1.coupling`)."
        )

    p1 = exp.get("paper1")
    p2 = exp.get("paper2")
    p1 = p1 if isinstance(p1, dict) else {}
    p2 = p2 if isinstance(p2, dict) else {}

    struct = p1.get("struct")
    modeling = p1.get("modeling")
    coupling = p1.get("coupling")
    ablation = p2.get("ablation")

    if struct is None and modeling is None and ablation is None and coupling is None:
        return cfg

    if ablation is not None and (struct is not None or modeling is not None or coupling is not None):
        raise ValueError(
            "`experiment.paper2.ablation` cannot be combined with `experiment.paper1` preset axes."
        )

    if ablation is not None:
        key = str(ablation).strip()
        preset = LEGACY_ABLATION_PRESETS.get(key)
        if preset is None:
            raise ValueError(
                f"Unknown paper2.ablation {key!r}. Known: {sorted(LEGACY_ABLATION_PRESETS.keys())}"
            )
        return _deep_merge(preset, cfg)

    if coupling is not None and (struct is None or modeling is None):
        raise ValueError(
            "`experiment.paper1.coupling` requires BOTH `experiment.paper1.struct` "
            "and `experiment.paper1.modeling`."
        )

    if struct is None or modeling is None:
        raise ValueError(
            "Paper-1 presets require BOTH `experiment.paper1.struct` and `experiment.paper1.modeling`."
        )

    raw_struct = _norm_key(struct)
    akey = normalize_paper1_struct(raw_struct)
    mkey = normalize_paper1_modeling(_norm_key(modeling))
    ckey = _norm_key(coupling) if coupling is not None else _FULL_COUPLING

    # Control-variable consistency checks (avoid accidental full factorial).
    vary_struct = akey != _FULL_STRUCT
    vary_modeling = mkey != _FULL_MODELING
    vary_coupling = ckey != _FULL_COUPLING
    if (vary_struct + vary_modeling + vary_coupling) > 1:
        raise ValueError(
            "Paper-1 presets allow changing ONLY ONE axis at a time (3+3+3 design). "
            f"Got paper1.struct={akey!r}, paper1.modeling={mkey!r}, paper1.coupling={ckey!r}."
        )
    if vary_struct and (mkey != _FULL_MODELING or ckey != _FULL_COUPLING):
        raise ValueError(
            "Invalid struct-sweep combination: when varying `paper1.struct`, you must fix "
            f"`paper1.modeling={_FULL_MODELING}` and `paper1.coupling={_FULL_COUPLING}` "
            f"(got modeling={mkey!r}, coupling={ckey!r})."
        )
    if vary_modeling and (akey != _FULL_STRUCT or ckey != _FULL_COUPLING):
        raise ValueError(
            "Invalid modeling-sweep combination: when varying `paper1.modeling`, you must fix "
            f"`paper1.struct={_FULL_STRUCT}` and `paper1.coupling={_FULL_COUPLING}` "
            f"(got struct={akey!r}, coupling={ckey!r})."
        )
    if vary_coupling and (akey != _FULL_STRUCT or mkey != _FULL_MODELING):
        raise ValueError(
            "Invalid coupling-sweep combination: when varying `paper1.coupling`, you must fix "
            f"`paper1.struct={_FULL_STRUCT}` and `paper1.modeling={_FULL_MODELING}` "
            f"(got struct={akey!r}, modeling={mkey!r})."
        )
    ap = ARCH_VARIANT_PRESETS.get(akey)
    mp = MODEL_VARIANT_PRESETS.get(mkey)
    if ap is None:
        raise ValueError(
            f"Unknown paper1.struct {akey!r}. Known: {sorted(ARCH_VARIANT_PRESETS.keys())}"
        )
    if mp is None:
        raise ValueError(
            f"Unknown paper1.modeling {mkey!r}. Known: {sorted(MODEL_VARIANT_PRESETS.keys())}"
        )

    cp = COUPLING_VARIANT_PRESETS.get(ckey) or COUPLING_APPENDIX_PRESETS.get(ckey)
    if cp is None:
        main = sorted(COUPLING_VARIANT_PRESETS.keys())
        appendix = sorted(COUPLING_APPENDIX_PRESETS.keys())
        raise ValueError(
            f"Unknown paper1.coupling {ckey!r}. "
            f"Main (§4.2.2): {main}. Appendix/engineering: {appendix}."
        )
    if ckey in _COUPLING_REQUIRES_FULL_ARCH_MODEL and (
        akey != _FULL_STRUCT or mkey != _FULL_MODELING
    ):
        raise ValueError(
            f"paper1.coupling={ckey!r} is only defined under "
            f"`paper1.struct={_FULL_STRUCT}` and `paper1.modeling={_FULL_MODELING}` (control-variable sweep)."
        )

    # Preset stack (defaults only). Re-layer the *varying* axis so unset keys keep axis identity:
    # - Struct sweep: ARCH defaults over MODEL (avoid WCDL picking FDLC-like fast FSM).
    # - Modeling sweep: MODEL defaults over fixed FDLC struct.
    preset_layers: List[ConfigDict] = [ap, mp, cp]
    if struct is not None and not vary_coupling and vary_struct:
        preset_layers.append(ap)
    if modeling is not None and not vary_coupling and vary_modeling:
        preset_layers.append(mp)

    defaults = _merge_preset_defaults(*preset_layers)
    return _deep_merge(defaults, cfg)
