from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from uavlab.common.config import ConfigDict
from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds, dual_link_thresholds_from_comm
from uavlab.paper1.sim.energy_model import resolve_energy_coefficients


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class Paper1SimConfig:
    """
    Minimal config extracted from resolved YAML for the lightweight Paper1 simulator.
    """

    step_hz: int
    episode_steps: int
    v_xy_cruise: float
    v_xy_max: float
    approach_slowdown_radius_m: float

    gcs_ne: Point2D
    poi_list: List[Point2D]
    visit_radius_m: float
    key_bits_per_poi: float

    # comm profile knobs (mapped to Paper1 Eq. (5)-(7) in related_models)
    enable_distance_decay: bool
    distance_d0_m: float
    distance_d1_m: float
    distance_loss_min: float
    distance_loss_max: float
    base_loss: float
    loss_jitter_sigma: float
    use_scene_blackholes: bool
    blackhole_extra_loss: float
    delay_mean_s: float
    delay_jitter_s: float

    # energy knobs (engineering defaults)
    energy_per_meter: float
    energy_hover_per_s: float
    energy_safe_margin: float
    hover_speed_threshold_mps: float

    paper1_struct: str
    paper1_modeling: str
    paper1_coupling: str

    # collision penalty knobs (obstacles live in scene YAML; env decides what to do)
    collision_margin_m: float
    collision_energy_penalty: float
    terminate_on_collision: bool

    # boundary / termination knobs
    n_min: float
    n_max: float
    e_min: float
    e_max: float
    boundary_buffer_m: float
    terminate_on_oob: bool

    terminate_on_energy_depleted: bool
    energy_depleted_threshold: float

    return_radius_m: float
    return_reserve_s: float
    terminate_on_returned_home: bool

    # Paper1 §6.3.1 struct presets (B1: comm.waypoint_delta_max=0; full keeps taskA default)
    waypoint_delta_max_m: float
    # fast_upload_mode: "policy" = link-adaptive FSM for Stx/Srec; "fixed" = duty-cycled TX (see fixed_send_ratio).
    fast_upload_mode: str
    fixed_send_ratio: float

    # POI inspection dwell requirement (seconds) for a POI to count as "covered"
    poi_dwell_s: float

    dual_link: Paper1DualLinkThresholds


def _resolve_episode_steps(env: Dict[str, Any], *, step_hz: int) -> int:
    """``mission_time_s`` (wall-clock cap) overrides ``episode_steps`` when > 0."""

    mission_s = float(env.get("mission_time_s", 0.0) or 0.0)
    if mission_s > 0.0:
        return max(1, int(round(mission_s * float(max(1, step_hz)))))
    return max(1, int(env.get("episode_steps", 2500)))


def _resolve_xy_speeds(env: Dict[str, Any]) -> tuple[float, float]:
    """
    ``v_xy_cruise``: **average** cruise speed for energy budgeting and nominal en-route flight (m/s).
    ``v_xy_max``: hard cap for explicit velocity commands (e.g. PID / urgency); not the default pursuit speed.
    """

    vmax = float(env.get("v_xy_max", 1.5))
    cruise_raw = env.get("v_xy_cruise", env.get("v_xy_cruise_mps", vmax))
    vcruise = float(cruise_raw if cruise_raw is not None else vmax)
    if vcruise > vmax:
        raise ValueError(f"env.v_xy_cruise ({vcruise}) must be <= env.v_xy_max ({vmax}).")
    if vcruise <= 0.0 or vmax <= 0.0:
        raise ValueError("env.v_xy_cruise and env.v_xy_max must be positive.")
    return vcruise, vmax


def from_resolved_config(cfg: ConfigDict, scene: Dict[str, Any]) -> Paper1SimConfig:
    env = dict(cfg.get("env") or {})
    comm = dict(cfg.get("comm") or {})
    taskA = dict(cfg.get("taskA") or {})
    slow_loop = dict(cfg.get("slow_loop") or {})
    energy = dict(cfg.get("energy") or {})
    exp = dict(cfg.get("experiment") or {})
    p1 = exp.get("paper1") if isinstance(exp.get("paper1"), dict) else {}
    p1 = p1 or {}

    gcs_ne = tuple(scene.get("gcs_ne") or scene.get("start_ne") or (125.0, 125.0))  # type: ignore[assignment]
    poi_list = list(scene.get("poi_list") or [])
    # Key bits per POI: keep it constant for the lightweight platform for now.
    raw_bits = comm.get("data_chunk_bits", 16_000_000.0)
    if isinstance(raw_bits, str):
        raw_bits = raw_bits.replace(" ", "").replace("_", "")
    key_bits_per_poi = float(raw_bits)

    wpd_raw = comm.get("waypoint_delta_max", taskA.get("waypoint_delta_max", 6.0))
    waypoint_delta_max_m = float(wpd_raw)

    poi_dwell_s = float(
        slow_loop.get(
            "poi_dwell_s",
            slow_loop.get("t_obs_s", energy.get("t_hov_s", 10.0)),
        )
    )

    _fum = str(env.get("fast_upload_mode", "policy")).strip().lower()
    if _fum not in ("policy", "fixed"):
        _fum = "policy"
    _fsr = float(env.get("fixed_send_ratio", 0.5))
    _fsr = min(1.0, max(0.05, _fsr))

    step_hz = int(env.get("step_hz", 20))
    v_xy_cruise, v_xy_max = _resolve_xy_speeds(env)
    visit_radius_m = float(scene.get("visit_radius_m", 8.0))
    approach_slowdown_radius_m = float(
        env.get("approach_slowdown_radius_m", max(40.0, 20.0 * visit_radius_m))
    )
    episode_steps = _resolve_episode_steps(env, step_hz=step_hz)
    energy_per_meter, energy_hover_per_s, energy_safe_margin = resolve_energy_coefficients(
        env=env, energy=energy
    )

    return Paper1SimConfig(
        step_hz=step_hz,
        episode_steps=episode_steps,
        v_xy_cruise=v_xy_cruise,
        v_xy_max=v_xy_max,
        approach_slowdown_radius_m=approach_slowdown_radius_m,
        gcs_ne=(float(gcs_ne[0]), float(gcs_ne[1])),
        poi_list=[(float(p[0]), float(p[1])) for p in poi_list],
        visit_radius_m=visit_radius_m,
        key_bits_per_poi=key_bits_per_poi,
        enable_distance_decay=bool(comm.get("enable_distance_decay", False)),
        distance_d0_m=float(comm.get("distance_d0_m", 0.0)),
        distance_d1_m=float(comm.get("distance_d1_m", 250.0)),
        distance_loss_min=float(comm.get("distance_loss_min", 0.05)),
        distance_loss_max=float(comm.get("distance_loss_max", 0.60)),
        base_loss=float(comm.get("base_loss", 0.15)),
        loss_jitter_sigma=float(comm.get("loss_jitter_sigma", 0.0)),
        use_scene_blackholes=bool(comm.get("use_scene_blackholes", False)),
        blackhole_extra_loss=float(comm.get("blackhole_extra_loss", 0.5)),
        delay_mean_s=float(comm.get("delay_mean_s", 0.10)),
        delay_jitter_s=float(comm.get("delay_jitter_s", 0.08)),
        energy_per_meter=energy_per_meter,
        energy_hover_per_s=energy_hover_per_s,
        energy_safe_margin=energy_safe_margin,
        hover_speed_threshold_mps=float(energy.get("hover_speed_threshold_mps", 0.5)),
        paper1_struct=str(p1.get("struct", "")),
        paper1_modeling=str(p1.get("modeling", "")),
        paper1_coupling=str(p1.get("coupling", "full_coupling")),
        collision_margin_m=float(scene.get("collision_margin_m", env.get("collision_margin_m", 0.5))),
        collision_energy_penalty=float(env.get("collision_energy_penalty", 0.02)),
        terminate_on_collision=bool(env.get("terminate_on_collision", False)),

        n_min=float(scene.get("n_min", 0.0)),
        n_max=float(scene.get("n_max", 250.0)),
        e_min=float(scene.get("e_min", 0.0)),
        e_max=float(scene.get("e_max", 250.0)),
        boundary_buffer_m=float(scene.get("boundary_buffer_m", env.get("boundary_buffer_m", 0.0))),
        terminate_on_oob=bool(env.get("terminate_on_oob", False)),

        terminate_on_energy_depleted=bool(env.get("terminate_on_energy_depleted", True)),
        energy_depleted_threshold=float(env.get("energy_depleted_threshold", 0.0)),

        return_radius_m=float(env.get("return_radius_m", scene.get("goal_radius_m", 8.0))),
        return_reserve_s=float(env.get("return_reserve_s", 300.0)),
        terminate_on_returned_home=bool(env.get("terminate_on_returned_home", False)),

        waypoint_delta_max_m=waypoint_delta_max_m,
        fast_upload_mode=_fum,
        fixed_send_ratio=_fsr,
        poi_dwell_s=poi_dwell_s,
        dual_link=dual_link_thresholds_from_comm(comm),
    )

