"""Per-architecture slow-loop constraint profiles (CDSL / WCDL / FDLC)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping


def _normalize_struct_key(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    legacy = {
        "b1_centralized_single_loop": "cdsl",
        "b3_decouple_dual_loop": "wcdl",
        "full_architecture": "fdlc",
    }
    return legacy.get(s, s)


@dataclass(frozen=True)
class StructAxisProfile:
    """
    Structure-axis slow-loop parameters (see ``CDSL_WCDL_FDLC_三种架构实现修改清单.md``).

    Feasible set should satisfy F_CDSL ⊆ F_WCDL ⊆ F_FDLC under identical scenes.
    """

    use_struct_comm_profile: bool = True
    control_cmd_min_ratio: float = 0.85
    control_loss_exposure_max: float = 0.05
    use_data_return_hard_mask: bool = False
    comm_ret_objective_weight: float = 1.0
    energy_plan_margin_frac: float = 0.08
    candidate_degrade_max_level: int = 2
    # Legacy ``comm_constraint`` maps to control-path hard mask when struct profile is on.
    use_control_path_hard_mask: bool = True

    def relaxed_for_level(self, level: int) -> "StructAxisProfile":
        """Return mask settings for candidate-window degrade level (0 = strictest)."""

        lv = int(max(0, level))
        cmd = float(self.control_cmd_min_ratio)
        loss = float(self.control_loss_exposure_max)
        data_hard = bool(self.use_data_return_hard_mask)
        ctrl_mask = bool(self.use_control_path_hard_mask)
        if lv >= 1:
            cmd = float(max(0.55, cmd - 0.10 * lv))
        if lv >= 2:
            data_hard = False
        if lv >= 3:
            ctrl_mask = False
            loss = float(min(1.0, loss + 0.05))
        return StructAxisProfile(
            use_struct_comm_profile=self.use_struct_comm_profile,
            control_cmd_min_ratio=cmd,
            control_loss_exposure_max=loss,
            use_data_return_hard_mask=data_hard,
            comm_ret_objective_weight=self.comm_ret_objective_weight,
            energy_plan_margin_frac=self.energy_plan_margin_frac,
            candidate_degrade_max_level=self.candidate_degrade_max_level,
            use_control_path_hard_mask=ctrl_mask,
        )


_STRUCT_AXIS_DEFAULTS: Dict[str, StructAxisProfile] = {
    "cdsl": StructAxisProfile(
        control_cmd_min_ratio=0.95,
        control_loss_exposure_max=0.0,
        use_data_return_hard_mask=True,
        comm_ret_objective_weight=2.0,
        energy_plan_margin_frac=0.15,
        candidate_degrade_max_level=1,
    ),
    "wcdl": StructAxisProfile(
        control_cmd_min_ratio=0.85,
        control_loss_exposure_max=0.05,
        use_data_return_hard_mask=False,
        comm_ret_objective_weight=1.0,
        energy_plan_margin_frac=0.08,
        candidate_degrade_max_level=2,
    ),
    "fdlc": StructAxisProfile(
        control_cmd_min_ratio=0.70,
        control_loss_exposure_max=0.10,
        use_data_return_hard_mask=False,
        comm_ret_objective_weight=0.5,
        energy_plan_margin_frac=0.08,
        candidate_degrade_max_level=3,
    ),
}


def struct_axis_profile_defaults(struct_key: str) -> StructAxisProfile:
    key = _normalize_struct_key(struct_key)
    return _STRUCT_AXIS_DEFAULTS.get(key, _STRUCT_AXIS_DEFAULTS["fdlc"])


def struct_axis_profile_from_mapping(
    struct_key: str,
    raw: Mapping[str, Any] | None,
) -> StructAxisProfile:
    base = struct_axis_profile_defaults(struct_key)
    if not isinstance(raw, dict) or not raw:
        return base
    fields: Dict[str, Any] = {}
    for f in (
        "use_struct_comm_profile",
        "control_cmd_min_ratio",
        "control_loss_exposure_max",
        "use_data_return_hard_mask",
        "comm_ret_objective_weight",
        "energy_plan_margin_frac",
        "candidate_degrade_max_level",
        "use_control_path_hard_mask",
    ):
        if f in raw:
            fields[f] = raw[f]
    if fields:
        return replace(base, **fields)
    return base
