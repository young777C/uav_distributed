from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


Point2D = Tuple[float, float]


class FastCommState(str, Enum):
    """
    Paper-1 Eq. (43): S_f = {S_ins, S_tx, S_rec, S_safe, S_back}.
    """

    INS = "Sins"
    TX = "Stx"
    REC = "Srec"
    SAFE = "Ssafe"
    BACK = "Sback"


@dataclass(frozen=True)
class CommState:
    """
    Paper-1 Eq. (35): C_t = (ℓ_t, τ_t, b_t).

    - ℓ_t: packet loss probability in [0,1]
    - τ_t: delay in seconds
    - b_t: bandwidth in bps
    """

    loss_p: float
    delay_s: float
    bandwidth_bps: float
    jitter_s: float = 0.0


@dataclass(frozen=True)
class FastState:
    """
    Paper-1 Eq. (35): x_t^f = (p_t, v_t, ψ_t, C_t, E_t^{loc}, m_t).

    In the lightweight simulator we represent:
    - p_t: 2D position (N,E)
    - v_t: 2D velocity (VN,VE)
    - ψ_t: heading (rad)
    - C_t: current link state (loss/delay/bandwidth)
    - E_t^{loc}: remaining local energy ratio in [0,1]
    - m_t: execution/FSM mode
    """

    p_ne: Point2D
    v_ne: Point2D
    yaw_rad: float
    comm: CommState
    energy_loc: float
    mode: FastCommState


@dataclass(frozen=True)
class SlowGoal:
    """
    Paper-1 Eq. (36): g_t = (i*_t, m^s_t).
    """

    poi_id: Optional[int]
    slow_mode: str = "default"


@dataclass(frozen=True)
class Poi:
    poi_id: int
    pos_ne: Point2D
    cover_radius_m: float
    key_bits: float


@dataclass
class EpisodeLog:
    """
    Minimal log bundle for analysis.
    """

    steps: int = 0
    covered_ids: List[int] | None = None
    effective_ids: List[int] | None = None

    def __post_init__(self) -> None:
        if self.covered_ids is None:
            self.covered_ids = []
        if self.effective_ids is None:
            self.effective_ids = []

