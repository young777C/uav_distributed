"""Per-POI FCFS return buffers with incremental (sharded) transmission."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set

from uavlab.related_models.comm_link_state import LinkState
from uavlab.related_models.key_data_return import ReturnDecisionParams, effective_bandwidth_bps


@dataclass
class ReturnBuffer:
    poi_id: int
    total_bits: float
    remaining_bits: float
    covered_time_s: float
    first_tx_time_s: float | None = None
    last_progress_time_s: float | None = None
    returned_time_s: float | None = None
    expired: bool = False


@dataclass(frozen=True)
class ReturnProgressResult:
    """Outcome of one ``progress_step`` call."""

    completed_ids: tuple[int, ...] = ()
    expired_ids: tuple[int, ...] = ()
    transmitted_bits: float = 0.0

    @property
    def return_progress_bits(self) -> float:
        return float(self.transmitted_bits)


@dataclass
class ReturnQueue:
    """
    FCFS queue of pending key-data returns.

    Each simulation step may transfer up to ``b_eff * dt_s`` bits when the data link
    satisfies ``loss_p <= max_loss_p_for_return``.

    Expiry (P0 patch): never from ``covered_time`` alone. After the first byte is sent,
    expire only on **stall** — no transfer progress for ``max_return_time_s`` since
    ``last_progress_time_s``.
    """

    _buffers: Dict[int, ReturnBuffer] = field(default_factory=dict)
    _order: Deque[int] = field(default_factory=deque)

    def reset(self) -> None:
        self._buffers.clear()
        self._order.clear()

    def enqueue(self, *, poi_id: int, bits: float, covered_time_s: float) -> None:
        pid = int(poi_id)
        if pid in self._buffers:
            return
        b = float(bits)
        self._buffers[pid] = ReturnBuffer(
            poi_id=pid,
            total_bits=b,
            remaining_bits=b,
            covered_time_s=float(covered_time_s),
        )
        self._order.append(pid)

    @property
    def pending_bits(self) -> float:
        return float(sum(buf.remaining_bits for buf in self._buffers.values() if not buf.expired))

    @property
    def pending_count(self) -> int:
        return int(sum(1 for buf in self._buffers.values() if buf.remaining_bits > 0 and not buf.expired))

    def oldest_pending_age_s_at(self, now_s: float) -> float:
        ages: List[float] = []
        for buf in self._buffers.values():
            if buf.remaining_bits <= 0 or buf.expired:
                continue
            t0 = buf.first_tx_time_s if buf.first_tx_time_s is not None else buf.covered_time_s
            ages.append(float(now_s) - float(t0))
        if not ages:
            return 0.0
        return float(max(ages))

    def pending_poi_ids(self) -> Set[int]:
        return {
            int(buf.poi_id)
            for buf in self._buffers.values()
            if buf.remaining_bits > 0 and not buf.expired
        }

    def remaining_bits_by_poi(self) -> Dict[int, float]:
        return {
            int(buf.poi_id): float(buf.remaining_bits)
            for buf in self._buffers.values()
            if float(buf.remaining_bits) > 0 and not buf.expired
        }

    def returned_time_s_for(self, poi_id: int) -> float | None:
        buf = self._buffers.get(int(poi_id))
        if buf is None:
            return None
        return buf.returned_time_s

    def _mark_progress(self, buf: ReturnBuffer, *, now_s: float, delta_bits: float) -> None:
        if delta_bits <= 0.0:
            return
        if buf.first_tx_time_s is None:
            buf.first_tx_time_s = float(now_s)
        buf.last_progress_time_s = float(now_s)

    def _expire_stalled(self, *, now_s: float, stall_s: float) -> List[int]:
        expired: List[int] = []
        for pid in list(self._order):
            buf = self._buffers.get(pid)
            if buf is None or buf.expired or buf.remaining_bits <= 0.0:
                continue
            if buf.first_tx_time_s is None:
                continue
            last = buf.last_progress_time_s
            if last is None:
                continue
            if float(now_s) - float(last) > float(stall_s):
                buf.expired = True
                buf.remaining_bits = 0.0
                expired.append(int(pid))
        return expired

    def progress_step(
        self,
        *,
        link: LinkState,
        dt_s: float,
        now_s: float,
        params: ReturnDecisionParams | None = None,
    ) -> ReturnProgressResult:
        p = params or ReturnDecisionParams()
        completed: List[int] = []
        transmitted = 0.0
        stall_s = float(p.max_return_time_s)

        max_loss = float(p.max_loss_p_for_return)

        if float(link.loss_p) <= max_loss:
            budget = float(effective_bandwidth_bps(link)) * float(max(0.0, dt_s))
            remaining_budget = budget
            for pid in list(self._order):
                if remaining_budget <= 0.0:
                    break
                buf = self._buffers.get(pid)
                if buf is None or buf.expired or buf.remaining_bits <= 0.0:
                    continue
                take = float(min(buf.remaining_bits, remaining_budget))
                if take <= 0.0:
                    continue
                self._mark_progress(buf, now_s=float(now_s), delta_bits=take)
                buf.remaining_bits -= take
                remaining_budget -= take
                transmitted += take
                if buf.remaining_bits <= 1e-9:
                    buf.remaining_bits = 0.0
                    buf.returned_time_s = float(now_s)
                    completed.append(int(pid))

        expired = self._expire_stalled(now_s=float(now_s), stall_s=stall_s)

        return ReturnProgressResult(
            completed_ids=tuple(completed),
            expired_ids=tuple(expired),
            transmitted_bits=float(transmitted),
        )
