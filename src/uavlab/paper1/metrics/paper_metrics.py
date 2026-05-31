"""
Paper §4.2.4 episode metrics (Eq. 57–62).

Operational definitions: see ``docs/paper1_metrics_operationalization.md``.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, Mapping, Optional, Set


def compute_mean_key_return_delay_s(
    *,
    returned: Set[int],
    poi_cov_time_s: Mapping[int, float],
    poi_return_time_s: Mapping[int, float],
    eps: float = 1e-9,
) -> float:
    """
    Paper Eq. (61): mean wall-clock delay from on-site coverage to effective key return.

    T_ret = sum_i c_i r_i (t_i^ret - t_i^cov) / (sum_i c_i r_i + eps)
    Only POIs in ``returned`` with both timestamps contribute.
    """

    num = 0.0
    den = 0.0
    for pid in returned:
        ip = int(pid)
        if ip not in poi_cov_time_s or ip not in poi_return_time_s:
            continue
        lag = float(poi_return_time_s[ip]) - float(poi_cov_time_s[ip])
        if lag < 0.0:
            lag = 0.0
        num += lag
        den += 1.0
    if den <= 0.0:
        return float("nan")
    return float(num / (den + eps))


def compute_paper_episode_metrics(
    *,
    n_pois: int,
    covered: Set[int],
    returned: Set[int],
    poi_cov_time_s: Optional[Mapping[int, float]] = None,
    poi_return_time_s: Optional[Mapping[int, float]] = None,
    nofly_dwell_s: float,
    link_recovery_latencies_s: Iterable[float] | None = None,
    nofly_entry_count: int = 0,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    """
  Compute paper-aligned scalars from per-POI sets.

  - ``covered`` implements ``c_i`` (on-site observation complete).
  - ``returned`` implements ``r_i`` (key data returned); must be subset of ``covered``.
  - ``e_i = c_i * r_i`` so ``|returned| = sum_i e_i`` when ``returned ⊆ covered``.
    """

    n = int(max(1, n_pois))
    covered_set = set(int(x) for x in covered)
    returned_set = set(int(x) for x in returned if int(x) in covered_set)
    c_count = len(covered_set)
    e_count = len(returned_set)
    failed_after_cov = len(covered_set - returned_set)

    r_cov = float(c_count) / float(n)
    r_task = float(e_count) / float(n)
    r_fail_cov = float(failed_after_cov) / float(c_count + eps) if c_count > 0 else 0.0

    lat_list = [float(x) for x in (link_recovery_latencies_s or []) if float(x) >= 0.0]
    if lat_list:
        link_latency_mean = float(statistics.mean(lat_list))
        link_latency_max = float(max(lat_list))
    else:
        link_latency_mean = float("nan")
        link_latency_max = float("nan")

    cov_t = dict(poi_cov_time_s or {})
    ret_t = dict(poi_return_time_s or {})
    t_ret_s = compute_mean_key_return_delay_s(
        returned=returned_set,
        poi_cov_time_s=cov_t,
        poi_return_time_s=ret_t,
        eps=eps,
    )

    return {
        # Paper symbols (primary)
        "R_cov": r_cov,
        "R_task": r_task,
        "R_fail_given_cov": r_fail_cov,
        "R_fail_cov": r_fail_cov,
        "T_ret_s": float(t_ret_s),
        "mean_key_return_delay_s": float(t_ret_s),
        "T_nf_s": float(nofly_dwell_s),
        # Legacy / runner aliases
        "coverage_ratio": r_cov,
        "effective_ratio": r_task,
        "nofly_dwell_s": float(nofly_dwell_s),
        # Diagnostics
        "covered_count": int(c_count),
        "returned_count": int(e_count),
        "failed_after_cov_count": int(failed_after_cov),
        "nofly_entry_count": int(nofly_entry_count),
        "link_recovery_latency_s": link_latency_mean,
        "link_recovery_latency_max_s": link_latency_max,
        "link_recovery_event_count": int(len(lat_list)),
        "effective_return_count": int(len(returned_set)),
        # Eq. (60) check helper: R_task ≈ R_cov * (1 - R_fail_given_cov)
        "R_task_check_product": float(r_cov * (1.0 - r_fail_cov)),
    }
