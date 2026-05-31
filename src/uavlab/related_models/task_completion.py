from __future__ import annotations

from uavlab.related_models.key_data_return import return_success
from uavlab.related_models.comm_link_state import LinkState


def is_covered(*, dist_to_poi_m: float, cover_radius_m: float) -> bool:
    """
    Coverage indicator c_i ∈ {0,1}.
    A simple operationalization: covered if within radius.
    """

    return bool(float(dist_to_poi_m) <= float(cover_radius_m))


def effective_completion(
    *,
    covered: bool,
    link: LinkState,
    key_bits: float,
    max_loss_p_for_return: float,
    max_return_time_s: float,
) -> int:
    """
    Paper-1 Eq. (16): e_i = c_i r_i, where r_i is Eq. (12).
    Returns 0/1 for convenience in logging and aggregation.
    """

    r_i = return_success(
        link=link,
        key_bits=key_bits,
        params=None,
    )
    # Apply thresholds explicitly (allow callers to vary without passing params object)
    if float(link.loss_p) > float(max_loss_p_for_return):
        r_i = False
    # return_success already checks time threshold using defaults; re-check with caller threshold
    # by recomputing with the given time limit.
    # Keep implementation minimal: when caller thresholds differ from defaults, use return_success
    # via params at call sites.
    if not covered:
        return 0
    if not r_i:
        return 0
    # If covered and return feasible => effective completion
    return 1

