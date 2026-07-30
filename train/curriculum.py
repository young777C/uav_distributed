"""Difficulty-curriculum scheduling for Stage 2 (design §4.3).

Tiers (by scene attrs), from easy to language-critical:
  a  : distinct / num_similar==0            — single meaningful target
  b  : same_class                            — similar kind, still distinguishable
  c  : same_shape_diff_color / same_color_diff_shape — near-identical, language REQUIRED

Rather than hard staging (which risks forgetting), we PROGRESSIVELY MIX: start
weighted toward tier a, anneal toward tier c over training. A WeightedRandomSampler
is rebuilt each epoch from these tier weights.
"""

from __future__ import annotations

import numpy as np


def tier_of(strategy: str, num_similar: int) -> str:
    s = (strategy or "").lower()
    if num_similar == 0 or s == "distinct":
        return "a"
    if s == "same_class":
        return "b"
    return "c"                         # same_shape_diff_color / same_color_diff_shape


def tier_weights(epoch: int, total_epochs: int, a_floor: float = 0.15,
                 ramp: float = 1.2) -> dict:
    """Annealed per-tier sampling weight. Early: a-heavy. Late: c-heavy.
    Tier a keeps a floor (`a_floor`) so basic tracking isn't forgotten; `ramp`
    controls how fast a decays (smaller = gentler a->c transition, later saturation).
    Defaults (0.15, 1.2) reproduce the original schedule."""
    p = epoch / max(total_epochs - 1, 1)            # 0 -> 1
    w_a = max(a_floor, 1.0 - ramp * p)
    w_c = min(1.0, 0.15 + 1.1 * p)
    w_b = 0.5                                        # medium tier stays moderate
    tot = w_a + w_b + w_c
    return {"a": w_a / tot, "b": w_b / tot, "c": w_c / tot}


def sample_weights(tiers, epoch, total_epochs, a_floor: float = 0.15,
                   ramp: float = 1.2) -> np.ndarray:
    """Per-sample weight array for WeightedRandomSampler, given each sample's tier."""
    tw = tier_weights(epoch, total_epochs, a_floor, ramp)
    # normalize by tier population so weights reflect target mixture, not raw counts
    from collections import Counter
    pop = Counter(tiers)
    w = np.array([tw[t] / max(pop[t], 1) for t in tiers], dtype=np.float64)
    return w / w.sum()
