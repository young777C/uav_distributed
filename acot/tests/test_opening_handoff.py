"""1006 F-c / G3: opening hand-over. legacy: only the TAH memory is reset at the non-final opening ticks; uniform: every
identity head is reset (and so re-seeded from the accumulated opening pick through init_slot)."""
import numpy as np
import pytest

from acot.runtime.policy import AcotPolicy


class FakeOpening:
    def __init__(self, picks):
        self.picks, self.i = picks, 0

    def scores(self, rgb, dets, language, rgb_hi):
        z = np.zeros(len(dets))
        z[self.picks[self.i]] = 1.0
        self.i += 1
        return z


class FakeHead:
    def __init__(self):
        self.resets = self.mem_resets = 0

    def reset(self):
        self.resets += 1

    def reset_memory(self):
        self.mem_resets += 1


class Det:
    def __init__(self, k):
        self.key = k


def policy(kind, handoff, window=3):
    p = AcotPolicy.__new__(AcotPolicy)
    p.kind, p.head, p.pgif = kind, FakeHead(), None
    p.opening = FakeOpening([0, 1, 1])
    p.open_window, p.open_agg, p.open_final_reset = window, "sum", False
    p.open_reset_pgif, p.open_force_final, p.open_seed_pgif = False, False, 0.0
    p.open_handoff = handoff
    p.open_k, p.open_acc, p.open_cnt, p.open_force = 0, {}, {}, False
    return p


class Inp:
    rgb = rgb_hi = None
    language = ""


@pytest.mark.parametrize("handoff", ["legacy", "uniform"])
def test_tah_unchanged(handoff):
    p = policy("tah", handoff)
    dets = [Det(5), Det(6)]
    for _ in range(3):
        p._opening(Inp(), dets)
    assert p.head.mem_resets == 2 and p.head.resets == 0       # ticks 1, 2 (non-final) in both modes


@pytest.mark.parametrize("kind", ["assoc", "dam4sam"])
def test_assoc_dam4sam(kind):
    p = policy(kind, "legacy")
    dets = [Det(5), Det(6)]
    picks = [p._opening(Inp(), dets) for _ in range(3)]
    assert p.head.resets == 0 and p.head.mem_resets == 0        # legacy: seeded once (first tick) by the policy
    q = policy(kind, "uniform")
    picks_u = [q._opening(Inp(), dets) for _ in range(3)]
    assert q.head.resets == 2 and q.head.mem_resets == 0        # uniform: reset at the two non-final ticks
    assert picks == picks_u                                     # the opening pick itself is unchanged
