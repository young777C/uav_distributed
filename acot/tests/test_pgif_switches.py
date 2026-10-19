"""1006 §5.2 / G7: gate_commit and force_conf switches of the PGIF hand-off in AcotPolicy._filter."""
import pytest

from acot.runtime.policy import AcotPolicy


class Head:
    def __init__(self):
        self.gated, self.undone = [], 0

    def gate(self, slot):
        self.gated.append(slot)

    def undo(self):
        self.undone += 1


class FakePgif:
    def __init__(self, out):
        self.out = out
        self.belief = {10: 0.5, 11: 3.0}

    def step(self, keys, slot, conf):
        return dict(self.out)


class D:
    def __init__(self, k):
        self.key = k


def pol(out, gate_commit=True, force_conf=True):
    p = AcotPolicy.__new__(AcotPolicy)
    p.kind, p.head, p.pgif = "tah", Head(), FakePgif(out)
    p.oracle_absent, p.gate_commit, p.force_conf = "off", gate_commit, force_conf
    p.pred_slot, p.pred_conf, p.committed, p.events = 0, 0.3, None, 0
    return p


DETS = [D(10), D(11)]
FIRE = {"fire": True, "slot": 1, "kind": "gate", "gate_memory": True}
ABSENT = {"fire": False, "slot": None, "kind": "absent", "gate_memory": True}


def test_default_unchanged():
    p = pol(FIRE)
    p._filter(DETS, [10, 11], None, {})
    assert (p.pred_slot, p.pred_conf, p.committed, p.head.gated) == (1, 1.0, 11, [1])
    q = pol(ABSENT)
    q._filter(DETS, [10, 11], None, {})
    assert (q.pred_slot, q.pred_conf, q.head.undone) == (-1, 0.0, 1)


def test_no_gate_commit():
    p = pol(FIRE, gate_commit=False)
    p._filter(DETS, [10, 11], None, {})
    assert (p.pred_slot, p.pred_conf, p.committed, p.head.gated) == (0, 0.3, None, [1])   # memory gated only
    q = pol(ABSENT, gate_commit=False)
    q._filter(DETS, [10, 11], None, {})
    assert (q.pred_slot, q.pred_conf, q.head.undone) == (0, 0.3, 1)                       # write undone, pick kept


def test_no_force_conf():
    p = pol(FIRE, force_conf=False)
    p._filter(DETS, [10, 11], None, {})
    assert (p.pred_slot, p.pred_conf, p.committed, p.head.gated) == (1, pytest.approx(2.5), None, [1])
