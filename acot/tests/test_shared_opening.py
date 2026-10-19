"""1006 §3.4 / G4: shared opening — the harness maps the reference actor to a slot; during the opening window every
head is re-seeded from it and the identity output / commitment are forced to it; fallback is flagged."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from acot.runtime.policy import AcotPolicy
from acot.runtime.server import Runtime


def test_harness_slot_mapping():
    rt = Runtime.__new__(Runtime)
    rt.shared = {"95000": {"0": -1, "6": 3, "12": 3}}
    rt.seed = 95000
    rt.dets, rt.info = ["a", "b", "c"], NS(gt_idx=(3, -1, 7))
    assert rt.shared_slot(0) == 1 and rt.shared_slot(6) == 0
    assert rt.shared_slot(18) is None                       # not an opening tick
    rt.info = NS(gt_idx=(7, 8, -1))
    assert rt.shared_slot(12) == -1                         # reference actor not among the detections
    rt.seed = 95001
    assert rt.shared_slot(0) is None                        # seed without reference


class Head:
    def __init__(self):
        self.resets = self.mem_resets = self.writes = 0
        self.mem = None

    def reset(self):
        self.resets += 1

    def reset_memory(self):
        self.mem_resets += 1
        self.mem = None

    def cold(self):
        return self.mem is None

    def score(self, dets, *a, init_slot=None, rgb=None):
        import torch
        z = np.zeros(len(dets), np.float32)
        z[-1] = 5.0                                         # the head itself would pick the LAST slot
        return z if init_slot is not None else (torch.from_numpy(z), 0.9)   # assoc: scores; tah: (logits, conf)

    def write(self, slot):
        self.writes += 1
        self.mem = slot

    def gate(self, slot, *a):
        self.mem = slot

    def undo(self):
        pass


class Opening:
    def scores(self, *a):
        return np.zeros(3)


class FakePgif:
    def __init__(self):
        self.steps = []

    def step(self, keys, slot, conf):
        self.steps.append(slot)
        return {"fire": True, "slot": 0, "kind": "gate", "gate_memory": True}


def pol(kind):
    import torch
    p = AcotPolicy.__new__(AcotPolicy)
    p.kind, p.head, p.pgif, p.opening = kind, Head(), FakePgif(), Opening()
    p.open_window, p.open_agg, p.open_final_reset, p.open_reset_pgif = 3, "sum", False, False
    p.open_force_final, p.open_seed_pgif, p.open_handoff = False, 0.0, "uniform"
    p.open_k, p.open_acc, p.open_cnt, p.open_force = 0, {}, {}, False
    p.committed, p.events, p.W, p.H = None, 0, 512, 512
    p.memory_seed = "committed"
    p.gate_commit = p.force_conf = True
    p.oracle_absent = "off"
    p.device = "cpu"
    p.vlm_ctx = p.ctx_mask = None
    p.fe = NS(tid=lambda cf, *a: torch.zeros(1, cf.shape[1]))
    return p


def dets(n=3):
    return tuple(NS(key=10 + i, vlm_feat=np.zeros(4, np.float32)) for i in range(n))


@pytest.mark.parametrize("kind", ["tah", "assoc"])
def test_forced_window_then_release(kind):
    p = pol(kind)
    D = dets()
    for _ in range(3):
        tr = p._identity(NS(shared_open_slot=1, rgb=None, rgb_hi=None, language=""), D, None)
        assert tr["forced"] and p.pred_slot == 1 and p.committed == 11 and p.pred_conf == 1.0
    assert p.pgif.steps == [1, 1, 1]                          # PGIF only accumulates the forced vote
    if kind == "tah":
        assert p.head.mem_resets >= 3 and p.head.mem == 1     # memory seeded by the reference at every opening tick
    else:
        assert p.head.resets >= 3
    tr = p._identity(NS(shared_open_slot=None, rgb=None, rgb_hi=None, language=""), D, None)
    assert "forced" not in tr                                 # released after the window


def test_fallback_flagged_not_forced():
    p = pol("tah")
    tr = p._identity(NS(shared_open_slot=-1, rgb=None, rgb_hi=None, language=""), dets(), None)
    assert tr.get("open_fallback") and "forced" not in tr
