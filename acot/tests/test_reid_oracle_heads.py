"""1006 G8: ReidHead (gallery store rule, select-with-past, EMA, gating) and the oracle / GT-memory diagnostic branches."""
from types import SimpleNamespace as NS

import numpy as np

from acot.runtime.heads import ReidHead


def dets(feats, uv=None):
    feats = np.asarray(feats, np.float64)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    uv = uv or [(100.0 * i, 0.0) for i in range(len(feats))]
    return [NS(key=10 + i, crop_feat=f, vlm_feat=f, u=u, v=v) for i, (f, (u, v)) in enumerate(zip(feats, uv))]


def test_select_with_past_then_store():
    h = ReidHead("dinov2", 2, 2, 0.0, 0.0, 160.0)
    D = dets([[1, 0], [0, 1]])
    assert h.score(D, init_slot=1, committed_slot=None, committed=False) is None   # empty gallery → no scores
    assert len(h.bank) == 1 and np.allclose(h.bank[0], D[1].crop_feat)            # stored the opening pick
    rl = h.score(D, init_slot=0, committed_slot=1, committed=True)                  # committed → store committed view
    assert int(np.argmax(rl)) == 1


def test_fresh_after_reset_stores_handover_pick():
    h = ReidHead("dinov2", 2, 2, 0.0, 0.0, 160.0)
    D = dets([[1, 0], [0, 1]])
    h.score(D, init_slot=1, committed_slot=None, committed=False)
    h.reset()                                                                        # opening hand-over
    h.score(D, init_slot=0, committed_slot=1, committed=True)                       # fresh → store init_slot
    assert np.allclose(h.bank[0], D[0].crop_feat)


def test_tavg_ema_per_key_and_gate():
    h = ReidHead("dinov2", 2, 2, 0.5, 0.0, 160.0)
    D = dets([[1, 0], [0.9, 0.1]])
    h.score(D, init_slot=0, committed_slot=None, committed=False)
    r1 = h.score(D, init_slot=0, committed_slot=0, committed=True)
    assert set(h.ema) == {10, 11} and r1.shape == (2,)
    bank, ema = [b.copy() for b in h.bank], dict(h.ema)
    h.gate(1, D)                                               # PGIF gating is a no-op for reid (legacy parity)
    assert len(h.bank) == len(bank) and all(np.allclose(a, b) for a, b in zip(h.bank, bank)) and h.ema == ema


class Oracle:
    def __init__(self, t):
        self.t, self.calls = t, []

    def true_slot(self, purpose):
        self.calls.append(purpose)
        return self.t


def _pol(kind, memory_seed="committed"):
    import torch
    from acot.runtime.policy import AcotPolicy
    p = AcotPolicy.__new__(AcotPolicy)
    p.kind, p.pgif, p.opening, p.open_window, p.open_k = kind, None, None, 3, 3
    p.memory_seed, p.committed, p.device, p.W, p.H = memory_seed, None, "cpu", 512, 512
    p.vlm_ctx = p.ctx_mask = None
    p.fe = NS(tid=lambda cf, *a: torch.zeros(1, cf.shape[1]))
    p.open_force = False
    return p


def test_oracle_head_u1():
    p = _pol("oracle")
    p.head = None
    D = dets([[1, 0], [0, 1], [1, 1]])
    o = Oracle(2)
    p._identity(NS(shared_open_slot=None), D, o)
    assert p.pred_slot == 2 and p.pred_conf == 1.0 and o.calls == ["oracle_head"]
    p._identity(NS(shared_open_slot=None), D, Oracle(-1))
    assert p.pred_slot == -1


def test_gt_memory_u2_writes_true_slot():
    import torch

    class Tah:
        def __init__(self):
            self.written = []

        def score(self, dets, W, H, rgb=None):
            return torch.tensor([3.0, 0.0, 0.0]), 0.9

        def cold(self):
            return False

        def write(self, slot):
            self.written.append(slot)

    p = _pol("tah", memory_seed="gt")
    p.head = Tah()
    D = dets([[1, 0], [0, 1], [1, 1]])
    p._identity(NS(shared_open_slot=None, rgb=None), D, Oracle(1))
    assert p.pred_slot == 0 and p.head.written == [1]            # selection unchanged, memory written with GT
    p._identity(NS(shared_open_slot=None, rgb=None), D, Oracle(-1))
    assert p.head.written[-1] is None                            # target absent → no write
