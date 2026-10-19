"""1016 §3 step 0: the sticky ("never switch") identity head of the acot runtime and the client scorer's frame_state.
  PYTHONPATH=. python -m pytest acot/tests/test_sticky_head.py -q"""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch

from acot.runtime.policy import AcotPolicy

_CUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "carla_uav_tracking")
if _CUT not in sys.path:
    sys.path.insert(0, _CUT)


def _policy(window=3, opening_scores=None):
    p = AcotPolicy.__new__(AcotPolicy)
    p.kind, p.head, p.pgif, p.device = "sticky", None, None, "cpu"
    p.frame_input, p.mot_frame_ms = None, 0.0
    p.fe = NS(tid=lambda cf, ctx, cm, mask: torch.tensor([[float(i) for i in range(cf.shape[1])]]))  # favours the LAST slot
    p.vlm_ctx = p.ctx_mask = None
    p.committed, p.chal, p.chal_n, p.commit_k = None, None, 0, 5
    p.sticky_key, p.dets = None, None
    p.open_k, p.open_acc, p.open_cnt, p.open_force = 0, {}, {}, False
    p.open_agg, p.open_final_reset, p.open_handoff = "sum", False, "uniform"
    p.open_reset_pgif, p.open_seed_pgif, p.open_force_final = False, 0.0, False
    p.open_window = window
    p.opening = None if opening_scores is None else NS(scores=lambda rgb, dets, lang, hi: opening_scores.pop(0))
    return p


def _dets(keys):
    return tuple(NS(key=k, u=100.0 + 10 * i, v=100.0, w=24.0, h=24.0, vlm_feat=np.zeros(4, np.float32),
                    crop_feat=np.zeros(4)) for i, k in enumerate(keys))


INP = NS(shared_open_slot=None, rgb=None, rgb_hi=None, language="")


def _tick(p, keys):
    d = _dets(keys)
    tr = p._identity(INP, d, None)
    p.dets = d
    if p.pred_conf >= 0.05:
        p._update_commit()
    return tr


def test_binds_at_every_opening_tick_final_rebind():
    # opening sums (agg sum of zs − max): tick 1 → key 5; tick 2 → key 6; tick 3 (final) → key 5 again (codex R1:
    # the final winner differs from the penultimate one, so a missing final-tick re-binding would fail)
    p = _policy(opening_scores=[np.array([1.0, 0.0]), np.array([0.0, 3.0]), np.array([5.0, 0.0])])
    tr = _tick(p, [5, 6])
    assert tr["open"] == 0 and p.sticky_key == 5 and p.pred_slot == 0 and p.pred_conf == 1.0 and p.committed == 5
    _tick(p, [5, 6])
    assert p.sticky_key == 6 and p.pred_slot == 1 and p.committed == 5 and p.chal_n == 1
    tr = _tick(p, [5, 6])                                    # final opening tick: re-bound to the accumulated pick
    assert "open" in tr and p.sticky_key == 5 and p.pred_slot == 0 and p.committed == 5
    for keys in ([9, 5], [5, 9, 6], [6, 9, 5], [5]):        # grounding favours the last slot; sticky ignores it
        tr = _tick(p, keys)
        assert "open" not in tr and p.sticky_key == 5 and keys[p.pred_slot] == 5 and tr["sticky_key"] == 5
        assert p.committed == 5


def test_commitment_moves_to_bound_key_at_fifth_vote():
    # tick 1 commits key 5; ticks 2–3 (final) bind key 6 → challenger votes 1, 2; then 3, 4, 5 after the opening
    p = _policy(opening_scores=[np.array([1.0, 0.0]), np.array([0.0, 3.0]), np.array([0.0, 0.5])])
    _tick(p, [5, 6])
    _tick(p, [5, 6])
    _tick(p, [5, 6])
    assert p.sticky_key == 6 and p.committed == 5 and p.chal_n == 2
    _tick(p, [5, 6])
    _tick(p, [6, 5])
    assert p.committed == 5 and p.chal_n == 4                # just before the fifth vote
    _tick(p, [5, 6])
    assert p.committed == 6 and p.chal_n == 0                # at the fifth vote (K = 5)


def test_empty_tick_shared_opening_and_reset():
    p = _policy(window=1, opening_scores=[np.array([0.0, 1.0])])
    _tick(p, [3, 4])
    tr = p._identity(INP, None, None)                        # no detections this tick
    assert tr == {"n": 0} and p.pred_slot == -1 and p.pred_conf == 0.0 and p.sticky_key == 4
    q = _policy(window=2, opening_scores=[np.array([1.0, 0.0]), np.array([1.0, 0.0])])
    sh = NS(shared_open_slot=1, rgb=None, rgb_hi=None, language="")
    for _ in range(2):                                       # forced shared opening overrides the opening pick
        tr = q._identity(sh, _dets([7, 8]), None)
        assert tr["forced"] and q.sticky_key == 8 and q.committed == 8
    q.estimator = NS(reset=lambda: None)
    q.reset()
    assert q.sticky_key is None and q.committed is None


def test_abstains_without_the_key_never_confident():
    p = _policy(window=1, opening_scores=[np.array([0.0, 1.0])])
    _tick(p, [3, 4])
    assert p.sticky_key == 4 and p.committed == 4
    for keys in ([3], [3, 8], [8, 3, 7]):                     # bound key absent (also a single candidate)
        _tick(p, keys)
        assert p.pred_slot == -1 and p.pred_conf == 0.0 and p.committed == 4 and p.sticky_key == 4
    _tick(p, [8, 4])
    assert p.pred_slot == 1 and p.committed == 4


def test_no_opening_module_binds_first_grounding_pick():
    p = _policy(window=1, opening_scores=None)
    _tick(p, [1, 2, 3])                                      # grounding argmax = last slot → key 3
    assert p.sticky_key == 3 and p.committed == 3
    _tick(p, [1, 2, 7])
    assert p.sticky_key == 3 and p.pred_slot == -1 and p.committed == 3


def test_commitment_never_moves_away_after_opening():
    rng = np.random.default_rng(0)
    p = _policy(opening_scores=[np.array([2.0, 0.0])] * 3)
    for _ in range(3):
        _tick(p, [11, 12])
    assert p.sticky_key == 11 and p.committed == 11
    for _ in range(200):
        keys = list(rng.choice([11, 12, 13, 14, 15], size=int(rng.integers(1, 5)), replace=False))
        _tick(p, keys)
        assert p.committed == 11 and p.sticky_key == 11


def test_scorer_frame_state():
    from acot_client.scoring import AcotScorer
    s = AcotScorer(fps=10, commit_k=5, t_fail_s=2.0, max_lost_s=18.0, central_frac=0.3, horizon_frames=900)
    s.reset()
    info = NS(cand_set=None, pred_slot=-1, committed_key=None)
    seq = [(0, True, True), (1, True, False), (2, False, None), (4, True, None), (5, False, False)]   # step 3 unseen
    for step, present, vis in seq:
        s.update(step=step, target_present=present, target_central=False, in_loss_window=False, info=info,
                 target_visible=vis)
    out = s.finalize()
    assert out["frame_state"] == "vox-px"
    s.reset()
    assert s.finalize()["frame_state"] == ""
    s.reset()                                               # a legacy runtime (no committed_key): still logged
    s.update(step=0, target_present=False, target_central=False, in_loss_window=False,
             info=NS(cand_set=None, pred_slot=-1), target_visible=None)
    out = s.finalize()
    assert out["frame_state"] == "x" and out["pc_correct"] is None
