"""1006 §4 / G6: policy-commitment metrics (AcotScorer) and SR(τ) / AUC-SR on hand-built golden sequences."""
import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "carla_uav_tracking"))
sys.path.insert(0, ROOT)

from acot.analysis import metrics as M  # noqa: E402
from acot_client.scoring import AcotScorer  # noqa: E402


def cset(*cands):
    """cands: (tracker key, is_target, gt_idx)."""
    return NS(cands=[NS(idx=k, is_target=t, gt_idx=g, depth=10.0) for k, t, g in cands],
              true_idx=next((i for i, c in enumerate(cands) if c[1]), -1))


def run(frames, horizon=20, fps=10, max_lost_s=0.5):
    """frames: (target_present, cand_set | None, committed key | 'NA' (legacy: no field), pred_slot)."""
    s = AcotScorer(fps=fps, commit_k=5, t_fail_s=2.0, max_lost_s=max_lost_s, central_frac=0.3, horizon_frames=horizon)
    s.reset({})
    for i, (present, cs, key, pred) in enumerate(frames):
        info = NS(cand_set=cs, pred_slot=pred)
        if key != "NA":
            info.committed_key = key
        s.update(step=i, target_present=present, target_central=False, in_loss_window=False, info=info)
    return s.finalize()


A = cset((7, True, -1), (8, False, 0))          # key 7 = target, key 8 = distractor 0
B = cset((8, False, 0))                         # target occluded: only the distractor has a box


def test_policy_commit_counts_and_runs():
    fr = ([(True, A, None, 0)] * 2            # no commitment yet → not correct
          + [(True, A, 7, 0)] * 4             # committed to the target
          + [(True, B, 7, -1)] * 2            # key 7 has no box → keeps its last actor (target)
          + [(False, B, 7, -1)]               # target out of frame: not counted, runs reset
          + [(True, A, 8, 1)] * 3)            # committed to the distractor
    e = run(fr)
    assert e["pc_correct_frames"] == 6 and e["pc_wrong_frames"] == 3 and e["pc_none_frames"] == 2
    assert e["pc_correct"] == pytest.approx(6 / 11)
    assert e["pc_max_correct_run_s"] == pytest.approx(0.6) and e["pc_max_wrong_run_s"] == pytest.approx(0.3)
    assert e["tr90"] == pytest.approx(6 / 20)
    assert e["pc_switches"] == 1 and e["truncated"] is False


def test_unmatched_key_never_seen_is_none():
    e = run([(True, A, 99, 0)] * 3)
    assert e["pc_none_frames"] == 3 and e["pc_correct"] == 0.0


def test_legacy_runtime_gives_none():
    e = run([(True, A, "NA", 0)] * 3)
    assert e["pc_correct"] is None and e["tr90"] is None
    assert M.value("pc_correct", e) is None and M.value("sr@2", e) is None and M.value("auc_sr", e) is None


def test_truncation_and_sr():
    fr = [(True, A, 8, 1)] * 3 + [(False, None, 8, -1)] * 5        # wrong 0.3 s, then lost 0.5 s → truncated
    e = run(fr)
    assert e["truncated"] is True
    assert M.value("sr@1", e) == 0.0                                 # truncated → failure at every τ
    e2 = run([(True, A, 8, 1)] * 3 + [(True, A, 7, 0)] * 3)
    assert e2["truncated"] is False and e2["pc_max_wrong_run_s"] == pytest.approx(0.3)
    assert M.value("sr@1", e2) == 1.0 and M.value("auc_sr", e2) == 1.0
    e3 = run([(True, A, 8, 1)] * 12 + [(True, A, 7, 0)] * 2)         # wrong run 1.2 s
    assert M.value("sr@1", e3) == 0.0 and M.value("sr@2", e3) == 1.0
    # τ grid 0.5 … 20 s: SR = 1 for τ > 1.2 s
    assert M.value("auc_sr", e3) == pytest.approx(sum(t > 1.2 for t in M.AUC_TAUS) / len(M.AUC_TAUS))


def test_old_scorer_fields_untouched():
    e = run([(True, A, 7, 0)] * 6)
    assert "mis_follow_sustained" in e and e["mis_follow_sustained"] == 0.0
