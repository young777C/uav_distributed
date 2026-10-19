"""Unit tests for rollout/tracklets.py (GT-free tracklet assigner).  python test_tracklets.py"""
import numpy as np

from tracklets import TrackletAssigner


def _f(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def test_stable_ids_under_shuffle():
    """Two well-separated moving objects, detections shuffled every tick → ids follow the objects, not the order."""
    tr = TrackletAssigner(); rng = np.random.default_rng(0)
    fa, fb = _f([1, 0, 0]), _f([0, 1, 0])
    first = None
    for t in range(20):
        dets = [((100 + 5 * t, 200), 30, fa, "A"), ((400 - 5 * t, 100), 30, fb, "B")]
        order = rng.permutation(2); dets = [dets[i] for i in order]
        ids = tr.update([d[0] for d in dets], [d[1] for d in dets], np.stack([d[2] for d in dets]))
        m = {d[3]: i for d, i in zip(dets, ids)}
        first = first or m
        assert m == first, (t, m, first)


def test_short_gap_keeps_id_long_gap_new_id():
    tr = TrackletAssigner(max_age=5); f = _f([1, 1, 0])
    ids = [tr.update([(100 + 4 * t, 100)], [30], f[None])[0] for t in range(5)]
    assert len(set(ids)) == 1
    for _ in range(3):
        tr.update(np.zeros((0, 2)), [], None)                      # 3 missed ticks (occlusion)
    same = tr.update([(100 + 4 * 8, 100)], [30], f[None])[0]       # re-appears where CV predicts
    assert same == ids[0], (same, ids[0])
    for _ in range(7):
        tr.update(np.zeros((0, 2)), [], None)                      # 7 missed ticks > max_age → deleted
    new = tr.update([(100 + 4 * 16, 100)], [30], f[None])[0]
    assert new != ids[0]


def test_appearance_prevents_swap_when_close():
    """Two objects pass close by; distinct appearance keeps identities."""
    tr = TrackletAssigner(); fa, fb = _f([1, 0, 0.1]), _f([0, 1, 0.1])
    first = None
    for t in range(15):
        pa, pb = (100 + 10 * t, 200), (240 - 10 * t, 205)          # cross around t = 7
        ids = tr.update([pa, pb], [30, 30], np.stack([fa, fb]))
        first = first or ids
        assert ids == first, (t, ids, first)


def test_no_gt_fields_used():
    """The assigner's public API takes geometry + features only (no is_target / actor index)."""
    import inspect
    src = inspect.getsource(TrackletAssigner)
    assert "is_target" not in src and ".idx" not in src and "true_idx" not in src


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
