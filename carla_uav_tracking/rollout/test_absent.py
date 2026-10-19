"""Unit tests for PGIF's absence hypothesis (asd.ASD absent_hold).  python test_absent.py"""
from asd import ASD


def _run(absent_hold, seq):
    f = ASD(mode="filter", filter_kind="tempered", resid_decay=0.98, absent_hold=absent_hold, absent_max=8)
    f.reset(); out = []
    for ids, pick in seq:
        o = f.step(None, ids, ids.index(pick), mem_conf=0.5)
        out.append("ABSENT" if o.get("kind") == "absent" else ids[o["slot"]])
    return out


# target 7 tracked for 20 ticks, then OCCLUDED for 4 ticks (only distractors 3, 5 visible; the memory picks 3),
# then visible again
SEQ = [([7, 3, 5], 7)] * 20 + [([3, 5], 3)] * 4 + [([7, 3, 5], 7)] * 3


def test_hold_through_occlusion():
    out = _run(True, SEQ)
    assert out[:20] == [7] * 20
    assert out[20:24] == ["ABSENT"] * 4, out[20:24]       # target's evidence still dominates → no forced switch
    assert out[24:] == [7] * 3


def test_legacy_forced_switch():
    out = _run(False, SEQ)
    assert out[20:24] == [3] * 4, out[20:24]               # legacy: forced to pick a visible tracklet


def test_unseen_too_long_releases():
    """beyond absent_max ticks the unseen tracklet leaves the competition → a visible tracklet is output."""
    seq = [([7, 3], 7)] * 5 + [([3], 3)] * 12
    out = _run(True, seq)
    assert out[5:5 + 9] == ["ABSENT"] * 9 or out[5] == "ABSENT"
    assert out[-1] == 3, out


def test_evidence_can_overtake():
    """a visible tracklet that accumulates MORE evidence than the decaying unseen one is output (no infinite hold)."""
    seq = [([7, 3], 7)] * 2 + [([3], 3)] * 6
    out = _run(True, seq)
    assert out[-1] == 3, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
