"""1007 opt: PGIF evidence inheritance across tracker-id changes (filter.inherit; off by default)."""
from acot.runtime.pgif import Pgif

BASE = dict(filter_kind="tempered", resid_decay=0.98, absent_hold=True, absent_max=8, conf_window=30, gate_memory=True,
            eps_switch=0.01, q_lo=0.2, q_hi=0.7)


def run(p, ticks):
    """ticks = [(keys, mem_slot, uv)] → outputs"""
    return [p.step(k, m, 0.5, uv=uv) for k, m, uv in ticks]


# target = key 1 at (100, 100) for 6 ticks next to a distractor key 2; then key 1 vanishes and the target re-appears as
# NEW key 7 at (130, 110); the memory picks the distractor on that tick
HIST = [([1, 2], 0, [(100, 100), (300, 300)])] * 6
SWITCH = [([7, 2], 1, [(130, 110), (300, 300)])]


def test_off_is_unchanged_with_or_without_uv():
    a, b = Pgif(**BASE), Pgif(**BASE)
    oa = run(a, HIST + SWITCH)
    ob = [b.step(k, m, 0.5) for k, m, _ in HIST + SWITCH]
    assert oa == ob and a.belief == b.belief
    assert oa[-1]["kind"] == "absent"                       # without inheritance: key 1 is the absent MAP


def test_successor_inherits_and_is_picked():
    p = Pgif(**BASE, inherit=True)
    out = run(p, HIST + SWITCH)
    assert out[-1]["kind"] == "gate" and out[-1]["slot"] == 0 and out[-1]["fire"]   # re-picks key 7 over the pick 2
    assert p.last_inherit == (1, 7) and 1 not in p.belief and p.belief[7] > p.belief[2]


def test_far_successor_not_inherited():
    p = Pgif(**BASE, inherit=True)
    out = run(p, HIST + [([7, 2], 1, [(450, 450), (300, 300)])])
    assert out[-1]["kind"] == "absent" and p.last_inherit is None


def test_old_key_is_not_a_successor():
    p = Pgif(**BASE, inherit=True)
    hist = [([1, 2, 7], 0, [(100, 100), (300, 300), (400, 50)])] * 6     # key 7 already existed elsewhere
    out = run(p, hist + [([7, 2], 1, [(130, 110), (300, 300)])])
    assert out[-1]["kind"] == "absent" and p.last_inherit is None


def test_from_method_reads_switches():
    F = dict(BASE, inherit=True, inherit_gate_px=100.0, inherit_px_per_tick=10.0)
    p = Pgif.from_method(F)
    assert (p.inherit, p.inherit_gate_px, p.inherit_px_per_tick) == (True, 100.0, 10.0)
    q = Pgif.from_method(dict(BASE))
    assert q.inherit is False
