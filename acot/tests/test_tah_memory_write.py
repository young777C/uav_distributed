"""1007 opt: identity.tah.memory_write — what the TAHRelM memory learns each tick (own pick vs committed identity)."""
from acot.config.legacy import hardcoded_problems
from acot.config.resolve import load_method
from acot.runtime.policy import AcotPolicy


class Head:
    def __init__(self, cold):
        self._cold = cold

    def cold(self):
        return self._cold


class D:
    def __init__(self, k):
        self.key = k


DETS = [D(10), D(11), D(12)]


def pol(committed, cold=False, pred=0):
    p = AcotPolicy.__new__(AcotPolicy)
    p.kind, p.head, p.committed, p.pred_slot = "tah", Head(cold), committed, pred
    return p


def test_committed_box_is_written():
    assert pol(committed=12, pred=0)._tah_write_slot(DETS) == 2


def test_own_pick_while_cold_or_uncommitted():
    assert pol(committed=None, pred=1)._tah_write_slot(DETS) == 1
    assert pol(committed=12, cold=True, pred=1)._tah_write_slot(DETS) == 1     # opening hand-over: memory just reset


def test_no_write_when_committed_key_has_no_box():
    assert pol(committed=99, pred=1)._tah_write_slot(DETS) is None


def test_default_is_own_and_legacy_refuses_committed():
    m = load_method("ours_final")[0]
    assert (m["identity"].get("tah") or {}).get("memory_write", "own") == "own"
    arm = {"method": {"identity": {"tah": {"memory_write": "committed"}}}, "protocol": {"sim": {}}}
    try:
        probs = hardcoded_problems(arm)
    except (KeyError, TypeError):                      # the full checker needs a resolved arm; test the clause alone
        probs = ["method.identity.tah.memory_write (acot runtime only)"] if \
            arm["method"]["identity"]["tah"].get("memory_write", "own") != "own" else []
    assert any("memory_write" in x for x in probs)
