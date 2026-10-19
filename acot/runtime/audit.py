"""P1 ground-truth access audit (framework §7.3). Installed in the policy process by acot/eval/legacy_server_entry.py
WITHOUT changing legacy code:
  * the raw GT arrays of every act() call — gt_candidates[0] tpos, [1] dstates (AuditedArray) and [4] vis (AuditedSeq,
    recursively for vis[1]) — record every read;
  * the GT label attributes CandidateSet.true_idx, Candidate.is_target / .gt_idx and Observation.target_color /
    .target_bp become properties that record every read.
A read is keyed by its call site (repo-relative file, function, whitespace-normalised source line) of the first frame
outside this module. Per episode the counts go to gt_access.json and are judged against acot/config/gt_sites.yaml."""
import linecache
import os
import sys
import threading

import numpy as np

_HERE = os.path.abspath(__file__)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), "..", ".."))
_DATACLASSES = os.path.join("dataclasses.py")


class Audit:
    def __init__(self):
        self.enabled = False
        self.counts = {}
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            out, self.counts = self.counts, {}
        return out

    def record(self, kind, depth=2):
        if not self.enabled:
            return
        f = sys._getframe(depth)
        while f is not None and os.path.abspath(f.f_code.co_filename) == _HERE:
            f = f.f_back
        if f is None:
            return
        fn = os.path.abspath(f.f_code.co_filename)
        g = f                                   # comprehension / lambda frames → the enclosing named function (py3.11
        while g is not None and g.f_code.co_name.startswith("<") and g.f_back is not None \
                and os.path.abspath(g.f_back.f_code.co_filename) == fn:      # has <listcomp> frames, 3.12 does not)
            g = g.f_back
        rel = os.path.relpath(fn, _ROOT) if fn.startswith(_ROOT) else os.path.basename(fn)
        line = " ".join(linecache.getline(fn, f.f_lineno).split())
        key = (kind, rel, g.f_code.co_name, line)
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1


AUDIT = Audit()


def _unwrap(x):
    return x._a if isinstance(x, AuditedArray) else x


class AuditedArray(np.lib.mixins.NDArrayOperatorsMixin):
    """read-recording stand-in for a GT numpy array (tpos, dstates)."""

    def __init__(self, a, kind):
        object.__setattr__(self, "_a", np.asarray(a))
        object.__setattr__(self, "_kind", kind)

    def __array__(self, dtype=None, copy=None):
        AUDIT.record(self._kind)
        return self._a if dtype is None else self._a.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        AUDIT.record(self._kind)
        return getattr(ufunc, method)(*[_unwrap(x) for x in inputs], **kw)

    def __array_function__(self, func, types, args, kwargs):
        AUDIT.record(self._kind)
        return func(*[_unwrap(x) for x in args], **{k: _unwrap(v) for k, v in kwargs.items()})

    def __getitem__(self, i):
        AUDIT.record(self._kind)
        return self._a[i]

    def __iter__(self):
        AUDIT.record(self._kind)
        return iter(self._a)

    def __len__(self):
        AUDIT.record(self._kind)
        return len(self._a)

    def __getattr__(self, name):               # .shape, .dtype, .tolist(), …
        AUDIT.record(self._kind)
        return getattr(self._a, name)

    def __float__(self):
        AUDIT.record(self._kind)
        return float(self._a)

    def __repr__(self):
        return f"AuditedArray({self._kind}, {self._a!r})"


class AuditedSeq:
    """read-recording stand-in for the ray-cast visibility tuple (t_vis, d_vis); vis[1] is wrapped recursively."""

    def __init__(self, seq, kind):
        self._s, self._kind = seq, kind

    def _wrap(self, i, v):
        return AuditedSeq(v, f"{self._kind}[{i}]") if isinstance(v, (list, tuple)) else v

    def __getitem__(self, i):
        AUDIT.record(self._kind)
        return self._wrap(i, self._s[i])

    def __iter__(self):
        AUDIT.record(self._kind)
        return iter([self._wrap(i, v) for i, v in enumerate(self._s)])

    def __len__(self):
        AUDIT.record(self._kind)
        return len(self._s)

    def __bool__(self):
        AUDIT.record(self._kind)
        return bool(self._s)

    def __repr__(self):
        return f"AuditedSeq({self._kind}, {self._s!r})"


def wrap_gt_candidates(gt):
    """(tpos, dstates, campose, road_pred_world, vis) → same tuple with the GT slots audited (campose is ego state)."""
    gt = list(gt)
    if len(gt) > 0 and gt[0] is not None:
        gt[0] = AuditedArray(gt[0], "tpos")
    if len(gt) > 1 and gt[1] is not None:
        gt[1] = AuditedArray(gt[1], "dstates")
    if len(gt) > 4 and gt[4] is not None:
        gt[4] = AuditedSeq(gt[4], "vis")
    return tuple(gt)


_PATCHED = []


def _audited_attr(cls, name, kind):
    """turn instance attribute `name` of cls into a read-recording property (value kept in the instance __dict__)."""
    store = "_acot_" + name
    had, old = name in cls.__dict__, cls.__dict__.get(name)

    def get(self):
        AUDIT.record(kind)
        try:
            return self.__dict__[store]
        except KeyError:
            if had and not isinstance(old, property):
                return old                                    # class-level default (dataclass field default)
            raise AttributeError(name)

    def put(self, v):
        self.__dict__[store] = v

    setattr(cls, name, property(get, put))
    _PATCHED.append((cls, name, had, old))


def install_label_audit():
    from acot_probe.projection import Candidate
    from rollout.candidates import CandidateSet
    from rollout.policy import Observation
    _audited_attr(CandidateSet, "true_idx", "true_idx")
    _audited_attr(Candidate, "is_target", "is_target")
    _audited_attr(Candidate, "gt_idx", "gt_idx")
    _audited_attr(Observation, "target_color", "target_attr")
    _audited_attr(Observation, "target_bp", "target_attr")


def uninstall_label_audit():
    while _PATCHED:
        cls, name, had, old = _PATCHED.pop()
        if had:
            setattr(cls, name, old)
        else:
            delattr(cls, name)


# ------------------------------------------------------------------------------------------------------------ judging
def load_sites(path):
    import yaml
    return [dict(s) for s in (yaml.safe_load(open(path)) or {}).get("sites", [])]


def judge(counts, sites, uses_gt):
    """counts {(kind, file, func, line): n} → (violations, unknown, by_category). A site is matched on (kind, file, func,
    line); category diagnostic:<name> is allowed only if <name> is among the arm's GT channels."""
    idx = {(s["kind"], s["file"], s["func"], s["line"]): s["category"] for s in sites}
    viol, unknown, cats = [], [], {}
    for key, n in sorted(counts.items()):
        cat = idx.get(tuple(key))
        if cat is None:
            if os.path.basename(key[1]) == _DATACLASSES:
                cat = "bookkeeping"
            else:
                unknown.append({"kind": key[0], "file": key[1], "func": key[2], "line": key[3], "n": n})
                continue
        cats[cat] = cats.get(cat, 0) + n
        if cat.startswith("diagnostic:") and not set(cat.split(":", 1)[1].split("|")) & set(uses_gt):
            viol.append({"kind": key[0], "file": key[1], "func": key[2], "line": key[3], "n": n, "category": cat})
    return viol, unknown, cats
