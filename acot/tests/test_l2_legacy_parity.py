"""L2 parity (framework §9): the new importer + metrics + statistics reproduce the legacy analysis scripts exactly —
episode sets and per-seed metrics (iter_gate), tick-log alignment (opening_cl), and every number printed by
arch_compare.py and opening_ds.py (re-computed with acot.analysis, driven by the same RNG stream)."""
import contextlib
import io
import math
import os
import re
import sys

import numpy as np
import pytest

from acot import ROOT
from acot.analysis import legacy_import as L
from acot.analysis import metrics as M
from acot.analysis import stats as S

ROLLOUT = os.path.join(ROOT, "carla_uav_tracking", "rollout")
sys.path.insert(0, ROLLOUT)
import arch_compare as ac  # noqa: E402
import iter_gate as g  # noqa: E402
import opening_cl as oc  # noqa: E402
import opening_ds as od  # noqa: E402

STEMS = ["v2_pgif_sim", "v2t_pgif_trk", "v2h_pgif_trk", "zoa_pgif_trk_a", "zoa_pgif_trk_b", "zoa_pgif_trk_c",
         "zo3_pgif_trk_a", "zo3_pgif_trk_b", "zoo_pgif_trk_a", "zo4_pgif_trk_a"]


def same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def numbers(text):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+|(?<![\w.])[-+]?\d+(?![\w.])", text)]


def run(fn, *argv):
    old = sys.argv
    sys.argv = ["x", *argv]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    finally:
        sys.argv = old
    return buf.getvalue()


def test_episode_import_matches_iter_gate():
    names = sorted({t[1] for t in g.TESTS} | {t[2] for t in g.TESTS} | set(STEMS))
    checked = 0
    for n in names:
        legacy = g.load(n) or {}
        new, _, _ = L.episodes(n)
        assert sorted(legacy) == sorted(new), n
        for s, e in new.items():
            assert e == legacy[s], (n, s)
            for k in g.METRICS:
                assert same(g.METRICS[k](e), M.value(k, e)), (n, s, k)
        checked += len(new)
    assert checked > 100


def test_tick_alignment_matches_opening_cl():
    n = 0
    for stem in STEMS:
        legacy = oc.load(oc.pairs_for(stem))
        new, how = L.ticks(L.tick_pairs(stem))
        assert legacy == new, stem
        assert set(how.values()) <= {"by_order_verified"}, (stem, how)
        for s, R in new.items():
            assert M.opening_lock(R) == ac.opening([stem]).get(s), (stem, s)
        n += len(new)
    assert n > 50


def _arch_compare_report():
    """arch_compare.main() re-computed with acot.analysis (same rows, same RNG order)."""
    data = []
    for lab, arms, logs in ac.STEPS:
        E = {}
        for a in arms:
            E.update(L.episodes(a)[0])
        if logs is None:
            op = None
        else:
            T = {}
            for p in ([logs] if isinstance(logs, str) else logs) if not (isinstance(logs, list) and isinstance(logs[0], tuple)) else [None]:
                T.update(L.ticks(L.tick_pairs(p))[0] if p is not None else L.ticks(logs)[0])
            op = {s: v for s, v in ((s, M.opening_lock(R)) for s, R in T.items()) if v is not None}
        data.append((lab, E, op))
    common = sorted(set.intersection(*[set(E) for _, E, _ in data]))
    out = [f"common seeds n={len(common)}: {[s % 1000 for s in common]}"]
    for lab, E, op in data:
        o = "—" if op is None else f"{sum(op.get(s, False) for s in common)}/{sum(s in op for s in common)}"
        vals = " ".join(f"{np.nanmean([M.value(m, E[s]) for s in common]):9.{p}f}" for m, _, p in ac.M)
        out.append(f"{lab} {o} {vals} {np.mean([E[s]['steps'] for s in common]):8.0f}")
    out.append("paired Δ (95% seed-bootstrap CI; sign-flip p):")
    rng = np.random.default_rng(20261003)
    for (la, A, _), (lb, B, _) in [(data[i], data[i + 1]) for i in range(1, len(data) - 1)] + [(data[1], data[-1])]:
        parts = []
        for m, h, p in (ac.M[0], ac.M[3], ac.M[4]):
            d = np.array([M.value(m, B[s]) - M.value(m, A[s]) for s in common], float)
            mu, lo, hi = S.bootstrap_ci(d, rng)
            pv = S.sign_flip_p(d, rng)
            parts.append(f"{h} {mu:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}] p={pv:.3f}")
        out.append(f"  {la[:2]} → {lb[:2]}: " + " | ".join(parts))
    return "\n".join(out)


def test_arch_compare_numbers_reproduced():
    legacy = run(ac.main)
    ours = _arch_compare_report()
    assert numbers(legacy) == numbers(ours)


def test_opening_ds_numbers_reproduced():
    legacy = run(od.main)
    A = {}
    for s in od.ARM:
        A.update(L.episodes(s)[0])
    B = L.episodes(od.REF)[0]
    Sd = sorted(set(A) & set(B))

    def lock(stems):
        T = {}
        for st in stems:
            T.update(L.ticks(L.tick_pairs(st))[0])
        return {s: v for s, v in ((s, M.opening_lock(R)) for s, R in T.items()) if v is not None}
    oa, ob = lock(od.ARM), lock([od.REF])
    so = [s for s in Sd if s in oa and s in ob]
    out = [f"paired seeds n={len(Sd)} (attr arm has {len(A)}, H-A has {len(B)})",
           f"opening lock (t≥12): attr {sum(oa[s] for s in so)}/{len(so)} | head {sum(ob[s] for s in so)}/{len(so)}"]
    rng = np.random.default_rng(20261003)
    for m in ("correct", "longC", "longW", "correct_s"):
        a = np.array([M.value(m, A[s]) for s in Sd], float)
        b = np.array([M.value(m, B[s]) for s in Sd], float)
        d = a - b
        d = d[np.isfinite(d)]
        mu, lo, hi = S.bootstrap_ci(d, rng)
        pv = S.sign_flip_p(d, rng)
        out.append(f"{M.label(m)} attr {np.nanmean(a):.3f} | head {np.nanmean(b):.3f} | Δ {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={pv:.3f}")
    out.append(" ".join(f"{s % 1000}:{'✓' if oa.get(s) else '✗'}{'✓' if ob.get(s) else '✗'}"
                        f"({M.value('correct', A[s]):.2f}/{M.value('correct', B[s]):.2f})" for s in Sd))
    ours = "\n".join(out)
    assert numbers(legacy) == numbers(ours)


def test_iter_gate_arm_table_reproduced(tmp_path):
    js = tmp_path / "ig.json"
    run(g.main, "--json", str(js))
    import json
    res = json.load(open(js))
    rng = np.random.default_rng(20261003)                    # iter_gate's module RNG, consumed in main() order
    names = sorted({t[1] for t in g.TESTS} | {t[2] for t in g.TESTS})
    for n in names:
        E = L.episodes(n)[0]
        if not E:
            continue
        for k in g.METRICS:
            v = S.bootstrap_ci([M.value(k, e) for e in E.values()], rng)
            assert np.allclose(v, res["arms"][n][k], rtol=0, atol=1e-12), (n, k)
        assert res["arms"][n]["n"] == len(E)
