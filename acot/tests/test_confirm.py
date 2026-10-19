"""1006 §8 / G11: confirmatory statistics and decision rule."""
import numpy as np

from acot.analysis import confirm as C


def test_signflip_and_boot():
    rng = np.random.default_rng(0)
    d = np.full(40, 0.2) + rng.normal(0, 0.05, 40)
    assert C.signflip_p(d, rng, b=20000) < 0.001
    m, lo, hi = C.stratified_boot_ci(d, ["T5"] * 20 + ["T3"] * 20, rng, b=2000)
    assert lo < m < hi and lo > 0.1
    z = rng.normal(0, 1, 40)
    assert C.signflip_p(z - z.mean(), rng, b=5000) > 0.5           # exactly-zero mean → p ≈ 1


def test_stratified_boot_keeps_strata_sizes():
    rng = np.random.default_rng(1)
    d = np.array([1.0] * 10 + [0.0] * 30)                          # all variance between strata
    m, lo, hi = C.stratified_boot_ci(d, ["a"] * 10 + ["b"] * 30, rng, b=2000)
    assert lo == hi == m == 0.25                                    # within-stratum resampling → no variance


def _res(mean, p, tr_lo):
    return {"primary": {"mean": mean, "p": p}, "safety": {"ci": [tr_lo, tr_lo + 0.05]}}


def test_decide_holm_and_safety_gate():
    res = [_res(0.10, 0.001, 0.0), _res(0.08, 0.02, -0.05), _res(-0.05, 0.001, 0.01), _res(0.05, 0.04, 0.0)]
    C.decide(res, delta=0.02)
    assert [r["primary"]["p_holm"] for r in res] == [0.004, 0.04, 0.004, 0.04]
    assert [r["confirmed"] for r in res] == [True, False, False, True]   # 2nd fails TR90 gate; 3rd wrong sign
    assert res[1]["safety_ok"] is False
