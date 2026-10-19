"""Pre-writing statistical rigor (data-completeness #1): Wilson 95% CIs for SR, and PAIRED
same-seed significance tests for TAHRelM vs each key baseline — on SR (McNemar exact) and on the
continuous metrics (paired bootstrap on the per-seed difference). No CARLA; reuses eval_out JSONs.

Why: SR is binary over n=19 seeds → point estimates are noisy; a headline needs CIs + paired tests
(same seeds → paired, not independent). Continuous metrics (track_correct, max_wrong) are per-frame
aggregates → far less noisy; paired bootstrap gives their difference CIs.
"""
import json
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "eval_out")
ARMS = {
    "TAHRelM":  "paper_ext_tahrelm_gt",
    "borrow#1": "paper_ext_reidmotion03",
    "DeepSORT": "paper_ext_assoc_deepsort",
    "DAM4SAM":  "paper_ext_dam4sam",
    "OC-SORT":  "paper_ext_assoc_motion",
    "l5":       "paper_l5_dino_conf_tavg",
    "xattn":    "paper_l1_xattn",
}
# deterministic bootstrap RNG seed passed in (no Math.random / Date in this env is fine — numpy ok on host)
RNG = np.random.default_rng(20260916)


def load(stem):
    d = json.load(open(os.path.join(ROOT, stem + ".json")))
    return {int(e["episode_seed"]): e for e in d["episodes"] if "mis_follow_inst" in e}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(a_win, b_win):
    """exact two-sided binomial p on discordant pairs (a=1,b=0) vs (a=0,b=1)."""
    from math import comb
    b, c = a_win, b_win
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def boot_diff(x, y, B=20000):
    """paired bootstrap 95% CI of mean(x-y) over seeds."""
    d = x - y
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    data = {k: load(v) for k, v in ARMS.items()}
    common = sorted(set.intersection(*[set(v) for v in data.values()]))
    n = len(common)
    print(f"paired seeds n={n}\n")

    print("== SR (latch) with Wilson 95% CI ==")
    print(f"{'arm':<10}{'SR':>7}{'k/n':>8}{'95% CI':>18}")
    for k in ARMS:
        sr = np.array([1 if data[k][s]["success"] else 0 for s in common])
        p, lo, hi = wilson(sr.sum(), n)
        print(f"{k:<10}{p:>7.3f}{str(int(sr.sum()))+'/'+str(n):>8}   [{lo:.3f}, {hi:.3f}]")
    print("  ⚠ n=19 → CIs are WIDE (±~0.20); SR alone is a fragile headline.\n")

    base = "TAHRelM"
    print(f"== PAIRED tests: {base} vs each baseline (same {n} seeds) ==")
    st = np.array([1 if data[base][s]["success"] else 0 for s in common])
    tc = np.array([1 - data[base][s]["mis_follow_inst"] for s in common])
    mw = np.array([data[base][s]["max_wrong_run_s"] for s in common])
    for k in ARMS:
        if k == base:
            continue
        sb = np.array([1 if data[k][s]["success"] else 0 for s in common])
        a_win = int(((st == 1) & (sb == 0)).sum()); b_win = int(((st == 0) & (sb == 1)).sum())
        pmc = mcnemar_exact(a_win, b_win)
        tcb = np.array([1 - data[k][s]["mis_follow_inst"] for s in common])
        mwb = np.array([data[k][s]["max_wrong_run_s"] for s in common])
        dtc, lo_tc, hi_tc = boot_diff(tc, tcb)
        dmw, lo_mw, hi_mw = boot_diff(mw, mwb)
        sig_tc = "*" if (lo_tc > 0 or hi_tc < 0) else "ns"
        sig_mw = "*" if (lo_mw > 0 or hi_mw < 0) else "ns"
        print(f"\n  {base} vs {k}:")
        print(f"    SR McNemar: win/lose discordant = {a_win}/{b_win}, exact p={pmc:.3f}"
              + (" (SR too sparse to separate)" if pmc > 0.05 else " *"))
        print(f"    Δtrack_correct = {dtc:+.3f}  95%CI [{lo_tc:+.3f},{hi_tc:+.3f}]  {sig_tc}")
        print(f"    Δmax_wrong_s   = {dmw:+.2f}   95%CI [{lo_mw:+.2f},{hi_mw:+.2f}]  {sig_mw}")
    print("\n[* = 95% CI excludes 0 (significant); ns = not separable at n=19]")
    print("READOUT: continuous metrics (track_correct/max_wrong) separate arms far better than binary SR")
    print("→ headline should LEAD with continuous metrics + paired CIs; SR as a conservative operating point.")


if __name__ == "__main__":
    main()
