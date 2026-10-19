"""Simulated power for paired arm comparisons under the BIMODAL episode outcome observed in CARLA closed loop
(1004_ablation_design.md §3.5). Per seed: opening correct ~ Bernoulli(p); open-right → hold (U[0.9,1]) w.p. h else
drift (U[0,0.6]); open-wrong → recover (U[0.3,0.9]) w.p. r else stay (U[0,0.1]). Full: h=0.95, r=0.05; the ablated arm
lowers h so that the total mean difference is Δ. Natural opening: each arm flips the reference opening independently
w.p. `flip`. Hold events of the two arms are independent within a seed (conservative). Test = the pre-registered
paired SIGN-FLIP permutation test, via its large-sample form z = Σd / sqrt(Σd²) (the permutation null of Σ±d_i has
variance Σd_i²), two-sided at α (0.025 / 0.0125 = strictest Holm step for 2 / 4 comparisons).
`--sens` prints the pessimistic sensitivity grid (p, flip, h_full).
  python power_sim.py"""
import numpy as np

RNG = np.random.default_rng(0)
Z = {0.05: 1.96, 0.025: 2.24, 0.0125: 2.50}


def arm(open_ok, h, r):
    n = len(open_ok); u = RNG.random(n)
    hold, drift = RNG.uniform(0.9, 1.0, n), RNG.uniform(0.0, 0.6, n)
    rec, stay = RNG.uniform(0.3, 0.9, n), RNG.uniform(0.0, 0.1, n)
    return np.where(open_ok, np.where(u < h, hold, drift), np.where(u < r, rec, stay))


def power(n, dtot, natural, alpha, p=0.55, h_full=0.95, r=0.05, flip=0.10, sims=20000):
    mean_right = lambda h: h * 0.95 + (1 - h) * 0.3
    h_abl = (mean_right(h_full) - dtot / p - 0.3) / 0.65
    hits = 0
    for _ in range(sims):
        o = RNG.random(n) < p
        o_f = o ^ (RNG.random(n) < flip) if natural else o
        o_a = o ^ (RNG.random(n) < flip) if natural else o
        d = arm(o_f, h_full, r) - arm(o_a, h_abl, r)
        hits += d.sum() / np.sqrt((d ** 2).sum() + 1e-12) > Z[alpha]
    return hits / sims


if __name__ == "__main__":
    import sys
    if "--sims" in sys.argv:   # override the per-cell simulation count (default 20000, as in 1004 §3.5)
        _S = int(sys.argv[sys.argv.index("--sims") + 1])
        power.__defaults__ = power.__defaults__[:-1] + (_S,)
    if "--sens" in sys.argv:
        print("setting  n   Δ     p     flip  h_full  power(α)")
        for nat, n, a in ((True, 80, 0.025), (False, 60, 0.0125)):
            for dt in (0.15, 0.20):
                for p in (0.45, 0.55):
                    for fl in ((0.10, 0.15) if nat else (0.0,)):
                        for hf in (0.85, 0.95):
                            print(f"{'natural' if nat else 'shared '} {n:3d} {dt:.2f}  {p:.2f}  {fl:.2f}  {hf:.2f}   "
                                  f"{power(n, dt, nat, a, p=p, h_full=hf, flip=fl):.2f} (α={a})")
        raise SystemExit
    print("n   Δ     setting   α=0.05  α=0.025  α=0.0125")
    for n in (40, 60, 80):
        for dt in (0.10, 0.15, 0.20):
            for nat in (False, True):
                print(f"{n:3d} {dt:.2f}  {'natural' if nat else 'shared '}   "
                      + "  ".join(f"{power(n, dt, nat, a):.2f}" for a in (0.05, 0.025, 0.0125)))
