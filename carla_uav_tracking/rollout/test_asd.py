"""Unit test: ASD fires on synthetic confident drift, stays silent on a stable correct track."""
import numpy as np
from asd import ASD

def make(rng, n=5, d=32):
    base = rng.normal(size=(n, d)); base /= np.linalg.norm(base, axis=1, keepdims=True)
    # look-alikes: candidates 1..n-1 = target + perturbation (cos ~0.6-0.8)
    for i in range(1, n):
        v = base[0] + 0.9 * base[i]; base[i] = v / np.linalg.norm(v)
    return base

def frame(rng, base):
    f = base + 0.05 * rng.normal(size=base.shape)
    return f / np.linalg.norm(f, axis=1, keepdims=True)

def run(drift_at, ticks=60, seed=0, mode="rule"):
    rng = np.random.default_rng(seed); base = make(rng); a = ASD(mode=mode)
    fires = []
    for t in range(ticks):
        mem = 0 if (drift_at is None or t < drift_at) else 2          # memory confidently switches to look-alike 2
        lang = np.array([1.0, 0.2, 0.3, 0.1, 0.0]) + 0.5 * rng.normal(size=5)   # weak, noisy, unbiased spec signal
        o = a.step(frame(rng, base), list(range(5)), mem, mem_conf=0.9, lang=lang, true_slot=0, t=t)
        if o["fire"]:
            fires.append((t, o["slot"]))
    return fires

S = 30
fa = sum(len(run(None, ticks=150, seed=s)) for s in range(S))              # false alarms on stable tracks
det = [run(20, seed=s) for s in range(S)]
hit = [d for d in det if d and d[0][1] == 0]
delay = np.mean([d[0][0] - 20 for d in hit]) if hit else float("nan")
print(f"FA = {fa} over {S}x150 stable ticks | detect {len(hit)}/{S} (correct slot) | mean delay {delay:.1f} ticks")
assert fa <= 2 and len(hit) >= 0.9 * S and delay <= 4
