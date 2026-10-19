"""The ONE implementation of the statistics (framework §12.3): seed-bootstrap CI of a mean, sign-flip permutation p,
Wilson interval, exact McNemar, Holm. Every function takes the numpy Generator explicitly; reports derive it from the
experiment / comparison name (rng_for), so every number is reproducible."""
import hashlib
import math

import numpy as np

B = 20000


def rng_for(*keys):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()[:16], 16))


def bootstrap_ci(x, rng, b=B):
    """mean and 95% percentile CI over seeds (non-finite values dropped)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = x[rng.integers(0, len(x), (b, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def sign_flip_p(d, rng, b=B):
    """two-sided paired permutation p (random sign flips of the per-seed differences)."""
    d = np.asarray(d, float)
    return float((np.abs((d[None] * rng.choice([-1, 1], (b, len(d)))).mean(1)) >= abs(d.mean())).mean())


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    """exact two-sided McNemar p for discordant counts b (A✓B✗) and c (A✗B✓)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(pvals):
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i])
        adj[i] = min(1.0, run)
    return adj
