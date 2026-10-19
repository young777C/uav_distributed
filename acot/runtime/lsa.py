"""Rectangular linear sum assignment (Hungarian, shortest augmenting path with potentials) in numpy — the policy image
deliberately has no scipy (framework §20), so perception.tracker.assign = hungarian needs its own solver. Same contract
as scipy.optimize.linear_sum_assignment(C): → (rows, cols) of a minimum-cost matching of size min(n, m), rows sorted.
O(n²·m); the tracker's matrices are ≤ ~15 × 15."""
import numpy as np


def linear_sum_assignment(cost):
    C = np.asarray(cost, np.float64)
    if C.ndim != 2:
        raise ValueError("cost must be 2-D")
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, int)                 # p[j] = row (1-based) matched to column j; column 0 is the virtual root
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows = np.array([p[j] - 1 for j in range(1, m + 1) if p[j] > 0], int)
    cols = np.array([j - 1 for j in range(1, m + 1) if p[j] > 0], int)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows, kind="stable")
    return rows[order], cols[order]
