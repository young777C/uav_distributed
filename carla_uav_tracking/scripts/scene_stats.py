"""1026 per-episode scene statistics of a recorded scene probe (no carla import: also used on the host for the
stress_v1 reference)."""
import math

import numpy as np


def scene_stats(tpos, tvis, tin, clones, others, fps=10):
    """per-episode scene statistics (same turn / stop definitions as the 10-25 town scan of stress_v1):
    tpos (N,3) target; tvis[i] ray-visible; tin[i] inside the frame; clones / others: per frame list of (x, y, speed,
    visible)."""
    p = np.asarray(tpos, dtype=float)[:, :2]
    out = {"frames": int(len(p))}
    if len(p) < 30:
        return out
    v = np.diff(p, axis=0) * fps
    sp = np.linalg.norm(v, axis=1)
    mov = sp > 1.0
    hd = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turns, i, n = 0, 0, len(hd)
    while i < n - 50:                                   # ≥ 45° heading change within 5 s while moving, once per event
        if mov[i] and mov[i + 50] and abs(hd[i + 50] - hd[i]) >= math.radians(45):
            turns += 1
            i += 50
        else:
            i += 1
    stops, c = 0, 0
    for m in ~mov:
        c = c + 1 if m else 0
        if c == 20:
            stops += 1                                  # stationary ≥ 2 s
    tvis, tin = np.asarray(tvis, bool), np.asarray(tin, bool)
    occ = tin & ~tvis                                   # in the frame but hidden (structure / vehicle in front)
    lost = ~tvis                                        # not visible for any reason (incl. E3 look-away, out of frame)

    def runs(mask, k):
        r, c = 0, 0
        for x in mask:
            c = c + 1 if x else 0
            if c == k:
                r += 1
        return r

    def still(rows):
        xs = [s < 0.5 for fr in rows for (_x, _y, s, _v) in fr]
        return float(np.mean(xs)) if xs else None

    cov = [any(vv for (_x, _y, _s, vv) in fr) for fr, ok in zip(clones, tvis) if ok]
    out.update({
        "still_frac": round(float(1 - mov.mean()), 3), "stops_2s": stops, "turns_45": turns,
        "speed_moving": round(float(sp[mov].mean()), 2) if mov.any() else 0.0,
        "occ_frac_inframe": round(float(occ.sum() / max(tin.sum(), 1)), 3),
        "occ_runs_1s": runs(occ, 10), "occ_runs_2s": runs(occ, 20),
        "lost_frac": round(float(lost.mean()), 3), "lost_runs_2s": runs(lost, 20),
        "clone_still_frac": None if still(clones) is None else round(still(clones), 3),
        "other_still_frac": None if still(others) is None else round(still(others), 3),
        "clone_covis_frac": round(float(np.mean(cov)), 3) if cov else None,
    })
    return out
