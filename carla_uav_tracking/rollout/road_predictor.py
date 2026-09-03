"""Env-side road-graph target predictor for closed-loop predict-intercept (P2a-1).

The DEPLOYABLE counterpart of the offline `annotation/road_pred` (M0 spec): instead
of straight-line const-velocity dead-reckoning (rollout.target_state, the CV prior),
snap the LAST-SEEN target to its lane and TRAVERSE the lane graph by arc =
speed*(loss_elapsed + horizon), picking the straightest branch at junctions. This
follows the curving road instead of flying off it — the road-graph WM's value region
is precisely turns/junctions (offline M2 gate: road-FDE << cv_frame-FDE there).

Lives ENV-SIDE (cyh-carla, py3.6) because the CARLA `carla.Map` lane graph is only
here; the (K,3) WORLD prediction is shipped over the wire (as gt_candidates[3]) and
the policy projects it into the current camera frame → z_ex (same slot as the CV
prior). torch-free.

PRIVILEGE / honesty: `update()` is fed the GT target world pos+heading on VISIBLE
ticks (same privilege as the harness GT candidate boxes and the CV estimator). During
a LOSS it NEVER peeks — it traverses from the last-seen anchor only. So this is the
deployable road prediction (from last-seen), NOT the offline oracle-current upper
bound. Difference vs the CV prior is EXACTLY straight-vs-lane-follow (same anchor,
same speed) → a clean closed-loop isolation of the road-graph value.
"""
from __future__ import annotations

import numpy as np


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class RoadPredictor:
    def __init__(self, carla_map, offsets_s, fps, step_m: float = 2.0):
        self._map = carla_map
        self.off_s = [float(o) for o in offsets_s]     # [1,2,4,6] s
        self.fps = float(fps)
        self.step_m = float(step_m)                    # lane-traversal granularity (m)
        self.reset()

    def reset(self):
        self.p_last = None                             # (3,) last-visible world pos
        self.speed_last = 0.0                          # m/s at last-visible
        self.yaw_last = 0.0                            # deg heading at last-visible
        self.t_last = None

    @property
    def seen(self) -> bool:
        return self.p_last is not None

    def update(self, tpos_world, tvel_world, tyaw_deg, step: int):
        """Call on VISIBLE ticks with the observed target world pos / velocity / heading."""
        self.p_last = np.asarray(tpos_world, np.float64)
        v = np.asarray(tvel_world, np.float64)
        self.speed_last = float(np.linalg.norm(v))
        self.yaw_last = float(tyaw_deg)
        self.t_last = int(step)

    def _pick_straightest(self, nxts, ref_yaw):
        best, bad = nxts[0], 1e9
        for wp in nxts:
            d = abs(_wrap180(wp.transform.rotation.yaw - ref_yaw))
            if d < bad:
                bad, best = d, wp
        return best

    def predict(self, step: int):
        """(K,3) WORLD lane-traversal prediction from the last-seen anchor, or None.

        arc = speed * (loss_elapsed + horizon); junction branch = straightest vs the
        running heading. SINGLE-PASS to the max horizon, sampling the K cumulative-arc
        thresholds along the way (off_s is increasing) → ~4x fewer carla.next() RPCs
        than re-traversing per horizon. Mirrors the offline M0 algorithm, anchored at
        last-seen (or, in the oracle positive-control, the true current pos).
        """
        if self.p_last is None:
            return None
        import carla
        loss_s = 0.0 if self.t_last is None else max(0.0, (step - self.t_last) / self.fps)
        loc = carla.Location(x=float(self.p_last[0]), y=float(self.p_last[1]), z=float(self.p_last[2]))
        wp0 = self._map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None:                                                # off-road → freeze (lane unknown)
            return np.repeat(self.p_last[None], len(self.off_s), axis=0).astype(np.float64)
        arcs = [self.speed_last * (loss_s + off) for off in self.off_s]  # increasing cumulative arcs
        out = [None] * len(arcs)
        cur, ref_yaw, travelled, ai = wp0, self.yaw_last, 0.0, 0
        while ai < len(arcs) and arcs[ai] <= travelled + 1e-9:         # zero-length horizons → at wp0
            lc = cur.transform.location; out[ai] = [lc.x, lc.y, lc.z]; ai += 1
        while ai < len(arcs):
            d = min(self.step_m, arcs[ai] - travelled)
            nxts = cur.next(d)
            if not nxts:                                              # dead-end → freeze remaining at cur
                lc = cur.transform.location
                for j in range(ai, len(arcs)):
                    out[j] = [lc.x, lc.y, lc.z]
                break
            cur = self._pick_straightest(nxts, ref_yaw)
            ref_yaw = cur.transform.rotation.yaw                      # follow the road's bend
            travelled += d
            while ai < len(arcs) and arcs[ai] <= travelled + 1e-9:    # crossed threshold(s) → record
                lc = cur.transform.location; out[ai] = [lc.x, lc.y, lc.z]; ai += 1
        return np.asarray(out, np.float64)
