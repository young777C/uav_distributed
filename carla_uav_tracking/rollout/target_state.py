"""Training-free target-state prior for out-of-frame predict-intercept.

Maintains the last-VISIBLE target world position + a finite-difference world
velocity, and dead-reckons the target's future position (② zero-velocity or
③ constant-velocity) into the CURRENT camera frame — the SAME (K,3)/scale
convention as EAR's waypoint (dataset_stage2.py: inverse_pose_matrix + /wp_scale).
The policy injects this into its z_ex slot during a loss (policy.py `--ex-source`)
so DiT flies toward the predicted position instead of EAR's blind guess (offline
EAR loss-frame FDE is 2-6x worse than this CV baseline; see memory
acot-uav-wm-roadgraph / transition-diary §8.6).

Dead-reckoning is done in the WORLD frame then reprojected through the current
pose => UAV self-motion is fully compensated and `v_last` is the target's ABSOLUTE
world velocity (not relative). This is the exact z_ex-during-loss injection point
the road-graph WM will later occupy (Scheme A); CV here is the baseline WM must
beat — it solves straight losses but collapses on turns/junctions.

Privilege note: `update()` is fed GT target world pos on VISIBLE ticks (same
privilege level as the harness's GT candidate boxes). During a LOSS the estimator
NEVER peeks — it dead-reckons from the last-seen state. So loss-frame prediction is
honest; only the last-seen anchor is GT-exact rather than detector-estimated (=
the CV upper bound / 档A). Swap update() to a bbox+depth back-projection for the
realizable 档C version.
"""
from __future__ import annotations

import numpy as np

from acot_probe.projection import inverse_pose_matrix


class TargetStateEstimator:
    def __init__(self, offsets_s, wp_scale, fps, conf_horizon_s: float = 4.0):
        self.off_s = np.asarray(offsets_s, np.float64)     # waypoint horizons (s)
        self.wp_scale = float(wp_scale)
        self.fps = float(fps)
        self.conf_h = float(conf_horizon_s)                # CV trusted up to ~here
        self.reset()

    def reset(self):
        self.p_last = None                                 # (3,) last-visible world pos
        self.v_last = np.zeros(3)                          # (3,) world velocity (finite diff)
        self.t_last = None                                 # step of last visible obs

    @property
    def seen(self) -> bool:
        return self.p_last is not None

    def update(self, tpos_world, step: int):
        """Call on VISIBLE ticks with the observed target world position."""
        p = np.asarray(tpos_world, np.float64)
        if self.p_last is not None and self.t_last is not None and step > self.t_last:
            dt = (step - self.t_last) / self.fps
            if dt > 1e-3:
                self.v_last = (p - self.p_last) / dt
        self.p_last = p
        self.t_last = step

    def loss_seconds(self, step: int) -> float:
        return 0.0 if self.t_last is None else max(0.0, (step - self.t_last) / self.fps)

    def confidence(self, step: int) -> float:
        """1 at last-seen → 0 once the loss exceeds conf_horizon (widen search past this)."""
        return max(0.0, 1.0 - self.loss_seconds(step) / self.conf_h) if self.seen else 0.0

    def predict_wp(self, campose, step: int, mode: str = "cv"):
        """(K,3) cam-frame /wp_scale waypoint prior, or None if target never seen.

        campose = (cx,cy,cz,pitch,yaw). mode: 'cv' (const-vel) | 'zerovel' (frozen).
        """
        if not self.seen:
            return None
        if mode == "zerovel":
            pw = np.repeat(self.p_last[None], len(self.off_s), axis=0)          # (K,3) frozen
        else:                                                                    # cv
            hor = self.loss_seconds(step) + self.off_s                          # (K,) since last-seen
            pw = self.p_last[None] + self.v_last[None] * hor[:, None]           # (K,3) world
        return self.project_world(campose, pw)

    def project_world(self, campose, pw):
        """Project (K,3) WORLD points → (K,3) cam-frame /wp_scale (EAR waypoint convention).

        The road-graph WM (P2a-1) traverses the lane graph env-side and ships WORLD (K,3);
        the policy reuses this exact reprojection to fill z_ex, so road and cv priors land
        in the identical frame/scale — the only difference is straight-vs-lane-follow."""
        pw = np.asarray(pw, np.float64)
        M = inverse_pose_matrix(*campose)                                       # world -> cam
        homog = np.concatenate([pw, np.ones((pw.shape[0], 1))], -1)
        return ((homog @ M.T)[:, :3] / self.wp_scale).astype(np.float32)
