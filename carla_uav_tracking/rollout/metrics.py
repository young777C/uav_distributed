"""RolloutScorer — score a closed-loop episode from GT, per guide §6 / design §7.2.

The controller never sees GT; the scorer does, every tick. Key rule (§3.4 族B, §7):
closed-loop mis-follow is NOT per-frame argmax (that punishes single-frame flicker
from flow-matching stochasticity). We maintain a COMMITTED target with K-frame
hysteresis; a new candidate only takes over after winning K consecutive ticks.

Reported:
  mis_follow_inst      per-tick argmax≠target over target-present ticks (diagnostic, strict)
  mis_follow_sustained committed≠target over target-present ticks (main; == MOT ID-switch)
  id_switches          # times the committed target flips to a wrong car (held ≥K)
  success              reached episode end with no committed-wrong run ≥ T_fail_s and
                       never lost the target past truncation
  track_frames         ticks the target stayed in frame
  centering_rate       fraction of in-frame ticks with the target in the central region
  reacquire_*          convergence time (delay OK) vs lock-wrong rate (hard failure, §7)
  shortcut_*           "always nearest car" / "target dead-center" occupancy (捷径 checks, §7)

The scorer maps the policy's predicted candidate SLOT → identity via the shuffled
CandidateSet (cands[slot].is_target / .idx); it does not read any slot ordering
into the control path.
"""

from __future__ import annotations

import numpy as np


def _identity_key(cand) -> str:
    return "target" if cand.is_target else f"distractor_{cand.idx}"


class RolloutScorer:
    def __init__(self, fps: int = 10, commit_k: int = 5, t_fail_s: float = 2.0,
                 max_lost_s: float = 5.0, central_frac: float = 0.3):
        self.fps = fps
        self.commit_k = commit_k
        self.t_fail = int(t_fail_s * fps)
        self.max_lost = int(max_lost_s * fps)
        self.central_frac = central_frac

    def reset(self, metadata: dict | None = None):
        self.meta = metadata or {}
        self.n = 0
        # committed-target hysteresis
        self._committed = None
        self._challenger = None
        self._chal_count = 0
        self._id_switches = 0
        self._wrong_run = 0          # consecutive committed-wrong ticks (target present)
        self._max_wrong_run = 0
        self._off_run = 0            # consecutive target-off-screen ticks
        self._failed = False
        # counters
        self._present = 0            # target-present (in-frame) ticks
        self._inst_wrong = 0
        self._sustain_wrong = 0
        self._central = 0
        self._nearest_pred = 0       # ticks the pred == nearest candidate (shortcut)
        self._nearest_denom = 0
        self._central_target = 0     # ticks target is dead-center (shortcut)
        self._dist_sum = 0.0
        self._dist_n = 0
        # per-frame grounding stratifiers (why is online mis_follow > offline?):
        self._byN = {}          # N candidates -> [wrong, total]
        self._byC = {"central": [0, 0], "off": [0, 0]}   # target central vs off-center
        self._byD = {}          # target depth bucket -> [wrong, total]
        # re-acquisition tracking: EVERY target-loss is one attempt, resolved on
        # recovery (success) or left as timeout at episode end (no-WM baseline needs
        # the failures, not just the successes).
        self._was_lost = False
        self._reacq_open = False
        self._reacq_start = 0
        self._reacq_wrong_counted = False
        self._reacq_attempts = 0
        self._reacq_success = 0      # re-locked the TRUE target
        self._reacq_wrong = 0        # committed to a WRONG car during recovery (≥1 tick)
        self._reacq_conv = []        # convergence frames, successful re-acquisitions only

    # ------------------------------------------------------------------
    def update(self, *, step: int, target_present: bool, target_central: bool,
               in_loss_window: bool, info, uav_target_dist: float | None = None):
        """info: policy.ActInfo (cand_set + pred_slot). target_present: target in-frame."""
        self.n = step + 1
        cset = getattr(info, "cand_set", None)
        pred_slot = getattr(info, "pred_slot", -1)
        pred_cand = cset.cands[pred_slot] if (cset is not None and 0 <= pred_slot < len(cset.cands)) else None
        pred_key = _identity_key(pred_cand) if pred_cand is not None else None

        # --- hysteresis commit ---
        if pred_key is not None:
            if self._committed is None:
                self._committed = pred_key
            elif pred_key == self._committed:
                self._challenger, self._chal_count = None, 0
            else:
                if pred_key == self._challenger:
                    self._chal_count += 1
                else:
                    self._challenger, self._chal_count = pred_key, 1
                if self._chal_count >= self.commit_k:
                    if self._committed == "target" and pred_key != "target":
                        self._id_switches += 1
                    self._committed = pred_key
                    self._challenger, self._chal_count = None, 0

        # --- off-screen / truncation bookkeeping ---
        if target_present:
            self._off_run = 0
        else:
            self._off_run += 1

        # --- mis-follow accounting (only over target-present ticks) ---
        if target_present:
            self._present += 1
            if target_central:
                self._central += 1
                self._central_target += 1
            if pred_cand is not None and not pred_cand.is_target:
                self._inst_wrong += 1
            # per-frame stratified mis_follow (matches offline argmax≠target over target-present)
            ti = getattr(cset, "true_idx", -1) if cset is not None else -1
            if cset is not None and ti >= 0:
                N = len(cset.cands)
                wrong = int(pred_slot != ti)
                nb = min(N, 6)
                b = self._byN.setdefault(nb, [0, 0]); b[0] += wrong; b[1] += 1
                ck = "central" if target_central else "off"
                self._byC[ck][0] += wrong; self._byC[ck][1] += 1
                td = float(cset.cands[ti].depth)
                db = "d<30" if td < 30 else ("d30-60" if td < 60 else "d>=60")
                bd = self._byD.setdefault(db, [0, 0]); bd[0] += wrong; bd[1] += 1
            committed_wrong = self._committed is not None and self._committed != "target"
            if committed_wrong:
                self._sustain_wrong += 1
                self._wrong_run += 1
                self._max_wrong_run = max(self._max_wrong_run, self._wrong_run)
                if self._wrong_run >= self.t_fail:
                    self._failed = True
            else:
                self._wrong_run = 0
            # shortcut: does pred coincide with the nearest (smallest-depth) candidate?
            if cset is not None and len(cset.cands) > 1 and pred_cand is not None:
                nearest = min(range(len(cset.cands)), key=lambda i: cset.cands[i].depth)
                self._nearest_pred += int(pred_slot == nearest)
                self._nearest_denom += 1

        if uav_target_dist is not None:
            self._dist_sum += float(uav_target_dist); self._dist_n += 1

        # --- re-acquisition: each target-loss is one attempt, resolved on recovery ---
        lost_now = (not target_present) or in_loss_window
        if lost_now and not self._was_lost:            # a loss STARTS → open a new attempt
            self._reacq_open = True
            self._reacq_start = step
            self._reacq_wrong_counted = False
            self._reacq_attempts += 1
        if self._reacq_open:
            if (not self._reacq_wrong_counted and target_present
                    and self._committed is not None and self._committed != "target"):
                self._reacq_wrong += 1                  # locked a wrong car while recovering (once/attempt)
                self._reacq_wrong_counted = True
            if (not lost_now) and target_present and self._committed == "target":
                self._reacq_success += 1                # correctly re-acquired the TRUE target
                self._reacq_conv.append(step - self._reacq_start)
                self._reacq_open = False                # attempt resolved
        self._was_lost = lost_now

    def should_truncate(self) -> bool:
        return self._off_run >= self.max_lost

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        present = max(self._present, 1)
        att = self._reacq_attempts
        conv = self._reacq_conv
        success = (not self._failed) and (self._off_run < self.max_lost)
        return {
            "success": bool(success),
            "mis_follow_inst": self._inst_wrong / present,
            "mis_follow_sustained": self._sustain_wrong / present,
            "id_switches": self._id_switches,
            "track_frames": self._present,
            "track_seconds": self._present / self.fps,
            "centering_rate": self._central / present,
            "max_wrong_run_s": self._max_wrong_run / self.fps,
            "mean_uav_target_dist": (self._dist_sum / self._dist_n) if self._dist_n else None,
            # re-acquisition (guide §7): each target-loss = one attempt (deliberate loss
            # window OR model-caused loss). success = re-locked TRUE target; the no-WM
            # baseline's failure modes are timeout (never re-acquired) + lock-wrong.
            "reacquire_attempts": att,
            "reacquire_success_rate": (self._reacq_success / att) if att else None,
            "reacquire_timeout_rate": ((att - self._reacq_success) / att) if att else None,
            "reacquire_lock_wrong_rate": (self._reacq_wrong / att) if att else None,
            "reacquire_convergence_s_mean": (float(np.mean(conv)) / self.fps) if conv else None,
            # shortcut checks (§7): high ⇒ a shortcut survived, re-harden data (§2.3)
            "shortcut_nearest_rate": (self._nearest_pred / self._nearest_denom) if self._nearest_denom else None,
            "shortcut_target_central_rate": self._central_target / present,
            # per-frame grounding stratifiers (online-vs-offline diagnosis)
            "mis_by_N": {k: v[0] / v[1] for k, v in self._byN.items() if v[1]},
            "n_dist": {k: v[1] for k, v in self._byN.items()},
            "mis_by_central": {k: (v[0] / v[1] if v[1] else None) for k, v in self._byC.items()},
            "c_dist": {k: v[1] for k, v in self._byC.items()},
            "mis_by_depth": {k: v[0] / v[1] for k, v in self._byD.items() if v[1]},
            "d_dist": {k: v[1] for k, v in self._byD.items()},
            "steps": self.n,
        }
