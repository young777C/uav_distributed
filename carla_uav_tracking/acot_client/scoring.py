"""Policy-commitment metrics (1006 §4; py3.8 inside cyh-carla). RolloutScorer scores the per-frame OUTPUT identity
(`pred_slot`) after its own 5-FRAME hysteresis (0.5 s); the identity the drone actually follows is the policy's internal
commitment (updated only at S2 ticks: conf ≥ conf_tau and a challenger for K = 5 S2 ticks ≈ 3 s, or a direct PGIF
re-commit). AcotScorer additionally scores that commitment, sent by the acot server with every reply as
`info.committed_key` (a tracker id, None before the first commitment):

  key → actor   the key's candidate in the current candidate set (the latest S2 tick) gives the true actor; a key with
                no box this tick keeps the actor it was last matched to (None if never matched)
  pc_*          over target-present frames: correct = the committed actor is the target; an absent commitment counts as
                NOT correct (conservative, identical for all arms); longest correct / wrong runs reset when the target
                leaves the frame (correct) or the commitment is correct / absent (wrong)
  tr90          correct-commitment frames / the episode horizon (frames after a truncation count as not tracked)
  truncated     the episode ended because the target was lost for max_lost_s (SR(τ) needs it)
  frame_state   (1016 §3 step 0; logging only, no metric reads it) one character per scored step, index = step:
                v = in frame and ray-cast visible, o = in frame but occluded, p = in frame (visibility not measured),
                x = out of frame (off-scene), - = a step the scorer did not see
Legacy runtimes send no `committed` → every pc_* field is None."""
from rollout.metrics import RolloutScorer, _identity_key


class AcotScorer(RolloutScorer):
    def __init__(self, *args, horizon_frames=900, **kw):
        super().__init__(*args, **kw)
        self.horizon = int(horizon_frames)

    def reset(self, metadata=None):
        super().reset(metadata)
        self._pc_seen = False            # did the policy ever send a `committed` field
        self._pc_actor_of = {}           # tracker key → last matched actor ("target" / "distractor_<gt>")
        self._pc_correct = 0
        self._pc_wrong = 0
        self._pc_none = 0
        self._pc_correct_run = self._pc_max_correct = 0
        self._pc_wrong_run = self._pc_max_wrong = 0
        self._pc_prev_actor = None
        self._pc_switches = 0            # committed actor changes target → other (while committed)
        self._fs = []                    # frame_state characters, index = step

    def _pc_actor(self, info):
        key = getattr(info, "committed_key", None)
        cset = getattr(info, "cand_set", None)
        if cset is not None:
            for c in cset.cands:
                self._pc_actor_of[int(c.idx)] = _identity_key(c)
        if key is None:
            return None
        return self._pc_actor_of.get(int(key))

    def update(self, *, step, target_present, target_central, in_loss_window, info, uav_target_dist=None,
               target_visible=None):
        super().update(step=step, target_present=target_present, target_central=target_central,
                       in_loss_window=in_loss_window, info=info, uav_target_dist=uav_target_dist,
                       target_visible=target_visible)
        step = int(step)
        if step >= 0:
            while len(self._fs) <= step:
                self._fs.append("-")
            self._fs[step] = ("x" if not target_present else "p" if target_visible is None
                              else "v" if target_visible else "o")
        if not hasattr(info, "committed_key"):
            return
        self._pc_seen = True
        actor = self._pc_actor(info)
        if self._pc_prev_actor == "target" and actor is not None and actor != "target":
            self._pc_switches += 1
        if actor is not None:
            self._pc_prev_actor = actor
        if not target_present:
            self._pc_correct_run = 0
            self._pc_wrong_run = 0
            return
        if actor == "target":
            self._pc_correct += 1
            self._pc_correct_run += 1
            self._pc_max_correct = max(self._pc_max_correct, self._pc_correct_run)
            self._pc_wrong_run = 0
        elif actor is None:
            self._pc_none += 1
            self._pc_correct_run = 0
            self._pc_wrong_run = 0
        else:
            self._pc_wrong += 1
            self._pc_wrong_run += 1
            self._pc_max_wrong = max(self._pc_max_wrong, self._pc_wrong_run)
            self._pc_correct_run = 0

    def finalize(self):
        out = super().finalize()
        out["frame_state"] = "".join(self._fs)
        present = self._present
        truncated = self._off_run >= self.max_lost
        if not self._pc_seen:
            out.update(pc_correct=None, pc_correct_frames=None, pc_wrong_frames=None, pc_none_frames=None,
                       pc_max_correct_run_s=None, pc_max_wrong_run_s=None, pc_switches=None, tr90=None,
                       truncated=bool(truncated), horizon_frames=self.horizon)
            return out
        out.update(pc_correct=(self._pc_correct / present) if present else None,
                   pc_correct_frames=self._pc_correct, pc_wrong_frames=self._pc_wrong, pc_none_frames=self._pc_none,
                   pc_max_correct_run_s=self._pc_max_correct / self.fps, pc_max_wrong_run_s=self._pc_max_wrong / self.fps,
                   pc_switches=self._pc_switches, tr90=self._pc_correct / float(self.horizon),
                   truncated=bool(truncated), horizon_frames=self.horizon)
        return out
