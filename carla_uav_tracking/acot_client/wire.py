"""RemotePolicy for the acot client (framework §7.3 P1, §8.4): blanks the GT wire fields a non-diagnostic arm must not
receive, and sends the episode seed / arm / config hash with every reset (the legacy server ignores the extra keys)."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from rollout.bridge import recv_msg, send_msg
from rollout.remote_policy import RemotePolicy


class AcotRemotePolicy(RemotePolicy):
    def __init__(self, host, port, blank, meta, timeout=120.0):
        super().__init__(host, port, timeout=timeout)
        self.blank = dict(blank)          # {"target_attrs": bool, "road": bool, "retarget": bool}
        self.meta = dict(meta)            # arm, config_hash, experiment; "seed" is set per episode

    def reset(self):
        msg = {"cmd": "reset"}
        msg.update(self.meta)
        send_msg(self.sock, msg)
        recv_msg(self.sock)

    def act(self, obs, gt_candidates):
        if self.blank.get("target_attrs"):
            obs = replace(obs, target_color="", target_bp="")
        if self.blank.get("retarget"):
            obs = replace(obs, retarget_event=False)
        if self.blank.get("road") and len(gt_candidates) > 3 and gt_candidates[3] is not None:
            gt_candidates = tuple(gt_candidates[:3]) + (None,) + tuple(gt_candidates[4:])
        return self._exchange(obs, gt_candidates)

    def _exchange(self, obs, gt_candidates):
        """= RemotePolicy.act, plus the policy's internal commitment (`committed`, acot runtime only) in the scorer
        info (1006 §4.1)."""
        send_msg(self.sock, {"cmd": "act", "rgb": obs.rgb, "language": obs.language,
                             "proprio": np.asarray(obs.proprio), "step": int(obs.step),
                             "target_color": getattr(obs, "target_color", ""),
                             "target_bp": getattr(obs, "target_bp", ""),
                             "rgb_hi": getattr(obs, "rgb_hi", None),
                             "retarget_event": bool(getattr(obs, "retarget_event", False)),
                             "gt_candidates": gt_candidates})
        r = recv_msg(self.sock)
        cl = r.get("cand_lite")
        cand_set = None if cl is None else SimpleNamespace(cands=[SimpleNamespace(**c) for c in cl["cands"]],
                                                            true_idx=cl["true_idx"])
        info = SimpleNamespace(cand_set=cand_set, pred_slot=r["pred_slot"])
        if "committed" in r:
            info.committed_key = r["committed"]
        return tuple(r["action"]), r["search_mode"], info
