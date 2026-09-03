"""Env-side stub that forwards .act() to the policy server over TCP (bridge.py).

Implements the same .reset()/.act() contract StudentRolloutEnv expects, so the env
loop is unchanged whether the policy is local or remote. Reconstructs a lightweight
`info` (cand_set with per-slot identity/depth + pred_slot) for the scorer — the big
VLM features never cross the wire.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import numpy as np

from rollout.bridge import send_msg, recv_msg


class RemotePolicy:
    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def reset(self):
        send_msg(self.sock, {"cmd": "reset"})
        recv_msg(self.sock)

    def act(self, obs, gt_candidates):
        send_msg(self.sock, {"cmd": "act", "rgb": obs.rgb, "language": obs.language,
                             "proprio": np.asarray(obs.proprio), "step": int(obs.step),
                             "target_color": getattr(obs, "target_color", ""),
                             "target_bp": getattr(obs, "target_bp", ""),
                             "gt_candidates": gt_candidates})
        r = recv_msg(self.sock)
        cl = r.get("cand_lite")
        if cl is None:
            cand_set = None
        else:
            cand_set = SimpleNamespace(
                cands=[SimpleNamespace(**c) for c in cl["cands"]],
                true_idx=cl["true_idx"])
        info = SimpleNamespace(cand_set=cand_set, pred_slot=r["pred_slot"])
        return tuple(r["action"]), r["search_mode"], info

    def close(self):
        try:
            send_msg(self.sock, {"cmd": "close"})
            recv_msg(self.sock)
        except OSError:
            pass
        self.sock.close()
