"""PGIF — Posterior-Gated Identity Filtering (framework §7.2, P2): port of the filter mode of the legacy ASD class
(carla_uav_tracking/rollout/asd.py, mode="filter") without any label argument. Per decision tick the memory's pick is
an observation of the target's tracklet key:
  tempered  L_k ← λ·L_k + (0.5 + rank(conf))·1[k = pick]     (main; rank = position in a sliding confidence window)
  resid     the same with weight 1                              (ablation)
  bayes     sticky HMM posterior with a confidence-ranked error rate  (ablation)
The output is the MAP key: if it differs from the pick the policy commits to it and (gate_memory) re-does the memory
update with it. absent_hold: the MAP ranges over keys seen within absent_max ticks; if the MAP key is not visible
this tick the output is "absent" (no pick, commitment held, the memory write undone).
inherit (1007 opt, off by default): a target re-appearing under a NEW tracker id inherits the evidence of the
vanished id it most plausibly continues (position gate; see _inherit) — tempered / resid only."""
import math

import numpy as np


class Pgif:
    def __init__(self, filter_kind, resid_decay, absent_hold, absent_max, conf_window, gate_memory, eps_switch, q_lo,
                 q_hi, inherit=False, inherit_gate_px=160.0, inherit_px_per_tick=30.0):
        assert filter_kind in ("tempered", "resid", "bayes"), filter_kind
        self.kind, self.decay = filter_kind, float(resid_decay)
        self.absent_hold, self.absent_max = bool(absent_hold), int(absent_max)
        self.conf_window, self.gate_memory = int(conf_window), bool(gate_memory)
        self.eps_switch, self.q_lo, self.q_hi = float(eps_switch), float(q_lo), float(q_hi)
        self.inherit, self.inherit_gate_px = bool(inherit), float(inherit_gate_px)
        self.inherit_px_per_tick = float(inherit_px_per_tick)
        self.reset()

    @classmethod
    def from_method(cls, F):
        kw = {k: F[k] for k in ("filter_kind", "resid_decay", "absent_hold", "absent_max", "conf_window",
                                "gate_memory", "eps_switch", "q_lo", "q_hi")}
        for k in ("inherit", "inherit_gate_px", "inherit_px_per_tick"):           # 1007 opt; absent = off
            if F.get(k) is not None:
                kw[k] = F[k]
        return cls(**kw)

    def reset(self):
        self.belief = {}
        self.conf_hist = []
        self.last_seen = {}
        self.tick = 0
        self.fires = 0
        self.absents = 0
        self.seen, self.first_seen = {}, {}          # inherit: key → (u, v, tick last seen) / tick first seen
        self.inherits, self.last_inherit = 0, None

    def _rank(self, conf):
        c = 0.0 if conf is None else float(conf)
        h = self.conf_hist
        pct = float(np.searchsorted(np.sort(h), c)) / len(h) if len(h) >= 5 else 0.5
        self.conf_hist = (h + [c])[-self.conf_window:]
        return pct

    def _inherit(self, ids, uv):
        """the strongest-evidence key NOT visible this tick, last seen within absent_max ticks and stronger than every
        visible key (i.e. the would-be "absent" MAP), hands its evidence to a visible key first seen AFTER it vanished
        whose box lies within inherit_gate_px + inherit_px_per_tick × (ticks since) of its last position (nearest
        wins). Position only: the tracker's own association (motion + appearance) has already failed here."""
        b, shown = self.belief, set(ids)
        gone = [k for k in b if k not in shown and k in self.seen and b[k] > 0
                and self.tick - self.seen[k][2] <= self.absent_max]
        if not gone:
            return
        g = max(gone, key=lambda k: b[k])
        if b[g] <= max(b.get(i, 0.0) for i in ids):
            return
        gu, gv, gt = self.seen[g]
        lim = self.inherit_gate_px + self.inherit_px_per_tick * (self.tick - gt)
        cand = [(math.hypot(u - gu, v - gv), i) for i, (u, v) in zip(ids, uv)
                if self.first_seen.get(i, self.tick) > gt and math.hypot(u - gu, v - gv) <= lim]
        if cand:
            _, i = min(cand)
            b[i] = b.get(i, 0.0) + b.pop(g)
            self.inherits += 1
            self.last_inherit = (int(g), int(i))

    def step(self, keys, mem_slot, mem_conf, uv=None):
        """keys: the detections' keys; mem_slot: the memory's pick → {fire, slot, kind: gate|absent, gate_memory}.
        uv: the detections' image positions (only read when inherit is on)."""
        ids = [int(a) for a in keys]
        act = ids[mem_slot]
        b = self.belief
        self.last_inherit = None
        if self.kind in ("resid", "tempered"):
            if self.inherit and uv is not None:
                for i in ids:
                    self.first_seen.setdefault(i, self.tick)
                self._inherit(ids, uv)
                for i, (u, v) in zip(ids, uv):
                    self.seen[i] = (float(u), float(v), self.tick)
            pct = self._rank(mem_conf)
            w = 1.0 if self.kind == "resid" else 0.5 + pct
            for k in b:
                b[k] *= self.decay
            b[act] = b.get(act, 0.0) + w
            map_slot = max(range(len(ids)), key=lambda s_: b.get(ids[s_], 0.0))
            if self.absent_hold:
                for i in ids:
                    self.last_seen[i] = self.tick
                alive = [k for k in b if self.tick - self.last_seen.get(k, -10 ** 9) <= self.absent_max]
                gmap = max(alive, key=lambda k: b[k]) if alive else None
                if gmap is not None and gmap not in ids and b[gmap] > b.get(ids[map_slot], 0.0):
                    self.absents += 1
                    self.tick += 1
                    return {"fire": False, "slot": None, "kind": "absent", "gate_memory": self.gate_memory,
                            "absent_id": int(gmap)}
            fire = map_slot != mem_slot
            self.fires += int(fire)
            self.tick += 1
            return {"fire": fire, "slot": map_slot, "kind": "gate", "gate_memory": self.gate_memory}
        for a in ids:                                   # bayes: new keys enter with a tiny prior
            b.setdefault(a, 1e-3 if b else 1.0)
        tot = sum(b.values())
        m = len(b)
        for k in b:
            b[k] = (1 - self.eps_switch) * b[k] / tot + self.eps_switch / m
        pct = self._rank(mem_conf)
        q = self.q_hi - (self.q_hi - self.q_lo) * pct
        nv = len(ids)
        for k in b:
            b[k] *= (1 - q) if k == act else q / max(nv - 1, 1)
        tot = sum(b.values())
        for k in b:
            b[k] /= tot
        map_slot = max(range(nv), key=lambda s_: b[ids[s_]])
        fire = map_slot != mem_slot
        self.fires += int(fire)
        self.tick += 1
        return {"fire": fire, "slot": map_slot, "kind": "gate", "post": round(b[ids[map_slot]], 4),
                "gate_memory": self.gate_memory}
