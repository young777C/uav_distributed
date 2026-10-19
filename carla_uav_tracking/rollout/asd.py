"""ASD module — drift correction for closed-loop self-updating identity memories.

MAIN METHOD (mode="filter", filter_kind="tempered") = PGIF, Posterior-Gated Identity Filtering:
  a memory-agnostic wrapper around ANY identity memory that (i) picks a candidate, (ii) reports a confidence on any
  scale, (iii) exposes an update hook. Per decision tick it keeps a tempered, confidence-weighted evidence count per
  persistent tracklet  L_k <- lam*L_k + (0.5 + rank(conf))*1[k = memory pick]  and, when argmax_k L_k disagrees with
  the memory's pick, re-does the memory update with the argmax crop and commits control to it (posterior gating).
  Rationale (acot_note/1001_innovation-design.md): self-update makes the memory's errors autocorrelated, so evidence
  must be tempered (untempered Bayes = ablation, gets captured); gating stops wrong picks being written into memory.
ABLATIONS: filter_kind="resid" (equal weights), "bayes" (untempered sticky HMM), gate_memory=False (commit-only).
LEGACY (negative results, kept for reproducibility): modes "rule"/"learned" = frozen appearance+language anchor with
  multi-hypothesis CUSUM; "observe" = log statistics only (no intervention).
GT is never used at decision time; `true_slot` is only written to the log for offline scoring."""
from __future__ import annotations

import json

import numpy as np

STAT_NAMES = ["disagree", "margin", "anc_dissim", "lang_disfavor", "disagree_x_margin"]


class ASD:
    def __init__(self, mode: str = "rule", k_anchor: int = 3, beta_lang: float = 0.25, lang_ema: float = 0.7,
                 tau: float = 0.1, kappa: float = 0.25, h: float = 2.0, refractory: int = 3,
                 learned_ckpt: str | None = None, log_path: str | None = None, anchor_mode: str = "first",
                 eps_switch: float = 0.01, q_lo: float = 0.2, q_hi: float = 0.7, conf_window: int = 30,
                 gate_memory: bool = True, filter_kind: str = "bayes", resid_decay: float = 0.98,
                 alpha: dict | None = None, naive: bool = False, gate_tah: bool = True, gate_dam: bool = True,
                 absent_hold: bool = False, absent_max: int = 8):
        assert mode in ("observe", "rule", "learned", "filter"), mode
        self.mode, self.k_anchor, self.beta, self.lema = mode, int(k_anchor), float(beta_lang), float(lang_ema)
        self.tau, self.kappa, self.h, self.refractory = float(tau), float(kappa), float(h), int(refractory)
        self.w = self.b = None
        if mode == "learned":
            assert learned_ckpt, "mode=learned needs learned_ckpt"
            ck = json.load(open(learned_ckpt))
            self.w, self.b = np.asarray(ck["w"], np.float64), float(ck["b"])
            self.prior_logit = float(ck.get("prior_logit", 0.0))
            self.h = float(ck.get("h", self.h))
            self.mu, self.sd = np.asarray(ck["mu"]), np.asarray(ck["sd"])
        self.log_path = log_path
        # anchor_mode: 'first' = anchor = M's picks during the window (inherits cold-start errors) |
        # 'spec' = sequential SPEC identification: integrate language evidence per actor over the window, anchor
        # to the argmax actor's crops and correct M if it disagrees (also fixes wrong cold-start locks).
        assert anchor_mode in ("first", "spec"), anchor_mode
        self.anchor_mode = anchor_mode
        # filter mode (ASD-IF): sticky Bayes filter over persistent tracklets; the memory's pick is a noisy
        # observation whose error rate q is set by the rank of its confidence within a sliding window
        # (scale-free → memory-agnostic, no per-memory calibration). Output = MAP tracklet (posterior gating).
        self.eps_switch, self.q_lo, self.q_hi, self.conf_window = float(eps_switch), float(q_lo), float(q_hi), int(conf_window)
        self.gate_memory = bool(gate_memory)          # ablation: False = override commitment only, memory still self-updates
        assert filter_kind in ("bayes", "resid", "tempered"), filter_kind
        self.filter_kind, self.resid_decay = filter_kind, float(resid_decay)   # 'resid' = longest-residence heuristic
        # V3 fusion: per-source weights (default equal), naive = per-tick vote (no tempering, ensemble control)
        self.alpha = dict(alpha or {}); self.naive = bool(naive)
        self.gate_tah, self.gate_dam = bool(gate_tah), bool(gate_dam)
        # 1004 deployable v2 · ABSENCE HYPOTHESIS: the posterior also ranges over tracklets NOT visible this tick (seen
        # within the last `absent_max` ticks). If its MAP tracklet is currently unseen (occluded / out of frame), PGIF
        # outputs "absent" (no pick → the commitment is held, the memory is not written) instead of being forced to
        # pick the best VISIBLE tracklet — the forced choice is what turns a target occlusion into a wrong lock.
        # The evidence update itself is unchanged.
        self.absent_hold, self.absent_max = bool(absent_hold), int(absent_max)
        self.ep = -1
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.anchor: list[np.ndarray] = []      # frozen early-track appearance bank (drift-immune reference)
        self.win_feats: dict[int, list] = {}    # spec-anchor window: actor id -> crops seen
        self.win_ticks = 0
        self.belief: dict[int, float] = {}      # filter: posterior over tracklet ids
        self.conf_hist: list[float] = []
        self.src_hist: dict[str, list] = {}     # V3 fusion: per-source confidence windows
        self.lang_score: dict[int, float] = {}  # per-actor EMA of z-scored language logit (spec integration)
        self.S: dict[int, float] = {}           # multi-hypothesis CUSUM: per challenger identity (actor id)
        self.cool = 0                           # refractory ticks after an alarm
        self.tick = 0
        self.fires = 0
        self.last_seen: dict[int, int] = {}     # absence hypothesis: tracklet id -> last tick it was visible
        self.absents = 0
        self.ep += 1

    # ------------------------------------------------------------------ stats
    def _lang_z(self, aidx, lang):
        if lang is None or len(lang) == 0:
            return np.zeros(len(aidx))
        l = np.asarray(lang, np.float64)
        z = (l - l.mean()) / (l.std() + 1e-6) if len(l) > 1 else np.zeros_like(l)
        for a, v in zip(aidx, z):
            p = self.lang_score.get(int(a), float(v))
            self.lang_score[int(a)] = self.lema * p + (1 - self.lema) * float(v)
        return np.array([self.lang_score[int(a)] for a in aidx])

    def stats(self, feats, aidx, mem_slot, lang):
        zl = self._lang_z(aidx, lang)
        cosA = (feats @ np.stack(self.anchor).T).max(axis=1) if self.anchor else np.zeros(len(aidx))
        a = cosA + self.beta * zl                                    # anchor (spec-side) score
        anc = int(a.argmax())
        dis = float(anc != mem_slot)
        margin = float(a[anc] - a[mem_slot])
        x = np.array([dis, margin, 1.0 - float(cosA[mem_slot]), float(zl.max() - zl[mem_slot]), dis * margin])
        self._last = (cosA, zl)                                      # per-candidate view (offline ablations)
        return x, anc

    def increment(self, x):
        if self.mode == "learned":
            z = (x - self.mu) / self.sd
            return float(z @ self.w + self.b - self.prior_logit)      # log-likelihood ratio (wrong vs right)
        return float(x[0] * min(x[1] / self.tau, 2.0) - self.kappa)  # rule: disagreement weighted by margin

    # ------------------------------------------------------------------ main
    def step(self, feats, aidx, mem_slot, mem_conf=None, lang=None, true_slot=None, t=None):
        out = {"fire": False, "slot": mem_slot}
        if mem_slot is None or mem_slot < 0 or len(aidx) == 0:
            return out
        if self.mode == "filter":
            return self._filter_step(aidx, mem_slot, mem_conf, true_slot, t)
        feats = np.asarray(feats, np.float64)
        if self.anchor_mode == "first" and len(self.anchor) < self.k_anchor:   # window: anchor = M's early picks
            self._lang_z(aidx, lang)
            self.anchor.append(feats[mem_slot].copy())
            self.tick += 1
            self._log(t, aidx, mem_slot, mem_conf, true_slot, None, None, False)
            return out
        if self.anchor_mode == "spec" and not self.anchor:                  # window: integrate spec evidence
            self._lang_z(aidx, lang)
            for s_, a_ in enumerate(aidx):
                self.win_feats.setdefault(int(a_), []).append(feats[s_].copy())
            self.win_ticks += 1
            self.tick += 1
            fire = False
            if self.win_ticks >= self.k_anchor:
                seen = [a_ for a_ in self.lang_score if len(self.win_feats.get(a_, [])) >= max(1, self.win_ticks // 2)]
                best = max(seen or list(self.lang_score), key=lambda a_: self.lang_score[a_])
                self.anchor = self.win_feats[best][-self.k_anchor:]
                if best in [int(a_) for a_ in aidx] and int(aidx[mem_slot]) != best and self.mode != "observe":
                    fire = True; self.fires += 1; self.cool = self.refractory
                    out = {"fire": True, "slot": [int(a_) for a_ in aidx].index(best)}
                self.win_feats = {}
            self._log(t, aidx, mem_slot, mem_conf, true_slot, None, None, fire)
            return out
        x, anc = self.stats(feats, aidx, mem_slot, lang)
        g = self.increment(x)
        fire = False
        # Multi-hypothesis CUSUM: evidence accumulates only for the SPECIFIC identity the anchor prefers over M.
        # Noise-driven disagreements scatter over different look-alikes and decay; a true drift concentrates
        # on one identity (the real target) → that hypothesis crosses h.
        ca = int(aidx[anc]) if anc != mem_slot else None
        for k in list(self.S):
            if k != ca:
                self.S[k] = max(0.0, self.S[k] - self.kappa)
        if ca is not None:
            self.S[ca] = max(0.0, self.S.get(ca, 0.0) + g)
        if self.cool > 0:
            self.cool -= 1
        elif self.mode != "observe" and ca is not None and self.S[ca] > self.h:
            fire = True
            self.S, self.cool, self.fires = {}, self.refractory, self.fires + 1
            out = {"fire": True, "slot": anc}
        self.tick += 1
        self._log(t, aidx, mem_slot, mem_conf, true_slot, x, anc, fire)
        return out

    def step_multi(self, aidx, obs, true_slot=None, t=None):
        """V3 fusion. obs = [(slot|None, conf, source), ...] from heterogeneous memories + language. Tempered,
        rank-weighted per-source votes into ONE tracklet evidence; naive=True -> per-tick vote (ensemble control).
        Returns {"slot": MAP slot or None, "gate": {source: MAP != that source's pick}}."""
        ids = [int(a) for a in aidx]
        b = {} if self.naive else self.belief
        if not self.naive:
            for k in b:
                b[k] *= self.resid_decay
        picks = {}
        for slot, conf, src in obs:
            if slot is None or slot < 0 or slot >= len(ids):
                picks[src] = None; continue
            c = 0.0 if conf is None or conf != conf else float(conf)
            h = self.src_hist.get(src, [])
            pct = float(np.searchsorted(np.sort(h), c)) / len(h) if len(h) >= 5 else 0.5
            self.src_hist[src] = (h + [c])[-self.conf_window:]
            b[ids[slot]] = b.get(ids[slot], 0.0) + float(self.alpha.get(src, 1.0)) * (0.5 + pct)
            picks[src] = int(slot)
        self.belief = b
        vis = [s_ for s_ in range(len(ids)) if b.get(ids[s_], 0.0) > 0]
        map_slot = max(vis, key=lambda s_: b[ids[s_]]) if vis else None
        gate = {src: (map_slot is not None and p != map_slot) for src, p in picks.items()}
        self.tick += 1
        if map_slot is not None and any(gate.values()):
            self.fires += 1
        if self.log_path:
            rec = {"ep": self.ep, "t": t, "n": len(ids), "true": None if true_slot is None else int(true_slot),
                   "map": map_slot, "picks": picks, "gate": gate, "aidx": ids,
                   "conf": {src: (None if c is None or c != c else round(float(c), 4)) for _, c, src in obs}}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return {"slot": map_slot, "gate": gate}

    def override(self, new_actor_id: int, prior: float = 3.0, t=None):
        """1003_dynamic-retarget-design.md §2.3 — event-driven hard reset (S1: GT-driven).
        Not evidence: an external re-designation command bypasses the tempered accumulator
        entirely (a legitimate switch must not have to out-accumulate the old target's
        evidence against the sticky prior that drift-resistance deliberately creates). Wipes
        ALL tracklet evidence (not just the old target's) and seeds the new target with a
        prior equivalent to a few committed ticks, so the very next step() doesn't immediately
        flip back. conf_hist is cleared too (rank-confidence restarts fresh, as at cold-start)."""
        self.belief = {int(new_actor_id): float(prior)}
        self.conf_hist = []
        self.fires += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"ep": self.ep, "t": t, "kind": "override",
                                    "new_actor_id": int(new_actor_id)}) + "\n")

    def _filter_step(self, aidx, mem_slot, mem_conf, true_slot, t):
        ids = [int(a) for a in aidx]; act = ids[mem_slot]; b = self.belief
        if self.filter_kind in ("resid", "tempered"):
            # TEMPERED evidence accumulator (PGIF main): L_k <- lam*L_k + w_t*1[k = memory pick]. Self-update makes
            # the memory's errors autocorrelated, so per-tick evidence must be tempered (lam ~ 1 - 1/tau_corr)
            # instead of multiplied as if independent (the untempered 'bayes' ablation flips in 1-2 ticks).
            # w_t = 0.5 + rank(conf) (scale-free) for 'tempered'; w_t = 1 for the 'resid' ablation.
            c = 0.0 if mem_conf is None else float(mem_conf); h = self.conf_hist
            pct = float(np.searchsorted(np.sort(h), c)) / len(h) if len(h) >= 5 else 0.5
            self.conf_hist = (h + [c])[-self.conf_window:]
            w = 1.0 if self.filter_kind == "resid" else 0.5 + pct
            for k in b:
                b[k] *= self.resid_decay
            b[act] = b.get(act, 0.0) + w
            map_slot = max(range(len(ids)), key=lambda s_: b.get(ids[s_], 0.0))
            if self.absent_hold:
                for i in ids:
                    self.last_seen[i] = self.tick
                alive = [k for k in b if self.tick - self.last_seen.get(k, -10 ** 9) <= self.absent_max]
                gmap = max(alive, key=lambda k: b[k]) if alive else None
                if gmap is not None and gmap not in ids and b[gmap] > b.get(ids[map_slot], 0.0):
                    self.absents += 1; self.tick += 1
                    self._log(t, aidx, mem_slot, mem_conf, true_slot, None, None, False, absent=int(gmap))
                    return {"fire": False, "slot": None, "kind": "absent", "gate_memory": self.gate_memory,
                            "absent_id": int(gmap)}
            fire = map_slot != mem_slot
            self.fires += int(fire); self.tick += 1
            self._log(t, aidx, mem_slot, mem_conf, true_slot, None, map_slot, fire)
            return {"fire": fire, "slot": map_slot, "kind": "gate", "gate_memory": self.gate_memory}
        for a in ids:                                            # new tracklets enter with a tiny prior
            b.setdefault(a, 1e-3 if b else 1.0)
        tot = sum(b.values()); m = len(b)
        for k in b:                                              # sticky identity transition
            b[k] = (1 - self.eps_switch) * b[k] / tot + self.eps_switch / m
        c = 0.0 if mem_conf is None else float(mem_conf)
        h = self.conf_hist
        pct = float(np.searchsorted(np.sort(h), c)) / len(h) if len(h) >= 5 else 0.5
        self.conf_hist = (h + [c])[-self.conf_window:]
        q = self.q_hi - (self.q_hi - self.q_lo) * pct           # confident pick → lower error rate
        nv = len(ids)
        for k in b:
            b[k] *= (1 - q) if k == act else q / max(nv - 1, 1)
        tot = sum(b.values())
        for k in b:
            b[k] /= tot
        map_slot = max(range(nv), key=lambda s_: b[ids[s_]])
        fire = map_slot != mem_slot
        if fire:
            self.fires += 1
        self.tick += 1
        self._log(t, aidx, mem_slot, mem_conf, true_slot, None, map_slot, fire)
        return {"fire": fire, "slot": map_slot, "kind": "gate", "post": round(b[ids[map_slot]], 4),
                "gate_memory": self.gate_memory}

    def _log(self, t, aidx, mem_slot, mem_conf, true_slot, x, anc, fire, absent=None):
        if not self.log_path:
            return
        rec = {"ep": self.ep, "t": t, "n": len(aidx), "mem": int(mem_slot),
               "true": None if true_slot is None else int(true_slot),
               "conf": None if mem_conf is None else float(mem_conf),
               "anc": anc, "S": round(max(self.S.values(), default=0.0), 4), "fire": fire,
               "x": None if x is None else [round(float(v), 5) for v in x],
               "aidx": [int(a) for a in aidx]}
        if absent is not None:
            rec["absent"] = absent                       # PGIF held: MAP tracklet (this id) not visible this tick
        if x is not None and getattr(self, "_last", None) is not None:
            rec["cosA"] = [round(float(v), 4) for v in self._last[0]]
            rec["zl"] = [round(float(v), 4) for v in self._last[1]]
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
