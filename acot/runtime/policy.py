"""The acot runtime policy (framework §7.2, P2): the decision logic of the methods the 1004 experiments use, built from
the RESOLVED configuration (no environment variables, no constructor defaults). It receives PolicyInput — the frame,
language, proprio, own camera pose and the detections of the protocol's perception — and, for a diagnostic arm, an
OracleHandle restricted to the GT channels the method declared. Labels never reach it.

Per S2 tick (step % s2_period == 0): EAR waypoint sample + IAR → grounding logits over the detections → opening
(protocol frontend: accumulate the attribute scores per key over the opening window) → identity head (tah / assoc /
dam4sam / …; xattn = the grounding pick; sticky = the opening key, never re-picked) → pick confidence = top1 − top2
margin → PGIF (tempered / resid / bayes, absence hold, posterior-gated memory update) → K-tick commitment hysteresis → WHERE: the committed key's matched box world
position feeds the constant-velocity target estimator whose waypoints replace the EAR sample. Every tick: DiT action.
Each step mirrors carla_uav_tracking/rollout/policy.py (Policy.act) in the same order and with the same RNG use;
acot/eval/replay_parity.py checks decision equality on recorded act messages. Unsupported settings are refused
(unsupported()) — such arms run on the legacy runtime."""
import hashlib
import os
import time

import numpy as np
import torch

from acot import ROOT
from acot.runtime.heads import AssocHead, Dam4samHead, ReidHead, TahHead
from acot.runtime.io import Decision
from acot.runtime.pgif import Pgif
from acot.runtime.support import unsupported

ORACLE_PURPOSE = {"abstain": "oracle_absent_abstain", "hygiene": "oracle_absent_hygiene"}


class UnsupportedConfig(ValueError):
    pass


class AcotPolicy:
    def __init__(self, protocol, method, frontend, device, services=None, log=print):
        bad = unsupported(protocol, method)
        if bad:
            raise UnsupportedConfig("acot runtime cannot run this arm (use runtime: legacy): " + "; ".join(bad))
        from rollout.target_state import TargetStateEstimator
        P, M = protocol, method
        C, I, F = M["control"], M["identity"], M["filter"]
        self.fe, self.device = frontend, device
        self.s2_period = int(P["sim"]["s2_period"])
        self.policy_rng = P["sim"].get("policy_rng", "unseeded")
        self.rng_base = None
        self.ablate = C["ablate"]
        self.commit_k = int(C["commit_k"])
        self.frame_gain = float(C["frame_gain"])
        self.absent_where = C.get("absent_where", "ear")
        self.conf_tau = float(I["conf_tau"])
        self.kind = I["head"]
        if self.kind == "tah":
            fm = None
            if I["tah"].get("feature", "dinov2_crop") == "dinov3_crop":
                from acot.runtime.head_oavat import dinov3_extractor
                fm = dinov3_extractor("dinov3_vitb16", os.path.join(
                    ROOT, "external/OA-VAT/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"), device).model
            self.head = TahHead(os.path.join(ROOT, I["tah"]["ckpt"]), device, log=log, feat_model=fm)
        elif self.kind == "assoc":
            a = I["assoc"]
            self.head = AssocHead(a["mode"], a["gate_px"], a["lambda"])
        elif self.kind == "dam4sam":
            d, svc = I["dam4sam"], (services or {}).get("dam4sam")
            if svc is None:
                raise UnsupportedConfig("identity.head=dam4sam needs a DAM4SAM service instance (infra.services)")
            self.head = Dam4samHead(svc["host"], svc["port"], d["gate_px"], d["every_frame"], d["reinit_cooldown"],
                                    log=log)
        elif self.kind == "oavat":
            from acot.runtime.head_oavat import OavatHead, make_extractor
            o = I["oavat"]
            ex = make_extractor(o["model"], os.path.join(ROOT, o["checkpoint"]) if o["checkpoint"] else "", device)
            self.head = OavatHead(ex, o["match_thresh"], o["reacq_thresh"], o["iou_pred"], o["fullframe_after"],
                                  o["max_predict_ticks"], o["enhance_every"], o["enhance_alpha"], o["history"])
        elif self.kind == "reid":
            r = I["reid"]
            self.head = ReidHead(r["feature"], r["bank"], r["topm"], r["tavg"], r["motion"], r["motion_gate_px"])
        elif self.kind == "deepsort":                 # 1016 DeepSORT identity adaptation (official tracker)
            from acot.runtime.head_mot import DeepSortHead
            self.head = DeepSortHead(**I["deepsort"])
        elif self.kind == "ocsort":                   # 1016 OC-SORT identity adaptation (official tracker)
            from acot.runtime.head_mot import OcSortHead
            self.head = OcSortHead(**I["ocsort"], W=frontend.image_cfg["width"], H=frontend.image_cfg["height"])
        else:
            self.head = None                  # xattn (grounding pick) / oracle (diagnostic U1) / sticky (null)
        self.memory_seed = I["memory_seed"]   # committed | gt (diagnostic U2: TAH memory written with the true slot)
        self.tah_write = (I.get("tah") or {}).get("memory_write", "own") if self.kind == "tah" else "own"
        # 1016 every-frame MOT heads: what the server must hand over on the frames between S2 ticks (detections only,
        # via PolicyInput.detections; + crop features for DeepSORT's appearance metric); None = nothing
        self.frame_input = None
        if self.kind in ("deepsort", "ocsort") and self.head.every_frame:
            self.frame_input = "boxes+feats" if self.kind == "deepsort" else "boxes"
        self.mot_frame_ms = 0.0
        self.pgif = Pgif.from_method(F) if F["mode"] == "filter" else None
        self.gate_commit = bool(F.get("gate_commit", True))
        self.force_conf = bool(F.get("force_conf", True))
        self.oracle_absent = M["oracle_absent"]
        o = P["frontend"]["opening"]
        self.opening = frontend.opening
        self.open_window = int(o["window"]) if self.opening is not None else 1
        self.open_agg, self.open_final_reset = str(o["agg"]), bool(o["final_reset"])
        self.open_reset_pgif, self.open_force_final = bool(o["reset_asd"]), bool(o["force_final"])
        self.open_seed_pgif = float(o["seed_asd"])
        self.open_handoff = o.get("handoff", "legacy")
        wp = frontend.cfg["waypoint"]
        self.estimator = TargetStateEstimator(wp["offsets_s"], wp["scale"], frontend.cfg.get("fps", 10))
        img = frontend.image_cfg
        self.W, self.H, self.fov = img["width"], img["height"], img["fov_deg"]
        self.reset()

    # ------------------------------------------------------------------------------------------------------- state
    def reset(self):
        self.z_ex = self.z_im = self.vlm_ctx = self.ctx_mask = None
        self.logits = np.zeros(0, np.float32)
        self.pred_slot = -1
        self.pred_conf = 0.0
        self.dets = None
        self.committed = None                          # committed identity key (K-tick hysteresis)
        self.chal, self.chal_n = None, 0
        self.frame_cw = None
        self.open_k, self.open_acc, self.open_cnt, self.open_force = 0, {}, {}, False
        self.events = 0
        self.mot_frame_ms = 0.0
        self.sticky_key = None                         # sticky null: the key bound at the opening
        if self.head is not None:          # MOT heads: the tracker lives per episode, reset() only drops the binding
            getattr(self.head, "reset_episode", self.head.reset)()
        if self.pgif is not None:
            self.pgif.reset()
        self.estimator.reset()

    def set_episode(self, rng_base):
        """opaque per-episode RNG base for the per-step policy RNG (1006 §3.5), = H(scene, repeat) computed by the
        server — the policy never sees the scene identity. None = unseeded."""
        self.rng_base = None if rng_base is None else int(rng_base)

    def _reseed(self, stream, step):
        """per_step: torch reseeded from H(rng_base, stream, step) — arm-independent, so arms share the
        sampling noise of every step (common random numbers) whatever their history."""
        if self.policy_rng != "per_step" or self.rng_base is None:
            return
        h = int.from_bytes(hashlib.sha256(f"{self.rng_base}|{stream}|{int(step)}".encode()).digest()[:8],
                           "little") & (2 ** 63 - 1)
        torch.manual_seed(h)          # EAR / DiT sampling draws from torch only (numpy global RNG unused)

    def is_s2(self, step):
        return step % self.s2_period == 0 or self.z_ex is None

    # -------------------------------------------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, inp, oracle=None):
        """PolicyInput (+ OracleHandle for diagnostic arms) → Decision."""
        from train import flow_matching
        from train.stage2 import action_sample
        fe = self.fe
        prop = torch.from_numpy(inp.proprio.astype(np.float32))[None].to(self.device)
        if self.kind == "dam4sam" and self.head.every_frame and self.head.inited and not inp.s2:
            self.head.advance(inp.rgb)
        if self.frame_input is not None and not inp.s2:          # 1016 every-frame MOT: tracker step on this frame
            t_f = time.perf_counter()
            self.head.advance(inp.detections)
            self.mot_frame_ms += 1000 * (time.perf_counter() - t_f)
        trace = None
        if inp.s2:
            t_s2 = time.perf_counter()
            layers, ctx = inp.vlm
            self.vlm_ctx = ctx[None]
            ctx_layers = [x[None] for x in layers]
            self.ctx_mask = torch.ones(1, ctx.shape[0], dtype=torch.bool, device=self.device)
            if self.ablate == "ear":
                self.z_ex = torch.zeros(1, fe.K, 3, device=self.device)
            else:
                self._reseed("ear", inp.step)
                self.z_ex = flow_matching.sample(fe.ear, self.vlm_ctx, self.ctx_mask, k=fe.K, steps=fe.steps,
                                                 proprio=prop)
            self.z_im, _ = fe.iar(ctx_layers, self.ctx_mask)
            if self.ablate == "iar":
                self.z_im = torch.zeros_like(self.z_im)
            trace = self._identity(inp, inp.detections, oracle)
            self.dets = inp.detections
            if self.ablate != "ear":
                self._where(inp)
            self._sync()
            trace["t_s2_ms"] = round(1000 * (time.perf_counter() - t_s2), 2)        # 1006 §4.2 cost ②
        self._reseed("dit", inp.step)
        act = action_sample(fe.dit, self.z_ex, self.z_im, self.vlm_ctx, prop, self.ctx_mask, fe.H, fe.A, fe.steps)
        a0 = act[0, 0].float().cpu().numpy()
        dx, dy, dz, dyaw = (a0[:4] * fe.a_scale).tolist()
        search = float(a0[4] > 0.5)
        if self.frame_gain and self.frame_cw is not None:
            from acot_probe.projection import project_point
            pr = project_point(tuple(float(x) for x in self.frame_cw), tuple(float(x) for x in inp.campose),
                               self.W, self.H, self.fov)
            if pr is not None and pr[2] > 0:
                dyaw = dyaw + self.frame_gain * (pr[0] / self.W - 0.5)
        if trace is not None:
            trace.update(pred=self.pred_slot, conf=self.pred_conf, committed=self.committed, search=search)
        return Decision(action=(dx, dy, dz, dyaw), search=search, pred_slot=self.pred_slot, trace=trace)

    # --------------------------------------------------------------------------------------------------- identity
    def _identity(self, inp, dets, oracle):
        if dets is None:
            tr0 = {"n": 0}
            if self.kind in ("deepsort", "ocsort"):
                t_id = time.perf_counter()
                self.head.no_detections()             # the tracker steps on every tick (predict, no measurement)
                tr0.update(self._mot_trace(), t_identity_ms=round(1000 * (time.perf_counter() - t_id), 3))
            self.logits, self.pred_slot, self.pred_conf = np.zeros(0, np.float32), -1, 0.0
            return tr0
        n = len(dets)
        keys = [d.key for d in dets]
        cf = torch.from_numpy(np.stack([d.vlm_feat for d in dets]))[None].to(self.device)
        cmask = torch.from_numpy(np.ones(n, dtype=bool))[None].to(self.device)
        lg = self.fe.tid(cf, self.vlm_ctx, self.ctx_mask, cmask)[0]           # language grounding (all heads)
        self.logits = lg.float().cpu().numpy()
        self.pred_slot = int(lg.argmax().item())
        lang_slot = self.pred_slot
        tr = {"n": n, "keys": keys, "lang": lang_slot}
        shared = inp.shared_open_slot if (self.opening is not None and self.open_k < self.open_window) else None
        if self.opening is not None and self.open_k < self.open_window and n > 0:
            lang_slot = self._opening(inp, dets)
            tr["open"] = lang_slot
            if shared is not None:                    # 1006 §3.4: shared opening — every head re-seeded from the ref
                tr["shared"] = int(shared)
                if 0 <= shared < n:
                    lang_slot = shared
                    if self.kind == "tah":
                        self.head.reset_memory()
                    elif self.head is not None:
                        self.head.reset()
                else:
                    tr["open_fallback"] = True
        self._sync()
        t_id = time.perf_counter()
        if self.kind == "assoc":
            al = self.head.score(dets, init_slot=lang_slot)
            self.logits = al
            self.pred_slot = int(al.argmax())
        elif self.kind == "dam4sam":
            dl = self.head.score(dets, inp.rgb, init_slot=lang_slot)
            self.logits = dl
            self.pred_slot = int(dl.argmax()) if float(dl.max()) > -1e8 else -1
        elif self.kind == "reid":
            cs = None if self.committed is None else next((i for i, d in enumerate(dets) if d.key == self.committed),
                                                          None)
            rl = self.head.score(dets, init_slot=lang_slot, committed_slot=cs, committed=self.committed is not None)
            if rl is not None:
                self.logits = rl
                self.pred_slot = int(rl.argmax())
        elif self.kind == "oavat":
            z = self.head.score(dets, inp.rgb, inp.rgb_hi, init_slot=lang_slot)
            self.logits = z
            tr["oavat_mode"] = self.head.mode                                  # G9(b′) diagnostics only
            if self.head.last_sims is not None:
                tr["oavat_top"] = int(np.argmax(self.head.last_sims))
            self.pred_slot = int(z.argmax()) if z.size and float(z.max()) > -1e8 else -1
        elif self.kind in ("deepsort", "ocsort"):
            ml = self.head.score(dets, init_slot=lang_slot)
            self.logits = ml
            self.pred_slot = int(ml.argmax()) if float(ml.max()) > -1e8 else -1
            tr.update(self._mot_trace())
        elif self.kind == "sticky":
            # 1016 §3 step 0 null ("never switch"): bound to the opening pick's key at every opening tick (also the
            # final one: the opening's accumulated conclusion; no opening module → the first grounding pick); after
            # the opening it votes only for that key and abstains when the key has no box. The shared commitment
            # rule applies unchanged, so the commitment can only move TOWARD the bound key, never away from it.
            if ("open" in tr or self.sticky_key is None) and 0 <= lang_slot < n:
                self.sticky_key = keys[lang_slot]
            s = next((i for i, k in enumerate(keys) if k == self.sticky_key), None)
            self.logits = np.full(n, -1e9, np.float32)
            if s is not None:
                self.logits[:] = 0.0
                self.logits[s] = 1.0
            self.pred_slot = -1 if s is None else s
            tr["sticky_key"] = self.sticky_key
        elif self.kind == "oracle":                                           # U1 diagnostic: the true target
            t = oracle.true_slot("oracle_head")
            if t >= 0:
                self.logits = np.zeros(n, np.float32)
                self.logits[t] = 1.0
                self.pred_slot = t
            else:
                self.pred_slot = -1
        elif self.kind == "tah":
            lt, conf = self.head.score(dets, self.W, self.H, rgb=inp.rgb)
            self.logits = lt.float().cpu().numpy()
            self.pred_slot = int(lt.argmax())
            self.pred_conf = conf
            if self.open_force:                                               # opening v4: commit the opening pick
                self.pred_slot = lang_slot
                self.open_force = False
            if self.head.cold():                                              # empty memory → the opening pick
                self.pred_slot = lang_slot
            if self.memory_seed == "gt":                                      # U2 diagnostic: GT-refreshed memory
                t = oracle.true_slot("gallery_seed_gt")
                self.head.write(t if t >= 0 else None)
            else:
                self.head.write(self._tah_write_slot(dets) if getattr(self, "tah_write", "own") == "committed"
                                else self.pred_slot)
        tr["mem"] = self.pred_slot
        tl = self.logits
        self.pred_conf = float(np.sort(tl)[-1] - np.sort(tl)[-2]) if tl.size >= 2 else 1.0
        if self.kind in ("deepsort", "ocsort", "sticky") and self.pred_slot < 0:
            self.pred_conf = 0.0                  # abstention is never confident (also with a single candidate)
        if shared is not None and 0 <= shared < n:
            # forced opening tick: identity output, commitment (→ WHERE) = the reference actor; PGIF accumulates the
            # same vote in every PGIF arm but cannot gate / re-pick during the window
            if self.pgif is not None:
                self.pgif.step(keys, shared, 1.0)
            self.pred_slot, self.pred_conf, self.committed = shared, 1.0, keys[shared]
            tr["forced"] = True
            if self.kind in ("deepsort", "ocsort"):   # the tracker stepped on this tick too (cost accounting)
                tr["t_identity_ms"] = round(1000 * (time.perf_counter() - t_id), 3)
            return tr
        if self.pgif is not None and n > 0 and 0 <= self.pred_slot < n:
            self._filter(dets, keys, oracle, tr)
        self._sync()
        tr["t_identity_ms"] = round(1000 * (time.perf_counter() - t_id), 3)          # 1006 §4.2 cost ①
        return tr

    def _mot_trace(self):
        """MOT heads: the bound track and the tracker time on the camera frames since the last S2 tick."""
        out = {"mot_bound": self.head.bound}
        if self.frame_input is not None:
            out["mot_frame_ms"] = round(self.mot_frame_ms, 3)
            self.mot_frame_ms = 0.0
        return out

    def _tah_write_slot(self, dets):
        """1007 opt (identity.tah.memory_write = committed): the memory learns the COMMITTED identity's box, as the
        re-ID gallery does; the own pick only while the memory is cold (opening hand-over) or nothing is committed;
        no write when the committed key has no box this tick (PGIF gating may still write the MAP detection)."""
        if self.committed is None or self.head.cold():
            return self.pred_slot
        return next((i for i, d in enumerate(dets) if d.key == self.committed), None)

    def _sync(self):
        if torch.cuda.is_available() and str(getattr(self, "device", "cpu")).startswith("cuda"):
            torch.cuda.synchronize()

    def _opening(self, inp, dets):
        """opening window: accumulate the opening scorer's per-detection scores by key → the opening pick."""
        zs = self.opening.scores(inp.rgb, dets, inp.language, inp.rgb_hi)
        if self.open_agg != "mean":
            zs = zs - zs.max()
        for d, z in zip(dets, zs):
            self.open_acc[d.key] = self.open_acc.get(d.key, 0.0) + float(z)
            self.open_cnt[d.key] = self.open_cnt.get(d.key, 0) + 1
        if self.open_agg == "last":
            pick = int(np.argmax(zs))
        else:
            den = (lambda k: self.open_cnt[k]) if self.open_agg == "mean" else (lambda k: 1)
            pick = max(range(len(dets)), key=lambda s: self.open_acc[dets[s].key] / den(dets[s].key))
        self.pred_slot = pick
        self.open_k += 1
        if self.open_window > 1 and (self.open_k < self.open_window or self.open_final_reset):
            if self.kind == "tah":
                self.head.reset_memory()          # cold memory → this tick's accumulated opening pick seeds it
            elif self.open_handoff == "uniform" and self.head is not None:
                self.head.reset()                 # 1006 F-c: assoc / dam4sam re-seeded from the same pick (init_slot)
        if self.open_reset_pgif and self.pgif is not None:
            self.pgif.belief, self.pgif.conf_hist = {}, []
        if self.open_k >= self.open_window:
            self.open_force = self.open_force_final
            if self.open_seed_pgif > 0 and self.pgif is not None:
                self.pgif.belief, self.pgif.conf_hist = {dets[pick].key: self.open_seed_pgif}, []
        return pick

    def _filter(self, dets, keys, oracle, tr):
        mconf = self.head.iou if self.kind == "dam4sam" else self.pred_conf
        orc = self.oracle_absent != "off" and oracle.true_slot(ORACLE_PURPOSE[self.oracle_absent]) < 0
        if orc:
            b0, c0 = dict(self.pgif.belief), list(self.pgif.conf_hist)
        kw = {"uv": [(float(d.u), float(d.v)) for d in dets]} if getattr(self.pgif, "inherit", False) else {}
        o = self.pgif.step(keys, self.pred_slot, mconf, **kw)
        if getattr(self.pgif, "last_inherit", None) is not None:
            tr["inherit"] = list(self.pgif.last_inherit)               # 1007 opt: (vanished key, successor)
        if orc:                                         # diagnostic: perfect "target not among the detections"
            self.pgif.belief, self.pgif.conf_hist = b0, c0
            if self.oracle_absent == "hygiene":
                if self.kind == "tah":
                    self.head.undo()
                if o.get("kind") == "gate":
                    o["gate_memory"] = False
            else:
                o = {"fire": False, "slot": None, "kind": "absent"}
            tr["oracle_absent"] = True
        tr["pgif"] = o.get("kind")
        if o.get("kind") == "absent":
            if self.gate_commit:
                self.pred_slot, self.pred_conf = -1, 0.0
            if self.kind == "tah":
                self.head.undo()
        if o["fire"] and o.get("kind") == "gate":
            if o.get("gate_memory", True):
                if self.kind == "tah":
                    self.head.gate(int(o["slot"]))
                elif self.head is not None:
                    self.head.gate(int(o["slot"]), dets)
            if self.gate_commit:
                self.pred_slot = int(o["slot"])
                if self.force_conf:
                    self.committed = dets[self.pred_slot].key
                    self.pred_conf = 1.0
                else:                                 # 1006 §5.2: evidence margin, normal commitment rule
                    ev = sorted((self.pgif.belief.get(int(k), 0.0) for k in keys), reverse=True)
                    self.pred_conf = float(ev[0] - ev[1]) if len(ev) > 1 else float(ev[0])
            self.events += 1
            tr["fire"] = True

    # ------------------------------------------------------------------------------------------------------ WHERE
    def _update_commit(self):
        dets = self.dets
        if dets is None or not (0 <= self.pred_slot < len(dets)):
            return
        sel = dets[self.pred_slot].key
        if self.committed is None or sel == self.committed:
            self.committed, self.chal, self.chal_n = sel, None, 0
        else:
            self.chal_n = self.chal_n + 1 if sel == self.chal else 1
            self.chal = sel
            if self.chal_n >= self.commit_k:
                self.committed, self.chal, self.chal_n = sel, None, 0

    def _committed_world(self):
        if self.committed is None or self.dets is None:
            return None
        d = next((d for d in self.dets if d.key == self.committed), None)
        if d is None or d.world_xyz is None:
            return None
        return np.asarray(d.world_xyz, np.float64)

    def _in_frame(self, world_xyz, campose):
        from acot_probe.projection import project_point
        pr = project_point(tuple(float(x) for x in world_xyz), tuple(float(x) for x in campose), self.W, self.H,
                           self.fov)
        if pr is None:
            return False
        u, v, _ = pr
        return bool(0 <= u < self.W and 0 <= v < self.H)

    def _where(self, inp):
        """commit (confident picks only), then fly toward the committed key's matched box (CV waypoints); when the
        committed key has no box this tick the EAR sample stays (legacy deploy-v2 behaviour) unless
        control.absent_where = cv: then the estimator extrapolates the last confident position at constant velocity."""
        confident = self.pred_conf >= self.conf_tau
        if confident:
            self._update_commit()
        cw = self._committed_world()
        self.frame_cw = cw if (confident and cw is not None) else None
        if cw is not None:
            if confident and self._in_frame(cw, inp.campose):
                self.estimator.update(cw, inp.step)
            if self.estimator.seen:
                wp = self.estimator.predict_wp(inp.campose, inp.step, mode="cv")
                if wp is not None:
                    self.z_ex = torch.from_numpy(wp)[None].to(self.device)
        elif self.absent_where == "cv" and self.committed is not None and self.estimator.seen:
            wp = self.estimator.predict_wp(inp.campose, inp.step, mode="cv")
            if wp is not None:
                self.z_ex = torch.from_numpy(wp)[None].to(self.device)
