"""Identity heads of the acot runtime (framework §7.2, P2) — the memories that pick WHICH detection is the target.
Ports of the legacy Policy branches (carla_uav_tracking/rollout/policy.py) for memory_seed=committed, i.e. the memory
is initialised from the opening pick and then updated with the policy's own belief, never with a label:
  TahHead      learned TAHRelM association (tid_head=tah)
  AssocHead    DeepSORT / OC-SORT-style motion (+ appearance) association (tid_head=assoc)
  Dam4samHead  DAM4SAM (distractor-aware SAM2 memory) over the socket service (tid_head=dam4sam)
  ReidHead     K-view appearance gallery re-ID (+ optional motion fusion / per-key temporal EMA) (tid_head=reid;
               1006 G8 port of Policy._reid_select / the reid branch of Policy.act, deployable gallery)
A head sees Detections only. Hooks used by the policy: score(...), write(slot) (tah), gate(slot, ...) = posterior-gated
memory update (PGIF), undo() = drop this tick's memory write (PGIF absence)."""
import numpy as np
import torch


class TahHead:
    name = "tah"

    def __init__(self, ckpt, device, log=print, feat_model=None):
        from train.tah import TAHRel, TAHRelM
        self.feat_model = feat_model          # None = the shared crop-DINOv2 feature (Detection.crop_feat)
        tck = torch.load(ckpt, map_location=device, weights_only=False)
        cls = TAHRelM if tck.get("arch") == "relm" else TAHRel
        self.net = cls(d_feat=tck.get("d_feat", 384)).to(device).eval()
        self.net.load_state_dict(tck["model"])
        self.device = device
        log(f"[acot-policy] {cls.__name__} ({sum(p.numel() for p in self.net.parameters())} params) from {ckpt}")
        self.reset()

    def reset(self):
        self.mem = self.net.reset(self.device)
        self.prev = None
        self.last = None

    def reset_memory(self):
        self.mem = self.net.reset(self.device)

    def cold(self):
        return self.mem is None or self.mem.get("app") is None

    def score(self, dets, W, H, rgb=None):
        """→ (logits tensor (n,), learned confidence)."""
        if self.feat_model is not None:       # 1006 backbone control: same crops (224 px, 1.3× box) via DINOv3
            from train.reid_resolution_probe import _dino_feats
            fe = _dino_feats(self.feat_model, rgb, dets, W, H, 224, 1.3, self.device).astype(np.float64)
            fe = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-8)
        else:
            fe = np.stack([d.crop_feat for d in dets])
        feat = torch.tensor(fe, dtype=torch.float32, device=self.device)
        pos = torch.tensor(np.array([[d.u / W, d.v / H, d.depth_m / 100.0] for d in dets], np.float32),
                           device=self.device)
        self.last = (feat, pos)
        logits, conf, _ = self.net.step(feat, pos, self.mem)
        return logits, float(conf)

    def write(self, slot):
        """the memory learns the believed target of this tick (the snapshot allows gating / undo)."""
        feat, pos = self.last
        self.prev = self.mem
        if slot is not None and 0 <= slot < feat.shape[0]:
            self.mem = self.net.update(self.mem, feat[slot], pos[slot])

    def gate(self, slot):
        """posterior gating: redo this tick's update with the filter's MAP detection."""
        if self.last is None:
            return
        feat, pos = self.last
        prev = self.prev if self.prev is not None else self.net.reset(self.device)
        self.mem = self.net.update(prev, feat[slot], pos[slot])

    def undo(self):
        if self.prev is not None:
            self.mem = self.prev


class AssocHead:
    name = "assoc"

    def __init__(self, mode, gate_px, lam):
        assert mode in ("deepsort", "motion"), mode
        self.mode, self.gate_px, self.lam = mode, float(gate_px), float(lam)
        self.reset()

    def reset(self):
        self.uv = self.vel = self.feat = None

    def score(self, dets, init_slot):
        """select with the PAST track state, then update it with this tick's own association (the opening pick
        before any track state exists) → (n,) scores (−1e9 outside the motion gate)."""
        uv = np.array([[d.u, d.v] for d in dets], np.float64)
        n = len(dets)
        if self.uv is not None:
            pred = self.uv + (self.vel if self.vel is not None else 0.0)
            dist = np.linalg.norm(uv - pred[None], axis=1)
            score = -dist / self.gate_px
            gate = dist < self.gate_px
        else:
            score = np.zeros(n)
            gate = np.ones(n, bool)
        fn = None
        if self.mode == "deepsort":
            fn = np.stack([d.crop_feat for d in dets])
            if self.feat is not None:
                score = score + self.lam * (fn @ self.feat)
        if self.uv is not None and gate.any():
            score = np.where(gate, score, -1e9)
        if self.uv is None:
            ti = init_slot
        else:
            ti = int(score.argmax()) if float(score.max()) > -1e8 else None
        if ti is not None and ti >= 0:
            self.vel = (uv[ti] - self.uv) if self.uv is not None else np.zeros(2)
            self.uv = uv[ti].copy()
            if fn is not None:
                t = fn[ti]
                self.feat = t if self.feat is None else \
                    (0.9 * self.feat + 0.1 * t) / (np.linalg.norm(0.9 * self.feat + 0.1 * t) + 1e-8)
        return score.astype(np.float32)

    def gate(self, slot, dets):
        pass                                  # no memory to re-seed (legacy _asd_gate_memory is a no-op for assoc)

    def undo(self):
        pass


class Dam4samHead:
    name = "dam4sam"

    def __init__(self, host, port, gate_px, every_frame, reinit_cooldown, log=print):
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, int(port)))
        self.gate_px, self.every_frame, self.cooldown = float(gate_px), bool(every_frame), int(reinit_cooldown)
        log(f"[acot-policy] connected to DAM4SAM {host}:{port} (every_frame={self.every_frame})")
        self.reset()

    def reset(self):
        self.inited = False
        self.iou = 0.0
        self.cool = 0
        self.rgb = None
        self.ef_n = self.ef_empty = 0

    def _rpc(self, msg):
        from rollout.bridge import recv_msg, send_msg
        send_msg(self.sock, msg)
        return recv_msg(self.sock)

    def advance(self, rgb):
        """every-frame mode: feed an intermediate frame (memory update only, no identity decision)."""
        r = self._rpc({"cmd": "track", "rgb": np.asarray(rgb, np.uint8)})
        self.ef_n += 1
        self.ef_empty += int(r.get("bbox") is None)

    def score(self, dets, rgb, init_slot):
        n = len(dets)
        bbox = None
        if not self.inited:
            if init_slot is not None and init_slot >= 0:
                d = dets[init_slot]
                bbox = self._rpc({"cmd": "init", "rgb": np.asarray(rgb, np.uint8),
                                  "box": (d.u - d.w / 2.0, d.v - d.h / 2.0, d.w, d.h)}).get("bbox")
                self.inited = True
                self.iou = 0.5
            else:
                return np.full(n, -1e9, np.float32)
        else:
            r = self._rpc({"cmd": "track", "rgb": np.asarray(rgb, np.uint8)})
            bbox = r.get("bbox")
            iou = r.get("iou")
            self.iou = float(iou) if iou is not None and iou == iou else 0.0
        self.rgb = rgb
        self.cool = max(0, self.cool - 1)
        if bbox is None:
            return np.full(n, -1e9, np.float32)
        bx, by, bw, bh = bbox
        pc = np.array([bx + bw / 2.0, by + bh / 2.0])
        uv = np.array([[d.u, d.v] for d in dets], np.float64)
        dist = np.linalg.norm(uv - pc[None], axis=1)
        return np.where(dist < self.gate_px, -dist / self.gate_px, -1e9).astype(np.float32)

    def gate(self, slot, dets):
        """posterior gating: re-initialise SAM2 on the filter's MAP box (cool-down between re-inits)."""
        if self.rgb is None or self.cool != 0:
            return
        d = dets[slot]
        self._rpc({"cmd": "init", "rgb": np.asarray(self.rgb, np.uint8),
                   "box": (d.u - d.w / 2.0, d.v - d.h / 2.0, d.w, d.h)})
        self.cool = self.cooldown

    def undo(self):
        pass


class ReidHead:
    """Deployable appearance-gallery re-ID (legacy `tid_head=reid --gallery-seed committed`): SELECT with the PAST
    gallery, then store the view of the identity the policy currently believes is the target (the opening / language
    pick before any commitment — or right after an opening hand-over reset — else the committed key's box)."""
    name = "reid"

    def __init__(self, feature, bank, topm, tavg, motion, motion_gate_px):
        assert feature in ("dinov2", "vlmpool"), feature
        self.feature, self.K, self.topm = feature, int(bank), int(topm)
        self.tavg, self.motion, self.gate_px = float(tavg), float(motion), float(motion_gate_px)
        self.reset()

    def reset(self):
        self.bank, self.ema, self.mvu, self.mvel, self.fn = [], {}, None, None, None
        self.fresh = True                     # just reset → the next store is the hand-over / opening pick

    def _feats(self, dets):
        f = np.stack([d.crop_feat if self.feature == "dinov2" else d.vlm_feat for d in dets]).astype(np.float64)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

    def score(self, dets, init_slot, committed_slot, committed):
        """→ (n,) scores, or None while the gallery is empty (the policy keeps the language / opening pick)."""
        from train.reid_resolution_probe import bank_score, bank_update
        fn = self._feats(dets)
        rl = bank_score(self.bank, fn, self.topm) if self.bank else None
        self.fn = fn
        store = init_slot if (not committed or self.fresh) else committed_slot
        if store is not None and 0 <= store < len(fn):
            bank_update(self.bank, fn[store], self.K, 0.9)
            self.fresh = False
        if rl is None:
            return None
        if self.motion > 0:
            uv = np.array([[d.u, d.v] for d in dets], np.float64)
            ms = np.zeros(len(dets)) if self.mvu is None else \
                -np.linalg.norm(uv - (self.mvu + (self.mvel if self.mvel is not None else 0.0))[None], axis=1) / self.gate_px
            rl = rl + self.motion * ms
            if store is not None and 0 <= store < len(dets):
                nuv = uv[store]
                self.mvel = (nuv - self.mvu) if self.mvu is not None else np.zeros(2)
                self.mvu = nuv
        if self.tavg > 0:
            a = self.tavg
            for s, d in enumerate(dets):
                prev = self.ema.get(d.key, float(rl[s]))
                self.ema[d.key] = a * prev + (1 - a) * float(rl[s])
            rl = np.array([self.ema.get(d.key, -1e9) for d in dets], np.float64)
        return np.asarray(rl, np.float32)

    def gate(self, slot, dets):
        """posterior gating = no-op, as legacy _asd_gate_memory for reid: the gallery stores the COMMITTED identity,
        which PGIF overrides, so the filter acts on the gallery through the commitment from the next tick on (the
        gallery re-seed of legacy _asd_reset_memory belongs to the non-gating ASD modes, not to PGIF)."""
        return

    def undo(self):
        pass
