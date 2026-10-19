"""Load the 4-module Stage-2 policy and run it online, one tick at a time.

Assembles EAR + IAR + DiT + TargetIDHead from a `stage2_best.pt` (dims block +
training cfg["model"]) — there is no existing loader (guide §4). Per tick:

  S2 (slow, ~1 Hz): VLM encode → EAR z_ex (flow sample) → IAR z_im → tid logits.
  S1 (fast, 10 Hz): DiT flow-sample the action chunk on fresh proprio; take step 0.

`s2_period` controls the split. DEFAULT 1 = run the full pipeline every tick
(correct-first; CARLA is sync so wall-clock, not realtime, is the only cost). Set
>1 for true async — the DiT was trained with staleness noise on z_ex to tolerate a
stale slow-stream (guide §4). The target-id choice is NOT used for control (the
action is conditioned on z_ex/z_im/vlm_ctx/proprio, not tid); it is returned raw
for the scorer, which applies the K-frame hysteresis (guide §6, metrics.py).

Import root: repo root on sys.path (train.*, acot_probe.*).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from train.ear import EAR
from train.iar import IAR
from train.dit import DiT, TargetIDHead
from train import flow_matching
from train.stage2 import action_sample
from train.grounding_train import AttrBindHead, build_attr_vocab

from rollout.candidates import build_candidates, CandidateSet
from rollout.target_state import TargetStateEstimator

# --ex-source: what fills DiT's z_ex slot. 'ear' = the trained EAR (default). The
# rest inject a training-free memory+CV prior (rollout.target_state); '*_gated' only
# during a loss (EAR when visible), bare 'cv'/'zerovel' always (diagnostic).
# 'track' (Plan A): grounding-in-the-loop — z_ex is seeded from the tid-COMMITTED
# candidate (K-frame hysteresis), so tid QUALITY drives control (a wrong pick tracks
# the wrong car). The others seed from the GT target (tid stays scorer-only).
# 'road' (P2a-1): the road-graph WM prior — z_ex during a loss is the env-side
# lane-traversal prediction (gt_candidates[3], WORLD (K,3)) projected into cam frame,
# vs cv_gated's straight-line dead-reckon. Same last-seen anchor+speed → isolates
# follow-the-lane vs go-straight (offline M2 gate: road << cv_frame on turns/junctions).
_EX_MODES = {"ear": (None, None), "cv": ("cv", False), "cv_gated": ("cv", True),
             "zerovel": ("zerovel", False), "zerovel_gated": ("zerovel", True),
             "road": ("road", True), "track": ("cv", False)}


@dataclass
class Observation:
    rgb: np.ndarray          # (H,W,3) uint8 — sole perceptual input
    language: str            # fixed per episode (raw; vlm applies the language mode)
    proprio: np.ndarray      # (5,)
    step: int
    target_color: str = ""   # GT target identity (color+blueprint) — for the attrbind tid head
    target_bp: str = ""      # (privileged, same level as GT candidate boxes; = offline attr upper bound)
    retarget_event: bool = False   # 1003_dynamic-retarget-design.md §2.1: True on the exact tick
                                    # the scene's target was just reassigned (scripted, S1 GT-driven)
    rgb_hi: np.ndarray | None = None   # 1004 opening fix: k× resolution frame of the same camera (opening ticks)


@dataclass
class ActInfo:
    tid_logits: np.ndarray   # (N,) target-id scores over candidates
    pred_slot: int           # argmax candidate slot
    cand_set: CandidateSet | None   # carries true_idx + per-slot identity for the scorer
    z_ex: np.ndarray         # (K,3) EAR waypoint sample (diagnostic)


def load_policy_modules(ckpt_path: str, cfg: dict, device: str = "cuda:0"):
    """Reconstruct EAR/IAR/DiT/TargetIDHead from a stage2 ckpt (guide §4).

    Returns (ear, iar, dit, tid, dims, meta), all on `device` in eval mode. Hidden
    dims come from cfg["model"]; tensor dims from the ckpt's `dims` block.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    d, m = ck["dims"], cfg["model"]
    ear = EAR(d["cond_dim"], k=d["K"], d_model=m["ear"]["d_model"],
              n_layers=m["ear"]["n_layers"], n_heads=m["ear"]["n_heads"],
              proprio_dim=d["pdim"])
    iar = IAR(d["cond_dim"], d["n_layers_in"], d=m["iar"]["d"],
              n_im=m["iar"]["n_im"], n_heads=m["iar"]["n_heads"])
    dit = DiT(d["A"], d["H"], d["cond_dim"], d_ex=3, d_im=m["iar"]["d"],
              proprio_dim=d["pdim"], d=m["dit"]["d_model"],
              n_layers=m["dit"]["n_layers"], n_heads=m["dit"]["n_heads"])
    tid = TargetIDHead(d["cand_dim"], d["cond_dim"], d=m.get("tid_d", 256))
    ear.load_state_dict(ck["ear"]); iar.load_state_dict(ck["iar"])
    dit.load_state_dict(ck["dit"]); tid.load_state_dict(ck["tid"])
    mods = tuple(mod.to(device).eval() for mod in (ear, iar, dit, tid))
    meta = {k: ck.get(k) for k in ("epoch", "metrics")}
    return (*mods, d, meta)


class Policy:
    def __init__(self, ckpt_path: str, cfg: dict, vlm, *, device: str = "cuda:0",
                 s2_period: int = 1, shuffle_seed: int = 0, ablate: str | None = None,
                 ex_source: str = "ear", commit_k: int = 5,
                 tid_head: str = "xattn", tid_ckpt: str | None = None,
                 reid_feature: str = "vlmpool", reid_bank: int = 1, reid_topm: int = 1,
                 conf_tau: float = 0.0, conf_consensus: float = 0.0, frame_gain: float = 0.0, warmup_oracle_s: float = 0.0,
                 reid_tavg: float = 0.0, reid_motion: float = 0.0, reid_motion_gate_px: float = 160.0,
                 gallery_seed: str = "gt", reanchor: str = "off",
                 reanchor_patience: int = 5, reanchor_tavg: float = 0.0, assoc_mode: str = "deepsort",
                 assoc_gate_px: float = 160.0, assoc_lambda: float = 1.0,
                 dam4sam_host: str = "dam4sam-svc", dam4sam_port: int = 5601,
                 tah_ckpt: str = "runs/tah_rel.pt", asd: str = "off", asd_cfg: str = "",
                 tracklets: str = "sim", occl_cands: bool = False, event_override: bool = False):
        self.device = device
        self.cfg = cfg
        self.vlm = vlm                              # OnlineVLM (owns the frozen backbone + language mode)
        self.image_cfg = cfg["image"]
        self.a_scale = float(cfg["action"]["scale"])
        self.steps = int(cfg["flow"]["sample_steps"])
        self.s2_period = int(s2_period)
        self._shuffle_seed = int(shuffle_seed)
        # H1/H2 module ablation: zero the EAR/IAR stream into the DiT, matching a ckpt
        # trained with the same ablation (train/config_v5_no{ear,iar}.yaml). None = full.
        self.ablate = ablate                        # None | "ear" | "iar"
        # memory+CV target-state prior for z_ex during a loss (rollout.target_state).
        if ex_source not in _EX_MODES:
            raise ValueError(f"ex_source must be one of {list(_EX_MODES)}, got {ex_source!r}")
        self.ex_source = ex_source
        self._ex_mode, self._ex_gated = _EX_MODES[ex_source]   # (None,None) for 'ear'
        self._track = (ex_source == "track")                   # Plan A: grounding drives control
        self.commit_k = int(commit_k)                          # hysteresis frames to switch target
        self.conf_tau = float(conf_tau)                        # ①: min pick-margin to commit-fly (track)
        self.conf_consensus = float(conf_consensus)            # borrow#2: also require the pick's motion-consistency
                                                               # (dist < conf_consensus·gate) to commit-fly (0=off)
        self.frame_gain = float(frame_gain)                    # ②: yaw-centering gain on committed target
        self.warmup_oracle_s = float(warmup_oracle_s)          # diag: first N s use oracle id (est. tracking)
        self.reid_tavg = float(reid_tavg)                      # temporal re-ID: per-actor EMA of match score
        self.reid_motion = float(reid_motion)                  # borrow#1: motion-consensus weight fused into reid score (0=off)
        self.reid_motion_gate_px = float(reid_motion_gate_px)  # borrow#1: motion soft-gate radius (px)
        # deployable identity (paper2): gallery_seed 'gt'=legacy oracle-seeded template |
        # 'committed'=GT-free (store the believed-target crop). reanchor 'off'|'lang'=language re-anchor
        # on sustained collapse |'oracle'=GT re-anchor (headroom bound). See memory acot-uav-reid-reframe.
        self.gallery_seed = gallery_seed
        self.reanchor = reanchor
        self.reanchor_patience = int(reanchor_patience)        # consecutive low-margin S2 ticks before re-anchor fires
        self.reanchor_tavg = float(reanchor_tavg)              # temporal language re-anchor: per-actor EMA of language logit (0=single-frame)
        # 1003_dynamic-retarget-design.md §2.3 S1 (GT-driven): on obs.retarget_event, hard-reset
        # TAHRelM memory + PGIF belief to the (currently GT) new target instead of waiting for
        # evidence to out-accumulate the old target. Independent of `reanchor` (different trigger:
        # external event, not detected drift) — only wired for tid_head="tah"/"fusion" + committed.
        self.event_override = bool(event_override)
        self._override_events = 0
        self.fps = int(cfg.get("fps", 10))
        wp = cfg["waypoint"]
        self.estimator = TargetStateEstimator(wp["offsets_s"], wp["scale"], cfg.get("fps", 10))

        # DAgger collection: if DAGGER_DIR is set, log per-frame (cand_feats, true_idx,
        # target color/make) from the closed-loop distribution → retrain the tid on the
        # frames the policy actually visits (oracle label = GT true_idx, free).
        self._dagger_dir = os.environ.get("DAGGER_DIR")
        # path-B DAgger: log DEPLOYMENT-distribution TAH inputs (crop-DINOv2 feat + pos + GT id) to retrain
        # TAHRel on-policy (fixes the offline→online distribution shift). Set TAH_DAGGER_DIR to collect.
        self._tahdag_dir = os.environ.get("TAH_DAGGER_DIR"); self._tahdag_buf = []; self._tahdag_ep = 0
        self._dagger_ep = 0
        self._dagger_buf = []

        self.ear, self.iar, self.dit, self.tid, d, self.ckpt_meta = \
            load_policy_modules(ckpt_path, cfg, device)
        self.K, self.H, self.A = int(d["K"]), int(d["H"]), int(d["A"])
        # Optional: swap the (decoupled) tid head for the standalone-trained attrbind head
        # (CLIP-style identity match). No Stage-2 retrain — tid grads touch nothing else.
        # Vocab is rebuilt deterministically (build_attr_vocab = sorted attrs) so ids match.
        self.tid_head = tid_head
        # reid appearance feature + K-view gallery (offline milestone: dinov2+K2 breaks 0.5).
        self.reid_feature = reid_feature; self.reid_bank = int(reid_bank); self.reid_topm = int(reid_topm)
        self._dino = None
        # 1004 deployable v2: tracklets='tracker' → GT-free tracklet ids (rollout/tracklets.py) replace the simulator
        # actor index as the identity key everywhere (PGIF, commitment, gallery, WHERE); occl_cands → drop candidates
        # the env's ray-cast marks as occluded (needs run_rollout --vis-raycast). Defaults = legacy behaviour.
        assert tracklets in ("sim", "tracker"), tracklets
        self.tracklets = tracklets; self.occl_cands = bool(occl_cands)
        self._tracker = None; self._trk_log = None
        if tracklets == "tracker":
            from rollout.tracklets import TrackletAssigner
            # TRK_CFG (JSON kwargs) lets the tracker be tuned on dev seeds (perception module); default = pre-declared
            self._tracker = TrackletAssigner(**json.loads(os.environ.get("TRK_CFG") or "{}"))
            print(f"[policy] tracker cfg={os.environ.get('TRK_CFG') or '{}'}", flush=True)
            if os.environ.get("ASD_LOG"):
                self._trk_log = open(os.environ["ASD_LOG"].replace(".jsonl", "") + ".trk.jsonl", "a")
        print(f"[policy] tracklets={tracklets} occl_cands={self.occl_cands}", flush=True)
        # 1004 opening fix: OPENING=zoom → the episode's OPENING identity (first `window` S2 ticks with candidates) is
        # chosen by zoom-and-verify (rollout/opening_zoom.py: per-candidate crop → frozen Qwen3-VL yes/no log-odds)
        # instead of the grounding head's argmax; it replaces `lang_slot`, so every head's cold start uses it.
        self._zoom = None; self._open_window = 1
        if os.environ.get("OPENING", "head") in ("zoom", "attr"):
            # OPENING=attr: trained colour + model classifiers on DINOv2 crops (opening_cls.py; same scorer interface)
            from rollout.opening_zoom import AttrScorer, ZoomVerifier
            ocfg = json.loads(os.environ.get("OPENING_CFG") or "{}"); self._open_window = int(ocfg.pop("window", 1))
            # opening-window fixes (default off = the validated v1 behaviour): agg=mean → per-id MEAN of raw scores
            # over the ticks the id was seen (a tracker id that appears mid-window is not favoured); agg=last → each
            # opening tick decides from its OWN scores (tracker id swaps inside the window cannot carry a distractor
            # the target's history; the last opening tick's pick is the commitment); final_reset →
            # the TAH memory is also reset at the LAST opening tick, so the accumulated opening pick (not the memory's
            # own argmax) seeds it; reset_asd → PGIF evidence is cleared at every opening tick, so the posterior
            # starts from the final opening pick instead of resisting it after a tracker id switch.
            self._open_agg = str(ocfg.pop("agg", "sum")); self._open_final_reset = bool(ocfg.pop("final_reset", False))
            self._open_reset_asd = bool(ocfg.pop("reset_asd", False))
            # v4: force_final → at the LAST opening tick the TAH branch commits the opening pick (instead of its own
            # argmax) and the memory is UPDATED with it (keeps the earlier opening sample: 2 samples, as v1);
            # seed_asd=P → PGIF evidence is re-seeded to {final opening id: P} (inertia, no per-id history).
            self._open_force_final = bool(ocfg.pop("force_final", False)); self._open_seed_asd = float(ocfg.pop("seed_asd", 0.0))
            self._zoom = (AttrScorer if os.environ["OPENING"] == "attr" else ZoomVerifier)(
                vlm.bk.model, vlm.bk.processor, device, **ocfg)
            print(f"[policy] OPENING={os.environ['OPENING']} cfg={os.environ.get('OPENING_CFG') or '{}'} window={self._open_window}", flush=True)
        _need_dino = (tid_head == "reid" and reid_feature == "dinov2") or \
                     (tid_head == "assoc" and assoc_mode == "deepsort") or (tid_head in ("tah", "fusion")) or (asd != "off") \
                     or tracklets == "tracker"
        if _need_dino:
            from train.reid_resolution_probe import _load_dino
            self._dino = _load_dino("vit_small_patch14_dinov2.lvd142m", device)
            print(f"[policy] DINOv2 crop features loaded (tid_head={tid_head})", flush=True)
        # external-baseline: DeepSORT/OC-SORT-style tracking-by-detection association
        self.assoc_mode = assoc_mode; self.assoc_gate_px = float(assoc_gate_px); self.assoc_lambda = float(assoc_lambda)
        # path-B: TAHRel learned association head (replaces the heuristic WHICH stack)
        self._tah = None
        if tid_head in ("tah", "fusion"):
            from train.tah import TAHRel, TAHRelM
            tck = torch.load(tah_ckpt, map_location=device, weights_only=False)
            _cls = TAHRelM if tck.get("arch") == "relm" else TAHRel   # path-A motion-augmented vs base
            self._tah = _cls(d_feat=tck.get("d_feat", 384)).to(device).eval()
            self._tah.load_state_dict(tck["model"])
            print(f"[policy] {_cls.__name__} loaded ({sum(p.numel() for p in self._tah.parameters())} params) from {tah_ckpt}", flush=True)
        self._tah_mem = None
        self._cand_set_cur = None
        self.cold_window = int(os.environ.get("COLD_WINDOW", "0"))           # V2: deferred-commitment ticks (0 = off)
        self.cold_lambda = float(os.environ.get("COLD_LAMBDA", "0.9"))       # V2: tempering of the language votes
        self.cold_single = os.environ.get("COLD_SINGLE", "0") == "1"         # V2 control: commit single-frame pick at K
        self.v1_selfreset = os.environ.get("V1_SELFRESET", "0") == "1"       # V1 control: re-init on own pick, no identity change
        # ASD (Anchored Sequential Drift-correction): memory-agnostic confident-drift detector wrapped around
        # whichever identity memory is active (tah / reid). asd_cfg = JSON kwargs for rollout.asd.ASD.
        self._asd = None
        if asd != "off":
            import json as _json
            from rollout.asd import ASD
            self._asd = ASD(mode=asd, log_path=os.environ.get("ASD_LOG") or None,
                            **(_json.loads(asd_cfg or os.environ.get("ASD_CFG") or "{}")))
            print(f"[policy] ASD mode={asd} cfg={asd_cfg or os.environ.get('ASD_CFG') or '{}'} log={os.environ.get('ASD_LOG')}", flush=True)
        # DAM4SAM abstention diagnosis (DAM4SAM_DIAG=1): per S2 tick, WHY no candidate was picked (not init'd / empty
        # mask / nearest candidate outside the association gate) + geometry, to ASD_LOG. Logging only — no decision change.
        self._dam_diag = None; self._dam_ep = -1; self._dam_k = 0
        # G11 (1004): DAM4SAM_EVERY_FRAME=1 → feed EVERY frame to SAM2 (memory update only); the identity decision is
        # still read at S2 ticks. Default (0) = legacy S2-rate tracking (every s2_period-th frame).
        self._dam4sam_every_frame = os.environ.get("DAM4SAM_EVERY_FRAME") == "1"
        self._dam_ef_n = self._dam_ef_empty = 0
        if self._dam4sam_every_frame:
            print("[policy] DAM4SAM every-frame tracking ON (G11)", flush=True)
        if os.environ.get("DAM4SAM_DIAG") == "1" and os.environ.get("ASD_LOG"):
            self._dam_diag = open(os.environ["ASD_LOG"], "a")
            print(f"[policy] DAM4SAM diag log={os.environ['ASD_LOG']} gate={self.assoc_gate_px}px", flush=True)
        # external-baseline: DAM4SAM (SOTA distractor-aware SAM2 memory), served over a socket bridge
        self._dam4sam_sock = None
        if tid_head in ("dam4sam", "fusion"):
            import socket as _sk
            self._dam4sam_sock = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
            self._dam4sam_sock.connect((dam4sam_host, int(dam4sam_port)))
            print(f"[policy] connected to DAM4SAM service {dam4sam_host}:{dam4sam_port}", flush=True)
        self._cvoc = self._mvoc = None
        if tid_head == "attrbind":
            assert tid_ckpt, "tid_head=attrbind requires --tid-ckpt (the standalone attrbind head)"
            self._cvoc, self._mvoc = build_attr_vocab(cfg)
            m = cfg["model"]
            self.tid = AttrBindHead(d["cand_dim"], len(self._cvoc), len(self._mvoc),
                                    d=m.get("tid_d", 256)).to(device)
            tck = torch.load(tid_ckpt, map_location=device, weights_only=False)
            self.tid.load_state_dict(tck["tid"]); self.tid.eval()
            print(f"[policy] attrbind tid loaded from {tid_ckpt} "
                  f"(mis_follow={tck.get('mis_follow')}, n_color={len(self._cvoc)}, n_make={len(self._mvoc)})",
                  flush=True)
        self.reset()

    def reset(self):
        """Clear slow-stream caches + reseed the candidate shuffle for a new episode."""
        if getattr(self, "reanchor", "off") != "off" and getattr(self, "_reanchor_events", 0):
            print(f"[reanchor] episode fired {self._reanchor_events} re-anchors", flush=True)
        if getattr(self, "event_override", False) and getattr(self, "_override_events", 0):
            print(f"[event-override] episode fired {self._override_events} event-driven overrides", flush=True)
        self._override_events = 0
        self._rng = np.random.default_rng(self._shuffle_seed)
        self._z_ex = self._z_im = self._vlm_ctx = self._ctx_mask = None
        self._tid_logits = np.zeros(0, np.float32)
        self._pred_slot = -1
        self._cand_set = None
        self._committed_idx = None      # Plan A: committed target identity (-1=target, d=distractor)
        self._chal = None; self._chal_n = 0
        self._reid_bank = []            # reid: K-view diversity gallery of the tracked target
        self._reid_score = {}           # temporal re-ID: actor_idx -> EMA of gallery-match score
        self._reid_mvu = None           # borrow#1: last target image pos (2,) for motion consensus
        self._reid_mvel = None          # borrow#1: target image-space velocity (2,)
        self._reid_mscore = None        # borrow#2: stashed per-candidate motion score for the consensus gate
        self._tah_mem = self._tah.reset(self.device) if getattr(self, "_tah", None) is not None else None  # path-B TAHRel memory
        self._tah_anchor = None         # confident-drift: FROZEN initial target appearance (drift-immune anchor)
        self._lang_score = {}           # temporal language re-anchor: actor_idx -> EMA of language logit
        self._assoc_uv = None           # assoc baseline: last GT-target image pos (2,)
        self._assoc_vel = None          # assoc baseline: image-space velocity (2,)
        self._assoc_feat = None         # assoc baseline: appearance template (EMA, C,)
        self._dam4sam_inited = False     # dam4sam baseline: has the tracker been init'd this episode
        self._dam_ep = getattr(self, "_dam_ep", -1) + 1; self._dam_k = 0                  # DAM4SAM diag counters
        self._dam_ef_n = self._dam_ef_empty = 0                                            # G11 per-frame counters
        if getattr(self, "_tracker", None) is not None:                                   # 1004: GT-free tracklets
            self._tracker.reset()
        self._trk_ep = getattr(self, "_trk_ep", -1) + 1; self._trk_k = 0
        self._dump_ep = getattr(self, "_dump_ep", -1) + 1                                  # OPENING_DUMP episode counter
        self._open_k = 0; self._open_acc = {}; self._open_cnt = {}; self._open_force = False  # zoom opening state
        self._dam4sam_iou = 0.0; self._dam4sam_cool = 0; self._dam4sam_rgb = None   # V1: PGIF∘DAM4SAM state
        self._fus_dam = None                                                  # V3: DAM4SAM pick this tick
        self._sticky_idx = None                                               # Sticky-ID null: actor locked at cold start
        self._cold_n = 0; self._cold_G = {}; self._cold_hist = []; self._cold_active = False   # V2: deferred commitment
        self._pred_conf = 0.0           # ①: current pick top1−top2 margin
        self._frame_cw = None           # ②: committed target world pos to keep centered
        self._reid_fn = None            # deployable: stashed normalized crop feats (for re-anchor re-seed)
        self._lowconf_run = 0           # deployable re-anchor: consecutive low-margin S2 ticks
        self._reanchor_events = 0       # deployable re-anchor: fired-count this episode (diagnostic)
        if getattr(self, "_asd", None) is not None:
            if self._asd.tick:
                print(f"[asd] episode fired {self._asd.fires} drift corrections over {self._asd.tick} ticks", flush=True)
            self._asd.reset()
        self._asd_fe = None             # ASD: this tick's L2-normed candidate appearance feats
        self._tah_last = None           # ASD: (feat, pos) tensors of this tick (tah memory reset hook)
        self.estimator.reset()
        self._dagger_flush()            # DAgger: flush the finished episode's frames, start fresh
        self._dagger_buf = []
        self._tahdag_flush()            # path-B DAgger: flush deployment-distribution TAH frames

    def _dagger_flush(self):
        """Write the finished episode's collected frames to DAGGER_DIR (one .npz/episode)."""
        if not self._dagger_dir or not getattr(self, "_dagger_buf", None):
            return
        os.makedirs(self._dagger_dir, exist_ok=True)
        feats = [b[0] for b in self._dagger_buf]           # list of (N_i, C) — ragged
        tgt = np.array([b[1] for b in self._dagger_buf], np.int64)
        col = np.array([b[2] for b in self._dagger_buf])
        mk = np.array([b[3] for b in self._dagger_buf])
        n = np.array([f.shape[0] for f in feats], np.int64)
        out = os.path.join(self._dagger_dir, f"dagger_ep{self._dagger_ep:04d}.npz")
        np.savez_compressed(out, feats=np.concatenate(feats, 0).astype(np.float16),
                            n=n, true_idx=tgt, color=col, make=mk)
        print(f"[dagger] wrote {out}  ({len(feats)} frames)", flush=True)
        self._dagger_ep += 1

    def _tahdag_flush(self):
        """path-B DAgger: write the finished episode's deployment-distribution TAH frames (one .pt/ep,
        same schema as train/tah_cache.py so tah_train can consume it directly)."""
        if not self._tahdag_dir or not getattr(self, "_tahdag_buf", None):
            return
        import torch as _t
        os.makedirs(self._tahdag_dir, exist_ok=True)
        out = os.path.join(self._tahdag_dir, f"tahdag_{os.environ.get('HOSTNAME', 'x')}_ep{self._tahdag_ep:04d}.pt")
        _t.save({"episodes": [self._tahdag_buf], "d_feat": 384}, out)
        print(f"[tahdag] wrote {out}  ({len(self._tahdag_buf)} frames)", flush=True)
        self._tahdag_ep += 1; self._tahdag_buf = []

    def _reid_select(self, cset, rgb, store_slot=None):
        """Temporal appearance-memory re-ID (reframe → offline milestone: crop-DINOv2 feature +
        K-view gallery breaks the 0.5 wall without resolution). Returns (N,) match scores to the
        stored target gallery, or None if empty.

        Fair-by-construction (no GT leak): SELECT with the PAST gallery, THEN store this view from
        the GT-identified target (future only). Feature = crop-DINOv2 (instance-discriminative,
        `reid_feature=dinov2`) or the coarse VLM pool (`vlmpool`). Matching = top-m mean over a
        K-view diversity gallery (viewpoint-robust). Ported from train.reid_resolution_probe."""
        from train.reid_resolution_probe import _dino_feats, bank_update, bank_score
        if self.reid_feature == "dinov2":
            W, H = self.image_cfg["width"], self.image_cfg["height"]
            feats = _dino_feats(self._dino, rgb, cset.cands, W, H, 224, 1.3, self.device).astype(np.float64)
        else:
            feats = cset.feats.astype(np.float64)
        fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        logits = bank_score(self._reid_bank, fn, self.reid_topm) if self._reid_bank else None
        self._reid_fn = fn                                                    # stash for deployable re-anchor re-seed
        # gallery update source: 'committed'=deployable (store the believed-target crop passed as store_slot;
        # GT never touches the template) | 'gt'=legacy oracle-seeded (store the GT target view, future only).
        ss = store_slot if self.gallery_seed == "committed" else \
             (cset.true_idx if (cset.true_idx is not None and cset.true_idx >= 0) else None)
        if ss is not None and 0 <= ss < len(fn):
            bank_update(self._reid_bank, fn[ss], self.reid_bank, 0.9)
        return logits

    def _asd_feats(self, cset, rgb):
        """ASD appearance view of the candidates: reuse the active memory's frozen crop-DINOv2 feats when it
        computed them this tick (tah / reid-dinov2), else compute them (memory-agnostic)."""
        if self.tid_head == "tah" and self._asd_fe is not None:
            return self._asd_fe
        if self.tid_head == "reid" and self.reid_feature == "dinov2" and self._reid_fn is not None \
           and len(self._reid_fn) == len(cset.cands):
            return self._reid_fn
        from train.reid_resolution_probe import _dino_feats
        W, H = self.image_cfg["width"], self.image_cfg["height"]
        fe = _dino_feats(self._dino, rgb, cset.cands, W, H, 224, 1.3, self.device).astype(np.float64)
        return fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-8)

    def _asd_gate_memory(self, slot, which=None):
        """Posterior-gated memory update: redo this tick's update with the filter's MAP crop (tah); reid's gallery
        already stores the COMMITTED identity, which is overridden by the caller; dam4sam (V1): re-init SAM2 on the
        filter's MAP candidate box (drops its RAM/DRM), with a pre-registered 3-tick cooldown."""
        which = which or self.tid_head
        if which == "dam4sam" and self._dam4sam_rgb is not None and self._cand_set_cur is not None \
           and self._dam4sam_cool == 0:
            from rollout.bridge import send_msg, recv_msg
            c = self._cand_set_cur.cands[slot]
            send_msg(self._dam4sam_sock, {"cmd": "init", "rgb": np.asarray(self._dam4sam_rgb, np.uint8),
                                          "box": (c.u - c.w / 2.0, c.v - c.h / 2.0, c.w, c.h)})
            recv_msg(self._dam4sam_sock)
            self._dam4sam_cool = 3
            return
        if which in ("tah", "fusion") and self._tah_last is not None:
            feat, pos = self._tah_last
            prev = getattr(self, "_tah_mem_prev", None)
            prev = prev if prev is not None else self._tah.reset(self.device)
            self._tah_mem = self._tah.update(prev, feat[slot], pos[slot])

    def _cold_vote(self, aidx, lang_logits, lang_slot):
        """V2: one tempered, rank-weighted language vote per tick; returns the slot of argmax_k G_k (visible)."""
        l = np.asarray(lang_logits, np.float64); srt = np.sort(l)
        m = float(srt[-1] - srt[-2]) if l.size >= 2 else 0.0
        h = self._cold_hist; rho = float(np.searchsorted(np.sort(h), m)) / len(h) if len(h) >= 2 else 0.5
        self._cold_hist = h + [m]
        for k in self._cold_G:
            self._cold_G[k] *= self.cold_lambda
        a = int(aidx[lang_slot]); self._cold_G[a] = self._cold_G.get(a, 0.0) + 0.5 + rho
        return max(range(len(aidx)), key=lambda s_: self._cold_G.get(int(aidx[s_]), 0.0))

    def _asd_reset_memory(self, slot):
        """The ONE hook ASD needs from a memory: reset it to the anchor's pick (drop the drifted state)."""
        if self.tid_head == "tah" and self._tah_last is not None:
            feat, pos = self._tah_last
            self._tah_mem = self._tah.update(self._tah.reset(self.device), feat[slot], pos[slot])
        elif self.tid_head == "reid" and self._reid_fn is not None:
            from train.reid_resolution_probe import bank_update
            self._reid_bank = []
            bank_update(self._reid_bank, self._reid_fn[slot], self.reid_bank, 0.9)
            self._reid_score = {}
            self._reid_mvu = self._reid_mvel = None

    def _reid_motion_score(self, cset):
        """borrow#1 (DeepSORT): SOFT motion-consistency score to FUSE into the reid appearance score
        (pre-EMA). Predicts the target's next image position (constant velocity) from past state and
        softly penalizes candidates far from it → a look-alike spatially inconsistent with the track
        can't win on appearance alone (fixes ours' pure-appearance weakness; DeepSORT had the highest
        external SR). Returns (N,) ~[-∞,0] (0=at prediction), zeros if no prior. State NOT updated here."""
        cands = cset.cands
        if self._reid_mvu is None:
            return np.zeros(len(cands), np.float64)
        pred = self._reid_mvu + (self._reid_mvel if self._reid_mvel is not None else 0.0)
        uv = np.array([[c.u, c.v] for c in cands], np.float64)
        return -np.linalg.norm(uv - pred[None], axis=1) / self.reid_motion_gate_px

    def _reid_motion_update(self, cset, slot):
        """Update motion state from the seeding slot's image pos (SELECT used PAST state; same
        GT/committed privilege as the gallery). None/loss → hold + keep predicting with velocity."""
        if slot is None or not (0 <= slot < len(cset.cands)):
            return
        c = cset.cands[slot]; nuv = np.array([c.u, c.v], np.float64)
        self._reid_mvel = (nuv - self._reid_mvu) if self._reid_mvu is not None else np.zeros(2)
        self._reid_mvu = nuv

    def _assoc_select(self, cset, rgb, init_slot=None):
        """EXTERNAL BASELINE: DeepSORT/OC-SORT-style tracking-by-detection association.
        Motion (constant-velocity image-space prediction + gate) + optional appearance (crop-DINOv2
        cosine to an EMA template). SELECT with the PAST track state, THEN update from the GT-visible
        target (same oracle-seed privilege as reid's gallery → fair). Contrast to reid: an explicit
        motion model + per-frame argmin vs appearance gallery + temporal-EMA. assoc_mode: 'motion'
        (OC-SORT-like, no appearance) | 'deepsort' (motion+appearance). Returns (N,) scores."""
        cands = cset.cands
        uv = np.array([[c.u, c.v] for c in cands], np.float64)          # candidate image positions
        n = len(cands)
        if self._assoc_uv is not None:                                  # motion: predict + gate
            pred = self._assoc_uv + (self._assoc_vel if self._assoc_vel is not None else 0.0)
            dist = np.linalg.norm(uv - pred[None], axis=1)
            score = -dist / self.assoc_gate_px                          # higher = closer
            gate = dist < self.assoc_gate_px
        else:
            score = np.zeros(n); gate = np.ones(n, bool)
        fn = None
        if self.assoc_mode == "deepsort":                              # + appearance cosine
            from train.reid_resolution_probe import _dino_feats
            W, H = self.image_cfg["width"], self.image_cfg["height"]
            feats = _dino_feats(self._dino, rgb, cands, W, H, 224, 1.3, self.device).astype(np.float64)
            fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            if self._assoc_feat is not None:
                score = score + self.assoc_lambda * (fn @ self._assoc_feat)
        if self._assoc_uv is not None and gate.any():                  # hard motion gate (fall back if empty)
            score = np.where(gate, score, -1e9)
        # UPDATE track state. gallery_seed='gt' (oracle, legacy): from the GT target. 'committed' (DEPLOYABLE, GT-free):
        # from the tracker's OWN association this tick (language pick at cold start, before any track state exists).
        # SELECT always used the PAST state.
        if self.gallery_seed == "committed":
            if self._assoc_uv is None:
                ti = init_slot
            else:
                ti = int(score.argmax()) if float(score.max()) > -1e8 else None   # no gated match → hold state
        else:
            ti = cset.true_idx
        if ti is not None and ti >= 0:
            self._assoc_vel = (uv[ti] - self._assoc_uv) if self._assoc_uv is not None else np.zeros(2)
            self._assoc_uv = uv[ti].copy()
            if fn is not None:
                t = fn[ti]
                self._assoc_feat = t if self._assoc_feat is None else \
                    (0.9 * self._assoc_feat + 0.1 * t) / (np.linalg.norm(0.9 * self._assoc_feat + 0.1 * t) + 1e-8)
        return score.astype(np.float32)

    def _dam4sam_select(self, cset, rgb, init_slot=None):
        """EXTERNAL BASELINE: DAM4SAM (CVPR'25 SOTA distractor-aware SAM2 memory) as the WHICH module,
        served over a socket bridge (separate torch2.1/SAM2 env). On the first target-visible frame,
        init the tracker with the GT target box (same oracle-seed privilege as reid/assoc); each S2
        tick, track → predicted target box → assign the candidate whose center is nearest (within a
        gate) → pred_slot. Returns (N,) scores (higher=nearer the DAM4SAM box); -1e9 gate for far."""
        from rollout.bridge import send_msg, recv_msg
        cands = cset.cands
        n = len(cands)
        # init box: gallery_seed='gt' (oracle, legacy) = GT target box; 'committed' (DEPLOYABLE) = the language pick's box
        ti = init_slot if self.gallery_seed == "committed" else cset.true_idx
        bbox = None
        if not self._dam4sam_inited:
            if ti is not None and ti >= 0:                              # init from the chosen target box
                c = cands[ti]
                box = (c.u - c.w / 2.0, c.v - c.h / 2.0, c.w, c.h)
                send_msg(self._dam4sam_sock, {"cmd": "init", "rgb": np.asarray(rgb, np.uint8), "box": box})
                bbox = recv_msg(self._dam4sam_sock).get("bbox")
                self._dam4sam_inited = True
                self._dam4sam_iou = 0.5                                 # init frame: no predicted IoU yet
            else:
                self._dam_log(cset, "no_init", None, None)
                return np.full(n, -1e9, np.float32)                     # target never seen yet → no pick
        else:
            send_msg(self._dam4sam_sock, {"cmd": "track", "rgb": np.asarray(rgb, np.uint8)})
            resp = recv_msg(self._dam4sam_sock); bbox = resp.get("bbox")
            iou = resp.get("iou")
            self._dam4sam_iou = float(iou) if iou is not None and iou == iou else 0.0   # SAM2 predicted IoU = confidence
        self._dam4sam_rgb = rgb                                         # for the PGIF reset hook (re-init on ŝ)
        self._dam4sam_cool = max(0, getattr(self, "_dam4sam_cool", 0) - 1)
        if bbox is None:
            self._dam_log(cset, "empty", None, None)
            return np.full(n, -1e9, np.float32)                         # lost (empty mask) → hold
        bx, by, bw, bh = bbox
        pc = np.array([bx + bw / 2.0, by + bh / 2.0])                   # DAM4SAM predicted center
        uv = np.array([[c.u, c.v] for c in cands], np.float64)
        dist = np.linalg.norm(uv - pc[None], axis=1)
        gate = float(self.assoc_gate_px)
        self._dam_log(cset, "pick" if float(dist.min()) < gate else "gate", dist, bbox)
        return np.where(dist < gate, -dist / gate, -1e9).astype(np.float32)

    def _dam_log(self, cset, reason, dist, bbox):
        """DAM4SAM abstention diagnosis record (offline scoring only: true_idx is NOT used by any decision)."""
        if self._dam_diag is None:
            return
        ti = cset.true_idx if cset.true_idx is not None else -1
        rec = {"ep": self._dam_ep, "k": self._dam_k, "reason": reason, "n": len(cset.cands), "true": int(ti),
               "inited": bool(self._dam4sam_inited), "iou": round(float(self._dam4sam_iou), 4),
               "ef_n": self._dam_ef_n, "ef_empty": self._dam_ef_empty}
        if dist is not None:
            j = int(dist.argmin())
            rec.update(nearest=j, d_min=round(float(dist[j]), 1), nearest_is_true=bool(j == ti),
                       d_true=round(float(dist[ti]), 1) if ti >= 0 else None,
                       box_wh=[round(float(bbox[2]), 1), round(float(bbox[3]), 1)])
        self._dam_diag.write(json.dumps(rec) + "\n"); self._dam_diag.flush()
        self._dam_k += 1

    def _update_commit(self, cset):
        """Commit the tid-selected candidate IDENTITY with K-frame hysteresis (Plan A).
        A challenger identity must win commit_k consecutive S2 ticks to take over; during
        a loss (no valid pick) the current commitment is held."""
        sel = None
        if cset is not None and 0 <= self._pred_slot < len(cset.cands):
            sel = int(cset.cands[self._pred_slot].idx)          # -1 = target, else distractor idx
        if sel is None:
            return                                              # loss -> hold commitment
        if self._committed_idx is None or sel == self._committed_idx:
            self._committed_idx = sel; self._chal = None; self._chal_n = 0
        else:
            self._chal_n = self._chal_n + 1 if sel == self._chal else 1
            self._chal = sel
            if self._chal_n >= self.commit_k:
                self._committed_idx = sel; self._chal = None; self._chal_n = 0

    def _committed_slot(self, cset):
        """Candidate slot of the currently committed identity this frame (None if not visible / uncommitted).
        Deployable gallery seeds its template from THIS crop — the identity we believe, never GT."""
        if self._committed_idx is None or cset is None:
            return None
        for s, c in enumerate(cset.cands):
            if int(c.idx) == int(self._committed_idx):
                return s
        return None

    def _committed_world(self, tpos, dstates):
        """World position of the committed identity (GT position; the IDENTITY is tid's)."""
        if self._committed_idx is None:
            return None
        if self._tracker is not None or self.occl_cands:
            # deployable v2: localise the committed identity ONLY through a box matched to it THIS tick (ideal
            # localisation of a visible box); unmatched / occluded → None (the estimator coasts on CV).
            cs = self._cand_set
            c = next((c for c in (cs.cands if cs is not None else []) if int(c.idx) == int(self._committed_idx)), None)
            if c is None:
                return None
            gi = int(getattr(c, "gt_idx", c.idx))
            if gi < 0:
                return np.asarray(tpos, np.float64)
            ds = np.asarray(dstates)
            return np.asarray(ds[gi][:3], np.float64) if gi < ds.shape[0] else None
        if self._committed_idx < 0:
            return np.asarray(tpos, np.float64)
        ds = np.asarray(dstates)
        return np.asarray(ds[self._committed_idx][:3], np.float64) if self._committed_idx < ds.shape[0] else None

    def _in_frame(self, world_xyz, campose):
        from acot_probe.projection import project_point
        W = self.image_cfg["width"]; H = self.image_cfg["height"]; fov = self.image_cfg["fov_deg"]
        pr = project_point(tuple(float(x) for x in world_xyz), tuple(float(x) for x in campose), W, H, fov)
        if pr is None:
            return False
        u, v, _ = pr
        return bool(0 <= u < W and 0 <= v < H)

    @torch.no_grad()
    def act(self, obs: Observation, gt_candidates) -> tuple[tuple, float, ActInfo]:
        """gt_candidates = (tpos(3,), dstates(D,6), campose(5,)). Returns
        ((dx,dy,dz,dyaw), search_mode, ActInfo)."""
        prop = torch.from_numpy(obs.proprio.astype(np.float32))[None].to(self.device)

        if getattr(self, "_dam4sam_every_frame", False) and self.tid_head == "dam4sam" and self._dam4sam_inited \
           and not (obs.step % self.s2_period == 0 or self._z_ex is None):
            # G11: intermediate frame → advance SAM2's memory on it; no identity decision here (read at S2 ticks).
            from rollout.bridge import send_msg, recv_msg
            send_msg(self._dam4sam_sock, {"cmd": "track", "rgb": np.asarray(obs.rgb, np.uint8)})
            resp = recv_msg(self._dam4sam_sock)
            self._dam_ef_n += 1; self._dam_ef_empty += int(resp.get("bbox") is None)

        if obs.step % self.s2_period == 0 or self._z_ex is None:
            # ---- S2 slow stream: VLM → EAR → IAR → target-id ----
            layers, vlm_ctx, grid_last = self.vlm.encode(obs.rgb, obs.language)   # (M,C) float32
            self._vlm_ctx = vlm_ctx[None]                                          # (1,M,C)
            self._ctx_layers = [x[None] for x in layers]
            self._ctx_mask = torch.ones(1, vlm_ctx.shape[0], dtype=torch.bool, device=self.device)
            if self.ablate == "ear":
                self._z_ex = torch.zeros(1, self.K, 3, device=self.device)                # w/o-EAR
            else:
                self._z_ex = flow_matching.sample(self.ear, self._vlm_ctx, self._ctx_mask,
                                                   k=self.K, steps=self.steps, proprio=prop)  # (1,K,3)
            self._z_im, _ = self.iar(self._ctx_layers, self._ctx_mask)                     # (1,n_im,d)
            if self.ablate == "iar":
                self._z_im = torch.zeros_like(self._z_im)                                  # w/o-IAR

            tpos, dstates, campose = gt_candidates[:3]
            # P2a-1: env-side road-graph prediction (WORLD (K,3)) rides in slot 3 when
            # --ex-source road; None otherwise (or for legacy 3-tuples).
            road_pred_world = gt_candidates[3] if len(gt_candidates) > 3 else None
            cset = build_candidates(grid_last, tpos, dstates, campose, self.image_cfg, self._rng)
            vis = gt_candidates[4] if len(gt_candidates) > 4 else None
            if cset is not None and self.occl_cands and vis is not None:
                # occlusion-aware detector: a vehicle the env's ray-cast marks occluded produces no box this tick
                t_vis, d_vis = vis
                keep = [i for i, c in enumerate(cset.cands)
                        if (t_vis if c.is_target else (c.idx < len(d_vis) and d_vis[c.idx]))]
                if not keep:
                    cset = None
                else:
                    cands = [cset.cands[i] for i in keep]
                    cset = type(cset)(feats=cset.feats[keep], cands=cands,
                                      true_idx=next((j for j, c in enumerate(cands) if c.is_target), -1),
                                      mask=cset.mask[keep])
            if cset is not None and self._tracker is not None:
                # GT-free identity keys: re-key every candidate with the tracker's id; keep the simulator actor index
                # in gt_idx (scorer / ideal localisation of the CURRENTLY matched visible box only).
                from train.reid_resolution_probe import _dino_feats
                Wt, Ht = self.image_cfg["width"], self.image_cfg["height"]
                ft = _dino_feats(self._dino, obs.rgb, cset.cands, Wt, Ht, 224, 1.3, self.device).astype(np.float64)
                ft = ft / (np.linalg.norm(ft, axis=1, keepdims=True) + 1e-8)
                tids = self._tracker.update([(c.u, c.v) for c in cset.cands], [c.w for c in cset.cands], ft)
                for c, t in zip(cset.cands, tids):
                    c.gt_idx = int(c.idx); c.idx = int(t)
                if self._trk_log is not None:
                    self._trk_log.write(json.dumps({"ep": self._trk_ep, "k": self._trk_k, "t": int(obs.step),
                                                    "tid": [int(c.idx) for c in cset.cands],
                                                    "gt": [int(c.gt_idx) for c in cset.cands],
                                                    # replay data for offline tracker tuning on dev (perception only)
                                                    "uv": [[round(float(c.u), 1), round(float(c.v), 1)] for c in cset.cands],
                                                    "sz": [round(float(c.w), 1) for c in cset.cands],
                                                    "f": np.round(ft, 3).astype(np.float16).tolist()}) + "\n")
                    self._trk_log.flush()
                self._trk_k += 1
            if cset is not None:
                cf = torch.from_numpy(cset.feats)[None].to(self.device)                    # (1,N,C)
                cmask = torch.from_numpy(cset.mask)[None].to(self.device)
                if self.tid_head == "attrbind":
                    if obs.target_color not in self._cvoc or obs.target_bp not in self._mvoc:
                        print(f"[attrbind-attr] <UNK> color='{obs.target_color}' make='{obs.target_bp}'", flush=True)
                    ci = torch.tensor([self._cvoc.get(obs.target_color, self._cvoc["<unk>"])], device=self.device)
                    mi = torch.tensor([self._mvoc.get(obs.target_bp, self._mvoc["<unk>"])], device=self.device)
                    logits = self.tid(cf, ci, mi, cmask)[0]                                # (N,)
                else:
                    logits = self.tid(cf, self._vlm_ctx, self._ctx_mask, cmask)[0]         # (N,)
                self._tid_logits = logits.float().cpu().numpy()
                self._pred_slot = int(logits.argmax().item())
                lang_slot = self._pred_slot   # language grounding pick — saved before any reid override
                lang_logits = self._tid_logits.copy()   # full language logits — for TEMPORAL language re-anchor
                if self._zoom is not None and self._open_k < self._open_window and len(cset.cands) > 0:
                    zs = self._zoom.scores(obs.rgb, cset.cands, str(obs.language), getattr(obs, "rgb_hi", None))
                    if self._open_agg != "mean":
                        zs = zs - zs.max()
                    for c, z in zip(cset.cands, zs):                       # accumulate over the opening window by id
                        self._open_acc[int(c.idx)] = self._open_acc.get(int(c.idx), 0.0) + float(z)
                        self._open_cnt[int(c.idx)] = self._open_cnt.get(int(c.idx), 0) + 1
                    _den = (lambda i_: self._open_cnt[i_]) if self._open_agg == "mean" else (lambda i_: 1)
                    if self._open_agg == "last":     # this tick's scores only — immune to tracker id swaps in the window
                        lang_slot = int(np.argmax(zs))
                    else:
                        lang_slot = max(range(len(cset.cands)),
                                        key=lambda s_: self._open_acc[int(cset.cands[s_].idx)] / _den(int(cset.cands[s_].idx)))
                    self._pred_slot = lang_slot; self._open_k += 1
                    if self._open_window > 1 and (self._open_k < self._open_window or self._open_final_reset) \
                       and self._tah is not None:
                        self._tah_mem = self._tah.reset(self.device)       # keep the memory seeded by the FINAL pick
                    if self._open_reset_asd and self._asd is not None and getattr(self._asd, "mode", "") == "filter":
                        self._asd.belief = {}; self._asd.conf_hist = []   # PGIF restarts from this opening pick
                    if self._open_k >= self._open_window:                  # last opening tick
                        self._open_force = self._open_force_final
                        if self._open_seed_asd > 0 and self._asd is not None and getattr(self._asd, "mode", "") == "filter":
                            self._asd.belief = {int(cset.cands[lang_slot].idx): self._open_seed_asd}; self._asd.conf_hist = []
                if os.environ.get("OPENING_DUMP") and obs.step in (0, 6, 12):
                    # opening benchmark snapshot (offline scorer development): frame + candidate boxes + language +
                    # current grounding logits; is_target / gt ids are saved ONLY for offline scoring
                    _d = os.environ["OPENING_DUMP"]; os.makedirs(_d, exist_ok=True)
                    np.savez_compressed(
                        os.path.join(_d, f"{os.environ.get('HOSTNAME', 'x')}_{self._dump_ep:03d}_t{int(obs.step):03d}.npz"),
                        rgb=np.asarray(obs.rgb, np.uint8), language=str(obs.language),
                        rgb_hi=(np.zeros((0,), np.uint8) if getattr(obs, "rgb_hi", None) is None
                                else np.asarray(obs.rgb_hi, np.uint8)),
                        target_color=str(getattr(obs, "target_color", "")), target_bp=str(getattr(obs, "target_bp", "")),
                        u=np.array([c.u for c in cset.cands], np.float32), v=np.array([c.v for c in cset.cands], np.float32),
                        w=np.array([c.w for c in cset.cands], np.float32), h=np.array([c.h for c in cset.cands], np.float32),
                        depth=np.array([c.depth for c in cset.cands], np.float32),
                        is_target=np.array([bool(c.is_target) for c in cset.cands]),
                        gt_idx=np.array([int(getattr(c, "gt_idx", c.idx)) for c in cset.cands], np.int32),
                        lang_logits=np.asarray(lang_logits, np.float32), step=int(obs.step), ep=int(self._dump_ep))
                rl = None                     # reid match scores (None = no gallery yet / non-reid head)
                # TEMPORAL language re-anchor: single-frame language picks wrong at re-appearance (H0 ceiling) →
                # accumulate per-actor EMA of the language logit so the re-anchor commits the temporally-integrated
                # language vote, not one noisy frame (mirrors reid_tavg, which fixed single-frame reid).
                if self.reanchor == "lang" and self.reanchor_tavg > 0:
                    a = self.reanchor_tavg
                    for s, c in enumerate(cset.cands):
                        prev = self._lang_score.get(c.idx, float(lang_logits[s]))
                        self._lang_score[c.idx] = a * prev + (1 - a) * float(lang_logits[s])
                # reid override (reframe test): temporal appearance-memory re-ID instead of
                # per-frame language grounding — select the candidate matching the STORED target
                # appearance (built while tracking), not the language description. Tests whether
                # the WHICH wall is the single-frame paradigm. Falls back to tid before a template
                # exists (cold start). See memory acot-uav-reid-reframe.
                if self.tid_head == "reid":
                    # deployable gallery: store the crop of the identity we currently BELIEVE is the
                    # target (prior-frame commitment; language pick at cold-start), never GT (GT-free WHICH).
                    store_slot = None
                    if self.gallery_seed == "committed":
                        store_slot = lang_slot if self._committed_idx is None \
                                     else self._committed_slot(cset)
                    rl = self._reid_select(cset, obs.rgb, store_slot)
                    if rl is not None:
                        if self.reid_motion > 0:                      # borrow#1: fuse motion consensus into rl (pre-EMA)
                            ms = self._reid_motion_score(cset)
                            self._reid_mscore = ms                    # borrow#2: stash for the consensus commit-fly gate
                            rl = rl + self.reid_motion * ms
                            mslot = store_slot if self.gallery_seed == "committed" else \
                                    (cset.true_idx if (cset.true_idx is not None and cset.true_idx >= 0) else None)
                            self._reid_motion_update(cset, mslot)
                        self._tid_logits = rl
                        self._pred_slot = int(rl.argmax())
                        # TEMPORAL re-ID: the gallery template is temporal but the QUERY is single-frame
                        # (argmax) → an off-center/far frame lets a look-alike fluke-win → mis-commit →
                        # cascade. Accumulate per-ACTOR (stable cand.idx) EMA of the match score and pick
                        # the TEMPORALLY-INTEGRATED best visible candidate → robust to single bad frames,
                        # persists evidence across brief losses. reid_tavg=decay (0=off=single-frame).
                        if self.reid_tavg > 0:
                            a = self.reid_tavg
                            for s, c in enumerate(cset.cands):
                                prev = self._reid_score.get(c.idx, float(rl[s]))
                                self._reid_score[c.idx] = a * prev + (1 - a) * float(rl[s])
                            best_s, best_v = self._pred_slot, -1e9
                            for s, c in enumerate(cset.cands):
                                v = self._reid_score.get(c.idx, -1e9)
                                if v > best_v:
                                    best_v, best_s = v, s
                            self._pred_slot = int(best_s)
                            self._tid_logits = np.array([self._reid_score.get(c.idx, -1e9)
                                                         for c in cset.cands], np.float32)
                # external baseline: DeepSORT/OC-SORT-style association (motion + appearance),
                # a common tracking-by-detection identity mechanism — the natural comparison for
                # our temporal appearance-memory re-ID. Same z_ex/control path; only WHICH differs.
                if self.tid_head == "assoc":
                    al = self._assoc_select(cset, obs.rgb, init_slot=lang_slot)
                    self._tid_logits = al
                    self._pred_slot = int(al.argmax())
                if self.tid_head == "fusion":                          # V3: DAM4SAM runs alongside TAHRelM
                    dl = self._dam4sam_select(cset, obs.rgb, init_slot=lang_slot)
                    self._fus_dam = int(dl.argmax()) if float(dl.max()) > -1e8 else None
                if self.tid_head == "dam4sam":
                    dl = self._dam4sam_select(cset, obs.rgb, init_slot=lang_slot)
                    self._tid_logits = dl
                    self._pred_slot = int(dl.argmax()) if float(dl.max()) > -1e8 else -1
                # NULL BASELINE (Sticky-ID): language cold-start pick, then never switch. Identity is carried
                # entirely by the harness's persistent cand.idx; abstain (-1) while that actor is not in frame.
                if self.tid_head == "sticky":
                    if self._sticky_idx is None:
                        self._sticky_idx = int(cset.cands[lang_slot].idx)
                    self._tid_logits = np.full(len(cset.cands), -1e9, np.float32)
                    self._pred_slot = -1
                    for s, c in enumerate(cset.cands):
                        if int(c.idx) == self._sticky_idx:
                            self._tid_logits[s] = 1.0
                            self._pred_slot = s
                if self.tid_head in ("tah", "fusion"):                # path-B: learned TAHRel association
                    from train.reid_resolution_probe import _dino_feats
                    Wi, Hi = self.image_cfg["width"], self.image_cfg["height"]
                    fe = _dino_feats(self._tah_dino if hasattr(self, "_tah_dino") else self._dino,
                                     obs.rgb, cset.cands, Wi, Hi, 224, 1.3, self.device).astype(np.float64)
                    fe = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-8)
                    self._asd_fe = fe
                    feat = torch.tensor(fe, dtype=torch.float32, device=self.device)
                    pos = torch.tensor(np.array([[c.u / Wi, c.v / Hi, c.depth / 100.0] for c in cset.cands], np.float32),
                                       device=self.device)
                    self._tah_last = (feat, pos)
                    with torch.no_grad():
                        logits, conf, _ = self._tah.step(feat, pos, self._tah_mem)
                    self._tid_logits = logits.float().cpu().numpy()
                    self._pred_slot = int(logits.argmax())
                    self._pred_conf = float(conf)
                    if getattr(self, "_open_force", False):          # v4: last opening tick → commit the opening pick
                        self._pred_slot = lang_slot; self._open_force = False
                    if self.gallery_seed == "committed":
                        # DEPLOYABLE (GT-free): the tah memory is empty at cold-start → its logits are
                        # meaningless (mlp of zeros → const) → LANGUAGE grounds the initial lock. Then a
                        # language RE-ANCHOR on a sustained low-margin (ambiguous / drifting) streak: reset
                        # the drifted memory + re-pick via language (mirrors the reid deployable path).
                        # gt-seeded skips both (oracle memory every frame). See eval_out/paper_ext_tahrelm_committed.
                        cold = (self._tah_mem is None) or (self._tah_mem.get("app") is None)
                        self._cold_active = cold and self.cold_window > 0 and self._cold_n < self.cold_window
                        if self.event_override and getattr(obs, "retarget_event", False) and cset.true_idx is not None \
                           and cset.true_idx >= 0:
                            # S1 GT-driven event override (1003_dynamic-retarget-design.md §2.3): the scene's
                            # target just changed (scripted, GT-known this tick) — hard-reset memory + PGIF
                            # belief to it NOW, instead of waiting for drift-detection machinery to notice.
                            # Takes priority over cold-start / reanchor handling below on this one tick.
                            self._tah_mem = self._tah.reset(self.device)
                            self._pred_slot = cset.true_idx
                            self._cold_active = False
                            if self._asd is not None and getattr(self._asd, "mode", "") == "filter":
                                self._asd.override(int(cset.cands[cset.true_idx].idx), t=int(obs.step))
                            self._override_events += 1
                        elif self._cold_active:
                            # V2 DEFERRED COMMITMENT: integrate tempered, rank-weighted LANGUAGE votes per tracklet for
                            # cold_window ticks; fly toward the provisional argmax but do NOT write identity memory.
                            k_best = self._cold_vote([int(c.idx) for c in cset.cands], lang_logits, lang_slot)
                            self._pred_slot = lang_slot               # window flight = per-tick language pick (as V0)
                            self._cold_n += 1
                            if self._cold_n >= self.cold_window:      # commit: seed memory + PGIF evidence
                                k_c = lang_slot if self.cold_single else k_best   # control: single-frame pick at tick K
                                self._pred_slot = k_c
                                self._committed_idx = int(cset.cands[k_c].idx)
                                self._tah_mem = self._tah.update(self._tah.reset(self.device), feat[k_c], pos[k_c])
                                if self._asd is not None and getattr(self._asd, "mode", "") == "filter" and not self.cold_single:
                                    self._asd.belief = {k: min(v, 3.0) for k, v in self._cold_G.items()}   # capped prior
                                self._cold_active = False
                                self._cold_done_now = True
                                if self.tid_head == "fusion" and self._dam4sam_inited:
                                    self._dam4sam_cool = 0; self._cand_set_cur = cset; self._dam4sam_rgb = obs.rgb
                                    self._asd_gate_memory(k_c, which="dam4sam")
                        elif cold:
                            self._pred_slot = lang_slot                       # language cold-start
                            if self.reanchor == "anchor" and self._tah_anchor is None:
                                self._tah_anchor = feat[lang_slot].clone()    # freeze initial target appearance
                        elif self.reanchor == "anchor":
                            # GT-FREE confident-drift detector: the running memory can drift to a look-alike;
                            # the FROZEN initial anchor can't. Detect drift when the anchor's best-match
                            # candidate persistently disagrees with the running-memory pick → reset memory +
                            # re-anchor to the anchor's match. Approximates the GT-drift upper bound w/o labels.
                            if self._tah_anchor is not None:
                                apick = int((feat @ self._tah_anchor).argmax())
                                self._lowconf_run = self._lowconf_run + 1 if apick != self._pred_slot else 0
                                if self._lowconf_run >= self.reanchor_patience:
                                    self._tah_mem = self._tah.reset(self.device)
                                    self._pred_slot = apick
                                    self._lowconf_run = 0; self._reanchor_events += 1
                        elif self.reanchor == "lang":
                            self._lowconf_run = self._lowconf_run + 1 if float(conf) < 0.60 else 0
                            if self._lowconf_run >= self.reanchor_patience:
                                self._tah_mem = self._tah.reset(self.device)  # drop the drifted memory
                                self._pred_slot = lang_slot                   # re-anchor via language
                                self._lowconf_run = 0; self._reanchor_events += 1
                        elif self.reanchor == "oracle":
                            # ★confident-drift UPPER BOUND: perfect (GT) drift detection. conf-gate reanchor
                            # can't catch a CONFIDENTLY-wrong lock (high margin) → detect drift by GT instead
                            # (pred != true target for a sustained streak) → reset the drifted memory + re-anchor
                            # to the true target. Measures whether drift-detection is the deployable lever at all.
                            gt = cset.true_idx
                            if gt is not None and gt >= 0:
                                self._lowconf_run = self._lowconf_run + 1 if self._pred_slot != gt else 0
                                if self._lowconf_run >= self.reanchor_patience:
                                    self._tah_mem = self._tah.reset(self.device)
                                    self._pred_slot = gt                      # re-anchor to the true target
                                    self._lowconf_run = 0; self._reanchor_events += 1
                    # memory update source: gt=oracle GT every frame | committed=believed target (lang at cold-start)
                    us = (cset.true_idx if self.gallery_seed != "committed" else self._pred_slot)
                    if self._cold_active or getattr(self, "_cold_done_now", False):
                        us = None                                     # V2: no memory write inside / at the end of the window
                        self._cold_done_now = False
                    self._tah_mem_prev = self._tah_mem                    # ASD-IF: snapshot for posterior-gated update
                    if us is not None and 0 <= us < feat.shape[0]:
                        with torch.no_grad():
                            self._tah_mem = self._tah.update(self._tah_mem, feat[us], pos[us])
                    if self._tahdag_dir:                             # path-B DAgger: log deployment-dist frame
                        self._tahdag_buf.append({"feat": fe.astype(np.float16),
                                                 "pos": pos.cpu().numpy(),
                                                 "aidx": np.array([c.idx for c in cset.cands], np.int16),
                                                 "tidx": int(cset.true_idx)})
                # oracle-identity diagnostic: always pick the TRUE target (isolates the control/WHERE
                # ceiling — if identity were perfect, can deployable control keep the target in frame?).
                if self.tid_head == "oracle":
                    if cset.true_idx is not None and cset.true_idx >= 0:
                        self._tid_logits = np.zeros(len(cset.cands), np.float32)
                        self._tid_logits[cset.true_idx] = 1.0
                        self._pred_slot = int(cset.true_idx)
                    else:
                        self._pred_slot = -1                              # target off-screen this frame
                # WARM-START diagnostic: first N seconds use ORACLE identity (establish good tracking/
                # framing before handing off to reid) — the gallery still builds from the GT target
                # during warmup (reid_select ran above). Tests "does reid sustain if the cascade
                # isn't initiated early" → is the deployable gap cascade-INITIATION or reid-intrinsic.
                if self.warmup_oracle_s > 0 and obs.step < self.warmup_oracle_s * self.fps:
                    if cset.true_idx is not None and cset.true_idx >= 0:
                        self._tid_logits = np.zeros(len(cset.cands), np.float32)
                        self._tid_logits[cset.true_idx] = 1.0
                        self._pred_slot = int(cset.true_idx)
                # pick confidence = top1−top2 margin (used to gate commit-fly in track mode).
                tl = self._tid_logits
                self._pred_conf = float(np.sort(tl)[-1] - np.sort(tl)[-2]) if tl.size >= 2 else 1.0
                # ★ DEPLOYABLE LANGUAGE RE-ANCHOR (paper2 甲): appearance re-ID drifts onto a look-alike and
                # cannot recover on its own (its own drifted gallery keeps confirming the wrong car). When the
                # pick collapses for `reanchor_patience` consecutive ticks (or there is no gallery match), fall
                # back to the drift-free LANGUAGE description for ONE shot: re-pick, reset+re-seed the gallery
                # from that crop, force-commit. Language earns its keep here (not per-frame). 'oracle'=GT bound.
                if self.reanchor != "off" and self.tid_head == "reid" and len(cset.cands) > 0:
                    low = self._pred_conf < self.conf_tau
                    self._lowconf_run = (self._lowconf_run + 1) if low else 0
                    fire = (rl is None) or (self._lowconf_run >= self.reanchor_patience)
                    if fire and self._reid_fn is not None:
                        if self.reanchor == "lang":
                            if self.reanchor_tavg > 0:          # temporally-integrated language pick (visible cands)
                                ra, bv = lang_slot, -1e9
                                for s, c in enumerate(cset.cands):
                                    v = self._lang_score.get(c.idx, -1e9)
                                    if v > bv:
                                        bv, ra = v, int(s)
                            else:
                                ra = lang_slot                  # single-frame language pick
                        else:
                            ra = int(cset.true_idx) if (cset.true_idx is not None and cset.true_idx >= 0) else -1
                        if ra is not None and 0 <= ra < len(cset.cands):
                            from train.reid_resolution_probe import bank_update
                            self._pred_slot = int(ra)
                            self._reid_bank = []                                 # drop the drifted gallery
                            bank_update(self._reid_bank, self._reid_fn[ra], self.reid_bank, 0.9)  # re-seed from re-anchor pick
                            self._reid_score = {}                                # clear stale temporal EMA
                            self._committed_idx = int(cset.cands[ra].idx)        # force commit to re-anchored id
                            self._pred_conf = 1.0                                # high-confidence assertion → control commits
                            self._lowconf_run = 0
                            self._reanchor_events += 1
                # ★ ASD: memory-agnostic confident-drift correction (wraps whichever memory picked _pred_slot).
                if self._asd is not None and len(cset.cands) > 0 and 0 <= self._pred_slot < len(cset.cands) \
                   and not self._cold_active:
                    self._cand_set_cur = cset
                    if self.tid_head == "fusion":
                        srt = np.sort(np.asarray(self._tid_logits)); tconf = float(srt[-1] - srt[-2]) if srt.size >= 2 else 0.0
                        ls = np.sort(np.asarray(lang_logits)); lconf = float(ls[-1] - ls[-2]) if ls.size >= 2 else 0.0
                        tah_pick = self._pred_slot
                        fo = self._asd.step_multi([int(c.idx) for c in cset.cands],
                                                  [(tah_pick, tconf, "tah"), (self._fus_dam, self._dam4sam_iou, "dam"),
                                                   (lang_slot, lconf, "lang")], true_slot=cset.true_idx, t=int(obs.step))
                        sh = fo["slot"]
                        if sh is not None:
                            if self._asd.gate_tah and fo["gate"].get("tah") and self.gallery_seed == "committed":
                                self._asd_gate_memory(int(sh), which="tah")
                            if self._asd.gate_dam and (fo["gate"].get("dam") or self._fus_dam is None) \
                               and self.gallery_seed == "committed" and self._dam4sam_inited:
                                self._asd_gate_memory(int(sh), which="dam4sam")
                            self._pred_slot = int(sh)
                            self._committed_idx = int(cset.cands[self._pred_slot].idx)
                            self._pred_conf = 1.0
                            self._reanchor_events += int(any(fo["gate"].values()))
                        o = {"fire": False}
                    else:
                        fe_ = self._asd_feats(cset, obs.rgb) if self._asd.mode != "filter" else None
                        mconf = self._dam4sam_iou if self.tid_head == "dam4sam" else self._pred_conf
                        _orc_abs = os.environ.get("ORACLE_ABSENT", "0") in ("1", "2") and (cset.true_idx is None or cset.true_idx < 0)
                        if _orc_abs:                       # snapshot PGIF state (restored below)
                            _b0, _c0 = dict(self._asd.belief), list(self._asd.conf_hist)
                        o = self._asd.step(fe_, [int(c.idx) for c in cset.cands], self._pred_slot,
                                           mem_conf=mconf, lang=lang_logits,
                                           true_slot=cset.true_idx, t=int(obs.step))
                        if _orc_abs:
                            # DIAGNOSTIC UPPER BOUND ONLY (uses GT): a perfect "target is not among the candidates"
                            # detector → abstain like PGIF's absence hypothesis (hold the commitment, undo this tick's
                            # memory write) and leave the PGIF evidence untouched by this tick's distractor pick.
                            self._asd.belief, self._asd.conf_hist = _b0, _c0
                            if os.environ.get("ORACLE_ABSENT") == "2":
                                # identity hygiene only: commitment / control as usual (follows the pick), but nothing
                                # is learned from it — the TAH memory write and any PGIF memory gating are undone
                                if self.tid_head in ("tah", "fusion") and getattr(self, "_tah_mem_prev", None) is not None:
                                    self._tah_mem = self._tah_mem_prev
                                if o.get("kind") == "gate":
                                    o["gate_memory"] = False
                            else:
                                o = {"fire": False, "slot": None, "kind": "absent"}
                        if o.get("kind") == "absent":
                            # absence hypothesis: PGIF's MAP tracklet is not visible this tick → make NO pick (the
                            # commitment is held; WHERE coasts on its CV estimate) and undo this tick's memory write
                            # (the memory's own pick is, by the posterior, not the target).
                            self._pred_slot = -1; self._pred_conf = 0.0
                            if self.tid_head in ("tah", "fusion") and getattr(self, "_tah_mem_prev", None) is not None \
                               and self.gallery_seed == "committed":
                                self._tah_mem = self._tah_mem_prev
                    if o["fire"] and o.get("kind") == "gate":
                        # ASD-IF posterior gating: the memory learns from (and control commits to) the filter's MAP
                        # tracklet, not its own raw pick → breaks the self-confirming update loop.
                        # gt-seeded (oracle memory) runs keep the oracle update by definition → PGIF gates the
                        # control commitment only; deployable (committed) runs also gate the memory update.
                        if self.v1_selfreset and self.tid_head == "dam4sam":
                            # CONTROL: re-init SAM2 at the same moments but on its OWN pick (no identity change)
                            self._asd_gate_memory(int(self._pred_slot)); self._reanchor_events += 1
                            o = {"fire": False}
                    if o["fire"] and o.get("kind") == "gate":
                        if o.get("gate_memory", True) and self.gallery_seed == "committed":
                            self._asd_gate_memory(int(o["slot"]))
                        self._pred_slot = int(o["slot"])
                        self._committed_idx = int(cset.cands[self._pred_slot].idx)
                        self._pred_conf = 1.0
                        self._reanchor_events += 1
                    elif o["fire"]:
                        self._asd_reset_memory(int(o["slot"]))
                        self._pred_slot = int(o["slot"])
                        self._committed_idx = int(cset.cands[self._pred_slot].idx)   # force commit (as reanchor)
                        self._pred_conf = 1.0
                        self._reanchor_events += 1
            else:
                self._tid_logits, self._pred_slot = np.zeros(0, np.float32), -1
                self._pred_conf = 0.0
            self._cand_set = cset

            # DAgger: log this closed-loop frame's candidate feats + GT identity (oracle).
            if self._dagger_dir is not None and cset is not None and cset.true_idx >= 0:
                self._dagger_buf.append((cset.feats.astype(np.float16), int(cset.true_idx),
                                         obs.target_color, obs.target_bp))

            # ---- seed z_ex from a target-state prior (memory+CV) ----
            if self.ablate != "ear" and (self._ex_mode is not None or self._track):
                if self._track:
                    # Plan A: commit the tid-selected candidate (K-frame hysteresis), then
                    # track ITS world position. Grounding drives control — a wrong tid pick
                    # tracks the wrong car (lock-error), a right pick re-acquires the target.
                    # Confidence gate (①): on a LOW-margin (ambiguous) pick, DON'T switch the
                    # commitment and DON'T re-anchor control to it — keep dead-reckoning the last
                    # CONFIDENT trajectory (CV) instead of flying toward an uncertain look-alike.
                    confident = self._pred_conf >= self.conf_tau
                    # borrow#2: CONSENSUS gate — commit-fly only if the pick is ALSO motion-consistent
                    # (appearance AND motion agree). ms[slot]=-dist/gate; require dist<conf_consensus·gate.
                    # On disagreement (a look-alike wins appearance but is spatially off-track) → hold+search.
                    if confident and self.conf_consensus > 0 and self._reid_mscore is not None \
                       and 0 <= self._pred_slot < len(self._reid_mscore):
                        confident = float(self._reid_mscore[self._pred_slot]) > -self.conf_consensus
                    if confident:
                        self._update_commit(cset)
                    cw = self._committed_world(tpos, dstates)
                    self._frame_cw = cw if (confident and cw is not None) else None   # ②: keep this centered
                    if cw is not None:
                        if confident and self._in_frame(cw, campose):
                            self.estimator.update(cw, obs.step)
                        if self.estimator.seen:
                            wp = self.estimator.predict_wp(campose, obs.step, mode="cv")
                            if wp is not None:
                                self._z_ex = torch.from_numpy(wp)[None].to(self.device)
                else:
                    # (baseline) prior seeded from the GT target, gated by visibility; tid
                    # stays scorer-only (does NOT affect control).
                    visible = cset is not None and cset.true_idx >= 0
                    if visible:
                        self.estimator.update(np.asarray(tpos, np.float64), obs.step)
                    apply = (not self._ex_gated) or (not visible)
                    if apply and self._ex_mode == "road":
                        # road-graph WM: env already traversed the lane graph (from its own
                        # last-seen anchor) → just project the WORLD (K,3) into cam frame.
                        if road_pred_world is not None:
                            wp_prior = self.estimator.project_world(
                                campose, np.asarray(road_pred_world, np.float64))
                            self._z_ex = torch.from_numpy(wp_prior)[None].to(self.device)
                    elif apply and self.estimator.seen:
                        wp_prior = self.estimator.predict_wp(campose, obs.step, mode=self._ex_mode)
                        if wp_prior is not None:
                            self._z_ex = torch.from_numpy(wp_prior)[None].to(self.device)

        # ---- S1 fast stream: DiT action chunk on fresh proprio ----
        act = action_sample(self.dit, self._z_ex, self._z_im, self._vlm_ctx,
                            prop, self._ctx_mask, self.H, self.A, self.steps)              # (1,H,5)
        a0 = act[0, 0].float().cpu().numpy()
        dx, dy, dz, dyaw = (a0[:4] * self.a_scale).tolist()
        search_mode = float(a0[4] > 0.5)
        # ②: yaw-centering correction on the committed target (breaks the framing→reid cascade —
        # diagnostic: deployable control only centers 8% of frames → off-center mis 0.39). Nudge
        # the camera toward the committed target's projected horizontal offset; gated by confidence.
        if self._track and self.frame_gain and self._frame_cw is not None:
            from acot_probe.projection import project_point
            W = self.image_cfg["width"]; H = self.image_cfg["height"]; fov = self.image_cfg["fov_deg"]
            pr = project_point(tuple(float(x) for x in self._frame_cw),
                               tuple(float(x) for x in gt_candidates[2]), W, H, fov)
            if pr is not None and pr[2] > 0:
                dyaw = dyaw + self.frame_gain * (pr[0] / W - 0.5)   # +gain turns toward a right-of-center target
        info = ActInfo(tid_logits=self._tid_logits, pred_slot=self._pred_slot,
                       cand_set=self._cand_set, z_ex=self._z_ex[0].float().cpu().numpy())
        return (dx, dy, dz, dyaw), search_mode, info
