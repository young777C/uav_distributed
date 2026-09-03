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
                 conf_tau: float = 0.0):
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
        wp = cfg["waypoint"]
        self.estimator = TargetStateEstimator(wp["offsets_s"], wp["scale"], cfg.get("fps", 10))

        # DAgger collection: if DAGGER_DIR is set, log per-frame (cand_feats, true_idx,
        # target color/make) from the closed-loop distribution → retrain the tid on the
        # frames the policy actually visits (oracle label = GT true_idx, free).
        self._dagger_dir = os.environ.get("DAGGER_DIR")
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
        if tid_head == "reid" and reid_feature == "dinov2":
            from train.reid_resolution_probe import _load_dino
            self._dino = _load_dino("vit_small_patch14_dinov2.lvd142m", device)
            print(f"[policy] reid=dinov2 crop features + K{reid_bank}/top{reid_topm} gallery", flush=True)
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
        self._rng = np.random.default_rng(self._shuffle_seed)
        self._z_ex = self._z_im = self._vlm_ctx = self._ctx_mask = None
        self._tid_logits = np.zeros(0, np.float32)
        self._pred_slot = -1
        self._cand_set = None
        self._committed_idx = None      # Plan A: committed target identity (-1=target, d=distractor)
        self._chal = None; self._chal_n = 0
        self._reid_bank = []            # reid: K-view diversity gallery of the tracked target
        self._pred_conf = 0.0           # ①: current pick top1−top2 margin
        self.estimator.reset()
        self._dagger_flush()            # DAgger: flush the finished episode's frames, start fresh
        self._dagger_buf = []

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

    def _reid_select(self, cset, rgb):
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
        if cset.true_idx is not None and cset.true_idx >= 0:                   # store this view (future only)
            bank_update(self._reid_bank, fn[cset.true_idx], self.reid_bank, 0.9)
        return logits

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

    def _committed_world(self, tpos, dstates):
        """World position of the committed identity (GT position; the IDENTITY is tid's)."""
        if self._committed_idx is None:
            return None
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
                # reid override (reframe test): temporal appearance-memory re-ID instead of
                # per-frame language grounding — select the candidate matching the STORED target
                # appearance (built while tracking), not the language description. Tests whether
                # the WHICH wall is the single-frame paradigm. Falls back to tid before a template
                # exists (cold start). See memory acot-uav-reid-reframe.
                if self.tid_head == "reid":
                    rl = self._reid_select(cset, obs.rgb)
                    if rl is not None:
                        self._tid_logits = rl
                        self._pred_slot = int(rl.argmax())
                # pick confidence = top1−top2 margin (used to gate commit-fly in track mode).
                tl = self._tid_logits
                self._pred_conf = float(np.sort(tl)[-1] - np.sort(tl)[-2]) if tl.size >= 2 else 1.0
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
                    if confident:
                        self._update_commit(cset)
                    cw = self._committed_world(tpos, dstates)
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
        info = ActInfo(tid_logits=self._tid_logits, pred_slot=self._pred_slot,
                       cand_set=self._cand_set, z_ex=self._z_ex[0].float().cpu().numpy())
        return (dx, dy, dz, dyaw), search_mode, info
