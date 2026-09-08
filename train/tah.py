"""Temporal Association Head (TAH) — path-B: ONE learned module replacing the heuristic WHICH stack
(K-view gallery + top-m + temporal-EMA + motion-consensus fusion + consensus-gate). Given per-frame
candidates {crop-DINOv2 appearance ⊕ image position} and a recurrent target MEMORY, learn to associate
the tracked target across time (appearance + position + memory) → identity logits + a confidence.

Subsumes: gallery/EMA = the GRU memory (learned recurrence); motion-consensus = position is an input
feature the model learns to weight vs appearance; consensus-gate = the confidence output. No hand-tuned
knobs (RMOT / conf-tau / gate-px). Trained SUPERVISED on GT identity (the WHICH signal is learnable —
DINOv2 separates look-alikes; unlike the failed BC/RL control)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TAH(nn.Module):
    def __init__(self, d_feat=384, d_pos=3, d=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_feat + d_pos, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Linear(d, d)                       # query from memory
        self.k = nn.Linear(d, d)                       # key from candidate
        self.gru = nn.GRUCell(d, d)                    # learned memory update (replaces gallery+EMA)
        self.m0 = nn.Parameter(torch.zeros(d))         # learned initial memory
        self.conf = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))  # learned confidence
        self.d = d

    def reset(self, device):
        return self.m0.to(device).clone()

    def step(self, feat, pos, m):
        """One frame. feat:(n,d_feat) pos:(n,d_pos) m:(d,). Returns logits:(n,), conf:scalar, emb:(n,d)."""
        e = self.enc(torch.cat([feat, pos], dim=-1))                    # (n,d)
        logits = (self.k(e) @ self.q(m)) / (self.d ** 0.5)             # (n,) attention target↔candidates
        conf = torch.sigmoid(self.conf(m)).squeeze(-1)                 # learned confidence for control gating
        return logits, conf, e

    def update(self, m, target_emb):
        """Recurrent memory update from the (GT-teacher-forced or committed) target embedding."""
        return self.gru(target_emb.unsqueeze(0), m.unsqueeze(0)).squeeze(0)


class TAHRel(nn.Module):
    """TAH v2 (Tier-1 generalization fix): consume RELATIVE / vehicle-INVARIANT features instead of
    absolute candidate embeddings — the root-cause fix for the 64-episode open-set overfitting (CE on
    absolute embeddings memorizes training vehicles). The ONLY learned parts are (a) a scalar appearance-
    memory update gate and (b) a TINY MLP that WEIGHTS relative signals — a vehicle-agnostic gating
    function (learn the DeepSORT gating, not the encoder). Appearance comparison is offloaded to the
    frozen DINOv2 cosine (generalizes to any vehicle). Memory is a dict{app(384),pos(2),vel(2)}.
    ~5K params → cannot overfit 64 episodes."""
    def __init__(self, d_feat=384):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(-1.0))          # learned appearance-memory EMA gate (sigmoid)
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def reset(self, device):
        return {"app": None, "pos": None, "vel": None}

    def step(self, feat, pos, mem):
        """feat:(n,384) L2-normed; pos:(n,3)=[u/W,v/H,depth]; mem dict. Returns logits:(n,), conf, None."""
        n = feat.shape[0]; dev = feat.device
        if mem["app"] is None:                                # cold start: no target memory yet
            cos = torch.zeros(n, device=dev); relp = torch.zeros(n, 2, device=dev); rank = torch.zeros(n, device=dev)
        else:
            cos = feat @ mem["app"]                           # relative appearance (vehicle-agnostic cosine)
            pred = mem["pos"] + (mem["vel"] if mem["vel"] is not None else torch.zeros(2, device=dev))
            relp = pos[:, :2] - pred[None]                    # relative motion (candidate vs CV-predicted target)
            rank = torch.softmax(cos * 5.0, dim=0)            # relative-to-neighbors (star-topology, GNN-style)
        x = torch.stack([cos, relp[:, 0], relp[:, 1], pos[:, 2], rank], dim=-1)  # (n,5) ALL vehicle-invariant
        logits = self.mlp(x).squeeze(-1)
        srt = torch.sort(logits, descending=True).values
        conf = torch.sigmoid(srt[0] - srt[1]) if n >= 2 else torch.tensor(1.0, device=dev)
        return logits, conf, None

    def update(self, mem, target_feat, target_pos):
        g = torch.sigmoid(self.gate)
        m = dict(mem)
        if m["app"] is None:
            m["app"] = target_feat.clone()
        else:
            a = (1 - g) * m["app"] + g * target_feat
            m["app"] = a / (a.norm() + 1e-8)
        npos = target_pos[:2]
        m["vel"] = (npos - m["pos"]) if m["pos"] is not None else torch.zeros(2, device=target_feat.device)
        m["pos"] = npos
        return m


class TAHRelM(nn.Module):
    """TAHRel + LEARNED motion-consensus (path-A upgrade, distills borrow#1). TAHRel already had a raw
    relative-displacement feature but (a) fed the MLP only unbounded relp_x/relp_y — not a well-scaled
    distance/gated consensus — and (b) used a noisy single-step velocity. This adds the two things
    borrow#1's hand-tuned motion gate supplies, but LEARNED: an explicit gate-normalized soft motion
    score msoft = exp(−‖relp‖/gate) (bounded 0–1, learned radius) + the raw distance ‖relp‖, and an EMA
    velocity (learned decay) robust across losses. Targets the ONE metric TAHRel loses (q_reacq / re-
    acquisition), where DeepSORT & borrow#1 win via an explicit motion prior. 7 vehicle-invariant
    features; ~4.7K params (still cannot overfit). Same step/update signature → train & deploy unchanged."""
    def __init__(self, d_feat=384):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(-1.0))          # appearance-memory EMA gate (sigmoid)
        self.vbeta = nn.Parameter(torch.tensor(0.0))          # EMA velocity decay (sigmoid)
        self.mgate = nn.Parameter(torch.tensor(-2.3))         # soft motion-gate radius (softplus, img-frac)
        self.mlp = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def reset(self, device):
        return {"app": None, "pos": None, "vel": None}

    def step(self, feat, pos, mem):
        """feat:(n,384) L2-normed; pos:(n,3)=[u/W,v/H,depth]; mem dict. Returns logits:(n,), conf, None."""
        n = feat.shape[0]; dev = feat.device
        if mem["app"] is None:                                # cold start: no target memory yet
            cos = torch.zeros(n, device=dev); relp = torch.zeros(n, 2, device=dev)
            rank = torch.zeros(n, device=dev); dmot = torch.zeros(n, device=dev); msoft = torch.zeros(n, device=dev)
        else:
            cos = feat @ mem["app"]                           # relative appearance (vehicle-agnostic cosine)
            pred = mem["pos"] + (mem["vel"] if mem["vel"] is not None else torch.zeros(2, device=dev))
            relp = pos[:, :2] - pred[None]                    # relative motion (candidate vs CV-predicted target)
            rank = torch.softmax(cos * 5.0, dim=0)            # relative-to-neighbors (star-topology, GNN-style)
            dmot = relp.norm(dim=-1)                          # distance from CV prediction
            gate = F.softplus(self.mgate) + 1e-3
            msoft = torch.exp(-dmot / gate)                   # borrow#1-style bounded soft motion consensus (learned radius)
        x = torch.stack([cos, relp[:, 0], relp[:, 1], pos[:, 2], rank, dmot, msoft], dim=-1)  # (n,7) all invariant
        logits = self.mlp(x).squeeze(-1)
        srt = torch.sort(logits, descending=True).values
        conf = torch.sigmoid(srt[0] - srt[1]) if n >= 2 else torch.tensor(1.0, device=dev)
        return logits, conf, None

    def update(self, mem, target_feat, target_pos):
        g = torch.sigmoid(self.gate); m = dict(mem)
        if m["app"] is None:
            m["app"] = target_feat.clone()
        else:
            a = (1 - g) * m["app"] + g * target_feat
            m["app"] = a / (a.norm() + 1e-8)
        npos = target_pos[:2]
        step_v = (npos - m["pos"]) if m["pos"] is not None else torch.zeros(2, device=target_feat.device)
        if m["vel"] is None:
            m["vel"] = step_v
        else:
            b = torch.sigmoid(self.vbeta)
            m["vel"] = (1 - b) * m["vel"] + b * step_v        # EMA velocity — robust across losses
        m["pos"] = npos
        return m
