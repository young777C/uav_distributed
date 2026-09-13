"""Tier-0 Probe B-2 (912 plan): does a LEARNED, cross-episode-amortized target-conditioned adapter
beat both the uniform baseline AND the naive per-episode fit — on HELD-OUT (unseen) episodes? This is
the decisive headroom test that probe B left open (in-sample oracle +0.11/+0.35 vs naive-online +0.03/+0.06).

Setup (fully deployable): target_emb = mean of target crop-DINOv2 feats over the first K=30 frames (what
you have at task start). A small head g = f(target_emb) outputs a per-channel modulation; candidates in
LATER frames are scored by cos(g⊙cand, g⊙target_emb) → pick target. Trained with CE over candidates on
TRAIN episodes, evaluated on disjoint VAL episodes (unseen targets). Compares:
  baseline (g=1)           uniform cosine
  diag  (g = softplus MLP) FiLM-γ learned modulation
  lowrank (g + U V^T proj) richer learned adapter (bounds the LDA headroom, still amortized)

  GO   : val Δpick (diag or lowrank) ≥ +0.05  → learned adapter generalizes → build full Tier-1 + closed-loop
  NO-GO: val Δpick ~0                          → the amortized adapter doesn't transfer either → ceiling for this idea
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


def load_episodes(cache, K, maxc):
    data = torch.load(cache, weights_only=False, map_location="cpu")
    eps = []
    for ep in data["episodes"]:
        frs = []
        for fr in ep:
            f = fr["feat"]; f = f.numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
            f = f.astype(np.float32); t = int(fr["tidx"])
            if t < 0 or t >= f.shape[0] or f.shape[0] < 2 or f.shape[0] > maxc:
                continue
            f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            frs.append((f, t))
        if len(frs) > K + 4:
            temb = np.mean([frs[i][0][frs[i][1]] for i in range(K)], axis=0)
            temb = temb / (np.linalg.norm(temb) + 1e-8)
            eps.append((temb.astype(np.float32), frs[K:]))       # (target_emb, eval-frames)
    return eps, data["d_feat"]


def batchify(eps, maxc, dev):
    """Flatten eval-frames into padded tensors: feats(N,maxc,D), mask(N,maxc), tgt(N,), temb(N,D)."""
    F, M, Y, E = [], [], [], []
    for temb, frs in eps:
        for f, t in frs:
            n = f.shape[0]; pad = np.zeros((maxc, f.shape[1]), np.float32); pad[:n] = f
            m = np.zeros(maxc, np.float32); m[:n] = 1.0
            F.append(pad); M.append(m); Y.append(t); E.append(temb)
    return (torch.tensor(np.array(F), device=dev), torch.tensor(np.array(M), device=dev),
            torch.tensor(np.array(Y), device=dev), torch.tensor(np.array(E), device=dev))


class Adapter(nn.Module):
    def __init__(self, d=384, rank=16, mode="diag"):
        super().__init__()
        self.mode = mode
        self.g = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d))   # → log-scale γ
        if mode == "lowrank":
            self.u = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d * rank))
            self.v = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d * rank))
            self.rank = rank; self.d = d

    def score(self, feats, mask, temb):
        g = torch.exp(0.5 * torch.tanh(self.g(temb)))                 # (B,D) positive γ near 1
        fw = feats * g[:, None, :]; tw = temb * g
        fw = fw / (fw.norm(dim=-1, keepdim=True) + 1e-8)
        tw = tw / (tw.norm(dim=-1, keepdim=True) + 1e-8)
        s = (fw * tw[:, None, :]).sum(-1)                             # (B,maxc) cosine
        if self.mode == "lowrank":
            B = temb.shape[0]
            U = self.u(temb).view(B, self.d, self.rank); V = self.v(temb).view(B, self.d, self.rank)
            pf = torch.einsum("bnd,bdr->bnr", feats, U)               # (B,maxc,r)
            pt = torch.einsum("bd,bdr->br", temb, V)                  # (B,r)
            s = s + (pf * pt[:, None, :]).sum(-1)
        return s.masked_fill(mask < 0.5, -1e9)


def pick_acc(model, F, M, Y, E, bs=4096):
    model.eval(); correct = tot = 0
    with torch.no_grad():
        for i in range(0, F.shape[0], bs):
            s = model.score(F[i:i+bs], M[i:i+bs], E[i:i+bs])
            correct += (s.argmax(1) == Y[i:i+bs]).sum().item(); tot += s.shape[0]
    return correct / tot


def baseline_acc(F, M, Y, E, bs=8192):
    correct = tot = 0
    for i in range(0, F.shape[0], bs):
        f, m, e = F[i:i+bs], M[i:i+bs], E[i:i+bs]
        s = (f * e[:, None, :]).sum(-1).masked_fill(m < 0.5, -1e9)
        correct += (s.argmax(1) == Y[i:i+bs]).sum().item(); tot += s.shape[0]
    return correct / tot


def train_eval(mode, tr, va, dev, epochs=40):
    model = Adapter(mode=mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Ft, Mt, Yt, Et = tr; bs = 4096
    for _ in range(epochs):
        model.train(); perm = torch.randperm(Ft.shape[0], device=dev)
        for i in range(0, Ft.shape[0], bs):
            idx = perm[i:i+bs]
            s = model.score(Ft[idx], Mt[idx], Et[idx])
            loss = nn.functional.cross_entropy(s, Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return pick_acc(model, *tr), pick_acc(model, *va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/tah_cache.pt")
    ap.add_argument("--K", type=int, default=30); ap.add_argument("--maxc", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    eps, d = load_episodes(a.cache, a.K, a.maxc)
    val_eps = eps[::5]; tr_eps = [e for i, e in enumerate(eps) if i % 5 != 0]   # cross-episode split
    print(f"[probeB2] {len(eps)} eps (train {len(tr_eps)} / val {len(val_eps)}), d={d}, dev={dev}")
    tr = batchify(tr_eps, a.maxc, dev); va = batchify(val_eps, a.maxc, dev)
    print(f"  train {tr[0].shape[0]} frames / val {va[0].shape[0]} frames\n")
    b_tr, b_va = baseline_acc(*tr), baseline_acc(*va)
    print(f"{'scheme':<20}{'train':>8}{'val':>8}{'val Δ':>9}")
    print("-" * 45)
    print(f"{'baseline (g=1)':<20}{b_tr:>8.3f}{b_va:>8.3f}{'—':>9}")
    for mode in ("diag", "lowrank"):
        t_tr, t_va = train_eval(mode, tr, va, dev)
        tag = "diag (FiLM-γ)" if mode == "diag" else "lowrank adapter"
        print(f"{tag:<20}{t_tr:>8.3f}{t_va:>8.3f}{t_va-b_va:>+9.3f}")
    print("\nGO if val Δ ≥ +0.05 (learned adapter generalizes to unseen targets) → build Tier-1 + closed-loop.")


if __name__ == "__main__":
    main()
