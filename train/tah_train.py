"""Train the Temporal Association Head (TAH, path-B). Supervised association over episode sequences:
roll the recurrent memory over frames, predict the target candidate (CE vs GT tidx), teacher-force the
memory update from the GT target embedding. Off-screen frames (tidx=-1) hold memory + skip loss.
Reports val target-accuracy + mis_follow (1-acc) to compare vs the heuristic stack."""
from __future__ import annotations
import argparse, os, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)


def run_epoch(model, episodes, opt, device, train=True, rel=False, fnoise=0.0, pjit=0.0, perm=False):
    import torch
    model.train(train)
    tot_loss = n_loss = correct = total = 0
    order = np.random.permutation(len(episodes)) if train else range(len(episodes))
    for ei in order:
        frames = episodes[ei]
        m = model.reset(device)
        losses = []
        for fr in frames:
            feat = torch.tensor(np.asarray(fr["feat"], np.float32), device=device)
            pos = torch.tensor(fr["pos"], device=device)
            tidx = int(fr["tidx"])
            if train:                                               # Tier-2 augmentation (train only)
                if fnoise > 0:
                    feat = feat + fnoise * torch.randn_like(feat); feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                if pjit > 0:
                    pos = pos.clone(); pos[:, :2] = pos[:, :2] + pjit * torch.randn_like(pos[:, :2])
                if perm and feat.shape[0] > 1:                      # candidate-order permutation invariance
                    p = torch.randperm(feat.shape[0], device=device)
                    feat, pos = feat[p], pos[p]
                    if tidx >= 0: tidx = int((p == tidx).nonzero()[0])
            logits, conf, e = model.step(feat, pos, m)
            if tidx >= 0:
                losses.append(torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                              torch.tensor([tidx], device=device)))
                pred = int(logits.argmax()); correct += (pred == tidx); total += 1
                if rel:
                    m = model.update(m, feat[tidx], pos[tidx])                   # TAHRel: DINOv2 feat + pos
                else:
                    m = model.update(m, e[tidx].detach() if not train else e[tidx])
            # tidx<0 (target off-screen): hold memory, no loss
        if losses:
            loss = torch.stack(losses).mean()
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot_loss += float(loss); n_loss += 1
    return tot_loss / max(n_loss, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/tah_cache.pt")
    ap.add_argument("--out", default="runs/tah.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--rel", action="store_true", help="Tier-1: use TAHRel (relative/vehicle-invariant features)")
    ap.add_argument("--aug", action="store_true", help="Tier-2: candidate-permutation + feature-noise + pos-jitter")
    ap.add_argument("--fnoise", type=float, default=0.05)
    ap.add_argument("--pjit", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import torch
    from train.tah import TAH, TAHRel
    data = torch.load(a.cache, weights_only=False)
    eps = data["episodes"]
    nval = max(1, int(len(eps) * a.val_frac))
    val, tr = eps[:nval], eps[nval:]
    print(f"[tah] {len(tr)} train / {len(val)} val episodes; d_feat={data['d_feat']}; "
          f"rel={a.rel} aug={a.aug} (fnoise={a.fnoise} pjit={a.pjit})", flush=True)
    model = (TAHRel(d_feat=data["d_feat"]) if a.rel else TAH(d_feat=data["d_feat"])).to(a.device)
    print(f"[tah] params: {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    fn, pj, pm = (a.fnoise, a.pjit, True) if a.aug else (0.0, 0.0, False)
    best = 0.0
    for ep in range(a.epochs):
        tl, ta = run_epoch(model, tr, opt, a.device, train=True, rel=a.rel, fnoise=fn, pjit=pj, perm=pm)
        with torch.no_grad():
            vl, va = run_epoch(model, val, opt, a.device, train=False, rel=a.rel)
        if va > best:
            best = va; torch.save({"model": model.state_dict(), "d_feat": data["d_feat"]}, a.out)
        print(f"[tah] ep{ep+1}/{a.epochs} train_loss={tl:.3f} train_acc={ta:.3f} | "
              f"val_acc={va:.3f} (mis={1-va:.3f}) best={best:.3f}", flush=True)
    print(f"[tah] done. best val_acc={best:.3f} (mis_follow={1-best:.3f}) → {a.out}", flush=True)
    print(f"[tah] compare vs heuristic stack: l5 offline mis ~0.26 (1-0.738); reid-dino gallery.", flush=True)


if __name__ == "__main__":
    main()
