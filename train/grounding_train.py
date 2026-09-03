"""Fast standalone tid-head trainer — the grounding main-attack substrate.

tid gradients don't touch any other module (VLM frozen, cand_feats/vlm_ctx cached),
so training tid ALONE on the cached features reaches the same optimum as joint
Stage-2 but in minutes — a fast loop to try grounding levers:

  --loss ce        plain cross-entropy (reproduces Stage-2 mis_follow ~0.58 = control)
  --loss hardneg   CE upweighted on hard frames (tier-c and/or many candidates)
  --loss margin    max-margin: push target logit above the BEST distractor by --margin

Reports mis_follow overall + by tier each epoch (grounding_probe metric). If every
lever plateaus at ~0.58 → the wall is candidate-FEATURE resolution (ctx_grid), not
the head → switch to the re-precompute lever. If a lever breaks below → cheap win.
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import torch.nn as nn
from train.dataset_stage2 import RealStage2Dataset, collate_stage2
from train.dit import TargetIDHead


class AttrBindHead(nn.Module):
    """Explicit language-binding head: score each candidate by how well its feature
    MATCHES the target's (color, make) embedding. Tests whether the tid failure is a
    BINDING problem (attr-probe showed features encode color/make ~50% in-dist, well
    above chance) rather than features/capacity. Ignores vlm_ctx on purpose — isolates
    'can we bind language identity to candidate features'. GT color/make = the parsed
    language identity (neutralize_language keeps exactly color+make)."""
    def __init__(self, cand_dim, n_color, n_make, d=256, dropout=0.1):
        super().__init__()
        self.cand = nn.Sequential(nn.Linear(cand_dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.color_emb = nn.Embedding(n_color, d)
        self.make_emb = nn.Embedding(n_make, d)
        self.desc = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.scale = d ** -0.5

    def forward(self, cand_feats, color_id, make_id, cand_mask=None):
        cq = self.cand(cand_feats)                                          # (B,N,d)
        desc = self.desc(torch.cat([self.color_emb(color_id), self.make_emb(make_id)], -1))  # (B,d)
        s = (cq * desc[:, None, :]).sum(-1) * self.scale                    # (B,N) match score
        if cand_mask is not None:
            s = s.masked_fill(~cand_mask, float("-inf"))
        return s


class CombinedHead(nn.Module):
    """attrbind(identity match) ⊕ xattn(context cross-attn): sum of per-candidate logits.
    Wires straight into Stage-2 as the tid head if it wins (identity + spatial context)."""
    def __init__(self, cand_dim, ctx_dim, n_color, n_make, d=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.attr = AttrBindHead(cand_dim, n_color, n_make, d=d, dropout=dropout)
        self.xattn = TargetIDHead(cand_dim, ctx_dim, d=d, n_heads=n_heads, dropout=dropout)

    def forward(self, cand_feats, color_id, make_id, vlm_ctx, ctx_mask, cand_mask=None):
        s = self.attr(cand_feats, color_id, make_id) + self.xattn(cand_feats, vlm_ctx, ctx_mask)
        if cand_mask is not None:
            s = s.masked_fill(~cand_mask, float("-inf"))
        return s


def build_attr_vocab(cfg):
    """color/make vocab from all episodes' target identity attrs (+ <unk>)."""
    import h5py
    from train.dataset import _select_episodes
    cols, makes = set(), set()
    for ep in _select_episodes(cfg, "train") + _select_episodes(cfg, "val"):
        try:
            with h5py.File(ep, "r") as f:
                cols.add(str(f.attrs.get("target_color", "?")))
                makes.add(str(f.attrs.get("target_bp", "?")))
        except Exception:
            pass
    cvoc = {v: i for i, v in enumerate(sorted(cols))}; cvoc["<unk>"] = len(cvoc)
    mvoc = {v: i for i, v in enumerate(sorted(makes))}; mvoc["<unk>"] = len(mvoc)
    return cvoc, mvoc


def _attr_ids(b, cvoc, mvoc, dev):
    ci = torch.tensor([cvoc.get(c, cvoc["<unk>"]) for c in b["target_colors"]], device=dev)
    mi = torch.tensor([mvoc.get(m, mvoc["<unk>"]) for m in b["target_makes"]], device=dev)
    return ci, mi


def _eval(head, logits_fn, dl, dev):
    head.eval()
    tot = [0, 0]; by_tier = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for b in dl:
            logits = logits_fn(b)
            pred = logits.argmax(1).cpu().numpy()
            tgt = b["target_idx"].numpy(); tv = (b["tid_valid"] > 0.5).numpy()
            tiers = b["tiers"]
            for i in range(len(pred)):
                if not tv[i]:
                    continue
                w = int(pred[i] != tgt[i]); tot[0] += w; tot[1] += 1
                by_tier[tiers[i]][0] += w; by_tier[tiers[i]][1] += 1
    r = lambda c: c[0] / c[1] if c[1] else float("nan")
    return r(tot), {t: r(by_tier[t]) for t in by_tier}


def _loss(logits, b, dev, kind, margin, hard_w):
    tv = (b["tid_valid"] > 0.5).to(dev)
    if not tv.any():
        return None
    tgt = b["target_idx"].to(dev)
    lg = logits[tv]; tg = tgt[tv]
    if kind == "margin":
        pos = lg.gather(1, tg[:, None]).squeeze(1)                 # target logit
        neg = lg.clone(); neg.scatter_(1, tg[:, None], float("-inf"))
        neg_max = neg.max(1).values                                # best distractor
        return F.relu(margin - (pos - neg_max)).mean()
    ce = F.cross_entropy(lg, tg, reduction="none")                 # (Bv,)
    if kind == "hardneg":
        tiers = [t for t, keep in zip(b["tiers"], tv.cpu().numpy()) if keep]
        wts = torch.tensor([hard_w if t == "c" else 1.0 for t in tiers], device=dev)
        return (ce * wts).sum() / wts.sum()
    return ce.mean()                                               # ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default=None, help="warmstart tid from a stage2 ckpt (no param inherit)")
    ap.add_argument("--from-run", default=None,
                    help="reference stage2 ckpt: INHERIT its hparams (lr/epochs/...) + warmstart tid "
                         "+ POSITIVE-CONTROL init eval vs its recorded mis_follow (rigor guardrail)")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--loss", default="ce", choices=["ce", "hardneg", "margin"])
    # lr/epochs/wd default None = 'not explicitly set' -> resolved from --from-run then fallback (OFAT)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--hard-w", type=float, default=3.0)
    ap.add_argument("--head", default="xattn", choices=["xattn", "attrbind", "combined"],
                    help="xattn=cand→ctx cross-attn | attrbind=cand·(color+make embed) | combined=attrbind⊕xattn")
    ap.add_argument("--tid-d", type=int, default=None)
    ap.add_argument("--tid-heads", type=int, default=4, help="capacity gate: keep d/heads=64/head (d512->h8)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=0.03, help="positive-control tolerance on init mis_follow")
    ap.add_argument("--strict", action="store_true", help="abort (not warn) if positive control fails")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(a.config))
    # tid needs ONLY the last VLM layer (vlm_ctx) + cand_feats. Restricting the dataset
    # to L=1 cuts the per-sample tensor 5x -> avoids DataLoader /dev/shm blow-up (the
    # 5-layer vlm_ctx_layers is 13MB/sample) and is faster.
    cfg.setdefault("data", {})["use_last_layer_only"] = True

    # --- resolve hyperparams: explicit CLI > --from-run hparams > fallback (OFAT) ---
    ref = torch.load(a.from_run, map_location="cpu") if a.from_run else None
    ref_hp = (ref or {}).get("hparams", {}) if ref else {}
    def resolve(cli, key, fallback):
        if cli is not None: return cli, "cli"
        if key in ref_hp and ref_hp[key] is not None: return ref_hp[key], f"from-run:{a.from_run}"
        return fallback, "fallback"
    lr, lr_src = resolve(a.lr, "lr", 2e-4)
    epochs, ep_src = resolve(a.epochs, "epochs", 40)
    wd, wd_src = resolve(a.wd, "tid_weight_decay", 1e-2)
    if ref_hp:
        print(f"[gtrain] inherited hparams from {a.from_run}: lr={lr}({lr_src}) "
              f"epochs={epochs}({ep_src}) wd={wd}({wd_src})")

    ds_tr = RealStage2Dataset(cfg, "train"); ds_va = RealStage2Dataset(cfg, "val")
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True, collate_fn=collate_stage2,
                       num_workers=4, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=128, collate_fn=collate_stage2, num_workers=2)

    cand_dim = ds_tr[0]["cand_feats"].shape[-1]; cond_dim = ds_tr.cond_dim
    td = a.tid_d or cfg["model"].get("tid_d", 256)
    cvoc = mvoc = None
    if a.head == "attrbind":
        cvoc, mvoc = build_attr_vocab(cfg)
        head = AttrBindHead(cand_dim, len(cvoc), len(mvoc), d=td, dropout=a.dropout).to(dev)
        logits_fn = lambda b: head(b["cand_feats"].to(dev), *_attr_ids(b, cvoc, mvoc, dev), b["cand_mask"].to(dev))
        print(f"[gtrain] head=attrbind  n_color={len(cvoc)} n_make={len(mvoc)}")
    elif a.head == "combined":
        cvoc, mvoc = build_attr_vocab(cfg)
        head = CombinedHead(cand_dim, cond_dim, len(cvoc), len(mvoc), d=td, n_heads=a.tid_heads, dropout=a.dropout).to(dev)
        logits_fn = lambda b: head(b["cand_feats"].to(dev), *_attr_ids(b, cvoc, mvoc, dev),
                                   b["vlm_ctx"].to(dev), b["ctx_mask"].to(dev), b["cand_mask"].to(dev))
        print(f"[gtrain] head=combined  n_color={len(cvoc)} n_make={len(mvoc)}")
    else:
        head = TargetIDHead(cand_dim, cond_dim, d=td, n_heads=a.tid_heads, dropout=a.dropout).to(dev)
        logits_fn = lambda b: head(b["cand_feats"].to(dev), b["vlm_ctx"].to(dev),
                                   b["ctx_mask"].to(dev), b["cand_mask"].to(dev))
    warm = ref if ref is not None else (torch.load(a.ckpt, map_location="cpu") if a.ckpt else None)
    warm_src = a.from_run or a.ckpt
    if warm is not None and a.head == "xattn":
        try: head.load_state_dict(warm["tid"]); print(f"[gtrain] warmstart tid from {warm_src}")
        except Exception as e: print(f"[gtrain] warmstart skipped ({e})")
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"[gtrain] head={a.head} loss={a.loss} epochs={epochs} lr={lr} d={td} heads={a.tid_heads} "
          f"| train={len(ds_tr)} val={len(ds_va)}  (margin={a.margin} hard_w={a.hard_w})")
    mf0, byt0 = _eval(head, logits_fn, dl_va, dev)
    print(f"[gtrain] init mis_follow={mf0:.3f}  " + " ".join(f"{k}={v:.3f}" for k, v in sorted(byt0.items())))

    # --- POSITIVE CONTROL: a warmstarted head MUST reproduce the ref's recorded mis_follow.
    # If not, the substrate (split/eval/cache) differs from how the ref was trained -> STOP;
    # any margin/hardneg delta measured against a broken substrate is meaningless.
    ref_mf = (ref or {}).get("metrics", {}).get("mis_follow") if (ref and a.head == "xattn") else None
    if ref_mf is not None:
        delta = abs(mf0 - ref_mf)
        ok = delta <= a.tol
        print(f"[gtrain] POSITIVE-CONTROL: init={mf0:.3f} vs recorded={ref_mf:.3f} "
              f"|Δ|={delta:.3f} tol={a.tol} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            msg = ("substrate mismatch (split/eval/cache differ from the ref's training) — "
                   "fix before trusting any loss comparison")
            if a.strict:
                raise SystemExit(f"[gtrain] ABORT: positive control FAILED — {msg}")
            print(f"[gtrain] WARNING: positive control FAILED — {msg}")
    best = 1.0
    for ep in range(epochs):
        head.train()
        for b in dl_tr:
            logits = logits_fn(b)
            L = _loss(logits, b, dev, a.loss, a.margin, a.hard_w)
            if L is None:
                continue
            opt.zero_grad(); L.backward(); opt.step()
        sched.step()
        mf, byt = _eval(head, logits_fn, dl_va, dev)
        tag = ""
        if mf < best:
            best = mf; tag = " *best"
            if a.out:
                torch.save({"tid": head.state_dict(), "mis_follow": mf, "by_tier": byt, "head": a.head}, a.out)
        print(f"[gtrain] ep{ep:02d} mis_follow={mf:.3f}  "
              + " ".join(f"{k}={v:.3f}" for k, v in sorted(byt.items())) + tag)
    print(f"[gtrain] BEST mis_follow={best:.3f}  (head={a.head} loss={a.loss})")


if __name__ == "__main__":
    main()
