"""Train the Temporal Association Head (TAH, path-B). Supervised association over episode sequences:
roll the recurrent memory over frames, predict the target candidate (CE vs GT tidx), teacher-force the
memory update from the GT target embedding. Off-screen frames (tidx=-1) hold memory + skip loss.
Reports val target-accuracy + mis_follow (1-acc) to compare vs the heuristic stack."""
from __future__ import annotations
import argparse, os, sys
from collections import Counter
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
    ap.add_argument("--cache", nargs="+", default=["runs/tah_cache.pt"],
                    help="one or more caches; episodes are concatenated in the given order. The validation split is "
                         "taken from the FIRST cache only (first val_frac of its episodes) — identical to the "
                         "single-cache behaviour when one cache is given")
    ap.add_argument("--val-cache", nargs="+", default=None,
                    help="2026-10-12 P0-2: validate (and select the best epoch) on ALL episodes of these caches and "
                         "train on ALL episodes of --cache. For a deployment-distribution (DAgger) validation set — "
                         "the offline teacher-forced val saturates at ~0.99 from epoch 1. Default: legacy split")
    ap.add_argument("--group-phases", action="store_true",
                    help="2026-10-12 P0-2: training sequences cut from one source episode into phases (tah_cache "
                         "--tick-stride; frames carry 'ep') form one group, and each epoch trains on ONE randomly drawn "
                         "sequence per group → one gradient step per source episode, as without phases (keeps the "
                         "offline : DAgger step ratio of the recorded recipe). Sequences without 'ep' = own group")
    ap.add_argument("--seed", type=int, default=None, help="torch / numpy seed (training-seed replicates s0/s1/s2)")
    ap.add_argument("--save-last", action="store_true", help="also save the final-epoch weights to <out>_last.pt "
                    "(the offline val split can saturate early, so best-by-val may pick an under-trained epoch)")
    ap.add_argument("--out", default="runs/tah.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--rel", action="store_true", help="Tier-1: use TAHRel (relative/vehicle-invariant features)")
    ap.add_argument("--motion", action="store_true", help="path-A: use TAHRelM (TAHRel + learned motion-consensus + EMA vel)")
    ap.add_argument("--aug", action="store_true", help="Tier-2: candidate-permutation + feature-noise + pos-jitter")
    ap.add_argument("--fnoise", type=float, default=0.05)
    ap.add_argument("--pjit", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import torch
    from train.tah import TAH, TAHRel, TAHRelM
    if a.seed is not None:
        np.random.seed(a.seed); torch.manual_seed(a.seed)
    datas = [torch.load(c, weights_only=False) for c in a.cache]
    data = datas[0]
    assert all(d["d_feat"] == data["d_feat"] for d in datas), "caches disagree on d_feat"
    first = data["episodes"]
    if a.val_cache:
        same = {os.path.realpath(c) for c in a.val_cache} & {os.path.realpath(c) for c in a.cache}
        if same:
            raise SystemExit(f"--val-cache overlaps --cache: {sorted(same)}")
        vdatas = [torch.load(c, weights_only=False) for c in a.val_cache]
        assert all(d["d_feat"] == data["d_feat"] for d in vdatas), "val caches disagree on d_feat"
        val = [e for d in vdatas for e in d["episodes"]]
        tr = [e for d in datas for e in d["episodes"]]
        print(f"[tah] val caches: " + ", ".join(f"{c} ({len(d['episodes'])} eps)" for c, d in zip(a.val_cache, vdatas)),
              flush=True)
    else:
        nval = max(1, int(len(first) * a.val_frac))
        val = first[:nval]
        tr = first[nval:] + [e for d in datas[1:] for e in d["episodes"]]
    print(f"[tah] caches: " + ", ".join(f"{c} ({len(d['episodes'])} eps)" for c, d in zip(a.cache, datas)), flush=True)
    groups = None
    if a.group_phases:
        if not a.val_cache:
            raise SystemExit("--group-phases needs --val-cache (the legacy val split could cut through a group)")
        gidx, groups = {}, []
        for ci, d in enumerate(datas):
            for e in d["episodes"]:
                k = (ci, int(e[0]["ep"])) if "ep" in e[0] else None
                if k is not None:
                    assert all(int(fr["ep"]) == k[1] for fr in e), f"sequence mixes source episodes ({a.cache[ci]} {k})"
                if k is None or k not in gidx:
                    if k is not None:
                        gidx[k] = len(groups)
                    groups.append([e])
                else:
                    groups[gidx[k]].append(e)
        sizes = sorted(Counter(len(g) for g in groups).items())
        print(f"[tah] --group-phases: {len(tr)} train sequences in {len(groups)} groups (size: count {sizes}); "
              f"one sequence per group per epoch", flush=True)
    # checkpoint metadata: the 2026-10-12 options appear only when used (legacy runs keep the legacy schema)
    recipe = {k: v for k, v in vars(a).items() if not (k in ("val_cache", "group_phases") and not v)}
    extra = {} if groups is None else {"n_groups": len(groups)}
    arch = "relm" if a.motion else ("rel" if a.rel else "abs")
    print(f"[tah] {len(tr)} train / {len(val)} val episodes; d_feat={data['d_feat']}; "
          f"arch={arch} aug={a.aug} (fnoise={a.fnoise} pjit={a.pjit})", flush=True)
    rel = a.rel or a.motion   # both TAHRel/TAHRelM use the relative update path in run_epoch
    a.rel = rel
    model = (TAHRelM(d_feat=data["d_feat"]) if a.motion else
             TAHRel(d_feat=data["d_feat"]) if a.rel else TAH(d_feat=data["d_feat"])).to(a.device)
    print(f"[tah] params: {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    fn, pj, pm = (a.fnoise, a.pjit, True) if a.aug else (0.0, 0.0, False)
    best = 0.0
    for ep in range(a.epochs):
        tr_ep = tr if groups is None else [g[np.random.randint(len(g))] for g in groups]
        tl, ta = run_epoch(model, tr_ep, opt, a.device, train=True, rel=a.rel, fnoise=fn, pjit=pj, perm=pm)
        with torch.no_grad():
            vl, va = run_epoch(model, val, opt, a.device, train=False, rel=a.rel)
        if va > best:
            best = va
            torch.save({"model": model.state_dict(), "d_feat": data["d_feat"], "arch": arch,
                        "recipe": recipe, "n_train_eps": len(tr), "n_val_eps": len(val), **extra,
                        "best_epoch": ep + 1, "best_val_acc": float(va)}, a.out)
        print(f"[tah] ep{ep+1}/{a.epochs} train_loss={tl:.3f} train_acc={ta:.3f} | "
              f"val_acc={va:.3f} (mis={1-va:.3f}) best={best:.3f}", flush=True)
    if a.save_last:
        lp = a.out[:-3] + "_last.pt" if a.out.endswith(".pt") else a.out + "_last"
        torch.save({"model": model.state_dict(), "d_feat": data["d_feat"], "arch": arch,
                    "recipe": recipe, "n_train_eps": len(tr), "n_val_eps": len(val), **extra,
                    "epoch": a.epochs, "val_acc": float(va)}, lp)
        print(f"[tah] last-epoch weights → {lp}", flush=True)
    print(f"[tah] done. best val_acc={best:.3f} (mis_follow={1-best:.3f}) → {a.out}", flush=True)
    print(f"[tah] compare vs heuristic stack: l5 offline mis ~0.26 (1-0.738); reid-dino gallery.", flush=True)


if __name__ == "__main__":
    main()
