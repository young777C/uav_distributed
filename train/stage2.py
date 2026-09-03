"""Stage 2 — end-to-end joint training with difficulty curriculum.

Trains EAR (warm-started) + IAR + AGP/DiT + target-id head. Multi-task loss:

  L = L_action + 0.3·L_ear + 0.1·(L_visibility + L_maneuver) + 0.2·L_target_id

with: visibility-masked L_action (no blind-follow supervision), staleness-augmented
Z^ex, and a curriculum sampler annealing a->c. `run_training_stage2` is shared with
smoke_stage2.py so the whole loop is testable on synthetic data (no VLM/CARLA/GPU).

    python -m train.stage2 --config train/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from .curriculum import sample_weights
from .dataset_stage2 import collate_stage2
from .dit import DiT, TargetIDHead
from .ear import EAR
from .flow_matching import cfm_loss
from .iar import IAR
from .stage1_ear import safe_device


# --- action-chunk flow matching (DiT with multi-conditioning) ---------------
def action_flow_loss(dit, a1, z_ex, z_im, vlm_ctx, proprio, vlm_mask, visible):
    b = a1.shape[0]
    z = torch.randn_like(a1)
    t = torch.rand(b, device=a1.device)
    tt = t.view(b, 1, 1)
    a_t = (1 - tt) * z + tt * a1
    v = a1 - z
    v_hat = dit(a_t, t, z_ex, z_im, vlm_ctx, proprio, vlm_mask)
    per = ((v_hat - v) ** 2).mean(dim=(1, 2))            # per-sample
    w = visible                                          # supervise mask: visible OR intercept
    return (per * w).sum() / (w.sum() + 1e-6)            # (see dataset_stage2 §3.2 act_supervise)


@torch.no_grad()
def action_sample(dit, z_ex, z_im, vlm_ctx, proprio, vlm_mask, h, a_dim, steps):
    b = z_ex.shape[0]
    x = torch.randn(b, h, a_dim, device=z_ex.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=z_ex.device)
        x = x + dt * dit(x, t, z_ex, z_im, vlm_ctx, proprio, vlm_mask)
    return x


def _to(v, device):
    """Move tensors / lists-of-tensors to device; leave strings/other untouched."""
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, list) and v and torch.is_tensor(v[0]):
        return [x.to(device) for x in v]
    return v


def _ctx_vec(vlm_ctx, ctx_mask):
    """Masked mean of VLM context tokens -> a language-conditioned scene summary."""
    m = ctx_mask.unsqueeze(-1).float()
    return (vlm_ctx * m).sum(1) / m.sum(1).clamp_min(1.0)


def _stale(z_ex, p, noise):
    """Staleness augmentation: with prob p, degrade the coarse guidance (placeholder
    for real t-δ reuse). Matches the S1/S2 async gap at inference."""
    if p <= 0:
        return z_ex
    mask = (torch.rand(z_ex.shape[0], 1, 1, device=z_ex.device) < p).float()
    return z_ex + mask * noise * torch.randn_like(z_ex)


def run_training_stage2(train_ds, val_ds, *, cfg, ear_ckpt=None, tag="stage2"):
    device = safe_device(cfg["train"].get("device", "auto"))
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)
    import random as _random; _random.seed(seed)
    s = train_ds[0]
    cond_dim = s["vlm_ctx"].shape[-1]
    n_layers_in = len(s["vlm_ctx_layers"])
    cand_dim = s["cand_feats"].shape[-1]
    K = s["waypoint"].shape[0]
    H, A = s["action"].shape
    m = cfg["model"]
    print(f"[{tag}] device={device} train={len(train_ds)} val={len(val_ds)} "
          f"cond_dim={cond_dim} L={n_layers_in} cand_dim={cand_dim}")

    pdim = int(s["proprio"].shape[0])
    ear = EAR(cond_dim, k=K, d_model=m["ear"]["d_model"],
              n_layers=m["ear"]["n_layers"], n_heads=m["ear"]["n_heads"], proprio_dim=pdim).to(device)
    if ear_ckpt and Path(ear_ckpt).exists():
        ear.load_state_dict(torch.load(ear_ckpt, map_location=device)); print(f"[{tag}] EAR warm-start <- {ear_ckpt}")
    iar = IAR(cond_dim, n_layers_in, d=m["iar"]["d"], n_im=m["iar"]["n_im"],
              n_heads=m["iar"]["n_heads"]).to(device)
    dit = DiT(A, H, cond_dim, d_ex=3, d_im=m["iar"]["d"], proprio_dim=s["proprio"].shape[0],
              d=m["dit"]["d_model"], n_layers=m["dit"]["n_layers"], n_heads=m["dit"]["n_heads"]).to(device)
    tid = TargetIDHead(cand_dim, cond_dim, d=m.get("tid_d", 256)).to(device)
    # Separate, stronger weight decay on the tid head (it overfits the thin language
    # signal much faster than the action stack); cosine LR decay settles late training
    # so the early mis_follow peak is held instead of drifting back to chance.
    tid_params = list(tid.parameters())
    other_params = list(ear.parameters()) + list(iar.parameters()) + list(dit.parameters())
    params = other_params + tid_params
    opt = torch.optim.AdamW(
        [{"params": other_params, "weight_decay": cfg["train"].get("weight_decay", 1e-4)},
         {"params": tid_params, "weight_decay": cfg["train"].get("tid_weight_decay", 1e-2)}],
        lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

    lw = cfg["loss"]
    tiers = [train_ds[i]["tier"] for i in range(len(train_ds))]
    nw = int(cfg["train"].get("workers", 0))       # mmap cache -> workers>0 safe (shared pages)
    dl_kw = dict(num_workers=nw, collate_fn=collate_stage2, pin_memory=("cuda" in str(device)))
    if nw > 0:
        dl_kw["persistent_workers"] = True; dl_kw["prefetch_factor"] = 4
    vl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], **dl_kw)
    stale_p = cfg["train"].get("staleness_p", 0.3)
    steps = cfg["flow"]["sample_steps"]

    ckpt_dir = Path(cfg["train"].get("stage2_ckpt_dir", "runs/stage2"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Full hyperparams stored in the ckpt so downstream runs INHERIT them (rigor
    # guardrail: reproduce/compare by inheriting a reference run's params + OFAT,
    # never hand-retype lr/epochs/split/fov — see memory acot-uav-experiment-rigor).
    hparams = {
        "lr": cfg["train"]["lr"], "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "weight_decay": cfg["train"].get("weight_decay", 1e-4),
        "tid_weight_decay": cfg["train"].get("tid_weight_decay", 1e-2),
        "split_manifest": cfg["data"].get("split_manifest"),
        "stage2_context_cache": cfg["data"].get("stage2_context_cache"),
        "ctx_grid": cfg["backbone"].get("ctx_grid"), "vlm_layer": cfg["backbone"].get("layer"),
        "image_wh": [cfg["image"]["width"], cfg["image"]["height"]], "fov_deg": cfg["image"]["fov_deg"],
        "wp_scale": cfg["waypoint"]["scale"], "wp_offsets_s": cfg["waypoint"]["offsets_s"],
        "flow_steps": cfg["flow"]["sample_steps"], "tid_d": cfg["model"].get("tid_d", 256),
        "supervise_intercept": cfg["train"].get("supervise_intercept", False),
    }

    def _save(name, ep, met):
        torch.save({"ear": ear.state_dict(), "iar": iar.state_dict(),
                    "dit": dit.state_dict(), "tid": tid.state_dict(),
                    "epoch": ep, "metrics": met, "hparams": hparams,
                    "dims": {"cond_dim": cond_dim, "K": K, "H": H, "A": A,
                             "n_layers_in": n_layers_in, "cand_dim": cand_dim, "pdim": pdim}},
                   ckpt_dir / name)

    cw = cfg.get("curriculum", {})
    a_floor = float(cw.get("a_floor", 0.15)); ramp = float(cw.get("ramp", 1.2))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    # Module ablations (H1/H2): zero the reasoner's conditioning into the DiT so the
    # action head learns WITHOUT it, and drop that module's own loss. The tid head is
    # untouched (mis_follow unaffected) — the H1/H2 signal is action quality, not消歧.
    abl = cfg.get("ablate", {})
    abl_ear = bool(abl.get("ear", False))     # w/o-EAR (H1): z_ex -> 0, drop L_ear
    abl_iar = bool(abl.get("iar", False))     # w/o-IAR (H2): z_im -> 0 into DiT, drop L_vis/L_man
    print(f"[{tag}] seed={seed} a_floor={a_floor} ramp={ramp} grad_clip={grad_clip}"
          f"{' ABLATE=EAR' if abl_ear else ''}{' ABLATE=IAR' if abl_iar else ''}")

    best = float("inf"); metrics = {}
    for ep in range(cfg["train"]["epochs"]):
        w = sample_weights(tiers, ep, cfg["train"]["epochs"], a_floor, ramp)
        sampler = WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)
        tl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
                        collate_fn=collate_stage2, num_workers=nw,
                        pin_memory=("cuda" in str(device)),
                        **({"prefetch_factor": 4} if nw > 0 else {}))
        for m_ in (ear, iar, dit, tid): m_.train()
        agg = {"act": 0, "ear": 0, "vis": 0, "man": 0, "tid": 0}
        for b in tl:
            b = {k: _to(v, device) for k, v in b.items()}
            z_ex = _stale(b["waypoint"], stale_p, cfg["train"].get("staleness_noise", 0.2))
            if abl_ear:
                z_ex = torch.zeros_like(z_ex)                # w/o-EAR: no waypoint guidance to DiT
            z_im, aux = iar(b["vlm_ctx_layers"], b["ctx_mask"])
            z_im_dit = torch.zeros_like(z_im) if abl_iar else z_im   # w/o-IAR: no prior to DiT
            l_act = action_flow_loss(dit, b["action"], z_ex, z_im_dit, b["vlm_ctx"],
                                     b["proprio"], b["ctx_mask"], b["act_supervise"])
            l_ear = cfm_loss(ear, b["waypoint"], b["vlm_ctx"], b["ctx_mask"], proprio=b["proprio"])
            l_vis = F.mse_loss(torch.sigmoid(aux["occ_structural"]), b["occ_structural"])
            l_man = F.binary_cross_entropy_with_logits(aux["maneuver"], b["maneuver"])
            logits = tid(b["cand_feats"], b["vlm_ctx"], b["ctx_mask"], b["cand_mask"])
            tv = b["tid_valid"]                              # 0 on target-absent (loss) frames
            l_tid = (F.cross_entropy(logits, b["target_idx"], reduction="none") * tv).sum() / tv.sum().clamp_min(1.0)
            loss = (l_act + (0.0 if abl_ear else lw["ear"]) * l_ear
                    + (0.0 if abl_iar else lw["vis"]) * l_vis
                    + (0.0 if abl_iar else lw["maneuver"]) * l_man
                    + lw["target_id"] * l_tid)
            opt.zero_grad(); loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            for k, val in zip(agg, [l_act, l_ear, l_vis, l_man, l_tid]): agg[k] += val.item()
        if ep % cfg["eval"].get("every", 5) == 0 or ep == cfg["train"]["epochs"] - 1:
            metrics = _evaluate(ear, iar, dit, tid, vl, device, steps, K, H, A,
                                abl_ear=abl_ear, abl_iar=abl_iar)
            n = max(len(tl), 1)
            # select best on mis_follow (the language-grounding metric this experiment
            # targets), NOT act_mse — act_mse keeps improving while mis_follow overfits,
            # so an act_mse-best ckpt saved the WORST language epoch.
            if metrics["mis_follow"] < best:
                best = metrics["mis_follow"]
                _save(f"{tag}_best.pt", ep, metrics)
            _save(f"{tag}_last.pt", ep, metrics)
            print(f"[{tag}] ep{ep:3d} " + " ".join(f"{k}={agg[k]/n:.3f}" for k in agg)
                  + f" | val mis_follow={metrics['mis_follow']:.3f} act_mse={metrics['act_mse']:.3f}"
                  + f" best_mis={best:.3f}")
        sched.step()
    return {"best_mis_follow": best, "final": metrics}


@torch.no_grad()
def _evaluate(ear, iar, dit, tid, loader, device, steps, K, H, A, abl_ear=False, abl_iar=False):
    for m_ in (ear, iar, dit, tid): m_.eval()
    mis, n_grp, se, n = 0, 0, 0.0, 0
    for b in loader:
        b = {k: _to(v, device) for k, v in b.items()}
        z_ex = torch.zeros_like(b["waypoint"]) if abl_ear else b["waypoint"]
        z_im, _ = iar(b["vlm_ctx_layers"], b["ctx_mask"])
        if abl_iar:
            z_im = torch.zeros_like(z_im)
        pred = action_sample(dit, z_ex, z_im, b["vlm_ctx"], b["proprio"],
                             b["ctx_mask"], H, A, steps)
        se += F.mse_loss(pred, b["action"], reduction="sum").item(); n += b["action"].numel()
        logits = tid(b["cand_feats"], b["vlm_ctx"], b["ctx_mask"], b["cand_mask"])
        tv = b["tid_valid"] > 0.5                        # mis_follow only over target-present frames
        mis += int(((logits.argmax(1) != b["target_idx"]) & tv).sum()); n_grp += int(tv.sum())
    return {"mis_follow": mis / max(n_grp, 1), "act_mse": se / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    ap.add_argument("--ckpt-dir", default=None, help="override cfg.train.stage2_ckpt_dir")
    ap.add_argument("--device", default=None, help="override cfg.train.device, e.g. cuda:3")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.ckpt_dir:
        cfg["train"]["stage2_ckpt_dir"] = a.ckpt_dir
    if a.device:
        cfg["train"]["device"] = a.device
    from .dataset_stage2 import RealStage2Dataset  # noqa: built at real-run time
    train_ds = RealStage2Dataset(cfg, "train"); val_ds = RealStage2Dataset(cfg, "val")
    if len(train_ds) == 0:
        raise SystemExit("Empty Stage-2 set. Precompute multi-layer context first:\n"
                         "  python -m train.backbone_kv --config train/config.yaml --layers 4 8 12 16")
    run_training_stage2(train_ds, val_ds, cfg=cfg,
                        ear_ckpt=cfg["train"].get("ear_ckpt", "runs/stage1_ear/stage1_ear_best.pt"))


if __name__ == "__main__":
    main()
