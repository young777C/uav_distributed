"""Stage 1 — EAR warmup.

Freeze VLM/IAR/DiT; train EAR to denoise K coarse 3D waypoints from the cached
VLM context (flow matching). Gate: 6s-horizon waypoint error < threshold.

    python -m train.stage1_ear --config train/config.yaml

`run_training` is shared with smoke_stage1.py so the exact loop is testable with
synthetic data (no VLM, no CARLA, CPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import RealEARDataset, collate
from .ear import EAR
from .flow_matching import cfm_loss, sample


def safe_device(pref="auto") -> str:
    """Return a device that actually works (this box has A40s but a too-old driver)."""
    if pref not in ("auto", "cuda"):
        return pref
    try:
        if torch.cuda.is_available():
            (torch.zeros(2, device="cuda") + 1).sum().item()
            return "cuda"
    except Exception:
        pass
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, k, steps, scale, device):
    model.eval()
    se, n = 0.0, 0
    per_h = [0.0] * k                       # per-horizon L2 error (meters): +2s/+4s/+6s
    for b in loader:
        cond = b["cond"].to(device); mask = b["cond_mask"].to(device)
        wp = b["wp"].to(device); prop = b["proprio"].to(device)
        pred = sample(model, cond, mask, k, steps, proprio=prop)
        se += torch.nn.functional.mse_loss(pred, wp, reduction="sum").item()
        n += wp.numel()
        d = torch.linalg.norm((pred - wp) * scale, dim=-1)   # (B, K) meters per horizon
        for h in range(k):
            per_h[h] += d[:, h].sum().item()
    model.train()
    ns = max(len(loader.dataset), 1)
    out = {"wp_mse": se / max(n, 1), "err_h": [per_h[h] / ns for h in range(k)]}
    out["err_6s"] = per_h[k - 1] / ns       # alias for the gate/print (last = longest horizon)
    return out


def run_training(train_ds, val_ds, *, cond_dim, k, cfg, scale, tag="stage1"):
    device = safe_device(cfg["train"].get("device", "auto"))
    print(f"[{tag}] device={device}  train={len(train_ds)} val={len(val_ds)} cond_dim={cond_dim}")
    nw = cfg["train"].get("workers", 0)
    dev_gpu = device == "cuda"
    tl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    collate_fn=collate, num_workers=nw,
                    persistent_workers=nw > 0, pin_memory=dev_gpu,
                    prefetch_factor=(4 if nw > 0 else None))
    vl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate,
                    num_workers=nw, persistent_workers=nw > 0, pin_memory=dev_gpu)
    pdim = int(train_ds[0]["proprio"].shape[0]) if "proprio" in train_ds[0] else 0
    model = EAR(cond_dim=cond_dim, k=k, d_model=cfg["ear"]["d_model"],
                n_layers=cfg["ear"]["n_layers"], n_heads=cfg["ear"]["n_heads"],
                proprio_dim=pdim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 1e-4))
    steps = cfg["flow"]["sample_steps"]
    gate = cfg["eval"].get("gate_mse_6s_m", 10.0)
    best = float("inf")
    m = {"wp_mse": float("nan"), "err_6s": float("nan")}
    ckpt_dir = Path(cfg["train"].get("ckpt_dir", "runs/stage1_ear"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(cfg["train"]["epochs"]):
        tot = 0.0
        for b in tl:
            cond = b["cond"].to(device); mask = b["cond_mask"].to(device)
            wp = b["wp"].to(device); prop = b["proprio"].to(device)
            loss = cfm_loss(model, wp, cond, mask, proprio=prop)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % cfg["eval"].get("every", 5) == 0 or ep == cfg["train"]["epochs"] - 1:
            m = evaluate(model, vl, k, steps, scale, device)
            flag = "  ✓GATE" if m["err_6s"] < gate else ""
            print(f"[{tag}] ep{ep:3d} loss={tot/max(len(tl),1):.4f} "
                  f"wp_mse={m['wp_mse']:.4f} err_6s={m['err_6s']:.2f}m{flag}")
            if m["wp_mse"] < best:
                best = m["wp_mse"]
                torch.save(model.state_dict(), ckpt_dir / f"{tag}_best.pt")
    return {"best_wp_mse": best, "final": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    train_ds = RealEARDataset(cfg, "train")
    val_ds = RealEARDataset(cfg, "val")
    if len(train_ds) == 0:
        raise SystemExit(
            "Empty training set. Run context precompute first:\n"
            "  python -m train.backbone_kv --config train/config.yaml\n"
            "(needs finished data + a working GPU).")
    run_training(train_ds, val_ds, cond_dim=train_ds.cond_dim,
                 k=cfg["waypoint"]["K"], cfg=cfg,
                 scale=cfg["waypoint"].get("scale", 50.0), tag="stage1_ear")


if __name__ == "__main__":
    main()
