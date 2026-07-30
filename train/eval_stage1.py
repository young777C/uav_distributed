"""Evaluate a trained EAR checkpoint with per-horizon waypoint error (no training).

    python -m train.eval_stage1 --config train/config.yaml [--ckpt ...] [--split val]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import RealEARDataset, collate
from .ear import EAR
from .stage1_ear import evaluate, safe_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="runs/stage1_ear/stage1_ear_best.pt")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    device = safe_device(cfg["train"].get("device", "auto"))

    ds = RealEARDataset(cfg, a.split)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate)
    k = cfg["waypoint"]["K"]
    m = cfg
    pdim = int(ds[0]["proprio"].shape[0]) if "proprio" in ds[0] else 0
    model = EAR(ds.cond_dim, k=k, d_model=cfg["ear"]["d_model"],
                n_layers=cfg["ear"]["n_layers"], n_heads=cfg["ear"]["n_heads"],
                proprio_dim=pdim).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    print(f"[eval] {a.ckpt} · split={a.split} · n={len(ds)} · device={device}")

    res = evaluate(model, dl, k, cfg["flow"]["sample_steps"], cfg["waypoint"].get("scale", 50.0), device)
    offsets_s = cfg["waypoint"].get("offsets_s", [2, 4, 6])
    gate = cfg["eval"].get("gate_mse_6s_m", 10.0)
    print(f"  wp_mse (normalized): {res['wp_mse']:.4f}")
    for off, e in zip(offsets_s, res["err_h"]):
        flag = " ✓<10m" if e < gate else ""
        print(f"  err_{off}s (m): {e:6.2f}{flag}")


if __name__ == "__main__":
    main()
