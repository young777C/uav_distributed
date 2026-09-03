"""Cheap CONFIRMATORY probe: is the wall the candidate FEATURES (resolution) or the
tid HEAD? Both cheap head levers (loss ce/margin/hardneg, capacity d512/h8) plateaued
at mis_follow ~0.50 with tier-c stuck ~0.62. Before paying for a 768/1024 resolution
bump, confirm the features actually can't resolve look-alikes.

For each tier-c strategy, linearly probe (in-distribution) the DISCRIMINATING attribute
of a candidate from its cached cand_feats (the exact vector the tid head sees):
  same_shape_diff_color  -> distractors share shape, differ in COLOR -> probe color
  same_color_diff_shape  -> share color, differ in TYPE/make        -> probe make(bp)
Verdict:
  attribute NOT readable (~chance)  -> 27px features can't tell look-alikes apart
                                       -> RESOLUTION WALL confirmed, the bump is justified
  attribute readable (>>chance)     -> features DO encode it, tid head can't use it
                                       -> a head/binding problem, do NOT pay for resolution

Sources cand_feats from the cache (no VLM encode) -> runs on the HOST, CPU, in ~1 min.
Mirrors the v4->v5 in-distribution methodology (make 55% in-dist vs 6% by-episode).
"""
from __future__ import annotations
import argparse, json

import numpy as np
import torch
import torch.nn as nn
import yaml

from train.dataset_stage2 import RealStage2Dataset
from train.dataset import _select_episodes
from acot_probe.backbones import pool_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ncap", type=int, default=12000, help="max candidates to collect")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    W, H = cfg["image"]["width"], cfg["image"]["height"]
    ds = RealStage2Dataset(cfg, "val")            # for _ctx/_h5/_candidates helpers
    ctx_dir = ds.ctx_dir
    stride = int(cfg["data"].get("frame_stride", 3)) * 4
    cap = int(cfg["data"].get("max_frames_per_episode", 500))
    fps = cfg.get("fps", 10)
    horizon = int(cfg["waypoint"].get("max_horizon_s", 6) * fps)

    import h5py
    attr_cache = {}
    def ep_attr(ep):
        if ep not in attr_cache:
            with h5py.File(ep, "r") as f:
                attr_cache[ep] = (f.attrs["target_color"], f.attrs["target_bp"],
                                  json.loads(str(f.attrs["distractors"])),
                                  str(f.attrs.get("strategy", "")))
        return attr_cache[ep]

    feats, cols, makes, strat, istgt, boxpx = [], [], [], [], [], []
    eps = _select_episodes(cfg, "train") + _select_episodes(cfg, "val")   # all (in-distribution)
    for ep in eps:
        if len(feats) >= a.ncap:
            break
        from pathlib import Path
        npz = ctx_dir / (Path(ep).stem + ".npz")
        if not npz.exists():
            continue
        try:
            h = ds._h5(ep)
        except Exception:
            continue
        tcol, tbp, distr, strategy = ep_attr(ep)
        frames, (gh, gw), ctx_all = ds._ctx(str(npz))
        n = len(h["tgt"])
        for i in list(range(0, max(0, n - horizon), stride))[:cap]:
            if len(feats) >= a.ncap:
                break
            cands = ds._candidates(h, i)
            if len(cands) < 2 or not any(c.is_target for c in cands):
                continue
            row = min(int(np.searchsorted(frames, i)), len(frames) - 1)
            grid = np.ascontiguousarray(ctx_all[row][-1]).astype(np.float32).reshape(gh, gw, ctx_all.shape[-1])
            for c in cands:
                feats.append(pool_box(grid, c.frac_box(W, H)).astype(np.float32))
                if c.is_target:
                    cols.append(str(tcol)); makes.append(str(tbp))
                else:
                    dd = distr[c.idx] if c.idx < len(distr) else {"color": "?", "bp": "?"}
                    cols.append(str(dd.get("color", "?"))); makes.append(str(dd.get("bp", "?")))
                strat.append(strategy); istgt.append(c.is_target); boxpx.append(float(c.w))

    X = np.stack(feats); cols = np.array(cols); makes = np.array(makes)
    strat = np.array(strat); boxpx = np.array(boxpx)
    coarse = np.array([s.split()[-1] for s in cols])          # "dark red"->red
    print(f"[attr-probe] collected {len(X)} candidates over {len(set(strat))} strategies; "
          f"strategies={sorted(set(strat))}")
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd

    def probe(mask, y_raw, name):
        idx = np.where(mask)[0]
        if len(idx) < 200:
            print(f"  {name}: n={len(idx)} too few, skip"); return
        y_raw = y_raw[idx]; Xi = Xn[idx]; bxi = boxpx[idx]
        voc = {v: k for k, v in enumerate(sorted(set(y_raw)))}
        y = np.array([voc[v] for v in y_raw])
        g = np.random.default_rng(0).permutation(len(idx)); ntr = int(len(idx) * 0.8)
        tri, tei = g[:ntr], g[ntr:]
        net = nn.Linear(Xi.shape[1], len(voc)); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
        Xg, yg = torch.tensor(Xi[tri]), torch.tensor(y[tri])
        for _ in range(150):
            opt.zero_grad(); nn.functional.cross_entropy(net(Xg), yg).backward(); opt.step()
        with torch.no_grad():
            pred = net(torch.tensor(Xi[tei])).argmax(1).numpy()
        ok = (pred == y[tei]); chance = 100.0 / len(voc)
        print(f"  {name}: classes={len(voc)} chance={chance:.1f}%  IN-DIST acc={100*ok.mean():.1f}%  (n_te={len(tei)})")
        bxt = bxi[tei]
        for lo, hi in [(0, 25), (25, 40), (40, 300)]:
            m = (bxt >= lo) & (bxt < hi)
            if m.sum() > 30:
                print(f"      box {lo}-{hi}px: {100*ok[m].mean():.1f}%  (n={m.sum()})")

    ss = np.char.find(strat.astype(str), "same_shape") >= 0     # differ in COLOR
    sc = np.char.find(strat.astype(str), "same_color") >= 0     # differ in TYPE
    print("\n=== 判别属性可读性(in-distribution 线性探针)===")
    print("--- same_shape_diff_color 档:区分靠 COLOR ---")
    probe(ss, coarse, "  color(coarse)")
    probe(ss, cols, "  color(fine)  ")
    print("--- same_color_diff_shape 档:区分靠 TYPE/make ---")
    probe(sc, makes, "  make(bp)     ")
    print("--- 参考:全体候选 ---")
    probe(np.ones(len(X), bool), coarse, "  color(coarse)")
    probe(np.ones(len(X), bool), makes, "  make(bp)     ")
    print("\n判读: 各档'区分属性'IN-DIST 若接近 chance → 27px 特征分不出 look-alike → 分辨率墙坐实(值得提到768/1024); "
          "若 >>chance → 特征已含判别信息、tid 用不上 → 是头/绑定问题,别花钱提分辨率。")


if __name__ == "__main__":
    main()
