"""Build a proper train/val/test split manifest (Phase A2).

Fixes the naive "first-N episodes = val" slice, which put ALL of Town01 in val
(town leakage + unrepresentative). Instead:

  - TEST  = one whole HELD-OUT town  -> measures cross-town generalization (§7.2).
  - VAL   = strategy-stratified sample of the remaining towns (deterministic).
  - TRAIN = the rest.

Episode-level (no frame leakage). Reads only attrs. Writes a JSON manifest that
RealEARDataset consumes. Run after data generation is finished:

    python -m train.make_split --config train/config.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml


def _attrs(ep):
    with h5py.File(ep, "r") as f:
        a = f.attrs
        def g(k):
            v = a.get(k, "")
            return v.decode() if isinstance(v, bytes) else str(v)
        return {"town": g("town"), "strategy": g("strategy"), "cls": g("target_class")}


def build(cfg):
    eps = sorted(glob.glob(cfg["data"]["episodes_glob"]))
    if not eps:
        raise SystemExit(f"no episodes at {cfg['data']['episodes_glob']}")
    meta = {ep: _attrs(ep) for ep in eps}
    towns = Counter(m["town"] for m in meta.values())

    # choose held-out test town: configured, else the smallest town (least train loss)
    test_town = cfg["data"].get("test_town") or min(towns, key=lambda t: towns[t])
    test = [ep for ep in eps if meta[ep]["town"] == test_town]
    pool = [ep for ep in eps if meta[ep]["town"] != test_town]

    # strategy-stratified val from the pool (deterministic per strategy group)
    val_frac = cfg["data"].get("val_frac", 0.15)
    rng = np.random.default_rng(cfg.get("seed", 0))
    by_strat = defaultdict(list)
    for ep in pool:
        by_strat[meta[ep]["strategy"]].append(ep)
    val, train = [], []
    for _strat, group in sorted(by_strat.items()):
        group = list(group)
        order = rng.permutation(len(group))
        n_val = max(1, math.floor(len(group) * val_frac))
        vi = set(order[:n_val].tolist())
        for i, ep in enumerate(group):
            (val if i in vi else train).append(ep)

    manifest = {
        "test_town": test_town,
        "train": sorted(Path(e).name for e in train),
        "val": sorted(Path(e).name for e in val),
        "test": sorted(Path(e).name for e in test),
    }
    out = Path(cfg["data"].get("split_manifest", "train/splits/mvp_split.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    def brk(names):
        ms = [meta[e] for e in eps if Path(e).name in set(names)]
        return dict(Counter(m["town"] for m in ms)), dict(Counter(m["strategy"] for m in ms))
    print(f"[split] test_town={test_town}  ->  {out}")
    for k in ["train", "val", "test"]:
        tt, ss = brk(manifest[k])
        print(f"  {k:5s} n={len(manifest[k]):3d}  towns={tt}")
        print(f"        strat={ss}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    build(cfg)


if __name__ == "__main__":
    main()
