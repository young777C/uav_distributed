"""Orchestrate the layer-16 grounding-probe comparison of PaliGemma-2-3B vs Qwen3-VL.

Usage:
    python -m acot_probe.run_probe --config acot_probe/config.yaml
    python -m acot_probe.run_probe --config acot_probe/config.yaml --only paligemma
    python -m acot_probe.run_probe --config acot_probe/config.yaml --skip-cache  # reuse features

Stages
  1. cache : load each frozen backbone, run all frames, pool per-candidate layer
             features -> <cache_dir>/<name>.npz   (skippable)
  2. probe : train a tiny selection head per (backbone, layer); report TSA vs chance
  3. report: print a backbone × layer table + a selection recommendation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .backbones import build_backbone
from .data import build_cache, load_cache, resolve_episode_paths, save_cache
from .probe import run_layer_sweep


def _cache_one(spec, cfg, episodes, cache_path):
    bk = build_backbone(spec["kind"], spec["model_id"], spec["device"]).load()
    try:
        cache = build_cache(bk, cfg, episodes, cfg["layers"])
    finally:
        bk.free()
    save_cache(cache, cache_path)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="probe only this backbone name")
    ap.add_argument("--skip-cache", action="store_true", help="reuse cached features")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_dir = Path(cfg["cache_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = resolve_episode_paths(cfg)
    if not episodes:
        raise SystemExit(f"No episodes matched {cfg['data']['episodes_glob']!r}. "
                         "Generate a small similar-distractor probe set first "
                         "(see acot_probe/README.md).")
    print(f"[probe] {len(episodes)} episodes")

    names = [args.only] if args.only else list(cfg["backbones"])
    results = {}
    for name in names:
        spec = cfg["backbones"][name]
        cache_path = cache_dir / f"{name}.npz"
        if args.skip_cache and cache_path.exists():
            print(f"[cache] reuse {cache_path}")
            cache = load_cache(cache_path)
        else:
            print(f"[cache] {name}: {spec['model_id']} on {spec['device']}")
            cache = _cache_one(spec, cfg, episodes, cache_path)
        sweep = run_layer_sweep(cache, cfg)
        results[name] = {r.layer: r for r in sweep}

    _report(results, cfg, out_dir)


def _report(results, cfg, out_dir):
    layers = sorted({L for r in results.values() for L in r})
    thr = cfg.get("pass_tsa_at_16", 0.80)

    print("\n=== Target-Selection Accuracy (val)  [chance in brackets] ===")
    header = "backbone".ljust(12) + "".join(f"L{L:>7}" for L in layers)
    print(header)
    dump = {}
    for name, per in results.items():
        row = name.ljust(12)
        dump[name] = {}
        for L in layers:
            r = per.get(L)
            if r is None or r.tsa != r.tsa:            # nan
                row += " " * 8
                continue
            row += f" {r.tsa:.2f}[{r.chance:.2f}]"[:8]
            dump[name][L] = {"tsa": r.tsa, "chance": r.chance,
                             "n_val": r.n_groups_val, "n_train": r.n_groups_train}
        print(row)

    print("\n=== Selection verdict (design §3.4 gate) ===")
    for name, per in results.items():
        r16 = per.get(16)
        if r16 is None or r16.tsa != r16.tsa:
            print(f"  {name:<12} layer 16: n/a")
            continue
        ok = r16.tsa >= thr and r16.tsa > r16.chance + 0.15
        verdict = "PASS ✓ (layer-16 grounding usable)" if ok else \
            "WEAK ✗ (relax truncation deeper / prefer other backbone)"
        print(f"  {name:<12} layer 16 TSA={r16.tsa:.2f} (chance {r16.chance:.2f}) → {verdict}")

    (out_dir / "probe_results.json").write_text(json.dumps(dump, indent=2))
    print(f"\n[probe] wrote {out_dir/'probe_results.json'}")


if __name__ == "__main__":
    main()
