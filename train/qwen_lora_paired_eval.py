"""Paired, episode-clustered significance for Stage 0 (r16 LoRA vs r0 frozen probe).
Both checkpoints are evaluated on the SAME frames with the SAME candidate shuffles (same seeds as
qwen_lora_grounding.evaluate); per-frame correctness is saved, then the r16−r0 difference is bootstrapped
over EPISODES (frames within an episode are correlated).
  python train/qwen_lora_paired_eval.py --out runs/qwen_lora/paired_eval.json"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.qwen_lora_grounding import Grounder, build_index, desc, sample, split_eps  # noqa: E402


@torch.no_grad()
def run(model, idx, seen, swap_pool=None):
    model.eval(); rng = np.random.default_rng(0); rows = []
    sel = [idx[j] for j in np.random.default_rng(1).choice(len(idx), min(1500, len(idx)), replace=False)]
    for p, i, look in sel:
        override = None
        if swap_pool is not None:
            d0 = desc(p)[0]; cand = [l for d, l in swap_pool if d != d0]; override = cand[rng.integers(len(cand))]
        rgb, instr, boxes, y = sample(p, i, rng, override)
        rows.append({"ep": os.path.basename(p), "frame": int(i), "ok": int(model(rgb, instr, boxes).argmax().item() == y),
                     "look": bool(look), "unseen": desc(p)[0] not in seen, "n": len(boxes)})
    return rows


def load(rank, ck, device):
    m = Grounder("Qwen/Qwen3-VL-4B-Instruct", device, rank, 16, 32)
    [x.to(device) for x in (m.ln_c, m.ln_t, m.mlp, m.pc, m.pt)]
    c = torch.load(ck, map_location=device, weights_only=False)
    m.load_state_dict({**{f"vlm.{k}": v for k, v in c["lora"].items()}, **c["head"]}, strict=False)
    return m


def cluster_boot(a, b, key=None, B=10000, seed=0):
    eps = sorted(set(r["ep"] for r in a))
    da = {e: [r["ok"] for r in a if r["ep"] == e and (key is None or r[key])] for e in eps}
    db = {e: [r["ok"] for r in b if r["ep"] == e and (key is None or r[key])] for e in eps}
    eps = [e for e in eps if da[e]]
    diff = np.mean([x for e in eps for x in db[e]]) - np.mean([x for e in eps for x in da[e]])
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        s = rng.choice(len(eps), len(eps), replace=True)
        xa = [x for j in s for x in da[eps[j]]]; xb = [x for j in s for x in db[eps[j]]]
        bs.append(np.mean(xb) - np.mean(xa))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"base": float(np.mean([x for e in eps for x in da[e]])), "arm": float(np.mean([x for e in eps for x in db[e]])),
            "diff": float(diff), "ci": [float(lo), float(hi)], "p_le0": float(np.mean(np.array(bs) <= 0)), "episodes": len(eps)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    S = split_eps(); seen = set(desc(p)[0] for p in S["train"]); swap = [desc(p) for p in S["val"] + S["test"]]
    idx = {"val": build_index(S["val"], 9), "test": build_index(S["test"], 9)}
    res = {}
    for name, rank in (("r0", 0), ("r16", 16)):
        m = load(rank, f"runs/qwen_lora/{name}/best.pt", a.device)
        res[name] = {sp: run(m, idx[sp], seen) for sp in ("val", "test")}
        res[name]["test_langswap"] = run(m, idx["test"], seen, swap)
        del m; torch.cuda.empty_cache()
        print(name, {k: round(np.mean([r["ok"] for r in v]), 3) for k, v in res[name].items()}, flush=True)
    out = {}
    for sp in ("val", "test"):
        for key in (None, "look", "unseen"):
            out[f"{sp}/{key or 'all'}"] = cluster_boot(res["r0"][sp], res["r16"][sp], key)
    out["langswap_drop/r0"] = cluster_boot(res["r0"]["test_langswap"], res["r0"]["test"])
    out["langswap_drop/r16"] = cluster_boot(res["r16"]["test_langswap"], res["r16"]["test"])
    for k, v in out.items():
        print(f"{k:22s} base {v['base']:.3f} → arm {v['arm']:.3f}  Δ {v['diff']:+.3f}  95%CI [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  "
              f"P(Δ≤0)={v['p_le0']:.3f}  eps={v['episodes']}", flush=True)
    json.dump({"paired": out, "frames": res}, open(a.out, "w"))


if __name__ == "__main__":
    main()
