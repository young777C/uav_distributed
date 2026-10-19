"""Offline opening-identity benchmark (1004 opening-fix plan). Input: runs/opening_dump/npz/<host>_<ep>_t<step>.npz
(frame + candidate boxes + language + current grounding logits; is_target only for scoring) and hostmap.txt +
eval_out/open_dump_<k>.json (episode → seed alignment; skipped episodes never reach the policy).
Scorers (all GT-free; Qwen3-VL-4B-Instruct, the same frozen model the policy already loads):
  S0  current grounding head (argmax lang_logits)                                   [baseline]
  S1  zoom-and-verify: crop each candidate (side = scale·box), resize, ask "Is the vehicle in the center of this image
      a <desc>? Answer yes or no." → score = logit(yes) − logit(no)
  S2  multiple choice: all candidate crops in ONE prompt ("Image k:"), ask which one is the <desc> → digit logits
  S3  attribute-decomposed yes/no: colour question + model question, summed log-odds
  +agg  sum of a scorer's per-candidate scores over the opening ticks t=0,6,12 (association by simulator id offline —
        stands in for the tracker's ids, which are near-perfect over 1.2 s)
  python opening_score.py --out eval_out/opening_score.json [--max-eps N]"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "carla_uav_tracking"))
from train.backbone_kv import neutralize_language                                     # noqa: E402

DUMP = os.path.join(REPO, "runs", "opening_dump")
COLOR2 = {"dark red", "dark green", "light blue", "royal blue", "dark blue"}
COLOR1 = {"black", "white", "gray", "grey", "silver", "red", "orange", "yellow", "green", "blue", "brown", "beige",
          "purple", "gold"}


def describe(lang):
    n = neutralize_language(lang); i = n.lower().rfind(" the ")
    d = (n[i + 5:] if i >= 0 else n).rstrip(".").strip()
    w = d.split()
    if len(w) >= 2 and " ".join(w[:2]).lower() in COLOR2:
        return d, " ".join(w[:2]), " ".join(w[2:])
    if w and w[0].lower() in COLOR1:
        return d, w[0], " ".join(w[1:])
    return d, "", d


def episodes(sub="npz", jprefix="open_dump_"):
    """→ list of dicts {split, seed, frames: {step: npz}}; aligned to non-skipped seeds of each dump job."""
    host = {}
    hm = os.path.join(DUMP, f"hostmap_{sub}.txt")
    for line in open(hm if os.path.exists(hm) else os.path.join(DUMP, "hostmap.txt")):
        cid, name = line.split(); host[cid] = int(name.split("-")[1])
    by = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(DUMP, sub, "*.npz")):
        m = re.match(r"(\w+)_(\d+)_t(\d+)\.npz", os.path.basename(p))
        if m and m.group(1) in host:
            by[(host[m.group(1)], int(m.group(2)))][int(m.group(3))] = p
    out = []
    for k in sorted({k for k, _ in by}):
        js = os.path.join(HERE, "eval_out", f"{jprefix}{k}.json")
        if not os.path.exists(js):
            js += ".partial"
        if not os.path.exists(js):
            continue
        seeds = [e["episode_seed"] for e in json.load(open(js))["episodes"] if not e.get("skipped")]
        eps = sorted(ep for kk, ep in by if kk == k)
        for ep, seed in zip(eps, seeds):
            out.append({"split": "dev" if 91000 <= seed < 92000 else "dev2", "seed": seed, "frames": by[(k, ep)]})
    return out


def crop(rgb, u, v, w, h, scale, R, spotlight=False):
    H, W = rgb.shape[:2]; side = max(32.0, max(w, h) * scale); half = side / 2
    if spotlight:                     # dim everything outside the candidate's own box (no added colour cue)
        rgb = rgb.astype(np.float32); m = np.full(rgb.shape[:2], 0.35, np.float32); b = max(w, h) * 0.6
        m[max(0, int(v - b)):int(v + b), max(0, int(u - b)):int(u + b)] = 1.0
        rgb = (rgb * m[..., None]).clip(0, 255).astype(np.uint8)
    x0, y0, x1, y1 = int(round(u - half)), int(round(v - half)), int(round(u + half)), int(round(v + half))
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    img = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge") if pad else rgb
    c = img[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return Image.fromarray(c).resize((R, R), Image.BICUBIC)


class Qwen:
    def __init__(self, device):
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as M
        except Exception:
            from transformers import AutoModelForImageTextToText as M
        mid = "Qwen/Qwen3-VL-4B-Instruct"
        self.proc = AutoProcessor.from_pretrained(mid); self.proc.tokenizer.padding_side = "left"
        self.model = M.from_pretrained(mid, torch_dtype=torch.bfloat16).to(device).eval(); self.dev = device
        tok = self.proc.tokenizer
        self.yes = [tok.encode(t, add_special_tokens=False)[0] for t in ("Yes", "yes", " Yes", " yes")]
        self.no = [tok.encode(t, add_special_tokens=False)[0] for t in ("No", "no", " No", " no")]
        self.digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)]

    @torch.no_grad()
    def last_logits(self, contents, images):
        """contents: list of message-content lists (one per batch item); images: list of image lists."""
        texts = [self.proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
                 for c in contents]
        flat = [im for ims in images for im in ims]
        inp = self.proc(text=texts, images=flat, return_tensors="pt", padding=True).to(self.dev)
        return self.model(**inp).logits[:, -1].float()

    def yesno(self, crops, question):
        lg = self.last_logits([[{"type": "image"}, {"type": "text", "text": question}]] * len(crops), [[c] for c in crops])
        return (torch.logsumexp(lg[:, self.yes], 1) - torch.logsumexp(lg[:, self.no], 1)).cpu().numpy()

    def choice(self, crops, desc):
        content = []
        for i in range(len(crops)):
            content += [{"type": "text", "text": f"Image {i + 1}:"}, {"type": "image"}]
        content.append({"type": "text", "text": f"Each image is centered on one vehicle seen from a drone. Which image "
                                                f"shows the {desc}? Answer with the image number only."})
        lg = self.last_logits([content], [crops])[0]
        return lg[self.digits[:len(crops)]].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--max-eps", type=int, default=0)
    ap.add_argument("--R", type=int, default=336); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fast", action="store_true", help="only S0 / S1_2.0 / S1s_3.0 / S3 at t=0,6,12")
    ap.add_argument("--dump", default="npz", help="sub-directory of runs/opening_dump with the snapshots")
    ap.add_argument("--jprefix", default="open_dump_", help="eval_out json prefix of the dump jobs")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--report", nargs="*", help="only merge these shard jsons and print the report")
    a = ap.parse_args()
    if a.report is not None:
        rows = [r for p in a.report for r in json.load(open(p))]
        for r in rows:                                   # json turns int tick keys into strings
            r["ticks"] = {int(k): v for k, v in r["ticks"].items()}
        if a.out:
            json.dump(rows, open(a.out, "w"))
        report(rows); return
    eps = episodes(a.dump, a.jprefix)[a.shard::a.nshard]
    if a.max_eps:
        eps = eps[:a.max_eps]
    print(f"{len(eps)} episodes ({sum(e['split'] == 'dev2' for e in eps)} dev2, {sum(e['split'] == 'dev' for e in eps)} dev)", flush=True)
    q = Qwen(a.device)
    rows = []
    for n_ep, e in enumerate(eps):
        rec = {"seed": e["seed"], "split": e["split"], "ticks": {}}
        for step, p in sorted(e["frames"].items()):
            z = np.load(p, allow_pickle=True)
            if not z["is_target"].any():
                continue
            rgb = z["rgb"]; desc, color, model = describe(str(z["language"]))
            boxes = list(zip(z["u"], z["v"], z["w"], z["h"]))
            if "rgb_hi" in z.files and z["rgb_hi"].size > 0:      # crop REAL pixels from the k× opening camera
                f = z["rgb_hi"].shape[1] / float(rgb.shape[1]); rgb = z["rgb_hi"]
                boxes = [(u * f, vv * f, w * f, h * f) for u, vv, w, h in boxes]
            t = {"gt": z["gt_idx"].tolist(), "tgt": int(np.argmax(z["is_target"])), "S0": z["lang_logits"].tolist(),
                 "desc": desc}
            for sc in ((2.0,) if a.fast else (1.5, 2.0, 3.0)):
                crops = [crop(rgb, *b, sc, a.R) for b in boxes]
                t[f"S1_{sc}"] = q.yesno(crops, f"Is the vehicle in the center of this image a {desc}? Answer yes or no.").tolist()
            for sc in ((1.5, 3.0) if a.fast else (2.0, 3.0)):
                sp = [crop(rgb, *b, sc, a.R, spotlight=True) for b in boxes]
                t[f"S1s_{sc}"] = q.yesno(sp, f"Is the brightly lit vehicle in the center of this image a {desc}? "
                                              f"Answer yes or no.").tolist()
            crops = [crop(rgb, *b, 2.0, a.R) for b in boxes]
            if len(crops) <= 9 and not a.fast:
                t["S2"] = q.choice(crops, desc).tolist()
            if color and model:
                t["S3"] = (q.yesno(crops, f"Is the vehicle in the center of this image {color}? Answer yes or no.")
                           + q.yesno(crops, f"Is the vehicle in the center of this image a {model}? Answer yes or no.")).tolist()
            rec["ticks"][step] = t
        rows.append(rec)
        if (n_ep + 1) % 10 == 0:
            print(f"  scored {n_ep + 1}/{len(eps)}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))
    report(rows)


def report(rows):
    methods = ["S0", "S1_1.5", "S1_2.0", "S1_3.0", "S1s_1.5", "S1s_2.0", "S1s_3.0", "S2", "S3"]
    for split in ("dev2", "dev"):
        R = [r for r in rows if r["split"] == split and 0 in r["ticks"]]
        if not R:
            continue
        print(f"\n## {split}: {len(R)} episodes (opening tick t=0)")
        for m in methods:
            ok = [int(np.argmax(r["ticks"][0][m]) == r["ticks"][0]["tgt"]) for r in R if m in r["ticks"][0]]
            agg = []
            for r in R:                                  # +agg over t=0,6,12, association by simulator id
                tot = collections.defaultdict(float); last = None
                for s, t in sorted(r["ticks"].items()):
                    if m not in t:
                        continue
                    sc = np.asarray(t[m], float); sc = sc - sc.max()    # per-tick normalisation (log-odds offset)
                    for g, v in zip(t["gt"], sc):
                        tot[g] += v
                    last = t
                if last is not None and tot:
                    vis = {g for g in last["gt"]}
                    best = max((g for g in tot if g in vis), key=lambda g: tot[g])
                    agg.append(int(best == -1))
            print(f"  {m:7s} t0: {np.mean(ok):.1%} (n={len(ok)})   +agg(0/6/12): {np.mean(agg):.1%} (n={len(agg)})")


if __name__ == "__main__":
    main()
