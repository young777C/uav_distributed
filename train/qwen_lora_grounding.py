"""Stage 0 — LoRA on Qwen3-VL so the layer-24 representation carries WHICH (language-specified target among
look-alikes). OFFLINE, on recorded mvp_full_v5 episodes; supervised by GT identity (training-time privilege only).

Model: Qwen3-VL-4B (bf16), text-before-image prompt with the neutralized instruction (same as the deployed
encoder, acot_probe/backbones.py::Qwen3VLBackbone.encode). Forward runs only the first 24 decoder layers; the
readout = output of decoder layer 23 (0-based) == hidden_states[24] of the full model (pre-norm). LoRA on the
attention projections of layers [lora_lo, 23] (the layers feeding the readout). Candidate features = mean-pool
of the image-token grid inside each GT-projected candidate box (same boxes as the closed-loop harness).
Head: per-candidate score = MLP(LN(cand)) + bilinear(LN(cand), LN(text_mean)); CE over in-frame candidates.
rank 0 = frozen-Qwen linear/MLP probe (the baseline that isolates what LoRA adds).

Eval (val/test from train/splits/mvp_full_split.json, episodes on disk only): per-frame pick accuracy over
target-present frames with >=2 candidates; subsets: look-alike frames (a `similar` distractor in frame),
unseen target description (not in train); language-swap control (instruction replaced by another episode's
target description → should fall toward chance if the model uses language).
  python train/qwen_lora_grounding.py --rank 16 --out runs/qwen_lora/r16
  python train/qwen_lora_grounding.py --rank 0  --out runs/qwen_lora/r0
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "carla_uav_tracking")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from acot_probe.projection import project_point  # noqa: E402
from train.backbone_kv import neutralize_language  # noqa: E402

DATA = "/nvidia/hque/data/carla_data/mvp_full_v5"
W = H = 512; FOV = 90.0; VSZ = 4.0; BMIN, BMAX = 10.0, 140.0
READ_LAYER = 24                       # hidden_states index used by the deployed policy (config_v5 backbone.layer)


# ----------------------------------------------------------------------------------------------- data
def candidates(fh, i):
    """GT-projected candidate boxes for frame i → list of (fu0,fv0,fu1,fv1), label index or -1, similar flags."""
    cam = tuple(float(fh[f"state/cam_{k}"][i]) for k in ("x", "y", "z", "pitch", "yaw"))
    tgt = (float(fh["target/tx"][i]), float(fh["target/ty"][i]), float(fh["target/tz"][i]))
    dpos = fh["distractors/positions"][i]; sim = fh["distractors/similar"][:]
    out = []

    def mk(world, is_t, simf):
        pr = project_point(world, cam, W, H, FOV)
        if pr is None:
            return
        u, v, d = pr
        if not (0 <= u < W and 0 <= v < H):
            return
        f = W / (2 * math.tan(math.radians(FOV) / 2)); s = float(np.clip(f * VSZ / max(d, 0.1), BMIN, BMAX))
        out.append(((max(0, u - s / 2) / W, max(0, v - s / 2) / H, min(W, u + s / 2) / W, min(H, v + s / 2) / H), is_t, simf))

    mk(tgt, True, False)
    for d in range(dpos.shape[0]):
        if np.any(dpos[d, :3] != 0):
            mk(tuple(float(x) for x in dpos[d, :3]), False, bool(sim[d]))
    return out


def build_index(eps, stride):
    idx = []
    for p in eps:
        with h5py.File(p, "r") as fh:
            for i in range(0, fh["rgb"].shape[0], stride):
                c = candidates(fh, i)
                if len(c) >= 2 and any(x[1] for x in c):
                    idx.append((p, i, any(x[2] for x in c)))
    return idx


def split_eps():
    man = json.load(open(os.path.join(REPO, "train/splits/mvp_full_split.json")))
    on = {os.path.basename(p): p for p in glob.glob(f"{DATA}/episode_*.h5")}
    S = {k: [on[n] for n in man[k] if n in on] for k in ("train", "val", "test")}
    return S


def desc(p):
    with h5py.File(p, "r") as fh:
        return str(fh.attrs["target_desc"]), str(fh.attrs["language"])


# ----------------------------------------------------------------------------------------------- model
class Grounder(nn.Module):
    def __init__(self, model_id, device, rank, lora_lo, alpha):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.vlm = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        lm = self.vlm.model.language_model
        lm.layers = lm.layers[:READ_LAYER]                         # truncate: only layers 0..23 are needed
        for p in self.vlm.parameters():
            p.requires_grad_(False)
        self._cap = {}
        lm.layers[READ_LAYER - 1].register_forward_hook(lambda m, i, o: self._cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        self.rank = rank
        if rank > 0:
            from peft import LoraConfig, inject_adapter_in_model
            tm = [f"model.language_model.layers.{l}.self_attn.{p}" for l in range(lora_lo, READ_LAYER)
                  for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
            self.vlm = inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                                          target_modules=tm), self.vlm)
            for n, p in self.vlm.named_parameters():
                p.requires_grad_("lora_" in n)
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.vlm.enable_input_require_grads()
        C = self.vlm.config.text_config.hidden_size
        self.ln_c, self.ln_t = nn.LayerNorm(C), nn.LayerNorm(C)
        self.mlp = nn.Sequential(nn.Linear(C, 256), nn.GELU(), nn.Linear(256, 1))
        self.pc, self.pt = nn.Linear(C, 128, bias=False), nn.Linear(C, 128, bias=False)
        self.img_tok = self.vlm.config.image_token_id
        self.device = device

    def head_params(self):
        return [p for m in (self.ln_c, self.ln_t, self.mlp, self.pc, self.pt) for p in m.parameters()]

    def lora_params(self):
        return [p for n, p in self.vlm.named_parameters() if p.requires_grad]

    def forward(self, rgb, instruction, boxes):
        msgs = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image"}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[Image.fromarray(rgb)], return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(self.rank > 0 and self.training):
            self.vlm.model(**inp)                                  # base model only (no LM head); hook grabs layer 23
        h = self._cap["h"][0].float()                              # (S, C)
        ids = inp["input_ids"][0]
        img = (ids == self.img_tok).nonzero(as_tuple=True)[0]
        thw = inp["image_grid_thw"][0].tolist(); m = self.vlm.config.vision_config.spatial_merge_size
        gh, gw = thw[1] // m, thw[2] // m
        grid = h[img][: gh * gw].reshape(gh, gw, -1)
        txt = h[(ids != self.img_tok)].mean(0)
        feats = []
        for fu0, fv0, fu1, fv1 in boxes:                           # differentiable pool_box
            c0 = int(math.floor(fu0 * gw)); c1 = max(c0 + 1, int(math.ceil(fu1 * gw)))
            r0 = int(math.floor(fv0 * gh)); r1 = max(r0 + 1, int(math.ceil(fv1 * gh)))
            c0, c1 = min(max(c0, 0), gw - 1), min(max(c1, 1), gw); r0, r1 = min(max(r0, 0), gh - 1), min(max(r1, 1), gh)
            feats.append(grid[r0:r1, c0:c1].reshape(-1, grid.shape[-1]).mean(0))
        c = self.ln_c(torch.stack(feats)); t = self.ln_t(txt)
        return self.mlp(c).squeeze(-1) + (self.pc(c) * self.pt(t)[None]).sum(-1) / math.sqrt(128)


# ----------------------------------------------------------------------------------------------- loop
def sample(p, i, rng, lang_override=None):
    with h5py.File(p, "r") as fh:
        rgb = np.asarray(fh["rgb"][i]); c = candidates(fh, i); lang = str(fh.attrs["language"])
    order = rng.permutation(len(c)); c = [c[o] for o in order]
    y = next(k for k, x in enumerate(c) if x[1])
    instr = neutralize_language(lang_override or lang)
    return rgb, instr, [x[0] for x in c], y


@torch.no_grad()
def evaluate(model, idx, seen_desc, swap_pool=None, max_n=None, seed=0):
    model.eval(); rng = np.random.default_rng(seed); res = {"all": [], "lookalike": [], "unseen": [], "chance": []}
    sel = idx if max_n is None else [idx[j] for j in np.random.default_rng(1).choice(len(idx), min(max_n, len(idx)), replace=False)]
    for p, i, look in sel:
        override = None
        if swap_pool is not None:                                  # language-swap control
            d0 = desc(p)[0]; cand = [l for d, l in swap_pool if d != d0]; override = cand[rng.integers(len(cand))]
        rgb, instr, boxes, y = sample(p, i, rng, override)
        ok = int(model(rgb, instr, boxes).argmax().item() == y)
        res["all"].append(ok); res["chance"].append(1.0 / len(boxes))
        if look:
            res["lookalike"].append(ok)
        if desc(p)[0] not in seen_desc:
            res["unseen"].append(ok)
    model.train()
    return {k: (float(np.mean(v)) if v else None, len(v)) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-lo", type=int, default=16, help="first (0-based) layer with LoRA; through layer 23")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--eval-stride", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--eval-max", type=int, default=1500)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train frames (smoke test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "a")

    def say(*s):
        msg = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in s)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    torch.manual_seed(0); random.seed(0)
    S = split_eps()
    seen = set(desc(p)[0] for p in S["train"])
    swap_pool = [desc(p) for p in S["val"] + S["test"]]
    tr, va, te = build_index(S["train"], a.stride), build_index(S["val"], a.eval_stride), build_index(S["test"], a.eval_stride)
    if a.max_train:
        tr = [tr[j] for j in np.random.default_rng(0).choice(len(tr), min(a.max_train, len(tr)), replace=False)]
    say(f"rank={a.rank} lora_layers={a.lora_lo}-23 | train frames {len(tr)} ({len(S['train'])} eps) | val {len(va)} | test {len(te)}")
    model = Grounder("Qwen/Qwen3-VL-4B-Instruct", a.device, a.rank, a.lora_lo, a.alpha)
    model.ln_c.to(a.device); [m.to(a.device) for m in (model.ln_t, model.mlp, model.pc, model.pt)]
    groups = [{"params": model.head_params(), "lr": a.lr_head}]
    if a.rank > 0:
        groups.append({"params": model.lora_params(), "lr": a.lr_lora})
    say("trainable params: head", sum(p.numel() for p in model.head_params()),
        "lora", sum(p.numel() for p in model.lora_params()) if a.rank > 0 else 0)
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    total = a.epochs * len(tr) // a.accum; step = 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    rng = np.random.default_rng(0); best = -1
    for ep in range(a.epochs):
        order = rng.permutation(len(tr)); run_loss = run_acc = 0.0; n = 0; t0 = time.time()
        for k, j in enumerate(order):
            p, i, _ = tr[j]
            rgb, instr, boxes, y = sample(p, i, rng)
            logit = model(rgb, instr, boxes)
            loss = F.cross_entropy(logit[None], torch.tensor([y], device=a.device)) / a.accum
            loss.backward(); run_loss += loss.item() * a.accum; run_acc += float(logit.argmax().item() == y); n += 1
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if (k + 1) % 500 == 0:
                say(f"ep{ep} {k + 1}/{len(tr)} loss {run_loss / n:.3f} train-acc {run_acc / n:.3f} ({(time.time() - t0) / n:.2f}s/frame)")
                run_loss = run_acc = 0.0; n = 0
        v = evaluate(model, va, seen, max_n=a.eval_max)
        say(f"== ep{ep} VAL", json.dumps(v))
        if v["all"][0] > best:
            best = v["all"][0]
            torch.save({"head": {k2: v2 for k2, v2 in model.state_dict().items() if not k2.startswith("vlm.")},
                        "lora": {k2: v2 for k2, v2 in model.vlm.state_dict().items() if "lora_" in k2},
                        "args": vars(a), "val": v, "epoch": ep}, os.path.join(a.out, "best.pt"))
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=a.device, weights_only=False)
    model.load_state_dict({**{f"vlm.{k}": v for k, v in ck["lora"].items()}, **ck["head"]}, strict=False)
    final = {"best_epoch": ck["epoch"], "val": evaluate(model, va, seen, max_n=a.eval_max),
             "test": evaluate(model, te, seen, max_n=a.eval_max),
             "test_langswap": evaluate(model, te, seen, swap_pool=swap_pool, max_n=a.eval_max)}
    say("== FINAL", json.dumps(final))
    json.dump(final, open(os.path.join(a.out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
