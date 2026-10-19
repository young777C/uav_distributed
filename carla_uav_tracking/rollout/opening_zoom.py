"""Zoom-and-verify opening identity (1004 opening fix). GT-free: uses the camera frame, the candidate boxes and the
language instruction only. Each candidate is cropped (side = scale·box), optionally spot-lit (outside of its own box
dimmed), resized, and the frozen Qwen3-VL-4B-Instruct the policy already loads is asked whether it is the described
vehicle; score = logit(yes) − logit(no). Shared by the offline benchmark (opening_score.py) and policy.py."""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

COLOR2 = {"dark red", "dark green", "light blue", "royal blue", "dark blue"}
COLOR1 = {"black", "white", "gray", "grey", "silver", "red", "orange", "yellow", "green", "blue", "brown", "beige",
          "purple", "gold"}


def describe(lang: str):
    """'Keep tracking the yellow Jeep Wrangler directly ahead, ...' → ('yellow Jeep Wrangler', 'yellow', 'Jeep Wrangler')."""
    from train.backbone_kv import neutralize_language
    n = neutralize_language(lang); i = n.lower().rfind(" the ")
    d = (n[i + 5:] if i >= 0 else n).rstrip(".").strip()
    w = d.split()
    if len(w) >= 2 and " ".join(w[:2]).lower() in COLOR2:
        return d, " ".join(w[:2]), " ".join(w[2:])
    if w and w[0].lower() in COLOR1:
        return d, w[0], " ".join(w[1:])
    return d, "", d


def crop(rgb, u, v, w, h, scale, R, spotlight=False):
    H, W = rgb.shape[:2]; side = max(32.0, max(w, h) * scale); half = side / 2
    if spotlight:
        rgb = rgb.astype(np.float32); m = np.full(rgb.shape[:2], 0.35, np.float32); b = max(w, h) * 0.6
        m[max(0, int(v - b)):int(v + b), max(0, int(u - b)):int(u + b)] = 1.0
        rgb = (rgb * m[..., None]).clip(0, 255).astype(np.uint8)
    x0, y0, x1, y1 = int(round(u - half)), int(round(v - half)), int(round(u + half)), int(round(v + half))
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    img = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge") if pad else rgb
    return Image.fromarray(np.ascontiguousarray(img[y0 + pad:y1 + pad, x0 + pad:x1 + pad])).resize((R, R), Image.BICUBIC)


class ZoomVerifier:
    """Wraps an already-loaded Qwen3-VL (model + processor) — no second copy of the weights."""

    def __init__(self, model, processor, device, scale=3.0, R=336, spotlight=True):
        self.model, self.proc, self.dev = model, processor, device
        self.scale, self.R, self.spotlight = float(scale), int(R), bool(spotlight)
        tok = processor.tokenizer
        self.yes = [tok.encode(t, add_special_tokens=False)[0] for t in ("Yes", "yes", " Yes", " yes")]
        self.no = [tok.encode(t, add_special_tokens=False)[0] for t in ("No", "no", " No", " no")]

    @torch.no_grad()
    def yesno(self, crops, q) -> np.ndarray:
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]] * len(crops)
        texts = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        side = self.proc.tokenizer.padding_side; self.proc.tokenizer.padding_side = "left"
        try:
            inp = self.proc(text=texts, images=crops, return_tensors="pt", padding=True).to(self.dev)
        finally:
            self.proc.tokenizer.padding_side = side
        lg = self.model(**inp).logits[:, -1].float()
        return (torch.logsumexp(lg[:, self.yes], 1) - torch.logsumexp(lg[:, self.no], 1)).cpu().numpy()

    @torch.no_grad()
    def scores(self, rgb, cands, language, rgb_hi=None) -> np.ndarray:
        desc, _, _ = describe(language)
        src, f = np.asarray(rgb, np.uint8), 1.0
        if rgb_hi is not None:                       # same camera at k× resolution → crop real pixels, not upsampling
            src = np.asarray(rgb_hi, np.uint8); f = src.shape[1] / float(np.asarray(rgb).shape[1])
        crops = [crop(src, c.u * f, c.v * f, c.w * f, c.h * f, self.scale, self.R, self.spotlight) for c in cands]
        q = (f"Is the brightly lit vehicle in the center of this image a {desc}? Answer yes or no." if self.spotlight
             else f"Is the vehicle in the center of this image a {desc}? Answer yes or no.")
        return self.yesno(crops, q)


def norm(s):
    s = " ".join(s.lower().split())
    return "gray" if s == "grey" else s


class Dino:
    """Frozen DINOv2 ViT-S/14 crop features (CLS + mean patch, L2-normalised)."""

    def __init__(self, dev):
        import timm
        self.m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(dev).eval()
        self.dev = dev
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def feats(self, imgs, flip=False):
        x = torch.from_numpy(np.stack([np.asarray(i, np.float32) / 255.0 for i in imgs])).permute(0, 3, 1, 2).to(self.dev)
        if flip:
            x = x.flip(3)
        x = (x - self.mean) / self.std
        t = self.m.forward_features(x)                                       # (B, 1+P, C)
        f = torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
        return torch.nn.functional.normalize(f, dim=1).cpu().numpy()


def hist(rgb, u, v, w, h, bins=6):
    """RGB histogram of the inner 70% of the box (car body, little road) — sqrt-normalised, flip-invariant."""
    a = np.asarray(crop(rgb, u, v, w * 0.7, h * 0.7, 1.0, 48), np.float32).reshape(-1, 3)
    q = np.clip((a / 256.0 * bins).astype(int), 0, bins - 1)
    hst = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(np.float32)
    return np.sqrt(hst / hst.sum())


class AttrScorer:
    """Opening identity from TRAINED attribute classifiers (opening_cls.py → eval_out/opening_attr_cls.pt): per candidate
    score = wm·log p(model noun of the instruction) + wc·log p(colour word) [+ wq·Qwen colour yes/no log-odds].
    Same interface as ZoomVerifier; GT-free at run time (crops of the k× opening frame + candidate boxes + language)."""

    def __init__(self, model, processor, device, ckpt="", wm=1.0, wc=1.0, wq=0.0, ft=0.0):
        eo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_out")
        ck = torch.load(ckpt or os.path.join(eo, "opening_attr_cls.pt"), map_location=device, weights_only=False)
        self.ft = float(ft); self.ftnet = None
        if self.ft:                      # fine-tuned DINOv2 (opening_ft.py), blended: (1-ft)·linear + ft·fine-tuned
            import timm
            fk = torch.load(os.path.join(eo, "opening_attr_ft.pt"), map_location=device, weights_only=False)
            b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=224)
            net = torch.nn.ModuleDict({"b": b, "hm": torch.nn.Linear(768, len(fk["cls_m"])),
                                       "hc": torch.nn.Linear(768, len(fk["cls_c"]))})
            net.load_state_dict(fk["state"]); self.ftnet = net.to(device).eval()
            self.ft_idx = ({c: i for i, c in enumerate(fk["cls_m"])}, {c: i for i, c in enumerate(fk["cls_c"])})
            self.ft_scale = float(fk["test_scale"])
        self.dino, self.dev = Dino(device), device
        self.scales = tuple(ck["scales"])
        self.heads = {}
        for k in ("m", "c"):
            sd = ck[f"W{k}"]; W = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).to(device)
            W.load_state_dict(sd); W.eval()
            self.heads[k] = (W, {c: i for i, c in enumerate(ck[f"cls_{k}"])}, float(ck[f"T{k}"]))
        self.wm, self.wc, self.wq = float(wm), float(wc), float(wq)
        self.zv = ZoomVerifier(model, processor, device, scale=2.0, spotlight=False) if self.wq else None

    @torch.no_grad()
    def scores(self, rgb, cands, language, rgb_hi=None) -> np.ndarray:
        _, color, model = describe(language)
        src, f = np.asarray(rgb, np.uint8), 1.0
        if rgb_hi is not None:
            src = np.asarray(rgb_hi, np.uint8); f = src.shape[1] / float(np.asarray(rgb).shape[1])
        boxes = [(c.u * f, c.v * f, c.w * f, c.h * f) for c in cands]
        F = sum(self.dino.feats([crop(src, *b, sc, 224) for b in boxes], fl) for sc in self.scales for fl in (False, True))
        Ft = torch.tensor(F / (2 * len(self.scales)), device=self.dev)
        Ht = torch.tensor(np.stack([hist(src, *b) for b in boxes]), device=self.dev)
        s = np.zeros(len(cands)); lin = {"m": Ft, "c": torch.cat([Ft, Ht], 1)}; fto = {}
        if self.ftnet is not None:
            x = torch.from_numpy(np.stack([np.asarray(crop(src, *b, self.ft_scale, 224), np.float32) / 255.0 for b in boxes]))
            x = ((x.permute(0, 3, 1, 2).to(self.dev) - self.dino.mean) / self.dino.std)
            t = self.ftnet["b"].forward_features(torch.cat([x, x.flip(3)]))
            f = torch.cat([t[:, 0], t[:, 1:].mean(1)], 1); n = len(boxes)
            for k in ("m", "c"):
                lp = torch.log_softmax(self.ftnet["h" + k](f).float(), 1); fto[k] = ((lp[:n] + lp[n:]) / 2).cpu().numpy()
        for k, w, word in (("m", self.wm, model), ("c", self.wc, color)):
            if not (w and word):
                continue
            W, idx, T = self.heads[k]; v = np.zeros(len(cands))
            if norm(word) in idx and self.ft < 1:
                v += (1 - self.ft) * torch.log_softmax(W(lin[k]) * 20.0 / T, 1)[:, idx[norm(word)]].cpu().numpy()
            fi = self.ft_idx[0 if k == "m" else 1] if self.ftnet is not None else {}
            if norm(word) in fi:
                v += self.ft * fto[k][:, fi[norm(word)]]
            s += w * v
        if self.zv is not None and color:
            s += self.wq * self.zv.yesno([crop(src, *b, 2.0, 336) for b in boxes],
                                         f"Is the vehicle in the center of this image {color}? Answer yes or no.")
        return s
