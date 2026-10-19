"""Fine-tuned attribute classifiers for the opening identity (1004 opening fix, step 3b). Same data / labels / eval as
opening_cls.py (train: npz_tr seeds 96000+; eval: npz_hi dev2 93000–99 + dev 91000–19), but the last `--blocks` blocks
of DINOv2 ViT-S/14 are fine-tuned jointly with a MODEL head and a COLOUR head (augment: scale/shift jitter, h-flip,
brightness). Run-time score per candidate = log p(model noun) + log p(colour word) [+ wq·Qwen colour, from the
opening_cls.py output cache]. Saves eval_out/opening_attr_ft.pt.
  python opening_ft.py --out eval_out/opening_ft.json --qwen-cache eval_out/opening_cls_v2.json"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "carla_uav_tracking")); sys.path.insert(0, HERE)
from opening_cls import COMBOS, frame_src, report                            # noqa: E402,F401
from opening_score import episodes                                           # noqa: E402
from opening_zoom import crop, describe, norm                                # noqa: E402
from scene.language_generator import LanguageGenerator                       # noqa: E402

TYPE = LanguageGenerator()._type_noun
BIG, R, TEST_SCALE = 2.4, 224, 1.9          # stored crop side = 2.4·box at 256 px; train scale jitter 1.4–2.4
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Net(torch.nn.Module):
    def __init__(self, n_m, n_c, blocks):
        super().__init__()
        import timm
        self.b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=R)
        for p in self.b.parameters():
            p.requires_grad = False
        for blk in self.b.blocks[-blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.b.norm.parameters():
            p.requires_grad = True
        self.hm = torch.nn.Linear(768, n_m); self.hc = torch.nn.Linear(768, n_c)

    def forward(self, x):
        t = self.b.forward_features(x)
        f = torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)
        return self.hm(f), self.hc(f)


def to_tensor(imgs, dev):
    x = torch.from_numpy(np.stack(imgs)).float().div(255).permute(0, 3, 1, 2)
    return ((x - MEAN) / STD).to(dev)


def build():
    lab = os.path.join(HERE, "eval_out", "labels_tr"); X, ym, yc, g = [], [], [], []
    for e in episodes("npz_tr", "open_dumpt_"):
        lp = os.path.join(lab, f"{e['seed']}.json")
        if not os.path.exists(lp):
            continue
        L = json.load(open(lp))
        for step, p in e["frames"].items():
            z = np.load(p, allow_pickle=True); rgb, boxes = frame_src(z)
            for i, gi in enumerate(z["gt_idx"]):
                d = {"bp": L["target_bp"], "color": L["target_color"]} if gi == -1 else \
                    (L["distractors"][gi] if gi < len(L["distractors"]) else None)
                if d and d["bp"].startswith("vehicle."):
                    X.append(np.asarray(crop(rgb, *boxes[i], BIG, 256), np.uint8))
                    ym.append(norm(TYPE(d["bp"]))); yc.append(norm(d["color"]) if d.get("color") else ""); g.append(e["seed"])
    return np.stack(X), np.array(ym), np.array(yc), np.array(g)


def augment(batch, rng):
    out = []
    for im in batch:
        s = rng.uniform(1.4, BIG) / BIG; side = int(256 * s); j = int(256 * 0.04)
        cx = 128 + rng.integers(-j, j + 1); cy = 128 + rng.integers(-j, j + 1)
        x0 = int(np.clip(cx - side // 2, 0, 256 - side)); y0 = int(np.clip(cy - side // 2, 0, 256 - side))
        c = im[y0:y0 + side, x0:x0 + side]
        c = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(c).resize((R, R), 3), np.float32)
        if rng.random() < 0.5:
            c = c[:, ::-1]
        c = np.clip(c * rng.uniform(0.85, 1.15), 0, 255)
        out.append(c.astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", type=int, default=4); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--qwen-cache", default=""); ap.add_argument("--full", action="store_true", help="train on all seeds")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(0); torch.manual_seed(0)
    X, ym, yc, g = build()
    cm = sorted(set(ym)); cc = sorted(set(yc) - {""}); im = {c: i for i, c in enumerate(cm)}; ic = {c: i for i, c in enumerate(cc)}
    Ym = np.array([im[c] for c in ym]); Yc = np.array([ic.get(c, -100) for c in yc])
    seeds = np.unique(g); val = set(rng.choice(seeds, len(seeds) // 5, replace=False)) if not a.full else set()
    tr = np.array([s not in val for s in g]); va = ~tr
    print(f"{len(X)} candidate crops, {len(seeds)} episodes | {len(cm)} models {len(cc)} colours | train {tr.sum()} val {va.sum()}", flush=True)
    net = Net(len(cm), len(cc), a.blocks).to(dev)
    bp = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("b.")]
    hp = list(net.hm.parameters()) + list(net.hc.parameters())
    opt = torch.optim.AdamW([{"params": bp, "lr": 2e-5}, {"params": hp, "lr": 1e-3}], weight_decay=0.05)
    idx = np.where(tr)[0]; B = 96; steps = a.epochs * (len(idx) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps, pct_start=0.1)
    k = 0
    for ep in range(a.epochs):
        net.train(); rng.shuffle(idx); tl = []
        for s in range(0, len(idx) - B + 1, B):
            bi = idx[s:s + B]; x = to_tensor(augment(X[bi], rng), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lm, lc = net(x)
            loss = Fn.cross_entropy(lm.float(), torch.tensor(Ym[bi], device=dev)) + \
                Fn.cross_entropy(lc.float(), torch.tensor(Yc[bi], device=dev), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item()); k += 1
        msg = f"epoch {ep + 1}/{a.epochs} loss {np.mean(tl):.3f}"
        if va.any():
            am, ac = evaluate_crops(net, X[va], Ym[va], Yc[va], dev); msg += f" | val acc model {am:.3f} colour {ac:.3f}"
        print(msg, flush=True)
    torch.save({"state": net.state_dict(), "cls_m": cm, "cls_c": cc, "blocks": a.blocks, "test_scale": TEST_SCALE},
               os.path.join(HERE, "eval_out", "opening_attr_ft.pt"))
    qc = {}
    if a.qwen_cache and os.path.exists(a.qwen_cache):
        for r in json.load(open(a.qwen_cache)):
            for s, t in r["ticks"].items():
                qc[(r["seed"], int(s))] = t.get("Q", t.get("color"))
    net.eval(); rows = []
    for e in episodes("npz_hi", "open_dumph_"):
        rec = {"seed": e["seed"], "split": e["split"], "ticks": {}}
        for step, p in sorted(e["frames"].items()):
            z = np.load(p, allow_pickle=True)
            if not z["is_target"].any():
                continue
            desc, color, model = describe(str(z["language"])); rgb, boxes = frame_src(z)
            lm, lc = predict(net, [np.asarray(crop(rgb, *b, TEST_SCALE, R), np.uint8) for b in boxes], dev)
            n = len(boxes); mi, ci = im.get(norm(model)), ic.get(norm(color)) if color else None
            M = lm[:, mi] if mi is not None else np.zeros(n); C = lc[:, ci] if ci is not None else np.zeros(n)
            Q = np.asarray(qc.get((e["seed"], step), np.zeros(n)), float)
            rec["ticks"][step] = {"tgt": int(np.argmax(z["is_target"])), "gt": z["gt_idx"].tolist(), "M": M.tolist(),
                                  "C": C.tolist(), "Q": Q.tolist(), "desc": desc,
                                  "model_known": mi is not None, "colour_known": ci is not None or not color}
        rows.append(rec)
    if a.out:
        json.dump(rows, open(a.out, "w"))
    report(rows)


@torch.no_grad()
def predict(net, crops, dev):
    x = to_tensor(crops, dev); xf = x.flip(3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lm, lc = net(torch.cat([x, xf]))
    n = len(crops)
    lm = Fn.log_softmax(lm.float(), 1); lc = Fn.log_softmax(lc.float(), 1)
    return ((lm[:n] + lm[n:]) / 2).cpu().numpy(), ((lc[:n] + lc[n:]) / 2).cpu().numpy()


@torch.no_grad()
def evaluate_crops(net, X, Ym, Yc, dev):
    net.eval(); pm, pc = [], []
    for s in range(0, len(X), 128):
        cr = [np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(
            im[int(128 - 128 * TEST_SCALE / BIG):int(128 + 128 * TEST_SCALE / BIG),
               int(128 - 128 * TEST_SCALE / BIG):int(128 + 128 * TEST_SCALE / BIG)]).resize((R, R), 3), np.uint8) for im in X[s:s + 128]]
        lm, lc = predict(net, cr, dev); pm.append(lm.argmax(1)); pc.append(lc.argmax(1))
    net.train(); pm = np.concatenate(pm); pc = np.concatenate(pc); ok = Yc >= 0
    return float((pm == Ym).mean()), float((pc[ok] == Yc[ok]).mean())


if __name__ == "__main__":
    main()
