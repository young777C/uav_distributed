"""Vehicle MODEL + COLOUR classifiers for the opening identity (1004 opening fix, step 3). Qwen3-VL zoom-verify resolves
colour only partly and CARLA car models poorly (~41% in same-colour scenes) → train both attributes on simulator crops.
Train: hi-res opening snapshots on SEPARATE seeds (runs/opening_dump/npz_tr, seeds 96000+, labels from env DUMP_LABELS);
features = frozen DINOv2 ViT-S/14 (CLS + mean patch) on crops of the k× camera (+ horizontal flips), plus an RGB
histogram of the box interior for the colour head; linear softmax heads over the language's type nouns / 18 colour
words, temperature-calibrated on held-out seeds.
Eval (dev2 93000–99, dev 91000–19 hi-res snapshots): per candidate
  M = log p(type noun named in the instruction), C = log p(colour word), Q = Qwen3-VL yes/no log-odds on the colour word
  combined = sums; accuracy of the argmax at t=0 and summed over t=0/6/12.
GT labels are used only to TRAIN the classifiers and to SCORE.
  python opening_cls.py --out eval_out/opening_cls.json [--reuse-feats] [--qwen-cache eval_out/opening_cls_q.json]"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "carla_uav_tracking")); sys.path.insert(0, HERE)
from opening_score import episodes, Qwen                                      # noqa: E402
from opening_zoom import Dino, crop, describe, hist, norm                   # noqa: E402
from scene.language_generator import LanguageGenerator                       # noqa: E402

TYPE = LanguageGenerator()._type_noun
SCALES = (1.5, 2.0)
FEATS = os.path.join(HERE, "eval_out", "opening_feats_tr.npz")
CKPT = os.path.join(HERE, "eval_out", "opening_attr_cls.pt")


def frame_src(z):
    rgb = z["rgb"]; f = 1.0
    if "rgb_hi" in z.files and z["rgb_hi"].size > 0:
        f = z["rgb_hi"].shape[1] / float(rgb.shape[1]); rgb = z["rgb_hi"]
    return rgb, [(u * f, v * f, w * f, h * f) for u, v, w, h in zip(z["u"], z["v"], z["w"], z["h"])]


def build_train(dino):
    lab_dir = os.path.join(HERE, "eval_out", "labels_tr")
    D, H, ym, yc, g = [], [], [], [], []
    for e in episodes("npz_tr", "open_dumpt_"):
        lp = os.path.join(lab_dir, f"{e['seed']}.json")
        if not os.path.exists(lp):
            continue
        L = json.load(open(lp))
        for step, p in e["frames"].items():
            z = np.load(p, allow_pickle=True)
            rgb, boxes = frame_src(z)
            keep, mn, cn = [], [], []
            for i, gi in enumerate(z["gt_idx"]):
                d = {"bp": L["target_bp"], "color": L["target_color"]} if gi == -1 else \
                    (L["distractors"][gi] if gi < len(L["distractors"]) else None)
                if d and d["bp"].startswith("vehicle."):
                    keep.append(i); mn.append(norm(TYPE(d["bp"]))); cn.append(norm(d["color"]) if d.get("color") else "")
            if not keep:
                continue
            hs = np.stack([hist(rgb, *boxes[i]) for i in keep])
            for sc in SCALES:
                crops = [crop(rgb, *boxes[i], sc, 224) for i in keep]
                for flip in (False, True):
                    D.append(dino.feats(crops, flip)); H.append(hs); ym += mn; yc += cn; g += [e["seed"]] * len(keep)
    return np.concatenate(D), np.concatenate(H), np.array(ym), np.array(yc), np.array(g)


def train(X, y, g, dev, name):
    ok = np.array([bool(s) for s in y]); X, y, g = X[ok], y[ok], g[ok]
    classes = sorted(set(y)); ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[c] for c in y])
    seeds = np.unique(g); rng = np.random.default_rng(0)
    val = set(rng.choice(seeds, max(1, len(seeds) // 5), replace=False))
    tr = np.array([s not in val for s in g])
    Xt = torch.tensor(X, device=dev); yt = torch.tensor(yi, device=dev); trt = torch.tensor(tr, device=dev)

    def fit(mask, epochs):
        torch.manual_seed(0)
        W = torch.nn.Linear(X.shape[1], len(classes)).to(dev)
        opt = torch.optim.AdamW(W.parameters(), lr=3e-3, weight_decay=1e-2)
        for _ in range(epochs):
            opt.zero_grad(); torch.nn.functional.cross_entropy(W(Xt[mask]) * 20.0, yt[mask]).backward(); opt.step()
        return W

    W = fit(trt, 400)
    with torch.no_grad():
        lv = W(Xt[~trt]) * 20.0
        acc_tr = (W(Xt[trt]).argmax(1) == yt[trt]).float().mean().item()
        acc_va = (lv.argmax(1) == yt[~trt]).float().mean().item()
        Ts = [0.25, 0.5, 1, 2, 3, 5, 8, 12]
        nll = [torch.nn.functional.cross_entropy(lv / T, yt[~trt]).item() for T in Ts]
    T = Ts[int(np.argmin(nll))]
    print(f"{name} classifier: {len(classes)} classes, {int(tr.sum())} train / {int((~tr).sum())} val crops "
          f"({len(seeds) - len(val)}/{len(val)} seeds) | acc train {acc_tr:.3f} val {acc_va:.3f} | temperature {T}", flush=True)
    return fit(torch.ones_like(trt), 400), classes, T, acc_va


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reuse-feats", action="store_true"); ap.add_argument("--qwen-cache", default="")
    ap.add_argument("--no-qwen", action="store_true")
    a = ap.parse_args(); dev = a.device
    dino = Dino(dev)
    if a.reuse_feats and os.path.exists(FEATS):
        f = np.load(FEATS); D, H, ym, yc, g = f["D"], f["H"], f["ym"], f["yc"], f["g"]
    else:
        D, H, ym, yc, g = build_train(dino)
        np.savez(FEATS, D=D, H=H, ym=ym, yc=yc, g=g)
    print(f"train crops {len(ym)} from {len(set(g))} episodes | {len(set(ym))} models, {len(set(yc) - {''})} colours", flush=True)
    Wm, cls_m, Tm, va_m = train(D, ym, g, dev, "model ")
    Wc, cls_c, Tc, va_c = train(np.concatenate([D, H], 1), yc, g, dev, "colour")
    torch.save({"Wm": Wm.state_dict(), "cls_m": cls_m, "Tm": Tm, "Wc": Wc.state_dict(), "cls_c": cls_c, "Tc": Tc,
                "scales": SCALES, "val_acc": {"model": va_m, "colour": va_c}}, CKPT)
    im, ic = {c: i for i, c in enumerate(cls_m)}, {c: i for i, c in enumerate(cls_c)}
    qc = {}
    if a.qwen_cache and os.path.exists(a.qwen_cache):
        for r in json.load(open(a.qwen_cache)):
            for s, t in r["ticks"].items():
                qc[(r["seed"], int(s))] = t.get("color", t.get("Q"))
    q = None if a.no_qwen else Qwen(dev)
    rows = []
    for e in episodes("npz_hi", "open_dumph_"):
        rec = {"seed": e["seed"], "split": e["split"], "ticks": {}}
        for step, p in sorted(e["frames"].items()):
            z = np.load(p, allow_pickle=True)
            if not z["is_target"].any():
                continue
            desc, color, model = describe(str(z["language"]))
            cm, cc = im.get(norm(model)), ic.get(norm(color)) if color else None
            rgb, boxes = frame_src(z)
            F = sum(dino.feats([crop(rgb, *b, sc, 224) for b in boxes], fl) for sc in SCALES for fl in (False, True)) / 4
            Hs = np.stack([hist(rgb, *b) for b in boxes])
            with torch.no_grad():
                Ft = torch.tensor(F, device=dev)
                lm = torch.log_softmax(Wm(Ft) * 20.0 / Tm, 1).cpu().numpy()
                lc = torch.log_softmax(Wc(torch.cat([Ft, torch.tensor(Hs, device=dev)], 1)) * 20.0 / Tc, 1).cpu().numpy()
            n = len(boxes)
            M = lm[:, cm] if cm is not None else np.zeros(n)
            C = lc[:, cc] if cc is not None else np.zeros(n)
            if (e["seed"], step) in qc:
                Q = np.asarray(qc[(e["seed"], step)], float)
            elif q is not None and color:
                Q = q.yesno([crop(rgb, *b, 2.0, 336) for b in boxes],
                            f"Is the vehicle in the center of this image {color}? Answer yes or no.")
            else:
                Q = np.zeros(n)
            rec["ticks"][step] = {"tgt": int(np.argmax(z["is_target"])), "gt": z["gt_idx"].tolist(), "M": M.tolist(),
                                  "C": C.tolist(), "Q": np.asarray(Q).tolist(), "desc": desc,
                                  "model_known": cm is not None, "colour_known": cc is not None or not color}
        rows.append(rec)
    if a.out:
        json.dump(rows, open(a.out, "w"))
    report(rows)


COMBOS = (("M  model", lambda t: t["M"]), ("C  colour(cls)", lambda t: t["C"]), ("Q  colour(Qwen)", lambda t: t["Q"]),
          ("C+M", lambda t: t["C"] + t["M"]), ("Q+M", lambda t: t["Q"] + t["M"]),
          ("C+M+0.5Q", lambda t: t["C"] + t["M"] + 0.5 * t["Q"]), ("C+M+Q", lambda t: t["C"] + t["M"] + t["Q"]))


def report(rows):
    for split in ("dev2", "dev", "all"):
        R = [r for r in rows if (split == "all" or r["split"] == split) and 0 in r["ticks"]]
        if not R:
            continue
        print(f"\n## {split}: {len(R)} episodes (t=0) — model known {sum(r['ticks'][0]['model_known'] for r in R)}, "
              f"colour known {sum(r['ticks'][0]['colour_known'] for r in R)}")
        for name, fn in COMBOS:
            def sc(t):
                return np.asarray(fn({k: np.asarray(t[k], float) for k in ("M", "C", "Q")}), float)
            acc = np.mean([int(np.argmax(sc(r["ticks"][0])) == r["ticks"][0]["tgt"]) for r in R])
            agg = []
            for r in R:
                tot = collections.defaultdict(float); last = None
                for s, t in sorted(r["ticks"].items()):
                    v = sc(t); v = v - v.max()
                    for gg, x in zip(t["gt"], v):
                        tot[gg] += x
                    last = t
                vis = set(last["gt"]); agg.append(int(max((gg for gg in tot if gg in vis), key=lambda k: tot[k]) == -1))
            print(f"  {name:16s} t0 {acc:.1%}   +agg(0/6/12) {np.mean(agg):.1%}")


if __name__ == "__main__":
    main()
