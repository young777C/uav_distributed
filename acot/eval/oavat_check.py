"""1006 G9(a): the OA-VAT identity adaptation vs the original OA-VAT code, OPEN LOOP on recorded act messages.
The acot runtime (arm ext_oavat, protocol sim_proxy_v2@2) is replayed on a recording; every OavatHead.score call is
intercepted and re-computed independently:
  reference   the original offline recipe (extract_target_feature_dinov3.py 246–326, rectangle-mask fallback):
              rectangle-masked BGR image → DINOv3FeatureExtractor.extract_features(img, [box]) → L2-normalise; the
              horizontally flipped image / box likewise; mean → L2-normalise   vs   OavatHead.refs[-1]
  similarity  tracking/detection.py::match_candidates_with_reference(boxes, None, bgr, extractor, [prototype], cfg)
              (the original boxes-only matcher) vs the head's scores (its +1 on the chosen slot removed)
DINOv3 weights are loaded by the ORIGINAL DINOv3FeatureExtractor (torch.hub local, strict state-dict load).
  python -m acot.eval.oavat_check --record <act_record.pkl> --out <json> [--device cuda:0]"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

from acot import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-acts", type=int, default=400)
    a = ap.parse_args()
    for p in (ROOT, os.path.join(ROOT, "carla_uav_tracking")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import torch.nn.functional as F
    from acot.config.resolve import load_protocol, resolve_arm
    from acot.runtime import head_oavat as HO
    from acot.runtime.server import Runtime

    proto = load_protocol("sim_proxy_v2@2")
    proto = proto[0] if isinstance(proto, tuple) else proto
    arm = resolve_arm(proto, "oavat", {"method": "ext_oavat"})
    rt = Runtime(arm, {}, a.device)
    head = rt.policy.head
    _, match, _, _ = HO._oavat()
    ex = head.ex
    rec = {"ref": [], "sim": [], "n_calls": 0, "weights": arm["method"]["identity"]["oavat"]["checkpoint"],
           "model": arm["method"]["identity"]["oavat"]["model"]}

    orig_score = head.score

    def ref_original(rgb, rgb_hi, d):
        src = rgb_hi if rgb_hi is not None else rgb
        f = np.asarray(src).shape[1] / float(np.asarray(rgb).shape[1])
        img = np.ascontiguousarray(np.asarray(src, np.uint8)[..., ::-1])            # BGR, as cv2.imread
        H, W = img.shape[:2]
        x1, y1 = max(0, int(round((d.u - d.w / 2) * f))), max(0, int(round((d.v - d.h / 2) * f)))
        x2, y2 = min(W, int(round((d.u + d.w / 2) * f))), min(H, int(round((d.v + d.h / 2) * f)))
        mask = np.zeros((H, W), bool)
        mask[y1:y2, x1:x2] = True                                                   # rectangle-mask fallback
        masked = img.copy()
        masked[~mask] = 0
        f_base = F.normalize(ex.extract_features(masked, np.array([[x1, y1, x2, y2]]))[0].float().cpu(), dim=0)
        fm = mask[:, ::-1]
        ys, xs = np.where(fm)
        fb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]  # tight box of the flipped mask
        fl = np.ascontiguousarray(masked[:, ::-1])
        f_flip = F.normalize(ex.extract_features(fl, np.array([fb]))[0].float().cpu(), dim=0)
        return F.normalize(torch.stack([f_base, f_flip]).mean(0), dim=0)

    def wrapped(dets, rgb, rgb_hi, init_slot):
        fresh = head.fresh
        proto_before = head.store.get_feature_vector()
        z = orig_score(dets, rgb, rgb_hi, init_slot)
        rec["n_calls"] += 1
        if fresh and head.refs and init_slot is not None and 0 <= init_slot < len(dets):
            r0 = ref_original(rgb, rgb_hi, dets[init_slot])
            rec["ref"].append(float((r0 - head.refs[-1]).abs().max()))
        P = head.store.get_feature_vector() if fresh else proto_before
        if P is not None and len(dets) and float(np.max(z)) > -1e8 and not head.redetect:
            boxes = [(d.u - d.w / 2, d.v - d.h / 2, d.w, d.h) for d in dets]
            bgr = np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., ::-1])
            _, _, sims, _ = match(boxes, None, bgr, ex, [P], {})
            s = sims.detach().float().cpu().numpy()
            zz = np.asarray(z, float).copy()
            zz[int(np.argmax(zz))] -= 1.0
            m = zz > -1.0 + 1e-9                                                    # candidates the head scored
            rec["sim"].append(float(np.max(np.abs(zz[m] - s[m]))) if m.any() else 0.0)
        return z

    head.score = wrapped
    n = 0
    with open(a.record, "rb") as f:
        while n < a.max_acts:
            try:
                msg = pickle.load(f)
            except EOFError:
                break
            if msg.get("cmd") == "reset":
                rt.reset(seed=msg.get("seed"), repeat=0)
                continue
            rt.step(msg)
            n += 1
    out = {"n_acts": n, "n_head_calls": rec["n_calls"], "weights": rec["weights"], "model": rec["model"],
           "reference": {"n": len(rec["ref"]), "max_abs_diff": max(rec["ref"]) if rec["ref"] else None},
           "similarity": {"n": len(rec["sim"]), "max_abs_diff": max(rec["sim"]) if rec["sim"] else None}}
    out["ok"] = bool(rec["ref"] and rec["sim"] and out["reference"]["max_abs_diff"] < 1e-4
                     and out["similarity"]["max_abs_diff"] < 1e-4)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
