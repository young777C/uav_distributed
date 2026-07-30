"""Audit: does the episode-level language's stated BEARING (front-left / ahead /
front-right) match the target's ACTUAL bearing at each sampled frame?

Quantifies the "stale spatial clause" noise the user flagged: the prompt is fixed
per episode, but samples are cut at arbitrary frames where the target may have
drifted to a different bearing. Read-only (projection + h5 metadata); no GPU.

    python -m train.audit_bearing --config train/config.yaml
"""
from __future__ import annotations
import argparse, math, re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml

from acot_probe.projection import project_point
from train.dataset import _select_episodes  # reuse the exact train/val episode set


def stated_bearing(lang: str):
    s = lang.lower()
    # compound forms first (they contain 'front'/'left'/'right')
    if re.search(r"front[- ]?left|to the left|on the left|left[- ]side", s):
        return "left"
    if re.search(r"front[- ]?right|to the right|on the right|right[- ]side", s):
        return "right"
    if re.search(r"directly ahead|straight ahead|\bahead\b|in front|front[- ]?center|directly in front", s):
        return "ahead"
    return None


def actual_bearing(u: float, W: int):
    if u < W / 3.0:
        return "left"
    if u < 2.0 * W / 3.0:
        return "ahead"
    return "right"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    img = cfg["image"]; W, H, fov = img["width"], img["height"], img["fov_deg"]
    stride = int(cfg["data"].get("frame_stride", 3))
    cap = int(cfg["data"].get("max_frames_per_episode", 200))
    horizon = int(cfg["waypoint"].get("max_horizon_s", 6) * cfg.get("fps", 10))

    eps = list(_select_episodes(cfg, "train")) + list(_select_episodes(cfg, "val"))
    print(f"[audit] episodes={len(eps)}  W={W} fov={fov}  bearing=image-thirds (left/ahead/right)")

    n_samp = 0; n_offscreen = 0; n_no_stated = 0
    conf = Counter()                     # (stated, actual) -> count  (on-screen, stated present)
    by_third = defaultdict(lambda: [0, 0])   # time-third -> [mismatch, total]
    per_strategy = defaultdict(lambda: [0, 0])
    ep_stated = Counter()

    for ep in eps:
        with h5py.File(ep, "r") as f:
            lang = f.attrs.get("language", "")
            lang = lang.decode() if isinstance(lang, bytes) else str(lang)
            strat = str(f.attrs.get("strategy", ""))
            sb = stated_bearing(lang)
            ep_stated[sb] += 1
            n = f["rgb"].shape[0]
            tx = f["target/tx"][:]; ty = f["target/ty"][:]; tz = f["target/tz"][:]
            cam = np.stack([f["state/cam_x"][:], f["state/cam_y"][:], f["state/cam_z"][:],
                            f["state/cam_pitch"][:], f["state/cam_yaw"][:]], -1)
        valid = max(0, n - horizon)
        idxs = list(range(0, valid, stride))[:cap]
        for i in idxs:
            n_samp += 1
            pr = project_point((float(tx[i]), float(ty[i]), float(tz[i])), tuple(cam[i]), W, H, fov)
            if pr is None or not (0 <= pr[0] < W and 0 <= pr[1] < H):
                n_offscreen += 1
                continue
            if sb is None:
                n_no_stated += 1
                continue
            ab = actual_bearing(pr[0], W)
            conf[(sb, ab)] += 1
            mism = int(ab != sb)
            third = min(2, int(3 * i / max(valid, 1)))
            by_third[third][0] += mism; by_third[third][1] += 1
            per_strategy[strat][0] += mism; per_strategy[strat][1] += 1

    on_screen_stated = sum(conf.values())
    mismatch = sum(v for (s, a), v in conf.items() if s != a)
    opposite = sum(v for (s, a), v in conf.items() if {s, a} == {"left", "right"})
    print(f"\n[样本] 总采样帧={n_samp}  目标出画={n_offscreen} ({100*n_offscreen/max(n_samp,1):.1f}%)  "
          f"prompt无方位词={n_no_stated}  可比对={on_screen_stated}")
    print(f"[episode 声明方位分布] {dict(ep_stated)}")
    print(f"\n=== 方位不一致率(可比对样本中)===")
    print(f"  声明 ≠ 实际:  {mismatch}/{on_screen_stated} = {100*mismatch/max(on_screen_stated,1):.1f}%")
    print(f"  其中左右相反(严重): {opposite}/{on_screen_stated} = {100*opposite/max(on_screen_stated,1):.1f}%")
    print(f"\n=== 混淆矩阵 (声明 row -> 实际 col) ===")
    cats = ["left", "ahead", "right"]
    print("           " + "".join(f"{c:>8}" for c in cats))
    for s in cats:
        row = conf.get((s, "left"), 0), conf.get((s, "ahead"), 0), conf.get((s, "right"), 0)
        print(f"  声明 {s:5} " + "".join(f"{v:8d}" for v in row))
    print(f"\n=== 不一致率 vs 视频切入时刻(验证'越往后越陈旧')===")
    for t, name in [(0, "前 1/3"), (1, "中 1/3"), (2, "后 1/3")]:
        mm, tot = by_third[t]
        print(f"  {name}: {100*mm/max(tot,1):.1f}%  (n={tot})")
    print(f"\n=== 按 strategy 分档 ===")
    for st, (mm, tot) in sorted(per_strategy.items()):
        print(f"  {st:24} 不一致 {100*mm/max(tot,1):.1f}%  (n={tot})")


if __name__ == "__main__":
    main()
