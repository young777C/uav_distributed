#!/usr/bin/env python3
"""Automated training-side acceptance review of a CARLA data SAMPLE batch.

Run by the training-reviewer sub-agent (or standalone) to decide, WITHOUT a
human, whether a freshly-generated sample is good enough to train on. Emits a
machine verdict (verdict.json) + a human-readable table. This is the EXECUTABLE
half of acot_note/data-acceptance-criteria.md — the thresholds here ARE the
contract, so the doc and this script never drift.

    python -m iter.review_sample --dir /nvidia/hque/data/carla_data/mvp_review_r1 \
        --config train/config.yaml --out iter/reviews/r1.json

Exit 0 = APPROVED, 2 = CHANGES_REQUESTED, 1 = could not review (broken/empty).

Checks (see CRITERIA below for thresholds + rationale):
  integrity   HARD  every episode has the full schema, sane shapes, no NaN/Inf
  frames      HARD  each episode long enough to hold a 6s waypoint horizon
  leak_floor  SOFT  position-only MLP mis_follow — higher = geometry hides target
  covis       SOFT  frames with >=2 on-screen candidates (target + look-alike)
  offscreen   SOFT  target off-frame fraction — needs some, but bounded
  recovery    SOFT  >=3s loss segments that get re-acquired (predict-intercept)
  jitter      SOFT  camera yaw shake during occlusion/off-screen (deg/frame)
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from train.benchmark_leak_probe import build, train_position_classifier, _proj

# ---- the acceptance contract (single source of truth) -----------------------
# Each SOFT check maps a measured value to pass / waived / fail against these.
CRITERIA = {
    "leak_floor":  {"pass": 0.60, "waive": 0.40, "dir": "up",
                    "why": "position-only MLP mis_follow; low => geometry solves "
                           "target selection => language not load-bearing"},
    "covis":       {"pass": 0.40, "waive": 0.25, "dir": "up",
                    "why": "co-visible look-alikes force the model to disambiguate"},
    "offscreen":   {"lo": 0.05, "hi": 0.45, "dir": "band",
                    "why": "some loss needed for recovery skill; too much = untrackable"},
    "recovery":    {"pass": 0.20, "waive": 0.08, "dir": "up",
                    "why": "fraction of episodes with a >=3s loss that is re-acquired"},
    "jitter":      {"pass": 1.0, "waive": 2.5, "dir": "down",
                    "why": "camera yaw JERK (|d2yaw|, deg/frame^2) during occlusion/"
                           "off-screen; jerk (not rate) so a smooth fast pan is not "
                           "flagged, only oscillating shake is"},
}
MIN_EPISODES = 8            # a sample smaller than this can't be judged
OFFSCREEN_LOSS_S = 3.0      # a "loss segment" worth counting toward recovery


def _grade(name, val):
    """-> (status, ok_for_gate). status in pass/waived/fail; None val -> fail."""
    c = CRITERIA[name]
    if val is None:
        return "fail", False
    if c["dir"] == "up":
        if val >= c["pass"]:
            return "pass", True
        return ("waived", True) if val >= c["waive"] else ("fail", False)
    if c["dir"] == "down":
        if val <= c["pass"]:
            return "pass", True
        return ("waived", True) if val <= c["waive"] else ("fail", False)
    # band
    return ("pass", True) if c["lo"] <= val <= c["hi"] else ("fail", False)


def _cfg_for_dir(cfg_path, sample_dir):
    cfg = yaml.safe_load(open(cfg_path))
    cfg = copy.deepcopy(cfg)
    cfg["data"]["episodes_glob"] = str(Path(sample_dir) / "episode_*.h5")
    cfg["data"]["split_manifest"] = None          # naive slice within the sample
    cfg["data"]["val_frac"] = 0.35
    return cfg


def integrity_and_scene(cfg, sample_dir):
    """Per-episode schema/NaN checks + scene stats (covis/offscreen/recovery/jitter)."""
    img = cfg["image"]; W, H, fov = img["width"], img["height"], img["fov_deg"]
    fps = cfg.get("fps", 10)
    horizon = int(cfg["waypoint"].get("max_horizon_s", 6) * fps)
    req_state = ["cam_x", "cam_y", "cam_z", "cam_pitch", "cam_yaw", "occ_raycast"]
    eps = sorted(glob.glob(str(Path(sample_dir) / "episode_*.h5")))
    problems, per_ep = [], []
    covis_num = covis_den = off_num = off_den = 0
    jit_vals, rec_eps = [], 0
    for ep in eps:
        name = Path(ep).name
        try:
            with h5py.File(ep, "r") as f:
                if "rgb" not in f:
                    problems.append(f"{name}: no rgb"); continue
                T = f["rgb"].shape[0]
                if tuple(f["rgb"].shape[1:]) != (H, W, 3):
                    problems.append(f"{name}: rgb shape {f['rgb'].shape[1:]} != {(H, W, 3)}")
                if T < horizon + 10:
                    problems.append(f"{name}: only {T} frames (< horizon {horizon}+10)")
                for g in ("state", "action", "target"):
                    if g not in f:
                        problems.append(f"{name}: missing group {g}")
                for k in req_state:
                    if f"state/{k}" not in f:
                        problems.append(f"{name}: missing state/{k}")
                if "distractors/positions" not in f:
                    problems.append(f"{name}: no distractors/positions (no look-alikes)")
                # NaN/Inf sweep on numeric fields
                for grp in ("state", "action", "target"):
                    if grp in f:
                        for k in f[grp]:
                            a = f[f"{grp}/{k}"][:]
                            if a.dtype.kind == "f" and not np.isfinite(a).all():
                                problems.append(f"{name}: NaN/Inf in {grp}/{k}")

                tgt = np.stack([f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]], -1)
                cam = np.stack([f["state/cam_x"][:], f["state/cam_y"][:], f["state/cam_z"][:],
                                f["state/cam_pitch"][:], f["state/cam_yaw"][:]], -1)
                distr = f["distractors/positions"][:] if "distractors/positions" in f else None
                occ = f["state/occ_raycast"][:] if "state/occ_raycast" in f else np.zeros(T)
                cyaw = f["state/cam_yaw"][:]
        except Exception as e:
            problems.append(f"{name}: unreadable ({e})"); continue

        in_frame = np.zeros(T, dtype=bool)
        n_cand = np.zeros(T, dtype=int)
        for i in range(T):
            t = _proj(tgt[i], cam[i], W, H, fov)
            in_frame[i] = t is not None
            k = 1 if t is not None else 0
            if distr is not None:
                for d in range(distr.shape[1]):
                    if _proj(distr[i, d, :3], cam[i], W, H, fov) is not None:
                        k += 1
            n_cand[i] = k
        covis_num += int(((n_cand >= 2) & in_frame).sum())
        covis_den += int(in_frame.sum())
        off_num += int((~in_frame).sum()); off_den += T

        # recovery: a >=OFFSCREEN_LOSS_S off-screen run followed by re-acquisition
        min_run = int(OFFSCREEN_LOSS_S * fps)
        i, recovered = 0, False
        while i < T:
            if not in_frame[i]:
                j = i
                while j < T and not in_frame[j]:
                    j += 1
                if (j - i) >= min_run and j < T:      # re-acquired after a long loss
                    recovered = True
                i = j
            else:
                i += 1
        rec_eps += int(recovered)

        # jitter: yaw JERK (|d2 cam_yaw|) during occluded OR off-screen frames.
        # Jerk (rate-of-change of rate), NOT rate: a smooth fast re-acquisition pan
        # has high |dyaw| but low jerk; only oscillating shake has high jerk.
        rate = ((np.diff(cyaw) + 180) % 360) - 180          # deg/frame  (len T-1)
        jerk = np.abs(np.diff(rate))                         # deg/frame^2 (len T-2)
        bad = (~in_frame[2:]) | (occ[2:] > 0.5)             # align to jerk index
        if bad.any():
            jit_vals.append(float(jerk[bad].mean()))
        per_ep.append({"ep": name, "T": int(T),
                       "covis_frac": round(float(((n_cand >= 2) & in_frame).mean()), 3),
                       "off_frac": round(float((~in_frame).mean()), 3),
                       "recovered": recovered})

    scene = {
        "n_episodes": len(eps),
        "covis": (covis_num / covis_den) if covis_den else None,
        "offscreen": (off_num / off_den) if off_den else None,
        "recovery": (rec_eps / len(eps)) if eps else None,
        "jitter": float(np.mean(jit_vals)) if jit_vals else 0.0,
    }
    return problems, scene, per_ep


def leak_floor(cfg, split_from="train", split_eval="val"):
    tr = build(cfg, split_from); va = build(cfg, split_eval)
    if not tr or not va:
        return None, {"note": "too few hard (>=2 candidate) samples to fit probe"}
    floor = train_position_classifier(tr, va) / 100.0    # -> mis_follow fraction
    avg_k = float(np.mean([len(f) for f, _ in va]))
    return floor, {"hard_train": len(tr), "hard_val": len(va), "avg_candidates": round(avg_k, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="sample dir (host path)")
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--out", default="", help="write verdict json here")
    ap.add_argument("--round", type=int, default=0)
    a = ap.parse_args()

    cfg = _cfg_for_dir(a.config, a.dir)
    n_have = len(glob.glob(str(Path(a.dir) / "episode_*.h5")))
    if n_have < MIN_EPISODES:
        v = {"verdict": "CANNOT_REVIEW", "round": a.round,
             "reason": f"only {n_have} episodes (< {MIN_EPISODES}); generate more"}
        print(json.dumps(v, ensure_ascii=False, indent=2))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(v, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    problems, scene, per_ep = integrity_and_scene(cfg, a.dir)
    floor, leak_meta = leak_floor(cfg)

    measured = {"leak_floor": floor, "covis": scene["covis"],
                "offscreen": scene["offscreen"], "recovery": scene["recovery"],
                "jitter": scene["jitter"]}
    checks, changes = {}, []
    for name, val in measured.items():
        status, ok = _grade(name, val)
        checks[name] = {"value": (round(val, 4) if isinstance(val, float) else val),
                        "status": status, "why": CRITERIA[name]["why"]}
        if not ok:
            changes.append({"check": name, "value": val, "status": status,
                            "target": CRITERIA[name], "why": CRITERIA[name]["why"]})

    hard_ok = len(problems) == 0
    if not hard_ok:
        verdict = "CHANGES_REQUESTED"
    elif changes:
        verdict = "CHANGES_REQUESTED"
    else:
        verdict = "APPROVED"

    out = {
        "verdict": verdict, "round": a.round, "dir": a.dir,
        "n_episodes": scene["n_episodes"],
        "integrity": {"ok": hard_ok, "problems": problems[:40]},
        "checks": checks,
        "leak_meta": leak_meta,
        "changes_requested": changes,
        "per_episode": per_ep,
    }
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    raise SystemExit(0 if verdict == "APPROVED" else 2)


if __name__ == "__main__":
    main()
