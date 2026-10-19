"""Automated QC + split manifest for the v6 dataset (acot_note/1003_data-regen-v6-spec.md §QC).

  python3 carla_uav_tracking/scripts/qc_v6.py <dataset_dir> [--split-out train/splits/v6_fov90_split.json] [--viz N]
  (also imported by generate_batch.py for the per-episode gate before the atomic rename)

Per-episode gates (FAIL → episode excluded from the manifest):
  G1 attrs: fov/rgb_resolution/label_version/git_commit/subset/config_snapshot present; fov == 90
  G2 rgb shape (T,512,512,3), T >= 300, no all-black frame among 16 sampled frames
  G3 projection ↔ instance alignment: for actor-frames with vis_px >= 50, the instance (modal) box lies
     inside the projected 3D (amodal) box (≥ 90 % of its area, 3 px margin) in ≥ 95 % of cases
     → catches any FOV / pose mismatch between rendering and projection automatically
  G4 association: clearly visible actors (amodal box ≥ 12 px, occ_raycast < 0.2, center in frame) get an
     instance blob in ≥ 85 % of cases
  G5 postprocess done: actors/visible, actors/tracklet_id, events/target_reentry present
  G6 view: frames with the camera pressed against geometry (block_frac > 0.8 AND optical-axis hit < 8 m) ≤ 1 % of T
  G7 association stability: target instance-id switches between consecutive visible frames ≤ 3
  G9 motion: the target moves (speed > 1 m/s) in ≥ 50 % of frames
  G8 subset gates: all T ≥ 600 (60 s); reacq ≥ 1 target re-entry after ≥ 3 s with a look-alike visible;
     occ ≥ 20 in-frame structurally-occluded target frames or ≥ 1 occlusion-caused re-entry
Dataset-level report: per subset × town counts, target frame classes, re-entry events, look-alike
co-visibility, box sizes. Split: train = training towns minus val (every 10th episode id per town),
test = held-out town (attrs test_town)."""
import argparse
import collections
import glob
import json
import os

import h5py
import numpy as np

REQ_ATTRS = ("fov", "K", "rgb_resolution", "label_version", "git_commit", "subset", "config_snapshot", "town", "seed")
REQ_DS = ("view/axis_hit_m", "actors/yaw", "actors/pos", "actors/extent", "seg/iid", "seg/semantic", "view/block_frac",
          "state/loss_window", "state/action_valid", "instance_mask", "traffic/blobs")


def check_episode(p):
    r = {"file": os.path.basename(p), "fails": []}
    with h5py.File(p, "r") as f:
        a = dict(f.attrs)
        miss = [k for k in REQ_ATTRS if k not in a]
        if miss or float(a.get("fov", -1)) != 90.0:
            r["fails"].append(f"G1 attrs missing={miss} fov={a.get('fov')}")
        for k in ("town", "subset", "episode_id", "test_town", "language", "strategy", "num_similar"):
            v = a.get(k)
            r[k] = v.item() if hasattr(v, "item") else v
        T = f["rgb"].shape[0]; r["T"] = int(T)
        if f["rgb"].shape[1:] != (512, 512, 3) or T < 300:
            r["fails"].append(f"G2 rgb shape {f['rgb'].shape}")
        idx = np.linspace(0, T - 1, 16).astype(int)
        if any(f["rgb"][i].max() < 5 for i in idx):
            r["fails"].append("G2 black frame")
        if "actors" not in f:
            r["fails"].append("no actors group"); return r
        miss_ds = [k for k in REQ_DS if k not in f]
        if miss_ds:
            r["fails"].append(f"G1 datasets missing={miss_ds}")
        if "view/block_frac" in f:
            bf = f["view/block_frac"][:]
            ah = f["view/axis_hit_m"][:] if "view/axis_hit_m" in f else np.zeros_like(bf)
            r["view_blocked_frames"] = int(((bf > 0.8) & (ah < 8.0)).sum())
            r["view_building_frames"] = int((bf > 0.8).sum())          # building fills view (kept, reported)
            if r["view_blocked_frames"] > 0.01 * T:
                r["fails"].append(f"G6 view blocked {r['view_blocked_frames']} frames")
        g = f["actors"]
        am, mo, px = g["amodal_box"][:], g["modal_box"][:], g["vis_px"][:]
        occ, inf_ = g["occ_raycast"][:], g["in_frame"][:]
        role = g["role"][:]
        # G3 containment of modal in amodal
        sel = px >= 50
        mx0, my0, mx1, my1 = (mo[..., i][sel] for i in range(4))
        ax0, ay0, ax1, ay1 = (am[..., i][sel] for i in range(4))
        ix = np.clip(np.minimum(mx1, ax1 + 3) - np.maximum(mx0, ax0 - 3) + 1, 0, None)
        iy = np.clip(np.minimum(my1, ay1 + 3) - np.maximum(my0, ay0 - 3) + 1, 0, None)
        area = (mx1 - mx0 + 1) * (my1 - my0 + 1)
        cont = ix * iy / np.maximum(area, 1)
        r["g3_n"], r["g3_ok"] = int(sel.sum()), float((cont >= 0.9).mean()) if sel.any() else float("nan")
        if sel.sum() >= 50 and r["g3_ok"] < 0.95:
            r["fails"].append(f"G3 alignment {r['g3_ok']:.3f}")
        # G4 association
        w = am[..., 2] - am[..., 0]; h = am[..., 3] - am[..., 1]
        clear = (am[..., 0] >= 0) & (np.minimum(w, h) >= 12) & (np.nan_to_num(occ, nan=1) < 0.2) & (inf_ == 1)
        r["g4_n"], r["g4_ok"] = int(clear.sum()), float((px[clear] > 0).mean()) if clear.any() else float("nan")
        if clear.sum() >= 50 and r["g4_ok"] < 0.85:
            r["fails"].append(f"G4 association {r['g4_ok']:.3f}")
        if not all(k in g for k in ("visible", "tracklet_id")) or "events/target_reentry" not in f:
            r["fails"].append("G5 postprocess missing"); return r
        vis = g["visible"][:].astype(bool)
        tv = vis[:, 0]
        toff = (inf_[:, 0] == 0) & ~tv                       # classes are disjoint: visible > off > occluded
        tocc = (inf_[:, 0] == 1) & ~tv
        r["frames"] = {"visible": int(tv.sum()), "occluded_or_tiny": int(tocc.sum()), "off_screen": int(toff.sum())}
        r["target_struct_occ_frames"] = int((tocc & (np.nan_to_num(occ[:, 0]) >= 0.5)).sum())
        ii = g["inst_id"][:, 0]
        both = tv[1:] & tv[:-1] & (ii[1:] >= 0) & (ii[:-1] >= 0)
        r["target_iid_switches"] = int((both & (ii[1:] != ii[:-1])).sum())
        if r["target_iid_switches"] > 3:
            r["fails"].append(f"G7 target iid switches {r['target_iid_switches']}")
        r["target_partial_occ"] = int(((np.nan_to_num(occ[:, 0]) > 0) & tv).sum())
        ev = f["events/target_reentry"][:]
        r["reentry"] = {"n": int(len(ev)), "gap>=30": int((ev[:, 1] >= 30).sum()) if len(ev) else 0,
                        "gap>=30_lookalike": int(((ev[:, 1] >= 30) & (ev[:, 3] >= 1)).sum()) if len(ev) else 0,
                        "cause_occ": int((ev[:, 2] >= 1).sum()) if len(ev) else 0}
        look = vis[:, 1:] & (role[1:] == 1)[None]
        r["lookalike_covis"] = float((look.any(1) & tv).sum() / max(tv.sum(), 1))
        tb = am[tv, 0]
        r["target_box_px_median"] = float(np.median(np.maximum(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]))) if tv.any() else float("nan")
        r["n_tracklets_target"] = int(len(set(g["tracklet_id"][:, 0].tolist()) - {-1}))
        sp = f["target/tspeed"][:]
        r["target_moving_frac"] = float((sp > 1.0).mean())
        if r["target_moving_frac"] < 0.5:
            r["fails"].append(f"G9 target moving only {r['target_moving_frac']:.2f}")
        sub = str(r.get("subset") or "")
        if T < 600:
            r["fails"].append(f"G8 short episode T={T}")
        if sub == "reacq" and r["reentry"]["gap>=30_lookalike"] < 1:
            r["fails"].append("G8 reacq: no >=3 s re-entry with look-alike")
        if sub == "occ" and r["target_struct_occ_frames"] < 20 and r["reentry"]["cause_occ"] < 1:
            r["fails"].append(f"G8 occ: struct-occ frames {r['target_struct_occ_frames']}, occ re-entries 0")
    return r


def viz(p, out_png, n=3):
    from PIL import Image, ImageDraw
    with h5py.File(p, "r") as f:
        T = f["rgb"].shape[0]; g = f["actors"]; tiles = []
        vis = g["vis_px"][:]
        cand = np.where((vis[:, 0] > 50) & ((vis[:, 1:] > 50).sum(1) >= 2))[0]
        idx = cand[np.linspace(0, len(cand) - 1, n).astype(int)] if len(cand) >= n else np.linspace(0, T - 1, n).astype(int)
        for i in idx:
            im = Image.fromarray(f["rgb"][i]); d = ImageDraw.Draw(im)
            m = f["instance_mask"][i]
            tint = np.array(im).copy(); tint[m == 1] = (255, 0, 0); tint[(m >= 2) & (m < 253)] = (0, 128, 255)
            im = Image.blend(im, Image.fromarray(tint), 0.35); d = ImageDraw.Draw(im)
            for a in range(g["amodal_box"].shape[1]):
                b = g["amodal_box"][i, a]
                if b[0] >= 0:
                    d.rectangle(list(b), outline="lime" if a == 0 else "white", width=1)
                mb = g["modal_box"][i, a]
                if mb[0] >= 0:
                    d.rectangle(list(mb), outline="yellow", width=1)
            d.text((4, 4), f"t={i}", fill="white")
            tiles.append(im)
        W = Image.new("RGB", (512 * len(tiles), 512))
        for k, t in enumerate(tiles):
            W.paste(t, (512 * k, 0))
        W.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--split-out"); ap.add_argument("--viz", type=int, default=0)
    ap.add_argument("--reject-report", action="store_true", help="also QC *.rejected.h5 → reject_report.json")
    a = ap.parse_args()
    files = sorted(p for p in glob.glob(os.path.join(a.root, "episode_*.h5"))
                   if not p.endswith((".tmp.h5", ".rejected.h5")))
    if a.reject_report:
        rej = [check_episode(p) for p in sorted(glob.glob(os.path.join(a.root, "episode_*.rejected.h5")))]
        led = collections.defaultdict(lambda: {"n": 0, "reasons": collections.Counter()})
        for r in rej:
            k = f"{r.get('subset')}/{r.get('town')}"
            led[k]["n"] += 1
            for fl in r["fails"]:
                led[k]["reasons"][fl.split(" ")[0] + " " + fl.split(" ")[1] if " " in fl else fl] += 1
        json.dump({"episodes": rej, "by_subset_town": {k: {"n": v["n"], "reasons": dict(v["reasons"])} for k, v in led.items()}},
                  open(os.path.join(a.root, "reject_report.json"), "w"), indent=1, default=str)
        print(f"reject ledger: {len(rej)} rejected episodes → {a.root}/reject_report.json")
    rows = [check_episode(p) for p in files]
    good = [r for r in rows if not r["fails"]]
    print(f"episodes {len(rows)}  pass {len(good)}  fail {len(rows) - len(good)}")
    for r in rows:
        if r["fails"]:
            print("  FAIL", r["file"], r["fails"])
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in good:
        k = (r["subset"], r["town"])
        c = agg[k]; c["eps"] += 1; c["frames"] += r["T"]
        for kk, vv in r["frames"].items():
            c[kk] += vv
        for kk, vv in r["reentry"].items():
            c["re_" + kk] += vv
    print(f"\n{'subset':6s} {'town':9s} {'eps':>4s} {'frames':>7s} {'vis%':>6s} {'occ%':>6s} {'off%':>6s} {'re>=3s':>7s} {'re>=3s+look':>11s} {'re_occ':>6s}")
    for (s, t), c in sorted(agg.items()):
        F = c["frames"]
        print(f"{s:6s} {t:9s} {c['eps']:4d} {F:7d} {c['visible'] / F:6.1%} {c['occluded_or_tiny'] / F:6.1%} {c['off_screen'] / F:6.1%} "
              f"{c['re_gap>=30']:7d} {c['re_gap>=30_lookalike']:11d} {c['re_cause_occ']:6d}")
    g3 = [r["g3_ok"] for r in rows if r.get("g3_n", 0) > 0]; g4 = [r["g4_ok"] for r in rows if r.get("g4_n", 0) > 0]
    print(f"\nG3 alignment (median over eps) {np.median(g3) if g3 else float('nan'):.3f}   G4 association {np.median(g4) if g4 else float('nan'):.3f}")
    print(f"target box px median {np.nanmedian([r.get('target_box_px_median', np.nan) for r in good]) if good else float('nan'):.1f}  "
          f"look-alike co-visibility {np.mean([r['lookalike_covis'] for r in good]) if good else float('nan'):.2f}")
    json.dump(rows, open(os.path.join(a.root, "qc_report.json"), "w"), indent=1, default=str)
    if a.split_out:
        test_town = next((r.get("test_town") for r in good if r.get("test_town")), "Town03")
        split = {"train": [], "val": [], "test": [], "test_town": test_town, "dataset": a.root,
                 "rule": "test = held-out town; val = episode_id % 10 == 9 within training towns; QC-failed excluded"}
        for r in good:
            if r["town"] == test_town:
                split["test"].append(r["file"])
            elif int(r.get("episode_id") or 0) % 10 == 9:
                split["val"].append(r["file"])
            else:
                split["train"].append(r["file"])
        json.dump(split, open(a.split_out, "w"), indent=1)
        print(f"split → {a.split_out}: train {len(split['train'])} val {len(split['val'])} test {len(split['test'])}")
    if a.viz:
        for r in good[: a.viz]:
            viz(os.path.join(a.root, r["file"]), os.path.join(a.root, f"viz_{r['file'][:-3]}.png"))
        print(f"viz → {a.root}/viz_*.png")


if __name__ == "__main__":
    main()
