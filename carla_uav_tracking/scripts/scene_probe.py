"""1026 scene probe (stress_v2): rebuild each seed's scenario under EXPERT flight + target GT and record it as a frame
sequence for human inspection (scripts/video_index.py), plus per-frame GT and per-episode scene statistics.
Runs INSIDE cyh-carla (py3.8), no policy / GPU model. Overlay: target box green (ray-visible) / red (occluded or out
of frame), clones yellow, other distractors grey; header = town · tier · speed · OCCLUDED / E3.
NOTE: scene + expert flight of the seed, NOT any method's own flight.
  python scripts/scene_probe.py --carla-port 2070 --scenario stress_v2 --record-dir <dir> \
      --plan Town01:94950-94953 Town02:94954-94957 ...
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import carla
import cv2
import numpy as np

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
from rollout.env import StudentRolloutEnv  # noqa: E402
from rollout.run_rollout import load_config  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_stats import scene_stats  # noqa: E402

FOV = 90.0


def project(campose, wp, w, h):
    """→ (u, v, depth) or None"""
    T = carla.Transform(carla.Location(float(campose[0]), float(campose[1]), float(campose[2])),
                        carla.Rotation(pitch=float(campose[3]), yaw=float(campose[4]), roll=0.0))
    p = np.array(T.get_inverse_matrix()) @ np.array([wp[0], wp[1], wp[2], 1.0])
    if p[0] <= 0.1:
        return None
    f = w / (2.0 * np.tan(np.radians(FOV) / 2.0))
    return f * p[1] / p[0] + w / 2.0, -f * p[2] / p[0] + h / 2.0, float(p[0])


class Tap:
    """stub policy: draws the overlay, writes frames + per-frame GT"""
    def __init__(self, seed, out, scene, town, tier):
        self.seed, self.out, self.scene, self.town, self.tier = seed, out, scene, town, tier
        self.rows, self.prev, self.nframes = [], None, 0

    def reset(self):
        pass

    def close(self):
        pass

    def act(self, obs, gt_candidates):
        tpos, dstates, campose, vis = gt_candidates[0], gt_candidates[1], gt_candidates[2], gt_candidates[4]
        img = np.ascontiguousarray(obs.rgb).copy()
        h, w = img.shape[:2]
        t_vis, d_vis = (vis if vis is not None else (False, []))
        meta = list(getattr(self.scene, "_distractor_meta", []))
        uvt = project(campose, tpos, w, h)
        t_in = bool(uvt and 0 <= uvt[0] < w and 0 <= uvt[1] < h)
        sp = 0.0 if self.prev is None else float(np.hypot(tpos[0] - self.prev[0], tpos[1] - self.prev[1]) * 10.0)
        self.prev = tpos
        clones, others = [], []
        for k, d in enumerate(dstates):
            if np.isnan(d[0]):
                continue
            m = meta[k] if k < len(meta) else {}
            dv = bool(d_vis[k]) if k < len(d_vis) else False
            row = (round(float(d[0]), 2), round(float(d[1]), 2), round(float(np.hypot(d[3], d[4])), 2), dv)
            (clones if m.get("clone") else others).append(row)
            uv = project(campose, d[:3], w, h)
            if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                r = int(np.clip(0.5 * (w / 2.0) * 4.0 / uv[2], 5, 70))
                col = (255, 220, 0) if m.get("clone") else (150, 150, 150)
                cv2.rectangle(img, (int(uv[0]) - r, int(uv[1]) - r), (int(uv[0]) + r, int(uv[1]) + r), col,
                              2 if dv else 1)
        if t_in:
            r = int(np.clip(0.5 * (w / 2.0) * 4.0 / uvt[2], 5, 70))
            col = (0, 255, 0) if t_vis else (255, 40, 40)
            cv2.rectangle(img, (int(uvt[0]) - r, int(uvt[1]) - r), (int(uvt[0]) + r, int(uvt[1]) + r), col, 2)
        e3 = bool(self.scene.in_loss_window(obs.step * 0.1))
        tag = "OCCLUDED" if (t_in and not t_vis) else ("OUT" if not t_in else "")
        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(img, "%s %s | %s | t %.1fs | v %.1f m/s %s%s" % (self.seed, self.town, self.tier or "-",
                    obs.step / 10.0, sp, tag, "  E3" if e3 else ""), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        cv2.imwrite(os.path.join(self.out, "%04d.jpg" % obs.step), np.ascontiguousarray(img[:, :, ::-1]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.nframes = obs.step + 1
        self.rows.append({"t": int(obs.step), "tp": [round(float(x), 2) for x in tpos], "tv": bool(t_vis),
                          "tin": t_in, "e3": e3, "cl": clones, "ot": others})
        return (0.0, 0.0, 0.0, 0.0), 0, SimpleNamespace(cand_set=None, pred_slot=-1)


class NoScore:
    def reset(self, meta): pass
    def update(self, **k): pass
    def should_truncate(self): return False
    def finalize(self): return {}


def parse_plan(items):
    """['Town01:94950-94953', 'Town03:94960'] → [(seed, town)]"""
    out = []
    for it in items:
        town, rng = it.split(":")
        a, _, b = rng.partition("-")
        out += [(s, town) for s in range(int(a), int(b or a) + 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-port", type=int, required=True)
    ap.add_argument("--scenario", default="stress_v2")
    ap.add_argument("--plan", nargs="+", required=True, help="Town:seedA-seedB …")
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--set-name", default="v2probe")
    a = ap.parse_args()
    plan = parse_plan(a.plan)
    os.makedirs(a.record_dir, exist_ok=True)
    mp = os.path.join(a.record_dir, "seeds_meta.json")
    meta_all = json.load(open(mp)) if os.path.exists(mp) else {}
    for s, t in plan:
        meta_all[str(s)] = {"set": a.set_name, "town": t}
    json.dump(meta_all, open(mp, "w"))
    client = carla.Client("127.0.0.1", a.carla_port)
    client.set_timeout(180.0)
    cur = None
    for seed, town in plan:
        out = os.path.join(a.record_dir, str(seed))
        if os.path.exists(os.path.join(out, "done.json")):
            continue
        os.makedirs(out, exist_ok=True)
        cfg = load_config(a.scenario)
        tier = next((k for k, v in (cfg["environment"].get("town_tiers") or {}).items() if town in v), None)
        if town not in cfg["environment"].get("towns", []):
            cfg["environment"].setdefault("towns", []).append(town)
        world = client.load_world(town) if town != cur else client.reload_world(False)
        cur = town
        cfg["environment"]["seed"] = seed
        scene = SceneManager(world, cfg, client)
        tap = Tap(seed, out, scene, town, tier)
        env = StudentRolloutEnv(scene, world, policy=tap, scorer=NoScore(), fps=10, resolution=(512, 512),
                                expert_control=True, fov_deg=FOV, vis_raycast=True)
        r = env.run(max_steps=a.steps)
        with open(os.path.join(out, "frames.jsonl"), "w") as f:
            for row in tap.rows:
                f.write(json.dumps(row) + "\n")
        st = scene_stats([x["tp"] for x in tap.rows], [x["tv"] for x in tap.rows], [x["tin"] for x in tap.rows],
                         [x["cl"] for x in tap.rows], [x["ot"] for x in tap.rows]) if tap.rows else {}
        st["unjam_relocated"] = int(getattr(scene, "_kin_relocated", 0) or 0)
        st["unjam_lane_kicks"] = int(getattr(scene, "_kin_lane_kicks", 0) or 0)
        rm = getattr(scene, "_route_meta", None) or {}
        done = {"seed": seed, "town": town, "tier": tier, "frames": tap.nframes, "skipped": r.get("skipped"),
                "reason": r.get("reason"), "error": r.get("error"), "stats": st,
                "route": {k: rm.get(k) for k in ("planned", "length_m", "occluders", "too_long", "turns",
                                                 "n_vehicles_set")},
                "polyline": rm.get("polyline"),
                "n_clones": sum(1 for m in getattr(scene, "_distractor_meta", []) if m.get("clone")),
                "n_distractors": len(getattr(scene, "_distractor_meta", []))}
        if r.get("skipped") or not r.get("error"):
            json.dump(done, open(os.path.join(out, "done.json"), "w"))
        print("[probe]", seed, town, tier, {k: r.get(k) for k in ("skipped", "reason", "error", "steps")},
              json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
