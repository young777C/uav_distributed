"""Dataset-generation dashboard (2026-10-09). Serves a self-contained web page on 127.0.0.1:<port>; view it from your
own machine through an SSH tunnel:  ssh -N -L <port>:localhost:<port> <user>@<server>  →  http://localhost:<port>
  · overall / per-run / per-town progress, QC pass rate, recorder rejections, in-progress episodes, rate + ETA, workers
  · every finished episode is rendered (background thread) to an H.264 MP4: target box coloured by its per-frame status
    (visible = green, occluded = orange, off-screen = red banner, no box), look-alike distractors red, other distractors
    yellow, a status chip, time / target speed / STILL flag, the instruction, and a per-frame status timeline + playhead.
Runs to watch are listed in a JSON config that is re-read on every request (add a run without restarting):
  {"runs": [{"name": "smoke2", "dir": "/nvidia/hque/data/carla_data/<out>", "planned": 10,
             "planned_towns": {"Town01": 2}, "logs": "runs/datagen_moving/smoke2_*.log", "seed_base": 42}],
   "qc": {"min_moving": 0.5, "max_still_s": 60, "still_speed": 0.3}, "video_dir": "runs/datagen_moving/dash_videos"}
Box convention (as scripts/h5_to_mp4.py): annotation/bbox_{u,v,w,h} = target box centre + size; distractors/bbox
(N, D, 4) the same, NaN = absent; distractors/similar (D,) 1 = look-alike. Status: off_screen > occluded > visible.
Needs h5py, numpy, opencv (drawing) and PyAV (libx264), e.g. PYTHONPATH=runs/pydeps_dash:
  PYTHONPATH=runs/pydeps_dash python carla_uav_tracking/scripts/datagen_dashboard.py \\
      --config runs/datagen_moving/dashboard.json --port 8767"""
import argparse
import glob
import json
import os
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import h5py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from recording.building_mask import MAX_INSIDE_FRAC, episode_inside   # noqa: E402  (QC v3, 2026-10-11)
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOCK = threading.Lock()
QC_CACHE = {}          # h5 path → (mtime, record)
RENDER = {}            # "<run>/<stem>" → "queued" | "rendering" | "done" | "failed: …"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
STATUS_RGB = {0: (40, 200, 70), 1: (255, 150, 0), 2: (235, 50, 50)}       # visible / occluded / off-screen
SIMILAR_RGB, DISTR_RGB = (235, 50, 50), (255, 210, 0)
STATUS_NAME = {0: "VISIBLE", 1: "OCCLUDED", 2: "OFF-SCREEN"}


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_cfg(path):
    cfg = json.load(open(path))
    cfg.setdefault("qc", {})
    cfg["qc"] = {"min_moving": 0.5, "max_still_s": 60.0, "still_speed": 0.3, **cfg["qc"]}
    cfg["video_dir"] = _abs(cfg.get("video_dir", "runs/datagen_moving/dash_videos"))
    for r in cfg.get("runs", []):
        r["dir"] = _abs(r["dir"])
        r.setdefault("planned", 0)
        r.setdefault("seed_base", 42)
    return cfg


def _episode_files(d):
    fs = glob.glob(os.path.join(d, "**", "episode_*.h5"), recursive=True)
    done = sorted(f for f in fs if not f.endswith((".tmp.h5", ".rejected.h5")))
    return done, sorted(f for f in fs if f.endswith(".tmp.h5")), sorted(f for f in fs if f.endswith(".rejected.h5"))


def _bins(x, nb, how):
    """per-frame array → nb bins (max for status, mean for flags)."""
    n = len(x)
    if n == 0:
        return []
    edges = np.linspace(0, n, min(nb, n) + 1).astype(int)
    f = np.max if how == "max" else np.mean
    return [round(float(f(x[a:b])), 3) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def episode_qc(path, qc):
    mt = os.path.getmtime(path)
    with LOCK:
        c = QC_CACHE.get(path)
        if c and c[0] == mt:
            return c[1]
    with h5py.File(path, "r") as h:
        sp = h["target/tspeed"][:]
        off = h["annotation/off_screen"][:] > 0.5
        occ = (h["annotation/occluded"][:] > 0.5) & ~off
        status = np.where(off, 2, np.where(occ, 1, 0)).astype(np.int8)
        still = sp < qc["still_speed"]
        best = cur = 0
        for s in still:
            cur = cur + 1 if s else 0
            best = max(best, cur)
        a = h.attrs
        try:
            nd = len(json.loads(a.get("distractors", "[]")))
        except (TypeError, ValueError):
            nd = int(h["distractors/bbox"].shape[1])
        # kinematics extras (2026-10-09 fast-speed smoke): target speed while moving, UAV horizontal speed from the
        # expert action (dx, dy per 0.1 s step; perturbation frames excluded), "convoy still" = target AND every
        # distractor below still_speed in the same frame (the cascade the unjam logic must prevent)
        mv = sp[~still]
        pert = h["state/perturbation"][:] > 0 if "state/perturbation" in h else np.zeros(len(sp), bool)
        uav_v = np.hypot(h["action/dx"][:], h["action/dy"][:]) * 10.0 if "action/dx" in h else np.zeros(0)
        uav_v = uav_v[~pert[:len(uav_v)]] if len(uav_v) else uav_v
        convoy = None
        if "distractors/positions" in h and len(sp) > 1:
            dp = h["distractors/positions"][:]
            if dp.ndim == 3 and dp.shape[1]:
                dv = np.linalg.norm(np.diff(dp[..., :2], axis=0), axis=-1) * 10.0
                dv = np.vstack([dv[:1], dv])
                dstill = np.all(np.where(np.isfinite(dv), dv < qc["still_speed"], True), axis=1)
                convoy = float((still & dstill).mean())
        try:
            ku = json.loads(a.get("kin_unjam", "null") or "null")
        except (TypeError, ValueError):
            ku = None
        nfull = int(round(float(a.get("max_duration_s", 150)) * int(a.get("fps", 10))))
        rec = {"seed": int(a.get("seed", -1)), "town": str(a.get("town", "?")), "frames": int(len(sp)),
               "truncated": bool(len(sp) < nfull - 1), "dur_s": len(sp) / 10.0,
               "v_med": float(np.median(mv)) if len(mv) else 0.0, "v_p90": float(np.percentile(mv, 90)) if len(mv) else 0.0,
               "uav_vmax": float(uav_v.max()) if len(uav_v) else None, "convoy_still": convoy,
               "relocated": (ku or {}).get("relocated"), "lane_kicks": (ku or {}).get("lane_kicks"),
               "reloc_in_view": sum(bool(e.get("in_view")) for e in (ku or {}).get("events", [])) if ku else None,
               "in_bldg": float(episode_inside(h).mean()),
               "fov": str(a.get("fov", "?")), "moving": float(1 - still.mean()), "longest_still_s": best / 10.0,
               "vis": float((status == 0).mean()), "occ": float((status == 1).mean()), "off": float((status == 2).mean()),
               "lang": str(a.get("language", "")), "desc": str(a.get("target_desc", "")),
               "strategy": str(a.get("strategy", "")), "n_distr": nd, "n_similar": int(a.get("num_similar", 0)),
               "timeline": _bins(status, 240, "max"), "still_tl": _bins(still.astype(np.float32), 240, "mean"),
               "mtime": mt}
    reasons = []
    if rec["moving"] < qc["min_moving"]:
        reasons.append(f"moving {rec['moving']:.0%} < {qc['min_moving']:.0%}")
    if rec["longest_still_s"] >= qc["max_still_s"]:
        reasons.append(f"standstill {rec['longest_still_s']:.0f} s ≥ {qc['max_still_s']:.0f} s")
    if rec["in_bldg"] > MAX_INSIDE_FRAC:
        reasons.append(f"camera inside building {rec['in_bldg']:.0%} > {MAX_INSIDE_FRAC:.0%}")
    if rec["truncated"]:
        reasons.append(f"truncated at {rec['dur_s']:.0f} s")
    rec["pass"], rec["reasons"] = not reasons, reasons
    with LOCK:
        QC_CACHE[path] = (mt, rec)
    return rec


def _workers(run):
    out = []
    for lg in sorted(glob.glob(_abs(run["logs"]))) if run.get("logs") else []:
        try:
            lines = open(lg, errors="replace").read().splitlines()
        except OSError:
            continue
        cur = None
        for ln in lines:
            m = re.search(r"=== seed (\d+) ep (\d+) town (\S+) \(try (\d+)\) ===", ln)
            if m:
                cur = {"seed": int(m.group(1)), "town": m.group(3), "try": int(m.group(4))}
        finished = any("list done" in ln for ln in lines[-5:])
        last = next((ln for ln in reversed(lines) if ln.strip()), "")
        age = time.time() - os.path.getmtime(lg)
        state = "done" if finished else ("stalled?" if age > 900 else "running")
        warn = sum("[warn]" in ln for ln in lines)
        out.append({"log": os.path.basename(lg), "state": state, "current": cur, "last": last[-160:],
                    "idle_s": round(age), "warnings": warn})
    return out


def run_status(run, cfg):
    qc = cfg["qc"]
    done, tmp, rej = _episode_files(run["dir"])
    eps = []
    for f in done:
        stem = os.path.splitext(os.path.basename(f))[0]
        key = f"{run['name']}/{stem}"
        try:
            e = dict(episode_qc(f, qc))
        except Exception as ex:                                  # noqa: BLE001
            e = {"seed": -1, "town": "?", "pass": False, "reasons": [f"unreadable: {ex}"], "timeline": [], "still_tl": []}
        vid = os.path.join(cfg["video_dir"], run["name"], stem + ".mp4")
        e["key"], e["file"] = key, os.path.basename(f)
        e["video"] = f"/video/{run['name']}/{stem}.mp4" if os.path.exists(vid) else None
        e["render"] = RENDER.get(key, "done" if e["video"] else "queued")
        eps.append(e)
    towns = {}
    for e in eps:
        t = towns.setdefault(e["town"], {"done": 0, "pass": 0, "planned": 0})
        t["done"] += 1
        t["pass"] += int(bool(e.get("pass")))
    for t, k in (run.get("planned_towns") or {}).items():
        towns.setdefault(t, {"done": 0, "pass": 0, "planned": 0})["planned"] = k
    mts = sorted(e.get("mtime", 0) for e in eps if e.get("mtime"))
    workers = _workers(run)
    nw = max(1, sum(w["state"] == "running" for w in workers)) if workers else 1
    # rate from the last 2 h only: reused / moved episodes keep their old mtimes (10-11: 10-09 files → "230 h" ETA)
    rec = [m for m in mts if m >= time.time() - 7200]
    rate = (len(rec) - 1) / (rec[-1] - rec[0]) * 3600 if len(rec) >= 3 and rec[-1] > rec[0] else None
    n_pass = sum(bool(e.get("pass")) for e in eps)
    remaining = max(0, run["planned"] - n_pass) if run["planned"] else 0
    eta = remaining / rate * 3600 if rate and remaining and any(w["state"] == "running" for w in workers) else None
    if run.get("orch_log") and remaining and any(w["state"] == "running" for w in workers):
        # orchestrator-aware ETA (full_fast.py log): episodes still needed / recent pass rate × median minutes per
        # episode (incl. world prepare) / running workers — robust to reused files and to the start-up burst
        try:
            res = re.findall(r"seed \d+ \S+ (PASS|FAIL|REJECTED|produced NO file)(?:.*? in ([\d.]+) min)?",
                             open(_abs(run["orch_log"]), errors="replace").read())[-40:]
            mins = [float(m) for _, m in res if m]
            if len(res) >= 3 and mins:
                p_pass = max(0.2, sum(r == "PASS" for r, _ in res) / len(res))
                eta = remaining / p_pass * float(np.median(mins)) * 60 / nw
                rate = nw * 60 / float(np.median(mins)) * p_pass
        except OSError:
            pass
    inprog = []
    for f in tmp:
        m = re.search(r"episode_(\d+)\.tmp", f)
        ep = int(m.group(1)) if m else -1
        inprog.append({"file": os.path.basename(f), "seed": run["seed_base"] + ep if ep >= 0 else None,
                       "size_mb": round(os.path.getsize(f) / 2 ** 20), "age_s": round(time.time() - os.path.getmtime(f))})
    return {"name": run["name"], "dir": run["dir"], "planned": run["planned"], "done": len(eps), "pass": n_pass,
            "fail": len(eps) - n_pass, "rejected": len(rej), "in_progress": inprog, "towns": towns,
            "rate_per_h": round(rate, 2) if rate else None, "eta_s": round(eta) if eta else None,
            "workers": workers, "episodes": sorted(eps, key=lambda e: (e.get("town", ""), e.get("seed", 0)))}


# ----------------------------------------------------------------------------------------------- video rendering
def _put(img, txt, org, scale=0.45, color=(255, 255, 255), thick=1):
    import cv2
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def render_episode(h5p, out, qc, fps=10, panel=64):
    import av
    import cv2
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part.mp4"
    with h5py.File(h5p, "r") as h:
        rgb = h["rgb"]
        n, H, W = rgb.shape[0], rgb.shape[1], rgb.shape[2]
        tu, tv = h["annotation/bbox_u"][:], h["annotation/bbox_v"][:]
        tw, th = h["annotation/bbox_w"][:], h["annotation/bbox_h"][:]
        off = h["annotation/off_screen"][:] > 0.5
        occ = (h["annotation/occluded"][:] > 0.5) & ~off
        occr = h["annotation/occlusion"][:] if "annotation/occlusion" in h else np.zeros(n)
        status = np.where(off, 2, np.where(occ, 1, 0))
        sp = h["target/tspeed"][:]
        dbox = h["distractors/bbox"][:] if "distractors/bbox" in h else np.zeros((n, 0, 4))
        sim = h["distractors/similar"][:] if "distractors/similar" in h else np.zeros(dbox.shape[1])
        lang = str(h.attrs.get("language", ""))
        head = f"seed {h.attrs.get('seed', '?')} | {h.attrs.get('town', '?')} | {h.attrs.get('target_desc', '')}"
        strip = np.zeros((10, W, 3), np.uint8)                            # per-frame status timeline
        idx = np.minimum((np.arange(W) * n / W).astype(int), n - 1)
        for x, i in enumerate(idx):
            strip[:, x] = STATUS_RGB[int(status[i])]
            if sp[i] < qc["still_speed"]:
                strip[7:, x] = (120, 120, 120)                           # grey underline = target standing still
        lines = _wrap(lang, 66)[:2]
        c = av.open(tmp, "w")
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = W, H + panel, "yuv420p"
        s.options = {"crf": "26", "preset": "veryfast"}
        for i in range(n):
            img = np.zeros((H + panel, W, 3), np.uint8)
            img[:H] = rgb[i]
            fr = img[:H]
            for d in range(dbox.shape[1]):                               # distractors (look-alike red / other yellow)
                u, v, w, hh = dbox[i, d]
                if not np.all(np.isfinite([u, v, w, hh])):
                    continue
                x0, y0, x1, y1 = int(u - w / 2), int(v - hh / 2), int(u + w / 2), int(v + hh / 2)
                if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                    continue
                col = SIMILAR_RGB if d < len(sim) and sim[d] else DISTR_RGB
                cv2.rectangle(fr, (x0, y0), (x1, y1), col, 1)
                _put(fr, ("S" if d < len(sim) and sim[d] else "D") + str(d), (max(x0, 0), max(y0 - 3, 9)), 0.35, col)
            st = int(status[i])
            if st != 2 and np.all(np.isfinite([tu[i], tv[i], tw[i], th[i]])):  # target box coloured by status
                x0, y0 = int(tu[i] - tw[i] / 2), int(tv[i] - th[i] / 2)
                x1, y1 = int(tu[i] + tw[i] / 2), int(tv[i] + th[i] / 2)
                cv2.rectangle(fr, (x0, y0), (x1, y1), STATUS_RGB[st], 2)
                lab = "TARGET" + (f" occ {occr[i]:.2f}" if st == 1 else "")
                _put(fr, lab, (max(x0, 0), max(y0 - 4, 10)), 0.42, STATUS_RGB[st])
            cv2.rectangle(fr, (6, 6), (128, 26), STATUS_RGB[st], -1)        # status chip
            _put(fr, STATUS_NAME[st], (11, 21), 0.5, (255, 255, 255), 1)
            _put(fr, f"t={i / fps:5.1f}s  v={sp[i]:4.1f} m/s", (W - 190, 21), 0.45)
            if sp[i] < qc["still_speed"]:
                _put(fr, "STILL", (W - 190, 40), 0.45, (255, 90, 90))
            if st == 2:
                cv2.rectangle(fr, (W // 2 - 110, H // 2 - 16), (W // 2 + 110, H // 2 + 10), (235, 50, 50), -1)
                _put(fr, "TARGET OFF-SCREEN", (W // 2 - 100, H // 2), 0.6, (255, 255, 255), 1)
            p = img[H:]                                                    # bottom panel
            p[:] = (24, 24, 28)
            _put(p, head[:84], (6, 14), 0.38, (200, 200, 200))
            for k, ln in enumerate(lines):
                _put(p, ln, (6, 30 + 13 * k), 0.38)
            p[panel - 12:panel - 2] = strip
            x = int(i * W / n)
            p[panel - 14:panel, max(x - 1, 0):x + 1] = (255, 255, 255)        # playhead
            for pk in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
        c.close()
    os.replace(tmp, out)


def render_loop(cfg_path):
    while True:
        try:
            cfg = load_cfg(cfg_path)
            todo = []
            for run in cfg.get("runs", []):
                done, _, _ = _episode_files(run["dir"])
                for f in done:
                    stem = os.path.splitext(os.path.basename(f))[0]
                    out = os.path.join(cfg["video_dir"], run["name"], stem + ".mp4")
                    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(f):
                        todo.append((os.path.getmtime(f), f"{run['name']}/{stem}", f, out))
            for _, key, f, out in sorted(todo):                          # oldest first
                if RENDER.get(key, "").startswith("failed"):
                    continue
                RENDER[key] = "rendering"
                try:
                    render_episode(f, out, cfg["qc"])
                    RENDER[key] = "done"
                except Exception as ex:                                  # noqa: BLE001
                    RENDER[key] = f"failed: {ex}"
                    traceback.print_exc()
        except Exception:                                                # noqa: BLE001
            traceback.print_exc()
        time.sleep(15)


# ---------------------------------------------------------------------------------------------------- training / eval
# 2026-10-12 (P0-2 identity retrain, acot_note/1012_p02_identity_retrain.md): page /train, data /api/train, configured by
# the optional "train" section of the dashboard config (re-read on every request), all paths repo-relative:
#   {"title": …, "caches": [glob of artefact dirs], "trains": [{"label": …, "glob": artefact-dir glob}],
#    "ref_val": {"value": 0.813, "label": …}, "selection": json, "decision": json,
#    "eval": {"exp": name, "arms": [arm, …], "expect": 40, "ref": [[exp, arm], …] (paired with arms by position)}}
TRAIN_SEEN = {}         # artefact dir → (first time seen training, epochs then) for the ETA
TAH_EP = re.compile(r"\[tah\] ep(\d+)/(\d+) train_loss=([\d.]+) train_acc=([\d.]+) \| val_acc=([\d.]+)")


def _tah_run(d):
    log = os.path.join(d, "logs", "train.log")
    rec = {"name": os.path.basename(d), "epochs": [], "total": None, "state": "queued", "best": None, "best_epoch": None}
    if not os.path.exists(log):
        return rec
    for line in open(log, errors="replace"):
        m = TAH_EP.search(line)
        if m:
            rec["total"] = int(m.group(2))
            rec["epochs"].append([int(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(5))])
    if rec["epochs"]:
        b = max(rec["epochs"], key=lambda r: (r[3], -r[0]))          # first epoch with the highest val (= tah_train)
        rec["best"], rec["best_epoch"] = b[3], b[0]
    done = os.path.exists(log + ".rc")
    rec["state"] = ("registered" if os.path.exists(os.path.join(d, "manifest.json")) else
                    "finished" if done else "training" if rec["epochs"] else "starting")
    n, tot = len(rec["epochs"]), rec["total"]
    if rec["state"] == "training" and n and tot:              # rate from this server's own observations
        t0, n0 = TRAIN_SEEN.setdefault(d, (time.time(), n))
        if n > n0:
            rec["eta_s"] = (time.time() - t0) / (n - n0) * (tot - n)
    rec["age_s"] = int(time.time() - os.path.getmtime(log))
    return rec


def _eval_rows(exp, arm):
    if REPO not in sys.path:
        sys.path.append(REPO)                                           # acot (appended: no shadowing)
    from acot.analysis import metrics as M
    base = os.path.join(REPO, "carla_uav_tracking", "rollout", "eval_out", "exp", exp, arm, "seeds")
    st, val = {}, {}
    for f in glob.glob(os.path.join(base, "*", "status.json")):
        s = int(os.path.basename(os.path.dirname(f)))
        st[s] = json.load(open(f)).get("status")
        ep = os.path.join(os.path.dirname(f), "episode.json")
        if st[s] == "complete" and os.path.exists(ep):
            e = json.load(open(ep))
            val[s] = {k: M.value(k, e) for k in ("pc_correct", "tr90")}
    return st, val


def train_status(cfg):
    T = cfg.get("train")
    if not T:
        return {"configured": False}
    out = {"configured": True, "title": T.get("title", ""), "ref_val": T.get("ref_val"), "caches": [], "trains": []}
    for g in T.get("caches", []):
        for d in sorted(glob.glob(_abs(g))):
            out["caches"].append({"name": os.path.basename(d),
                                  "state": "registered" if os.path.exists(os.path.join(d, "manifest.json")) else "building"})
    for t in T.get("trains", []):
        runs = [_tah_run(d) for d in sorted(glob.glob(_abs(t["glob"]))) if not d.endswith("_last") and "_last@" not in d]
        out["trains"].append({"label": t["label"], "runs": runs})
    for k in ("selection", "decision"):
        p = T.get(k) and _abs(T[k])
        out[k] = json.load(open(p)) if p and os.path.exists(p) else None
    E = T.get("eval")
    if E:
        arms, refs, n = [], [], int(E.get("expect", 40))
        for i, arm in enumerate(E["arms"]):
            st, val = _eval_rows(E["exp"], arm)
            row = {"arm": arm, "expect": n, "complete": sum(v == "complete" for v in st.values()),
                   "skipped": sum(v == "skipped" for v in st.values()), "failed": sum(v == "failed" for v in st.values()),
                   "running": sum(v not in ("complete", "skipped", "failed") for v in st.values())}
            if i < len(E.get("ref", [])):
                rexp, rarm = E["ref"][i]
                _, rval = _eval_rows(rexp, rarm)
                common = sorted(set(val) & set(rval))
                row["ref"] = f"{rexp}/{rarm}"
                row["paired_n"] = len(common)
                for k in ("pc_correct", "tr90"):
                    if common:
                        row[k] = float(np.mean([val[s][k] for s in common]))
                        row[k + "_ref"] = float(np.mean([rval[s][k] for s in common]))
            arms.append(row)
        out["eval"] = {"exp": E["exp"], "arms": arms}
    return out


# ---------------------------------------------------------------------------------------------------- web server
class Handler(BaseHTTPRequestHandler):
    cfg_path = None

    def log_message(self, fmt, *args):                                   # quiet
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/train":
                return self._send(200, TRAIN_PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/train":
                body = json.dumps(dict(train_status(load_cfg(self.cfg_path)), generated_at=time.time()))
                return self._send(200, body.encode(), "application/json")
            if path == "/api/status":
                cfg = load_cfg(self.cfg_path)
                runs = [run_status(r, cfg) for r in cfg.get("runs", [])]
                tot = {k: sum(r[k] for r in runs) for k in ("planned", "done", "pass", "fail", "rejected")}
                tot["in_progress"] = sum(len(r["in_progress"]) for r in runs)
                tot["rendered"] = sum(1 for r in runs for e in r["episodes"] if e["video"])
                body = json.dumps({"generated_at": time.time(), "qc": cfg["qc"], "totals": tot, "runs": runs})
                return self._send(200, body.encode(), "application/json")
            m = re.match(r"^/video/([^/]+)/([^/]+\.mp4)$", path)
            if m and SAFE.match(m.group(1)) and SAFE.match(m.group(2)):
                return self._video(os.path.join(load_cfg(self.cfg_path)["video_dir"], m.group(1), m.group(2)))
            return self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as ex:                                          # noqa: BLE001
            traceback.print_exc()
            return self._send(500, str(ex).encode(), "text/plain")

    def _video(self, fp):
        if not os.path.isfile(fp):
            return self._send(404, b"no video", "text/plain")
        size = os.path.getsize(fp)
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m:                                                            # HTTP range (seeking in the browser)
            a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
            b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            b = min(b, size - 1)
            with open(fp, "rb") as f:
                f.seek(a)
                body = f.read(b - a + 1)
            return self._send(206, body, "video/mp4", {"Content-Range": f"bytes {a}-{b}/{size}", "Accept-Ranges": "bytes"})
        with open(fp, "rb") as f:
            body = f.read()
        return self._send(200, body, "video/mp4", {"Accept-Ranges": "bytes"})


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数据生成看板</title><style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#16161a;--mut:#6b6b70;--line:#e2e1dc;--vis:#28c846;--occ:#ff9600;--off:#eb3232;--still:#787878;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#141417;--card:#1d1d22;--ink:#ecebe7;--mut:#9a9aa0;--line:#2e2e34}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{padding:16px 20px 8px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);font-size:12px}
main{padding:0 20px 40px;max-width:1500px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0 18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.tile b{display:block;font-size:22px}.tile span{color:var(--mut);font-size:12px}
.run{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:18px}
.run h2{font-size:16px;margin:0 0 6px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}.run h2 small{color:var(--mut);font-weight:normal;font-size:12px}
.bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden;display:flex;margin:6px 0}.bar i{display:block;height:100%}
.towns{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin:8px 0}.town{font-size:12px}.town .bar{height:8px;margin:3px 0}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600;color:#fff}.ok{background:var(--vis)}.bad{background:var(--off)}.run-s{background:var(--acc)}.warn{background:var(--occ)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;margin-top:8px}
.ep{border:1px solid var(--line);border-radius:10px;padding:9px;cursor:pointer;background:var(--bg)}.ep:hover{border-color:var(--acc)}
.ep .t{display:flex;justify-content:space-between;gap:6px;font-weight:600}.ep .m{color:var(--mut);font-size:12px;margin:3px 0}
.ep canvas{width:100%;height:14px;display:block;border-radius:3px;margin-top:4px}.ep .r{font-size:11px;color:var(--off)}
.ctrl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}.ctrl select,.ctrl button{font:inherit;padding:3px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--mut)}.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:4px}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.72);display:none;align-items:center;justify-content:center;padding:16px;z-index:9}
#modal .box{background:var(--card);border-radius:12px;padding:12px;max-width:min(96vw,760px);width:100%}#modal video{width:100%;border-radius:8px;background:#000}
#modal .h{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;gap:8px}#modal .h button{font:inherit;border:0;background:var(--line);border-radius:6px;padding:4px 10px;cursor:pointer;color:var(--ink)}
.empty{color:var(--mut);font-size:13px;padding:8px 0}
</style></head><body>
<header><h1>数据生成看板 <a href="/train" style="font-size:13px;font-weight:normal;color:var(--acc)">→ 训练与闭环验收</a></h1><div class="sub" id="sub">加载中…</div></header>
<main>
<div class="tiles" id="tiles"></div>
<div class="legend"><span><i style="background:var(--vis)"></i>可见 visible</span><span><i style="background:var(--occ)"></i>遮挡 occluded</span><span><i style="background:var(--off)"></i>出画 off-screen</span><span><i style="background:var(--still)"></i>时间轴下沿灰色 = 目标静止</span><span>视频：目标框按状态着色；红框 S = 相似干扰车，黄框 D = 其他干扰车</span></div>
<div class="ctrl"><label>显示 <select id="flt"><option value="all">全部</option><option value="fail">未通过质检</option><option value="pass">通过</option></select></label>
<label><input type="checkbox" id="auto" checked> 每 15 秒自动刷新</label><button id="now">立即刷新</button></div>
<div id="runs"></div></main>
<div id="modal"><div class="box"><div class="h"><b id="mt"></b><button id="close">关闭</button></div><video id="vid" controls playsinline preload="metadata"></video><div class="sub" id="mi" style="margin-top:6px"></div></div></div>
<script>
const $=s=>document.querySelector(s), pct=x=>x==null?'—':(100*x).toFixed(0)+'%', fmtS=s=>{if(s==null)return '—';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h} 小时 ${m} 分`:`${m} 分`};
const COL={0:getComputedStyle(document.documentElement).getPropertyValue('--vis').trim()||'#28c846',1:'#ff9600',2:'#eb3232'};
let DATA=null;
function tile(v,l){return `<div class="tile"><b>${v}</b><span>${l}</span></div>`}
function bar(done,pass,planned){const tot=Math.max(planned,done,1);return `<div class="bar" title="通过 ${pass} / 未通过 ${done-pass} / 计划 ${planned}"><i style="width:${100*pass/tot}%;background:var(--vis)"></i><i style="width:${100*(done-pass)/tot}%;background:var(--off)"></i></div>`}
function drawTL(cv,tl,st){const w=cv.width=cv.clientWidth*devicePixelRatio||240,h=cv.height=14*devicePixelRatio,c=cv.getContext('2d');const n=tl.length||1;
 tl.forEach((s,i)=>{c.fillStyle=COL[Math.round(s)]||'#888';c.fillRect(i*w/n,0,Math.ceil(w/n),h*0.7)});
 st.forEach((f,i)=>{if(f>0.5){c.fillStyle='#787878';c.fillRect(i*w/n,h*0.72,Math.ceil(w/n),h*0.28)}})}
function render(){const d=DATA;if(!d)return;const t=d.totals;
 $('#sub').textContent=`更新于 ${new Date(d.generated_at*1000).toLocaleTimeString()} · 质检标准：运动帧 ≥ ${pct(d.qc.min_moving)} 且无 ≥ ${d.qc.max_still_s} s 的静止段（车速 < ${d.qc.still_speed} m/s 记为静止）`;
 const eta=d.runs.map(r=>r.eta_s).filter(x=>x!=null);
 $('#tiles').innerHTML=tile(`${t.done} / ${t.planned}`,'已完成 / 计划')+tile(t.done?pct(t.pass/t.done):'—','质检通过率')+tile(t.pass,'通过')+tile(t.fail,'未通过')+tile(t.rejected,'被录制器拒绝')+tile(t.in_progress,'正在生成')+tile(`${t.rendered} / ${t.done}`,'已渲染视频')+tile(eta.length?fmtS(Math.max(...eta)):(t.in_progress?'估算中':'—'),'预计剩余');
 const flt=$('#flt').value;let html='';
 for(const r of d.runs){
  html+=`<section class="run"><h2>${r.name}<small>${r.dir}</small><small>完成 ${r.done}/${r.planned} · 通过 ${r.pass} · 未通过 ${r.fail} · 拒绝 ${r.rejected}${r.rate_per_h?` · ${r.rate_per_h} 集/小时`:''}${r.eta_s?` · 预计剩余 ${fmtS(r.eta_s)}`:''}</small></h2>${bar(r.done,r.pass,r.planned)}`;
  const tw=Object.entries(r.towns).sort();if(tw.length){html+='<div class="towns">'+tw.map(([k,v])=>`<div class="town">${k} · ${v.done}${v.planned?'/'+v.planned:''}（通过 ${v.pass}）${bar(v.done,v.pass,v.planned)}</div>`).join('')+'</div>'}
  if(r.workers.length){html+='<table><tr><th>worker 日志</th><th>状态</th><th>当前</th><th>最后一行</th><th>空闲</th></tr>'+r.workers.map(w=>`<tr><td>${w.log}</td><td><span class="pill ${w.state=='done'?'ok':w.state=='running'?'run-s':'warn'}">${w.state}</span>${w.warnings?` <span class="pill warn">重试 ${w.warnings}</span>`:''}</td><td>${w.current?`seed ${w.current.seed} · ${w.current.town} · 第 ${w.current.try} 次`:'—'}</td><td style="font-family:monospace;font-size:11px">${(w.last||'').replace(/</g,'&lt;')}</td><td>${w.idle_s}s</td></tr>`).join('')+'</table>'}
  if(r.in_progress.length){html+='<div class="sub">正在写入：'+r.in_progress.map(p=>`seed ${p.seed}（${p.size_mb} MB）`).join('，')+'</div>'}
  const eps=r.episodes.filter(e=>flt=='all'||(flt=='pass'?e.pass:!e.pass));
  html+=eps.length?'<div class="grid">'+eps.map(e=>`<div class="ep" data-key="${e.key}"><div class="t"><span>seed ${e.seed} · ${e.town}</span><span class="pill ${e.pass?'ok':'bad'}">${e.pass?'通过':'未通过'}</span></div>
   <div class="m">运动 ${pct(e.moving)} · 最长静止 ${e.longest_still_s!=null?e.longest_still_s.toFixed(0)+' s':'—'} · 全队静止 ${e.convoy_still!=null?pct(e.convoy_still):'—'} · 相机在楼内 ${e.in_bldg!=null?pct(e.in_bldg):'—'}<br>目标车速 中位 ${e.v_med!=null?e.v_med.toFixed(1):'—'} / p90 ${e.v_p90!=null?e.v_p90.toFixed(1):'—'} m/s · 无人机水平 max ${e.uav_vmax!=null?e.uav_vmax.toFixed(1)+' m/s':'—'}${e.relocated!=null?`<br>解卡：挪车 ${e.relocated}（画面内 ${e.reloc_in_view}）· 强制变道 ${e.lane_kicks}`:''}<br>可见 ${pct(e.vis)} · 遮挡 ${pct(e.occ)} · 出画 ${pct(e.off)} · 干扰车 ${e.n_distr??'—'}（相似 ${e.n_similar??'—'}）</div>
   <canvas></canvas>${e.reasons&&e.reasons.length?`<div class="r">${e.reasons.join('；')}</div>`:''}<div class="m">${e.video?'▶ 点击播放':'视频：'+e.render}</div></div>`).join('')+'</div>':'<div class="empty">暂无符合条件的集</div>';
  html+='</section>'}
 $('#runs').innerHTML=html;
 document.querySelectorAll('.ep').forEach(el=>{const e=findEp(el.dataset.key);drawTL(el.querySelector('canvas'),e.timeline||[],e.still_tl||[]);el.onclick=()=>openEp(e)})}
function findEp(k){for(const r of DATA.runs)for(const e of r.episodes)if(e.key==k)return e}
function openEp(e){if(!e.video){alert('视频还在渲染：'+e.render);return}$('#mt').textContent=`seed ${e.seed} · ${e.town} · ${e.desc||''}`;$('#mi').textContent=e.lang||'';const v=$('#vid');v.src=e.video;$('#modal').style.display='flex';v.play().catch(()=>{})}
$('#close').onclick=()=>{$('#vid').pause();$('#vid').removeAttribute('src');$('#modal').style.display='none'};
$('#modal').onclick=ev=>{if(ev.target.id=='modal')$('#close').onclick()};
async function load(){try{const r=await fetch('/api/status',{cache:'no-store'});DATA=await r.json();render()}catch(err){$('#sub').textContent='读取失败：'+err}}
$('#flt').onchange=render;$('#now').onclick=load;setInterval(()=>{if($('#auto').checked&&$('#modal').style.display!='flex')load()},15000);load();
</script></body></html>"""


TRAIN_PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练与闭环验收</title><style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#16161a;--mut:#6b6b70;--line:#e2e1dc;--ok:#28a745;--bad:#eb3232;--warn:#ff9600;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#141417;--card:#1d1d22;--ink:#ecebe7;--mut:#9a9aa0;--line:#2e2e34}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{padding:16px 20px 8px}h1{font-size:20px;margin:0 0 4px}h1 a{font-size:13px;font-weight:normal;color:var(--acc)}.sub{color:var(--mut);font-size:12px}
main{padding:0 20px 40px;max-width:1300px}section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:16px}
h2{font-size:16px;margin:0 0 8px}table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}td,th{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left}th{color:var(--mut);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;font-weight:600;color:#fff}.ok{background:var(--ok)}.bad{background:var(--bad)}.run{background:var(--acc)}.warn{background:var(--warn)}.idle{background:#8a8a90}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden;display:flex;min-width:120px}.bar i{display:block;height:100%}
svg text{fill:var(--mut);font-size:11px}.note{color:var(--mut);font-size:12px;margin-top:6px}.big{font-size:22px;font-weight:700}
</style></head><body>
<header><h1>训练与闭环验收 <a href="/">← 数据生成看板</a></h1><div class="sub" id="sub">加载中…</div></header>
<main id="m"></main>
<script>
const $=s=>document.querySelector(s),f3=x=>x==null?'—':x.toFixed(3),pct=x=>x==null?'—':(100*x).toFixed(0)+'%';
const fmtS=s=>{if(s==null)return '—';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h} 小时 ${m} 分`:`${m} 分`};
const COLS=['#2a78d6','#eb6834','#1baf7a','#7a5bc7','#c0392b','#8a6d00'];
const ST={registered:['已登记','ok'],finished:['已训完','ok'],training:['训练中','run'],starting:['启动中','warn'],queued:['排队','idle'],building:['构建中','run']};
const pill=s=>{const[t,c]=ST[s]||[s,'idle'];return `<span class="pill ${c}">${t}</span>`};
function chart(runs,ref){const W=640,H=200,L=44,R=10,T=10,B=26;const all=runs.flatMap(r=>r.epochs.map(e=>e[3]));if(!all.length)return '<div class="note">还没有 epoch 记录</div>';
 const tot=Math.max(...runs.map(r=>r.total||1));let lo=Math.min(...all,ref?ref.value:1),hi=Math.max(...all,ref?ref.value:0);lo=Math.floor(lo*100-1)/100;hi=Math.ceil(hi*100+0.5)/100;
 const X=e=>L+(W-L-R)*(e-1)/Math.max(tot-1,1),Y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo||1));let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">`;
 for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;s+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L-4}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(3)}</text>`}
 s+=`<text x="${(L+W)/2}" y="${H-6}" text-anchor="middle">epoch(验证集 = 部署分布 DAgger,teacher-forced 准确率)</text>`;
 if(ref){s+=`<line x1="${L}" x2="${W-R}" y1="${Y(ref.value)}" y2="${Y(ref.value)}" stroke="#888" stroke-dasharray="5 4"/><text x="${W-R}" y="${Y(ref.value)-4}" text-anchor="end">${ref.label} ${ref.value}</text>`}
 runs.forEach((r,i)=>{if(!r.epochs.length)return;const c=COLS[i%COLS.length];s+=`<polyline fill="none" stroke="${c}" stroke-width="1.8" points="${r.epochs.map(e=>X(e[0])+','+Y(e[3])).join(' ')}"/>`;
  if(r.best_epoch)s+=`<circle cx="${X(r.best_epoch)}" cy="${Y(r.best)}" r="3.5" fill="${c}"/>`});return s+'</svg>'}
function render(d){if(!d.configured){$('#m').innerHTML='<section>看板配置里没有 train 段</section>';return}
 $('#sub').textContent=`${d.title} · 更新于 ${new Date(d.generated_at*1000).toLocaleTimeString()} · 每 15 秒自动刷新`;let h='';
 h+='<section><h2>① TAH 序列缓存</h2>'+(d.caches.length?d.caches.map(c=>`${pill(c.state)} ${c.name}`).join('<br>'):'<div class="note">无</div>')+'</section>';
 for(const g of d.trains){h+=`<section><h2>② 训练:${g.label}</h2><table><tr><th></th><th>artefact</th><th>状态</th><th>进度</th><th>最新验证</th><th>最好验证(轮)</th><th>预计剩余</th></tr>`;
  g.runs.forEach((r,i)=>{const n=r.epochs.length,t=r.total||0,last=n?r.epochs[n-1][3]:null;
   h+=`<tr><td><i style="display:inline-block;width:12px;height:12px;border-radius:2px;background:${COLS[i%COLS.length]}"></i></td><td>${r.name}</td><td>${pill(r.state)}</td><td><div class="bar"><i style="width:${t?100*n/t:0}%;background:var(--acc)"></i></div>${n}/${t||'?'}</td><td>${f3(last)}</td><td>${f3(r.best)}${r.best_epoch?` (${r.best_epoch})`:''}</td><td>${r.state=='training'?fmtS(r.eta_s):'—'}</td></tr>`});
  h+='</table>'+chart(g.runs,d.ref_val)+'</section>'}
 const s=d.selection;h+='<section><h2>③ 选种(验证集最好的种子,best 轮;规则运行前固定)</h2>'+(s?`选中 <b>seed ${s.picked.seed}</b>:${s.picked.artefact},验证 ${f3(s.picked.best_val_acc)}(第 ${s.picked.best_epoch} 轮)→ ${s.method_file}<div class="note">候选:${s.candidates.map(c=>`s${c.seed} ${f3(c.best_val_acc)}@${c.best_epoch}`).join(' · ')}</div>`:'<div class="note">训练完成后运行 select_and_pin.py</div>')+'</section>';
 const e=d.eval;if(e){h+=`<section><h2>④ 闭环验收:${e.exp}(dev-B 40 种子,@3,r = 2)</h2><table><tr><th>臂</th><th>进度</th><th>完成 / 跳过 / 失败 / 进行中</th><th>对照</th><th>已配对</th><th>承诺正确 新 / 旧</th><th>TR90 新 / 旧</th></tr>`;
  for(const a of e.arms){const tot=a.expect||40;h+=`<tr><td>${a.arm}</td><td><div class="bar"><i style="width:${100*a.complete/tot}%;background:var(--ok)"></i><i style="width:${100*a.skipped/tot}%;background:#8a8a90"></i><i style="width:${100*a.failed/tot}%;background:var(--bad)"></i></div></td><td>${a.complete} / ${a.skipped} / ${a.failed} / ${a.running}</td><td>${a.ref||'—'}</td><td>${a.paired_n??0}</td><td>${f3(a.pc_correct)} / ${f3(a.pc_correct_ref)}</td><td>${f3(a.tr90)} / ${f3(a.tr90_ref)}</td></tr>`}
  h+='</table><div class="note">「新 / 旧」只在已完成且两边都有的种子上配对平均,是运行中的临时数字;正式结论只看下方判定(两次重复全部完成后由 p02_decision 计算)。</div></section>'}
 const D=d.decision;h+='<section><h2>⑤ 判定(门槛运行前固定:承诺正确 Δ ≥ +0.05 且 CI 下界 > 0;TR90 CI 下界 > -0.02;身份耗时 p95 < 100 ms)</h2>';
 if(D){h+=`<div class="big">${D.pass?'<span class="pill ok" style="font-size:16px">通过 → 替换论文权重</span>':'<span class="pill bad" style="font-size:16px">未通过 → 保留旧权重</span>'}</div><table><tr><th>指标</th><th>n</th><th>新</th><th>旧</th><th>Δ [95% CI]</th><th>p</th></tr>`;
  for(const[k,r]of Object.entries(D.metrics)){const[ln,lr]=[D.new[0],D.ref[0]];h+=`<tr><td>${k}</td><td>${r.n}</td><td>${f3(r[ln])}</td><td>${f3(r[lr])}</td><td>${(r.diff>=0?'+':'')+f3(r.diff)} [${f3(r.ci95[0])}, ${f3(r.ci95[1])}]</td><td>${f3(r.p)}</td></tr>`}
  h+='</table><div class="note">'+Object.entries(D.criteria).map(([k,v])=>`${k}: ${v?'✓':'✗'}`).join(' · ')+'</div>'}else h+='<div class="note">闭环两次重复全部完成后计算</div>';
 h+='</section>';$('#m').innerHTML=h}
async function load(){try{const r=await fetch('/api/train',{cache:'no-store'});render(await r.json())}catch(err){$('#sub').textContent='读取失败:'+err}}
setInterval(load,15000);load();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8767)   # 8765 perf_inspector, 8766 control-frontier (10-09)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    Handler.cfg_path = os.path.abspath(a.config)
    load_cfg(Handler.cfg_path)
    if not a.no_render:
        threading.Thread(target=render_loop, args=(Handler.cfg_path,), daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[dashboard] http://{a.host}:{a.port}  (config {Handler.cfg_path})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
