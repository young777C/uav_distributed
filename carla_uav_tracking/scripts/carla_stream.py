"""Live CARLA view of the stress scenario as MJPEG (route 2). Runs INSIDE cyh-carla (py3.8), no policy/GPU model:
re-spawns the seed's scenario (stress_v1) and flies the EXPERT through StudentRolloutEnv(expert_control=True) with a
stub policy that only taps the camera frame. Overlay = GT boxes (green target / yellow look-alike / grey other).
NOTE: scene + expert flight of the seed, NOT a replay of any method's own flight/commitments.
  python scripts/carla_stream.py --carla-port 2100 --seeds 94405 94406 --port 8771
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import carla
import cv2
import numpy as np

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
from rollout.env import StudentRolloutEnv  # noqa: E402
from rollout.run_rollout import load_config  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402

LOCK = threading.Lock()
STATE = {"jpg": None, "label": "starting"}
FOV = 90.0


def project(campose, wp, w, h):
    T = carla.Transform(carla.Location(float(campose[0]), float(campose[1]), float(campose[2])),
                        carla.Rotation(pitch=float(campose[3]), yaw=float(campose[4]), roll=0.0))
    p = np.array(T.get_inverse_matrix()) @ np.array([wp[0], wp[1], wp[2], 1.0])
    if p[0] <= 0.1:
        return None
    f = w / (2.0 * np.tan(np.radians(FOV) / 2.0))
    return int(f * p[1] / p[0] + w / 2.0), int(-f * p[2] / p[0] + h / 2.0)


class Tap:
    """Stub policy: no model. Draws the overlay, publishes the frame, paces to ~realtime."""
    def __init__(self, seed, pace, rec=None):
        self.seed, self.pace, self.t0, self.rec, self.vw = seed, pace, 0.0, rec, None

    def reset(self):
        self.t0 = time.time()

    def close(self):
        pass

    def act(self, obs, gt_candidates):
        tpos, dstates, campose = gt_candidates[0], gt_candidates[1], gt_candidates[2]
        img = np.ascontiguousarray(obs.rgb).copy()
        h, w = img.shape[:2]
        pts = [(tpos, (0, 255, 0), "T")] + [(d[:3], (0, 255, 255) if i < 99 else (160, 160, 160), str(i))
                                            for i, d in enumerate(dstates)]
        for wp, col, tag in pts:
            uv = project(campose, wp, w, h)
            if uv and 0 <= uv[0] < w and 0 <= uv[1] < h:
                cv2.rectangle(img, (uv[0] - 14, uv[1] - 14), (uv[0] + 14, uv[1] + 14), col, 2 if tag == "T" else 1)
        cv2.putText(img, "seed %s  step %d (expert flight)" % (self.seed, obs.step), (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.rec:
            cv2.imwrite(os.path.join(self.rec, "%04d.jpg" % obs.step), np.ascontiguousarray(img[:, :, ::-1]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.nframes = obs.step + 1
            return (0.0, 0.0, 0.0, 0.0), 0, SimpleNamespace(cand_set=None, pred_slot=-1)
        ok, jpg = cv2.imencode(".jpg", np.ascontiguousarray(img[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 80])   # obs.rgb is RGB-ordered as the policy sees it
        with LOCK:
            STATE["jpg"], STATE["label"] = jpg.tobytes(), "seed %s step %d" % (self.seed, obs.step)
        lag = self.t0 + (obs.step + 1) * self.pace - time.time()
        if lag > 0:
            time.sleep(lag)
        return (0.0, 0.0, 0.0, 0.0), 0, SimpleNamespace(cand_set=None, pred_slot=-1)


class NoScore:
    def reset(self, meta): pass
    def update(self, **k): pass
    def should_truncate(self): return False
    def finalize(self): return {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/":
            b = (b"<title>CARLA stress live</title><body style='background:#111;color:#ddd;font:14px system-ui'>"
                 b"<img src=/stream style='width:768px'><div id=l></div><script>setInterval(async()=>{l.textContent="
                 b"await(await fetch('/label')).text()},1000)</script>")
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(b)
        elif self.path == "/label":
            self.send_response(200); self.end_headers(); self.wfile.write(STATE["label"].encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
            last = None
            try:
                while True:
                    with LOCK:
                        j = STATE["jpg"]
                    if j is not None and j is not last:
                        last = j
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-port", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--record-dir", default=None, help="write <dir>/<seed>/%04d.jpg + done.json (900 steps, unpaced) instead of streaming")
    ap.add_argument("--exp-root", default="/workspace/rollout/eval_out/exp")
    a = ap.parse_args()
    if not a.record_dir:
        threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever, daemon=True).start()
    town = {}
    for p in glob.glob(a.exp_root + "/stress_*/*/seeds/*/status.json"):
        try:
            st = json.load(open(p)); town[st["seed"]] = st["town"]
        except Exception:
            pass
    cfg = load_config("stress_v1")
    client = carla.Client("127.0.0.1", a.carla_port)
    client.set_timeout(120.0)
    cur = None
    while True:
        for seed in a.seeds:
            out = os.path.join(a.record_dir, str(seed)) if a.record_dir else None
            if out and os.path.exists(os.path.join(out, "done.json")):
                continue
            t = town.get(seed, "Town05")
            if out:
                os.makedirs(out, exist_ok=True)
            tap = Tap(seed, 0.0 if out else 0.1, out)
            if t != cur:
                world = client.load_world(t); cur = t
            else:
                world = client.reload_world(False)
            cfg.setdefault("environment", {}).setdefault("towns", [])
            if t not in cfg["environment"]["towns"]:
                cfg["environment"]["towns"].append(t)
            cfg["environment"]["seed"] = seed
            scene = SceneManager(world, cfg, client)
            env = StudentRolloutEnv(scene, world, policy=tap, scorer=NoScore(), fps=10,
                                    resolution=(512, 512), expert_control=True, fov_deg=FOV)
            r = env.run(max_steps=a.steps)
            if out and (r.get("skipped") or not r.get("error")):
                json.dump({"seed": seed, "town": t, "frames": getattr(tap, "nframes", 0), "skipped": r.get("skipped"),
                           "reason": r.get("reason")}, open(os.path.join(out, "done.json"), "w"))
            print("[stream] seed", seed, t, {k: r.get(k) for k in ("skipped", "reason", "error", "steps")}, flush=True)
        if a.record_dir:
            return


if __name__ == "__main__":
    main()
