"""DAM4SAM socket service (runs in the isolated `dam4sam` conda env / dam4sam-svc container).

External-baseline bridge: the policy-server (cu118, tid_head=dam4sam) can't share a process with
DAM4SAM (torch2.1 + SAM2). This tiny server loads DAM4SAMTracker and serves it over the length-
prefixed-pickle bridge (rollout/bridge.py). Protocol (dicts):
  {cmd:init,  rgb:(H,W,3) uint8, box:(x,y,w,h)} -> {ok, bbox:(x,y,w,h)|None}   # reset+init from target box
  {cmd:track, rgb:(H,W,3) uint8}                -> {bbox:(x,y,w,h)|None}        # predicted target box
  {cmd:close}                                    -> {ok}
bbox is derived from DAM4SAM's predicted mask (min/max of nonzero). None if the mask is empty.

Launch (in dam4sam-svc): conda activate dam4sam && python .../dam4sam_service.py --model sam21pp-L --port 5601
"""
from __future__ import annotations
import argparse
import os
import socket
import sys

import numpy as np
from PIL import Image

_REPO = "/nvidia/hque/code/cyhe/uav-acot-track"
sys.path.insert(0, os.path.join(_REPO, "carla_uav_tracking"))
sys.path.insert(0, os.path.join(_REPO, "external/DAM4SAM"))
os.chdir(os.path.join(_REPO, "external/DAM4SAM"))

from rollout.bridge import send_msg, recv_msg  # noqa: E402


def _mask_to_bbox(m):
    m = np.asarray(m)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min()))


def _pred_mask(out):
    if isinstance(out, dict):
        return out.get("pred_mask", next(iter(out.values())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam21pp-L", choices=["sam21pp-L", "sam21pp-B", "sam21pp-S", "sam21pp-T"])
    ap.add_argument("--port", type=int, default=5601)
    a = ap.parse_args()
    from dam4sam_tracker import DAM4SAMTracker
    tracker = DAM4SAMTracker(a.model)
    print(f"[dam4sam-svc] tracker loaded ({a.model}, ckpt={os.path.basename(tracker.checkpoint)})", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port)); srv.listen(1)
    print(f"[dam4sam-svc] listening on 0.0.0.0:{a.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[dam4sam-svc] client {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn); cmd = msg.get("cmd")
                if cmd == "init":
                    img = Image.fromarray(np.asarray(msg["rgb"], dtype=np.uint8))
                    x, y, w, h = msg["box"]
                    out = tracker.initialize(img, None, bbox=(int(x), int(y), int(w), int(h)))
                    send_msg(conn, {"ok": True, "bbox": _mask_to_bbox(_pred_mask(out))})
                elif cmd == "track":
                    img = Image.fromarray(np.asarray(msg["rgb"], dtype=np.uint8))
                    out = tracker.track(img)
                    send_msg(conn, {"bbox": _mask_to_bbox(_pred_mask(out))})
                elif cmd == "close":
                    send_msg(conn, {"ok": True}); break
                else:
                    send_msg(conn, {"error": f"unknown cmd {cmd}"})
        except (ConnectionError, EOFError) as e:
            print(f"[dam4sam-svc] client disconnected: {e}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
