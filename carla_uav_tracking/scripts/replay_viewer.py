"""Read-only replay viewer for rollout eval_out experiments (no CARLA needed).

Reads <exp>/<method>/seeds/<seed>/{trk,ticks}.jsonl + scenario/episode.json and
serves a single page that replays all methods of one seed side by side.

    python replay_viewer.py --exp <eval_out/exp/NAME> --port 8770   # binds 127.0.0.1
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def jl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def jf(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


class Store:
    def __init__(self, exp):
        self.exp = exp

    def index(self):
        out = {}
        for m in sorted(os.listdir(self.exp)):
            sd = os.path.join(self.exp, m, "seeds")
            if not os.path.isdir(sd):
                continue
            for s in os.listdir(sd):
                if os.path.exists(os.path.join(sd, s, "trk.jsonl")):
                    out.setdefault(s, []).append(m)
        return dict(sorted(out.items()))

    def seed(self, seed):
        res = {}
        for m in self.index().get(seed, []):
            d = os.path.join(self.exp, m, "seeds", seed)
            ep = jf(d + "/episode.json")
            res[m] = {
                "trk": [{k: r[k] for k in ("t", "tid", "gt", "uv", "sz")} for r in jl(d + "/trk.jsonl")],
                "ticks": [{k: r.get(k) for k in ("t", "pred", "committed", "conf", "pgif", "search")}
                          for r in jl(d + "/ticks.jsonl")],
                "ep": {k: ep.get(k) for k in ("success", "id_switches", "track_seconds", "centering_rate",
                                              "max_wrong_run_s", "correct_committed_vis", "occluded_frames",
                                              "vis_frames", "reacquire_attempts", "reacquire_success_rate")},
            }
        sc = {}
        for m in res:
            sc = jf(os.path.join(self.exp, m, "seeds", seed, "scenario.json"))
            if sc:
                break
        return {"methods": res, "scenario": {k: sc.get(k) for k in ("target_desc", "strategy", "num_similar",
                                                                    "num_distractors")}}


PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_viewer.html")).read()


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype):
            b = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE, "text/html; charset=utf-8")
            elif u.path == "/api/index":
                self._send(json.dumps({"exp": os.path.basename(store.exp.rstrip("/")), "seeds": store.index()}),
                           "application/json")
            elif u.path == "/api/seed":
                s = parse_qs(u.query).get("seed", [""])[0]
                if not s.isdigit():
                    self.send_error(400)
                    return
                self._send(json.dumps(store.seed(s)), "application/json")
            else:
                self.send_error(404)

        def log_message(self, *a):
            pass
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Store(a.exp))).serve_forever()
