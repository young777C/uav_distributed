"""Policy server of the acot runtime (framework §8, P2), in the policy image:
  python -m acot.runtime.server --spec <arm>/resolved_spec.json --worker <tag>
Builds the frontend, the perception and the policy from the arm's RESOLVED configuration (<arm>/resolved.json; its
recomputed hash must equal the scheduler's config_hash) and refuses to start if any legacy behaviour environment
variable is set. Speaks the legacy wire protocol (rollout/bridge.py) so the acot CARLA client is unchanged; the reply's
cand_lite also carries the simulator actor id of every box (gt_idx — the scorer's identity key, framework §12.2).
Per seed (from the reset message): ticks.jsonl (S2 decision trace joined with the labels), trk.jsonl, gt_access.json
(oracle calls by purpose + the redundant raw-GT read audit: every read must come from the perception), and the first
record_act_ticks act messages (act_record.pkl, for replay tests). server_echo_<tag>.json: config hash, versions, GPU."""
import argparse
import hashlib
import json
import os
import pickle
import socket
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# every legacy behaviour / collection variable (acot.config.legacy.POLICY_ENV + the client-side label dump)
LEGACY_ENV = ("ASD_LOG", "ASD_CFG", "COLD_WINDOW", "COLD_LAMBDA", "COLD_SINGLE", "V1_SELFRESET", "DAM4SAM_DIAG",
              "DAM4SAM_EVERY_FRAME", "TRK_CFG", "OPENING_DUMP", "OPENING", "OPENING_CFG", "ORACLE_ABSENT",
              "TAH_DAGGER_DIR", "DAGGER_DIR", "DUMP_LABELS")
# raw-GT reads (tpos / dstates / vis) are legitimate only inside the protocol's perception
PERCEPTION_FILES = ("acot/runtime/perception.py", "carla_uav_tracking/rollout/candidates.py")


def setup_paths():
    for p in (REPO, os.path.join(REPO, "carla_uav_tracking")):
        if p not in sys.path:
            sys.path.insert(0, p)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def legacy_env_set(environ=None):
    environ = os.environ if environ is None else environ
    return sorted(k for k in LEGACY_ENV if k in environ)


class Runtime:
    """frontend + perception + policy (+ oracle for diagnostic arms) of one resolved arm; step(msg) → (Decision,
    ScoringInfo of the latest S2 tick, detections of the latest S2 tick)."""

    def __init__(self, arm, services, device, vlm=None, log=print):
        from acot.runtime.frontend import Frontend
        from acot.runtime.io import OracleHandle
        from acot.runtime.perception import SimProxyPerception
        from acot.runtime.policy import AcotPolicy
        P, M = arm["protocol"], arm["method"]
        self.frontend = Frontend(P, device, vlm=vlm, log=log)
        self.perception = SimProxyPerception(P, self.frontend.image_cfg, M["control"]["shuffle_seed"],
                                             self.frontend.dino, device)
        self.policy = AcotPolicy(P, M, self.frontend, device, services=services, log=log)
        self.oracle = OracleHandle(arm["uses_gt"]) if arm["uses_gt"] else None
        self.info = self.dets = None
        self.frame_perc_ms = 0.0
        sh = P["frontend"]["opening"].get("shared")
        self.shared = json.load(open(os.path.join(REPO, sh["file"]))) if sh else None    # {seed: {step: actor}}
        self.seed = None

    def reset(self, seed=None, repeat=0):
        self.seed = seed
        self.policy.reset()
        self.policy.set_episode(rng_base(seed, repeat))
        self.perception.reset()
        self.info = self.dets = None
        self.frame_perc_ms = 0.0
        if self.oracle is not None:
            self.oracle.bind(None)

    def step(self, msg, gt=None):
        import numpy as np
        from acot.runtime.io import PolicyInput
        step = int(msg["step"])
        rgb = np.asarray(msg["rgb"])
        gt = msg["gt_candidates"] if gt is None else gt
        s2 = self.policy.is_s2(step)
        vlm = None
        if s2:
            layers, ctx, grid = self.frontend.encode(rgb, msg["language"])
            self.dets, self.info = self.perception.process(step, gt, grid, rgb, msg.get("target_color", ""),
                                                           msg.get("target_bp", ""))
            vlm = (layers, ctx)
            if self.oracle is not None:
                self.oracle.bind(self.info)
        fdets, fi = None, self.policy.frame_input
        if fi is not None and not s2:            # 1016 every-frame MOT baselines: this frame's boxes (no tracker keys)
            t_f = time.perf_counter()
            fdets = self.perception.frame_detections(gt, rgb, feats=fi == "boxes+feats")
            self.frame_perc_ms += 1000 * (time.perf_counter() - t_f)
        inp = PolicyInput(step=step, s2=s2, rgb=rgb, rgb_hi=None if msg.get("rgb_hi") is None else np.asarray(msg["rgb_hi"]),
                          language=msg["language"], proprio=np.asarray(msg["proprio"], dtype=np.float32),
                          campose=msg["gt_candidates"][2], vlm=vlm, detections=self.dets if s2 else fdets,
                          shared_open_slot=self.shared_slot(step) if s2 else None)
        dec = self.policy.act(inp, self.oracle)
        if fi is not None and dec.trace is not None:          # frame perception time since the last S2 tick
            dec.trace["frame_perc_ms"] = round(self.frame_perc_ms, 3)
            self.frame_perc_ms = 0.0
        return dec

    def shared_slot(self, step):
        """1006 §3.4: the harness (not the method) maps the reference actor of this opening tick to the current slot;
        the policy only receives the slot index. None = not a shared-opening tick."""
        if self.shared is None or self.seed is None:
            return None
        actor = (self.shared.get(str(self.seed)) or {}).get(str(int(step)))
        if actor is None:
            return None
        if self.dets is None or self.info is None:
            return -1
        g = list(self.info.gt_idx)
        return g.index(int(actor)) if int(actor) in g else -1

    def cand_lite(self):
        if self.dets is None:
            return None
        i = self.info
        return {"cands": [{"is_target": bool(t), "idx": int(d.key), "depth": float(d.depth_m), "gt_idx": int(g)}
                          for d, t, g in zip(self.dets, i.is_target, i.gt_idx)],
                "true_idx": int(i.true_idx)}


def dagger_frame(rt):
    """1006 DINOv3 stage 2 (collect tah_dagger): this S2 tick's TAH inputs in the train/tah_cache.py schema — the
    features / positions the TAH head scored (head.last, so the feature backbone is the method's own), the scene actor
    of every candidate (−1 = target) and the target's candidate index (−1 = absent). The GT ids are training LABELS
    written to disk only, never returned to the policy. → (head.last, frame) or None (no TAH scoring this tick)."""
    import numpy as np
    last = getattr(rt.policy.head, "last", None)
    if last is None or rt.info is None or last[0].shape[0] != len(rt.info.gt_idx):
        return None
    feat, pos = last
    return last, {"feat": feat.detach().cpu().numpy().astype(np.float16),
                  "pos": pos.detach().cpu().numpy().astype(np.float32),
                  "aidx": np.array(rt.info.gt_idx, np.int16), "tidx": int(rt.info.true_idx)}


def rng_base(seed, repeat=0):
    """opaque per-episode policy-RNG base H(scene seed, repeat) (1006 §3.5); None when the seed is unknown."""
    if seed is None:
        return None
    return int.from_bytes(hashlib.sha256(f"acot-policy-rng|{int(seed)}|{int(repeat)}".encode()).digest()[:8], "little")


def resolved_hash(arm):
    from acot.config.resolve import config_hash
    sci = {"method": arm["method"], "protocol": arm["protocol"],
           "artefacts": {r["path"]: r["artefact"] for r in arm.get("artefacts", [])}}
    if "repeat" in arm:
        sci["repeat"] = int(arm["repeat"])
    return config_hash(sci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--service-slot", type=int, default=0)        # 1006 G10: which DAM4SAM instance this unit owns
    a = ap.parse_args()
    bad = legacy_env_set()
    if bad:
        print(f"[acot-server] FATAL: legacy behaviour environment variables are set: {bad} — the acot runtime reads "
              f"the resolved configuration only", flush=True)
        sys.exit(3)
    os.chdir(REPO)
    setup_paths()
    import numpy as np
    import torch
    from acot.runtime import audit as A
    from rollout.bridge import recv_msg, send_msg

    spec = json.load(open(a.spec))
    arm = json.load(open(os.path.join(spec["results_dir"], "resolved.json")))
    h = resolved_hash(arm)
    if h != spec["config_hash"] or h != arm["config_hash"]:
        print(f"[acot-server] FATAL: resolved configuration hashes to {h}, spec says {spec['config_hash']}", flush=True)
        sys.exit(3)
    collect = set(spec["collect"])
    from acot.runtime.support import COLLECTORS
    unknown = collect - set(COLLECTORS)
    if unknown:
        print(f"[acot-server] FATAL: collectors {sorted(unknown)} are not implemented by the acot runtime "
              f"(use runtime: legacy)", flush=True)
        sys.exit(3)
    services = dict(spec.get("services") or {})
    if "dam4sam" in services and services["dam4sam"].get("instances"):
        services["dam4sam"] = services["dam4sam"]["instances"][a.service_slot]
    rt = Runtime(arm, services, a.device)
    if "tah_dagger" in collect and getattr(rt.policy.head, "name", None) != "tah":
        print("[acot-server] FATAL: collect tah_dagger needs the TAH identity head", flush=True)
        sys.exit(3)
    vers = {}
    for mod in ("torch", "timm", "transformers", "numpy"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception as e:                              # noqa: BLE001
            vers[mod] = f"<{e}>"
    write_json(os.path.join(spec["results_dir"], f"server_echo_{a.worker}.json"), {
        "runtime": "acot", "worker": a.worker, "repo": REPO, "config_hash": h,
        "resolved_sha256": hashlib.sha256(open(os.path.join(spec["results_dir"], "resolved.json"), "rb").read()).hexdigest(),
        "uses_gt": arm["uses_gt"], "oracle": sorted(rt.oracle.allowed) if rt.oracle else [], "collect": sorted(collect),
        "image": os.environ.get("ACOT_IMAGE", ""), "image_id": os.environ.get("ACOT_IMAGE_ID", ""), "versions": vers,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "t": time.time()})

    S = {"seed": None, "ticks": None, "trk": None, "rec": None, "rec_n": 0, "dag": None, "dag_last": None}
    rec_max = int(spec.get("record_act_ticks", 120))

    def seed_dir(seed):
        d = os.path.join(spec["results_dir"], "seeds", str(seed))
        os.makedirs(d, exist_ok=True)
        return d

    def flush():
        if S["seed"] is None:
            return
        counts = A.AUDIT.reset()
        A.AUDIT.enabled = False
        if "gt_access" in collect:
            reads = [{"kind": k[0], "file": k[1], "func": k[2], "line": k[3], "n": n} for k, n in sorted(counts.items())]
            outside = [r for r in reads if r["file"] not in PERCEPTION_FILES]
            calls = rt.oracle.reset_calls() if rt.oracle else {}
            write_json(os.path.join(seed_dir(S["seed"]), "gt_access.json"), {
                "seed": S["seed"], "runtime": "acot", "worker": a.worker, "uses_gt": arm["uses_gt"],
                "oracle_calls": calls, "raw_reads": reads,
                "violations": [r for r in outside if r["kind"] != "label"],
                "unknown_sites": [], "fence": {"structural": True, "raw_audit": True}})
        for k in ("ticks", "trk", "rec"):
            if S[k] is not None:
                S[k].close()
                S[k] = None
        if S["dag"] is not None:                                           # one episode, train/tah_merge.py schema
            d_feat = int(S["dag"][0]["feat"].shape[1]) if S["dag"] else None
            torch.save({"episodes": [S["dag"]], "d_feat": d_feat, "seed": S["seed"]},
                       os.path.join(seed_dir(S["seed"]), "tah_dagger.pt"))
            S["dag"], S["dag_last"] = None, None
        if torch.cuda.is_available():                                      # 1006 §4.2 cost ③ (per episode)
            write_json(os.path.join(seed_dir(S["seed"]), "cost.json"),
                       {"gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                        "gpu_reserved_peak_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)})
            torch.cuda.reset_peak_memory_stats()
        S["seed"] = None

    def on_reset(msg):
        flush()
        rt.reset(seed=msg.get("seed"), repeat=int(arm.get("repeat", 0)))
        S["seed"] = msg.get("seed")
        if S["seed"] is None:
            return
        d = seed_dir(S["seed"])
        if "ticks" in collect:
            S["ticks"] = open(os.path.join(d, "ticks.jsonl"), "w")
        if "trk" in collect:
            S["trk"] = open(os.path.join(d, "trk.jsonl"), "w")
        if "record_act" in collect:
            S["rec"], S["rec_n"] = open(os.path.join(d, "act_record.pkl"), "wb"), 0
            pickle.dump({"cmd": "reset", "seed": S["seed"]}, S["rec"])
        if "tah_dagger" in collect:
            S["dag"], S["dag_last"] = [], None
        A.AUDIT.reset()
        A.AUDIT.enabled = "gt_access" in collect

    def on_act(msg):
        if S["rec"] is not None and S["rec_n"] < rec_max:
            pickle.dump(msg, S["rec"])
            S["rec_n"] += 1
        gt = A.wrap_gt_candidates(msg["gt_candidates"]) if A.AUDIT.enabled else msg["gt_candidates"]
        dec = rt.step(msg, gt)
        if dec.trace is not None:
            was, A.AUDIT.enabled = A.AUDIT.enabled, False
            try:
                i, tr = rt.info, dec.trace
                if S["ticks"] is not None:
                    S["ticks"].write(json.dumps(dict(tr, seed=S["seed"], t=int(msg["step"]),
                                                     keys=list(tr.get("keys", [])), gt=list(i.gt_idx),
                                                     true=int(i.true_idx)), default=float) + "\n")
                    S["ticks"].flush()
                if S["trk"] is not None and rt.dets is not None:
                    S["trk"].write(json.dumps({"seed": S["seed"], "t": int(msg["step"]),
                                               "tid": [d.key for d in rt.dets], "gt": list(i.gt_idx),
                                               "uv": [[round(float(d.u), 1), round(float(d.v), 1)] for d in rt.dets],
                                               "sz": [round(float(d.w), 1) for d in rt.dets],
                                               "f": [np.round(d.crop_feat, 3).astype(np.float16).tolist()
                                                     for d in rt.dets]}) + "\n")
                    S["trk"].flush()
                if S["dag"] is not None:
                    r = dagger_frame(rt)
                    if r is not None and r[0] is not S["dag_last"]:
                        S["dag"].append(r[1])
                        S["dag_last"] = r[0]
            finally:
                A.AUDIT.enabled = was
        return {"action": [float(x) for x in dec.action], "search_mode": float(dec.search),
                "pred_slot": int(dec.pred_slot), "cand_lite": rt.cand_lite(),
                "committed": None if rt.policy.committed is None else int(rt.policy.committed)}

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(1)
    print(f"[acot-server] listening on 0.0.0.0:{a.port} (arm {arm['name']}, config_hash {h}, runtime acot)", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[acot-server] client connected: {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                cmd = msg.get("cmd")
                if cmd == "reset":
                    on_reset(msg)
                    send_msg(conn, {"ok": True})
                elif cmd == "act":
                    send_msg(conn, on_act(msg))
                elif cmd == "close":
                    flush()
                    send_msg(conn, {"ok": True})
                    break
                else:
                    send_msg(conn, {"error": f"unknown cmd {cmd}"})
        except (ConnectionError, EOFError) as e:
            print(f"[acot-server] client disconnected: {e}", flush=True)
            flush()
        finally:
            conn.close()


if __name__ == "__main__":
    main()
