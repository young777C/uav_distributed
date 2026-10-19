"""Policy-container entry point for P1 (framework §7.3, §8.4–8.5). Runs the LEGACY carla_uav_tracking/rollout/
policy_server.py unchanged (runpy) after:
  1. asserting that its argv and behaviour env are byte-identical to the translator's (resolved.json);
  2. wrapping the constructors (Policy, OnlineVLM, ASD, TrackletAssigner, AttrScorer, ZoomVerifier) to echo the
     EFFECTIVE configuration → server_echo.json (the scheduler compares it with the resolved config: L1);
  3. installing the GT access audit (acot/runtime/audit.py) — raw tpos / dstates / vis proxies + label properties;
  4. keying everything by the episode seed carried by the reset message (acot client): per-seed ticks.jsonl,
     gt_access.json, and logmap.json (legacy ASD / trk log `ep` index → seed); optional act recording for the replay test.
  python -m acot.eval.legacy_server_entry --spec <arm>/resolved.json --worker w0 -- <policy_server argv>"""
import argparse
import functools
import json
import os
import pickle
import runpy
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CTX = {"seed": None, "pending": None, "ticks": None, "rec": None, "rec_n": 0, "policy": None}
CAPTURED = {}


def die(msg):
    print(f"[acot-entry] FATAL: {msg}", flush=True)
    sys.exit(3)


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return f"<{type(v).__name__}>"


def capture(cls, label, keep=None):
    orig = cls.__init__

    @functools.wraps(orig)
    def init(self, *args, **kw):
        CAPTURED.setdefault(label, []).append({"args": [_jsonable(a) if keep is None or i in keep else "<obj>"
                                                        for i, a in enumerate(args)],
                                               "kwargs": {k: _jsonable(v) for k, v in kw.items()}})
        return orig(self, *args, **kw)
    cls.__init__ = init


def seed_dir(spec, seed):
    d = os.path.join(spec["results_dir"], "seeds", str(seed))
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--worker", required=True)
    a, rest = ap.parse_known_args()
    argv = rest[1:] if rest[:1] == ["--"] else rest
    spec = json.load(open(a.spec))
    if argv != spec["policy_argv"]:
        die(f"policy argv differs from the translator's:\n  got  {argv}\n  want {spec['policy_argv']}")
    paths = ("ASD_LOG", "TAH_DAGGER_DIR", "DAGGER_DIR", "OPENING_DUMP")     # per-worker collection paths: presence only
    bad = {k: (os.environ.get(k, ""), v) for k, v in spec["policy_env"].items()
           if ((bool(os.environ.get(k)) != bool(v)) if k in paths else (os.environ.get(k, "") != v))}
    if "DAGGER_DIR" in os.environ and "DAGGER_DIR" not in spec["policy_env"]:
        bad["DAGGER_DIR"] = (os.environ["DAGGER_DIR"], "<unset>")       # `is not None` switch in policy.py
    if bad:
        die(f"behaviour env differs from the translator's: {bad}")
    os.chdir(REPO)
    for p in (REPO, os.path.join(REPO, "carla_uav_tracking")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import rollout.bridge as bridge
    from acot.runtime import audit as A
    from rollout.online_vlm import OnlineVLM
    from rollout.policy import Policy
    from rollout.asd import ASD
    from rollout.tracklets import TrackletAssigner
    from rollout.opening_zoom import AttrScorer, ZoomVerifier

    sites = A.load_sites(os.path.join(REPO, "acot", "config", "gt_sites.yaml"))
    learn = spec.get("audit_learn", False)
    collect = set(spec["collect"])
    rec_max = int(spec.get("record_act_ticks", 40))
    logmap = []

    capture(OnlineVLM, "OnlineVLM", keep=())
    capture(Policy, "Policy", keep={0})
    capture(ASD, "ASD")
    capture(TrackletAssigner, "TrackletAssigner")
    capture(AttrScorer, "AttrScorer", keep=())
    capture(ZoomVerifier, "ZoomVerifier", keep=())
    if "gt_access" in collect:
        A.install_label_audit()

    def flush_episode():
        seed = CTX["seed"]
        if seed is None:
            return
        counts = A.AUDIT.reset()
        A.AUDIT.enabled = False
        if "gt_access" in collect:
            viol, unknown, cats = A.judge(counts, sites, spec["uses_gt"])
            write_json(os.path.join(seed_dir(spec, seed), "gt_access.json"), {
                "seed": seed, "worker": a.worker, "uses_gt": spec["uses_gt"], "fence": spec["gt_fence"],
                "violations": viol, "unknown_sites": unknown, "by_category": cats,
                "counts": [{"kind": k[0], "file": k[1], "func": k[2], "line": k[3], "n": n}
                           for k, n in sorted(counts.items())]})
        for key in ("ticks", "rec"):
            if CTX[key] is not None:
                CTX[key].close()
                CTX[key] = None
        CTX["seed"] = None

    orig_recv = bridge.recv_msg

    def recv_msg(sock):
        msg = orig_recv(sock)
        if isinstance(msg, dict):
            cmd = msg.get("cmd")
            if cmd == "reset":
                CTX["pending"] = msg.get("seed")
            elif cmd == "close":
                flush_episode()
            if CTX["rec"] is not None and cmd in ("act", "reset") and CTX["rec_n"] < rec_max:
                pickle.dump(msg, CTX["rec"])
                CTX["rec_n"] += cmd == "act"
        return msg
    bridge.recv_msg = recv_msg

    orig_init, orig_reset, orig_act = Policy.__init__, Policy.reset, Policy.act

    def init(self, *args, **kw):
        orig_init(self, *args, **kw)
        CTX["policy"] = self
        attrs = {k: _jsonable(getattr(self, k)) for k in (
            "_open_window", "_open_agg", "_open_final_reset", "_open_reset_asd", "_open_force_final", "_open_seed_asd",
            "cold_window", "cold_lambda", "cold_single", "v1_selfreset", "_dam4sam_every_frame", "_shuffle_seed",
            "s2_period") if hasattr(self, k)}
        import torch
        vers = {}
        for mod in ("torch", "timm", "transformers", "numpy"):
            try:
                vers[mod] = __import__(mod).__version__
            except Exception as e:                      # noqa: BLE001
                vers[mod] = f"<{e}>"
        write_json(os.path.join(spec["results_dir"], f"server_echo_{a.worker}.json"), {
            "worker": a.worker, "repo": REPO, "argv": argv, "env": {k: os.environ.get(k, "") for k in spec["policy_env"]},
            "env_set": sorted(k for k in ("DAGGER_DIR", "TAH_DAGGER_DIR", "OPENING_DUMP", "ASD_LOG") if k in os.environ),
            "captured": CAPTURED, "policy_attrs": attrs, "versions": vers,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "audit": {"labels": "gt_access" in collect, "raw": "gt_access" in collect, "learn": learn},
            "config_hash": spec["config_hash"], "t": time.time()})
        print(f"[acot-entry] echo written (config_hash {spec['config_hash']})", flush=True)
    Policy.__init__ = init

    def reset(self, *args, **kw):
        if CTX["policy"] is self:                        # an episode reset (not the constructor's own reset)
            flush_episode()
            CTX["seed"], CTX["pending"] = CTX["pending"], None
        out = orig_reset(self, *args, **kw)
        seed = CTX["seed"]
        if CTX["policy"] is self and seed is not None:
            asd = getattr(self, "_asd", None)
            logmap.append({"seed": seed, "asd_ep": getattr(asd, "ep", None), "trk_ep": getattr(self, "_trk_ep", None),
                           "t": time.time()})
            write_json(os.path.join(spec["results_dir"], f"logmap_{a.worker}.json"), logmap)
            d = seed_dir(spec, seed)
            if "ticks" in collect:
                CTX["ticks"] = open(os.path.join(d, "ticks.jsonl"), "w")
            if "record_act" in collect:
                CTX["rec"], CTX["rec_n"] = open(os.path.join(d, "act_record.pkl"), "wb"), 0
                pickle.dump({"cmd": "reset", "seed": seed}, CTX["rec"])
            A.AUDIT.reset()
            A.AUDIT.enabled = "gt_access" in collect
        return out
    Policy.reset = reset

    def act(self, obs, gt_candidates):
        gt = A.wrap_gt_candidates(gt_candidates) if A.AUDIT.enabled else gt_candidates
        out = orig_act(self, obs, gt_candidates=gt)
        fh = CTX["ticks"]
        if fh is not None and obs.step % self.s2_period == 0:
            was, A.AUDIT.enabled = A.AUDIT.enabled, False      # our own logging reads are not attributed
            try:
                info = out[2]
                cs = info.cand_set
                fh.write(json.dumps({
                    "seed": CTX["seed"], "t": int(obs.step), "n": 0 if cs is None else len(cs.cands),
                    "keys": [] if cs is None else [int(c.idx) for c in cs.cands],
                    "gt": [] if cs is None else [int(getattr(c, "gt_idx", c.idx)) for c in cs.cands],
                    "true": None if cs is None else int(cs.true_idx), "pred": int(info.pred_slot),
                    "conf": float(getattr(self, "_pred_conf", float("nan"))),
                    "search": float(out[1])}) + "\n")
                fh.flush()
            finally:
                A.AUDIT.enabled = was
        return out
    Policy.act = act

    sys.argv = ["policy_server.py", *argv]
    runpy.run_path(os.path.join(REPO, "carla_uav_tracking", "rollout", "policy_server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
