"""Validation of the P1 GT audit (framework §7.3 item 5), inside the policy container. Replays the act messages recorded
for one episode (collector record_act) through the LEGACY Policy (built exactly as policy_server.py builds it):
  plain ×2   — no audit (the replay itself must be deterministic: same torch / numpy seeds at every reset);
  audited    — label properties + GT proxies; decisions must equal the plain run, and every read must be a reviewed
               non-diagnostic site;
  planted ×4 — positive controls: a decision-time read of tpos / dstates / vis[1] / true_idx from outside the reviewed
               sites must be flagged as an unknown site.
  python -m acot.eval.replay_neutral --spec <arm>/resolved_spec.json --record <seed dir>/act_record.pkl --out <json>"""
import argparse
import json
import os
import pickle
import random
import runpy
import socket
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


class _Stop(Exception):
    pass


def build_policy(spec):
    """run policy_server.py's main() up to serve(): the Policy is captured, the socket bind aborts the server."""
    from rollout.policy import Policy
    holder, oinit, obind = {}, Policy.__init__, socket.socket.bind

    def init(self, *a, **k):
        oinit(self, *a, **k)
        holder["p"] = self

    def bind(self, *a):
        raise _Stop()
    Policy.__init__, socket.socket.bind = init, bind
    sys.argv = ["policy_server.py", *spec["policy_argv"]]
    try:
        runpy.run_path(os.path.join(REPO, "carla_uav_tracking", "rollout", "policy_server.py"), run_name="__main__")
    except _Stop:
        pass
    finally:
        Policy.__init__, socket.socket.bind = oinit, obind
    return holder["p"]


def load_msgs(path):
    out = []
    with open(path, "rb") as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
    return out


def plant_tpos(policy, gt):
    return float(__import__("numpy").asarray(gt[0])[0])


def plant_dstates(policy, gt):
    return len(gt[1])


def plant_vis(policy, gt):
    if gt[4] is None:
        return None
    _, d = gt[4]
    return [bool(x) for x in d]


def plant_true_idx(policy, gt):
    cs = getattr(policy, "_cand_set", None)
    return None if cs is None else cs.true_idx


def replay(policy, msgs, audited, plant=None, limit=None):
    import numpy as np
    import torch
    from acot.runtime import audit as A
    from rollout.policy import Observation
    A.AUDIT.enabled = False
    A.AUDIT.reset()
    if audited:
        A.install_label_audit()
    outs, n = [], 0
    try:
        for m in msgs:
            if m.get("cmd") == "reset":
                policy.reset()
                torch.manual_seed(0)
                np.random.seed(0)
                random.seed(0)
                A.AUDIT.reset()
                A.AUDIT.enabled = audited
                continue
            if limit is not None and n >= limit:
                break
            n += 1
            obs = Observation(rgb=np.asarray(m["rgb"]), language=m["language"],
                              proprio=np.asarray(m["proprio"], dtype=np.float32), step=int(m["step"]),
                              target_color=m.get("target_color", ""), target_bp=m.get("target_bp", ""),
                              retarget_event=bool(m.get("retarget_event", False)),
                              rgb_hi=None if m.get("rgb_hi") is None else np.asarray(m["rgb_hi"]))
            gt = A.wrap_gt_candidates(m["gt_candidates"]) if audited else m["gt_candidates"]
            if plant is not None:
                plant(policy, gt)
            action, search, info = policy.act(obs, gt_candidates=gt)
            outs.append({"step": int(m["step"]), "action": [float(x) for x in action], "search": float(search),
                         "pred": int(info.pred_slot),
                         "logits": [float(x) for x in np.asarray(info.tid_logits).ravel()]})
        counts = A.AUDIT.reset()
    finally:
        A.AUDIT.enabled = False
        if audited:
            A.uninstall_label_audit()
    return outs, counts


def compare(a, b, atol=1e-5):
    import numpy as np
    bad = []
    for x, y in zip(a, b):
        if x["pred"] != y["pred"] or x["search"] != y["search"] or len(x["logits"]) != len(y["logits"]) \
                or not np.allclose(x["action"], y["action"], atol=atol) \
                or not np.allclose(x["logits"], y["logits"], atol=atol):
            bad.append({"step": x["step"], "a": x, "b": y})
    return {"n": min(len(a), len(b)), "length_equal": len(a) == len(b), "mismatches": bad[:5], "n_mismatch": len(bad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.chdir(REPO)
    for p in (REPO, os.path.join(REPO, "carla_uav_tracking")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from acot.runtime import audit as A
    spec = json.load(open(a.spec))
    msgs = load_msgs(a.record)
    policy = build_policy(spec)
    sites = A.load_sites(os.path.join(REPO, "acot", "config", "gt_sites.yaml"))
    p1, _ = replay(policy, msgs, audited=False)
    p2, _ = replay(policy, msgs, audited=False)
    au, counts = replay(policy, msgs, audited=True)
    viol, unknown, cats = A.judge(counts, sites, spec["uses_gt"])
    res = {"n_act": sum(m.get("cmd") == "act" for m in msgs), "plain_vs_plain": compare(p1, p2),
           "plain_vs_audited": compare(p1, au), "audited": {"violations": viol, "unknown": unknown, "by_category": cats},
           "positive_controls": {}}
    for name, fn in (("tpos", plant_tpos), ("dstates", plant_dstates), ("vis", plant_vis), ("true_idx", plant_true_idx)):
        _, c = replay(policy, msgs, audited=True, plant=fn, limit=12)
        _, unk, _ = A.judge(c, sites, spec["uses_gt"])
        res["positive_controls"][name] = sorted({(u["kind"], u["func"]) for u in unk})
    res["verdict"] = {
        "replay_deterministic": res["plain_vs_plain"]["n_mismatch"] == 0 and res["plain_vs_plain"]["length_equal"],
        "audit_decision_neutral": res["plain_vs_audited"]["n_mismatch"] == 0 and res["plain_vs_audited"]["length_equal"],
        "audit_clean": not viol and not unknown,
        "positive_controls_flagged": all(any(f.startswith("plant_") for _, f in v)
                                         for v in res["positive_controls"].values())}
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(res["verdict"]), flush=True)


if __name__ == "__main__":
    main()
