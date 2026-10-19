"""P2 acceptance (framework §9, §14): decision parity of the acot runtime with the legacy runtime, open loop on recorded
act messages (collect record_act). Inside ONE policy container (python -m acot replay-parity starts it), for every arm:
  legacy  carla_uav_tracking/rollout/policy_server.py's Policy, built exactly as the server builds it (translator argv
          + behaviour env), Observation per message;
  acot    acot.runtime.server.Runtime from the arm's resolved configuration (no env);
both share one frozen VLM and are replayed with torch / numpy / random seeded 0 at every reset. Per tick the pick, the
search flag, the action (|Δ| ≤ atol), the S2 identity logits and the scorer reply (cand_lite: is_target / key / depth /
true slot) must agree.
  python -m acot.eval.replay_parity --specs <dir>/<arm>/resolved_spec.json … --record <act_record.pkl> --out <json>"""
import argparse
import json
import os
import random
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ATOL = 1e-5


def seed_all():
    import numpy as np
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def lite_legacy(cs):
    if cs is None:
        return None
    return {"cands": [[bool(c.is_target), int(c.idx), round(float(c.depth), 6)] for c in cs.cands],
            "true_idx": int(cs.true_idx)}


def lite_acot(cl):
    if cl is None:
        return None
    return {"cands": [[c["is_target"], c["idx"], round(c["depth"], 6)] for c in cl["cands"]], "true_idx": cl["true_idx"]}


def run_legacy(spec, msgs, limit):
    import numpy as np
    from acot.eval.replay_neutral import build_policy
    from acot.runtime.server import LEGACY_ENV
    from rollout.policy import Observation
    for k in LEGACY_ENV:
        os.environ.pop(k, None)
    for k, v in spec["policy_env"].items():
        os.environ[k] = "" if k in ("ASD_LOG", "TAH_DAGGER_DIR", "OPENING_DUMP") else v
    policy = build_policy(spec)
    out, n = [], 0
    for m in msgs:
        if m.get("cmd") == "reset":
            policy.reset()
            seed_all()
            continue
        if limit and n >= limit:
            break
        n += 1
        obs = Observation(rgb=np.asarray(m["rgb"]), language=m["language"],
                          proprio=np.asarray(m["proprio"], dtype=np.float32), step=int(m["step"]),
                          target_color=m.get("target_color", ""), target_bp=m.get("target_bp", ""),
                          retarget_event=bool(m.get("retarget_event", False)),
                          rgb_hi=None if m.get("rgb_hi") is None else np.asarray(m["rgb_hi"]))
        action, search, info = policy.act(obs, gt_candidates=m["gt_candidates"])
        s2 = int(m["step"]) % policy.s2_period == 0
        out.append({"step": int(m["step"]), "action": [float(x) for x in action], "search": float(search),
                    "pred": int(info.pred_slot), "lite": lite_legacy(info.cand_set),
                    "logits": [float(x) for x in np.asarray(info.tid_logits).ravel()] if s2 else None,
                    "conf": float(getattr(policy, "_pred_conf", float("nan")))})
    sock = getattr(policy, "_dam4sam_sock", None)
    if sock is not None:
        sock.close()
    for k in LEGACY_ENV:
        os.environ.pop(k, None)
    return out


def run_acot(arm, spec, msgs, limit, vlm):
    import numpy as np
    from acot.runtime.server import Runtime
    rt = Runtime(arm, spec.get("services") or {}, "cuda:0", vlm=vlm)
    out, n = [], 0
    for m in msgs:
        if m.get("cmd") == "reset":
            rt.reset()
            seed_all()
            continue
        if limit and n >= limit:
            break
        n += 1
        dec = rt.step(m)
        s2 = dec.trace is not None
        out.append({"step": int(m["step"]), "action": [float(x) for x in dec.action], "search": float(dec.search),
                    "pred": int(dec.pred_slot), "lite": lite_acot(rt.cand_lite()),
                    "logits": [float(x) for x in np.asarray(rt.policy.logits).ravel()] if s2 else None,
                    "conf": float(rt.policy.pred_conf), "trace": dec.trace})
    head = rt.policy.head
    if getattr(head, "sock", None) is not None:
        head.sock.close()
    return out


def compare(L, N):
    import numpy as np
    bad, dmax = [], 0.0
    for x, y in zip(L, N):
        da = float(np.max(np.abs(np.asarray(x["action"]) - np.asarray(y["action"]))))
        dmax = max(dmax, da)
        why = []
        if x["pred"] != y["pred"]:
            why.append("pred")
        if x["search"] != y["search"]:
            why.append("search")
        if da > ATOL:
            why.append("action")
        if x["lite"] != y["lite"]:
            why.append("cand_lite")
        if (x["logits"] is None) != (y["logits"] is None) or (x["logits"] is not None and (
                len(x["logits"]) != len(y["logits"]) or not np.allclose(x["logits"], y["logits"], atol=ATOL))):
            why.append("logits")
        if why:
            bad.append({"step": x["step"], "why": why, "legacy": {k: x[k] for k in ("pred", "search", "action", "conf")},
                        "acot": {k: y[k] for k in ("pred", "search", "action", "conf", "trace")}})
    s2 = [y["trace"] for y in N if y.get("trace")]
    return {"n": min(len(L), len(N)), "length_equal": len(L) == len(N), "n_mismatch": len(bad), "first": bad[:4],
            "max_abs_action_diff": dmax, "s2_ticks": len(s2),
            "pgif_fires": sum(1 for t in s2 if t.get("fire")), "pgif_absent": sum(1 for t in s2 if t.get("pgif") == "absent"),
            "abstain_ticks": sum(1 for t in s2 if t.get("pred", 0) < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.chdir(REPO)
    for p in (REPO, os.path.join(REPO, "carla_uav_tracking")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import yaml
    import rollout.online_vlm as OV
    from acot.eval.replay_neutral import load_msgs
    msgs = load_msgs(a.record)
    arms = []
    for sp in a.specs:
        spec = json.load(open(sp))
        arms.append((spec, json.load(open(os.path.join(os.path.dirname(sp), "resolved.json")))))
    fv = {json.dumps(arm["protocol"]["frontend"]["vlm"], sort_keys=True) for _, arm in arms}
    assert len(fv) == 1, "all arms must share the VLM frontend"
    fr = arms[0][1]["protocol"]["frontend"]["vlm"]
    cfg = yaml.safe_load(open(os.path.join(REPO, fr["model_config"])))
    vlm = OV.OnlineVLM(cfg, device="cuda:0", language_mode=fr["language_mode"])
    real = OV.OnlineVLM
    OV.OnlineVLM = lambda *x, **k: vlm                          # the legacy server builds "its own" VLM → the shared one
    res = {"record": a.record, "n_act": sum(m.get("cmd") == "act" for m in msgs), "atol": ATOL, "arms": {}}
    try:
        for spec, arm in arms:
            t0 = time.time()
            try:
                L = run_legacy(spec, msgs, a.limit)
                torch.cuda.empty_cache()
                N = run_acot(arm, spec, msgs, a.limit, vlm)
                torch.cuda.empty_cache()
                r = compare(L, N)
                r["ok"] = r["n_mismatch"] == 0 and r["length_equal"]
            except Exception as e:                              # noqa: BLE001
                import traceback
                r = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            r["seconds"] = round(time.time() - t0, 1)
            res["arms"][spec["arm"]] = r
            print(json.dumps({"arm": spec["arm"], **{k: r.get(k) for k in ("ok", "n", "n_mismatch", "max_abs_action_diff",
                                                                          "s2_ticks", "pgif_fires", "pgif_absent",
                                                                          "abstain_ticks", "error", "seconds")}}),
                  flush=True)
            json.dump(res, open(a.out, "w"), indent=1, default=str)
    finally:
        OV.OnlineVLM = real
    res["ok"] = all(r.get("ok") for r in res["arms"].values())
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({"ok": res["ok"]}), flush=True)


if __name__ == "__main__":
    main()
