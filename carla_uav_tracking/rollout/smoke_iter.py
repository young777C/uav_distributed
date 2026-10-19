"""Offline smoke test for the 1003 iteration code paths (no CARLA): feeds real recorded frames + GT-projected
candidates of one mvp episode through Policy.act() for each configuration and checks it runs, picks are valid,
gating/reset hooks fire without errors. Usage (in acot-uav-train container on cyh-carla-net):
  python carla_uav_tracking/rollout/smoke_iter.py --dam4sam-port 5604"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "carla_uav_tracking")):
    sys.path.insert(0, p)
from rollout.online_vlm import OnlineVLM  # noqa: E402
from rollout.policy import Observation, Policy  # noqa: E402

EP = "/nvidia/hque/data/carla_data/mvp_full_v5/episode_000003.h5"


def run(name, vlm, cfg, env, a, **kw):
    for k, v in env.items():
        os.environ[k] = v
    os.environ["ASD_LOG"] = f"/tmp/smoke_{name}.jsonl"
    pol = Policy(os.path.join(REPO, "runs/stage2_v5/stage2_best.pt"), cfg, vlm, device="cuda:0", s2_period=6,
                 ex_source="track", conf_tau=0.5, gallery_seed="committed", reanchor="off",
                 tah_ckpt=os.path.join(REPO, "runs/tah_relm_dag.pt"), dam4sam_host="dam4sam-svc",
                 dam4sam_port=a.dam4sam_port, **kw)
    pol.reset()
    picks, t0 = [], time.time()
    with h5py.File(EP, "r") as fh:
        lang = str(fh.attrs["language"])
        for i in range(0, 90):
            tpos = np.array([fh["target/tx"][i], fh["target/ty"][i], fh["target/tz"][i]])
            d = fh["distractors/positions"][i]
            campose = np.array([fh[f"state/cam_{k}"][i] for k in ("x", "y", "z", "pitch", "yaw")])
            obs = Observation(rgb=np.asarray(fh["rgb"][i]), language=lang, proprio=np.zeros(5, np.float32), step=i)
            _, _, info = pol.act(obs, (tpos, d, campose))
            if i % 6 == 0:
                picks.append((info.pred_slot, None if info.cand_set is None else info.cand_set.true_idx))
    ok = sum(1 for p, t in picks if t is not None and t >= 0 and p == t)
    n = sum(1 for _, t in picks if t is not None and t >= 0)
    nlog = sum(1 for _ in open(f"/tmp/smoke_{name}.jsonl")) if os.path.exists(f"/tmp/smoke_{name}.jsonl") else 0
    print(f"[smoke] {name:14s} OK  ticks={len(picks)} correct={ok}/{n} gate_events={pol._reanchor_events} "
          f"asd_log_lines={nlog} ({time.time() - t0:.0f}s)", flush=True)
    for k in env:
        os.environ.pop(k, None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dam4sam-port", type=int, default=5604); a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(REPO, "train/config_v5.yaml")))
    vlm = OnlineVLM(cfg, device="cuda:0", language_mode="neutral")
    T = json.dumps({"filter_kind": "tempered", "resid_decay": 0.98})
    E = json.dumps({"filter_kind": "tempered", "resid_decay": 0.98, "naive": True, "gate_tah": False, "gate_dam": False})
    cases = [("V0_tah_pgif", {}, dict(tid_head="tah", asd="filter", asd_cfg=T)),
             ("V2_coldwin", {"COLD_WINDOW": "5"}, dict(tid_head="tah", asd="filter", asd_cfg=T)),
             ("V2_single", {"COLD_WINDOW": "5", "COLD_SINGLE": "1"}, dict(tid_head="tah", asd="filter", asd_cfg=T)),
             ("V1_dam_pgif", {}, dict(tid_head="dam4sam", asd="filter", asd_cfg=T)),
             ("V1_selfreset", {"V1_SELFRESET": "1"}, dict(tid_head="dam4sam", asd="filter", asd_cfg=T)),
             ("V3_fusion", {}, dict(tid_head="fusion", asd="filter", asd_cfg=T)),
             ("V3_fusion_V2", {"COLD_WINDOW": "5"}, dict(tid_head="fusion", asd="filter", asd_cfg=T)),
             ("V3_ensemble", {}, dict(tid_head="fusion", asd="filter", asd_cfg=E))]
    for name, env, kw in cases:
        try:
            run(name, vlm, cfg, env, a, **kw)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"[smoke] {name:14s} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
