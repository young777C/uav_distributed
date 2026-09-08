"""Policy server (training container, py3.11 + torch + VLM).

Holds the OnlineVLM + 4-module Policy and serves .act() over TCP to the CARLA env
running in the cyh-carla container. Runs on the shared `cyh-carla-net` so the env
can reach it by name. See bridge.py for the wire format.

    bash train/docker/run.sh  → doesn't attach the shared net; launch explicitly:
    docker run --rm --gpus all --network cyh-carla-net --name acot-policy-server \
      -v $REPO:$REPO -v $DATA:$DATA:ro -v $HF_CACHE:/root/.cache/huggingface -w $REPO \
      acot-uav-train:cu118 python carla_uav_tracking/rollout/policy_server.py \
        --config train/config_v5.yaml --ckpt runs/stage2_v5/stage2_best.pt \
        --language-mode neutral --port 5555

--language-mode: neutral (M0, matches v5 training) | none (M1 w/o-language ablation).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

import numpy as np
import yaml

_CARLA_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.dirname(_CARLA_PKG)
for _p in (_REPO, _CARLA_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rollout.bridge import send_msg, recv_msg  # noqa: E402
from rollout.online_vlm import OnlineVLM  # noqa: E402
from rollout.policy import Policy, Observation  # noqa: E402


def _cand_lite(cand_set):
    """Serialize only what the env-side scorer needs (identities + depth), not feats."""
    if cand_set is None:
        return None
    return {
        "cands": [{"is_target": bool(c.is_target), "idx": int(c.idx), "depth": float(c.depth)}
                  for c in cand_set.cands],
        "true_idx": int(cand_set.true_idx),
    }


def serve(policy: Policy, port: int, host: str = "0.0.0.0"):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[policy-server] listening on {host}:{port} "
          f"(ckpt epoch={policy.ckpt_meta.get('epoch')} "
          f"mis_follow={policy.ckpt_meta.get('metrics', {}).get('mis_follow')})", flush=True)

    while True:
        conn, addr = srv.accept()
        print(f"[policy-server] client connected: {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                cmd = msg.get("cmd")
                if cmd == "reset":
                    policy.reset()
                    send_msg(conn, {"ok": True})
                elif cmd == "close":
                    if hasattr(policy, "_dagger_flush"):
                        policy._dagger_flush()          # flush the LAST episode's DAgger frames
                    send_msg(conn, {"ok": True})
                    break
                elif cmd == "act":
                    obs = Observation(rgb=np.asarray(msg["rgb"]), language=msg["language"],
                                      proprio=np.asarray(msg["proprio"], dtype=np.float32),
                                      step=int(msg["step"]),
                                      target_color=msg.get("target_color", ""),
                                      target_bp=msg.get("target_bp", ""))
                    action, search_mode, info = policy.act(obs, gt_candidates=msg["gt_candidates"])
                    send_msg(conn, {"action": [float(x) for x in action],
                                    "search_mode": float(search_mode),
                                    "pred_slot": int(info.pred_slot),
                                    "cand_lite": _cand_lite(info.cand_set)})
                else:
                    send_msg(conn, {"error": f"unknown cmd {cmd}"})
        except (ConnectionError, EOFError) as e:
            print(f"[policy-server] client disconnected: {e}", flush=True)
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default="runs/stage2_v5/stage2_best.pt")
    ap.add_argument("--language-mode", default="neutral", choices=["neutral", "none", "full"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--s2-period", type=int, default=1)
    ap.add_argument("--ablate", default=None, choices=[None, "ear", "iar"],
                    help="H1/H2: zero EAR/IAR into DiT (must match the ckpt's training ablation)")
    ap.add_argument("--ex-source", default="ear",
                    choices=["ear", "cv", "cv_gated", "zerovel", "zerovel_gated", "road", "track"],
                    help="what fills DiT z_ex: ear (default) | memory+CV prior (*_gated=loss only, "
                         "seeded from GT target) | road (P2a-1 road-graph WM: env-side lane-traversal, "
                         "loss only; needs run_rollout --predict-road) | track (Plan A: seed from "
                         "tid-COMMITTED candidate → grounding drives control). no-WM baseline = cv_gated.")
    ap.add_argument("--commit-k", type=int, default=5, help="track: hysteresis frames to switch committed target")
    ap.add_argument("--tid-head", default="xattn", choices=["xattn", "attrbind", "reid", "oracle", "assoc", "dam4sam", "tah"],
                    help="xattn=language grounding (default) | attrbind=CLIP-style head (needs --tid-ckpt) "
                         "| reid=temporal appearance-memory re-ID (match the tracked instance, not the "
                         "language; reframe test — no ckpt, falls back to xattn before a template exists)")
    ap.add_argument("--tid-ckpt", default=None, help="attrbind head ckpt (e.g. runs/tid_attrbind.pt)")
    ap.add_argument("--reid-feature", default="vlmpool", choices=["vlmpool", "dinov2"],
                    help="reid appearance feature: vlmpool (coarse Qwen grid) | dinov2 (crop→DINOv2, "
                         "instance-discriminative; offline milestone K2 breaks 0.5)")
    ap.add_argument("--reid-bank", type=int, default=1, help="reid K-view gallery size (sweet spot ~2-3)")
    ap.add_argument("--reid-topm", type=int, default=1, help="reid match = top-m mean cosine to gallery")
    ap.add_argument("--warmup-oracle-s", type=float, default=0.0,
                    help="diag: first N seconds use oracle identity (establish tracking) then hand to reid — "
                         "tests if the deployable gap is cascade-INITIATION vs reid-intrinsic")
    ap.add_argument("--frame-gain", type=float, default=0.0,
                    help="②: yaw-centering gain on the committed target (deg per frac-offset from center; "
                         "breaks the framing→reid cascade). 0=off; sign per camera convention.")
    ap.add_argument("--conf-tau", type=float, default=0.0,
                    help="①: min pick-margin (top1-top2) to commit-fly in track mode; below it, hold "
                         "commitment + dead-reckon (don't chase an ambiguous look-alike). 0=off")
    ap.add_argument("--reid-tavg", type=float, default=0.0,
                    help="temporal re-ID: per-actor EMA decay of the reid match score (0=off=single-frame "
                         "argmax; e.g. 0.7 = integrate evidence over ~3 ticks → robust to off-center frames)")
    ap.add_argument("--conf-consensus", type=float, default=0.0,
                    help="borrow#2: consensus-gated control — commit-fly only if the pick is ALSO motion-"
                         "consistent (dist < conf_consensus·gate), i.e. appearance AND motion agree; on "
                         "disagreement hold+search. 0=off. Needs --reid-motion>0.")
    ap.add_argument("--reid-motion", type=float, default=0.0,
                    help="borrow#1 (DeepSORT): motion-consensus weight fused into the reid score (0=off=pure "
                         "appearance). >0 softly penalizes candidates far from the CV-predicted target image "
                         "position → spatially-inconsistent look-alikes can't win on appearance alone.")
    ap.add_argument("--reid-motion-gate-px", type=float, default=160.0, help="borrow#1 motion soft-gate radius (px)")
    ap.add_argument("--gallery-seed", default="gt", choices=["gt", "committed"],
                    help="reid gallery template source: gt=legacy oracle-seeded (per-frame GT target crop) | "
                         "committed=DEPLOYABLE GT-free (store the believed-target crop; language at cold-start)")
    ap.add_argument("--reanchor", default="off", choices=["off", "lang", "oracle"],
                    help="deployable language re-anchor (paper2 甲): on sustained reid collapse, re-pick via "
                         "language + reset gallery. off | lang | oracle (GT re-anchor = headroom upper bound)")
    ap.add_argument("--reanchor-patience", type=int, default=5,
                    help="consecutive low-margin (<conf-tau) S2 ticks before a re-anchor fires (rate-limit)")
    ap.add_argument("--reanchor-tavg", type=float, default=0.0,
                    help="TEMPORAL language re-anchor: per-actor EMA decay of the language logit so re-anchor "
                         "commits the temporally-integrated language vote (0=off=single-frame; e.g. 0.5)")
    ap.add_argument("--assoc-mode", default="deepsort", choices=["motion", "deepsort"],
                    help="external baseline (--tid-head assoc): motion (OC-SORT-like CV image-space) | "
                         "deepsort (motion gate + crop-DINOv2 appearance) tracking-by-detection association")
    ap.add_argument("--assoc-gate-px", type=float, default=160.0, help="assoc motion gate radius (px)")
    ap.add_argument("--assoc-lambda", type=float, default=1.0, help="assoc appearance weight vs motion")
    ap.add_argument("--dam4sam-host", default="dam4sam-svc", help="external baseline (--tid-head dam4sam) service host")
    ap.add_argument("--tah-ckpt", default="runs/tah_rel.pt", help="path-B: TAHRel learned association ckpt")
    ap.add_argument("--dam4sam-port", type=int, default=5601, help="DAM4SAM service port")
    ap.add_argument("--port", type=int, default=5555)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(_REPO, a.config)))
    vlm = OnlineVLM(cfg, device=a.device, language_mode=a.language_mode)
    tck = os.path.join(_REPO, a.tid_ckpt) if a.tid_ckpt else None
    policy = Policy(os.path.join(_REPO, a.ckpt), cfg, vlm, device=a.device,
                    s2_period=a.s2_period, ablate=a.ablate, ex_source=a.ex_source, commit_k=a.commit_k,
                    tid_head=a.tid_head, tid_ckpt=tck,
                    reid_feature=a.reid_feature, reid_bank=a.reid_bank, reid_topm=a.reid_topm,
                    conf_tau=a.conf_tau, frame_gain=a.frame_gain, warmup_oracle_s=a.warmup_oracle_s,
                    conf_consensus=a.conf_consensus,
                    reid_tavg=a.reid_tavg, reid_motion=a.reid_motion, reid_motion_gate_px=a.reid_motion_gate_px,
                    gallery_seed=a.gallery_seed, reanchor=a.reanchor,
                    reanchor_patience=a.reanchor_patience, reanchor_tavg=a.reanchor_tavg, assoc_mode=a.assoc_mode,
                    assoc_gate_px=a.assoc_gate_px, assoc_lambda=a.assoc_lambda,
                    dam4sam_host=a.dam4sam_host, dam4sam_port=a.dam4sam_port, tah_ckpt=a.tah_ckpt)
    print(f"[policy-server] loaded policy (language_mode={a.language_mode}, "
          f"s2_period={a.s2_period}, ablate={a.ablate}, ex_source={a.ex_source}, commit_k={a.commit_k}, "
          f"tid_head={a.tid_head})", flush=True)
    serve(policy, a.port)


if __name__ == "__main__":
    main()
