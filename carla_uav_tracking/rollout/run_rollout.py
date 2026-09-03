"""Closed-loop rollout entrypoint (CARLA container). Drives N episodes with the
remote VLA policy and writes per-episode + aggregate metrics (design §7).

    docker exec cyh-carla python /workspace/rollout/run_rollout.py \
      --policy-host acot-policy-server --policy-port 5555 \
      --episodes 5 --seed-base 90000 --steps 1500 --out /workspace/runs/rollout_m0.json

Seeds are OFFSET far from training (--seed-base) so scenarios are unseen (§7). Point
the policy server at language-mode neutral (M0) or none (M1 w/o-lang) to A/B test H0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

_CARLA_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARLA_PKG not in sys.path:
    sys.path.insert(0, _CARLA_PKG)

import carla  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402
from rollout.env import StudentRolloutEnv  # noqa: E402
from rollout.metrics import RolloutScorer  # noqa: E402
from rollout.remote_policy import RemotePolicy  # noqa: E402


def load_config():
    with open(os.path.join(_CARLA_PKG, "config", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    scen = os.path.join(_CARLA_PKG, "config", "scenarios", "mvp.yaml")
    if os.path.exists(scen):
        with open(scen) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def _merge(base, over):
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v)
        else:
            base[k] = v


def _aggregate(rows):
    done = [r for r in rows if not r.get("skipped")]
    n = max(len(done), 1)
    def mean(key):
        vals = [r[key] for r in done if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    tot_attempts = sum(r.get("reacquire_attempts", 0) for r in done)
    tot_success = sum(round((r.get("reacquire_success_rate") or 0) * r.get("reacquire_attempts", 0))
                      for r in done)
    tot_wrong = sum(round((r.get("reacquire_lock_wrong_rate") or 0) * r.get("reacquire_attempts", 0))
                    for r in done)
    return {
        "episodes": len(rows), "scored": len(done),
        "SR": sum(bool(r.get("success")) for r in done) / n,
        "mis_follow_sustained": mean("mis_follow_sustained"),
        "mis_follow_inst": mean("mis_follow_inst"),
        "id_switches_mean": mean("id_switches"),
        "track_seconds_mean": mean("track_seconds"),
        "centering_rate": mean("centering_rate"),
        # re-acquisition baseline (pooled over ALL attempts, not per-episode mean)
        "reacquire_attempts_total": tot_attempts,
        "reacquire_success_rate": (tot_success / tot_attempts) if tot_attempts else None,
        "reacquire_timeout_rate": (1 - tot_success / tot_attempts) if tot_attempts else None,
        "reacquire_lock_wrong_rate": (tot_wrong / tot_attempts) if tot_attempts else None,
        "reacquire_convergence_s_mean": mean("reacquire_convergence_s_mean"),
        "shortcut_nearest_rate": mean("shortcut_nearest_rate"),
        "shortcut_target_central_rate": mean("shortcut_target_central_rate"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-host", default="acot-policy-server")
    ap.add_argument("--policy-port", type=int, default=5555)
    ap.add_argument("--carla-port", type=int, default=int(os.environ.get("CARLA_PORT", "2012")))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--town", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--expert-control", action="store_true",
                    help="drive the drone with the EXPERT action (well-framed) while still "
                         "scoring the policy's tid — tests the framing→grounding→SR chain")
    ap.add_argument("--predict-road", action="store_true",
                    help="P2a-1: compute env-side road-graph target prediction (carla.Map lane "
                         "traversal) → gt_candidates[3]. MUST pair with the policy server's "
                         "--ex-source road; a no-op for the ear/cv_gated arms.")
    ap.add_argument("--road-oracle", action="store_true",
                    help="P2a-2 positive control: anchor the road prediction at the TRUE CURRENT "
                         "target pos every tick (bounds road's value if last-seen staleness+branch "
                         "error were removed). Implies --predict-road; pair with --ex-source road.")
    a = ap.parse_args()

    cfg = load_config()
    client = carla.Client("127.0.0.1", a.carla_port)
    client.set_timeout(30.0)
    world = client.load_world(a.town) if a.town else client.get_world()
    print(f"[rollout] map={world.get_map().name} policy={a.policy_host}:{a.policy_port} "
          f"episodes={a.episodes}", flush=True)

    policy = RemotePolicy(a.policy_host, a.policy_port)
    rows = []
    try:
        for ep in range(a.episodes):
            cfg.setdefault("environment", {})["seed"] = a.seed_base + ep
            scene = SceneManager(world, cfg, client)
            scorer = RolloutScorer(fps=10, max_lost_s=cfg["environment"].get("max_lost_seconds", 5))
            env = StudentRolloutEnv(scene, world, policy=policy, scorer=scorer,
                                    fps=10, resolution=(512, 512), expert_control=a.expert_control,
                                    predict_road=a.predict_road or a.road_oracle,
                                    road_oracle=a.road_oracle)
            summary = env.run(max_steps=a.steps)
            rows.append(summary)
            tag = "SKIP" if summary.get("skipped") else (
                "OK" if summary.get("success") else "FAIL")
            print(f"[rollout] ep{ep} seed={a.seed_base+ep} [{tag}] "
                  f"mis_follow_sus={summary.get('mis_follow_sustained')} "
                  f"track_s={summary.get('track_seconds')}", flush=True)
    finally:
        policy.close()

    agg = _aggregate(rows)
    print("\n[rollout] AGGREGATE:")
    for k, v in agg.items():
        print(f"    {k}: {v}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"aggregate": agg, "episodes": rows}, f, indent=2, default=str)
        print(f"[rollout] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
