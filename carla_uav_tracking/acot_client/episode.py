"""acot CARLA-side client (framework §8; py3.8 inside cyh-carla): the run_rollout.py episode loop for ONE block of
consecutive seeds of one town (one load_world per process, as run_rollout), with per-seed status / results, a scenario
fingerprint, explicit scorer parameters, GT wire blanking and seed-carrying resets. Reads the host-resolved client.json.
  docker exec -e CARLA_TM_PORT=… cyh-carla python /workspace/acot_client/episode.py --client <container path>"""
import argparse
import json
import os
import platform
import sys
import time
import traceback

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import carla  # noqa: E402
from acot_client.scoring import AcotScorer  # noqa: E402
from acot_client.wire import AcotRemotePolicy  # noqa: E402
from rollout.env import StudentRolloutEnv  # noqa: E402
from rollout.run_rollout import load_config  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402

FINGERPRINT = ("seed", "target_type", "target_class", "target_blueprint", "target_color", "target_desc", "strategy",
               "num_similar", "num_distractors", "distractors", "uav_start_altitude", "weather", "language",
               "occlusion_biased", "camera_pitch")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    a = ap.parse_args()
    c = json.load(open(a.client))
    sc, ev = c["scorer"], c["env"]
    cfg = load_config()
    towns = cfg.setdefault("environment", {}).setdefault("towns", [])
    if c["town"] not in towns:
        towns.append(c["town"])
    client = carla.Client("127.0.0.1", int(c["carla_port"]))
    client.set_timeout(60.0)
    world = client.load_world(c["town"])
    print(f"[acot] map={world.get_map().name} policy={c['policy_host']}:{c['policy_port']} seeds={c['seeds']}",
          flush=True)
    policy = AcotRemotePolicy(c["policy_host"], int(c["policy_port"]), c["blank"],
                              {"experiment": c["experiment"], "arm": c["arm"], "config_hash": c["config_hash"]})
    write_json(os.path.join(c["results_dir"], "client_echo_{}_p{}.json".format(c["worker"], c["part"])),
               dict(c, python=platform.python_version(), carla=getattr(carla, "__file__", ""), t=time.time()))
    try:
        for seed in c["seeds"]:
            d = os.path.join(c["results_dir"], "seeds", str(seed))
            os.makedirs(d, exist_ok=True)
            prev = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
            st = {"seed": seed, "town": c["town"], "status": "partial", "worker": c["worker"], "part": c["part"],
                  "config_hash": c["config_hash"], "attempts": prev.get("attempts", []), "t_start": time.time()}
            write_json(os.path.join(d, "status.json"), st)
            cfg["environment"]["seed"] = seed
            det = ev.get("determinism") or {}
            if det.get("reload_world"):                  # 1006: fresh physics / TM state for every seed
                world = client.reload_world(False)       # keep the (synchronous) settings
            if det.get("sync_setup"):                    # 1006: no wall-clock physics while the scene is spawned
                ws = world.get_settings()
                ws.synchronous_mode = True
                ws.fixed_delta_seconds = 1.0 / float(c["fps"])
                world.apply_settings(ws)
            scene = SceneManager(world, cfg, client)
            box = {}
            setup = scene.setup_episode

            def setup_and_keep(_setup=setup, _box=box):
                m = _setup()
                _box["meta"] = m
                return m
            scene.setup_episode = setup_and_keep
            scorer = AcotScorer(fps=c["fps"], commit_k=sc["commit_k"], t_fail_s=sc["t_fail_s"],
                                max_lost_s=sc["max_lost_s"], central_frac=sc["central_frac"],
                                horizon_frames=int(c["steps"]))
            env = StudentRolloutEnv(scene, world, policy=policy, scorer=scorer, fps=c["fps"],
                                    resolution=tuple(c["resolution"]), central_frac=sc["central_frac"],
                                    fov_deg=c["fov_deg"], expert_control=ev["expert_control"],
                                    predict_road=ev["predict_road"] or ev["road_oracle"], road_oracle=ev["road_oracle"],
                                    vis_raycast=ev["vis_raycast"], hires_k=ev["hires_k"],
                                    retarget_tick=ev["retarget_tick"])
            policy.meta["seed"] = seed
            try:
                summary = env.run(max_steps=c["steps"])
            except Exception:                                                    # noqa: BLE001
                summary = {"error": traceback.format_exc(), "episode_seed": seed}
            if summary.get("skipped"):
                status, reason = "skipped", str(summary.get("reason"))
            elif "error" in summary:
                status, reason = "failed", "episode_error: " + str(summary["error"])[-500:]
            else:
                status, reason = "complete", None
            write_json(os.path.join(d, "episode.json"), summary)
            m = box.get("meta") or {}
            write_json(os.path.join(d, "scenario.json"), dict({k: m.get(k) for k in FINGERPRINT},
                                                              map=world.get_map().name, town=c["town"]))
            st.update(status=status, reason=reason, t_end=time.time(),
                      attempts=st["attempts"] + [{"worker": c["worker"], "part": c["part"], "status": status,
                                                  "reason": reason}])
            write_json(os.path.join(d, "status.json"), st)
            print(f"[acot] seed={seed} status={status}" + (f" reason={reason[:120]}" if reason else ""), flush=True)
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
