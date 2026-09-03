"""Milestone-5a: CARLA-only inverted-loop smoke (no torch/VLM/acot_probe).

Validates the loop restructure (guide §1a): the student env captures the frame
BEFORE deciding, unlike the expert data-gen loop. Driven by an EXPERT adapter
(privileged GT action) + the real RolloutScorer, to confirm:
  - the reordered tick/capture/act/move runs in CARLA sync mode without breaking,
  - the sensor exposes a valid frame each tick at the acted-on pose,
  - the target stays largely in frame and the scorer produces sane numbers.

Candidate/tid scoring is skipped here (needs the VLM grid → policy side, M5b);
info.cand_set = None and the scorer degrades gracefully.

Runs INSIDE the CARLA container (interpreter `python`, /workspace = carla_uav_tracking):
    docker exec cyh-carla python /workspace/rollout/smoke_expert_loop.py
Env: CARLA_PORT (default 2012), SMOKE_STEPS (default 200), SMOKE_TOWN.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import yaml

_CARLA_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # carla_uav_tracking/ (=/workspace)
if _CARLA_PKG not in sys.path:
    sys.path.insert(0, _CARLA_PKG)

import carla  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402
from rollout.env import StudentRolloutEnv  # noqa: E402
from rollout.metrics import RolloutScorer  # noqa: E402


def load_config():
    with open(os.path.join(_CARLA_PKG, "config", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    scen = os.path.join(_CARLA_PKG, "config", "scenarios", "mvp.yaml")
    if os.path.exists(scen):
        with open(scen) as f:
            _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def _deep_merge(base, over):
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


class ExpertAdapter:
    """Policy stub honoring the real Policy contract (.reset()/.act()), driving the
    env with the privileged expert action so the loop + scorer run without torch."""

    def __init__(self, scene):
        self.scene = scene

    def reset(self):
        pass

    def act(self, obs, gt_candidates):
        tpos = gt_candidates[0]                         # (tpos, dstates, campose[, road_pred_world])
        ds = self.scene.drone.get_state()
        action, _ = self.scene.expert_action(tpos, ds.position, ds.yaw, True, obs.step * 0.1)
        info = SimpleNamespace(cand_set=None, pred_slot=-1,
                               tid_logits=np.zeros(0, np.float32), z_ex=np.zeros((4, 3), np.float32))
        return action, 0.0, info


def main() -> int:
    port = int(os.environ.get("CARLA_PORT", "2012"))
    steps = int(os.environ.get("SMOKE_STEPS", "200"))
    cfg = load_config()
    cfg.setdefault("environment", {})["seed"] = 12345

    print("[smoke] connecting to CARLA 127.0.0.1:%d ..." % port)
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(30.0)
    town = os.environ.get("SMOKE_TOWN")
    world = client.load_world(town) if town else client.get_world()
    print("[smoke] map=%s  running %d student-loop steps" % (world.get_map().name, steps))

    scene = SceneManager(world, cfg, client)
    policy = ExpertAdapter(scene)
    scorer = RolloutScorer(fps=10, max_lost_s=cfg["environment"].get("max_lost_seconds", 5))
    env = StudentRolloutEnv(scene, world, policy=policy, scorer=scorer,
                            fps=10, resolution=(512, 512))

    summary = env.run(max_steps=steps)
    print("\n[smoke] episode summary:")
    for k, v in summary.items():
        print("    %s: %s" % (k, v))

    if summary.get("skipped"):
        print("\n[smoke] SKIPPED (%s) — retry (traffic RNG); loop mechanics OK"
              % summary.get("reason"))
        return 0
    ok = (summary.get("steps", 0) >= steps - 1 and summary.get("track_frames", 0) > 0.3 * steps)
    print("\n[smoke] %s  (steps=%s track_frames=%s track_seconds=%.1f)" % (
        "PASS ✅ inverted loop runs; target tracked; scorer sane" if ok else "CHECK ⚠️",
        summary.get("steps"), summary.get("track_frames"), summary.get("track_seconds", 0)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
