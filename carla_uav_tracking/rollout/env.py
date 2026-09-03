"""Student-policy closed-loop rollout env (experiment-design §7 / training-plan Phase E.5).

A *fork* of `recording/recorder.py::EpisodeRecorder.run()` (NOT a reuse). Two
differences from the expert data-gen loop, neither expressible by a flag:

  (1) LOOP INVERSION. The expert picks its action from the GROUND-TRUTH target
      position, so data-gen computes the action BEFORE the frame is exposed
      (recorder.py:167 decide, :188 move, :197 tick, :198 capture). A VLA student's
      only input IS the frame, so it must capture FIRST, then decide. The
      tick/capture/act/move ordering is reversed (see the loop).

  (2) PRIVILEGED GT IS CONTROL-BYPASSED, SCORE-ONLY. get_target_state(),
      get_distractor_states(), _point_in_frame() feed the SCORER (and, for Option-A
      candidate boxes, the tid head's candidate SET) — never the controller. The
      expert's centroid framing / dead-reckoning / forced blind-pan are dropped.

This module is deliberately torch-free so it can run inside the CARLA container
(py3.6). The policy (torch/VLM, py3.11) is injected — for cross-container runs it is
a socket stub; for the CARLA-only smoke it is an expert adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from scene.scene_manager import SceneManager
from recording.recorder import EpisodeRecorder   # borrow _create_attached_sensor / _point_in_frame only


@dataclass
class Observation:
    """Everything the student policy is allowed to see. NO ground truth here."""
    rgb: np.ndarray          # (H, W, 3) uint8 — the ONLY perceptual input
    language: str            # fixed per episode (target identity)
    proprio: np.ndarray      # (5,) [cam_z/30, cam_pitch/90, vx/15, vy/15, vz/15]
    step: int                # tick index → drives the policy's S1/S2 scheduler
    target_color: str = ""   # GT target identity (color, blueprint) for the attrbind tid head —
    target_bp: str = ""      # privileged like the GT candidate boxes; = the offline attr upper bound


def _build_proprio(drone_state, cam_pitch: float) -> np.ndarray:
    """Replicate dataset_stage2.py:253 proprio (camera frame, 5-D — NOT 8)."""
    v = drone_state.velocity
    return np.array([
        drone_state.position[2] / 30.0,
        cam_pitch / 90.0,
        v[0] / 15.0, v[1] / 15.0, v[2] / 15.0,
    ], dtype=np.float32)


def _campose(cam_t) -> tuple:
    return (cam_t.location.x, cam_t.location.y, cam_t.location.z,
            cam_t.rotation.pitch, cam_t.rotation.yaw)


class StudentRolloutEnv:
    """Run one closed-loop episode driven by an injected policy (.reset()/.act())."""

    def __init__(self, scene: SceneManager, world, *, policy, scorer,
                 fps: int = 10, resolution=(512, 512), central_frac: float = 0.3,
                 fov_deg: float = 90.0, expert_control: bool = False,
                 predict_road: bool = False, road_oracle: bool = False,
                 waypoint_offsets_s=(1, 2, 4, 6)):
        self._scene = scene
        self._world = world
        self._policy = policy          # .reset(); .act(obs, gt_candidates) -> (action4, search_mode, info)
        # P2a-1: when True, compute the deployable road-graph target prediction env-side
        # (rollout.road_predictor, needs carla.Map) and ship it as gt_candidates[3] for the
        # policy's --ex-source road. Off by default → no CARLA-map cost for the other arms.
        self._predict_road = bool(predict_road)
        # road_oracle (P2a-2 positive control): update the road anchor from the TRUE CURRENT
        # target pos EVERY tick (not just visible) → traverse from a fresh anchor (loss_s≈0),
        # bounding road's value if the last-seen staleness + branch error were removed. Still
        # gated the same (applied only during a loss). If oracle-road doesn't beat CV, road is
        # not worth a learned which-way head; if it does, the gap is the learnable target.
        self._road_oracle = bool(road_oracle)
        self._road_offsets_s = list(waypoint_offsets_s)
        self._road = None
        # expert_control: still call the policy for tid SCORING (info.pred_slot), but drive
        # the drone with the EXPERT action (GT+PID, centers the target). Isolates "if framing
        # were perfect, does grounding/SR recover?" — tests the framing→grounding→SR chain.
        self._expert_control = bool(expert_control)
        self._scorer = scorer          # .reset(meta); .update(...); .should_truncate(); .finalize()
        self._fps = fps
        self._dt = 1.0 / fps
        self._resolution = resolution
        self._central_frac = central_frac
        self._fov = fov_deg
        self._rec = EpisodeRecorder.__new__(EpisodeRecorder)   # borrow pure helpers, don't run its loop
        self._rec._world = world
        self._rec._scene = scene
        self._rec._resolution = resolution
        self._rec._fov = fov_deg                               # _create_attached_sensor reads this

    def _target_central(self, cam_t, tpos) -> bool:
        """Target projects into the central `central_frac` box (inlined projection,
        same convention as EpisodeRecorder._point_in_frame — keeps env acot_probe-free)."""
        W, H = self._resolution
        M = np.array(cam_t.get_inverse_matrix())
        q = M @ np.array([float(tpos[0]), float(tpos[1]), float(tpos[2]), 1.0])
        if q[0] <= 0.1:
            return False
        f = W / (2.0 * np.tan(np.radians(self._fov) / 2.0))
        u = f * (q[1] / q[0]) + W / 2.0
        v = f * (-q[2] / q[0]) + H / 2.0
        m = self._central_frac
        return (0.5 - m / 2) * W <= u <= (0.5 + m / 2) * W and \
               (0.5 - m / 2) * H <= v <= (0.5 + m / 2) * H

    def run(self, max_steps: int | None = None) -> dict:
        metadata = self._scene.setup_episode()
        assert self._scene.drone is not None
        language = self._scene.episode_language

        cfg = self._scene._config
        max_duration = cfg.get("environment", {}).get("max_duration_seconds", 180)
        max_steps = max_steps or int(max_duration * self._fps)

        settings = self._world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._dt
        self._world.apply_settings(settings)

        # Warmup (recorder.py:97): fly the EXPERT to disperse traffic + confirm the
        # target moves. The scored episode starts after; warmup policy is irrelevant.
        max_speed = 0.0
        for _ in range(90):
            self._scene.step_target(0.1)
            tpos, _, _, _ = self._scene.get_target_state()
            ds = self._scene.drone.get_state()
            action, _ = self._scene.expert_action(tpos, ds.position, ds.yaw, True, 0.0)
            self._scene.step_drone(action)
            self._world.tick()
            _, _, _, ts = self._scene.get_target_state()
            max_speed = max(max_speed, ts)
        if max_speed < 1.2:
            return self._abort(original_settings, metadata, "target stationary")

        rgb_sensor = self._rec._create_attached_sensor()
        cam_pitch = float(metadata.get("camera_pitch", -50.0))
        self._policy.reset()
        self._scorer.reset(metadata)
        if self._predict_road:                         # P2a-1: env-side road-graph predictor
            from rollout.road_predictor import RoadPredictor
            self._road = RoadPredictor(self._world.get_map(), self._road_offsets_s, self._fps)
            self._road.reset()

        start = time.time()
        step = 0
        n_road = 0                                     # P2a-1: loss ticks that used a road prediction
        try:
            for step in range(max_steps):
                elapsed = step * self._dt

                # === LOOP INVERSION (cf. recorder.py:167-198) =================
                # Expert : decide → move → tick → capture.
                # Student: tick → capture → decide → move.
                # The camera sits at P(t) from the previous move, so this tick
                # exposes frame(t) for P(t) — the frame the policy acts on.
                # =============================================================
                self._scene.step_target(self._dt)
                self._world.tick()
                rgb = rgb_sensor.get_frame(timeout=0.5)               # frame(t) @ P(t)

                # GT — SCORING + candidate boxes ONLY, never fed to the controller.
                tpos, tvel, tyaw, _tspeed = self._scene.get_target_state()
                dstates = self._scene.get_distractor_states()          # (D,6)
                cam_t = self._scene.drone.get_camera_transform()
                campose = _campose(cam_t)
                drone_state = self._scene.drone.get_state()
                target_present = self._rec._point_in_frame(cam_t, tpos)
                target_central = self._target_central(cam_t, tpos)
                # Consumed only by the expert adapter's recovery framing (smoke);
                # the student policy never reads it (recorder.py:235 parity).
                self._scene._target_in_frame = bool(target_present)
                in_loss = self._scene.in_loss_window(elapsed)
                dist = float(np.linalg.norm(drone_state.position - np.asarray(tpos)))

                # P2a-1: deployable road-graph prediction (env-side, needs carla.Map). Update
                # the last-seen anchor while visible; traverse the lane graph during a loss.
                # Shipped as gt_candidates[3] (WORLD (K,3)); the policy projects it → z_ex.
                road_pred_world = None
                if self._road is not None:
                    if target_present or self._road_oracle:            # oracle: fresh anchor every tick
                        self._road.update(tpos, tvel, tyaw, step)
                    if not target_present and self._road.seen:         # apply only during a loss
                        road_pred_world = self._road.predict(step)     # (K,3) world or None
                        if road_pred_world is not None:
                            n_road += 1

                # STUDENT DECIDES from frame + language (+ its own proprio).
                obs = Observation(rgb=rgb, language=language,
                                  proprio=_build_proprio(drone_state, cam_pitch), step=step,
                                  target_color=str(metadata.get("target_color") or ""),
                                  target_bp=str(metadata.get("target_blueprint") or ""))
                action, _search_mode, info = self._policy.act(
                    obs, gt_candidates=(tpos, dstates, campose, road_pred_world))

                # expert-control probe: keep the policy's tid `info` for scoring, but drive
                # the drone with the EXPERT action (well-framed) instead of the DiT's.
                if self._expert_control:
                    action, _ = self._scene.expert_action(
                        tpos, drone_state.position, drone_state.yaw, True, elapsed)

                # SCORE this tick against GT (hysteresis mis-follow etc. inside).
                self._scorer.update(step=step, target_present=target_present,
                                    target_central=target_central, in_loss_window=in_loss,
                                    info=info, uav_target_dist=dist)

                # APPLY → move drone + camera to P(t+1).
                self._scene.step_drone(action)
                rgb_sensor._sensor.set_transform(self._scene.drone.get_camera_transform())

                if self._scorer.should_truncate():
                    break

        except Exception as e:  # noqa: BLE001 — mirror recorder's episode-level guard
            metadata["error"] = str(e)
        finally:
            try:
                rgb_sensor.destroy()
            except RuntimeError:
                pass
            self._scene.cleanup()
            try:
                self._world.tick()
            except RuntimeError:
                pass
            self._world.apply_settings(original_settings)

        summary = self._scorer.finalize()
        summary.update({"episode_seed": metadata.get("seed"), "steps": step,
                        "duration_seconds": time.time() - start, "language": language})
        if self._road is not None:
            summary["road_pred_ticks"] = n_road         # >0 confirms the road prior was live
            print(f"[env] road_pred used on {n_road} loss ticks", flush=True)
        if "error" in metadata:
            summary["error"] = metadata["error"]
        return summary

    def _abort(self, original_settings, metadata, reason: str) -> dict:
        self._scene.cleanup()
        try:
            self._world.tick()
        except RuntimeError:
            pass
        self._world.apply_settings(original_settings)
        return {"skipped": True, "reason": reason, "episode_seed": metadata.get("seed")}
