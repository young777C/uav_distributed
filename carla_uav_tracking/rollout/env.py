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

import os
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
    rgb_hi: np.ndarray | None = None   # 1004 opening fix: k× resolution frame of the SAME camera, opening ticks only
    retarget_event: bool = False   # 1003_dynamic-retarget-design.md §2.1: True on the exact tick
                                    # the scene's target was just reassigned (scripted, S1 GT-driven)


def _ray_visible(world, cam_loc, xyz, dz=(0.7, 1.4), start_off=1.0, margin=3.0) -> bool:
    """Occlusion check (1004 G13): is a vehicle at ground position `xyz` visible from the camera? Cast rays from the
    camera to points `dz` metres above the vehicle origin (body centre, roof); the vehicle counts as VISIBLE if at
    least one ray reaches within `margin` m of the point unobstructed (hits on the vehicle's own body lie inside the
    margin; buildings, vegetation and OTHER vehicles in front of it occlude). Partial visibility = visible."""
    import carla
    if xyz is None or not np.all(np.isfinite(np.asarray(xyz[:3], float))):
        return False
    c = np.array([cam_loc.x, cam_loc.y, cam_loc.z], float)
    for h in dz:
        p = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]) + h])
        d = float(np.linalg.norm(p - c))
        if d < start_off + margin:
            return True
        s0 = c + (p - c) / d * start_off
        hits = world.cast_ray(carla.Location(*s0), carla.Location(*p))
        if not any(float(np.linalg.norm(np.array([q.location.x, q.location.y, q.location.z]) - s0)) < d - start_off - margin
                   for q in hits):
            return True
    return False


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
                 expert_look_away: bool = True,
                 predict_road: bool = False, road_oracle: bool = False,
                 waypoint_offsets_s=(1, 2, 4, 6), vis_raycast: bool = False,
                 retarget_tick: int | None = None, hires_k: int = 1):
        self._scene = scene
        # 1004 G13: ray-cast occlusion flags for target + distractors (shipped to the policy as gt_candidates[4] for
        # occlusion-aware candidates; target visibility also goes to the scorer). Off = legacy frustum-only behaviour.
        self._vis_raycast = bool(vis_raycast)
        # 1004 opening fix: a second camera at hires_k× resolution (same pose / FOV) exists only for the opening ticks
        # (0, 6, 12) and is destroyed afterwards — the policy's zoom-and-verify crops candidates from it.
        self._hires_k = int(hires_k)
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
        # 1020 stress scenario: environment.loss_events.expert_mode == "look_away" → E3 group out-of-frame windows
        # under expert flight, and no target-anchored recovery framing (scene._no_target_recovery)
        _le = (scene._config.get("environment", {}).get("loss_events", {}) or {})
        _stress = bool(expert_control) and _le.get("enabled", False) and _le.get("expert_mode") == "look_away"
        self._look_away = _stress and bool(expert_look_away)
        if self._look_away:
            scene._no_target_recovery = True
            scene._centroid_radius_m = float(_le.get("centroid_radius_m", 25.0))
        elif _stress:
            # 1024 HOW ladder R2 (expert_look_away = False): the omniscient flight ceiling — the expert flies through
            # the E3 windows like every self-flight arm (the windows stay planned and annotated, in_loss_window is
            # unchanged), keeps the same 25 m look-alike centroid framing, and may servo the true target to recover it
            scene._centroid_radius_m = float(_le.get("centroid_radius_m", 25.0))
        # 1003_dynamic-retarget-design.md §2.2/§5 S1: scripted single-event retarget, GT-driven
        # (no language yet). At this tick, if a visible distractor exists, swap it in as the new
        # target and flag Observation.retarget_event=True for exactly that tick.
        self._retarget_tick = retarget_tick
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

    def _fire_retarget(self, cam_t, step: int) -> bool:
        """1003_dynamic-retarget-design.md §2.2 S1: pick a currently-visible distractor (similar
        ones preferred — the hard case) and swap it in as the target via SceneManager.retarget().
        GT-driven (no language): the chosen slot IS the new ground truth, so there is nothing to
        "get wrong" here — this only tests the retarget+override plumbing, not grounding."""
        dstates = self._scene.get_distractor_states()
        similar = self._scene.get_distractor_similar()
        order = sorted(range(len(dstates)), key=lambda d: (not bool(similar[d]) if d < len(similar) else True))
        for d in order:
            if not np.all(np.isfinite(dstates[d, :3])):
                continue
            if self._rec._point_in_frame(cam_t, dstates[d, :3]):
                return bool(self._scene.retarget(d, tick=step))
        return False      # no visible distractor at this tick — event does not fire (not retried in S1)

    def run(self, max_steps: int | None = None) -> dict:
        try:
            metadata = self._scene.setup_episode()
        except RuntimeError as e:          # e.g. "target spawn failed at 5 anchors" for this seed → skip, don't crash the run
            if "spawn failed" not in str(e):
                raise
            try:
                self._scene.cleanup()
            except Exception:
                pass
            return {"skipped": True, "reason": f"setup: {e}",
                    "episode_seed": getattr(self._scene, "_config", {}).get("environment", {}).get("seed")}
        if os.environ.get("DUMP_LABELS"):
            # 1004 opening fix: per-actor attribute LABELS for training a vehicle-model classifier on separate seeds
            # (offline only; never sent to the policy). Distractor order == dstates order == Candidate.gt_idx.
            import json as _json
            _d = os.environ["DUMP_LABELS"]; os.makedirs(_d, exist_ok=True)
            _json.dump({"seed": metadata.get("seed"), "target_bp": metadata.get("target_blueprint"),
                        "target_color": metadata.get("target_color"), "language": metadata.get("language"),
                        "distractors": [{"bp": m.get("bp"), "color": m.get("color"), "cls": m.get("cls")}
                                        for m in metadata.get("distractors", [])]},
                       open(os.path.join(_d, f"{metadata.get('seed')}.json"), "w"))
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
        hi_sensor = None
        if self._hires_k > 1:
            r0 = self._rec._resolution
            self._rec._resolution = (r0[0] * self._hires_k, r0[1] * self._hires_k)
            try:
                hi_sensor = self._rec._create_attached_sensor()
            finally:
                self._rec._resolution = r0
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
                # frame(t) @ P(t). 1020: 5 s (was 0.5 s): in synchronous mode a late frame is still frame(t) — under host
                # load the 0.5 s wait aborted whole episodes (queue.Empty) without changing any frame that did arrive.
                # 1022: 20 s — with 2 CARLA servers per GPU under host load ~130 the server still stalls > 5 s
                rgb = rgb_sensor.get_frame(timeout=20.0)
                rgb_hi = None
                if hi_sensor is not None:
                    rgb_hi = hi_sensor.get_frame(timeout=20.0)
                    if step >= 12:                                     # opening over → stop the extra render
                        hi_sensor.destroy(); hi_sensor = None
                    if step not in (0, 6, 12):
                        rgb_hi = None

                # GT — SCORING + candidate boxes ONLY, never fed to the controller.
                cam_t = self._scene.drone.get_camera_transform()
                retarget_event = False
                if self._retarget_tick is not None and step == self._retarget_tick:
                    retarget_event = self._fire_retarget(cam_t, step)
                tpos, tvel, tyaw, _tspeed = self._scene.get_target_state()
                dstates = self._scene.get_distractor_states()          # (D,6)
                campose = _campose(cam_t)
                drone_state = self._scene.drone.get_state()
                target_present = self._rec._point_in_frame(cam_t, tpos)
                vis = None; target_visible = None
                if self._vis_raycast:
                    t_vis = bool(target_present) and _ray_visible(self._world, cam_t.location, tpos)
                    d_vis = [bool(_ray_visible(self._world, cam_t.location, dstates[d, :3]))
                             for d in range(len(dstates))]
                    vis = (t_vis, d_vis); target_visible = t_vis
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
                                  target_bp=str(metadata.get("target_blueprint") or ""),
                                  retarget_event=retarget_event, rgb_hi=rgb_hi)
                action, _search_mode, info = self._policy.act(
                    obs, gt_candidates=(tpos, dstates, campose, road_pred_world, vis))

                # expert-control probe: keep the policy's tid `info` for scoring, but drive
                # the drone with the EXPERT action (well-framed) instead of the DiT's.
                if self._expert_control:
                    if self._look_away and in_loss:            # 1020 stress E3: the group leaves the frame
                        action = self._scene.expert_look_away(tpos, drone_state.position, drone_state.yaw, elapsed)
                    else:
                        if self._look_away:
                            self._scene.end_look_away()
                        action, _ = self._scene.expert_action(
                            tpos, drone_state.position, drone_state.yaw, True, elapsed)

                # SCORE this tick against GT (hysteresis mis-follow etc. inside).
                self._scorer.update(step=step, target_present=target_present,
                                    target_central=target_central, in_loss_window=in_loss,
                                    info=info, uav_target_dist=dist, target_visible=target_visible)

                # APPLY → move drone + camera to P(t+1).
                self._scene.step_drone(action)
                rgb_sensor._sensor.set_transform(self._scene.drone.get_camera_transform())
                if hi_sensor is not None:
                    hi_sensor._sensor.set_transform(self._scene.drone.get_camera_transform())

                if self._scorer.should_truncate():
                    break

        except Exception as e:  # noqa: BLE001 — mirror recorder's episode-level guard
            import traceback as _tb      # 1020: type + traceback tail (str(e) alone is empty for queue.Empty / asserts)
            metadata["error"] = "{}: {} | step {} | {}".format(type(e).__name__, e, step,
                                                              " / ".join(_tb.format_exc().splitlines()[-6:])[-900:])
        finally:
            try:
                rgb_sensor.destroy()
            except RuntimeError:
                pass
            if hi_sensor is not None:
                try:
                    hi_sensor.destroy()
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
        if self._retarget_tick is not None:
            summary["retarget_tick"] = self._retarget_tick
            summary["retarget_log"] = list(self._scene._retarget_log)   # [] if the event never found a visible distractor
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
