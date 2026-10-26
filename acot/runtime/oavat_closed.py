"""1024 OA-VAT FULL system, SAME-PERCEPTION variant, CLOSED LOOP (acot_note/1023_oavat_closedloop_plan.md v2): OA-VAT's
own loop (acot/runtime/oavat_full.py → main.start_mission) flies the drone with its own PID controller in self-flight.

Selected by server.main when method.identity.head == oavat_full. Same interface as server.Runtime (reset / step /
cand_lite / policy.committed / info / dets), but no Qwen / Stage-2 / opening scorer is loaded:
  perception  the shared SimProxyPerception (protocol boxes + ray-cast occlusion + crop-DINOv2 TrackletAssigner keys,
              the same shuffle stream as every arm) at the fixed S2 cadence step % s2_period == 0, with a dummy VLM grid
              (the pooled grid feature only feeds language grounding, which this arm does not run)
  detector    YOLOE.detect_all → this frame's protocol boxes (frame_detections' rule, own stream) — GTDetector
  workers     the online enhancer / lost-reacquire worker bodies run inline at their original enqueue points
              (zero-latency; reacq_parallel stays True) — identical to the codex-R5-passed open-loop r5
  opening     on every S2 tick of the shared-opening window whose keys contain the target (= server.shared_slot of
              gt_target): OA-VAT restarted with the reference built from the target's box on that frame; the frame's
              first-frame selection is offered only that box with the similarity threshold lifted for that frame;
              the commitment is forced to the target key. Before the first such tick: OA-VAT's no-reference start.
  action      forward = cmd[1] (vy; vx and vz are always 0 in drone_control), yaw = cmd[3]; world displacement
              dx, dy = g_forward·forward·dt·(cos ψ, sin ψ), dz = 0, dyaw = g_yaw·yaw·dt; no command → hover (0, 0, 0, 0)
              (VIDEO/unrealcv_stream.read steps a stop action for every frame read)
  commitment  scoring readout only (the flight never reads it): at an S2 tick, a FRESH OA-VAT box of THIS frame (the
              (re)initialisation box, else the Track box) → the protocol candidate whose box contains its centre,
              nearest centre, ties → smallest key (identity-blind); no fresh box / no hit → the previous key is kept
              (no actor re-binding). Native immediate switching; the K = 5 variant is computed offline from the trace.
Labels (ScoringInfo) never reach OA-VAT: the only GT-derived inputs are the protocol boxes (same privilege as the
DeepSORT / OC-SORT arms) and the shared-opening hand-over."""
import os
import random

import numpy as np
import yaml

from acot import ROOT
from acot.runtime import support
from acot.runtime.io import Decision

DT = 0.1                                   # s per frame (10 Hz) = OA-VAT ctrl_dt


# ----------------------------------------------------------------------------------------------- open-loop r5 parts
class GTDetector:
    """verbatim runs/oavat_full/openloop_stress.py:GTDetector (replaces YOLODetector.detect_all)."""

    def __init__(self, W=512, H=512):
        self.by_key = {}
        self.W, self.H = float(W), float(H)

    @staticmethod
    def key(a):
        return hash(np.ascontiguousarray(a[::23, ::29]).tobytes())

    def register(self, frame, boxes):
        self.by_key = {self.key(frame): boxes}

    def detect_all(self, frame, confidence_threshold=0.25):
        full, ox, oy = frame, 0, 0
        if frame.base is not None and isinstance(frame.base, np.ndarray) and frame.base.ndim == 3 \
                and frame.shape != frame.base.shape:
            full = frame.base
            off = frame.__array_interface__["data"][0] - full.__array_interface__["data"][0]
            oy, ox = off // full.strides[0], (off % full.strides[0]) // full.strides[1]
        boxes = self.by_key.get(self.key(full))
        if boxes is None:
            boxes = next(iter(self.by_key.values()), [])
            sy, sx = frame.shape[0] / self.H, frame.shape[1] / self.W
            boxes = [(a, (x * sx, y * sy, w * sx, h * sy)) for a, (x, y, w, h) in boxes]
        H, W = frame.shape[:2]
        out = []
        for _, (x, y, w, h) in boxes:
            x1, y1, x2, y2 = max(x - ox, 0), max(y - oy, 0), min(x + w - ox, W), min(y + h - oy, H)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return out, None


def reset_tracker(oa):
    """ORTrack keeps its loss counters across initialize(); reset every per-episode field (open-loop r5)."""
    for obj in (oa.tracker, getattr(oa.tracker, "tracker", None)):
        if obj is None:
            continue
        for k, v in (("consecutive_lost_frames", 0), ("is_lost", False), ("state", None), ("frame_id", 0)):
            if hasattr(obj, k):
                setattr(obj, k, v)


def associate(dets, bbox):
    """identity-blind readout of a FRESH OA-VAT box (x, y, w, h) onto THIS tick's tracker keys (open-loop r5
    `associate`, on Detections instead of trk.jsonl rows): candidate boxes containing the centre, nearest centre, ties →
    smallest key. → (key | None, n containing boxes). Numerical note (codex R3 #4): r5 read the boxes from trk.jsonl,
    rounded to 0.1 px; here the full-precision detections are used — identity-blind either way, and the two-episode
    replay parity (§6.3) found no tick where this changes the key."""
    if bbox is None or not dets:
        return None, 0
    cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    hit = [((d.u - cx) ** 2 + (d.v - cy) ** 2, int(d.key)) for d in dets
           if abs(cx - d.u) <= d.w / 2 and abs(cy - d.v) <= d.w / 2]
    return (min(hit)[1] if hit else None), len(hit)


def map_action(cmd, yaw_deg, g_forward, g_yaw):
    """OA-VAT command (vx, vy, vz, yaw_rate) | None → our world-frame per-frame action (dx, dy, dz, dyaw)."""
    if cmd is None:
        return (0.0, 0.0, 0.0, 0.0)
    vx, fwd, vz, yr = (float(c) for c in cmd)
    if vx != 0.0 or vz != 0.0:
        raise RuntimeError(f"OA-VAT sent vx={vx} vz={vz}; drone_control always sends 0 on these channels")
    psi = np.radians(float(yaw_deg))
    s = g_forward * fwd * DT
    return (float(s * np.cos(psi)), float(s * np.sin(psi)), 0.0, float(g_yaw * yr * DT))


# ------------------------------------------------------------- 1024 occlusion-aware planner (diffusion policy)
DP_REPO = os.path.join(ROOT, "external", "OA-VAT", "gym-unrealcv", "diffusion_policy-main")
DP_PYDEPS = os.path.join(ROOT, "runs", "pydeps_oavat_dp")
DP_CKPT = os.path.join(ROOT, "external", "OA-VAT", "weights", "dp", "dp_unrealcv_20k_ep40.ckpt")


class SimClock:
    """the `time` the planner's rescue loop sees (dp/dp_rescue.py): it advances by one frame period per frame the loop
    reads, so `dp_dt` is SIMULATION time (1 s = 10 frames at 10 Hz) instead of wall-clock time (which would make the
    number of frames per planner step depend on the host's speed). sleep(s) advances it by s, as a real clock would —
    a no-op sleep could spin forever when float rounding makes the loop's first `now − last ≥ read_dt` test false."""

    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, float(s))

    def tick(self):
        self.t += DT * (1.0 + 1e-9)        # margin: ten float-summed 0.1 steps must reach dp_dt = 1.0 (not 0.999…)

    def reset(self):
        self.t = 0.0


def planner_cmd(cmd, dp_active, held_cmd):
    """→ (command to execute, new held command, held?). Inside a planner step (dp_rescue) its velocity persists over
    the frames of the step (Tello rc semantics: one send_velocity, then the frames read for dp_dt) and its vz is
    dropped — OA-VAT's own UnrealCV evaluation drives a planar tracker that receives only [vy, yaw_rate]
    (DRONE/unreal_sim_controller.py:57). Outside: unchanged (no command → hover, map_action)."""
    if not dp_active:
        return cmd, None, False
    held = False
    if cmd is not None:
        held_cmd = cmd
    elif held_cmd is not None:
        cmd, held = held_cmd, True
    if cmd is not None:
        cmd = (0.0, float(cmd[1]), 0.0, float(cmd[3]))
    return cmd, held_cmd, held


def is_dp_reinit(found_box, box):
    """codex R6 provenance: ortrack_core reads one more frame after a planner match and initialises ORTrack there with
    the match box → that initialisation is a re-init from the previous frame's observation, not a new one."""
    return found_box is not None and tuple(float(v) for v in box) == tuple(found_box)


def install_dp(rt):
    """enable OA-VAT's own planner path (ortrack_core: predict-stage timeout → dp_rescue) with the released checkpoint:
    process-level compatibility (dependencies in DP_PYDEPS; `wandb` — imported by the training workspace, unused at
    inference — stubbed), the simulation clock above, and a wrapper marking the rescue loop so the runtime holds the
    planner's velocity across the frames of one planner step (Tello rc semantics: a command persists until the next)."""
    import importlib.machinery
    import sys
    import types
    if DP_PYDEPS not in sys.path:
        sys.path.append(DP_PYDEPS)
    from acot.runtime.oavat_full import PYDEPS                    # cv2 etc. (OA-VAT deps)
    if PYDEPS not in sys.path:
        sys.path.append(PYDEPS)
    import huggingface_hub as _hh                          # diffusers 0.25.0 (OA-VAT's `follow` env pin) imports
    if not hasattr(_hh, "cached_download"):                # cached_download (removed from the image's hub); never
        _hh.cached_download = _hh.hf_hub_download          # called on the inference path (alias = import compat only)
    if "wandb" not in sys.modules:
        w = types.ModuleType("wandb")
        w.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        sys.modules["wandb"] = w
    import dp.dp_rescue as DR
    import dp.dp_inference as DI
    import tracking.ortrack_core as OC
    clock = SimClock()
    DR.time = clock                                       # the rescue loop's only clock (time.time / time.sleep)
    orig = OC.dp_rescue

    def rescue(*a, **k):
        rt.dp_active, rt.dp_calls = True, rt.dp_calls + 1
        try:
            out = orig(*a, **k)
        finally:
            rt.dp_active = False
        if out and out[0] and out[1] is not None:
            rt.dp_found += 1
            # provenance (codex R6): native ortrack_core reads ONE more frame and initialises ORTrack there with this
            # match box — that initialisation is not a new observation (the match was already read out on its frame)
            rt.dp_found_box = tuple(float(v) for v in out[1])
        return out
    OC.dp_rescue = rescue
    cfg = rt.oa.cfg
    cfg.update(dp_enable=True, dp_checkpoint=DP_CKPT, dp_repo_root=DP_REPO)
    rt.oa.M.preload_dp_if_needed(cfg)                     # main.init_system's preload (model + one warm-up call)
    if DI.get_dp_cache().get("policy") is None:
        raise RuntimeError("OA-VAT diffusion policy did not load (dependencies / checkpoint) — refusing to run the "
                           "planner-on arm with the planner silently off")
    return clock


# ------------------------------------------------------------------------------------------------------- runtime
class _PolicyShim:
    """what server.main reads from rt.policy."""

    def __init__(self, s2_period):
        self.s2_period = int(s2_period)
        self.committed = None
        self.head = None

    def is_s2(self, step):
        return int(step) % self.s2_period == 0


class OavatClosedRuntime:
    def __init__(self, arm, services, device, vlm=None, log=print):
        from train.reid_resolution_probe import _load_dino
        from acot.runtime.oavat_full import OavatFull
        from acot.runtime.perception import SimProxyPerception
        P, M = arm["protocol"], arm["method"]
        o = M["identity"]["oavat_full"]
        if M["control"].get("expert_control"):
            raise ValueError("oavat_full flies with its own controller (control.expert_control must be false)")
        fr = P["frontend"]
        mcfg = yaml.safe_load(open(os.path.join(ROOT, fr["vlm"]["model_config"])))
        self.image_cfg = mcfg["image"]
        dino = _load_dino(support.CROP_FEATURES[fr["crop_features"]["id"]], device)
        self.perception = SimProxyPerception(P, self.image_cfg, M["control"]["shuffle_seed"], dino, device)
        self.policy = _PolicyShim(P["sim"]["s2_period"])
        self.g_forward, self.g_yaw = float(o["g_forward"]), float(o["g_yaw"])
        sh = fr["opening"].get("shared")
        if not (sh and sh.get("gt_target")):
            raise ValueError("oavat_full implements the gt_target shared opening only (protocol sim_proxy_v2@4)")
        self.open_steps = int(fr["opening"]["window"]) * int(P["sim"]["s2_period"])
        W, H = self.image_cfg["width"], self.image_cfg["height"]
        self.oa = OavatFull(device=device, frame_wh=(W, H), require_text_prompt=False,
                            work_dir=os.path.join(ROOT, "runs", "oavat_full_cache", "closed", str(os.getpid())),
                            extra_args=("--target_area_ratio", str(float(o["target_area_ratio"])),
                                        "--area_tolerance_ratio", str(float(o["area_tolerance_ratio"]))))
        # declared physical limits (plan v3 §2): ctrl_max_speed (±40 command units, drone_control saturation) × gain
        cmax = float(self.oa.cfg["ctrl_max_speed"])
        if cmax * self.g_forward > float(o["max_speed_mps"]) + 1e-9 or cmax * self.g_yaw > float(o["max_yaw_dps"]) + 1e-9:
            raise ValueError(f"oavat_full gains exceed the declared limits: {cmax}·{self.g_forward} m/s, "
                             f"{cmax}·{self.g_yaw} deg/s vs {o['max_speed_mps']} m/s, {o['max_yaw_dps']} deg/s")
        self.gtd = self._make_sync(W, H)
        self.dp = bool(o.get("dp", False))
        self.dp_active, self.dp_calls, self.dp_found, self.dp_clock = False, 0, 0, None
        if self.dp:
            # the planner's yaw is clipped at dp_yaw_max (dp_inference.dp_action_to_tello_vel), not at ctrl_max_speed
            ymax = max(cmax, float(self.oa.cfg.get("dp_yaw_max", 60.0)))
            if ymax * self.g_yaw > float(o["max_yaw_dps_dp"]) + 1e-9:
                raise ValueError(f"planner yaw {ymax}·{self.g_yaw} deg/s exceeds max_yaw_dps_dp {o['max_yaw_dps_dp']}")
            self.dp_clock = install_dp(self)
        self.oracle = None                                       # deployable arm: no oracle channel (server echo / audit)
        self.info = self.dets = None
        self.seed = None
        self.frame_log = None
        import torch
        torch.backends.cudnn.deterministic = True                # plan v3 §1.2 (codex R3 #3)
        torch.backends.cudnn.benchmark = False
        self._reset_state()

    # -------------------------------------------------------------------------- hooks (open-loop r5 make_sync)
    def _make_sync(self, W, H):
        import detectors
        import torch
        import tracking.ortrack_core as OC
        import tracking.workers as Wk
        oa, rt = self.oa, self
        gtd = GTDetector(W, H)
        oa.detector.detect_all = gtd.detect_all
        oa.detector.model.predict = lambda *a, **k: []

        def inline(self_, f, b, i):
            try:
                with torch.autocast(device_type="cuda", enabled=False):
                    self_._process_one(f, tuple(b), int(i), detectors.perform_yolo_detection_for_candidates)
            except Exception:                                      # noqa: BLE001 — workers.py _run behaviour
                n = type(self_).__name__
                rt.worker_errors[n] = rt.worker_errors.get(n, 0) + 1
        for cls in (Wk.OnlineEnhancerWorker, Wk.LostReacquireWorker):
            cls.start = lambda self_: None
            cls.stop = lambda self_: None
            cls.enqueue = inline
        oa.cfg["reacq_parallel"] = True
        orig_init = oa.tracker.initialize

        def init_hook(image, info):
            out = orig_init(image, info)
            box = [float(v) for v in info.get("init_bbox")]
            if is_dp_reinit(getattr(rt, "dp_found_box", None), box):
                rt.dp_inits[rt.cur] = box                      # re-init from the planner match of the previous frame
                rt.dp_found_box = None
            else:
                rt.selected[rt.cur] = box
            return out
        oa.tracker.initialize = init_hook
        orig_ca = OC.compute_action

        def ca_hook(point, area, *a, **k):
            rt.ctrl[rt.cur] = {"cyx": [float(point[0]), float(point[1])], "area": float(area),
                               "predict": bool(k.get("is_predict", False))}
            return orig_ca(point, area, *a, **k)
        OC.compute_action = ca_hook
        return gtd

    # ------------------------------------------------------------------------------------------------ episode
    def _reset_state(self):
        self.selected, self.ctrl, self.worker_errors = {}, {}, {}
        self.dp_inits, self.dp_found_box = {}, None
        self.cur = -1
        self.refs = []
        self.committed = None

    def reset(self, seed=None, repeat=0):
        from acot.runtime.server import rng_base
        import torch
        self.oa.stop()
        if self.oa.thread is not None:
            raise RuntimeError("previous OA-VAT mission still alive")
        reset_tracker(self.oa)
        self._reset_state()
        self.seed = seed
        self.perception.reset()
        b = rng_base(seed, repeat)                                 # 64-bit H(seed, repeat) → valid for all three RNGs
        s = 0 if b is None else b & 0x7FFFFFFF
        torch.manual_seed(s)
        np.random.seed(s)
        random.seed(s)
        self.info = self.dets = None
        self.policy.committed = None
        self.frame_log = []
        self.dp_active, self.dp_calls, self.dp_found, self.dp_held = False, 0, 0, None
        if self.dp_clock is not None:
            self.dp_clock.reset()                                  # per-episode (seed-order independent)

    def _start(self, bgr, box_xywh, step):
        self.oa.stop()
        reset_tracker(self.oa)
        self.cur = step
        xyxy = None if box_xywh is None else [box_xywh[0], box_xywh[1], box_xywh[0] + box_xywh[2],
                                               box_xywh[1] + box_xywh[3]]
        info = self.oa.start(bgr, xyxy, f"s{self.seed}_t{step}")
        self.refs.append({"step": step, "info": info})

    def shared_slot(self, step):
        """server.Runtime.shared_slot for gt_target: the target's slot on an opening S2 tick, −1 if absent, else None."""
        if int(step) >= self.open_steps:
            return None
        if self.dets is None or self.info is None:
            return -1
        g = list(self.info.gt_idx)
        return g.index(-1) if -1 in g else -1

    def step(self, msg, gt=None):
        step = int(msg["step"])
        gt = msg["gt_candidates"] if gt is None else gt
        rgb = np.asarray(msg["rgb"])
        bgr = np.ascontiguousarray(rgb[..., ::-1])                 # RGB → BGR once (reference, detector, mission)
        s2 = self.policy.is_s2(step)
        if s2:
            dummy = np.zeros((1, 1, 1), np.float32)               # no VLM: the pooled grid feeds grounding only
            self.dets, self.info = self.perception.process(step, gt, dummy, rgb, msg.get("target_color", ""),
                                                           msg.get("target_bp", ""))
        # this frame's protocol boxes through the perception's own frame_detections (its frame stream; the GT audit
        # attributes the reads to the perception) — the boxes of open-loop r5's frame_boxes, without actor labels
        fd = self.perception.frame_detections(gt, rgb, feats=False) or ()
        boxes = [(None, (d.u - d.w / 2, d.v - d.h / 2, d.w, d.h)) for d in fd]
        self.cur = step
        self.gtd.register(bgr, boxes)
        slot = self.shared_slot(step) if s2 else None
        # shared-opening hand-over: the reference actor's box = its S2 detection (same projection, same frame)
        tb0 = None
        if slot is not None and 0 <= slot < len(self.dets or ()):
            d = self.dets[slot]
            tb0 = (d.u - d.w / 2, d.v - d.h / 2, d.w, d.h)
        reseed = tb0 is not None
        if reseed:
            self.gtd.register(bgr, [(None, tb0)])
            self._start(bgr, tb0, step)
        elif not self.refs:
            self._start(bgr, None, step)
        if self.dp_clock is not None:
            self.dp_clock.tick()                                   # one frame period of simulation time per frame
        thr = self.oa.cfg["match_similarity_thresh"]
        if reseed:
            self.oa.cfg["match_similarity_thresh"] = -1.0
        try:
            cmd, cap = self.oa.step(bgr)
        finally:
            self.oa.cfg["match_similarity_thresh"] = thr
        raw = cmd
        cmd, self.dp_held, held = planner_cmd(cmd, self.dp_active, self.dp_held)
        action = map_action(cmd, self._own_yaw(msg), self.g_forward, self.g_yaw)
        st = cap.get("stage")
        # fresh boxes: Track, and a planner match (DP_Found: the rescue's matching box on THIS frame; codex R5 #4)
        tb = cap.get("track_bbox") if st in ("Track", "DP_Found") else None
        sel = self.selected.get(step)
        fresh = sel if sel is not None else (None if tb is None else [float(v) for v in tb])
        self.frame_log.append({"t": step, "stage": "Select" if sel is not None else st,
                               "track_bbox": None if tb is None else [float(v) for v in tb],
                               "predict_bbox": None if cap.get("predict_bbox") is None
                               else [float(v) for v in cap["predict_bbox"]],
                               "selected": sel, "ctrl": self.ctrl.get(step),
                               "cmd": None if raw is None else [float(c) for c in raw],
                               "dp": self.dp_active, "dp_held": held, "dp_init": self.dp_inits.get(step),
                               "action": [round(a, 5) for a in action], "reseed": reseed, "status": cap.get("status")})
        trace = None
        if s2:
            trace = self._commit(step, fresh, slot, st)
        return Decision(action=action, search=0.0, pred_slot=-1, trace=trace)

    def _own_yaw(self, msg):
        """the drone's own yaw (deg): the own camera pose (x, y, z, pitch, yaw) every arm receives as PolicyInput.campose
        (deployable; server.Runtime reads it from the same unwrapped message field). Camera yaw = drone yaw (drone.py)."""
        return float(msg["gt_candidates"][2][4])

    def _commit(self, step, fresh, slot, stage):
        dets = self.dets or ()
        keys = [int(d.key) for d in dets]
        k, n_hit = associate(dets, fresh)
        tr = {"n": len(keys), "keys": keys, "stage": stage, "fresh": fresh is not None, "geo_key": k, "n_hit": n_hit,
              "oavat_ended": self.oa.ended}
        if slot is not None:
            tr["shared"] = int(slot)
        if slot is not None and 0 <= slot < len(keys):
            self.committed = keys[slot]
            tr["forced"] = True
        elif k is not None:
            self.committed = k
        self.policy.committed = self.committed
        tr.update(pred=-1, conf=None, committed=self.committed, search=0.0,
                  worker_errors=dict(self.worker_errors), missions=int(self.oa.M.mission_counter),
                  dp_calls=self.dp_calls, dp_found=self.dp_found)
        return tr

    def cand_lite(self):
        if self.dets is None:
            return None
        i = self.info
        return {"cands": [{"is_target": bool(t), "idx": int(d.key), "depth": float(d.depth_m), "gt_idx": int(g)}
                          for d, t, g in zip(self.dets, i.is_target, i.gt_idx)],
                "true_idx": int(i.true_idx)}

    def close(self):
        self.oa.stop()
