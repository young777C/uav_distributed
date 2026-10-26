"""OA-VAT FULL system as an end-to-end external baseline (acot_note/1020_oavat_full_baseline_plan.md; Sun et al., CVPR 2026,
github.com/SHWplus/OA-VAT @ cea1f6b, MIT) — perception, identity, re-acquisition AND control are OA-VAT's own.

The ORIGINAL code runs unmodified (imported from external/OA-VAT, never edited):
  main.start_mission                → first-frame YOLOE detection + DINOv3 match to the reference list, online enhancer,
                                      tracking loop, REDETECT restart (recursion), exactly as `python main.py`
  tracking/ortrack_core.py          → per-frame ORTrack, confidence-based loss, conf-aware PKF / KF, LostReacquireWorker
                                      (background thread), predict-mode hover, PID control (control/drone_control.py)
  detectors/, dinov3_feature_extractor.py, tracking/workers.py, kalman_filter.py, control/  (as is)
Configuration = the README's UnrealCV closed-loop evaluation command (README :118-131) with text_query "car" and the
frame size of our camera (512 × 512; the pipeline never resizes a non-Unreal source, control/drone_control.py reads
cfg['desired_width'/'desired_height'] as the frame size). DINOv3 = dinov3_vitb16 (the README's extraction / eval weights;
the code default vith16plus is not released with the repo). Diffusion-policy rescue off (no released checkpoint; also the
code default).

What this module adds (no OA-VAT logic):
  FrameBridge   the `video` (read → (ok, BGR)) and `vehicle` (send_velocity(vx, vy, vz, yaw_rate)) objects the original
                loop drives; step() hands one frame in and returns after the loop asks for the next one, with the command
                it sent for that frame (None = no command: the original sends nothing in predict mode / while detecting).
  reference     the per-episode reference feature list, built with the recipe of extract_target_feature_dinov3.py:main
                (YOLOE best-IoU instance mask of the given box → masked DINOv3 feature + horizontal flip → mean → L2) from
                the box the protocol's shared opening picked (OA-VAT: a box the user draws on reference images).
  capture       a wrapper around tracking.ortrack_core.save_follow_stream_frame (the original's per-frame logging hook)
                that records the stage (Track / Lost / Predict) and the box of every frame, for scoring only.
Process-level compatibility only: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (the repo's torch.load calls predate torch 2.6's
weights_only default; the original env is torch 2.2), stub modules for the repo's unused hardware / simulator imports
(DRONE.drone_controller is not in the repo; VIDEO.unrealcv_stream needs gym-unrealcv), the repo's prints silenced, and
the vendored dinov3-main/hubconf.py import fix shared with the identity adaptation (head_oavat.dinov3_extractor: torch.hub
routed to the same vendored backbone builder)."""
import builtins
import os
import queue
import sys
import threading
import types

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OAVAT = os.path.join(_REPO, "external", "OA-VAT")
PYDEPS = os.path.join(_REPO, "runs", "pydeps_oavat_full")
CACHE = os.path.join(_REPO, "runs", "oavat_full_cache")
W_YOLO = os.path.join(OAVAT, "weights", "yolo", "yoloe-11l-seg.pt")
W_ORTRACK = os.path.join(OAVAT, "weights", "ortrack", "ORTrack_ep0300.pth.tar")
W_DINO = os.path.join(OAVAT, "weights", "dinov3", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
# README :118-131 (UnrealCV evaluation) — the closest published closed-loop configuration
README_EVAL = ["--text_query", "car", "--yolo_confidence_thresh", "0.1", "--ortrack_lost_thresh", "0.5",
               "--reacq_match_thresh", "0.5", "--match_similarity_thresh", "0.4", "--online_enhance",
               "--kf_mode", "confaware"]
REF_YOLO_CONF = 0.3          # README :219 (reference extraction --confidence 0.3)
_SILENCED = ("main", "tracking.ortrack_core", "tracking.first_frame", "tracking.workers", "tracking.detection",
             "control.drone_control", "detectors.yolo_detector", "detectors.ortrack_tracker", "dinov3_feature_extractor",
             "tracking.reference_store", "tracking.visualization", "dp.dp_rescue", "dp.dp_inference")


def _quiet(*a, **k):
    pass


def compute_iou(box_a, box_b):
    """verbatim copy of extract_target_feature_dinov3.py:compute_iou (:96-113); that script cannot be imported headless
    (it switches matplotlib to TkAgg at import)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class _StopMission(BaseException):
    """1023 codex R1 M9: raised from read() after stop(); a BaseException so the repo's `except Exception` retry loops
    (first-frame detection, worker polling) cannot swallow it — the mission thread always unwinds."""


class FrameBridge:
    """`video` + `vehicle` of the original loop. Thread contract: the mission thread calls read() when it is done with the
    previous frame; step() (caller thread) feeds one frame and waits for that next read() (or the mission's end)."""

    def __init__(self):
        self.q = queue.Queue(maxsize=1)
        self.asked = threading.Event()          # the loop is waiting in read() for a frame
        self.cmd = None

    # video
    def read(self):
        self.asked.set()
        fr = self.q.get()
        if fr is None:
            raise _StopMission()
        return True, fr

    # vehicle (control/drone_control.py: send_velocity(vx=, vy=, vz=, yaw_rate=); the battery check is in a try)
    def send_velocity(self, vx=0, vy=0, vz=0, yaw_rate=0):
        self.cmd = (float(vx), float(vy), float(vz), float(yaw_rate))

    def land(self):
        pass

    def get_battery(self):
        return 100


class OavatFull:
    def __init__(self, device="cuda", frame_wh=(512, 512), work_dir=None, extra_args=(), require_text_prompt=True):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        os.environ.setdefault("TORCH_HOME", CACHE)
        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(CACHE, "ultralytics"))
        if OAVAT not in sys.path:
            sys.path.insert(0, OAVAT)
        if PYDEPS not in sys.path:
            sys.path.append(PYDEPS)
        for name, attrs in (("DRONE.drone_controller", {"TelloController": None}),
                            ("VIDEO.unrealcv_stream", {"UnrealCVStream": type("UnrealCVStream", (), {})})):
            if name not in sys.modules:
                m = types.ModuleType(name)
                m.__dict__.update(attrs)
                sys.modules[name] = m
        self.work_dir = work_dir or os.path.join(CACHE, "episodes")
        os.makedirs(self.work_dir, exist_ok=True)
        w, h = frame_wh
        argv = ["main.py", "--path_to_video", "acot-carla", "--desired_width", str(w), "--desired_height", str(h),
                "--yolo_model_path", W_YOLO, "--ortrack_checkpoint_path", W_ORTRACK,
                "--dinov3_model_name", "dinov3_vitb16", "--dinov3_checkpoint_path", W_DINO,
                "--reference_feature_path", os.path.join(self.work_dir, "unset.pt"), *README_EVAL, *extra_args]
        cwd, saved_argv = os.getcwd(), sys.argv
        os.chdir(OAVAT)                                   # mobileclip text model / relative repo paths
        sys.argv = argv
        try:
            import main as M                              # parses argv at import (as `python main.py …`)
        finally:
            sys.argv = saved_argv
        self.M = M
        import importlib
        for mod in _SILENCED:
            try:
                importlib.import_module(mod).__dict__["print"] = _quiet
            except Exception:
                pass
        import torch
        self.device = torch.device(device)
        cfg = vars(M.args)
        # == main.init_system (:195-249) for the three models, verbatim arguments; the video / Tello / Unreal branches
        # are replaced by the FrameBridge
        self.tracker = M.initialize_ortrack_tracker(
            ortrack_prj_path=cfg["ortrack_prj_path"], model_name=cfg["ortrack_model_name"],
            checkpoint_path=cfg["ortrack_checkpoint_path"], lost_threshold=cfg["ortrack_lost_thresh"],
            max_lost_frames=cfg["ortrack_max_lost_frames"], template_size=cfg["ortrack_template_size"],
            search_size=cfg["ortrack_search_size"], keep_last_position=True)
        os.chdir(os.path.join(OAVAT, "weights", "yolo"))  # ultralytics looks up mobileclip_blt.ts in the cwd
        try:
            self.detector = M.initialize_yolo_model(model_path=cfg["yolo_model_path"],
                                                    target_classes=cfg["text_query"].strip().split())
        finally:
            os.chdir(OAVAT)
        names = getattr(self.detector.model, "names", None)
        names = list(names.values()) if isinstance(names, dict) else list(names or [])
        # 1023: the same-perception variant replaces YOLOE by the protocol boxes, so the text prompt is not needed there
        if require_text_prompt and (self.detector.selected_class_ids is not None
                                    or names != cfg["text_query"].strip().split()):
            raise RuntimeError(f"YOLOE text prompt not applied (names {names}); the class-index fallback is not OA-VAT's "
                               f"evaluated configuration")
        from acot.runtime.head_oavat import dinov3_extractor   # the original extractor; vendored hubconf import fix
        self.extractor = dinov3_extractor(cfg["dinov3_model_name"], cfg["dinov3_checkpoint_path"], self.device)
        os.chdir(cwd)
        cfg["video_writer"] = cfg["video_writer_size"] = cfg["video_writer_fps"] = None
        self.cfg = cfg
        self._install_capture()
        self.thread = self.bridge = None
        self.ended = None

    # ------------------------------------------------------------------------------------------------- capture hook
    def _install_capture(self):
        import tracking.ortrack_core as core
        orig = core.save_follow_stream_frame
        self.last = {}

        def hook(cfg, frame_bgr, stage, **kw):
            self.last = {"stage": stage, "track_bbox": kw.get("track_bbox"), "predict_bbox": kw.get("predict_bbox"),
                         "is_lost": bool(kw.get("is_lost", False))}
            return orig(cfg, frame_bgr, stage, **kw)
        core.save_follow_stream_frame = hook

    # ------------------------------------------------------------------------------------------------- reference
    def build_reference(self, frame_bgr, box_xyxy, path):
        """extract_target_feature_dinov3.py:main (:196-326) for ONE image and the default augmentation (hflip only),
        with `bbox` = the given box instead of a hand-drawn one."""
        import cv2
        import torch
        import torch.nn.functional as F
        bbox = [int(round(v)) for v in box_xyxy]
        det_bbox, mask_bool, best_iou, from_yolo = list(bbox), None, 0.0, False
        with self.M.YOLO_LOCK:
            results = self.detector.model.predict(frame_bgr, conf=REF_YOLO_CONF, verbose=False)   # the same
            # YOLOE + "car" text prompt as the script's YOLO(path).set_classes(classes, get_text_pe(classes))
        if results and results[0].boxes:
            r = results[0]
            masks_np = None
            if getattr(r, "masks", None) is not None and getattr(r.masks, "data", None) is not None:
                masks_np = (r.masks.data > 0.5).detach().cpu().numpy().astype("uint8")
                H, W = frame_bgr.shape[:2]
                masks_np = [cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) if m.shape != (H, W) else m
                            for m in masks_np]
            best_idx, best_iou = -1, 0.0
            for idx, box in enumerate(r.boxes):
                iou = compute_iou(bbox, box.xyxy[0].cpu().numpy().astype(int).tolist())
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0:
                det_bbox = r.boxes[best_idx].xyxy[0].cpu().numpy().astype(int).tolist()
                if masks_np is not None and best_idx < len(masks_np):
                    mask_bool, from_yolo = masks_np[best_idx].astype(bool), True
        if mask_bool is None:                                    # the script's rectangle fallback
            H, W = frame_bgr.shape[:2]
            mask_bool = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                mask_bool[y1:y2, x1:x2] = True
        masked = frame_bgr.copy()
        masked[~mask_bool] = 0
        f_list = [F.normalize(self.extractor.extract_features(masked, np.array([det_bbox]))[0], dim=0)]
        masked_h, mask_h = cv2.flip(masked, 1), np.fliplr(mask_bool)
        ys, xs = np.where(mask_h)
        if len(xs) and len(ys):
            hb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            f_list.append(F.normalize(self.extractor.extract_features(masked_h, np.array([hb]))[0], dim=0))
        f = F.normalize(torch.stack(f_list, dim=0).mean(dim=0), dim=0)
        torch.save([f], path)                                    # feature_set = [f.squeeze(0) for f in all_features]
        return {"det_bbox": det_bbox, "yolo_mask": from_yolo, "iou_with_given_box": float(best_iou)}

    # ------------------------------------------------------------------------------------------------- episode
    def start(self, frame_bgr, box_xyxy, tag):
        """build the reference from (frame, box), then start main.start_mission on the bridge (it reads the NEXT
        frames itself: first-frame detection, tracking, REDETECT restarts)."""
        self.stop()
        if box_xyxy is None:                                    # 1024 codex R2: no reference inside the shared
            info = {"reference": None}                          # opening → OA-VAT's own no-reference first-frame
            self.cfg["reference_feature_path"] = None           # selection (first_frame.py default candidate)
        else:
            path = os.path.join(self.work_dir, f"ref_{tag}.pt")
            info = self.build_reference(frame_bgr, box_xyxy, path)
            self.cfg["reference_feature_path"] = path
        self.M.mission_counter = 0
        self.bridge = FrameBridge()
        self.ended = None
        self.last = {}

        def run():
            try:
                res = self.M.start_mission(self.device, self.cfg, self.tracker, self.detector, self.bridge, self.bridge,
                                           self.extractor)
                self.ended = (res or {}).get("status", "ENDED") if isinstance(res, dict) else "ENDED"
            except _StopMission:
                self.ended = "STOPPED"
            except Exception as e:                             # never leave the caller waiting
                self.ended = f"ERROR {type(e).__name__}: {e}"
            finally:
                self.bridge.asked.set()
        self.thread = threading.Thread(target=run, name="oavat-mission", daemon=True)
        self.thread.start()
        if not self.bridge.asked.wait(timeout=300):
            raise RuntimeError("OA-VAT mission did not ask for its first frame")
        return info

    def step(self, frame_bgr, timeout=300):
        """→ (command (vx, vy, vz, yaw_rate) | None, capture dict). After the mission ended: (None, {'stage': 'ENDED'})."""
        if self.thread is None or not self.thread.is_alive() or self.ended is not None:
            return None, {"stage": "ENDED", "status": self.ended}
        b = self.bridge
        b.asked.clear()
        b.cmd = None
        self.last = {}
        b.q.put(frame_bgr)
        if not b.asked.wait(timeout=timeout):
            raise RuntimeError("OA-VAT loop did not finish a frame within the timeout")
        cap = dict(self.last) if self.last else {"stage": "Detect" if self.ended is None else "ENDED"}
        if self.ended is not None:
            cap["status"] = self.ended
        return b.cmd, cap

    def stop(self):
        if self.thread is not None and self.thread.is_alive():
            try:
                self.bridge.q.put_nowait(None)                  # read() → (False, None): the loop ends its mission
            except queue.Full:
                pass
            self.thread.join(timeout=60)
            if self.thread.is_alive():                          # 1023 codex R1 M9: never reuse a live mission
                raise RuntimeError("OA-VAT mission thread did not stop")
        self.thread = None

