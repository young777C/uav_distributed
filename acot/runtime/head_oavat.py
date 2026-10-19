"""OA-VAT identity adaptation (1006 §5.6; Sun et al., CVPR 2026, github.com/SHWplus/OA-VAT, MIT) — the WHICH slot only.

Original code reused verbatim from external/OA-VAT (imported, not copied):
  tracking/detection.py::match_candidates_with_reference   boxes-only path (candidate_masks=None): one DINOv3 forward,
                                                           ROI-pooled patch tokens, cosine to the reference set
  tracking/reference_store.py::ReferenceFeatureStore      prototype = normalised mean of the reference views; EMA update
  kalman_filter.py::KalmanFilter                          DeepSORT xyah constant-velocity filter (dt = 1 tick here)
  tracking/kalman_utils.py::estimate_velocity_robustly    velocity from the recent centre history on loss
  dinov3_feature_extractor.py::DINOv3FeatureExtractor      (weights: dinov3_vitb16 torch.hub checkpoint)
What is replaced (identical for every arm): the detector (YOLOE) → the protocol's shared candidate boxes; ORTrack (box
continuity) → the shared tracker ids; the occlusion-aware diffusion planner / PID → the shared WHERE + DiT.
Reference = the shared attribute opening's pick (OA-VAT: user-drawn reference images): its 3× opening-frame view with
everything outside the box zeroed (the repo's rectangle-mask fallback) + horizontal flip, mean → one reference entry.
Per S2 tick (0.6 s; frame-unit defaults of the repo converted):
  locked id visible  → keep it (no identity check while tracking, as with ORTrack); KF update; every `enhance_every`
                       ticks (repo: 20 frames) EMA-enhance the prototype with the locked box (alpha = old weight)
  locked id absent   → lost: KF predict; candidates with IoU ≥ iou_pred to the predicted box (all after
                       `fullframe_after` failed attempts); re-lock the best if cosine ≥ reacq_thresh, else abstain;
                       after `max_predict_ticks` (repo: 15 frames) → REDETECT: prototype reset to the reference list,
                       search all candidates against the list with match_thresh
Output scores: cosine of every candidate to the prototype, the chosen slot raised by +1 (argmax = OA-VAT's choice,
c = top1 − top2 margin > 0); abstention → all −1e9."""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OAVAT = os.path.join(_REPO, "external", "OA-VAT")
# OA-VAT needs scipy (kalman_filter) and cv2 (tracking/detection); the acot policy image has neither, so they live in an
# isolated target dir (scipy 1.14.1, opencv-python-headless 4.10.0.84, pip --no-deps) APPENDED to sys.path — it never
# shadows an image package; the tracker's assignment is pinned explicitly by the perception (not by scipy presence).
PYDEPS = os.path.join(_REPO, "runs", "pydeps_oavat")


def _oavat():
    if OAVAT not in sys.path:
        sys.path.insert(0, OAVAT)
    if os.path.isdir(PYDEPS) and PYDEPS not in sys.path:
        sys.path.append(PYDEPS)
    from kalman_filter import KalmanFilter
    from tracking.detection import match_candidates_with_reference
    from tracking.kalman_utils import estimate_velocity_robustly
    from tracking.reference_store import ReferenceFeatureStore
    return KalmanFilter, match_candidates_with_reference, estimate_velocity_robustly, ReferenceFeatureStore


class _TimmDinov2:
    """DINOv2 ViT-S/14 (timm, the same pretrained weights as the shared crop features) behind the DINOv3 backbone's
    get_intermediate_layers(img, n=1, reshape=True) interface (last block, final norm, patch tokens as (B, C, H, W))."""

    def __init__(self, device):
        import timm
        self.m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).to(device).eval()

    def get_intermediate_layers(self, x, n=1, reshape=True):
        return self.m.get_intermediate_layers(x, n=n, reshape=reshape, norm=True)


def dinov2_extractor(device):
    """1006 backbone control: OA-VAT's ORIGINAL extractor class (preprocessing 518 / ImageNet norm, box scaling,
    ROI-align pooling) with the backbone swapped for DINOv2 ViT-S/14 (37×37 patch grid at 518 px)."""
    import torch.nn as nn
    from torchvision import transforms
    for p in (OAVAT,):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dinov3_feature_extractor import DINOv3FeatureExtractor
    ex = DINOv3FeatureExtractor.__new__(DINOv3FeatureExtractor)
    nn.Module.__init__(ex)
    ex.device, ex.model_name, ex.verbose = device, "dinov2_vits14", False
    ex.model = _TimmDinov2(device)
    ex.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize(518, antialias=True),
                                        transforms.CenterCrop(518),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return ex


def make_extractor(model_name, checkpoint, device):
    return dinov2_extractor(device) if model_name.startswith("dinov2") else \
        dinov3_extractor(model_name, checkpoint, device)


def dinov3_extractor(model_name, checkpoint, device):
    """the ORIGINAL DINOv3FeatureExtractor (strict state-dict load of the Meta torch.hub checkpoint). Its
    torch.hub.load(DINOV3_PATH, name, source='local') imports the vendored dinov3-main/hubconf.py, which fails on an
    import the vendored copy lacks (dinov3.eval.detection, only needed by the detector entries); hubconf merely
    re-exports dinov3.hub.backbones, so torch.hub.load is routed to the SAME vendored backbone builder for the duration
    of the constructor — model code = OA-VAT's own copy, nothing else changed."""
    import torch
    vend = os.path.join(OAVAT, "dinov3-main")
    for p in (OAVAT, vend):
        if p not in sys.path:
            sys.path.insert(0, p)
    import dinov3.hub.backbones as B
    from dinov3_feature_extractor import DINOv3FeatureExtractor
    orig = torch.hub.load

    def hub_load(repo, name, source=None, pretrained=False, **kw):
        assert os.path.realpath(repo).endswith("dinov3-main") and source == "local", (repo, source)
        return getattr(B, name)(pretrained=pretrained, **kw)
    torch.hub.load = hub_load
    try:
        return DINOv3FeatureExtractor(model_name=model_name, checkpoint_path=checkpoint, device=device)
    finally:
        torch.hub.load = orig


class OavatHead:
    name = "oavat"

    def __init__(self, extractor, match_thresh=0.75, reacq_thresh=0.70, iou_pred=0.2, fullframe_after=3,
                 max_predict_ticks=3, enhance_every=3, enhance_alpha=0.8, history=2):
        KF, self._match, self._vel, self._Store = _oavat()
        self.kf = KF()
        self.ex = extractor                    # .extract_features(bgr image, (n, 4) xyxy) → (n, C) torch
        self.match_thresh, self.reacq_thresh, self.iou_pred = float(match_thresh), float(reacq_thresh), float(iou_pred)
        self.fullframe_after, self.max_predict = int(fullframe_after), int(max_predict_ticks)
        self.enhance_every, self.enhance_alpha, self.hist_len = int(enhance_every), float(enhance_alpha), int(history)
        self.cfg = {}
        self.reset()

    def reset(self):
        self.refs, self.store = [], self._Store()
        self.locked, self.mean = None, None
        self.cov, self.hist = None, []
        self.lost, self.fails, self.redetect, self.tick = 0, 0, False, 0
        self.fresh = True
        self.last_sims, self.mode = None, "init"          # diagnostics for the trace (G9(b′)); no decision reads them

    # -------------------------------------------------------------------------------------------- features
    @staticmethod
    def _bgr(rgb):
        return np.ascontiguousarray(np.asarray(rgb, np.uint8)[..., ::-1])

    @staticmethod
    def _xyxy(d, f=1.0):
        return [(d.u - d.w / 2) * f, (d.v - d.h / 2) * f, (d.u + d.w / 2) * f, (d.v + d.h / 2) * f]

    def _reference_view(self, rgb, rgb_hi, d):
        """the repo's offline reference: rectangle-masked full image (+ horizontal flip), normalised mean."""
        import torch
        import torch.nn.functional as F
        src = rgb_hi if rgb_hi is not None else rgb
        f = np.asarray(src).shape[1] / float(np.asarray(rgb).shape[1])
        img = self._bgr(src)
        x1, y1, x2, y2 = [int(round(v)) for v in self._xyxy(d, f)]
        H, W = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        m = np.zeros(img.shape[:2], bool)
        m[y1:y2, x1:x2] = True
        masked = img.copy()
        masked[~m] = 0
        views = [F.normalize(self.ex.extract_features(masked, np.array([[x1, y1, x2, y2]]))[0].float().cpu(), dim=0)]
        fl = np.ascontiguousarray(masked[:, ::-1])
        views.append(F.normalize(self.ex.extract_features(fl, np.array([[W - x2, y1, W - x1, y2]]))[0].float().cpu(),
                                 dim=0))
        return F.normalize(torch.stack(views).mean(0), dim=0)

    def _sims(self, rgb, dets, refs, idx=None):
        """the repo's matcher (boxes-only) on a candidate subset → (full-length sims array, best slot, best sim)."""
        idx = list(range(len(dets))) if idx is None else idx
        boxes = [(dets[i].u - dets[i].w / 2, dets[i].v - dets[i].h / 2, dets[i].w, dets[i].h) for i in idx]
        bi, best, sims, _ = self._match(boxes, None, self._bgr(rgb), self.ex, refs, self.cfg)
        out = np.full(len(dets), -1.0)
        out[idx] = np.asarray(sims.detach().float().cpu().numpy(), float)
        self.last_sims = out
        return out, idx[bi], float(best)

    def _lock(self, d):
        self.locked = d.key
        meas = self.kf.xyxy_to_xyah(self._xyxy(d))
        self.mean, self.cov = self.kf.initiate(np.asarray(meas, float))
        self.hist = [tuple(meas)]
        self.lost = self.fails = 0
        self.redetect = False

    # ---------------------------------------------------------------------------------------------- per tick
    def score(self, dets, rgb, rgb_hi, init_slot):
        n = len(dets)
        self.tick += 1
        self.last_sims = None
        if self.fresh:                                         # hand-over: the opening pick is the reference
            if init_slot is None or not (0 <= init_slot < n):
                return np.full(n, -1e9, np.float32)
            self.refs = [self._reference_view(rgb, rgb_hi, dets[init_slot])]
            self.store.initialize_from_list(self.refs)
            self._lock(dets[init_slot])
            self.fresh = False
            self.mode = "init"
            return self._out(*self._sims(rgb, dets, [self.store.get_feature_vector()])[:1], init_slot)
        proto = [self.store.get_feature_vector()]
        slot = next((i for i, d in enumerate(dets) if d.key == self.locked), None) if self.locked is not None else None
        if slot is not None:                                   # tracking: keep the locked id, no identity check
            self.mode = "track"
            sims, _, _ = self._sims(rgb, dets, proto)
            meas = self.kf.xyxy_to_xyah(self._xyxy(dets[slot]))
            self.mean, self.cov = self.kf.predict(self.mean, self.cov)
            self.mean, self.cov = self.kf.update(self.mean, self.cov, np.asarray(meas, float))
            self.hist = (self.hist + [tuple(meas)])[-self.hist_len:]
            if self.enhance_every > 0 and self.tick % self.enhance_every == 0:
                import torch
                f = self.ex.extract_features(self._bgr(rgb), np.array([self._xyxy(dets[slot])]))[0]
                self.store.update_with_feature(torch.as_tensor(f), alpha=self.enhance_alpha)
            return self._out(sims, slot)
        # lost
        self.lost += 1
        if self.lost == 1 and self.hist:
            vx, vy, va, vh = self._vel(self.hist)
            self.mean, self.cov = self.kf.initiate(np.asarray(self.hist[-1], float), init_velocity=(vx, vy, va, vh))
        if self.mean is not None:
            self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        if n == 0:
            return np.full(0, -1e9, np.float32)
        if self.lost > self.max_predict and not self.redetect:  # REDETECT: drop online updates, search with the list
            self.store = self._Store()
            self.store.initialize_from_list(self.refs)
            self.redetect = True
        self.mode = "redetect" if self.redetect else "lost"
        if self.redetect:
            sims, bi, best = self._sims(rgb, dets, self.refs)
            if best > self.match_thresh:
                self._lock(dets[bi])
                return self._out(sims, bi)
            return np.full(n, -1e9, np.float32)
        idx = list(range(n))
        if self.fails < self.fullframe_after and self.mean is not None:
            ious = self.kf.compute_iou(self.mean[:4], [self._xyxy(d) for d in dets])
            near = [i for i in idx if ious[i] >= self.iou_pred]
            idx = near or idx
        sims, bi, best = self._sims(rgb, dets, proto, idx)
        if best >= self.reacq_thresh:
            self._lock(dets[bi])
            return self._out(sims, bi)
        self.fails += 1
        return np.full(n, -1e9, np.float32)

    @staticmethod
    def _out(sims, slot):
        z = np.asarray(sims, np.float32).copy()
        z[slot] = 1.0 + z[slot]
        return z

    def gate(self, slot, dets):
        """posterior gating (cross-memory use): re-lock the filter's MAP detection."""
        if 0 <= slot < len(dets):
            self._lock(dets[slot])

    def undo(self):
        pass
