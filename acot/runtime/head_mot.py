"""DeepSORT / OC-SORT identity adaptations (1016) — two published multi-object trackers in the WHICH slot, the official
code imported unmodified (not copied) from external/ (1006 §5.6 porting rules):
  DeepSORT  Wojke, Bewley & Paulus, ICIP 2017 — github.com/nwojke/deep_sort @ f08cf1d (GPL-3.0)
            deep_sort.tracker.Tracker: KF (x, y, a, h) + chi² Mahalanobis gate, matching cascade on the nearest-neighbour
            cosine appearance metric (per-track gallery of up to nn_budget features), IoU matching of unconfirmed /
            just-missed tracks, n_init / max_age track management
  OC-SORT   Cao, Pang, Weng, Khirodkar & Kitani, CVPR 2023 — github.com/noahcao/OC_SORT @ 8462e7e (MIT)
            trackers/ocsort_tracker/ocsort.py::OCSort: KF (x, y, s, r) + observation-centric re-update (ORU), momentum
            (OCM) in the association cost, observation-centric recovery (OCR); motion only, no appearance
Single-target use of a multi-object tracker (tracking-by-detection): the tracker runs over ALL candidate boxes of every
S2 tick; the target is the track the opening pick was associated with, re-bound at every opening tick (1006 F-c);
afterwards the pick = the detection that track was associated with this tick (matched, any track state); the track not
associated this tick → abstain (all −1e9, the commitment holds); the track deleted → abstain for the rest of the episode
(no re-identification beyond the tracker's own association, as in the original systems).
What is replaced (identical for every arm): the detector → the protocol's shared candidate boxes (square pseudo-boxes,
score 1.0); DeepSORT's appearance CNN (MARS pedestrian re-ID, 128-d) → the shared crop-DINOv2 (384-d, L2-normalised).
Rate: every_frame = true (main) — one tracker step per camera frame (10 Hz; between S2 ticks the server's
SimProxyPerception.frame_detections gives the same boxes, advance()), the identity decision still at S2 ticks;
every_frame = false — one step per S2 tick (0.6 s; at that rate the IoU association fails, runs/id_iter/replay_mot.py).
A step happens on every frame / tick, also without candidates. Frame-unit repo defaults are converted to the step rate
in the method configs. The tracker state lives for the whole episode: the policy's per-episode reset calls
reset_episode(); reset() (the opening hand-over) only drops the binding."""
import importlib
import os
import sys
import types

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEEPSORT = os.path.join(_REPO, "external", "deep_sort", "deep_sort")
OCSORT = os.path.join(_REPO, "external", "OC_SORT", "trackers", "ocsort_tracker")
# deep_sort needs scipy (kalman_filter, linear_assignment); OC-SORT needs filterpy (kalmanfilter.py: filterpy.stats,
# which imports scipy) and uses lap if present, else scipy. The acot policy image has neither: isolated target dirs
# (pip --no-deps) — the OA-VAT port's (scipy 1.14.1) and runs/pydeps_mot (filterpy 1.4.5) — are APPENDED to sys.path,
# they never shadow an image package (see head_oavat.py).
PYDEPS = (os.path.join(_REPO, "runs", "pydeps_oavat"), os.path.join(_REPO, "runs", "pydeps_mot"))


def _package(name, path):
    """import the repo's tracker package under a private name (its relative imports resolve inside `path`; the repos'
    top-level names — tools, utils, trackers, … — never enter sys.path)."""
    if name not in sys.modules:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} missing (clone the official repo into external/, see the module docstring)")
        for p in PYDEPS:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
        pkg = types.ModuleType(name)
        pkg.__path__ = [path]
        sys.modules[name] = pkg
    return sys.modules[name]


def _deepsort():
    _package("_ext_deep_sort", DEEPSORT)
    nn = importlib.import_module("_ext_deep_sort.nn_matching")
    tr = importlib.import_module("_ext_deep_sort.tracker")
    det = importlib.import_module("_ext_deep_sort.detection")
    return nn, tr, det


def _ocsort(pkg="_ext_ocsort_tracker", record=True):
    """the repo's ocsort module under a private package name. record: its module-global KalmanBoxTracker (the name
    OCSort.update creates trackers with) is replaced by a subclass that only RECORDS (track, detection row) for every
    creation and every measurement update into KalmanBoxTracker.sink — the tracker logic itself is untouched.
    record=False gives a pristine copy (tests: state equality with the upstream code)."""
    _package(pkg, OCSORT)
    mod = importlib.import_module(f"{pkg}.ocsort")
    if record and not getattr(mod, "_acot_recording", False):
        base = mod.KalmanBoxTracker

        class KalmanBoxTracker(base):
            sink = None

            def __init__(self, bbox, delta_t=3, orig=False):
                super().__init__(bbox, delta_t=delta_t, orig=orig)
                if KalmanBoxTracker.sink is not None:
                    KalmanBoxTracker.sink.append((self, bbox))

            def update(self, bbox):
                if bbox is not None and KalmanBoxTracker.sink is not None:
                    KalmanBoxTracker.sink.append((self, bbox))
                super().update(bbox)

        setattr(mod, "KalmanBoxTracker", KalmanBoxTracker)
        setattr(mod, "_acot_recording", True)
    return mod


class _MotHead:
    """binding / pick logic shared by both trackers; subclasses implement _new_tracker() and _step(dets) → {det slot:
    track id} for every detection the tracker associated this tick (matched or newly initiated) and _alive() → ids."""

    def __init__(self):
        self.reset_episode()

    def _new_tracker(self):
        raise NotImplementedError

    def _step(self, dets):
        raise NotImplementedError

    def _alive(self):
        raise NotImplementedError

    def reset_episode(self):
        self.tracker = self._new_tracker()
        self.reset()

    def reset(self):
        self.bound = None                     # track id of the target
        self.lost = False                     # the bound track was deleted (no re-binding after the opening)
        self.last_assoc = {}

    def no_detections(self):
        """a tick without candidates: the tracker still steps (predict, no measurement) as the repos require."""
        self.last_assoc = self._step([])
        self._check_alive()

    def advance(self, dets):
        """every_frame: a camera frame between S2 ticks — tracker step only (no binding, no identity decision)."""
        if not dets:
            self.no_detections()
            return
        self.last_assoc = self._step(dets)
        self._check_alive()

    def _check_alive(self):
        if self.bound is not None and self.bound not in self._alive():
            self.bound, self.lost = None, True

    def score(self, dets, init_slot):
        """→ (n,) scores: 1 at the bound track's detection, 0 elsewhere; all −1e9 when the head abstains."""
        n = len(dets)
        assoc = self._step(dets)
        self.last_assoc = assoc
        if self.bound is None and not self.lost and init_slot is not None and 0 <= init_slot < n:
            self.bound = assoc.get(int(init_slot))           # every detection is associated (matched or new track)
        self._check_alive()
        out = np.full(n, -1e9, np.float32)
        if self.bound is None:
            return out
        slot = next((s for s, t in assoc.items() if t == self.bound), None)
        if slot is not None:
            out[:] = 0.0
            out[slot] = 1.0
        return out

    def gate(self, slot, dets):
        pass                                  # no memory to re-seed (PGIF is off for the external baselines)

    def undo(self):
        pass


class DeepSortHead(_MotHead):
    name = "deepsort"

    def __init__(self, max_cosine_distance, nn_budget, max_iou_distance, max_age, n_init, every_frame=False):
        self.every_frame = bool(every_frame)
        self.cfg = (float(max_cosine_distance), None if nn_budget is None else int(nn_budget), float(max_iou_distance),
                    int(max_age), int(n_init))
        nn, tr, det = _deepsort()
        self._Detection = det.Detection
        self._nn = nn

        class Tracker(tr.Tracker):
            """the repo's Tracker, unchanged; only RECORDS which detection each track was associated with."""

            def update(self, detections):
                self.assoc = {}
                self._dets = detections
                super().update(detections)

            def _match(self, detections):
                matches, ut, ud = super()._match(detections)
                for ti, di in matches:
                    self.assoc[int(di)] = int(self.tracks[ti].track_id)
                return matches, ut, ud

            def _initiate_track(self, detection):
                di = next(i for i, d in enumerate(self._dets) if d is detection)
                self.assoc[di] = int(self._next_id)
                super()._initiate_track(detection)

        self._Tracker = Tracker
        super().__init__()

    def _new_tracker(self):
        cos, budget, iou, age, n_init = self.cfg
        metric = self._nn.NearestNeighborDistanceMetric("cosine", cos, budget)
        return self._Tracker(metric, max_iou_distance=iou, max_age=age, n_init=n_init)

    def _step(self, dets):
        D = [self._Detection(np.array([d.u - d.w / 2.0, d.v - d.h / 2.0, d.w, d.h], np.float64), 1.0,
                             np.asarray(d.crop_feat, np.float64)) for d in dets]
        self.tracker.predict()
        self.tracker.update(D)
        return dict(self.tracker.assoc)

    def _alive(self):
        return {int(t.track_id) for t in self.tracker.tracks}


class OcSortHead(_MotHead):
    name = "ocsort"

    def __init__(self, det_thresh, max_age, min_hits, iou_threshold, delta_t, asso_func, inertia, use_byte, W, H,
                 every_frame=False):
        self.every_frame = bool(every_frame)
        self.cfg = dict(det_thresh=float(det_thresh), max_age=int(max_age), min_hits=int(min_hits),
                        iou_threshold=float(iou_threshold), delta_t=int(delta_t), asso_func=str(asso_func),
                        inertia=float(inertia), use_byte=bool(use_byte))
        self.img = (int(H), int(W))
        self._mod = _ocsort()
        super().__init__()

    def _new_tracker(self):
        return self._mod.OCSort(**self.cfg)               # (resets the class-level KalmanBoxTracker id counter)

    def _step(self, dets):
        """OCSort.update returns boxes + ids of the tracks it REPORTS (hit streak ≥ min_hits); the association itself is
        read from the recording KalmanBoxTracker (_ocsort): every creation / measurement update receives a row view
        dets[i, :] of the update's detection array; with every detection above det_thresh that array has the input rows
        in input order, so the view's row offset is the slot (checked: the row equals the input row)."""
        arr = np.array([[d.u - d.w / 2.0, d.v - d.h / 2.0, d.u + d.w / 2.0, d.v + d.h / 2.0, 1.0] for d in dets],
                       np.float64).reshape(-1, 5)
        if not np.all(arr[:, 4] > self.cfg["det_thresh"]):
            raise RuntimeError("OC-SORT adaptation: every detection must pass det_thresh (row read-back)")
        K = self._mod.KalmanBoxTracker
        K.sink = sink = []
        try:
            self.tracker.update(arr.copy(), self.img, self.img)   # img_info = img_size → the repo's box scale = 1
        finally:
            K.sink = None
        assoc = {}
        for trk, row in sink:
            base = row.base
            if base is None or base.ndim != 2 or base.shape[0] != len(arr) or row.ndim != 1:
                raise RuntimeError("OC-SORT association read-back: detection row is not a row view of the update's array")
            off = row.__array_interface__["data"][0] - base.__array_interface__["data"][0]
            s, rem = divmod(off, base.strides[0])
            if rem or not 0 <= s < len(arr) or not np.array_equal(row, arr[s]) or s in assoc:
                raise RuntimeError(f"OC-SORT association read-back failed (slot {s}, offset {off})")
            assoc[int(s)] = int(trk.id)
        return assoc

    def _alive(self):
        return {int(t.id) for t in self.tracker.trackers}
