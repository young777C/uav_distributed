"""1016 DeepSORT / OC-SORT identity adaptations (acot/runtime/head_mot.py, SimProxyPerception.frame_detections):
(a) port consistency (1006 §5.6 a): the head's tracker evolves exactly as the repo's tracker fed the same detections
    (track ids, KF states, track management) — the head only reads the association back;
(b) the association read-back is right (crossing-free synthetic scene with known identities);
(c) binding / pick / abstain / deletion / opening re-binding / ticks without candidates;
(d) every-frame stepping and the frame boxes (same boxes and occlusion as the S2 tick, S2 stream untouched)."""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from acot.runtime import head_mot as M

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "carla_uav_tracking"))

pytestmark = pytest.mark.skipif(not (__import__("os").path.isdir(M.DEEPSORT) and __import__("os").path.isdir(M.OCSORT)),
                                reason="external/deep_sort or external/OC_SORT not cloned")
DS = dict(max_cosine_distance=0.2, nn_budget=100, max_iou_distance=0.7, max_age=2, n_init=3)
OC = dict(det_thresh=0.6, max_age=2, min_hits=3, iou_threshold=0.3, delta_t=1, asso_func="iou", inertia=0.2,
          use_byte=False, W=512, H=512)


def feat(i):
    f = np.zeros(384)
    f[i] = 1.0
    return f


def scene(T=12, n=3, seed=0, drop=()):
    """n cars on separate rows, slow constant velocity; detections shuffled per tick; (t, car) in drop = missed."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(T):
        cars = [(c, 60.0 + 3.0 * t + 40 * c, 80.0 + 120.0 * c) for c in range(n) if (t, c) not in drop]
        rng.shuffle(cars)
        out.append(([NS(u=u, v=v, w=24.0, h=24.0, crop_feat=feat(c)) for c, u, v in cars], [c for c, _, _ in cars]))
    return out


def test_deepsort_state_matches_repo_tracker():
    nn, tr, det = M._deepsort()
    h = M.DeepSortHead(**DS)
    ref = tr.Tracker(nn.NearestNeighborDistanceMetric("cosine", 0.2, 100), max_iou_distance=0.7, max_age=2, n_init=3)
    for dets, _ in scene(drop={(4, 1), (5, 1), (6, 1), (7, 1)}):
        h._step(dets)
        ref.predict()
        ref.update([det.Detection(np.array([d.u - 12, d.v - 12, 24, 24.0]), 1.0, d.crop_feat) for d in dets])
        a, b = h.tracker.tracks, ref.tracks
        assert [(t.track_id, t.state, t.time_since_update, t.hits) for t in a] == \
               [(t.track_id, t.state, t.time_since_update, t.hits) for t in b]
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.mean, y.mean)
            np.testing.assert_array_equal(x.covariance, y.covariance)


def test_ocsort_state_matches_repo_tracker():
    mod = M._ocsort("_ext_ocsort_ref", record=False)                # pristine upstream copy (no recording subclass)
    assert not getattr(mod, "_acot_recording", False) and mod.KalmanBoxTracker is not M._ocsort().KalmanBoxTracker
    h = M.OcSortHead(**OC)
    hs = []
    for dets, _ in scene(drop={(4, 1), (5, 1)}):
        h._step(dets)
        hs.append([(t.id, t.time_since_update, t.hit_streak, t.kf.x.copy()) for t in h.tracker.trackers])
    ref = mod.OCSort(**{k: v for k, v in OC.items() if k not in ("W", "H")})
    for k, (dets, _) in enumerate(scene(drop={(4, 1), (5, 1)})):
        arr = np.array([[d.u - 12, d.v - 12, d.u + 12, d.v + 12, 1.0] for d in dets]).reshape(-1, 5)
        ref.update(arr, (512, 512), (512, 512))
        rs = [(t.id, t.time_since_update, t.hit_streak, t.kf.x) for t in ref.trackers]
        assert [x[:3] for x in hs[k]] == [x[:3] for x in rs]
        for x, y in zip(hs[k], rs):
            np.testing.assert_array_equal(x[3], y[3])


@pytest.mark.parametrize("cls,kw", [(M.DeepSortHead, DS), (M.OcSortHead, OC)])
def test_readback_follows_identity(cls, kw):
    h = cls(**kw)
    car_of_track = {}
    for dets, cars in scene(T=15, drop={(6, 2)}):
        assoc = h._step(dets)
        assert sorted(assoc) == list(range(len(dets)))            # every detection associated (matched or new)
        for s, t in assoc.items():
            assert car_of_track.setdefault(t, cars[s]) == cars[s]  # a track id never changes car


@pytest.mark.parametrize("cls,kw", [(M.DeepSortHead, DS), (M.OcSortHead, OC)])
def test_bind_pick_abstain_delete_rebind(cls, kw):
    h = cls(**kw)
    sc = scene(T=14, drop={(5, 0), (8, 0), (9, 0), (10, 0), (11, 0)})
    dets, cars = sc[0]
    z = h.score(dets, init_slot=cars.index(0))                     # opening pick = car 0
    assert int(np.argmax(z)) == cars.index(0) and np.sort(z)[-1] - np.sort(z)[-2] == 1.0
    for t in range(1, 5):
        dets, cars = sc[t]
        z = h.score(dets, init_slot=cars.index(1))                 # later init_slot ignored while bound
        assert int(np.argmax(z)) == cars.index(0)
    dets, cars = sc[5]                                             # car 0 missed → abstain
    assert float(h.score(dets, init_slot=0).max()) < -1e8
    dets, cars = sc[6]                                             # back within max_age → same track
    assert int(np.argmax(h.score(dets, init_slot=0))) == cars.index(0)
    h.score(sc[7][0], init_slot=0)
    for t in (8, 9, 10):                                           # 3 misses > max_age 2 → deleted → lost
        h.score(sc[t][0], init_slot=0)
    assert h.lost and h.bound is None
    dets, cars = sc[12]                                            # car 0 back: new track, no re-binding
    assert float(h.score(dets, init_slot=cars.index(0)).max()) < -1e8
    h.reset()                                                      # opening hand-over → re-bind to the pick
    dets, cars = sc[13]
    assert int(np.argmax(h.score(dets, init_slot=cars.index(2)))) == cars.index(2)


@pytest.mark.parametrize("cls,kw", [(M.DeepSortHead, DS), (M.OcSortHead, OC)])
def test_empty_ticks_age_tracks(cls, kw):
    h = cls(**kw)
    dets, cars = scene(T=1)[0]
    h.score(dets, init_slot=0)
    for _ in range(3):
        h.no_detections()
    assert h.lost and not h._alive()
    h.reset_episode()
    assert h.bound is None and not h.lost and not h._alive()


@pytest.mark.parametrize("cls,kw", [(M.DeepSortHead, DS), (M.OcSortHead, OC)])
def test_every_frame_advance(cls, kw):
    """every_frame: advance() steps the tracker on the frames between S2 ticks (no binding there); the S2 pick follows
    the bound track through them; advance(None) = a frame without candidates."""
    h = cls(**dict(kw, every_frame=True, max_age=10))
    sc = scene(T=13)
    dets, cars = sc[0]
    assert h.advance(dets) is None and h.bound is None            # a frame never binds
    h.reset()
    h.score(sc[1][0], init_slot=sc[1][1].index(1))
    for t in range(2, 12):
        h.advance(sc[t][0] if t != 7 else None)
    dets, cars = sc[12]
    assert int(np.argmax(h.score(dets, init_slot=0))) == cars.index(1) and not h.lost


def _perception():
    import rollout.tracklets as T
    from acot.runtime.perception import SimProxyPerception
    p = SimProxyPerception.__new__(SimProxyPerception)
    p.image_cfg = {"width": 512, "height": 512, "fov_deg": 90.0, "vehicle_size_m": 4.0, "box_min_px": 10.0,
                   "box_max_px": 140.0}
    p.occlusion, p.shuffle_seed = True, 0
    T._lsa = None
    p.tracker = T.TrackletAssigner(max_age=8, gate_px=150.0, gate_box=2.0, w_app=0.85, min_cos=0.2, vel_beta=0.5,
                                   app_alpha=0.3)
    p.crop_feats = lambda rgb, boxes: np.stack([feat(int(b.u) % 384) for b in boxes])
    p.reset()
    return p


def _gt(t):
    tpos = np.array([30.0, 0.0 + 0.1 * t, 0.0])
    ds = np.array([[30.0, 6.0, 0.0, 0, 0, 0], [35.0, -6.0, 0.0, 0, 0, 0], [40.0, 3.0, 0.0, 0, 0, 0],
                   [30.0, -100.0, 0.0, 0, 0, 0]])                 # the last one is out of frame
    return (tpos, ds, (0.0, 0.0, 0.0, 0.0, 0.0), None, (True, [True, False, True, True]))   # distractor 1 occluded


def test_frame_detections_same_boxes_own_stream():
    a, b = _perception(), _perception()
    grid = np.zeros((4, 4, 8), np.float32)
    for t in range(4):
        da, ia = a.process(t * 6, _gt(t), grid, None)
        fa = a.frame_detections(_gt(t), None, feats=True)                 # interleaved frame calls (a only)
        fo = a.frame_detections(_gt(t), None, feats=False)
        db, ib = b.process(t * 6, _gt(t), grid, None)
        assert [(d.key, d.u, d.v) for d in da] == [(d.key, d.u, d.v) for d in db]   # S2 shuffle / tracker untouched
        assert ia.gt_idx == ib.gt_idx
        assert sorted((d.u, d.v, d.w) for d in fa) == sorted((d.u, d.v, d.w) for d in da)   # same boxes, occlusion
        assert len(fa) == 3 and all(d.key == -1 for d in fa) and fa[0].crop_feat is not None
        assert fo[0].crop_feat is None


def _upstream_matches(dets_seq):
    """pristine OC-SORT on the same detections, its association INSTRUMENTED: every KalmanBoxTracker creation / update
    of a fresh copy (own package name) is logged as (track id, detection row value) → per step {row tuple: id}."""
    mod = M._ocsort("_ext_ocsort_probe", record=False)
    base, log = mod.KalmanBoxTracker, []

    class Probe(base):
        def __init__(self, bbox, delta_t=3, orig=False):
            super().__init__(bbox, delta_t=delta_t, orig=orig)
            log.append((self, np.array(bbox)))

        def update(self, bbox):
            if bbox is not None:
                log.append((self, np.array(bbox)))
            super().update(bbox)

    mod.KalmanBoxTracker = Probe
    try:
        t = mod.OCSort(**{k: v for k, v in OC.items() if k not in ("W", "H")})
        out = []
        for dets in dets_seq:
            arr = np.array([[d.u - d.w / 2, d.v - d.h / 2, d.u + d.w / 2, d.v + d.h / 2, 1.0] for d in dets]).reshape(-1, 5)
            del log[:]
            t.update(arr.copy(), (512, 512), (512, 512))
            out.append([(trk.id, row) for trk, row in log])
        return out
    finally:
        mod.KalmanBoxTracker = base


def box(u, v=100.0):
    return NS(u=u, v=v, w=24.0, h=24.0, crop_feat=feat(0))


def test_ocsort_new_track_order_codex_r1():
    """codex R1 MAJOR: a detection Hungarian-matched but rejected by the IoU filter is appended AFTER the initially
    unmatched ones, so new ids are not in ascending slot order (track 1 ← u=300, track 2 ← u=116)."""
    seq = [[box(100.0)], [box(116.0), box(300.0)]]
    h = M.OcSortHead(**OC)
    got = [h._step(d) for d in seq]
    up = _upstream_matches(seq)
    for dets, a, u in zip(seq, got, up):
        for tid, row in u:
            s = next(i for i, d in enumerate(dets) if abs(d.u - (row[0] + row[2]) / 2) < 1e-9)
            assert a[s] == tid
    assert got[1] == {0: 2, 1: 1}


def test_ocsort_duplicate_boxes_codex_r1():
    """codex R1 MINOR: two identical boxes — the read-back follows the upstream assignment, not the first equal row.
    Expected slots come from the pristine upstream's own association indices (its `associate` output + OCR rematch are
    replayed through an instrumented copy of the module-global functions)."""
    seq = [[box(110.0), box(100.0)], [box(110.0), box(100.0)], [box(100.0), box(100.0)]]
    h = M.OcSortHead(**OC)
    got = [h._step(d) for d in seq]
    mod = M._ocsort("_ext_ocsort_assoc", record=False)
    t = mod.OCSort(**{k: v for k, v in OC.items() if k not in ("W", "H")})
    orig, calls = mod.associate, []

    def associate(dets, trks, *a, **k):
        r = orig(dets, trks, *a, **k)
        calls.append({int(d): int(t.trackers[int(j)].id) for d, j in r[0]})
        return r

    mod.associate = associate
    try:
        for dets in seq:
            arr = np.array([[d.u - 12, d.v - 12, d.u + 12, d.v + 12, 1.0] for d in dets])
            n_before = len(t.trackers)
            t.update(arr.copy(), (512, 512), (512, 512))
    finally:
        mod.associate = orig
    expected = calls[-1]                                       # last step: both duplicates matched in the first round
    assert len(expected) == 2 and n_before == 2 and len(t.trackers) == 2
    assert got[2] == expected


def test_policy_mot_abstention_not_confident():
    """codex R1 MINOR: a single candidate + abstention → pred_slot −1 AND pred_conf 0 (never a confident tick)."""
    import torch
    from acot.runtime.policy import AcotPolicy
    p = AcotPolicy.__new__(AcotPolicy)
    p.kind, p.pgif, p.opening, p.open_window, p.open_k = "ocsort", None, None, 1, 0
    p.head = M.OcSortHead(**OC)
    p.frame_input, p.mot_frame_ms, p.device, p.committed = None, 0.0, "cpu", None
    p.fe = NS(tid=lambda cf, ctx, cm, mask: torch.zeros(1, cf.shape[1]))
    p.vlm_ctx = p.ctx_mask = None
    one = lambda u: NS(key=7, u=u, v=100.0, w=24.0, h=24.0, vlm_feat=np.zeros(4, np.float32), crop_feat=feat(0))
    inp = NS(shared_open_slot=None)
    p._identity(inp, (one(100.0),), None)
    assert p.pred_slot == 0 and p.pred_conf == 1.0
    p.head.lost, p.head.bound = True, None                      # e.g. after deletion
    tr = p._identity(inp, (one(101.0),), None)
    assert p.pred_slot == -1 and p.pred_conf == 0.0 and tr["mot_bound"] is None
    tr = p._identity(inp, None, None)                            # empty S2 tick: tracker step + trace fields
    assert tr["n"] == 0 and "t_identity_ms" in tr and "mot_bound" in tr
