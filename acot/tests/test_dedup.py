"""P3b de-duplication: the legacy copies of the geometry and of the instruction parser equal the canonical
implementations in acot/data; every crop-DINO call site uses the parameters of its feature id."""
import glob

import h5py
import numpy as np

from acot.data import crops, geometry, language

IMG = {"width": 512, "height": 512, "fov_deg": 90.0, "vehicle_size_m": 4.0, "box_min_px": 10.0, "box_max_px": 140.0}


def test_pseudo_box_copies_agree():
    from acot_probe.projection import _pseudo_box as p2
    from recording import postprocess as pp
    rng = np.random.default_rng(0)
    for d in rng.uniform(0.05, 300, 500):
        for fov in (70.0, 90.0):
            ref = geometry.pseudo_box(d, 512, fov, 4.0, 10.0, 140.0)
            assert p2(d, 512, fov, 4.0, 10.0, 140.0) == ref
            assert pp._pseudo_box(d, 512, fov) == geometry.pseudo_box(d, 512, fov, pp._VEHICLE_SIZE_M, pp._BOX_MIN_PX,
                                                                         pp._BOX_MAX_PX)
    # the one documented discrepancy: the recorder's label boxes clip at 8 px, the model config at 10 px
    assert (pp._VEHICLE_SIZE_M, pp._BOX_MIN_PX, pp._BOX_MAX_PX) == (4.0, 8, 140)


def test_offline_candidate_builder_equals_runtime():
    from train.tah_cache import _mk
    rng = np.random.default_rng(1)
    n = 0
    for _ in range(400):
        cam = (rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(20, 40), -50.0, rng.uniform(-180, 180))
        w = (cam[0] + rng.uniform(-60, 60), cam[1] + rng.uniform(-60, 60), 0.5)
        a = _mk(w, True, -1, cam, 512, 512, 90.0, 4.0, 10.0, 140.0)
        b = geometry.make_candidate(w, True, -1, cam, IMG)
        assert (a is None) == (b is None)
        if a is not None:
            n += 1
            assert (a.u, a.v, a.w, a.h, a.depth, a.is_target, a.idx) == (b.u, b.v, b.w, b.h, b.depth, b.is_target, b.idx)
    assert n > 50


def test_instruction_parser_copies_agree_on_all_dataset_instructions():
    from rollout.opening_score import describe as d2
    langs = []
    for f in sorted(glob.glob("/nvidia/hque/data/carla_data/mvp_full_v5_fov90/episode_s*.h5"))[:32]:
        with h5py.File(f, "r") as h:
            langs.append(str(h.attrs["language"]))
    assert len(langs) == 32
    for s in langs:
        assert language.describe(s) == d2(s)
        assert language.describe(s)[1] != ""                     # every instruction names a colour


def test_crop_parameters_match_the_feature_id():
    fid = crops.FEATURES["dino_s14_lvd142m_cls384_c224_p1.3"]
    sites = crops.dino_call_sites()
    assert len(sites) >= 6
    for f, line, crop, pad in sites:
        if crop is not None:
            assert (crop, pad) == (fid["crop"], fid["pad"]), (f, line)
