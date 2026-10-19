"""P3b offline entry points: explicit splits reproduce the legacy implicit one, the 70°-with-90° projection and an
un-waived tick mix are refused, frozen data plans cannot change silently, QC catches a camera mismatch."""
import glob
import json
import os

import pytest
import yaml

from acot import CONFIGS, ROOT
from acot import artefacts as A


def test_naive_split_equals_the_legacy_fallback():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "carla_uav_tracking"))
    from train.dataset import _select_episodes
    m = A.lookup("artefact:split_v5_fov90_naive")
    sp = json.load(open(os.path.join(ROOT, m["files"][0]["path"])))
    cfg = {"data": {"episodes_glob": "/nvidia/hque/data/carla_data/mvp_full_v5_fov90/episode_s*.h5",
                    "split_manifest": "/no/such/file.json", "val_frac": 0.15}}
    legacy = [os.path.basename(p) for p in _select_episodes(cfg, "train") + _select_episodes(cfg, "val")]
    assert sp["train"] + sp["val"] == legacy and sp["test"] == []


def test_prep_refuses_70_degree_data_with_90_degree_projection():
    from acot.prep.tah_seq import plan
    with pytest.raises(A.ArtefactError, match="70°-images-with-90°-boxes"):
        plan(yaml.safe_load(open(os.path.join(CONFIGS, "prep", "tah_seq_v5_70_wrong.yaml"))))


def test_recipe_tick_mix_needs_a_waiver():
    r = yaml.safe_load(open(os.path.join(CONFIGS, "recipes", "tah_relm_v90_s0_repro.yaml")))
    assert any("tick_s" in e for e in A.check_inputs(r["inputs"]))
    assert A.check_inputs(r["inputs"], waive=tuple(r["waivers"])) == []


def test_frozen_plan_cannot_change(tmp_path, monkeypatch):
    from acot.data import gen
    monkeypatch.setattr(gen, "GEN_ROOT", str(tmp_path))
    spec = yaml.safe_load(open(os.path.join(CONFIGS, "datagen", "mvp_full_v5_fov90_replan.yaml")))
    spec = dict(spec, name="t", plan=dict(spec["plan"], limit=2))
    a = gen.run(spec, dry_run=True)
    assert gen.run(spec, dry_run=True) == a
    with pytest.raises(A.ArtefactError, match="DIFFERENT frozen plan"):
        gen.run(dict(spec, plan=dict(spec["plan"], limit=3)), dry_run=True)


def test_qc_flags_camera_mismatch():
    from acot.data.qc import check_episode
    cam90 = {"fov_deg": 90.0, "width": 512, "height": 512}
    f90 = sorted(glob.glob("/nvidia/hque/data/carla_data/mvp_full_v5_fov90/episode_s04*.h5"))[0]
    f70 = sorted(glob.glob("/nvidia/hque/data/carla_data/mvp_full_v5/episode_*.h5"))[0]
    assert check_episode(f90, cam90)["ok"]
    assert not check_episode(f70, cam90)["checks"]["attrs_match_camera"]
