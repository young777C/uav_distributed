"""Artefact store and compatibility matrix (framework §10, P3a acceptance): content addressing, conflict refusal,
unknown ≠ compatible, the 70°-as-90° and 0.2 s-vs-0.6 s cases are blocked unless waived, resolve-time checks."""
import json

import pytest

from acot import artefacts as A


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "STORE", str(tmp_path / "store"))
    monkeypatch.setattr(A, "INDEX", str(tmp_path / "store" / "index.json"))
    monkeypatch.setattr(A, "waivers", lambda: {})
    return tmp_path


def _file(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_content_addressing_and_conflict_refusal(store):
    f = _file(store, "a.bin", b"x" * 100)
    m1 = A.register("tah_seq", "c", files=[f], inputs={"stride": 2}, compat={"fov_deg": 90.0, "tick_s": 0.2})
    m2 = A.register("tah_seq", "c", files=[f], inputs={"stride": 2}, compat={"fov_deg": 90.0, "tick_s": 0.2})
    assert m1["id"] == m2["id"]                                             # same inputs → same artefact
    assert A.register("tah_seq", "c", files=[f], inputs={"stride": 3})["id"] != m1["id"]
    with pytest.raises(A.ArtefactError, match="DIFFERENT manifest"):         # same id, different content → refuse
        A.register("tah_seq", "c", files=[f], inputs={"stride": 2}, compat={"fov_deg": 70.0, "tick_s": 0.2})
    assert A.lookup(f)["name"] == "c" and A.lookup("artefact:c")["kind"] == "tah_seq"
    open(f, "wb").write(b"y" * 100)
    assert A.verify(m1) and "changed" in A.verify(m1)[0]


def test_unknown_and_mixed_are_not_compatible():
    m = {"compat": {"fov_deg": 90.0, "tick_s": [0.2, 0.6]}}
    assert A.compat_problems(m, {"fov_deg": 90}) == []
    assert A.compat_problems(m, {"tick_s": 0.6}) == [("tick_s", [0.2, 0.6], 0.6)]
    assert A.compat_problems(m, {"feature_id": "x"}) == [("feature_id", "unknown", "x")]


def test_b1_blocks_70_as_90_and_mixed_ticks(store):
    d70 = A.register("tah_seq", "s70", files=[_file(store, "s70", b"1")], compat={"fov_deg": 70.0, "tick_s": 0.2,
                                                                                    "feature_id": "f", "depth_scale": 100})
    d90 = A.register("tah_seq", "s90", files=[_file(store, "s90", b"2")], compat={"fov_deg": 90.0, "tick_s": 0.2,
                                                                                    "feature_id": "f", "depth_scale": 100})
    dag = A.register("dagger", "dag", files=[_file(store, "dag", b"3")], compat={"fov_deg": 90.0, "tick_s": 0.6,
                                                                                  "feature_id": "f", "depth_scale": 100})
    errs = A.check_inputs(["artefact:s70", "artefact:s90"])
    assert any("fov_deg" in e for e in errs)                                # 70° data mixed with 90° data
    errs = A.check_inputs(["artefact:s90", "artefact:dag"])
    assert any("tick_s" in e for e in errs) and not any("fov_deg" in e for e in errs)
    assert A.check_inputs(["artefact:s90", "artefact:dag"], waive=("tick_s",)) == []   # declared waiver


def test_b2_resolve_requires_waiver(store, monkeypatch):
    ck = A.register("tah_ckpt", "t", files=[_file(store, "t.pt", b"4")],
                    compat={"fov_deg": 90.0, "tick_s": [0.2, 0.6], "feature_id": "f", "depth_scale": 100.0})
    want = {"fov_deg": 90.0, "tick_s": 0.6, "feature_id": "f", "depth_scale": 100.0}
    recs, errs = A.check_refs([("identity.tah.ckpt", "artefact:t", want)])
    assert len(errs) == 1 and "tick_s" in errs[0]
    recs, errs = A.check_refs([("identity.tah.ckpt", "artefact:t", want)], accepted={"identity.tah.ckpt": "reason"})
    assert errs == [] and recs[0]["waived"] == [("tick_s", [0.2, 0.6], 0.6)]
    monkeypatch.setattr(A, "waivers", lambda: {"t": {"fields": ["tick_s"], "reason": "r"}})
    assert A.check_refs([("identity.tah.ckpt", "artefact:t", want)])[1] == []
    recs, errs = A.check_refs([("identity.tah.ckpt", "/no/such/file.pt", want)])
    assert "not a registered artefact" in errs[0]


def test_real_store_final_model_resolves_with_recorded_gaps():
    """the imported legacy store: the final model resolves only thanks to the two written waivers."""
    from acot.config.resolve import load_protocol, resolve_arm
    a = resolve_arm(load_protocol("sim_proxy_v2@1"), "x", {"method": "ours_final"})
    paths = {r["path"] for r in a["artefacts"]}
    assert {"frontend.vlm.stage2_ckpt", "identity.tah.ckpt", "frontend.opening.attr.ckpt_lin"} <= paths
    assert any("tick_s" in g for g in a["provenance_gaps"])
