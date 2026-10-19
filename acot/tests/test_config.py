"""Schema strictness, composition / override semantics, ownership rules, GT classification, hash stability, seed registry
(framework §5, §15)."""
import copy

import pytest
import yaml

from acot.config import schema as S
from acot.config.load import deep_merge, flatten, set_dotted
from acot.config.resolve import (ResolveError, check_compat, config_hash, load_protocol, resolve_arm,
                                 resolve_experiment)
from acot.config.seeds import SeedRegistry

P = load_protocol("sim_proxy_v2@1")


def exp_file(tmp_path, arms, seeds=None, **extra):
    d = {"name": "t", "protocol": "sim_proxy_v2@1", "seeds": seeds or {"set": "dev", "subset": [[91000, 91001]]},
         "arms": arms, **extra}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


# ------------------------------------------------------------------------------------------------------------- schema
def test_schema_rejects_unknown_missing_and_bad_values():
    S.validate({"a": 1}, {"a": S.INT})
    with pytest.raises(S.SchemaError, match="unknown key"):
        S.validate({"a": 1, "b": 2}, {"a": S.INT})
    with pytest.raises(S.SchemaError, match="missing"):
        S.validate({}, {"a": S.INT})
    with pytest.raises(S.SchemaError, match="expected int"):
        S.validate({"a": True}, {"a": S.INT})                    # bool is not an int here
    with pytest.raises(S.SchemaError, match="not in"):
        S.validate({"a": "x"}, {"a": S.enum("y", "z")})
    S.validate({}, {"a": S.Opt(S.INT)})


def test_deep_merge_replaces_lists_and_merges_maps():
    assert deep_merge({"a": {"b": 1, "c": [1, 2]}}, {"a": {"c": [3]}}) == {"a": {"b": 1, "c": [3]}}
    d = {}
    set_dotted(d, "x.y.z", 1)
    assert flatten(d) == {"x.y.z": 1}


# ------------------------------------------------------------------------------------------------------------ methods
def test_final_method_resolves_fully_explicit():
    a = resolve_arm(P, "ours", {"method": "ours_final"})
    m = a["method"]
    assert a["role"] == "main" and a["uses_gt"] == []
    assert m["filter"]["filter_kind"] == "tempered" and m["filter"]["absent_hold"] is True
    assert set(m["identity"]) >= {"tah"} and "reid" not in m["identity"]          # inactive sections dropped
    assert m["control"]["ex_source"] == "track" and m["identity"]["memory_seed"] == "committed"


def test_filter_off_drops_fields_and_filter_kind_required():
    assert resolve_arm(P, "x", {"method": "ours_no_pgif"})["method"]["filter"] == {"mode": "off"}
    with pytest.raises(ResolveError, match="filter_kind"):
        resolve_arm(P, "x", {"method": "ours_final", "override": {"filter.filter_kind": None}})


def test_tah_ckpt_has_no_default():
    with pytest.raises(ResolveError, match="tah.ckpt must be set"):
        resolve_arm(P, "x", {"method": "ours_final", "override": {"identity.tah.ckpt": None}})


def test_inactive_head_section_cannot_be_set():
    with pytest.raises(ResolveError, match="does not read identity.reid"):
        resolve_arm(P, "x", {"method": "ours_final", "override": {"identity.reid.tavg": 0.5}})


def test_undeclared_gt_channel_is_an_error_and_declared_one_makes_a_diagnostic_arm():
    with pytest.raises(ResolveError, match="undeclared"):
        resolve_arm(P, "x", {"method": "ours_final", "override": {"identity.memory_seed": "gt"}})
    a = resolve_arm(P, "x", {"method": "ours_final",
                             "override": {"identity.memory_seed": "gt", "diagnostics": ["gallery_seed_gt"]}})
    assert a["role"] == "diagnostic" and a["uses_gt"] == ["gallery_seed_gt"]
    with pytest.raises(ResolveError, match="unused"):
        resolve_arm(P, "x", {"method": "ours_final", "override": {"diagnostics": ["warmup_oracle"]}})
    with pytest.raises(ResolveError, match="role must be diagnostic"):
        resolve_arm(P, "x", {"method": "diag_oracle_absent", "role": "main"})


def test_ex_source_other_than_track_is_gt():
    a = resolve_arm(P, "x", {"method": "ours_final",
                             "override": {"control.ex_source": "cv_gated", "diagnostics": ["ex_source_gt"]}})
    assert a["uses_gt"] == ["ex_source_gt"]
    assert resolve_arm(P, "x", {"method": "ours_final", "override": {"control.ex_source": "ear"}})["uses_gt"] == []


# ---------------------------------------------------------------------------------------------------------- ownership
def test_protocol_invariants_cannot_be_overridden():
    for key in ("camera.fov_deg", "sim.s2_period", "scorer.commit_k", "privileged_perception"):
        with pytest.raises(ResolveError, match="protocol invariant"):
            resolve_arm(P, "x", {"method": "ours_final", "override": {key: 1}})


def test_frontend_override_needs_matching_ablation():
    with pytest.raises(ResolveError, match="needs `ablation:`"):
        resolve_arm(P, "x", {"method": "ours_final", "override": {"frontend.opening.module": "head"}})
    with pytest.raises(ResolveError, match="needs `ablation:`"):
        resolve_arm(P, "x", {"method": "ours_final", "ablation": "opening",
                             "override": {"perception.occlusion": "none"}})
    a = resolve_arm(P, "x", {"method": "ours_final", "ablation": "opening",
                             "override": {"frontend.opening.module": "head", "frontend.opening.window": 1}})
    assert a["role"] == "ablation" and "attr" not in a["protocol"]["frontend"]["opening"]


def test_perception_ablation_recomputes_privileged_list():
    a = resolve_arm(P, "x", {"method": "ours_final", "ablation": "perception",
                             "override": {"perception.identity_keys": "sim"}})
    assert "identity_keys_sim_actor_ids" in a["protocol"]["privileged_perception"]
    with pytest.raises(ResolveError, match="where_localisation"):
        resolve_arm(P, "x", {"method": "ours_final", "ablation": "perception",
                             "override": {"perception.identity_keys": "sim", "perception.occlusion": "none"}})


# ------------------------------------------------------------------------------------------------------- hash / compat
def test_hash_ignores_key_order_and_infra(tmp_path):
    a = resolve_arm(P, "x", {"method": "ours_final"})
    sci = {"method": a["method"], "protocol": a["protocol"],
           "artefacts": {r["path"]: r["artefact"] for r in a["artefacts"]}}     # content ids are part of the hash
    assert config_hash(sci) == a["config_hash"]
    R1 = resolve_experiment(exp_file(tmp_path, {"a": {"method": "ours_final"}}))
    R2 = resolve_experiment(exp_file(tmp_path, {"a": {"method": "ours_final"}}), [("infra.policy_gpus", [7])])
    assert R1["arms"]["a"]["config_hash"] == R2["arms"]["a"]["config_hash"] and not R2["ad_hoc"]
    R3 = resolve_experiment(exp_file(tmp_path, {"a": {"method": "ours_final"}}), [("identity.conf_tau", 0.4)])
    assert R3["ad_hoc"] and R3["arms"]["a"]["config_hash"] != R1["arms"]["a"]["config_hash"]


def test_compat_rejects_protocol_model_config_mismatch():
    q = copy.deepcopy(P)
    q["camera"]["fov_deg"] = 70.0
    with pytest.raises(ResolveError, match="fov_deg"):
        check_compat(q, "t")


# -------------------------------------------------------------------------------------------------------------- seeds
def test_seed_registry_overlap_lock_and_subset(tmp_path):
    R = SeedRegistry()
    assert R.overlaps() == []
    pairs, purpose = R.select({"set": "dev", "subset": [[91000, 91004]]})
    assert purpose == "dev" and pairs == [("Town05", s) for s in range(91000, 91005)]
    with pytest.raises(PermissionError):
        R.select({"set": "test_town05"})
    assert len(R.select({"set": "test_town05"}, allow_test=True, prereg_ref="1004_ablation_design.md")[0]) == 80
    with pytest.raises(S.SchemaError, match="expects"):
        R.select({"set": "dev", "expect": 19})
    bad = tmp_path / "seeds.yaml"
    bad.write_text(yaml.safe_dump({"seedsets": {
        "a": {"purpose": "dev", "blocks": [{"town": "Town05", "ranges": [[1, 5]]}]},
        "b": {"purpose": "test", "blocks": [{"town": "Town05", "list": [5, 6]}]},
        "c": {"purpose": "test", "blocks": [{"town": "Town03", "list": [1]}]}}}))
    assert SeedRegistry(str(bad)).overlaps() == [("a", "b", 1)]        # same seed in another town is not an overlap


def test_locked_experiment_needs_allow_test_and_prereg(tmp_path):
    p = exp_file(tmp_path, {"a": {"method": "ours_final"}}, seeds={"set": "test_town05"})
    with pytest.raises(PermissionError):
        resolve_experiment(p, allow_test=True)                          # no prereg_ref
    p = exp_file(tmp_path, {"a": {"method": "ours_final"}}, seeds={"set": "test_town05"}, prereg_ref="1004_ablation_design.md")
    assert len(resolve_experiment(p, allow_test=True)["seeds"]) == 80
