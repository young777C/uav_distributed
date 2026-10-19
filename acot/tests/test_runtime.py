"""P2 runtime (framework §7.4, §14): structural GT separation of the acot runtime, the oracle handle, the PGIF port
against the legacy ASD filter, what the runtime supports, the server's refusal of legacy env, and the neutrality of
the cand_lite gt_idx fix for the primary metrics. Host-side (no GPU); decision parity of the full policy is checked
by acot/eval/replay_parity.py in the policy image."""
import ast
import os
import random
import sys

import numpy as np
import pytest

from acot import ROOT
from acot.config.resolve import load_protocol, resolve_arm
from acot.runtime import io as RIO
from acot.runtime import support

sys.path.insert(0, os.path.join(ROOT, "carla_uav_tracking"))
P = load_protocol("sim_proxy_v2@1")
METHOD_MODULES = ("policy.py", "heads.py", "pgif.py", "frontend.py")


# -------------------------------------------------------------------------------------------- structural separation
def test_policy_input_fields():
    for cls in (RIO.PolicyInput, RIO.Detection):
        assert not (RIO.field_names(cls) & RIO.FORBIDDEN), cls.__name__


def _names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            yield node.attr, node
        elif isinstance(node, ast.Name):
            yield node.id, node
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, node
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg, node


def test_method_no_gt_attr():
    for m in METHOD_MODULES:
        tree = ast.parse(open(os.path.join(ROOT, "acot", "runtime", m)).read())
        bad = sorted({n for n, _ in _names(tree) if n in RIO.FORBIDDEN})
        assert not bad, f"{m} touches label fields {bad}"
        imports = {a.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for a in node.names}
        assert "ScoringInfo" not in imports and "OracleHandle" not in imports, m
        assert not {"environ", "getenv"} & {n for n, _ in _names(tree)}, f"{m} reads environment variables"
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "true_slot"]
        assert all(isinstance(c.value, ast.Name) and c.value.id == "oracle" for c in calls), m


def test_oracle_handle():
    o = RIO.OracleHandle(["oracle_absent_abstain"])
    o.bind(RIO.ScoringInfo(step=0, true_idx=2, is_target=(False, False, True), gt_idx=(0, 1, -1), target_color="",
                           target_bp="", vis=None))
    assert o.true_slot("oracle_absent_abstain") == 2
    with pytest.raises(RIO.OracleError):
        o.true_slot("gallery_seed_gt")
    assert o.reset_calls() == {"oracle_absent_abstain": 1} and o.calls == {}
    o.bind(None)
    assert o.true_slot("oracle_absent_abstain") == -1
    a = resolve_arm(P, "x", {"method": "diag_oracle_absent"})
    from acot.runtime.policy import ORACLE_PURPOSE
    assert a["uses_gt"] == [ORACLE_PURPOSE[a["method"]["oracle_absent"]]]          # the purpose the policy asks for
    assert resolve_arm(P, "x", {"method": "ours_final"})["uses_gt"] == []           # → no handle at all


# ------------------------------------------------------------------------------------------------------ PGIF port
def _legacy_asd(**kw):
    from rollout.asd import ASD
    return ASD(mode="filter", **kw)


@pytest.mark.parametrize("kind", ["tempered", "resid", "bayes"])
@pytest.mark.parametrize("absent_hold", [False, True])
def test_pgif_port_matches_legacy_asd(kind, absent_hold):
    from acot.runtime.pgif import Pgif
    kw = dict(filter_kind=kind, resid_decay=0.98, absent_hold=absent_hold, absent_max=8, conf_window=30,
              gate_memory=True, eps_switch=0.01, q_lo=0.2, q_hi=0.7)
    new, old = Pgif(**kw), _legacy_asd(**kw)
    rng = random.Random(7)
    for ep in range(6):
        new.reset()
        old.reset()
        alive = list(range(4))
        for t in range(150):
            if rng.random() < 0.08:
                alive.append(max(alive) + 1)                     # tracker births
            if rng.random() < 0.05 and len(alive) > 2:
                alive.pop(rng.randrange(len(alive)))             # deaths
            keys = rng.sample(alive, k=rng.randint(1, len(alive)))
            slot = rng.randrange(len(keys))
            conf = rng.random() * 3
            a = new.step(keys, slot, conf)
            b = old.step(None, keys, slot, mem_conf=conf, lang=None, true_slot=None, t=t)
            assert a == b, (ep, t, a, b)
            assert new.belief == old.belief and new.conf_hist == old.conf_hist


# ------------------------------------------------------------------------------------------------- support matrix
@pytest.mark.parametrize("m", ["ours_final", "ours_no_pgif", "ours_no_absent", "ours_pgif_bayes", "ext_dam4sam",
                               "ext_assoc_deepsort", "ext_assoc_motion", "lang_xattn", "diag_oracle_absent"])
def test_1004_methods_supported(m):
    a = resolve_arm(P, "x", {"method": m})
    assert support.unsupported(a["protocol"], a["method"], ("ticks", "trk", "gt_access", "scenario")) == []


def test_open_head_ablation_supported_and_unsupported_settings_refused():
    a = resolve_arm(P, "x", {"method": "ours_final", "ablation": "opening",
                             "override": {"frontend.opening.module": "head", "frontend.opening.window": 1}})
    assert support.unsupported(a["protocol"], a["method"]) == []
    a = resolve_arm(P, "x", {"method": "ours_final", "override": {"identity.reanchor.mode": "lang"}})
    assert any("reanchor" in x for x in support.unsupported(a["protocol"], a["method"]))
    a = resolve_arm(P, "x", {"method": "ext_assoc_deepsort", "override": {"identity.memory_seed": "gt",
                                                                          "diagnostics": ["gallery_seed_gt"]}})
    assert any("memory_seed" in x for x in support.unsupported(a["protocol"], a["method"]))   # gt only for tah (U2)
    a = resolve_arm(P, "x", {"method": "diag_gt_memory"})
    assert support.unsupported(a["protocol"], a["method"]) == []
    assert any("dagger" in x for x in support.unsupported(P, resolve_arm(P, "x", {"method": "ours_final"})["method"],
                                                          ("ticks", "dagger")))
    v1 = load_protocol("deploy_v1@1")
    assert support.unsupported(v1, resolve_arm(v1, "x", {"method": "ours_final"})["method"])   # sim keys / no tracker


# ------------------------------------------------------------------------------------------------------ server env
def test_server_refuses_every_legacy_behaviour_env():
    from acot.config.legacy import POLICY_ENV
    from acot.runtime.server import LEGACY_ENV, legacy_env_set
    assert set(POLICY_ENV) <= set(LEGACY_ENV)
    assert legacy_env_set({"OPENING": "attr", "PATH": "/bin"}) == ["OPENING"]
    assert legacy_env_set({"DAGGER_DIR": ""}) == ["DAGGER_DIR"]                     # presence alone is refused


# ----------------------------------------------------------------------------------- cand_lite gt_idx (framework §12.2)
def test_cand_lite_gt_idx_is_neutral_for_primary_metrics_when_commit_k_le_s2_period():
    rng = random.Random(3)
    for _ in range(200):
        picks = [rng.choice([-1, -1, 0, 1, 2, None]) for _ in range(40)]
        churn = {(a, k): rng.randrange(100) for a in range(3) for k in range(40)}   # tracker ids that change
        legacy = _score_keys(picks, lambda a, k: 100 + churn[(a, k)])
        fixed = _score_keys(picks, None)
        assert legacy == fixed


def _score_keys(picks, tracker_key):
    """legacy wire: cands carry only the tracker key (no gt_idx); fixed wire: they carry gt_idx = actor."""
    from types import SimpleNamespace as NS
    from rollout.metrics import RolloutScorer
    sc = RolloutScorer(fps=10, commit_k=5, t_fail_s=2.0, max_lost_s=18.0, central_frac=0.3)
    sc.reset({})
    for t in range(len(picks) * 6):
        k = t // 6
        actor = picks[k]
        if tracker_key is None:
            ds = [NS(is_target=False, idx=50 + a, gt_idx=a, depth=12.0) for a in range(3)]
        else:
            ds = [NS(is_target=False, idx=tracker_key(a, k), depth=12.0) for a in range(3)]
        cands = [NS(is_target=True, idx=-1, depth=10.0)] + ds
        slot = -1 if actor is None else (0 if actor == -1 else 1 + actor)
        sc.update(step=t, target_present=True, target_central=True, in_loss_window=False,
                  info=NS(cand_set=NS(cands=cands, true_idx=0), pred_slot=slot), target_visible=True)
    return sc._sustain_wrong, sc._max_correct_run, sc._max_wrong_run, sc._id_switches, sc._present


# --------------------------------------------------------------------------------------- remaining §7.4 tests + L0 (P2)
def _exp(tmp_path, arms, **extra):
    import yaml
    d = {"name": "t", "protocol": "sim_proxy_v2@1", "seeds": {"set": "dev", "subset": [[91000, 91001]]}, "arms": arms,
         **extra}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


def test_diag_label_report_refuses_diagnostic_arm_outside_family_diagnostic(tmp_path):
    from acot.analysis.report import check_families
    from acot.config.resolve import resolve_experiment
    arms = {"ours": {"method": "ours_final"}, "orc": {"method": "diag_oracle_absent"}}
    R = resolve_experiment(_exp(tmp_path, arms, comparisons=[{"family": "baseline", "a": "ours", "b": ["orc"],
                                                              "metrics": ["correct"]}]))
    assert R["arms"]["orc"]["role"] == "diagnostic"
    with pytest.raises(ValueError, match="diagnostic arm"):
        check_families(R)
    R["comparisons"][0]["family"] = "diagnostic"
    check_families(R)


def test_wire_blanking(tmp_path, monkeypatch):
    from dataclasses import dataclass
    import acot_client.wire as W
    from acot.config.resolve import resolve_experiment
    from acot.eval.scheduler import Run

    @dataclass
    class Obs:
        rgb: object
        target_color: str
        target_bp: str
        retarget_event: bool

    got = {}
    monkeypatch.setattr(W.AcotRemotePolicy, "_exchange",
                        lambda self, obs, gt: got.update(obs=obs, gt=gt) or ((0.0,) * 4, 0.0, None))
    arms = {"ours": {"method": "ours_final"}, "orc": {"method": "diag_oracle_absent"}}
    run = Run(resolve_experiment(_exp(tmp_path, arms)), None)
    for arm in arms:                                   # neither declares attrbind / road / retarget → all blanked
        p = W.AcotRemotePolicy.__new__(W.AcotRemotePolicy)
        p.blank = run.blank_rules(arm)
        p.act(Obs(None, "red", "Audi TT", True), (1, 2, 3, [[0, 0, 0]], ("t_vis", ["d_vis"])))
        o, gt = got["obs"], got["gt"]
        assert (o.target_color, o.target_bp, o.retarget_event) == ("", "", False)
        assert gt[3] is None and gt[4] == ("t_vis", ["d_vis"]) and gt[:3] == (1, 2, 3)


def test_gt_audit_ratchet():
    from acot.runtime import audit as A
    logged = ("label", "carla_uav_tracking/rollout/policy_server.py", "_cand_lite", "x = c.is_target")
    new = ("tpos", "carla_uav_tracking/rollout/policy.py", "act", "y = tpos[0]")
    sites = [{"kind": logged[0], "file": logged[1], "func": logged[2], "line": logged[3], "category": "scoring"}]
    viol, unknown, cats = A.judge({logged: 2, new: 1}, sites, [])
    assert not viol and cats == {"scoring": 2} and [u["func"] for u in unknown] == ["act"]       # new site → unknown
    sites.append({"kind": new[0], "file": new[1], "func": new[2], "line": new[3], "category": "diagnostic:ex_source_gt"})
    viol, unknown, _ = A.judge({logged: 2, new: 1}, sites, [])
    assert not unknown and len(viol) == 1                                                      # undeclared channel
    assert A.judge({logged: 2, new: 1}, sites, ["ex_source_gt"])[0] == []


def test_acot_runtime_invocation_has_no_legacy_env_and_no_pip(tmp_path):
    from acot.config.resolve import resolve_experiment
    from acot.eval.scheduler import Run
    from acot.runtime.server import LEGACY_ENV
    R = resolve_experiment(_exp(tmp_path, {"ours": {"method": "ours_final"}}, infra={"runtime": "acot"}))
    run = Run(R, None)
    w = {"id": "w0", "gpu": 4, "carla_port": 2050, "tm_base": 9700}
    cmd = run.acot_policy_cmd("ours", w, "w0_p1", R["infra"]["docker"]["policy_image_acot"], "sha256:x")
    envs = {cmd[i + 1].split("=", 1)[0] for i, x in enumerate(cmd) if x == "-e"}
    assert run.runtime == "acot" and not envs & set(LEGACY_ENV)
    assert R["infra"]["docker"]["policy_image_acot"] in cmd and "acot.runtime.server" in cmd
    assert not any("pip" in x for x in cmd) and "--network" in cmd and cmd[cmd.index("--network") + 1] == "cyh-carla-net"


def test_legacy_runtime_refuses_values_the_legacy_policy_hard_codes(tmp_path):
    from acot.config.resolve import resolve_experiment
    from acot.eval.scheduler import runtime_problems
    arms = {"ours": {"method": "ours_final"},
            "thr": {"method": "ours_final", "override": {"identity.tah.lang_reanchor_thr": 0.5}},
            "cool": {"method": "ext_dam4sam", "override": {"identity.dam4sam.reinit_cooldown": 5}}}
    R = resolve_experiment(_exp(tmp_path, arms))
    assert set(runtime_problems(R, "legacy")) == {"thr", "cool"}
    assert set(runtime_problems(R, "acot")) == set()           # the acot runtime reads both fields (thr: reanchor off)
