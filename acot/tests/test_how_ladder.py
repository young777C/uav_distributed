"""1024 HOW-ceiling ladder (acot_note/1024_how_ceiling_prereg.md): the true-target-state WHERE channel (control.ex_source
= oracle, GT channel ex_source_gt) and the expert-without-E3 switch (control.expert_look_away)."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from acot.config import schema as S
from acot.config.resolve import GT_EX_SOURCES, classify_gt, load_protocol, resolve_arm
from acot.runtime import support
from acot.runtime.io import OracleError, OracleHandle, ScoringInfo

P4 = load_protocol("sim_proxy_v2@4")


def info(tw=None, true_idx=-1):
    return ScoringInfo(step=0, true_idx=true_idx, is_target=(), gt_idx=(), target_color="", target_bp="", vis=None,
                       target_world=tw)


def test_scoring_info_target_world_defaults_to_none():
    i = ScoringInfo(step=0, true_idx=-1, is_target=(), gt_idx=(), target_color="", target_bp="", vis=None)
    assert i.target_world is None


def test_true_world_is_gated_counted_and_unbound_none():
    with pytest.raises(OracleError):
        OracleHandle(["oracle_head"]).true_world("ex_source_gt")
    o = OracleHandle(["oracle_head", "ex_source_gt"])
    assert o.true_world("ex_source_gt") is None                                  # nothing bound yet
    o.bind(info((1.0, 2.0, 3.0)))
    assert np.allclose(o.true_world("ex_source_gt"), [1.0, 2.0, 3.0])
    assert o.reset_calls() == {"ex_source_gt": 2}


def test_oracle_ex_source_is_a_gt_channel_and_runs_on_acot_runtime():
    assert "oracle" in GT_EX_SOURCES and "ex_source_gt" in S.DIAGNOSTICS
    a = resolve_arm(P4, "x", {"method": "diag_oracle_head_gtwhere", "role": "diagnostic"})
    assert a["uses_gt"] == ["ex_source_gt", "oracle_head"]
    assert classify_gt(P4, a["method"]) == ["ex_source_gt", "oracle_head"]
    assert support.unsupported(a["protocol"], a["method"]) == []
    # a deployable method cannot switch it on silently
    with pytest.raises(Exception, match="undeclared"):
        resolve_arm(P4, "x", {"method": "ours_cv", "override": {"control.ex_source": "oracle"}})


def test_oracle_ex_source_refused_without_the_dit_flying():
    a = resolve_arm(P4, "x", {"method": "diag_oracle_head_gtwhere", "role": "diagnostic"})
    m = dict(a["method"], control=dict(a["method"]["control"], expert_control=True))
    assert any("ex_source='oracle'" in r for r in support.unsupported(a["protocol"], m))
    m = dict(a["method"], control=dict(a["method"]["control"], ex_source="cv"))
    assert any("control.ex_source='cv'" in r for r in support.unsupported(a["protocol"], m))


def test_ladder_arms_resolve_as_specified():
    r0 = resolve_arm(P4, "r0", {"method": "diag_oracle_head_cv", "role": "diagnostic"})["method"]["control"]
    r2 = resolve_arm(P4, "r2", {"method": "diag_oracle_head_expert_noe3", "role": "diagnostic"})["method"]["control"]
    assert (r0["ex_source"], r0["absent_where"], r0["expert_control"]) == ("track", "cv", False)
    assert (r2["ex_source"], r2["expert_control"], r2["expert_look_away"]) == ("track", True, False)
    # the switch is optional: absent everywhere else (existing resolved configs and hashes unchanged)
    old = resolve_arm(P4, "u1x", {"method": "diag_oracle_head_expert", "role": "diagnostic"})["method"]["control"]
    assert "expert_look_away" not in old


def test_r1f_resolves_and_wp_every_frame_needs_the_dit():
    a = resolve_arm(P4, "r1f", {"method": "diag_oracle_head_gtwhere_f", "role": "diagnostic"})
    c = a["method"]["control"]
    assert (c["ex_source"], c["wp_every_frame"]) == ("oracle", True) and a["uses_gt"] == ["ex_source_gt", "oracle_head"]
    assert support.unsupported(a["protocol"], a["method"]) == []
    m = dict(a["method"], control=dict(c, ex_source="track", expert_control=True))
    assert any("control.wp_every_frame" in r for r in support.unsupported(a["protocol"], m))
    old = resolve_arm(P4, "x", {"method": "diag_oracle_head", "role": "diagnostic"})["method"]["control"]
    assert "wp_every_frame" not in old


def _pol(ex_source, absent_where="ear"):
    import torch
    from acot.runtime.policy import AcotPolicy
    from acot.runtime.server import setup_paths
    setup_paths()
    from rollout.target_state import TargetStateEstimator
    p = AcotPolicy.__new__(AcotPolicy)
    p.ex_source, p.absent_where, p.device = ex_source, absent_where, "cpu"
    p.pred_conf, p.conf_tau, p.pred_slot, p.commit_k = 0.0, 0.05, -1, 5
    p.dets, p.committed, p.chal, p.chal_n = None, None, None, 0
    p.W, p.H, p.fov = 512, 512, 90.0
    p.estimator = TargetStateEstimator([1, 2, 4, 6], 10.0, 10)
    p.z_ex = torch.zeros(1, 4, 3)
    p.z_ex_est, p.wp_every_frame = False, False
    return p


class Orc:
    def __init__(self, tw):
        self.tw, self.calls = tw, []

    def true_world(self, purpose):
        self.calls.append(purpose)
        return None if self.tw is None else np.asarray(self.tw, np.float64)


def test_where_oracle_feeds_the_true_position_every_tick_even_with_no_box():
    p = _pol("oracle")
    cam = (0.0, 0.0, 40.0, -50.0, 0.0)
    p._where(NS(campose=cam, step=0), Orc((20.0, 0.0, 0.0)))
    p._where(NS(campose=cam, step=6), Orc((26.0, 0.0, 0.0)))               # no detections at all (target off-screen)
    assert np.allclose(p.estimator.p_last, [26.0, 0.0, 0.0]) and p.estimator.t_last == 6
    assert np.allclose(p.estimator.v_last, [10.0, 0.0, 0.0])                 # 6 m over 0.6 s
    want = p.estimator.predict_wp(cam, 6, mode="cv")
    assert np.allclose(p.z_ex[0].numpy(), want, atol=1e-5)
    assert p.committed is None and p.z_ex_est                                # the commitment is untouched


@pytest.mark.parametrize("tw", [None, (np.nan, 0.0, 0.0)])
def test_where_oracle_fails_loudly_without_a_finite_true_state(tw):
    p = _pol("oracle")
    with pytest.raises(RuntimeError, match="no finite true target position"):
        p._where(NS(campose=(0.0, 0.0, 40.0, -50.0, 0.0), step=0), Orc(tw))


def test_where_track_keeps_ear_sample_without_box_and_never_asks_the_oracle():
    import torch
    p = _pol("track")
    o = Orc((20.0, 0.0, 0.0))
    p._where(NS(campose=(0.0, 0.0, 40.0, -50.0, 0.0), step=0), o)
    assert torch.count_nonzero(p.z_ex) == 0 and o.calls == [] and not p.estimator.seen


def _act_frames(wp_every_frame, ex_source="oracle"):
    """drive AcotPolicy.act on one S2 tick (step 0) and the next non-S2 frame (step 1) with a stub frontend: the DiT
    sample is captured instead of computed, so the test sees exactly the z_ex each frame's action is conditioned on."""
    import torch
    from unittest import mock
    p = _pol(ex_source)
    p.wp_every_frame = wp_every_frame
    p.ablate, p.expert_flown, p.frame_gain, p.frame_cw = None, False, 0.0, None
    p.policy_rng, p.rng_base, p.frame_input, p.kind, p.head = "unseeded", None, None, "oracle", None
    p.fe = NS(K=4, steps=1, H=1, A=5, a_scale=np.ones(4, np.float32), ear=None, dit=None,
              iar=lambda layers, mask: (torch.zeros(1, 8), None))
    seen = []

    def fake_sample(dit, z_ex, *a, **k):
        seen.append(z_ex.clone())
        return torch.zeros(1, 1, 5)
    p._identity = lambda inp, dets, oracle: {"n": 0}
    o = Orc((20.0, 0.0, 0.0))
    base = dict(proprio=np.zeros(5, np.float32), rgb=None, detections=None)
    with mock.patch("train.flow_matching.sample", lambda *a, **k: torch.zeros(1, 4, 3)), \
            mock.patch("train.stage2.action_sample", fake_sample):
        p.act(NS(step=0, s2=True, vlm=([torch.zeros(3, 8)], torch.zeros(3, 8)), campose=(0.0, 0.0, 40.0, -50.0, 0.0),
                 **base), o)
        p.act(NS(step=1, s2=False, vlm=None, campose=(1.0, 0.0, 40.0, -50.0, 15.0), **base), o)
    return p, seen


def test_wp_every_frame_reprojects_between_s2_ticks_and_default_holds():
    p, seen = _act_frames(True)
    want = p.estimator.predict_wp((1.0, 0.0, 40.0, -50.0, 15.0), 1, mode="cv")
    assert np.allclose(seen[1][0].numpy(), want, atol=1e-5) and not np.allclose(seen[1], seen[0])
    p, seen = _act_frames(False)
    assert np.allclose(seen[1], seen[0])                                     # legacy: the S2 waypoints are held
