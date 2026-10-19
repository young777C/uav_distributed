"""GT access audit mechanics (framework §7.3–7.4): proxies are transparent to numpy code, every read is attributed to its
call site, nested vis[1] is covered, label properties record reads, and the judge flags undeclared / unknown reads
(positive controls)."""
import numpy as np

from acot.runtime import audit as A


def planted_decision(gt):                      # a "method" that reads tpos at decision time (must be caught)
    return float(np.asarray(gt[0])[0]) > 0


def legit_box_proxy(gt):                       # stands for build_candidates: reads tpos / dstates to make boxes
    t = (float(gt[0][0]), float(gt[0][1]), float(gt[0][2]))
    d = np.asarray(gt[1])
    return t, d.shape[0]


def occlusion_filter(gt):
    t_vis, d_vis = gt[4]
    return t_vis, [d_vis[i] for i in range(len(d_vis))]


def sneaky_vis(gt):
    _, d = gt[4]                               # unpack once …
    return any(d)                              # … then read the distractor list (must still be recorded)


def _gt():
    return A.wrap_gt_candidates((np.array([1.0, 2.0, 0.0]), np.zeros((3, 6)), (0, 0, 10, -50, 0), None,
                                 (True, [False, True, True])))


def test_proxies_are_transparent_and_record_sites():
    A.AUDIT.enabled = True
    A.AUDIT.reset()
    gt = _gt()
    assert legit_box_proxy(gt) == ((1.0, 2.0, 0.0), 3)
    assert np.allclose(np.asarray(gt[0], np.float64) * 2, [2, 4, 0]) and (gt[0] + 1)[0] == 2.0
    assert occlusion_filter(gt) == (True, [False, True, True])
    c = A.AUDIT.reset()
    funcs = {(k[0], k[2]) for k in c}
    assert ("tpos", "legit_box_proxy") in funcs and ("dstates", "legit_box_proxy") in funcs
    assert ("vis", "occlusion_filter") in funcs and ("vis[1]", "occlusion_filter") in funcs


def test_positive_controls_are_flagged():
    A.AUDIT.enabled = True
    A.AUDIT.reset()
    gt = _gt()
    legit_box_proxy(gt)
    occlusion_filter(gt)
    sites = [{"kind": k[0], "file": k[1], "func": k[2], "line": k[3], "category": "privileged"}
             for k in A.AUDIT.reset()]
    planted_decision(gt)
    sneaky_vis(gt)
    viol, unknown, _ = A.judge(A.AUDIT.reset(), sites, uses_gt=[])
    assert {(u["kind"], u["func"]) for u in unknown} == {("tpos", "planted_decision"), ("vis", "sneaky_vis"),
                                                         ("vis[1]", "sneaky_vis")}
    assert viol == []


def test_diagnostic_sites_need_the_declared_channel():
    key = ("true_idx", "carla_uav_tracking/rollout/policy.py", "act", "ti = cset.true_idx")
    sites = [{"kind": key[0], "file": key[1], "func": key[2], "line": key[3], "category": "diagnostic:gallery_seed_gt"}]
    v, u, _ = A.judge({key: 3}, sites, uses_gt=[])
    assert len(v) == 1 and u == []
    v, u, cats = A.judge({key: 3}, sites, uses_gt=["gallery_seed_gt"])
    assert v == [] and cats == {"diagnostic:gallery_seed_gt": 3}


def test_label_properties_record_reads_and_keep_values():
    from dataclasses import dataclass

    @dataclass
    class Cand:
        u: float
        is_target: bool
        color: str = "red"

    A._audited_attr(Cand, "is_target", "is_target")
    A._audited_attr(Cand, "color", "target_attr")
    try:
        A.AUDIT.enabled = True
        A.AUDIT.reset()
        c = Cand(1.0, True)
        assert c.is_target is True and c.color == "red"
        c.is_target = False
        assert c.is_target is False
        kinds = {k[0] for k in A.AUDIT.reset()}
        assert kinds == {"is_target", "target_attr"}
    finally:
        A.uninstall_label_audit()
    assert "is_target" not in Cand.__dict__ and Cand.__dict__["color"] == "red"


def test_disabled_audit_records_nothing():
    A.AUDIT.enabled = False
    A.AUDIT.reset()
    planted_decision(_gt())
    assert A.AUDIT.reset() == {}
