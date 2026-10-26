"""1024 OA-VAT FULL closed loop (acot/runtime/oavat_closed.py): action mapping channels / signs / units against our
camera geometry (plan v2 §6.1) and the identity-blind S2 readout (no actor re-binding; plan v2 §6.4). No GPU."""
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from acot.runtime.io import Detection                                          # noqa: E402
from acot.runtime.oavat_closed import DT, associate, map_action                # noqa: E402
from acot_probe.projection import project_point                                # noqa: E402

V0 = np.zeros(1, np.float32)


def det(key, u, v, w=27.0):
    return Detection(key=key, u=u, v=v, w=w, h=w, depth_m=40.0, world_xyz=None, vlm_feat=V0, crop_feat=None)


# ------------------------------------------------------------------------------------------------------- action
@pytest.mark.parametrize("yaw", [0.0, 90.0, -90.0, 180.0])
def test_forward_is_along_heading(yaw):
    dx, dy, dz, dyaw = map_action((0, 10, 0, 0), yaw, 0.25, 1.0)
    psi = math.radians(yaw)
    assert dz == 0.0 and dyaw == 0.0
    assert dx == pytest.approx(0.25 * 10 * DT * math.cos(psi), abs=1e-9)
    assert dy == pytest.approx(0.25 * 10 * DT * math.sin(psi), abs=1e-9)
    bx, by, _, _ = map_action((0, -10, 0, 0), yaw, 0.25, 1.0)                   # reverse
    assert (bx, by) == pytest.approx((-dx, -dy), abs=1e-9)


def test_units_and_saturation_envelope():
    """±40 (ctrl_max_speed) × gain = the declared maxima of the registered configs: the grid inside the configured
    expert envelope (12 m/s, 30 deg/s; config/default.yaml expert.velocity_limits), N0 with its own disclosed limit."""
    import yaml
    from acot import ROOT
    cfg = yaml.safe_load(open(os.path.join(ROOT, "carla_uav_tracking/config/default.yaml")))
    scen = yaml.safe_load(open(os.path.join(ROOT, "carla_uav_tracking/config/scenarios/stress_v1.yaml"))) or {}
    assert "velocity_limits" not in str((scen.get("uav") or {}).get("expert_policy", {}))   # stress keeps the defaults
    env = cfg["uav"]["expert_policy"]["velocity_limits"]
    assert (env["xy_max"], env["yaw_rate_max"]) == (12.0, 30.0)
    from acot.config.resolve import resolve_experiment
    arms = resolve_experiment(os.path.join(ROOT, "acot/configs/experiments/stress_tune_oavatcl_grid.yaml"))["arms"]
    want = {"g1": (4, 10), "g2": (8, 20), "g3": (12, 30)}
    for a, (vmax, wmax) in want.items():
        o = arms[a]["method"]["identity"]["oavat_full"]
        dx, dy, _, dyaw = map_action((0, 40, 0, 40), 0.0, o["g_forward"], o["g_yaw"])
        assert math.hypot(dx, dy) / DT == pytest.approx(vmax) and dyaw / DT == pytest.approx(wmax)
        assert vmax <= min(env["xy_max"], o["max_speed_mps"]) and wmax <= min(env["yaw_rate_max"], o["max_yaw_dps"])
    dx, dy, _, dyaw = map_action((0, 40, 0, 40), 0.0, 0.01, 1.0)                # N0 (assumed native units)
    assert math.hypot(dx, dy) / DT == pytest.approx(0.4) and dyaw / DT == pytest.approx(40.0)


def test_no_command_hovers_and_unused_channels_refused():
    assert map_action(None, 37.0, 0.25, 1.0) == (0.0, 0.0, 0.0, 0.0)
    assert map_action((0, 0, 0, 0), 37.0, 0.25, 1.0) == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError):
        map_action((5, 10, 0, 0), 0.0, 0.25, 1.0)
    with pytest.raises(RuntimeError):
        map_action((0, 10, 3, 0), 0.0, 0.25, 1.0)


@pytest.mark.parametrize("yaw", [0.0, 90.0, -90.0])
def test_yaw_sign_centres_a_target_on_the_right(yaw):
    """drone_control: target right of centre → error_x > 0 → yaw PID > 0. Mapped dyaw must then turn the camera so the
    target moves toward the image centre (our projection; CARLA yaw grows clockwise from above)."""
    W = H = 512
    cam = (0.0, 0.0, 30.0, -50.0, yaw)
    psi = math.radians(yaw)
    fwd, right = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
    tgt = np.r_[30.0 * fwd + 8.0 * right, 0.0]                                   # ahead, to the right, on the ground
    u0 = project_point(tgt, cam, W, H, 90.0)[0]
    assert u0 > W / 2                                                            # on the right of the image
    _, _, _, dyaw = map_action((0, 0, 0, +20), yaw, 0.25, 1.0)                   # PID output for error_x > 0
    assert dyaw > 0
    u1 = project_point(tgt, (*cam[:4], yaw + dyaw), W, H, 90.0)[0]
    assert abs(u1 - W / 2) < abs(u0 - W / 2)


def test_forward_command_approaches_target_ahead():
    """area below setpoint → vy > 0 → forward motion reduces the distance to a target ahead (box grows)."""
    yaw = 30.0
    psi = math.radians(yaw)
    tgt = np.array([40.0 * math.cos(psi), 40.0 * math.sin(psi)])
    dx, dy, _, _ = map_action((0, 40, 0, 0), yaw, 0.25, 1.0)
    assert np.linalg.norm(tgt - [dx, dy]) < np.linalg.norm(tgt)


# ------------------------------------------------------------------------------------------------------ readout
def test_readout_containing_nearest_then_smallest_key():
    dets = (det(7, 100, 100), det(3, 104, 100), det(9, 300, 300))
    k, n = associate(dets, [96, 92, 16, 16])                                     # centre (104, 100): both contain
    assert (k, n) == (3, 2)                                                      # nearest centre = key 3
    k, n = associate((det(7, 100, 100), det(3, 100, 100)), [92, 92, 16, 16])     # tie → smallest key
    assert (k, n) == (3, 2)
    assert associate(dets, [200, 10, 10, 10]) == (None, 0)                       # no hit
    assert associate(dets, None) == (None, 0) and associate((), [0, 0, 5, 5]) == (None, 0)


class _Shim:
    """the S2 commitment rule of OavatClosedRuntime._commit without the OA-VAT process."""

    def __init__(self):
        from acot.runtime.oavat_closed import OavatClosedRuntime
        self.rt = OavatClosedRuntime.__new__(OavatClosedRuntime)
        from acot.runtime.oavat_closed import _PolicyShim
        self.rt.policy = _PolicyShim(6)
        self.rt.committed = None
        self.rt.worker_errors = {}
        self.rt.dp_calls = self.rt.dp_found = 0

        class _M:
            mission_counter = 0

        class _OA:
            ended = None
            M = _M
        self.rt.oa = _OA

    def tick(self, dets, fresh, slot=None, stage="Track"):
        self.rt.dets = dets
        return self.rt._commit(0, fresh, slot, stage)


def test_no_actor_rebinding_key_change_and_transfer():
    """B2 probes: the committed KEY is kept when OA-VAT has no fresh box — it is never translated to the remembered
    actor's new key; a key transferred to another vehicle is scored as whatever it now denotes."""
    s = _Shim()
    s.tick((det(7, 100, 100), det(8, 300, 300)), None, slot=0)                  # forced opening → key 7
    assert s.rt.committed == 7
    s.tick((det(9, 100, 100), det(8, 300, 300)), None, stage="Lost")             # target re-keyed 7 → 9, no fresh box
    assert s.rt.committed == 7                                                   # stale key kept, no re-binding
    s.tick((det(9, 100, 100), det(8, 300, 300)), [92, 92, 16, 16])               # fresh box on the re-keyed target
    assert s.rt.committed == 9
    s.tick((det(8, 300, 300),), [92, 92, 16, 16])                                # key deleted, fresh box hits nothing
    assert s.rt.committed == 9
    tr = s.tick((det(9, 300, 300), det(4, 100, 100)), [292, 292, 16, 16])       # immediate native switch
    assert s.rt.committed == 9 and tr["geo_key"] == 9 and tr["n_hit"] == 1


def test_forced_only_on_opening_slot():
    s = _Shim()
    tr = s.tick((det(5, 10, 10), det(6, 200, 200)), [192, 192, 16, 16], slot=1)
    assert tr["forced"] and s.rt.committed == 6 and tr["geo_key"] == 6
    tr = s.tick((det(5, 10, 10), det(6, 200, 200)), [2, 2, 16, 16], slot=-1)     # target absent on an opening tick
    assert "forced" not in tr and s.rt.committed == 5


@pytest.mark.parametrize("yaw", [0.0, 90.0, -90.0])
def test_left_target_turns_left(yaw):
    W = H = 512
    cam = (0.0, 0.0, 30.0, -50.0, yaw)
    psi = math.radians(yaw)
    fwd, right = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
    tgt = np.r_[30.0 * fwd - 8.0 * right, 0.0]
    u0 = project_point(tgt, cam, W, H, 90.0)[0]
    assert u0 < W / 2
    _, _, _, dyaw = map_action((0, 0, 0, -20), yaw, 0.20, 0.50)
    assert dyaw < 0
    u1 = project_point(tgt, (*cam[:4], yaw + dyaw), W, H, 90.0)[0]
    assert abs(u1 - W / 2) < abs(u0 - W / 2)


def _native_cmd(area, cx=256, ratio=0.0028, tol=0.20):
    """OA-VAT's own drone_control (external/OA-VAT, unmodified) on one frame → the command it sends."""
    from acot.runtime.oavat_full import OAVAT
    if OAVAT not in sys.path:
        sys.path.insert(0, OAVAT)
    from control.drone_control import compute_drone_action_while_tracking
    from control.pid import PIDController  # noqa: F401 — probe the module path
    sent = {}

    class V:
        def send_velocity(self, vx=0, vy=0, vz=0, yaw_rate=0):
            sent["cmd"] = (vx, vy, vz, yaw_rate)

    cfg = {"desired_width": 512, "desired_height": 512, "ctrl_max_speed": 40, "target_area_ratio": ratio,
           "area_tolerance_ratio": tol, "ctrl_dt": 0.1}
    pids = {k: PIDController(Kp=kp, Ki=0.0, Kd=kd) for k, kp, kd in (("yaw", 0.45, 0.1), ("vz", 0.5, 0.05),
                                                                     ("vy", 0.35, 0.15))}
    compute_drone_action_while_tracking((256, cx), area, cfg, V(), pids)
    return sent.get("cmd")


def test_native_controller_approach_retreat_and_tolerance():
    """setpoint 0.0028·512² ≈ 734 px² (±20 %): a smaller box (target far) → forward, a larger one → backward, inside
    the tolerance → no forward command; mapped through G2 at −50° pitch the displacement follows the heading."""
    try:
        far, near, ok = _native_cmd(300), _native_cmd(2000), _native_cmd(734)
    except ImportError as e:                                                     # OA-VAT deps absent on this host
        pytest.skip(f"OA-VAT control module not importable here: {e}")
    assert far[1] > 0 and near[1] < 0 and ok[1] == 0
    assert far[0] == 0 and far[2] == 0
    dx, _, _, _ = map_action(far, 0.0, 0.20, 0.50)
    bx, _, _, _ = map_action(near, 0.0, 0.20, 0.50)
    assert dx > 0 > bx
    right = _native_cmd(734, cx=400)
    assert right[3] > 0                                                          # target right → yaw right


# ------------------------------------------------------------------------------------ 1024 planner (plan v4 §10)
def test_planner_hold_and_vz_drop():
    from acot.runtime.oavat_closed import planner_cmd
    assert planner_cmd(None, False, (0, 5, 3, 2)) == (None, None, False)            # outside: hover, nothing held
    assert planner_cmd((0, 5, 0, 2), False, None) == ((0, 5, 0, 2), None, False)
    c, h, held = planner_cmd((0, 12.0, 7.0, -9.0), True, None)                         # planner sends: vz dropped
    assert c == (0.0, 12.0, 0.0, -9.0) and h == (0, 12.0, 7.0, -9.0) and not held
    c2, h2, held2 = planner_cmd(None, True, h)                                         # drain frame: held
    assert c2 == (0.0, 12.0, 0.0, -9.0) and h2 == h and held2
    c3, h3, _ = planner_cmd((0, 0, 0, 0), True, h)                                     # found: zero
    assert c3 == (0.0, 0.0, 0.0, 0.0) and h3 == (0, 0, 0, 0)
    assert map_action(c2, 0.0, 0.2, 0.5)[2] == 0.0                                     # no vertical motion


@pytest.mark.parametrize("start", [0, 1, 7, 13, 99, 1234, 8999])
def test_sim_clock_planner_step_is_eleven_frames(start):
    """dp_rescue's drain loop under SimClock: dp_dt = 1 s → exactly 10 drain reads, then the one matching read; it
    terminates for every start time (float rounding in the first test is absorbed by the advancing sleep)."""
    from acot.runtime.oavat_closed import SimClock
    clk, reads = SimClock(), [0]
    for _ in range(start):
        clk.tick()

    def read():
        reads[0] += 1
        clk.tick()                                         # the runtime ticks once per frame fed
    dt, read_dt = 1.0, 1.0 / 30.0                          # dp_dt default; read_hz = fps (0) → 30 (dp_rescue)
    t_end = clk.time() + dt
    last = clk.time() - read_dt
    while clk.time() < t_end:                              # verbatim structure of dp_rescue's drain loop
        now = clk.time()
        if now - last >= read_dt:
            read()
            last = now
        else:
            clk.sleep(min(0.002, max(0.0, t_end - now)))
    read()                                                 # read_and_log_fn → find_instance_by_features
    assert reads[0] == 11


def test_planner_match_reinit_is_not_a_new_observation():
    from acot.runtime.oavat_closed import is_dp_reinit
    assert is_dp_reinit((10.0, 20.0, 27.0, 27.0), [10, 20, 27, 27])                     # init with the match box
    assert not is_dp_reinit(None, [10, 20, 27, 27])                                     # ordinary selection
    assert not is_dp_reinit((10.0, 20.0, 27.0, 27.0), [11, 20, 27, 27])                 # a different (new) box
