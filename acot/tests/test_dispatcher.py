"""1006 §3.6 / G10: randomised shared-queue dispatch, infra retries on a different worker, DAM4SAM slot leasing."""
import json
import os
import threading
import time

from acot.eval import scheduler as S


class FakeRun:
    def __init__(self, root, arms, seeds, bad_worker=None, always_fail=(), sch=None, dam_arms=()):
        self.root = root
        self.R = {"arms": {a: {"method": {"identity": {"head": "dam4sam" if a in dam_arms else "tah"}}} for a in arms},
                  "seeds": [{"town": "Town05", "seed": s} for s in seeds],
                  "schedule": sch or {"block": 2, "seed": 7, "max_infra_attempts": 3}}
        self.I = {"services": {"dam4sam": {"instances": [{"host": "h", "port": 1}, {"host": "h", "port": 2}]}}}
        self.bad, self.always = bad_worker, set(always_fail)
        self.ran, self.lock, self.concurrent_svc, self.max_svc = [], threading.Lock(), 0, 0

    def arm_dir(self, arm):
        return os.path.join(self.root, arm)

    def status(self, arm, seed):
        return S.read_json(os.path.join(self.arm_dir(arm), "seeds", str(seed), "status.json"), {}) or {}

    def finalize(self, arm, seed, status, reason, final=False):
        st = self.status(arm, seed)
        st.update(status=status, reason=reason, final=final)
        S.write_json(os.path.join(self.arm_dir(arm), "seeds", str(seed), "status.json"), st)

    needs_dam4sam = staticmethod(S.Run.needs_dam4sam)
    acquire_service = S.Run.acquire_service
    release_service = S.Run.release_service
    _svc_lock = threading.Lock()

    def run_unit(self, arm, w, seeds):
        if w.get("svc_slot") is not None:
            with self.lock:
                self.concurrent_svc += 1
                self.max_svc = max(self.max_svc, self.concurrent_svc)
        time.sleep(0.01)
        for s in seeds:
            st = self.status(arm, s["seed"])
            att = st.get("attempts", []) + [{"worker": w["id"]}]
            fail = w["id"] == self.bad or s["seed"] in self.always
            st.update(attempts=att, status="failed" if fail else "complete", final=bool(fail), infra_retry=bool(fail))
            S.write_json(os.path.join(self.arm_dir(arm), "seeds", str(s["seed"]), "status.json"), st)
            with self.lock:
                self.ran.append((arm, s["seed"], w["id"]))
        if w.get("svc_slot") is not None:
            with self.lock:
                self.concurrent_svc -= 1


def go(run, n_workers=3):
    W = [{"id": f"w{i}", "carla_port": 2000 + i, "gpu": i, "tm_base": 9700} for i in range(n_workers)]
    d = S.Dispatcher(run, W)
    ts = [threading.Thread(target=d.work, args=(w,)) for w in W]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    return d


def test_all_done_and_order_shuffled(tmp_path):
    run = FakeRun(str(tmp_path), ["a", "b"], range(10))
    go(run)
    assert all(run.status(a, s)["status"] == "complete" for a in "ab" for s in range(10))
    plan = json.loads(open(os.path.join(str(tmp_path), "schedule.jsonl")).readline())
    arms_in_order = [u[0] for u in plan["order"]]
    assert arms_in_order != sorted(arms_in_order)                  # arms interleaved, not arm-by-arm


def test_bad_worker_seeds_retried_elsewhere(tmp_path):
    run = FakeRun(str(tmp_path), ["a"], range(12), bad_worker="w1")
    go(run)
    for s in range(12):
        st = run.status("a", s)
        assert st["status"] == "complete", (s, st)
        assert st["attempts"][-1]["worker"] != "w1"


def test_permanent_failure_stops_after_max_attempts(tmp_path):
    run = FakeRun(str(tmp_path), ["a"], range(4), always_fail={2})
    go(run)
    st = run.status("a", 2)
    assert st["status"] == "failed" and st["final"] and len(st["attempts"]) == 3
    assert len({a["worker"] for a in st["attempts"]}) == 3          # three different workers
    assert all(run.status("a", s)["status"] == "complete" for s in (0, 1, 3))


def test_dam4sam_slots_bound_concurrency(tmp_path):
    run = FakeRun(str(tmp_path), ["d"], range(12), sch={"block": 1, "seed": 1, "max_infra_attempts": 3}, dam_arms=("d",))
    go(run, n_workers=4)
    assert run.max_svc <= 2                                          # two instances
    assert all(run.status("d", s)["status"] == "complete" for s in range(12))


def test_schedule_survives_resolution():
    """regression (2026-10-06): resolve_experiment dropped `schedule`, so the Dispatcher was never used."""
    from acot import ROOT
    from acot.config.resolve import resolve_experiment
    R = resolve_experiment(os.path.join(ROOT, "acot/configs/experiments/t0_pilot_deploy_devB.yaml"))
    assert R["schedule"] == {"block": 5, "seed": 10062, "max_infra_attempts": 3}
