"""1006 §3.6 / G10: randomised shared-queue dispatch, infra retries on a different worker, DAM4SAM slot leasing."""
import collections
import json
import os
import subprocess
import threading
import time
import types

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
    d.threads = ts
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


class CrashRun(FakeRun):
    """10-19 deadlock: a unit raises out of run_unit (restart_carla's preload timed out) after its first seed's part was
    booked as an infra failure; the unit's later seeds never started."""

    def __init__(self, *a, crash_workers=(), **k):
        super().__init__(*a, **k)
        self.crash, self.calls = set(crash_workers), collections.Counter()

    def run_unit(self, arm, w, seeds):
        with self.lock:
            self.calls[w["id"]] += 1
        if w["id"] not in self.crash:
            return super().run_unit(arm, w, seeds)
        s = seeds[0]["seed"]
        st = self.status(arm, s)
        st.update(attempts=st.get("attempts", []) + [{"worker": w["id"]}], status="failed", final=True, infra_retry=True)
        S.write_json(os.path.join(self.arm_dir(arm), "seeds", str(s), "status.json"), st)
        raise RuntimeError("preload timed out")


def _fast_sleep(monkeypatch):
    real = time.sleep
    monkeypatch.setattr(S.time, "sleep", lambda x: real(min(x, 0.02)))


def test_crashing_worker_retired_and_its_seeds_requeued(tmp_path, monkeypatch):
    _fast_sleep(monkeypatch)
    run = CrashRun(str(tmp_path), ["a"], range(10), crash_workers={"w1"}, sch={"block": 5, "seed": 3, "max_infra_attempts": 3})
    d = go(run)
    assert not any(t.is_alive() for t in d.threads) and d.active == 0 and d.q == []
    assert d.retired == {"w1"} and run.calls["w1"] == 1                 # retired after its one crash, never reused
    assert all(run.status("a", s)["status"] == "complete" for s in range(10))
    crashed = [s for s in range(10) if any(a["worker"] == "w1" for a in run.status("a", s)["attempts"])]
    assert len(crashed) == 1 and len(run.status("a", crashed[0])["attempts"]) == 2   # its in-progress seed: retried once
    assert all(len(run.status("a", s)["attempts"]) == 1 for s in range(10) if s not in crashed)   # unstarted: no charge
    ends = [json.loads(l) for l in open(os.path.join(str(tmp_path), "schedule.jsonl")) if '"end"' in l]
    assert any(e.get("retired") and e["worker"] == "w1" for e in ends)


def test_every_worker_crashing_never_deadlocks(tmp_path, monkeypatch):
    _fast_sleep(monkeypatch)
    run = CrashRun(str(tmp_path), ["a"], range(6), crash_workers={"w0", "w1", "w2"})
    d = go(run)
    assert not any(t.is_alive() for t in d.threads) and d.active == 0
    assert d.retired == {"w0", "w1", "w2"}
    st = [run.status("a", s) for s in range(6)]
    assert not any(x.get("status") == "complete" for x in st)
    assert not any(x.get("final") for x in st)                           # left resumable, nothing finalised as failed


class InterruptRun(FakeRun):
    """codex R1: a BaseException out of run_unit, or a failure of the retry bookkeeping, must not leave `active` held."""

    def __init__(self, *a, mode, **k):
        super().__init__(*a, **k)
        self.mode, self.fired = mode, False

    def run_unit(self, arm, w, seeds):
        with self.lock:
            fire = w["id"] == "w1" and not self.fired
            self.fired |= fire
        if not fire:
            return super().run_unit(arm, w, seeds)
        if self.mode == "base":
            raise KeyboardInterrupt
        real, me = self.status, threading.get_ident()  # "bookkeeping": the next status() of THIS thread (re-queueing

        def once(*a):                                  # this unit) fails, once; other threads are unaffected
            if threading.get_ident() != me:
                return real(*a)
            self.status = real
            raise OSError("disk full")
        self.status = once
        raise RuntimeError("preload timed out")


def test_base_exception_or_bookkeeping_failure_never_deadlocks(tmp_path, monkeypatch):
    _fast_sleep(monkeypatch)
    for mode in ("base", "bookkeeping"):
        root = os.path.join(str(tmp_path), mode)
        os.makedirs(root)
        run = InterruptRun(root, ["a"], range(10), mode=mode, sch={"block": 5, "seed": 3, "max_infra_attempts": 3})
        d = go(run)
        assert not any(t.is_alive() for t in d.threads) and d.active == 0, mode
        assert d.retired == {"w1"}, mode
        done = [s for s in range(10) if run.status("a", s).get("status") == "complete"]
        assert len(done) == 5, (mode, done)            # the other unit ran; the interrupted one is left resumable
        assert not any(run.status("a", s).get("final") for s in range(10) if s not in done)


def _fake_run(tmp_path, monkeypatch, client_dies=True, first_sleep_raises=True, poll_rc=None, probe_runs=True):
    """a real Run whose docker / Popen / sleep are faked: the local docker exec reports poll_rc (None = still running);
    the in-container probe prints pgrep rc 1 (gone) only if client_dies, and only if probe_runs (else a docker transport
    error: rc 1 without the marker); the first time.sleep (the watchdog loop) raises if asked."""
    calls = []
    r = S.Run.__new__(S.Run)
    r.R = {"name": "x", "collect": ["gt_access"], "schedule": {"block": 1, "seed": 1, "max_infra_attempts": 3},
           "seeds": [{"town": "T", "seed": 5}, {"town": "T", "seed": 6}],
           "arms": {"a": {"config_hash": "h", "uses_gt": [], "protocol": {
               "camera": {"fps": 10, "width": 8, "height": 8, "fov_deg": 90},
               "sim": {"episode_steps": 9, "max_lost_s": 18, "retarget_tick": None},
               "scorer": {"commit_k": 5, "t_fail_s": 2, "central_frac": .3}, "perception": {"occlusion": "raycast"},
               "frontend": {"opening": {"hires_k": 3}}},
               "method": {"control": {"expert_control": False, "predict_road": False, "road_oracle": False}}}}}
    r.I = {"watchdog": {"max_restarts": 2, "first_min": 10, "stall_min": 10}, "policy_port": 1,
           "tm_ports": {"per_restart": 10}, "docker": {"carla_container": "c"}, "python": {"carla_client": "py"}}
    r.cprefix, r.max_attempts = "acot-", 3
    r.arm_dir = lambda arm: os.path.join(str(tmp_path), arm)
    r.ctr = lambda p: p
    r.blank_rules = lambda arm: {}
    r.start_policy = lambda arm, w, tag: ("acot-p", None)
    r.restart_carla = lambda w, town: calls.append("restart")
    os.makedirs(os.path.join(str(tmp_path), "a", "logs"), exist_ok=True)

    class Proc:
        killed = False

        def poll(self):
            return poll_rc

        def kill(self):
            Proc.killed = True

        def wait(self, timeout=None):
            calls.append("wait")

    def sh(args, **k):
        calls.append(" ".join(args[3:5]))
        if args[3] == "sh":                                  # the probe
            if not probe_runs:
                return types.SimpleNamespace(returncode=1, stdout="Cannot connect to the Docker daemon")
            return types.SimpleNamespace(returncode=0, stdout=f"acot_probe_rc={1 if client_dies else 0}\n")
        return types.SimpleNamespace(returncode=0, stdout="")
    state = {"first": first_sleep_raises}

    def sleep(x):
        if state["first"]:
            state["first"] = False
            raise RuntimeError("log read failed")
    monkeypatch.setattr(S.subprocess, "Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(S.docker, "sh", sh)
    monkeypatch.setattr(S.docker, "rm", lambda name, prefix, timeout=None: calls.append(f"rm {name}"))
    monkeypatch.setattr(S.time, "sleep", sleep)
    S.write_json(os.path.join(str(tmp_path), "a", "seeds", "5", "status.json"), {"seed": 5, "status": "partial"})
    return r, calls, Proc


W0 = {"id": "w0", "carla_port": 2000, "tm_base": 9700, "gpu": 0}
SEEDS = [{"town": "T", "seed": 5}, {"town": "T", "seed": 6}]


def test_run_unit_stops_the_client_and_books_the_part_on_an_exception(tmp_path, monkeypatch):
    """codex R1/R2: an exception inside the watchdog loop (client still running) → verified stop + post_part, re-raise."""
    import pytest
    r, calls, Proc = _fake_run(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="log read failed"):
        r.run_unit("a", W0, SEEDS)
    assert Proc.killed and "pkill -TERM" in calls and "sh -c" in calls and "rm acot-p" in calls
    st = r.status("a", 5)
    assert st["status"] == "failed" and st["infra_retry"] and st["attempts"][-1]["reason"] == "client died mid-episode"
    assert r.status("a", 6) == {}                    # never started: untouched → re-queued without a charge


def test_unconfirmed_client_stop_quarantines_the_part(tmp_path, monkeypatch):
    """codex R2: pkill 'succeeds' but pgrep still finds the client → the part's seeds are quarantined, never retried."""
    import pytest
    r, calls, _ = _fake_run(tmp_path, monkeypatch, client_dies=False)
    with pytest.raises(RuntimeError, match="log read failed"):
        r.run_unit("a", W0, SEEDS)
    assert calls.count("pkill -KILL") == 2                        # escalated, bounded
    for s in (5, 6):
        st = r.status("a", s)
        assert st["status"] == "failed" and st["final"] and st["quarantined"] and not st["infra_retry"], st
    d = types.SimpleNamespace(max_att=3, run=r)
    assert S.Dispatcher._requeue(d, {"arm": "a", "seeds": SEEDS, "avoid": []}, W0, RuntimeError("x")) == []


def test_docker_transport_error_is_never_read_as_stopped(tmp_path, monkeypatch):
    """codex R3: `docker exec … pgrep` rc 1 from a docker failure (no probe marker) → unconfirmed → quarantine."""
    import pytest
    r, calls, _ = _fake_run(tmp_path, monkeypatch, probe_runs=False)
    with pytest.raises(RuntimeError, match="log read failed"):
        r.run_unit("a", W0, SEEDS)
    assert all(r.status("a", s)["quarantined"] for s in (5, 6))


def test_abnormal_docker_exec_exit_is_verified(tmp_path, monkeypatch):
    """codex R3: a nonzero docker-exec exit goes through the verified stop: confirmed → booked as usual;
    unconfirmed → ClientNotStopped + quarantine."""
    import pytest
    r, calls, _ = _fake_run(tmp_path, monkeypatch, first_sleep_raises=False, poll_rc=1)
    r.run_unit("a", W0, SEEDS)                                    # confirmed gone each part → ordinary bookkeeping
    assert "sh -c" in calls and r.status("a", 5)["infra_retry"] and not r.status("a", 5).get("quarantined")
    r2, _, _ = _fake_run(tmp_path / "b", monkeypatch, first_sleep_raises=False, poll_rc=1, client_dies=False)
    with pytest.raises(S.ClientNotStopped):
        r2.run_unit("a", W0, SEEDS)
    assert all(r2.status("a", s)["quarantined"] for s in (5, 6))


def test_quarantine_survives_a_late_client_write(tmp_path, monkeypatch):
    """codex R3: a surviving client rewriting status.json must not lift the quarantine (quarantine.json overlay)."""
    import pytest
    r, _, _ = _fake_run(tmp_path, monkeypatch, client_dies=False)
    with pytest.raises(RuntimeError):
        r.run_unit("a", W0, SEEDS)
    S.write_json(os.path.join(str(tmp_path), "a", "seeds", "5", "status.json"), {"seed": 5, "status": "complete"})
    S.write_json(os.path.join(str(tmp_path), "a", "seeds", "6", "status.json"), {"seed": 6, "status": "partial"})
    for s in (5, 6):
        assert r.done("a", s) and r.status("a", s)["status"] == "failed" and r.status("a", s)["quarantined"]
    r.reconcile()                                                 # persisted into status.json again
    raw = S.read_json(os.path.join(str(tmp_path), "a", "seeds", "6", "status.json"))
    assert raw["status"] == "failed" and raw["final"] and raw["quarantined"]


def test_cleanup_logging_failure_still_quarantines(tmp_path, monkeypatch):
    """codex R3: diagnostics on the cleanup path (say) failing must not skip the quarantine."""
    import pytest
    r, _, _ = _fake_run(tmp_path, monkeypatch, probe_runs=False)
    monkeypatch.setattr(S.docker, "rm", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("docker rm", 120)))
    real_say = S.say
    monkeypatch.setattr(S, "say", lambda msg: (_ for _ in ()).throw(BrokenPipeError())
                        if msg.startswith(("kill_client", "stop_part")) else real_say(msg))
    with pytest.raises(RuntimeError, match="log read failed"):
        r.run_unit("a", W0, SEEDS)
    assert all(r.status("a", s)["quarantined"] for s in (5, 6))


def test_finalize_failure_in_take_aborts(tmp_path, monkeypatch):
    """codex R3: the 'every live worker avoided this unit' finalisation failing must abort, not kill a worker silently."""
    run = FakeRun(str(tmp_path), ["a"], range(2), sch={"block": 2, "seed": 1, "max_infra_attempts": 3})
    W = [{"id": "w0", "carla_port": 2000, "gpu": 0, "tm_base": 9700}]
    d = S.Dispatcher(run, W)
    d.q = [{"arm": "a", "seeds": [{"town": "T", "seed": 0}], "avoid": ["w0"]}]
    run.finalize = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    assert d._take(W[0]) is None and d.abort and len(d.q) == 1


def test_bookkeeping_failure_raises_and_partial_is_never_rerun_uncharged(tmp_path, monkeypatch):
    import pytest
    r, _, _ = _fake_run(tmp_path, monkeypatch)
    real = S.write_json
    monkeypatch.setattr(S, "write_json", lambda path, obj: (_ for _ in ()).throw(OSError("disk full"))
                        if path.endswith("status.json") else real(path, obj))
    with pytest.raises(S.BookkeepingError):
        r.run_unit("a", W0, SEEDS)
    assert r.status("a", 5)["status"] == "partial"                # nothing written …
    d = types.SimpleNamespace(max_att=3, run=r)
    with pytest.raises(S.BookkeepingError):                       # … and the Dispatcher refuses to re-queue it uncharged
        S.Dispatcher._requeue(d, {"arm": "a", "seeds": SEEDS, "avoid": []}, W0, RuntimeError("x"))


def test_reconcile_books_partials_and_reapplies_the_gt_audit(tmp_path, monkeypatch):
    r, _, _ = _fake_run(tmp_path, monkeypatch)
    S.write_json(os.path.join(str(tmp_path), "a", "seeds", "6", "status.json"), {"seed": 6, "status": "complete"})
    S.write_json(os.path.join(str(tmp_path), "a", "seeds", "6", "gt_access.json"), {"violations": ["x"], "unknown_sites": []})
    r.reconcile()
    st5, st6 = r.status("a", 5), r.status("a", 6)
    assert st5["status"] == "failed" and not st5["final"] and len(st5["attempts"]) == 1   # charged, resumable
    assert st6["status"] == "failed" and st6["reason"] == "gt_access_violation" and st6["final"]
    r.reconcile()                                                 # idempotent
    assert r.status("a", 5) == st5 and r.status("a", 6) == st6


def test_log_failure_aborts_dispatch_without_deadlock(tmp_path, monkeypatch):
    """codex R2: a failing start / end log line must neither keep a service slot nor leave a dead, un-retired worker."""
    _fast_sleep(monkeypatch)
    for event in ("start", "end"):
        root = os.path.join(str(tmp_path), event)
        os.makedirs(root)
        run = FakeRun(root, ["d"], range(8), sch={"block": 1, "seed": 2, "max_infra_attempts": 3}, dam_arms=("d",))
        W = [{"id": f"w{i}", "carla_port": 2000 + i, "gpu": i, "tm_base": 9700} for i in range(3)]
        disp = S.Dispatcher(run, W)
        real, fired = disp._log, []

        def log(rec, real=real, fired=fired, event=event):
            if rec.get("event") == event and not fired:
                fired.append(1)
                raise OSError("disk full")
            return real(rec)
        disp._log = log
        ts = [threading.Thread(target=disp.work, args=(w,)) for w in W]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in ts) and disp.active == 0 and disp.abort, event
        assert all(not lk.locked() for lk in run._svc), event       # no DAM4SAM slot left held


def test_no_double_charge_without_schedule(tmp_path, monkeypatch):
    """codex R4: without `schedule`, a booked partial must not be charged again by reconcile."""
    import pytest
    r, _, _ = _fake_run(tmp_path, monkeypatch)
    r.R.pop("schedule")
    r.max_attempts = 2
    with pytest.raises(RuntimeError, match="log read failed"):
        r.run_unit("a", W0, SEEDS)
    st = r.status("a", 5)
    assert st["status"] == "failed" and len(st["attempts"]) == 1 and not st.get("final")
    r.reconcile()
    r.reconcile()
    assert len(r.status("a", 5)["attempts"]) == 1 and not r.done("a", 5)


def test_stale_gt_audit_never_validates_a_new_attempt(tmp_path, monkeypatch):
    """codex R4: the previous attempt's clean gt_access.json is moved away before a part starts."""
    import pytest
    r, _, _ = _fake_run(tmp_path, monkeypatch)
    ga = os.path.join(str(tmp_path), "a", "seeds", "5", "gt_access.json")
    S.write_json(ga, {"violations": [], "unknown_sites": []})
    with pytest.raises(RuntimeError, match="log read failed"):
        r.run_unit("a", W0, SEEDS)
    assert not os.path.exists(ga) and os.path.exists(ga + ".stale_w0_p1")
    S.write_json(os.path.join(str(tmp_path), "a", "seeds", "5", "status.json"), {"seed": 5, "status": "complete"})
    r.reconcile()                                                 # the new attempt left no audit → not accepted
    assert r.status("a", 5)["status"] == "failed" and r.status("a", 5)["reason"] == "gt_access record missing"


def test_orphan_clients_block_the_start_unless_confirmed_gone(tmp_path, monkeypatch):
    """codex R4: a client left by a crashed scheduler must be confirmed gone before the run starts."""
    import pytest
    r, calls, _ = _fake_run(tmp_path, monkeypatch, first_sleep_raises=False)
    r.root = os.path.join(str(tmp_path), "exp")
    r.kill_orphans()                                              # probe: gone
    assert any(c == "pkill -TERM" for c in calls)
    r2, _, _ = _fake_run(tmp_path / "b", monkeypatch, first_sleep_raises=False, client_dies=False)
    r2.root = os.path.join(str(tmp_path), "exp")
    with pytest.raises(S.ClientNotStopped):
        r2.kill_orphans()


def test_regex_metacharacters_are_never_confirmed(tmp_path, monkeypatch):
    """codex R5: a path such as review[1] must neither be probed as a regex nor let a run start."""
    import pytest
    r, calls, _ = _fake_run(tmp_path, monkeypatch, first_sleep_raises=False)
    assert r.kill_pattern("episode.py --client /w/exp/review[1]/a/parts/w0_p1.client.json") is False
    assert not any(c.startswith(("pkill", "sh")) for c in calls)   # nothing was even attempted
    assert r.kill_pattern("episode.py --client /w/exp/flight_ceiling_devB3/") is True
    for root, arm in (("exp/review[1]", "a"), ("exp/ok", "a(b)")):
        r.root = os.path.join(str(tmp_path), root)
        r.R["arms"] = {arm: r.R["arms"].get("a") or next(iter(r.R["arms"].values()))}
        r.arm_dir = lambda a, root=r.root: os.path.join(root, a)
        with pytest.raises(ValueError, match="regex metacharacters"):
            r.kill_orphans()


def test_schedule_survives_resolution():
    """regression (2026-10-06): resolve_experiment dropped `schedule`, so the Dispatcher was never used."""
    from acot import ROOT
    from acot.config.resolve import resolve_experiment
    R = resolve_experiment(os.path.join(ROOT, "acot/configs/experiments/t0_pilot_deploy_devB.yaml"))
    assert R["schedule"] == {"block": 5, "seed": 10062, "max_infra_attempts": 3}


def test_workers_two_policy_slots_per_gpu(tmp_path, monkeypatch):
    """10-22 user cap (≤4 GPUs): a GPU listed twice in --gpus hosts two policy containers under distinct leases."""
    from acot.infra import lease as L
    monkeypatch.setattr(L, "DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(S, "carla_up", lambda I, p: True)
    monkeypatch.setattr(S, "gpu_free_mib", lambda g: 40000)
    run = types.SimpleNamespace(
        I={"carla_servers": [{"port": 2000 + 10 * i, "adapter": i // 2} for i in range(4)],
           "reserved": {"carla_ports": []}, "tm_ports": {"base": 9700, "per_worker": 30}},
        a=types.SimpleNamespace(carla_ports=None, gpus="0,0,1,1"), R={"name": "t"})
    W = S.Run.workers(run, 4)
    assert [w["gpu"] for w in W] == [0, 0, 1, 1] and [w["slot"] for w in W] == [0, 1, 2, 3]
    assert sorted(os.listdir(tmp_path / "leases"))[:4] == ["carla_2000.lock", "carla_2010.lock", "carla_2020.lock",
                                                           "carla_2030.lock"]
    assert {"gpu_0.lock", "gpu_0.1.lock", "gpu_1.lock", "gpu_1.1.lock"} <= set(os.listdir(tmp_path / "leases"))
    run.a.gpus = "0,1"                                    # no duplicates → one container per GPU, as before
    for w in W:
        w["lease"].release()
    assert [w["gpu"] for w in S.Run.workers(run, 4)] == [0, 1]
