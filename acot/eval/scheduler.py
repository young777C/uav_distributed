"""`acot eval` (framework §8): resolve → preflight → per-arm spec → worker plan → for every (arm, worker) unit a policy
container (infra.runtime: legacy = legacy_server_entry with the L1 echo check | acot = acot/runtime/server.py, which
reads the resolved config and must echo its hash) + CARLA client parts (acot_client/episode.py) under a watchdog
(hang → kill client, restart that CARLA server, retry the seed); per-seed status (complete / skipped / failed /
partial), GT-audit verdicts, experiment manifest. Results: <results_root>/<experiment>/<arm>/seeds/<seed>/."""
import json
import os
import re
import shlex
import subprocess
import threading
import time

from acot import ROOT
from acot import provenance
from acot.config.legacy import semantic_from_echo, semantic_of, shell_join, translate
from acot.config.resolve import resolve_experiment
from acot.eval.dry_run import seed_blocks
from acot.infra import docker
from acot.infra.lease import Lease
from acot.infra.preflight import carla_up, check_runtime, check_static, gpu_free_mib

DONE = ("complete", "skipped")
PRINT = threading.Lock()
# pkill / pgrep -f read their pattern as an extended regex; a metacharacter in a client path could make the probe miss
# the real client and "confirm" it stopped (codex R5). '.' only over-matches, which is safe here.
REGEX_META = re.compile(r"[][(){}*+?|^$\\]")


class BookkeepingError(RuntimeError):
    """a per-seed status / attempt / GT-audit record could not be written: the run state is no longer trusted, so the
    Dispatcher stops handing out units (codex R2, 10-19); `Run.reconcile` repairs what it can at the next start."""


class ClientNotStopped(RuntimeError):
    """the client of an abandoned part could not be confirmed stopped inside the CARLA container; its seeds are
    quarantined (failed, final, not retried) so that no second client ever runs them concurrently."""


def say(msg):
    with PRINT:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def say_quiet(msg):
    """diagnostics on a cleanup path must never raise (codex R3: a BrokenPipe there skipped the quarantine)."""
    try:
        say(msg)
    except Exception:
        pass


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def parse_seed_filter(s):
    out = set()
    for part in (s or "").split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        elif part.strip():
            out.add(int(part))
    return out


class Run:
    _svc_lock = threading.Lock()

    def __init__(self, R, args):
        self.R, self.a = R, args
        I = R["infra"]
        self.I = I
        self.root = os.path.join(ROOT, I["results_root"], R["name"])
        self.max_attempts = int(R["retry"].get("infra_attempts", 2))
        self.cprefix = I["docker"]["policy_container_prefix"]
        self.runtime = I.get("runtime") or "legacy"

    # ------------------------------------------------------------------------------------------------------- paths
    def arm_dir(self, arm):
        return os.path.join(self.root, arm)

    def ctr(self, host_path):
        """host path under carla_uav_tracking/ → the same path inside cyh-carla (/workspace/…)."""
        base = os.path.join(ROOT, "carla_uav_tracking")
        assert host_path.startswith(base), host_path
        return "/workspace" + host_path[len(base):]

    def status(self, arm, seed):
        """the seed's status.json, overlaid by quarantine.json when present (codex R3: the client may still rewrite
        status.json after a quarantine; quarantine.json is written only here and is never touched by the client)."""
        sd = os.path.join(self.arm_dir(arm), "seeds", str(seed))
        st = read_json(os.path.join(sd, "status.json"), {}) or {}
        q = read_json(os.path.join(sd, "quarantine.json"))
        if q is not None:
            st.update(status="failed", reason=q.get("reason", "quarantined"), final=True, infra_retry=False,
                      quarantined=True)
        return st

    def done(self, arm, seed):
        st = self.status(arm, seed)
        return st.get("status") in DONE or (st.get("status") == "failed" and st.get("final"))

    # ----------------------------------------------------------------------------------------------------- planning
    def workers(self, n):
        """the first n CARLA servers that answer and n policy GPU slots with free memory, leased for this run. A GPU
        listed k times in --gpus hosts up to k policy containers (lease gpu_<g>, then gpu_<g>.1, …)."""
        I, out = self.I, []
        ports = [s for s in I["carla_servers"]
                 if not any(lo <= s["port"] <= hi for lo, hi in I["reserved"]["carla_ports"])]
        if getattr(self.a, "carla_ports", None):
            want = [int(p) for p in self.a.carla_ports.split(",")]
            ports = [s for p in want for s in I["carla_servers"] if s["port"] == p]
        gpus = [int(g) for g in self.a.gpus.split(",")] if getattr(self.a, "gpus", None) else \
            [g for g in I["policy_gpus"] if g not in I["reserved"]["gpus"]]
        for s in ports:
            if len(out) == n:
                break
            if not carla_up(I, s["port"]):
                continue
            slot = next((i for i, g in enumerate(gpus) if i not in [w["slot"] for w in out]
                         and gpu_free_mib(g) >= 20000), None)
            if slot is None:
                break
            g, dup = gpus[slot], gpus[:slot].count(gpus[slot])
            lease = Lease([f"carla_{s['port']}", f"gpu_{g}" if dup == 0 else f"gpu_{g}.{dup}"], owner=self.R["name"])
            try:
                lease.acquire()
            except RuntimeError as e:
                say(f"skip worker on port {s['port']} / gpu {g}: {e}")
                continue
            tm = None
            for blk in range(64):                    # TM port blocks are leased ACROSS concurrent runs
                try:
                    lease.fds += Lease([f"tm_block_{blk}"], owner=self.R["name"]).acquire().fds
                    tm = I["tm_ports"]["base"] + I["tm_ports"]["per_worker"] * blk
                    break
                except RuntimeError:
                    continue
            if tm is None:
                lease.release()
                say("no free TM port block")
                break
            k = len(out)
            out.append({"id": f"w{k}", "carla_port": s["port"], "adapter": s["adapter"], "gpu": g, "slot": slot,
                        "tm_base": tm, "lease": lease})
        return out

    def plan(self, seeds, nworkers):
        """per arm: seeds split into contiguous same-town chunks over the workers → [(arm, worker index, seeds)]."""
        units = []
        for arm in self.R["arms"]:
            for town, block in seed_blocks(seeds):
                k = min(nworkers, len(block))
                size = -(-len(block) // k)
                for i in range(k):
                    chunk = block[i * size:(i + 1) * size]
                    if chunk:
                        units.append((arm, i, chunk))
        return units

    # ------------------------------------------------------------------------------------------------- arm specs
    def write_specs(self):
        R = self.R
        for name, arm in R["arms"].items():
            inv = translate(R, name)
            d = self.arm_dir(name)
            os.makedirs(os.path.join(d, "logs"), exist_ok=True)
            fence = {"raw": "gt_access" in R["collect"], "labels": "gt_access" in R["collect"]}
            spec = {"experiment": R["name"], "arm": name, "config_hash": arm["config_hash"], "uses_gt": arm["uses_gt"],
                    "role": arm["role"], "results_dir": d, "collect": R["collect"], "policy_argv": inv["policy_argv"],
                    "policy_env": inv["policy_env"], "gt_fence": fence, "repo": ROOT,
                    "semantic": self._semantic(inv), "record_act_ticks": int(self.I.get("record_act_ticks") or 120),
                    "runtime": self.runtime,
                    "services": self.services(arm)}
            write_json(os.path.join(d, "resolved_spec.json"), spec)
            write_json(os.path.join(d, "resolved.json"), {k: v for k, v in arm.items()})

    def _semantic(self, inv):
        """the legacy-argparse reading of the arm (an informational record for the acot runtime; required for the
        legacy runtime). Arms the legacy policy cannot express (e.g. identity.head=oavat, 1006) make the legacy argparse
        exit — fatal on the legacy runtime, recorded as None on the acot runtime."""
        try:
            return semantic_of(inv)
        except (SystemExit, Exception) as e:                   # noqa: BLE001
            if self.runtime == "acot":
                return {"unavailable": f"legacy argparse cannot express this arm: {type(e).__name__} {e}"}
            raise

    def services(self, arm):
        """external services an arm's policy connects to (acot runtime: passed in the spec, not as argv)."""
        if self.needs_dam4sam(arm):
            insts = self.I["services"]["dam4sam"]["instances"]
            return {"dam4sam": {"host": insts[0]["host"], "port": insts[0]["port"],
                                "instances": [{"host": i["host"], "port": i["port"]} for i in insts]}}
        return {}

    @staticmethod
    def needs_dam4sam(arm):
        return arm["method"]["identity"]["head"] in ("dam4sam", "fusion")

    def acquire_service(self, arm_name):
        """1006 G10: one DAM4SAM instance per concurrently running unit (each instance accepts one client)."""
        if not self.needs_dam4sam(self.R["arms"][arm_name]):
            return None
        n = len(self.I["services"]["dam4sam"]["instances"])
        with self._svc_lock:
            if not hasattr(self, "_svc"):
                self._svc = [threading.Lock() for _ in range(n)]
        while True:
            for i, lk in enumerate(self._svc):
                if lk.acquire(blocking=False):
                    return i
            time.sleep(5)

    def release_service(self, slot):
        if slot is not None:
            self._svc[slot].release()

    def blank_rules(self, arm):
        uses = set(self.R["arms"][arm]["uses_gt"])
        labels = {"opening_snapshots", "dagger"} & set(self.R["collect"])
        return {"target_attrs": not ({"attrbind_gt_attrs"} & uses or labels),
                "road": not ({"predict_road", "road_oracle"} & uses), "retarget": "retarget" not in uses}

    # ------------------------------------------------------------------------------------------------ containers
    def cname(self, arm, w):
        return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{self.cprefix}{self.R['name']}-{arm}-{w['id']}")

    def start_policy(self, arm, w, tag):
        if self.runtime == "acot":
            return self.start_policy_acot(arm, w, tag)
        I, d = self.I, self.arm_dir(arm)
        spec = read_json(os.path.join(d, "resolved_spec.json"))
        env = dict(spec["policy_env"])
        for k in ("ASD_LOG", "TAH_DAGGER_DIR", "DAGGER_DIR", "OPENING_DUMP"):
            if env.get(k):
                env[k] = env[k].replace("asd_w0.jsonl", f"asd_{tag}.jsonl")
        name = self.cname(arm, w)
        if docker.running(name) or docker.inspect("container", name):
            docker.rm(name, self.cprefix)
        hf = os.path.expanduser(I["paths"]["hf_cache"])
        cmd = ["docker", "run", "-d", "--gpus", f"device={w['gpu']}", "--network", I["docker"]["network"], "--name", name,
               "--shm-size", I["docker"]["shm_size"], "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
               "-e", f"OMP_NUM_THREADS={I['threads']['omp']}", "-e", f"MKL_NUM_THREADS={I['threads']['omp']}"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["-v", f"{ROOT}:{ROOT}", "-v", f"{I['paths']['data']}:{I['paths']['data']}:ro",
                "-v", f"{hf}:/root/.cache/huggingface", "-w", ROOT, I["docker"]["policy_image"], "bash", "-c",
                f"pip install -q {I['pip_preinstall']} 2>/dev/null; python -m acot.eval.legacy_server_entry "
                f"--spec {os.path.join(d, 'resolved_spec.json')} --worker {tag} -- {shell_join(spec['policy_argv'])}"]
        docker.sh(cmd, check=True)
        t0, limit = time.time(), self.I["watchdog"]["first_min"] * 60
        while time.time() - t0 < limit:
            logs = docker.sh(["docker", "logs", name]).stdout or ""
            if "listening" in logs:
                break
            if not docker.running(name):
                return name, f"policy container exited:\n{logs[-1500:]}"
            time.sleep(5)
        else:
            return name, "policy server did not start listening in time"
        echo = read_json(os.path.join(d, f"server_echo_{tag}.json"))
        if echo is None:
            return name, "no server echo"
        inv = translate(self.R, arm)
        got = semantic_from_echo(echo, inv["client"]["argv"], inv["client"]["env"])
        want = semantic_of(inv)
        diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want) if k != "client" and got.get(k) != want.get(k)}
        write_json(os.path.join(d, f"echo_check_{tag}.json"), {"ok": not diff, "diff": diff, "config_hash": echo["config_hash"]})
        if diff:
            return name, f"L1 echo mismatch: {json.dumps(diff, default=str)[:1500]}"
        return name, None

    def _docker_run_base(self, name, gpu, image, env=()):
        I = self.I
        cmd = ["docker", "run", "-d", "--gpus", f"device={gpu}", "--network", I["docker"]["network"], "--name", name,
               "--shm-size", I["docker"]["shm_size"], "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
               "-e", f"OMP_NUM_THREADS={I['threads']['omp']}", "-e", f"MKL_NUM_THREADS={I['threads']['omp']}"]
        for k, v in env:
            cmd += ["-e", f"{k}={v}"]
        hf = os.path.expanduser(I["paths"]["hf_cache"])
        return cmd + ["-v", f"{ROOT}:{ROOT}", "-v", f"{I['paths']['data']}:{I['paths']['data']}:ro",
                      "-v", f"{hf}:/root/.cache/huggingface", "-w", ROOT, image]

    def _wait_listening(self, name):
        t0, limit = time.time(), self.I["watchdog"]["first_min"] * 60
        while time.time() - t0 < limit:
            logs = docker.sh(["docker", "logs", name]).stdout or ""
            if "listening" in logs:
                return None
            if not docker.running(name):
                return f"policy container exited:\n{logs[-1500:]}"
            time.sleep(5)
        return "policy server did not start listening in time"

    def acot_policy_cmd(self, arm, w, tag, image, image_id):
        """the acot runtime's policy container: no behaviour env, no pip install, the resolved spec only."""
        cmd = self._docker_run_base(self.cname(arm, w), w["gpu"], image,
                                    env=(("ACOT_IMAGE", image), ("ACOT_IMAGE_ID", image_id)))
        return cmd + ["python", "-m", "acot.runtime.server", "--spec",
                      os.path.join(self.arm_dir(arm), "resolved_spec.json"), "--worker", tag,
                      "--port", str(self.I["policy_port"]), "--service-slot", str(w.get("svc_slot") or 0)]

    def start_policy_acot(self, arm, w, tag):
        """acot runtime (framework §14 P2): the server reads <arm>/resolved.json and must echo the arm's config hash;
        no behaviour env, no pip install (timm is pinned in policy_image_acot)."""
        I, d = self.I, self.arm_dir(arm)
        image = I["docker"].get("policy_image_acot")
        img = docker.inspect("image", image) if image else None
        if img is None:
            return None, f"acot policy image {image!r} missing (bash acot/docker/build_acot_image.sh)"
        name = self.cname(arm, w)
        if docker.running(name) or docker.inspect("container", name):
            docker.rm(name, self.cprefix)
        docker.sh(self.acot_policy_cmd(arm, w, tag, image, img["Id"]), check=True)
        err = self._wait_listening(name)
        if err:
            return name, err
        echo = read_json(os.path.join(d, f"server_echo_{tag}.json"))
        if echo is None:
            return name, "no server echo"
        want = self.R["arms"][arm]["config_hash"]
        ok = echo.get("runtime") == "acot" and echo.get("config_hash") == want and echo.get("image_id") == img["Id"]
        write_json(os.path.join(d, f"echo_check_{tag}.json"), {"ok": ok, "runtime": "acot", "config_hash": echo.get("config_hash"),
                                                               "want": want, "image_id": echo.get("image_id")})
        if not ok:
            return name, f"L1 echo mismatch: acot server runs {echo.get('config_hash')} on {echo.get('image_id')}, " \
                         f"want {want} on {img['Id']}"
        return name, None

    def restart_carla(self, w, town):
        I, P, A = self.I, w["carla_port"], w["adapter"]
        c, py = I["docker"]["carla_container"], I["python"]["carla_client"]
        say(f"{w['id']}: restarting CARLA {P} (adapter {A}) and preloading {town}")
        docker.sh(["docker", "exec", c, "pkill", "-9", "-f", f"carla-rpc-port={P}( |$)"])
        time.sleep(5)
        docker.sh(["docker", "exec", "-d", c, "bash", "-c", f"/home/carla/CarlaUE4.sh -RenderOffScreen -graphicsadapter={A} "
                   f"-carla-rpc-port={P} -nosound -fps=20 > /tmp/carla_{P}.log 2>&1"])
        for _ in range(60):
            if carla_up(I, P):
                break
            time.sleep(10)
        time.sleep(20)
        # 10-22: a fresh server under host load can need > 4 min for its first load_world; one slow preload used to
        # retire the worker for the rest of the run (2030, 2050). Wait longer and retry once before giving up.
        for attempt in range(2):
            try:
                docker.exec_py(c, f"import carla; cl = carla.Client('127.0.0.1', {P}); cl.set_timeout(360); "
                                  f"cl.load_world('{town}')", python=py, timeout=420)
                return
            except subprocess.TimeoutExpired:
                if attempt:
                    raise
                say(f"{w['id']}: preloading {town} on CARLA {P} timed out — retrying once")

    # -------------------------------------------------------------------------------------------------------- unit
    def run_unit(self, arm, w, seeds):
        d, town = self.arm_dir(arm), seeds[0]["town"]
        all_seeds = [s["seed"] for s in seeds]
        restarts, part = 0, 0
        while True:
            pending = [s for s in all_seeds if not self.done(arm, s)]
            if not pending or restarts > self.I["watchdog"]["max_restarts"]:
                break
            part += 1
            tag = f"{w['id']}_p{part}"
            name, err = self.start_policy(arm, w, tag)
            if err:
                say(f"{arm}/{tag}: {err}")
                docker.rm(name, self.cprefix)
                if "L1 echo mismatch" in err or "argv differs" in err or "env differs" in err:
                    for s in pending:
                        self.finalize(arm, s, "failed", "config_echo_mismatch", final=True)
                    return
                restarts += 1
                continue
            client = {"experiment": self.R["name"], "arm": arm, "config_hash": self.R["arms"][arm]["config_hash"],
                      "worker": w["id"], "part": part, "town": town, "seeds": pending,
                      "policy_host": name, "policy_port": self.I["policy_port"], "carla_port": w["carla_port"],
                      "results_dir": self.ctr(d), "blank": self.blank_rules(arm)}
            proto = self.R["arms"][arm]["protocol"]
            meth = self.R["arms"][arm]["method"]
            client.update(fps=proto["camera"]["fps"], resolution=[proto["camera"]["width"], proto["camera"]["height"]],
                          fov_deg=proto["camera"]["fov_deg"], steps=proto["sim"]["episode_steps"],
                          scorer={"commit_k": proto["scorer"]["commit_k"], "t_fail_s": proto["scorer"]["t_fail_s"],
                                  "max_lost_s": proto["sim"]["max_lost_s"], "central_frac": proto["scorer"]["central_frac"]},
                          env={"vis_raycast": proto["perception"]["occlusion"] == "raycast",
                               "hires_k": proto["frontend"]["opening"]["hires_k"],
                               "expert_control": meth["control"]["expert_control"],
                               "expert_look_away": meth["control"].get("expert_look_away", True),   # 1024 HOW R2
                               "predict_road": meth["control"]["predict_road"],
                               "road_oracle": meth["control"]["road_oracle"],
                               "retarget_tick": proto["sim"]["retarget_tick"],
                               "determinism": dict(proto["sim"].get("env_determinism") or {})})
            if proto["sim"].get("scenario", "mvp") != "mvp":   # 1020: non-default scenario file (mvp: as before)
                client["scenario"] = proto["sim"]["scenario"]
            cj = os.path.join(d, "parts", f"{tag}.client.json")
            for sd in pending:                                  # codex R4: an earlier attempt's audit record must
                ga = os.path.join(d, "seeds", str(sd), "gt_access.json")   # never validate this attempt
                if os.path.exists(ga):
                    os.replace(ga, ga + f".stale_{tag}")
            write_json(cj, client)
            tm = w["tm_base"] + self.I["tm_ports"]["per_restart"] * restarts
            log = os.path.join(d, "logs", f"client_{tag}.log")
            dl = ["-e", f"DUMP_LABELS={self.ctr(os.path.join(d, 'collect', 'actor_labels'))}"] \
                if "actor_labels" in self.R["collect"] else []
            cmd = ["docker", "exec", "-e", f"CARLA_PORT={w['carla_port']}", "-e", f"CARLA_TM_PORT={tm}", *dl,
                   self.I["docker"]["carla_container"], self.I["python"]["carla_client"],
                   "/workspace/acot_client/episode.py", "--client", self.ctr(cj)]
            say(f"{arm}/{tag}: {len(pending)} seed(s) {pending[0]}… on CARLA {w['carla_port']} GPU {w['gpu']} TM {tm}")
            with open(log, "w") as fh:
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            hung, seen, last, limit = False, 0, time.time(), self.I["watchdog"]["first_min"] * 60
            try:
                while proc.poll() is None:
                    time.sleep(10)
                    n = sum(1 for x in open(log, errors="ignore") if x.startswith("[acot] seed="))
                    if n > seen:
                        seen, last, limit = n, time.time(), self.I["watchdog"]["stall_min"] * 60
                        say(f"{arm}/{tag}: {n}/{len(pending)} done")
                    # 10-22: a seed that still writes ticks is alive — slow self-flight episodes (> stall_min under
                    # GPU sharing) were killed mid-way although progressing; stall = no new tick for `limit`
                    tick_t = max((os.path.getmtime(p) for p in (os.path.join(self.arm_dir(arm), "seeds", str(s),
                                  "ticks.jsonl") for s in pending) if os.path.exists(p)), default=0)
                    if tick_t > last:
                        last = tick_t
                    if time.time() - last > limit:
                        hung = True
                        say(f"{arm}/{tag}: HANG after {seen} seed(s) — killing the client")
                        break
                if (hung or proc.poll() != 0) and not self.stop_part(proc, cj, name):   # a hang, or the docker exec
                    raise ClientNotStopped(f"{arm}/{tag}: the client could not be confirmed stopped "     # ended
                                           f"({'hang' if hung else f'docker exec rc {proc.poll()}'})")   # abnormally
            except BaseException as e:                          # codex R1/R2 (10-19): never leave this part's client
                stopped = not isinstance(e, ClientNotStopped) and self.stop_part(proc, cj, name)   # running when the
                self.book(arm, pending, quarantine=None if stopped else "client termination unconfirmed")  # unit is
                raise                                           # abandoned; book the interrupted part (or quarantine)
            try:
                docker.rm(name, self.cprefix, timeout=300)      # after close → the policy flushed its logs
            except Exception as e:
                say_quiet(f"{arm}/{tag}: docker rm {name}: {type(e).__name__}: {e}")
            self.book(arm, pending)
            if hung or seen == 0:
                restarts += 1
                self.restart_carla(w, town)

    def kill_client(self, cj, tries=(("-TERM", 5), ("-TERM", 10), ("-KILL", 5), ("-KILL", 10))):
        """stop the part's client inside the CARLA container and CONFIRM it is gone; bounded (every docker call has a
        timeout), escalating TERM → KILL. Confirmation needs an in-container probe that RAN (docker rc 0) and printed
        pgrep's rc 1 (codex R3: a docker transport error must never read as 'gone'); the [e] keeps the probe's own
        shell out of the match. → True only when confirmed."""
        return self.kill_pattern(f"episode.py --client {self.ctr(cj)}", tries)

    def kill_pattern(self, pat, tries=(("-TERM", 5), ("-TERM", 10), ("-KILL", 5), ("-KILL", 10))):
        """kill_client for any `pkill -f` pattern starting with 'episode.py' (see kill_client). A pattern with a regex
        metacharacter is never confirmed (→ False: quarantine / refuse)."""
        if REGEX_META.search(pat):
            say_quiet(f"kill_pattern: regex metacharacter in {pat!r} — termination cannot be confirmed")
            return False
        c = self.I["docker"]["carla_container"]
        probe = f"pgrep -f {shlex.quote('[e]' + pat[1:])} >/dev/null; echo acot_probe_rc=$?"
        for sig, wait_s in tries:
            try:
                docker.sh(["docker", "exec", c, "pkill", sig, "-f", pat], timeout=60)
                time.sleep(wait_s)
                p = docker.sh(["docker", "exec", c, "sh", "-c", probe], timeout=60)
                if p.returncode == 0 and "acot_probe_rc=1" in (p.stdout or ""):
                    return True
            except Exception as e:
                say_quiet(f"kill_client {pat}: {type(e).__name__}: {e}")
        return False

    def stop_part(self, proc, cj, name):
        """teardown of one part (hang, abnormal exit or exception): the client in the CARLA container (verified), the
        local docker-exec process, the policy container; bounded, every step attempted, never raises.
        → True only when the client is confirmed gone."""
        try:
            stopped = self.kill_client(cj)
        except Exception:
            stopped = False
        for step in (proc.kill, lambda: proc.wait(timeout=60), lambda: docker.rm(name, self.cprefix, timeout=120)):
            try:
                step()
            except Exception as e:
                say_quiet(f"stop_part {name}: {type(e).__name__}: {e}")
        return stopped

    def book(self, arm, seeds, quarantine=None):
        """post_part, or (client not confirmed stopped) quarantine the part's unfinished seeds: failed, final, never
        retried. Any write failure → BookkeepingError (the Dispatcher then stops handing out units)."""
        try:
            if quarantine is None:
                return self.post_part(arm, seeds)
            for s in seeds:
                st = self.status(arm, s)
                if st.get("status") not in DONE:         # unfinished now; quarantine.json overrides any later write
                    sd = os.path.join(self.arm_dir(arm), "seeds", str(s))   # of the surviving client
                    write_json(os.path.join(sd, "quarantine.json"), {"reason": quarantine, "t": time.time()})
                    st = self.status(arm, s)
                    st.setdefault("attempts", []).append({"status": "quarantined", "reason": quarantine})
                    write_json(os.path.join(sd, "status.json"), st)
        except Exception as e:
            raise BookkeepingError(f"{arm} {seeds}: {type(e).__name__}: {e}") from e

    def kill_orphans(self):
        """codex R4: a client of THIS experiment still running inside the CARLA container (left by a crashed / killed
        scheduler) must be confirmed gone before any of its seeds is reconciled or run again; the experiment lease
        (run_experiment) guarantees no other live invocation owns such a client. → raises if not confirmed."""
        bad = [p for p in [self.ctr(self.root)] + [self.ctr(self.arm_dir(a)) for a in self.R["arms"]]
               if REGEX_META.search(p)]
        if bad:                                          # codex R5: every client path must be matchable literally
            raise ValueError(f"regex metacharacters in result paths {bad} — rename the experiment / arms "
                             f"(client termination could not be confirmed)")
        if not self.kill_pattern(f"episode.py --client {self.ctr(self.root)}/"):
            raise ClientNotStopped(f"clients of {self.R['name']} in the CARLA container could not be confirmed stopped "
                                   f"— refusing to start (stop them by hand, then resume)")

    def reconcile(self):
        """idempotent repair at the start and end of a run (codex R2): a `partial` left by a crashed run is booked as an
        attempt (failed; final once the attempts are used up); a `complete` gets the GT-audit verdict re-applied."""
        cap = int((self.R.get("schedule") or {}).get("max_infra_attempts", self.max_attempts))
        uses_gt = {a: self.R["arms"][a]["uses_gt"] for a in self.R["arms"]}
        for arm in self.R["arms"]:
            for sd in self.R["seeds"]:
                s = sd["seed"]
                st = self.status(arm, s)
                if st.get("quarantined"):
                    raw = read_json(os.path.join(self.arm_dir(arm), "seeds", str(s), "status.json"), {}) or {}
                    if raw.get("quarantined") and raw.get("status") == "failed" and raw.get("final"):
                        continue
                elif st.get("status") == "partial":
                    st.setdefault("attempts", []).append({"status": "partial", "reason": "client died mid-episode (reconciled)"})
                    st.update(status="failed", infra_retry=False, final=len(st["attempts"]) >= cap)
                elif not (st.get("status") == "complete" and "gt_access" in self.R["collect"]):
                    continue
                elif (v := self.audit_verdict(arm, s, uses_gt[arm])) is not None:
                    st.update(status="failed", reason=v, final=True)
                else:
                    continue
                write_json(os.path.join(self.arm_dir(arm), "seeds", str(s), "status.json"), st)

    def audit_verdict(self, arm, seed, uses):
        g = read_json(os.path.join(self.arm_dir(arm), "seeds", str(seed), "gt_access.json"))
        if g is None:
            return "gt_access record missing"
        if g["violations"] or (g["unknown_sites"] and not uses):
            return "gt_access_violation"
        return None

    def post_part(self, arm, seeds):
        """per-attempt bookkeeping + GT-audit verdicts for the seeds of a finished part."""
        uses = self.R["arms"][arm]["uses_gt"]
        for s in seeds:
            st = self.status(arm, s)
            if not st:
                continue
            if st["status"] == "partial":
                st.setdefault("attempts", []).append({"status": "partial", "reason": "client died mid-episode"})
                st["status"] = "failed"              # booked (codex R4): reconcile only ever charges UNbooked partials
            if st["status"] in ("partial", "failed") and self.R.get("schedule") and not st.get("final"):
                st.update(status="failed", final=True, infra_retry=True)
            elif st["status"] in ("partial", "failed") and len(st.get("attempts", [])) >= self.max_attempts:
                st.update(status="failed", final=True)
            if st["status"] == "complete" and "gt_access" in self.R["collect"]:
                v = self.audit_verdict(arm, s, uses)
                if v is not None:
                    st.update(status="failed", reason=v, final=True)
            write_json(os.path.join(self.arm_dir(arm), "seeds", str(s), "status.json"), st)

    def finalize(self, arm, seed, status, reason, final=False):
        p = os.path.join(self.arm_dir(arm), "seeds", str(seed), "status.json")
        st = read_json(p, {}) or {"seed": seed}
        st.update(status=status, reason=reason, final=final)
        write_json(p, st)


class Dispatcher:
    """1006 §3.6 / G10: (arm, block) units of `schedule.block` seeds (same town, contiguous) shuffled with
    `schedule.seed` into ONE shared queue that every worker pulls from; a (seed, arm) that failed for an infrastructure
    reason is re-queued as a single-seed unit that avoids every worker it already failed on, until
    `schedule.max_infra_attempts` attempts — then it stays failed (and drops out of every paired analysis).
    Every unit start / end is logged to <experiment>/schedule.jsonl (worker, CARLA port, GPU, DAM4SAM slot)."""

    def __init__(self, run, workers):
        import random
        sch = run.R["schedule"]
        self.run, self.W = run, workers
        self.max_att = int(sch.get("max_infra_attempts", 3))
        units = []
        for arm in run.R["arms"]:
            for town, block in seed_blocks(run.R["seeds"]):
                for i in range(0, len(block), int(sch["block"])):
                    units.append({"arm": arm, "seeds": block[i:i + int(sch["block"])], "avoid": []})
        random.Random(int(sch["seed"])).shuffle(units)
        self.q, self.lock, self.log_lock, self.active = units, threading.Lock(), threading.Lock(), 0
        self.retired = set()                   # workers whose unit raised (e.g. their CARLA restart failed); never reused
        self.abort = False                     # set when bookkeeping / logging failed: hand out no further unit
        self.log = os.path.join(run.root, "schedule.jsonl")
        self._log({"event": "plan", "seed": int(sch["seed"]), "block": int(sch["block"]), "n_units": len(units),
                   "order": [[u["arm"], [s["seed"] for s in u["seeds"]]] for u in units]})

    def _log(self, rec):
        with self.log_lock:
            with open(self.log, "a") as f:
                f.write(json.dumps(dict(rec, t=time.time()), default=str) + "\n")

    def _take(self, w):
        with self.lock:
            if self.abort:
                return None
            for i, u in enumerate(self.q):
                if w["id"] not in u["avoid"]:
                    self.active += 1
                    return self.q.pop(i)
            live = set(x["id"] for x in self.W) - self.retired
            if self.q and live and all(live <= set(u["avoid"]) for u in self.q):
                try:
                    for u in self.q:
                        for s in u["seeds"]:
                            self.run.finalize(u["arm"], s["seed"], "failed", "infra: failed on every worker", final=True)
                    self.q = []
                except Exception as e:                   # codex R3: a write failure here must stop the dispatch
                    self.abort = True                    # (the queue is kept: the seeds stay resumable)
                    say_quiet(f"dispatch ABORTED while finalizing: {type(e).__name__}: {e}")
                    return None
            return None if (not self.q and self.active == 0) else False

    def work(self, w):
        while True:
            u = self._take(w)
            if u is None:
                return
            if u is False:
                time.sleep(10)
                continue
            crashed, retire, retry = None, True, []      # retired unless the unit's lifecycle ends normally
            try:                                         # codex R1/R2 (10-19): `active` is released on EVERY path
                slot = self.run.acquire_service(u["arm"])
                w["svc_slot"] = slot
                try:
                    self._log({"event": "start", "arm": u["arm"], "seeds": [s["seed"] for s in u["seeds"]],
                               "worker": w["id"], "carla_port": w["carla_port"], "gpu": w["gpu"],
                               "tm_base": w["tm_base"], "svc_slot": slot})
                    try:
                        self.run.run_unit(u["arm"], w, u["seeds"])
                    except BookkeepingError:
                        raise
                    except Exception as e:       # 10-19: an exception (CARLA restart / preload timeout) used to kill
                        crashed = e              # the thread with `active` never decremented → every worker waited
                        say(f"{w['id']}: unit {u['arm']} {[s['seed'] for s in u['seeds']]} raised "
                            f"{type(e).__name__}: {e} — re-queueing its unfinished seeds and retiring this worker")
                finally:
                    self.run.release_service(slot)
                retry = self._requeue(u, w, crashed)
                self._log({"event": "end", "arm": u["arm"], "worker": w["id"],
                           "requeued": [s["seed"] for r in retry for s in r["seeds"]],
                           **({"crashed": f"{type(crashed).__name__}: {crashed}", "retired": True} if crashed else {})})
                retire = crashed is not None
            except BaseException as e:                   # bookkeeping / logging failed (or an interrupt): the run
                with self.lock:                          # state is no longer trusted → no further unit is handed out
                    self.abort = True
                say(f"{w['id']}: dispatch ABORTED — {type(e).__name__}: {e} (resume after fixing; statuses are "
                    f"reconciled at the next start)")
                raise
            finally:
                with self.lock:
                    self.q.extend(retry)
                    self.active -= 1
                    if retire:
                        self.retired.add(w["id"])
            if retire:
                return

    def _requeue(self, u, w, crashed):
        """retry units for the unit's seeds: an infra failure (booked by post_part; after a crash run_unit has already
        stopped its client and booked the interrupted part, or quarantined it) → single-seed retry avoiding this worker,
        up to max_infra_attempts; after a crash, seeds that did not run in this unit (no status, or a resumable
        failed status from an earlier run) → one unit, no attempt charged. A `partial` here means the booking did not
        happen → BookkeepingError (never re-run uncharged)."""
        retry, rest = [], []
        for s in u["seeds"]:
            st = self.run.status(u["arm"], s["seed"])
            if st.get("status") == "failed" and st.get("infra_retry"):
                n = len(st.get("attempts", []))
                if n < self.max_att:
                    st.update(final=False, infra_retry=False)
                    write_json(os.path.join(self.run.arm_dir(u["arm"]), "seeds", str(s["seed"]), "status.json"), st)
                    retry.append({"arm": u["arm"], "seeds": [s], "avoid": u["avoid"] + [w["id"]]})
                else:
                    st.update(reason=(st.get("reason") or "") + f" | infra: {n} attempts")
                    write_json(os.path.join(self.run.arm_dir(u["arm"]), "seeds", str(s["seed"]), "status.json"), st)
            elif st.get("status") == "partial":
                raise BookkeepingError(f"{u['arm']} seed {s['seed']}: still partial after its part was booked")
            elif crashed is not None and not (st.get("status") in DONE or
                                              (st.get("status") == "failed" and st.get("final"))):
                rest.append(s)
        if rest:
            retry.append({"arm": u["arm"], "seeds": rest, "avoid": u["avoid"] + [w["id"]]})
        return retry


def runtime_problems(R, runtime):
    """{arm: [reasons]} for arms the chosen policy runtime cannot run as resolved (acot: not implemented; legacy: a
    method field the legacy policy hard-codes)."""
    from acot.config.legacy import hardcoded_problems
    from acot.runtime.support import unsupported
    out = {}
    for name, arm in R["arms"].items():
        bad = unsupported(arm["protocol"], arm["method"], R["collect"]) if runtime == "acot" else hardcoded_problems(arm)
        if bad:
            out[name] = bad
    return out


def run_experiment(a):
    from acot.cli import _overrides
    R = resolve_experiment(a.experiment, _overrides(a.overrides), allow_test=a.allow_test,
                           arms=a.arms.split(",") if a.arms else None)
    if a.seeds:
        keep = parse_seed_filter(a.seeds)
        R["seeds"] = [s for s in R["seeds"] if s["seed"] in keep]
    run = Run(R, a)
    bad = runtime_problems(R, run.runtime)
    if bad:
        print(f"arms the {run.runtime} runtime cannot run as configured:\n  " +
              "\n  ".join(f"{k}: {'; '.join(v)}" for k, v in bad.items()))
        return 2
    errs = check_static(R["infra"])
    if errs:
        print("preflight (static) failed:\n  " + "\n  ".join(errs))
        return 2
    nw = a.workers or 1
    W = run.workers(nw)
    if not W:
        print("no worker available (CARLA server up + free GPU + lease)")
        return 2
    exp_lease = None
    try:
        errs = check_runtime(R, W)
        if errs:
            print("preflight (runtime) failed:\n  " + "\n  ".join(errs))
            return 2
        exp_lease = Lease([f"exp_{R['name']}"], owner=R["name"]).acquire()   # codex R4: one live invocation per
        os.makedirs(run.root, exist_ok=True)                                 # experiment
        files = {name: provenance.referenced_files(arm) for name, arm in R["arms"].items()}
        write_json(os.path.join(run.root, "experiment.resolved.json"), R)
        image = R["infra"]["docker"]["policy_image_acot" if run.runtime == "acot" else "policy_image"]
        write_json(os.path.join(run.root, "manifest.json"), dict(provenance.manifest(image), runtime=run.runtime,
                                                                 files=files, workers=[{k: v for k, v in w.items() if k != "lease"} for w in W],
                                                                 argv=vars(a)))
        run.write_specs()
        run.kill_orphans()                               # codex R4: no client of a crashed earlier run survives
        run.reconcile()                                  # codex R2: repair what a crashed earlier run left behind
        if R.get("schedule"):
            work = Dispatcher(run, W).work
        else:
            units = run.plan(R["seeds"], len(W))
            queues = {w["id"]: [] for w in W}
            for arm, i, chunk in units:
                queues[W[i % len(W)]["id"]].append((arm, chunk))

            def work(w):
                for arm, chunk in queues[w["id"]]:
                    slot = run.acquire_service(arm)
                    w["svc_slot"] = slot
                    try:
                        run.run_unit(arm, w, chunk)
                    finally:
                        run.release_service(slot)

        threads = [threading.Thread(target=work, args=(w,), daemon=True) for w in W]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        run.reconcile()
    finally:
        for w in W:
            w["lease"].release()
        if exp_lease is not None:
            exp_lease.release()
    summary = {}
    for arm in R["arms"]:
        for s in R["seeds"]:
            st = run.status(arm, s["seed"]).get("status", "missing")
            summary.setdefault(arm, {}).setdefault(st, []).append(s["seed"])
    write_json(os.path.join(run.root, "summary.json"), summary)
    for arm, d in summary.items():
        say(f"{arm}: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(d.items())))
    return 0 if all(set(d) <= set(DONE) for d in summary.values()) else 1


def run_audit_replay(a):
    """`acot audit-replay <exp> --arm A --seed S --gpu G` — the P1 audit validation (acot/eval/replay_neutral.py) on the
    act messages recorded for that seed (collect record_act), in a throw-away policy container."""
    R = resolve_experiment(a.experiment)
    run = Run(R, a)
    d = run.arm_dir(a.arm)
    spec = os.path.join(d, "resolved_spec.json")
    rec = os.path.join(d, "seeds", str(a.seed), "act_record.pkl")
    out = os.path.join(d, "seeds", str(a.seed), "audit_replay.json")
    if not os.path.exists(rec):
        print(f"no recording {rec} (run the experiment with collect: [record_act])")
        return 2
    I = R["infra"]
    env = dict(read_json(spec)["policy_env"], ASD_LOG="")
    name = f"{I['docker']['policy_container_prefix']}audit-replay-{R['name']}-{a.arm}"
    cmd = ["docker", "run", "--rm", "--gpus", f"device={a.gpu}", "--network", I["docker"]["network"], "--name", name,
           "--shm-size", I["docker"]["shm_size"], "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
           "-e", f"OMP_NUM_THREADS={I['threads']['omp']}", "-e", f"MKL_NUM_THREADS={I['threads']['omp']}"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    hf = os.path.expanduser(I["paths"]["hf_cache"])
    cmd += ["-v", f"{ROOT}:{ROOT}", "-v", f"{I['paths']['data']}:{I['paths']['data']}:ro",
            "-v", f"{hf}:/root/.cache/huggingface", "-w", ROOT, I["docker"]["policy_image"], "bash", "-c",
            f"pip install -q {I['pip_preinstall']} 2>/dev/null; python -m acot.eval.replay_neutral --spec {spec} "
            f"--record {rec} --out {out}"]
    p = docker.sh(cmd, timeout=3600)
    print((p.stdout or "")[-3000:])
    res = read_json(out)
    if res is None:
        return 1
    print(json.dumps(res["verdict"], indent=1))
    return 0 if all(res["verdict"].values()) else 1


def run_replay_parity(a):
    """`acot replay-parity <exp> --record R --gpu G` — P2 acceptance: every arm of the experiment replayed through the
    legacy and the acot runtime on the recorded act messages, in one throw-away container of the acot image
    (acot/eval/replay_parity.py)."""
    from acot.cli import _overrides
    R = resolve_experiment(a.experiment, _overrides(a.overrides), arms=a.arms.split(",") if a.arms else None)
    run = Run(R, a)
    run.write_specs()
    rec = os.path.abspath(a.record)
    if not os.path.exists(rec):
        print(f"no recording {rec}")
        return 2
    out = os.path.abspath(a.out or os.path.join(run.root, "replay_parity_" + os.path.basename(os.path.dirname(rec)) + ".json"))
    I = R["infra"]
    image = I["docker"]["policy_image_acot"]
    img = docker.inspect("image", image)
    if img is None:
        print(f"acot policy image {image} missing (bash acot/docker/build_acot_image.sh)")
        return 2
    name = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run.cprefix}replay-parity-{R['name']}-{os.getpid()}")
    specs = [os.path.join(run.arm_dir(arm), "resolved_spec.json") for arm in R["arms"]]
    cmd = run._docker_run_base(name, a.gpu, image)
    cmd = [x for x in cmd if x != "-d"]
    cmd[2:2] = ["--rm"]
    cmd += ["python", "-m", "acot.eval.replay_parity", "--specs", *specs, "--record", rec, "--out", out]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    say(f"replay parity: {len(specs)} arm(s) on {rec} (GPU {a.gpu}, image {img['Id'][:19]})")
    p = docker.sh(cmd, timeout=6 * 3600)
    print((p.stdout or "")[-4000:])
    res = read_json(out)
    if res is None:
        return 1
    res["image"] = {"name": image, "id": img["Id"]}
    write_json(out, res)
    print(f"→ {out}")
    return 0 if res.get("ok") else 1
