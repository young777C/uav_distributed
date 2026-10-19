"""L0 golden harness (framework §9): run a LEGACY launcher in a scratch copy of its scripts with the fake `docker` /
`sleep` first on PATH, and extract every (policy container, CARLA client) invocation it would have made."""
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile

from acot import ROOT
from acot.config.legacy import POLICY_ENV

SHIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shim")
SUPPORT = ("carla_uav_tracking/rollout/eval_asd.sh", "runs/asd/iter/launch_wd.sh")
DIRS = ("carla_uav_tracking/rollout/eval_out", "runs/asd/iter", "runs/opening_dump/logs", "runs/opening_dump/val",
        "runs/tah_v90/dag_v2")
SCRUB = set(POLICY_ENV) | {"SEED_BASE", "EPISODES", "STEPS", "TOWN", "S2_PERIOD", "GPU", "SRV", "CARLA_PORT", "TM_PORT",
                           "OUT_NAME", "HEAD_ARGS", "VIS_RAYCAST", "HIRES", "DUMP_LABELS", "NTHR", "CONFIG",
                           "START_PART", "STALL_MIN", "FIRST_MIN", "HF_CACHE"}


def run_launcher(launcher, args=(), env=None, timeout=300):
    """→ (records, scratch repo path). The scratch copy rewrites the hard-coded repository path."""
    tmp = tempfile.mkdtemp(prefix="acot_golden_")
    repo = os.path.join(tmp, "repo")
    for rel in (launcher,) + SUPPORT:
        dst = os.path.join(repo, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(os.path.join(ROOT, rel)) as f:
            txt = f.read().replace(ROOT, repo)
        with open(dst, "w") as f:
            f.write(txt)
    for d in DIRS:
        os.makedirs(os.path.join(repo, d), exist_ok=True)
    state = os.path.join(tmp, "state")
    os.makedirs(state)
    log = os.path.join(tmp, "docker.jsonl")
    e = {k: v for k, v in os.environ.items() if k not in SCRUB}
    e.update(PATH=SHIM + os.pathsep + e.get("PATH", ""), SHIM_LOG=log, SHIM_STATE=state, SHIM_REPO=repo, HOME=tmp)
    e.update(env or {})
    p = subprocess.run(["bash", os.path.join(repo, launcher), *args], cwd=repo, env=e, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    recs = [json.loads(x) for x in open(log)] if os.path.exists(log) else []
    return recs, repo, p.stdout


def invocations(records, repo):
    """pair every `docker exec … run_rollout.py` with the policy container (`docker run`) it talks to."""
    current, out = {}, []
    for r in records:
        if r["op"] == "run":
            current[r["name"]] = r
        elif r["op"] == "exec" and any(c.endswith("run_rollout.py") for c in r["command"]):
            cmd = r["command"]
            i = next(k for k, c in enumerate(cmd) if c.endswith("run_rollout.py"))
            client_argv = cmd[i + 1:]
            run = current[client_argv[client_argv.index("--policy-host") + 1]]
            script = run["command"][-1]
            policy_argv = shlex.split(script.split("policy_server.py", 1)[1])
            pip = re.search(r"pip install -q (\S+)", script)
            out.append({"policy_argv": policy_argv, "policy_env": run["env"], "client_argv": client_argv,
                        "client_env": r["env"],
                        "docker": {"image": run["image"], "network": run["network"], "shm": run["shm"],
                                   "workdir": (run["workdir"] or "").replace(repo, ROOT),
                                   "mounts": sorted(m.replace(repo, ROOT).replace(os.path.dirname(repo), "$TMP")
                                                    for m in run["mounts"]),
                                   "pip": pip.group(1) if pip else None, "gpus": run["gpus"]}})
    return out


def cleanup(repo):
    shutil.rmtree(os.path.dirname(repo), ignore_errors=True)
