"""Offline jobs in the training image (framework §10, P3b): `docker run --rm` of acot-uav-train on the EXISTING network
(never a new one), GPU pinned, repository mounted at the same path, data read-only — the same container recipe as the
legacy training / caching launchers, so the legacy scripts run unchanged inside. The image is the acot policy image
(timm pinned in it; P2) when present, else the legacy image + the start-up pip install. Jobs run DETACHED and the
container itself writes the log and its exit code (<log>.rc): a job outlives this host process, whose death would
otherwise cut the log (and `--rm` drops the container's own logs) — `acot train --register-only` registers it after."""
import os
import shlex
import subprocess
import threading
import time

from acot import ROOT
from acot.config.load import read_yaml
from acot.infra import docker

from acot import CONFIGS


def infra():
    return read_yaml(os.path.join(CONFIGS, "infra", "default.yaml"))


def image_and_prefix(I):
    """(image, shell prefix): the acot image (timm pinned) if built, else the legacy image + pip install."""
    image = I["docker"].get("policy_image_acot")
    if image and docker.inspect("image", image) is not None:
        return image, ""
    return I["docker"]["policy_image"], f"pip install -q {I['pip_preinstall']} 2>/dev/null; "


def run_in_image(cmd, gpu, name, log, env=None, timeout=None, data_rw=False, poll_s=10):
    """run `cmd` (shell string, cwd = repo root) in the training image on `gpu`; stdout+stderr → log (written by the
    container), exit code → <log>.rc. → return code (125 if the container vanished without writing one)."""
    I = infra()
    hf = os.path.expanduser(I["paths"]["hf_cache"])
    cname = f"{I['docker']['policy_container_prefix']}job-{name}"
    if docker.inspect("container", cname):
        docker.rm(cname, I["docker"]["policy_container_prefix"])
    image, pre = image_and_prefix(I)
    rc_path = log + ".rc"
    if os.path.exists(rc_path):
        os.remove(rc_path)
    args = ["docker", "run", "-d", "--rm", "--gpus", f"device={gpu}", "--network", I["docker"]["network"],
            "--name", cname, "--shm-size", I["docker"]["shm_size"], "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "-e", f"OMP_NUM_THREADS={I['threads']['omp']}", "-e", f"MKL_NUM_THREADS={I['threads']['omp']}"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    args += ["-v", f"{ROOT}:{ROOT}", "-v", f"{I['paths']['data']}:{I['paths']['data']}{'' if data_rw else ':ro'}",
             "-v", f"{hf}:/root/.cache/huggingface", "-w", ROOT, image, "bash", "-c",
             f"( {pre}{cmd} ) >> {shlex.quote(log)} 2>&1; echo $? > {shlex.quote(rc_path)}"]
    os.makedirs(os.path.dirname(log), exist_ok=True)
    img = docker.inspect("image", image) or {}
    with open(log, "w") as fh:
        fh.write(f"# image {image} {img.get('Id', '?')}\n$ " + " ".join(shlex.quote(a) for a in args) + "\n")
    p = docker.sh(args)
    if p.returncode:
        with open(log, "a") as fh:
            fh.write(p.stdout or "")
        return p.returncode
    t0 = time.time()
    while not os.path.exists(rc_path):
        if not docker.running(cname):
            time.sleep(2)                                   # the rc file is written just before the container exits
            if not os.path.exists(rc_path):
                return 125
            break
        if timeout and time.time() - t0 > timeout:
            docker.rm(cname, I["docker"]["policy_container_prefix"])
            raise TimeoutError(f"job {cname} exceeded {timeout} s")
        time.sleep(poll_s)
    return int((open(rc_path).read().strip() or "125"))


def run_parallel(jobs):
    """jobs: [(cmd, gpu, name, log)] → list of return codes (one thread per job)."""
    rcs = [None] * len(jobs)

    def go(i, j):
        rcs[i] = run_in_image(*j)
    ts = [threading.Thread(target=go, args=(i, j)) for i, j in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return rcs


def code_hash(*paths):
    """sha256 over the source files a product depends on (part of an artefact's inputs)."""
    import hashlib
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode())
        h.update(open(os.path.join(ROOT, p), "rb").read())
    return h.hexdigest()[:16]


def git_head():
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True).stdout.strip()
