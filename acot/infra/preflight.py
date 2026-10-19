"""Infrastructure preflight (framework §13.2): static checks (network, cyh-carla running with the expected mounts,
CARLA-client python 3.8 + the acot client importable, policy image present) and run-time checks for a planned run
(CARLA servers answering on the leased ports, policy GPUs free, DAM4SAM instance healthy when needed)."""
import os
import subprocess

from acot import CONFIGS, ROOT
from acot.config.load import read_yaml
from acot.infra import docker


def infra_default():
    return read_yaml(os.path.join(CONFIGS, "infra", "default.yaml"))


def check_static(infra=None):
    I = infra or infra_default()
    d, errs = I["docker"], []
    if docker.inspect("network", d["network"]) is None:
        errs.append(f"docker network {d['network']} does not exist (acot never creates networks)")
    c = docker.inspect("container", d["carla_container"])
    if c is None or not c["State"]["Running"]:
        errs.append(f"container {d['carla_container']} is not running")
    else:
        mounts = {m["Destination"]: m["Source"] for m in c["Mounts"]}
        for dst, src in d["carla_mounts"].items():
            want = src if src.startswith("/") else os.path.join(ROOT, src)
            if os.path.normpath(mounts.get(dst, "")) != os.path.normpath(want):
                errs.append(f"{d['carla_container']} mount {dst} = {mounts.get(dst)} (expected {want})")
        p = docker.exec_py(d["carla_container"], "import sys; print('%d.%d' % sys.version_info[:2])",
                           python=I["python"]["carla_client"])
        if p.stdout.strip() != I["python"]["carla_client_version"]:
            errs.append(f"{d['carla_container']} `{I['python']['carla_client']}` is {p.stdout.strip()!r}, "
                        f"expected {I['python']['carla_client_version']}")
        p = docker.exec_py(d["carla_container"], "import sys; sys.path.insert(0, '/workspace'); import acot_client.episode",
                           python=I["python"]["carla_client"], workdir="/workspace")
        if p.returncode != 0:
            errs.append(f"acot_client does not import in {d['carla_container']}: {p.stdout[-400:]}")
    if docker.inspect("image", d["policy_image"]) is None:
        errs.append(f"policy image {d['policy_image']} is missing (rebuild: bash train/docker/build.sh)")
    return errs


def carla_up(infra, port):
    code = (f"import socket; s = socket.socket(); s.settimeout(3); s.connect(('127.0.0.1', {port})); print('up')")
    p = docker.exec_py(infra["docker"]["carla_container"], code, python=infra["python"]["carla_client"], timeout=30)
    return p.returncode == 0 and "up" in p.stdout


def gpu_free_mib(gpu):
    p = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True)
    used, total = (int(x) for x in p.stdout.strip().split(","))
    return total - used


def dam4sam_healthy(infra, host, port, timeout=10):
    """`close` round trip with the bridge protocol, from inside cyh-carla (on the same docker network; the service
    accepts one client at a time, so only call this while the instance is not leased to a policy server)."""
    code = ("import sys, socket; sys.path.insert(0, '/workspace'); from rollout.bridge import send_msg, recv_msg; "
            f"s = socket.create_connection(({host!r}, {port}), timeout={timeout}); send_msg(s, {{'cmd': 'close'}}); "
            "print('ok' if recv_msg(s).get('ok') else 'bad')")
    p = docker.exec_py(infra["docker"]["carla_container"], code, python=infra["python"]["carla_client"],
                       timeout=timeout + 20, workdir="/workspace")
    return p.returncode == 0 and "ok" in p.stdout


def check_runtime(R, workers, min_gpu_mib=20000):
    I, errs = R["infra"], []
    for w in workers:
        lo_hi = I["reserved"]["carla_ports"]
        if any(lo <= w["carla_port"] <= hi for lo, hi in lo_hi):
            errs.append(f"CARLA port {w['carla_port']} is reserved for another session")
        if not carla_up(I, w["carla_port"]):
            errs.append(f"no CARLA server answers on port {w['carla_port']} (start it, or let the scheduler restart it)")
        if gpu_free_mib(w["gpu"]) < min_gpu_mib:
            errs.append(f"GPU {w['gpu']} has < {min_gpu_mib} MiB free")
        if w["gpu"] in I["reserved"]["gpus"]:
            errs.append(f"GPU {w['gpu']} is reserved")
    heads = {a["method"]["identity"]["head"] for a in R["arms"].values()}
    if heads & {"dam4sam", "fusion"}:
        svc = I["services"]["dam4sam"]
        for inst in svc["instances"]:
            if not docker.running(inst["container"]):
                errs.append(f"DAM4SAM container {inst['container']} is not running "
                            f"(lifecycle {svc['lifecycle']}: docker start {inst['container']} + start the service)")
            elif not dam4sam_healthy(I, inst["host"], inst["port"]):
                errs.append(f"DAM4SAM service {inst['host']}:{inst['port']} does not answer")
    return errs
