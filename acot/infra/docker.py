"""Thin, logged wrappers around the only docker operations acot performs (framework §13.1): inspect, `docker run` of
policy containers on the EXISTING network, `docker exec` into cyh-carla, `docker start/exec` of DAM4SAM, `docker rm` of
acot's own containers. No network is ever created."""
import json
import subprocess

FORBIDDEN = ("network create", "compose")


def sh(args, check=False, capture=True, timeout=None):
    cmd = " ".join(args)
    if any(f in cmd for f in FORBIDDEN):
        raise PermissionError(f"forbidden docker operation: {cmd}")
    p = subprocess.run(args, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None,
                       text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd} → rc {p.returncode}: {(p.stdout or '')[-800:]}")
    return p


def inspect(kind, name):
    p = sh(["docker", kind, "inspect", name])
    return json.loads(p.stdout)[0] if p.returncode == 0 else None


def running(name):
    p = sh(["docker", "ps", "--format", "{{.Names}}"])
    return name in p.stdout.split()


def exec_py(container, code, python="python", timeout=120, workdir=None):
    args = ["docker", "exec"] + (["-w", workdir] if workdir else []) + [container, python, "-c", code]
    return sh(args, timeout=timeout)


def rm(name, prefix):
    if not name.startswith(prefix):
        raise PermissionError(f"acot only removes its own containers ({prefix}*), not {name}")
    return sh(["docker", "rm", "-f", name])
