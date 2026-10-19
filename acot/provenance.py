"""Provenance for every acot product (framework §1.3): git commit + hash of the uncommitted diff, image id, host, time,
and sha256 of the referenced files (checkpoints / configs; cached by path + size + mtime, except for files modified
within the last RACY_S seconds, whose timestamp cannot distinguish a same-size rewrite)."""
import hashlib
import json
import os
import platform
import socket
import subprocess
import time

from acot import ROOT

CACHE = os.path.join(ROOT, "runs", "infra", "filehash_cache.json")


def _git(*args):
    p = subprocess.run(["git", "-C", ROOT, *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return p.stdout


RACY_S = 2.0          # a file modified this recently is hashed afresh and not cached (timestamp granularity, "racy git")


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    path = os.path.join(ROOT, path)
    st = os.stat(path)
    if time.time() - st.st_mtime < RACY_S:
        return _hash(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    try:
        cache = json.load(open(CACHE))
    except (OSError, ValueError):
        cache = {}
    if key not in cache:
        cache[key] = _hash(path)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        tmp = CACHE + f".{os.getpid()}.tmp"
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, CACHE)
    return cache[key]


def referenced_files(arm):
    p, m = arm["protocol"], arm["method"]
    files = [p["frontend"]["vlm"]["model_config"], p["frontend"]["vlm"]["stage2_ckpt"],
             "carla_uav_tracking/config/default.yaml", "carla_uav_tracking/config/scenarios/mvp.yaml"]
    o = p["frontend"]["opening"]
    if o["module"] == "attr":
        files += [o["attr"]["ckpt_lin"], o["attr"]["ckpt_ft"]]
    for k in ("tah", "attrbind"):
        if m["identity"].get(k, {}).get("ckpt"):
            files.append(m["identity"][k]["ckpt"])
    return {f: file_sha256(f) for f in files if os.path.exists(os.path.join(ROOT, f))}


def manifest(image=None):
    diff = _git("diff", "HEAD")
    out = {"git_commit": _git("rev-parse", "HEAD").strip(), "git_dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
           "git_dirty_files": len([x for x in _git("status", "--porcelain").splitlines() if x.strip()]),
           "host": socket.gethostname(), "python": platform.python_version(), "time": time.time(),
           "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if image:
        p = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", image], stdout=subprocess.PIPE, text=True)
        out["image"] = {"name": image, "id": p.stdout.strip()}
    return out
