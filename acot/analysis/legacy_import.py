"""Import of legacy closed-loop results (framework §12.2) with DETERMINISTIC precedence: the merged
paper_it_<stem>.json if present, else its parts in NUMERIC order (any number of digits) with the final .json over the
.json.partial; per seed the first scored record wins (conflicting duplicates → warning). Legacy ASD tick logs carry no
seed: they are aligned to the non-skipped episodes of the matching part BY ORDER and verified by step counts."""
import collections
import glob
import json
import os
import re

import yaml

from acot import ROOT

EO = os.path.join(ROOT, "carla_uav_tracking", "rollout", "eval_out")
LG = os.path.join(ROOT, "runs", "asd")
ARM_MAP = os.path.join(ROOT, "acot", "legacy", "arm_map.yaml")


def arm_map():
    return (yaml.safe_load(open(ARM_MAP)) or {}).get("arms", {})


def stem_parts(stem):
    """→ ([files], [warnings])."""
    full = os.path.join(EO, f"paper_it_{stem}.json")
    if os.path.exists(full):
        return [full], []
    parts, warn = {}, []
    for p in glob.glob(os.path.join(EO, f"paper_it_{stem}_p*.json*")):
        m = re.search(r"_p(\d+)\.json(\.partial)?$", p)
        if m is None:
            continue
        k, final = int(m.group(1)), m.group(2) is None
        if k not in parts or final:
            parts[k] = p
    if parts:
        warn += [f"{os.path.basename(parts[k])}: only the .partial exists" for k in sorted(parts)
                 if parts[k].endswith(".partial")]
        return [parts[k] for k in sorted(parts)], warn
    if os.path.exists(full + ".partial"):
        return [full + ".partial"], [f"paper_it_{stem}.json.partial: only the .partial exists"]
    return [], [f"no files for stem {stem}"]


def files_for(name):
    M = arm_map().get(name)
    if M is None:
        return stem_parts(name)
    files, warn = [], []
    for f in M.get("files", []):
        if isinstance(f, dict):
            fs, w = stem_parts(f["stem"])
            files += fs
            warn += w
        else:
            files.append(os.path.join(EO, f))
    for s in M.get("stems", [M["stem"]] if "stem" in M else []):
        fs, w = stem_parts(s)
        files += fs
        warn += w
    return files, warn


def episodes(name):
    """→ ({seed: episode} of SCORED episodes (iter_gate semantics), {seed: status record}, [warnings])."""
    files, warn = files_for(name)
    scored, status = {}, {}
    for f in files:
        if not os.path.exists(f):
            warn.append(f"missing file {os.path.basename(f)}")
            continue
        for e in json.load(open(f))["episodes"]:
            s = int(e["episode_seed"]) if e.get("episode_seed") is not None else None
            if s is None:
                continue
            if e.get("skipped"):
                status.setdefault(s, {"status": "skipped", "reason": e.get("reason"), "file": os.path.basename(f)})
                continue
            if e.get("mis_follow_inst") is None:
                status.setdefault(s, {"status": "failed", "reason": "unscored", "file": os.path.basename(f)})
                continue
            if s in scored:
                if json.dumps(scored[s], sort_keys=True, default=str) != json.dumps(e, sort_keys=True, default=str):
                    warn.append(f"seed {s}: duplicate scored record with different content in {os.path.basename(f)} "
                                f"(first kept)")
                continue
            scored[s] = e
            status[s] = {"status": "complete", "file": os.path.basename(f),
                         **({"legacy_error": str(e["error"])[:200]} if e.get("error") else {})}
    return scored, status, warn


def tick_pairs(stem):
    """(ASD log, scorer json) per part, numeric part order, final json over partial."""
    out = []
    for jl in glob.glob(os.path.join(LG, "iter", f"{stem}_p*.jsonl")):
        m = re.search(r"_p(\d+)\.jsonl$", jl)
        if m is None:
            continue
        k = int(m.group(1))
        js = os.path.join(EO, f"paper_it_{stem}_p{k}.json")
        js = js if os.path.exists(js) else js + ".partial"
        if os.path.exists(js):
            out.append((k, jl, js))
    return [(jl, js) for _, jl, js in sorted(out)]


def ticks_for(name):
    M = arm_map().get(name, {})
    if "ticks" in M:
        return [(os.path.join(ROOT, a), os.path.join(EO, b)) for a, b in M["ticks"]]
    out = []
    for s in M.get("stems", [M.get("stem", name)]):
        out += tick_pairs(s)
    return out


def ticks(pairs, warnings=None):
    """→ ({seed: tick records sorted by t}, {seed: 'by_order_verified' | 'by_order_unverified'}). An episode is verified
    when its log group starts at t = 0 and ends at or before the scorer's step count; surplus TRAILING log groups (an
    episode killed by the watchdog, re-run in the next part) are reported in `warnings`."""
    eps, how = {}, {}
    for jl, js in pairs:
        try:
            S = [e for e in json.load(open(js))["episodes"] if not e.get("skipped")]
        except (OSError, ValueError):
            continue
        E = collections.defaultdict(list)
        for line in open(jl):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            E[r["ep"]].append(r)
        if len(E) > len(S) and warnings is not None:
            warnings.append(f"{os.path.basename(jl)}: {len(E) - len(S)} trailing log group(s) without a scored episode "
                            f"(killed episode)")
        for ep, e in zip(sorted(E), S):
            s = e["episode_seed"]
            if s in eps:
                continue
            R = sorted(E[ep], key=lambda r: r["t"])
            eps[s] = R
            ok = R[0]["t"] == 0 and R[-1]["t"] <= int(e.get("steps", 10 ** 9))
            how[s] = "by_order_verified" if ok else "by_order_unverified"
    return eps, how
