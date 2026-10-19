"""`acot prep` kind dagger (framework §10, P3b): merge the per-episode TAH-DAgger files collected by an acot experiment
(collect: [tah_dagger] → <results>/<exp>/<arm>/collect/tah_dagger/*.pt) into one dagger artefact with train/tah_merge.py.
compat is DERIVED from the experiment's resolved protocol (tick_s = s2_period / fps, fov, features, identity keys).
2026-10-14 (P0-2 stage B, acot_note/1012 §6.5): experiments on the acot runtime write one file per seed instead
(<arm>/seeds/<seed>/tah_dagger.pt, acot/runtime/server.py) → merged from there, complete seeds only."""
import glob
import json
import os

from acot import ROOT
from acot import artefacts as A
from acot import jobs
from acot.config.resolve import resolve_experiment


def _acot_parts(R, arm_dir):
    """acot runtime layout → (files, glob for train/tah_merge.py, part name). Only seeds whose final status is complete
    are merged; a seed with any other status (skipped / failed / partial) must not have left a file behind, and the glob
    must match exactly the complete seeds' files — otherwise refuse (tah_merge.py merges whatever the glob matches)."""
    base = os.path.join(arm_dir, "seeds")
    files, bad = [], []
    for s in sorted(int(x["seed"]) for x in R["seeds"]):
        d = os.path.join(base, str(s))
        sp, f = os.path.join(d, "status.json"), os.path.join(d, "tah_dagger.pt")
        st = json.load(open(sp)).get("status") if os.path.exists(sp) else None
        if st == "complete":
            # codex P0-2b R1 MAJOR: a retry keeps the seed directory, and the server writes the file only at flush (next
            # reset / close), AFTER the client marked the seed complete → a partial file of an earlier attempt survives
            # if the completing attempt's flush never ran. The file must be written after the completing attempt began
            # (t_start: set by the client per attempt, kept by the scheduler's post_part).
            t0 = json.load(open(sp)).get("t_start")
            if not os.path.exists(f):
                bad.append(f"seed {s}: complete but no tah_dagger.pt")
            elif t0 is None or os.path.getmtime(f) < float(t0):
                bad.append(f"seed {s}: tah_dagger.pt predates the completing attempt (t_start {t0}) — stale")
            else:
                files.append(f)
        elif os.path.exists(f):
            bad.append(f"seed {s}: status {st} but tah_dagger.pt exists")
    pat = os.path.join(base, "*", "tah_dagger.pt")
    stray = sorted(set(glob.glob(pat)) - set(files))
    if stray:
        bad.append(f"files outside the complete seeds: {stray[:5]}")
    if bad:
        raise A.ArtefactError("TAH-DAgger collection is not mergeable: " + "; ".join(bad))
    return files, pat, lambda f: os.path.relpath(f, base)


def build(spec, gpu, log=print):
    src = spec["source"]
    R = resolve_experiment(os.path.join(ROOT, src["experiment"]))
    arm = R["arms"][src["arm"]]
    p = arm["protocol"]
    arm_dir = os.path.join(ROOT, R["infra"]["results_root"], R["name"], src["arm"])
    if R["infra"].get("runtime") == "acot":
        files, pat, part_name = _acot_parts(R, arm_dir)
        d_in = os.path.join(arm_dir, "seeds")
    else:                                                       # legacy runtime: unchanged (same artefact ids)
        d_in = os.path.join(arm_dir, "collect", "tah_dagger")
        files = sorted(glob.glob(os.path.join(d_in, "*.pt")))
        pat, part_name = f"{d_in}/*.pt", os.path.basename
    if not files:
        raise A.ArtefactError(f"no collected TAH-DAgger files in {d_in}")
    compat = {"fov_deg": p["camera"]["fov_deg"], "tick_s": round(p["sim"]["s2_period"] / p["camera"]["fps"], 6),
              "feature_id": p["frontend"]["crop_features"]["id"],
              "depth_scale": p["frontend"]["crop_features"]["depth_scale"], "d_feat": 384,
              "identity_keys": p["perception"]["identity_keys"]}
    fps = [A.fingerprint(os.path.relpath(f, ROOT)) for f in files]
    inputs = {"experiment": R["name"], "arm": src["arm"], "config_hash": arm["config_hash"],
              "seeds_hash": R["seeds_hash"], "parts": [dict(fp, file=part_name(f)) for f, fp in zip(files, fps)],
              "code": jobs.code_hash("train/tah_merge.py")}
    aid = A.input_id("dagger", spec["name"], inputs)
    d = A.artefact_dir("dagger", spec["name"], aid)
    if os.path.exists(os.path.join(d, "manifest.json")):
        log(f"dagger/{spec['name']}@{aid} already merged")
        return A.manifest("dagger", spec["name"], aid)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "dagger.pt")
    rc = jobs.run_in_image(f"python train/tah_merge.py --glob '{pat}' --out {out}", gpu, f"merge-{aid}",
                           os.path.join(d, "logs", "merge.log"))
    if rc:
        raise RuntimeError(f"tah_merge failed (log {d}/logs/merge.log)")
    return A.register("dagger", spec["name"], files=[os.path.relpath(out, ROOT)], inputs=inputs, compat=compat,
                      producer={"tool": "acot prep dagger", "git": jobs.git_head(), "spec": spec})
