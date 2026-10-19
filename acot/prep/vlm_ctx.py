"""`acot prep` kind vlm_ctx (framework §10, P3b): the Stage-1 / Stage-2 VLM context caches of train/backbone_kv.py for a
dataset artefact under an explicit split, written INSIDE the artefact directory (never a shared runs/ctx_cache_* dir,
never 'skip if the file exists' across configurations: the directory is content-addressed by its inputs)."""
import os

import yaml

from acot import ROOT
from acot import artefacts as A
from acot import jobs
from acot.config.load import read_yaml

CODE = ("train/backbone_kv.py", "acot_probe/backbones.py")


def plan(spec):
    ds, sp = A.lookup(spec["dataset"]), A.lookup(spec["split"])
    if ds is None or ds["kind"] != "dataset" or sp is None or sp["kind"] != "split":
        raise A.ArtefactError("vlm_ctx needs a dataset and a split artefact")
    cfg = read_yaml(os.path.join(ROOT, spec["model_config"]))
    b, data = cfg["backbone"], cfg["data"]
    layers = spec.get("layers")
    inputs = {"dataset": f"dataset/{ds['name']}@{ds['id']}", "split": f"split/{sp['name']}@{sp['id']}",
              "backbone": {k: b.get(k) for k in ("kind", "model_id", "layer", "ctx_grid", "min_pixels", "max_pixels")},
              "layers": layers, "frame_stride": int(data.get("frame_stride", 3)),
              "max_frames": int(data.get("max_frames_per_episode", 200)),
              "language_mode": spec.get("language_mode", "neutral"), "code": jobs.code_hash(*CODE)}
    compat = {"fov_deg": ds["compat"]["fov_deg"], "tick_s": round(inputs["frame_stride"] / ds["compat"]["fps"], 6),
              "language_mode": inputs["language_mode"]}
    aid = A.input_id("vlm_ctx", spec["name"], inputs)
    d = A.artefact_dir("vlm_ctx", spec["name"], aid)
    derived = dict(cfg)
    derived["data"] = dict(data, episodes_glob=ds["inputs"]["glob"], split_manifest=os.path.join(ROOT, sp["files"][0]["path"]),
                           context_cache=os.path.join(d, "ctx"), stage2_context_cache=os.path.join(d, "ctx"))
    return inputs, compat, aid, d, derived


def build(spec, gpus, dry_run=False, log=print):
    inputs, compat, aid, d, derived = plan(spec)
    lay = f" --layers {' '.join(map(str, spec['layers']))}" if spec.get("layers") else ""
    cfgp = os.path.join(d, "config.yaml")
    cmds = [(f"python -m train.backbone_kv --config {cfgp}{lay} --shard {k} --num-shards {len(gpus)} --device cuda:0",
             g, f"vlmctx-{aid}-{k}", os.path.join(d, "logs", f"shard{k}.log")) for k, g in enumerate(gpus)]
    if dry_run:
        return {"id": aid, "dir": d, "inputs": inputs, "compat": compat, "commands": [c[0] for c in cmds]}
    if os.path.exists(os.path.join(d, "manifest.json")):
        return A.manifest("vlm_ctx", spec["name"], aid)
    os.makedirs(d, exist_ok=True)
    yaml.safe_dump(derived, open(cfgp, "w"), sort_keys=False)
    rcs = jobs.run_parallel(cmds)
    if any(rcs):
        raise RuntimeError(f"backbone_kv shard(s) failed: {rcs}")
    files = sorted(os.path.relpath(os.path.join(dp, f), ROOT) for dp, _, fs in os.walk(os.path.join(d, "ctx")) for f in fs)
    return A.register("vlm_ctx", spec["name"], files=files, inputs=inputs, compat=compat, full_hash=False,
                      producer={"tool": "acot prep vlm_ctx", "git": jobs.git_head(), "spec": spec})
