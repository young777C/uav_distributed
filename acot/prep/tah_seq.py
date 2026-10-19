"""`acot prep acot/configs/prep/<x>.yaml` for kind tah_seq (framework §10, P3b): TAH candidate sequences of a dataset
artefact under an EXPLICIT split, built by the legacy train/tah_cache.py (sharded over GPUs in the training image) and
merged by train/tah_merge.py. Content-addressed by (dataset, split, geometry, stride, feature id, code); the derived
config writes nowhere outside the artefact directory; compat (fov, tick_s, feature id, depth scale) is recorded."""
import os

import yaml

from acot import ROOT
from acot import artefacts as A
from acot import jobs
from acot.config.load import read_yaml

CODE = ("train/tah_cache.py", "train/tah_merge.py", "train/reid_resolution_probe.py", "acot_probe/projection.py",
        "train/dataset.py")
FEATURES = {"dino_s14_lvd142m_cls384_c224_p1.3": {"model": "vit_small_patch14_dinov2.lvd142m", "crop": 224,
                                                   "pad": 1.3, "d_feat": 384}}


def plan(spec):
    ds, sp = A.lookup(spec["dataset"]), A.lookup(spec["split"])
    if ds is None or ds["kind"] != "dataset":
        raise A.ArtefactError(f"{spec['dataset']}: not a dataset artefact")
    if sp is None or sp["kind"] != "split" or sp["inputs"]["dataset"] != f"dataset/{ds['name']}@{ds['id']}":
        raise A.ArtefactError(f"{spec['split']}: not a split of {spec['dataset']}")
    if spec["feature_id"] not in FEATURES:
        raise A.ArtefactError(f"feature_id {spec['feature_id']} is not what tah_cache.py computes ({list(FEATURES)})")
    if float(spec["depth_scale"]) != 100.0:
        raise A.ArtefactError("tah_cache.py hard-codes depth / 100")
    cfg = read_yaml(os.path.join(ROOT, spec["model_config"]))
    img = cfg["image"]
    cam = ds["compat"]
    bad = {k: (img.get(k), cam.get(k)) for k in ("fov_deg", "width", "height")
           if img.get(k) is None or float(img[k]) != float(cam[k])}
    if bad:
        raise A.ArtefactError(f"{spec['model_config']} projects with {bad} but the dataset camera is {cam} "
                              f"(the 70°-images-with-90°-boxes failure) — use a model config matching the data")
    # 2026-10-12 P0-2 (optional spec fields; absent = legacy): tick_stride K → one sequence per phase at the deployed
    # S2 tick K / fps; frame_mask "train" → recording/building_mask.py:train_keep_mask frames dropped
    tick = spec.get("tick_stride")
    fmask = spec.get("frame_mask")
    if tick is not None and (int(tick) <= 0 or int(tick) % int(spec["stride"])):
        raise A.ArtefactError(f"tick_stride {tick} must be a positive multiple of stride {spec['stride']}")
    if fmask not in (None, "train"):
        raise A.ArtefactError(f"frame_mask {fmask!r}: only 'train' (or absent) is implemented")
    inputs = {"dataset": f"dataset/{ds['name']}@{ds['id']}", "split": f"split/{sp['name']}@{sp['id']}",
              "geometry": {k: img[k] for k in ("width", "height", "fov_deg", "vehicle_size_m", "box_min_px", "box_max_px")},
              "stride": int(spec["stride"]), "feature_id": spec["feature_id"], "depth_scale": 100.0,
              "code": jobs.code_hash(*CODE)}
    if tick is not None:
        inputs["tick_stride"] = int(tick)
    if fmask is not None:
        inputs["frame_mask"] = fmask
    compat = {"fov_deg": float(img["fov_deg"]),
              "tick_s": round(int(tick if tick is not None else spec["stride"]) / int(cam.get("fps", 10)), 6),
              "feature_id": spec["feature_id"], "depth_scale": 100.0, "d_feat": FEATURES[spec["feature_id"]]["d_feat"]}
    aid = A.input_id("tah_seq", spec["name"], inputs)
    d = A.artefact_dir("tah_seq", spec["name"], aid)
    derived = dict(cfg)
    derived["data"] = dict(cfg.get("data", {}), episodes_glob=ds["inputs"]["glob"],
                           split_manifest=os.path.join(ROOT, sp["files"][0]["path"]),
                           context_cache=os.path.join(d, "unused_ctx"), stage2_context_cache=os.path.join(d, "unused_ctx2"))
    derived["data"].pop("frame_mask", None)
    if fmask is not None:
        derived["data"]["frame_mask"] = fmask
    return ds, sp, inputs, compat, aid, d, derived


def build(spec, gpus, only_shard=None, log=print):
    ds, sp, inputs, compat, aid, d, derived = plan(spec)
    out = os.path.join(d, "tah_seq.pt")
    if os.path.exists(os.path.join(d, "manifest.json")) and only_shard is None:
        log(f"tah_seq/{spec['name']}@{aid} already built (content-addressed) — nothing to do")
        return A.manifest("tah_seq", spec["name"], aid)
    n = int(spec.get("shards", len(gpus)))
    sd = os.path.join(d, ".shards" if only_shard is None else f".probe_{n}_{only_shard}")
    os.makedirs(sd, exist_ok=True)
    cfgp = os.path.join(d, "config.yaml")
    yaml.safe_dump(derived, open(cfgp, "w"), sort_keys=False)
    shards = range(n) if only_shard is None else [only_shard]
    tick = f" --tick-stride {inputs['tick_stride']}" if "tick_stride" in inputs else ""
    jobs_ = [(f"python train/tah_cache.py --config {cfgp} --out {sd}/shard{k}.pt --stride {inputs['stride']}{tick} "
              f"--device cuda:0 --shard {k} --nshard {n}", gpus[i % len(gpus)], f"tahseq-{aid}-{k}",
              os.path.join(d, "logs", f"shard{k}.log")) for i, k in enumerate(shards)]
    log(f"tah_seq/{spec['name']}@{aid}: {len(jobs_)} shard job(s) of {n} on GPUs {gpus}")
    rcs = jobs.run_parallel(jobs_)
    if any(rcs):
        raise RuntimeError(f"tah_cache shard(s) failed: {rcs} (logs in {d}/logs)")
    if only_shard is not None:
        return os.path.join(sd, f"shard{only_shard}.pt")
    rc = jobs.run_in_image(f"python train/tah_merge.py --glob '{sd}/shard*.pt' --out {out}", gpus[0],
                           f"tahseq-{aid}-merge", os.path.join(d, "logs", "merge.log"))
    if rc:
        raise RuntimeError(f"tah_merge failed (log {d}/logs/merge.log)")
    import torch
    order = []
    for k in range(n):
        order += list(torch.load(os.path.join(sd, f"shard{k}.pt"), map_location="cpu", weights_only=False)["order"])
    m = A.register("tah_seq", spec["name"], files=[os.path.relpath(out, ROOT)], inputs=inputs, compat=compat,
                   producer={"tool": "acot prep tah_seq", "git": jobs.git_head(), "spec": spec},
                   notes={"order": sorted(order), "n_shards": n})
    log(f"registered tah_seq/{spec['name']}@{m['id']}")
    return m
