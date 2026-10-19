"""`acot train` for trainers opening_attr_lin / opening_attr_ft (framework §10, P3b): the legacy
carla_uav_tracking/rollout/opening_cls.py / opening_ft.py (training snapshots = the opening_snapshots artefact
runs/opening_dump/npz_tr, which those scripts read) in the training image; the checkpoint is registered with its
feature id, vocabulary hash and hi-res factor."""
import hashlib
import json
import os

from acot import ROOT
from acot import artefacts as A
from acot import jobs

SCRIPT = {"opening_attr_lin": ("carla_uav_tracking/rollout/opening_cls.py", "opening_cls",
                               "dino_s14_lvd142m_clsmean768_s1.5-2.0_flip"),
          "opening_attr_ft": ("carla_uav_tracking/rollout/opening_ft.py", "opening_ft", "dinov2_s14_ft4_c224")}


def train(recipe, gpu, dry_run=False, log=print):
    script, kind, fid = SCRIPT[recipe["trainer"]]
    snaps = A.lookup(recipe["snapshots"])
    if snaps is None or snaps["kind"] != "opening_snapshots":
        raise A.ArtefactError(f"{recipe['snapshots']}: not an opening_snapshots artefact")
    args = dict(recipe.get("args") or {})
    inputs = {"snapshots": f"opening_snapshots/{snaps['name']}@{snaps['id']}", "args": args,
              "code": jobs.code_hash(script, "carla_uav_tracking/rollout/opening_zoom.py",
                                     "carla_uav_tracking/rollout/opening_score.py")}
    aid = A.input_id(kind, recipe["name"], inputs)
    d = A.artefact_dir(kind, recipe["name"], aid)
    out = os.path.join(d, "ckpt.pt")
    flags = " ".join(f"--{k.replace('_', '-')}" if v is True else f"--{k.replace('_', '-')} {v}"
                     for k, v in args.items() if v is not False)
    cmd = f"cd carla_uav_tracking/rollout && python {os.path.basename(script)} --out {out} --device cuda:0 {flags}"
    if dry_run:
        return {"id": aid, "command": cmd, "inputs": inputs}
    if os.path.exists(os.path.join(d, "manifest.json")):
        return A.manifest(kind, recipe["name"], aid)
    os.makedirs(d, exist_ok=True)
    rc = jobs.run_in_image(cmd, gpu, f"{kind}-{aid}", os.path.join(d, "logs", "train.log"))
    if rc:
        raise RuntimeError(f"{script} failed (log {d}/logs/train.log)")
    import torch
    ck = torch.load(out, map_location="cpu", weights_only=False)
    vocab = hashlib.sha256(json.dumps([list(ck["cls_m"]), list(ck["cls_c"])]).encode()).hexdigest()[:12]
    compat = dict(snaps.get("compat") or {}, feature_id=fid, vocab_hash=vocab)
    return A.register(kind, recipe["name"], files=[os.path.relpath(out, ROOT)], inputs=inputs, compat=compat,
                      producer={"tool": f"acot train {recipe['trainer']}", "git": jobs.git_head(), "recipe": recipe})
