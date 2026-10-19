"""`acot train acot/configs/recipes/<x>.yaml [seed=N]` for trainer tah_relm (framework §10, P3b): the legacy
train/tah_train.py in the training image, fed by REGISTERED input artefacts (order matters: the validation split is the
first val_frac of the first input). The inputs must agree on the compat fields unless the recipe waives a field with a
reason (B1). The best and last checkpoints are registered as two tah_ckpt artefacts whose compat is derived from the
inputs."""
import json
import os

from acot import ROOT
from acot import artefacts as A
from acot import jobs

CODE = ("train/tah_train.py", "train/tah.py")
FLAGS = ("rel", "motion", "aug")


def _compat(ms, arch):
    def one(k):
        vals = sorted({json.dumps((m.get("compat") or {}).get(k, A.UNKNOWN)) for m in ms})
        vals = [json.loads(v) for v in vals]
        return vals[0] if len(vals) == 1 else vals
    out = {k: one(k) for k in ("fov_deg", "tick_s", "feature_id", "depth_scale", "d_feat")}
    out["arch"] = arch
    return out


def train(recipe, gpu, log=print, register_only=False):
    """register_only: the training container ran to completion but the host process died before registering (the
    container outlives its `docker run` client) → register the finished outputs without training again."""
    waive = dict(recipe.get("waivers") or {})
    # 2026-10-12 P0-2: optional val_inputs = a separate validation set (e.g. deployment-distribution DAgger episodes)
    # used for best-epoch selection; then ALL episodes of `inputs` are trained on and hparams.val_frac is unused
    vrefs = list(recipe.get("val_inputs") or [])
    errs = A.check_inputs(recipe["inputs"] + vrefs, waive=tuple(waive))
    if errs:
        raise A.ArtefactError("recipe inputs are incompatible (declare a waiver with a reason only if intended): "
                              + "; ".join(errs))
    ms = [A.lookup(r) for r in recipe["inputs"]]
    vms = [A.lookup(r) for r in vrefs]
    hp = dict(recipe["hparams"])
    seed = recipe.get("seed")
    arch = "relm" if hp.get("motion") else ("rel" if hp.get("rel") else "abs")
    inputs = {"inputs": [f"{m['kind']}/{m['name']}@{m['id']}" for m in ms], "hparams": hp, "seed": seed,
              "waivers": waive, "code": jobs.code_hash(*CODE)}
    if vms:
        inputs["val_inputs"] = [f"{m['kind']}/{m['name']}@{m['id']}" for m in vms]
    aid = A.input_id("tah_ckpt", recipe["name"], inputs)
    d = A.artefact_dir("tah_ckpt", recipe["name"], aid)
    if os.path.exists(os.path.join(d, "manifest.json")):
        log(f"tah_ckpt/{recipe['name']}@{aid} already trained (content-addressed) — nothing to do")
        return A.manifest("tah_ckpt", recipe["name"], aid)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "ckpt.pt")
    logp = os.path.join(d, "logs", "train.log")
    if register_only:
        done = os.path.exists(out) and os.path.exists(logp) and "[tah] done." in open(logp).read()
        if not done:
            raise RuntimeError(f"tah_ckpt/{recipe['name']}@{aid}: training has not finished ({logp})")
    else:
        caches = " ".join(os.path.join(ROOT, A.path_of(r)) for r in recipe["inputs"])
        flags = " ".join(f"--{k}" for k in FLAGS if hp.get(k))
        if hp.get("group_phases"):
            flags += " --group-phases"
        if vrefs:
            flags += " --val-cache " + " ".join(os.path.join(ROOT, A.path_of(r)) for r in vrefs)
        cmd = (f"python train/tah_train.py --cache {caches} --out {out} --epochs {hp['epochs']} --lr {hp['lr']} "
               f"--val-frac {hp['val_frac']} {flags} --fnoise {hp['fnoise']} --pjit {hp['pjit']} --wd {hp['wd']} "
               f"--device cuda:0" + (f" --seed {seed}" if seed is not None else "") +
               (" --save-last" if recipe.get("save_last", True) else ""))
        log(f"tah_ckpt/{recipe['name']}@{aid}: training on GPU {gpu}")
        rc = jobs.run_in_image(cmd, gpu, f"tahtrain-{aid}", logp)
        if rc:
            raise RuntimeError(f"tah_train failed (log {logp})")
    import torch
    ck = torch.load(out, map_location="cpu", weights_only=False)
    compat = _compat(ms, arch)
    gaps = [f"waived {k}: {v}" for k, v in waive.items()]
    prod = {"tool": "acot train tah_relm", "git": jobs.git_head(), "recipe": recipe}
    best = A.register("tah_ckpt", recipe["name"], files=[os.path.relpath(out, ROOT)], inputs=inputs, compat=compat,
                      producer=prod, provenance_gaps=gaps,
                      notes={"select": "best", "best_epoch": ck.get("best_epoch"), "best_val_acc": ck.get("best_val_acc")})
    log(f"registered tah_ckpt/{recipe['name']}@{best['id']} (best epoch {ck.get('best_epoch')}, "
        f"val acc {ck.get('best_val_acc')})")
    last = os.path.join(d, "ckpt_last.pt")
    if os.path.exists(last):
        lk = torch.load(last, map_location="cpu", weights_only=False)
        if lk.get("epoch") != hp["epochs"]:
            raise RuntimeError(f"{last}: epoch {lk.get('epoch')} ≠ {hp['epochs']} (incomplete training)")
        A.register("tah_ckpt", recipe["name"] + "_last", files=[os.path.relpath(last, ROOT)], inputs=inputs,
                   compat=compat, producer=prod, provenance_gaps=gaps,
                   notes={"select": "last", "epoch": lk.get("epoch"), "val_acc": lk.get("val_acc")})
    return best
