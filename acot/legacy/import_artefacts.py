"""Register the legacy offline products listed in acot/legacy/artefacts.yaml in the artefact store (framework §10.5):
original files stay in place; the store keeps fingerprints, inputs, compat fields and provenance gaps."""
import glob
import os

import yaml

from acot import ROOT
from acot import artefacts as A


def dataset_inputs(a):
    import h5py
    files = sorted(glob.glob(a["glob"]))
    rows = []
    for f in files:
        with h5py.File(f, "r") as h:
            fov = h.attrs.get("fov")
            rows.append({"file": os.path.basename(f), "seed": int(h.attrs["seed"]), "town": str(h.attrs.get("town")),
                         "fov": None if fov is None else float(fov)})
    want = float(a["compat"]["fov_deg"])
    bad = [r["file"] for r in rows if r["fov"] is not None and abs(r["fov"] - want) > 1e-6]
    if bad:
        raise A.ArtefactError(f"{a['name']}: h5 fov attrs disagree with compat fov_deg {want}: {bad[:5]}")
    if any(r["fov"] is None for r in rows) and "camera_override" not in a:
        raise A.ArtefactError(f"{a['name']}: files without a fov attr need an explicit camera_override with evidence")
    return files, {"glob": a["glob"], "n_files": len(files), "episodes": rows, "plan": a.get("plan"),
                   "camera_override": a.get("camera_override")}


def import_all(names=None, log=print):
    spec = yaml.safe_load(open(os.path.join(ROOT, "acot", "legacy", "artefacts.yaml")))
    out = []
    for a in spec["artefacts"]:
        if names and a["name"] not in names:
            continue
        prod = {"tool": "legacy_import", "source": "acot/legacy/artefacts.yaml"}
        if a["kind"] == "dataset":
            files, inputs = dataset_inputs(a)
            m = A.register("dataset", a["name"], files=files, inputs=inputs, compat=a["compat"], producer=prod,
                           provenance_gaps=a.get("provenance_gaps", []), content_addressed_by="files", full_hash=False)
        elif "glob" in a:                                    # a directory of files (e.g. opening snapshots)
            files = sorted(os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(ROOT, a["glob"])))
            files += list(a.get("extra_files", []))
            m = A.register(a["kind"], a["name"], files=files, inputs=dict(a.get("inputs") or {}, glob=a["glob"],
                                                                         n_files=len(files)),
                           compat=a["compat"], producer=prod, provenance_gaps=a.get("provenance_gaps", []),
                           content_addressed_by="files", full_hash=False)
        else:
            inputs = dict(a.get("inputs") or {})
            if a.get("recipe_from_ckpt"):
                import torch
                ck = torch.load(os.path.join(ROOT, a["file"]), map_location="cpu", weights_only=False)
                inputs.update(recipe=ck.get("recipe"), best_epoch=ck.get("best_epoch"),
                              best_val_acc=ck.get("best_val_acc"))
            m = A.register(a["kind"], a["name"], files=[a["file"]], inputs=inputs, compat=a["compat"], producer=prod,
                           provenance_gaps=a.get("provenance_gaps", []), content_addressed_by="files")
        log(f"{m['kind']:12s} {m['name']:26s} @{m['id']}  files {len(m['files'])}  gaps {len(m['provenance_gaps'])}")
        out.append(m)
    return out
