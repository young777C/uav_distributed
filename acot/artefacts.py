"""Artefact store (framework §10, P3a): every offline product — dataset, split, vlm_ctx, crop_feats, tah_seq, dagger,
merge, opening_snapshots, opening_cls, opening_ft, tah_ckpt, stage2_ckpt — is registered under
runs/artefacts/<kind>/<name>@<id>/manifest.json with its inputs, files (+ fingerprints), compatibility fields and
provenance gaps. Produced artefacts are content-addressed by their inputs (same inputs → same id; an existing directory
with a DIFFERENT manifest is an error, never a silent reuse); imported legacy files are addressed by their content hash.
The index maps every registered file to its artefact, so configs may reference an artefact by `artefact:<alias>` or by
its path. compat_problems() implements the compatibility matrix (§10.2)."""
import hashlib
import json
import os
import time

from acot import CONFIGS, ROOT

STORE = os.path.join(ROOT, "runs", "artefacts")
INDEX = os.path.join(STORE, "index.json")
KINDS = ("dataset", "split", "vlm_ctx", "crop_feats", "tah_seq", "dagger", "merge", "opening_snapshots", "opening_cls",
         "opening_ft", "tah_ckpt", "stage2_ckpt")
UNKNOWN = "unknown"
CHUNK = 4 << 20


class ArtefactError(ValueError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _rel(path):
    a = os.path.abspath(os.path.join(ROOT, path))
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a


def fingerprint(path, full=None):
    """{size, sha256} (full content) for files ≤ 1 GiB, else {size, sha256_partial} over size + first / last 4 MiB."""
    p = os.path.join(ROOT, path)
    size = os.path.getsize(p)
    full = size <= (1 << 30) if full is None else full
    if full:
        from acot.provenance import file_sha256
        return {"size": size, "sha256": file_sha256(path)}
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(CHUNK))
        f.seek(max(0, size - CHUNK))
        h.update(f.read(CHUNK))
    return {"size": size, "sha256_partial": h.hexdigest()}


def _digest(fp):
    return fp.get("sha256") or fp.get("sha256_partial")


# --------------------------------------------------------------------------------------------------------------- index
def load_index():
    try:
        return json.load(open(INDEX))
    except (OSError, ValueError):
        return {"by_file": {}, "aliases": {}}


def _save_index(ix):
    os.makedirs(STORE, exist_ok=True)
    tmp = INDEX + f".{os.getpid()}.tmp"
    json.dump(ix, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, INDEX)


def input_id(kind, name, inputs):
    """the id a produced artefact WILL have (content-addressed by its inputs) — to write outputs before registering."""
    return hashlib.sha256(canonical({"kind": kind, "name": name, "inputs": inputs}).encode()).hexdigest()[:12]


def artefact_dir(kind, name, aid):
    return os.path.join(STORE, kind, f"{name}@{aid}")


def register(kind, name, *, files=(), inputs=None, compat=None, producer=None, provenance_gaps=(), notes=None,
             content_addressed_by="inputs", full_hash=None, alias=True):
    """write the manifest (idempotent; a DIFFERENT manifest under the same id is an error) and index its files."""
    if kind not in KINDS:
        raise ArtefactError(f"unknown artefact kind {kind!r}")
    fl = []
    for f in files:
        fp = fingerprint(f, full_hash)
        fl.append(dict({"path": _rel(f)}, **fp))
    inputs = inputs or {}
    if content_addressed_by == "inputs":
        aid = input_id(kind, name, inputs)
    else:                                                    # imported legacy artefact: its file contents
        aid = hashlib.sha256(canonical([_digest(x) for x in fl]).encode()).hexdigest()[:12]
    m = {"kind": kind, "name": name, "id": aid, "inputs": inputs, "files": fl, "compat": compat or {},
         "provenance_gaps": list(provenance_gaps), "producer": producer or {}, "notes": notes}
    d = artefact_dir(kind, name, aid)
    path = os.path.join(d, "manifest.json")
    if os.path.exists(path):
        old = json.load(open(path))
        same = {k: old.get(k) for k in ("kind", "name", "inputs", "files", "compat")} == \
               {k: m.get(k) for k in ("kind", "name", "inputs", "files", "compat")}
        if not same:
            raise ArtefactError(f"{kind}/{name}@{aid} already exists with a DIFFERENT manifest (stale or conflicting "
                                f"artefact) — refusing to overwrite")
        m["created"] = old.get("created")
    else:
        m["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(d, exist_ok=True)
    json.dump(m, open(path, "w"), indent=1, sort_keys=True, default=str)
    ix = load_index()
    for f in fl:
        ix["by_file"][os.path.abspath(os.path.join(ROOT, f["path"]))] = {"kind": kind, "name": name, "id": aid,
                                                                         "digest": _digest(f)}
    if alias:
        ix["aliases"][name] = f"{kind}/{name}@{aid}"
    _save_index(ix)
    return m


def manifest(kind, name, aid):
    return json.load(open(os.path.join(artefact_dir(kind, name, aid), "manifest.json")))


def lookup(ref):
    """`artefact:<alias>` or a file path → manifest (None if the path is not registered)."""
    ix = load_index()
    if ref.startswith("artefact:"):
        a = ix["aliases"].get(ref[len("artefact:"):])
        if a is None:
            raise ArtefactError(f"unknown artefact alias {ref!r} (python -m acot artefacts ls)")
        kind, rest = a.split("/", 1)
        name, aid = rest.rsplit("@", 1)
        return manifest(kind, name, aid)
    e = ix["by_file"].get(os.path.abspath(os.path.join(ROOT, ref)))
    return None if e is None else manifest(e["kind"], e["name"], e["id"])


def path_of(ref):
    """the (single) file of an artefact reference, or the path itself."""
    if not ref.startswith("artefact:"):
        return ref
    m = lookup(ref)
    if len(m["files"]) != 1:
        raise ArtefactError(f"{ref} has {len(m['files'])} files — reference a file explicitly")
    return m["files"][0]["path"]


def verify(m):
    """→ list of problems (missing file / fingerprint changed)."""
    bad = []
    for f in m["files"]:
        p = os.path.join(ROOT, f["path"])
        if not os.path.exists(p):
            bad.append(f"{f['path']}: missing")
            continue
        fp = fingerprint(f["path"], full="sha256" in f)
        if _digest(fp) != _digest(f):
            bad.append(f"{f['path']}: content changed since registration")
    return bad


def ls(incomplete=False):
    """registered manifests; incomplete=True also lists directories without a manifest (a build / training in
    progress or aborted — never usable downstream, §10.3) as {kind, name, id, incomplete: True}."""
    out = []
    for kind in KINDS:
        d = os.path.join(STORE, kind)
        for x in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, x, "manifest.json")
            if os.path.exists(p):
                out.append(json.load(open(p)))
            elif incomplete and "@" in x:
                name, aid = x.rsplit("@", 1)
                out.append({"kind": kind, "name": name, "id": aid, "incomplete": True})
    return out


# ------------------------------------------------------------------------------------------------------- compatibility
def waivers():
    from acot.config.load import read_yaml
    p = os.path.join(CONFIGS, "provenance_waivers.yaml")
    return (read_yaml(p) or {}).get("waivers", {}) if os.path.exists(p) else {}


def compat_problems(m, want):
    """compare an artefact's compat fields with the wanted values → [(field, have, want)] for mismatches; a field the
    artefact does not record, or records as unknown / mixed, is a problem too (unknown ≠ match)."""
    out = []
    for k, v in want.items():
        have = (m.get("compat") or {}).get(k, UNKNOWN)
        if isinstance(have, list) and not isinstance(v, list):
            ok = len(have) == 1 and have[0] == v
        elif isinstance(have, float) or isinstance(v, float):
            ok = isinstance(have, (int, float)) and isinstance(v, (int, float)) and abs(float(have) - float(v)) < 1e-9
        else:
            ok = have == v
        if not ok:
            out.append((k, have, v))
    return out


def check_refs(refs, accepted=None):
    """refs: [(config path, reference, wanted compat dict)] → (records, errors). A problem is accepted when the artefact
    name has a waiver for that field (acot/configs/provenance_waivers.yaml) or the config path is declared in the
    method's accept_provenance_gaps."""
    W, recs, errs = waivers(), [], []
    accepted = accepted or {}
    for cpath, ref, want in refs:
        m = lookup(ref)
        if m is None:
            errs.append(f"{cpath} = {ref}: not a registered artefact (python -m acot artefacts import)")
            continue
        probs = compat_problems(m, want)
        w = W.get(m["name"], {})
        open_ = [(k, h, v) for k, h, v in probs if k not in w.get("fields", []) and cpath not in accepted]
        recs.append({"path": cpath, "artefact": f"{m['kind']}/{m['name']}@{m['id']}", "gaps": m["provenance_gaps"],
                     "waived": [(k, h, v) for k, h, v in probs if (k, h, v) not in open_]})
        for k, h, v in open_:
            errs.append(f"{cpath} = {m['kind']}/{m['name']}: {k} is {h!r}, the protocol needs {v!r} "
                        f"(add a waiver with a reason to acot/configs/provenance_waivers.yaml only if this is intended)")
    return recs, errs


def check_inputs(inputs, fields=("fov_deg", "tick_s", "feature_id", "depth_scale"), waive=()):
    """B1 (training time): the input artefacts of a recipe must agree on the compat fields → list of problems."""
    ms = [lookup(r) for r in inputs]
    errs = [f"{r}: not a registered artefact" for r, m in zip(inputs, ms) if m is None]
    ms = [m for m in ms if m is not None]
    for k in fields:
        if k in waive:
            continue
        vals = {canonical((m.get("compat") or {}).get(k, UNKNOWN)) for m in ms}
        if len(vals) > 1:
            errs.append(f"inputs disagree on {k}: " + ", ".join(f"{m['name']}={(m.get('compat') or {}).get(k, UNKNOWN)!r}"
                                                               for m in ms))
        elif vals and canonical(UNKNOWN) in vals:
            errs.append(f"inputs have {k} unknown")
    return errs
