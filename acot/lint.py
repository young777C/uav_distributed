"""`acot lint` (framework §4): every protocol / method / experiment resolves; seed sets of different purposes do not overlap;
the legacy surface is fully mapped; legacy training configs are scanned for the known conflicts (shared output dirs,
missing split manifests). Errors fail the command; legacy findings are reported (they fail only with --strict-legacy)."""
import glob
import os

from acot import CONFIGS, ROOT
from acot.config import coverage
from acot.config.load import read_yaml
from acot.config.resolve import ResolveError, check_protocol, load_protocol, resolve_arm, resolve_experiment
from acot.config.schema import SchemaError
from acot.config.seeds import SeedRegistry

OUTPUT_KEYS = ("context_cache", "stage2_context_cache", "ckpt_dir", "stage2_ckpt_dir")


def legacy_findings():
    """known conflicts in train/config*.yaml: output dirs shared by configs over different data; missing split files."""
    out, owners = [], {}
    for p in sorted(glob.glob(os.path.join(ROOT, "train", "config*.yaml"))):
        cfg = read_yaml(p)
        name = os.path.relpath(p, ROOT)
        data, train = cfg.get("data", {}) or {}, cfg.get("train", {}) or {}
        for k in OUTPUT_KEYS:
            v = data.get(k, train.get(k))
            if v:
                owners.setdefault(v, []).append((name, data.get("episodes_glob")))
        sm = data.get("split_manifest")
        if sm and not os.path.exists(os.path.join(ROOT, sm)):
            out.append(f"{name}: split_manifest {sm} is missing → train/dataset.py falls back to a naive slice "
                       f"(empty test split)")
    for d, users in sorted(owners.items()):
        globs = {g for _, g in users}
        if len(users) > 1 and len(globs) > 1:
            out.append(f"output dir {d} shared by configs over different data: " + ", ".join(n for n, _ in users))
    return out


def run(strict_legacy=False, infra=False, quiet=False):
    errors, notes = [], []
    protos = {}
    for p in sorted(glob.glob(os.path.join(CONFIGS, "protocols", "*.yaml"))):
        ref = os.path.basename(p)[:-5]
        try:
            protos[ref] = load_protocol(ref)
            check_protocol(protos[ref], ref)
        except (ResolveError, SchemaError, FileNotFoundError) as e:
            errors.append(f"protocol {ref}: {e}")
    base = protos.get("sim_proxy_v2@1")
    for p in sorted(glob.glob(os.path.join(CONFIGS, "methods", "*.yaml"))):
        name = os.path.basename(p)[:-5]
        if name == "_base" or base is None:
            continue
        try:
            resolve_arm(base, name, {"method": name})
        except (ResolveError, SchemaError, FileNotFoundError) as e:
            errors.append(f"method {name}: {e}")
    for p in sorted(glob.glob(os.path.join(CONFIGS, "experiments", "**", "*.yaml"), recursive=True)):
        rel = os.path.relpath(p, CONFIGS)
        try:
            resolve_experiment(p)
        except PermissionError as e:
            notes.append(f"experiment {rel}: {e} (not resolved)")
        except (ResolveError, SchemaError, FileNotFoundError, KeyError) as e:
            errors.append(f"experiment {rel}: {e}")
    try:
        for a, b, n in SeedRegistry().overlaps():
            errors.append(f"seed sets {a} and {b} (different purposes) share {n} (town, seed) pairs")
    except (SchemaError, FileNotFoundError) as e:
        errors.append(f"seeds.yaml: {e}")
    for k, v in coverage.problems().items():
        errors.append(f"legacy surface {k}: {v[:8]}{' …' if len(v) > 8 else ''}")
    legacy = legacy_findings()
    if infra:
        from acot.infra.preflight import check_static
        errors += [f"infra: {e}" for e in check_static()]
    if not quiet:
        for e in errors:
            print(f"ERROR  {e}")
        for n in notes:
            print(f"note   {n}")
        for f in legacy:
            print(f"LEGACY {f}")
        print(f"\nlint: {len(errors)} error(s), {len(legacy)} legacy finding(s), {len(notes)} note(s)")
    return 1 if errors or (strict_legacy and legacy) else 0
