"""Seed registry (framework §5.4): named sets of (town, seed) with a purpose; sets of different purposes must not
overlap; `locked` sets (test) need --allow-test + the experiment's prereg_ref."""
import json
import os

from acot import CONFIGS, ROOT
from acot.config.load import read_yaml
from acot.config.schema import SEED_BLOCK, SEEDSET, SchemaError, validate


def _block_pairs(b):
    validate(b, SEED_BLOCK, "seeds.block")
    out = []
    for lo, hi in b.get("ranges", []):
        out += [(b["town"], s) for s in range(int(lo), int(hi) + 1)]
    out += [(b["town"], int(s)) for s in b.get("list", [])]
    if "list_file" in b:
        d = json.load(open(os.path.join(ROOT, b["list_file"])))
        towns = d.get("towns", {})
        for s in d["seeds"]:
            t = towns.get(str(s)) if b["town"] == "*" else b["town"]
            if t is None:
                raise SchemaError(f"{b['list_file']}: no town for seed {s}")
            out.append((t, int(s)))
    ex = set(b.get("exclude", []))
    return [(t, s) for t, s in out if s not in ex]


class SeedRegistry:
    def __init__(self, path=os.path.join(CONFIGS, "seeds.yaml")):
        self.path = path
        self.sets = {}
        for name, d in read_yaml(path)["seedsets"].items():
            validate(d, SEEDSET, f"seeds.{name}")
            pairs = []
            for b in d["blocks"]:
                pairs += _block_pairs(b)
            if len(set(pairs)) != len(pairs):
                raise SchemaError(f"seeds.{name}: duplicate (town, seed) entries")
            self.sets[name] = dict(d, pairs=pairs)

    def overlaps(self):
        """[(set_a, set_b, n_common)] for sets of DIFFERENT purposes that share (town, seed) pairs."""
        names = sorted(self.sets)
        out = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                A, B = self.sets[a], self.sets[b]
                if A["purpose"] != B["purpose"]:
                    n = len(set(A["pairs"]) & set(B["pairs"]))
                    if n:
                        out.append((a, b, n))
        return out

    def select(self, spec, allow_test=False, prereg_ref=None):
        """experiment `seeds:` → (ordered [(town, seed)], purpose). `subset` = inclusive ranges inside the set."""
        name = spec["set"]
        if name not in self.sets:
            raise SchemaError(f"unknown seed set {name!r} (have {sorted(self.sets)})")
        S = self.sets[name]
        if S.get("locked") and not (allow_test and prereg_ref):
            raise PermissionError(f"seed set {name!r} is locked: needs --allow-test and the experiment's prereg_ref")
        pairs = S["pairs"]
        if spec.get("subset"):
            keep = set()
            for lo, hi in spec["subset"]:
                keep |= set(range(int(lo), int(hi) + 1))
            pairs = [p for p in pairs if p[1] in keep]
            missing = keep - {s for _, s in pairs}
            if missing:
                raise SchemaError(f"seeds.subset: {sorted(missing)[:5]}… are not in set {name!r}")
        if spec.get("expect") is not None and len(pairs) != spec["expect"]:
            raise SchemaError(f"seed set {name!r} → {len(pairs)} seeds, experiment expects {spec['expect']}")
        return pairs, S["purpose"]
