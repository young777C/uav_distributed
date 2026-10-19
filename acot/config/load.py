"""YAML loading with `extends:` composition, deep merge and dotted `key=value` overrides (framework §5.1)."""
import copy
import os

import yaml


def read_yaml(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    return {} if d is None else d


def deep_merge(base, over):
    """new dict = base ⊕ over (mappings merge recursively; everything else — lists included — is replaced)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_dotted(d, key, default=KeyError):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is KeyError:
                raise KeyError(key)
            return default
        cur = cur[part]
    return cur


def set_dotted(d, key, value):
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise KeyError(f"{key}: '{part}' is not a mapping")
    cur[parts[-1]] = value


def flatten(d, prefix=""):
    """{"a": {"b": 1}} → {"a.b": 1} (leaves only)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def parse_override(s):
    """'a.b.c=VALUE' → ('a.b.c', yaml-parsed VALUE)."""
    if "=" not in s:
        raise ValueError(f"override {s!r} is not key=value")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def load_chain(name, directory, seen=()):
    """load <directory>/<name>.yaml following `extends:` (root first); returns (merged dict, [names root→leaf])."""
    path = os.path.join(directory, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} (referenced as {name!r})")
    if name in seen:
        raise ValueError(f"extends cycle: {' → '.join(seen + (name,))}")
    d = read_yaml(path)
    parent = d.get("extends")
    if parent is None:
        return d, [name]
    base, chain = load_chain(parent, directory, seen + (name,))
    return deep_merge(base, d), chain + [name]
