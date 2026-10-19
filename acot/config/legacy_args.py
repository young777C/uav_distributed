"""Static re-construction of a legacy script's argparse parser from its source (AST) — so argv can be parsed exactly as
the script would, on the host, without importing torch / carla (framework §9 L0)."""
import argparse
import ast
import os

from acot import ROOT

POLICY_SERVER = "carla_uav_tracking/rollout/policy_server.py"
RUN_ROLLOUT = "carla_uav_tracking/rollout/run_rollout.py"
DAM4SAM_SERVICE = "carla_uav_tracking/rollout/dam4sam_service.py"
_TYPES = {"int": int, "float": float, "str": str}
_CACHE = {}


def _value(node, env):
    try:
        return ast.literal_eval(node)
    except ValueError:
        pass
    # the few dynamic defaults in scope, e.g. int(os.environ.get("CARLA_PORT", "2012"))
    src = ast.unparse(node)
    return eval(src, {"int": int, "float": float, "str": str, "os": _Os(env)})   # noqa: S307 (own source, literal-only)


class _Os:
    def __init__(self, env):
        self.environ = dict(env)


def parser(path, env=None):
    """argparse.ArgumentParser equivalent to the add_argument calls of `path` (types, defaults, choices, actions)."""
    key = (path, tuple(sorted((env or {}).items())))
    if key in _CACHE:
        return _CACHE[key]
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    ap = argparse.ArgumentParser(prog=os.path.basename(path), add_help=False)
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"):
            continue
        names = [a.value for a in n.args if isinstance(a, ast.Constant)]
        kw = {}
        for k in n.keywords:
            if k.arg == "help":
                continue
            if k.arg == "type":
                kw["type"] = _TYPES[ast.unparse(k.value)]
            else:
                kw[k.arg] = _value(k.value, env or {})
        ap.add_argument(*names, **kw)
    _CACHE[key] = ap
    return ap


def parse(path, argv, env=None):
    """argv list → {dest: value} (unknown arguments raise)."""
    ns = parser(path, env).parse_args(list(argv))
    return vars(ns)


def defaults(path, env=None):
    return parse(path, [], env)
