"""Translator (framework §6, P0–P1): a resolved arm → the legacy invocation (policy_server argv + policy-container env,
run_rollout-equivalent client argv + env) — table-driven by acot/config/legacy_map.yaml, so the map and the translator
cannot diverge. semantic() normalises ANY invocation (captured from a legacy launcher or produced here) to the values
the legacy code actually uses (argparse parse + env defaults / empty-equals-unset / JSON-kwargs merged with constructor
defaults) — the L0 golden comparison (§9) is equality of semantic() outputs."""
import ast
import json
import os
import shlex

from acot.config import coverage
from acot.config.legacy_args import DAM4SAM_SERVICE, POLICY_SERVER, RUN_ROLLOUT, defaults, parse
from acot.config.load import get_dotted
from acot.tools.legacy_surface import signature_rows

MAP = coverage.load_map()
ALWAYS = ("--config", "--ckpt", "--language-mode", "--s2-period", "--ex-source", "--port")   # as eval_asd.sh
_ABSENT = object()
POLICY_ENV = ("ASD_LOG", "ASD_CFG", "COLD_WINDOW", "COLD_LAMBDA", "COLD_SINGLE", "V1_SELFRESET", "DAM4SAM_DIAG",
              "DAM4SAM_EVERY_FRAME", "TRK_CFG", "OPENING_DUMP", "OPENING", "OPENING_CFG", "ORACLE_ABSENT",
              "TAH_DAGGER_DIR", "DAGGER_DIR")
COLLECT_ENV = {"ASD_LOG": "ticks", "OPENING_DUMP": "opening_snapshots", "TAH_DAGGER_DIR": "tah_dagger",
               "DAGGER_DIR": "dagger", "DAM4SAM_DIAG": "dam4sam_diag", "DUMP_LABELS": "actor_labels"}


def _ctor_defaults():
    """{class: {kwarg: default}} from the legacy constructors' source (single source of truth for JSON builders)."""
    out = {}
    for _, _, k, d, _, cls in signature_rows():
        try:
            out.setdefault(cls, {})[k] = ast.literal_eval(d)
        except (ValueError, SyntaxError):
            out.setdefault(cls, {})[k] = d          # REQUIRED / injected
    return out


CTOR = _ctor_defaults()


def _from(arm, spec):
    paths = spec["from"] if isinstance(spec["from"], list) else [spec["from"]]
    for p in paths:
        top, rest = p.split(".", 1)
        v = get_dotted(arm[top], rest, _ABSENT)
        if v is not _ABSENT:
            return v
    return _ABSENT


def _mapped(spec, v):
    if v is _ABSENT or "map" not in spec:
        return v
    return spec["map"][v]


def _nondefault(values, cls, skip=()):
    d = CTOR[cls]
    return {k: v for k, v in sorted(values.items()) if k not in skip and d.get(k, _ABSENT) != v}


# --------------------------------------------------------------------------------------------------- JSON env builders
def asd_cfg(arm):
    f = arm["method"]["filter"]
    if f["mode"] == "off":
        return ""
    vals = {k: v for k, v in f.items() if k not in ("mode", "v1_selfreset", "gate_commit", "force_conf", "inherit",
                                                    "inherit_gate_px", "inherit_px_per_tick")}
    return json.dumps(_nondefault(vals, "ASD", skip=("mode", "log_path")), separators=(",", ":"))


def trk_cfg(arm):
    p = arm["protocol"]["perception"]
    if p["identity_keys"] != "tracker":
        return ""
    tk = {k: v for k, v in p["tracker"].items() if k != "assign"}     # assign: refused by hardcoded_problems
    return json.dumps(_nondefault(tk, "TrackletAssigner"), separators=(",", ":"))


def opening_cfg(arm):
    o = arm["protocol"]["frontend"]["opening"]
    if o["module"] == "head":
        return ""
    vals = {k: o[k] for k in ("window", "agg", "final_reset", "reset_asd", "force_final", "seed_asd")}
    out = _nondefault(vals, "OPENING_CFG")
    if o["module"] == "attr":
        a = dict(o["attr"])
        if a.pop("ckpt_ft") != "carla_uav_tracking/rollout/eval_out/opening_attr_ft.pt":
            raise ValueError("opening.attr.ckpt_ft is hard-coded in opening_zoom.py until P2")
        lin = a.pop("ckpt_lin")
        a["ckpt"] = "" if lin == "carla_uav_tracking/rollout/eval_out/opening_attr_cls.pt" else f"/nvidia/hque/code/cyhe/uav-acot-track/{lin}"
        out.update(_nondefault(a, "AttrScorer"))
    else:
        out.update(_nondefault(o["zoom"], "ZoomVerifier"))
    return json.dumps(out, separators=(",", ":"))


BUILDERS = {"asd_cfg": asd_cfg, "trk_cfg": trk_cfg, "opening_cfg": opening_cfg}


# --------------------------------------------------------------------------------------------------------- translation
def policy_argv(arm, infra):
    D = defaults(POLICY_SERVER)
    head = arm["method"]["identity"]["head"]
    svc = (infra.get("services", {}).get("dam4sam", {}).get("instances") or [{}])[0]
    infra_vals = {"policy_port": infra["policy_port"], "dam4sam_host": svc.get("host", "dam4sam-svc"),
                  "dam4sam_port": svc.get("port", 5601)}
    argv = []
    for flag, spec in MAP["policy_server"].items():
        dest = flag[2:].replace("-", "_")
        if spec.get("when_head") and head not in spec["when_head"]:
            continue
        if "fixed" in spec:
            v = spec["fixed"]
        elif "infra" in spec:
            v = infra_vals[spec["infra"]]
        else:
            v = _mapped(spec, _from(arm, spec))
        if v is _ABSENT or (v == D[dest] and flag not in ALWAYS):
            continue
        if isinstance(v, bool):
            if v:
                argv.append(flag)
            continue
        argv += [flag, str(v)]
    return argv


def policy_env(arm, collect_paths=None):
    """the behaviour + collection env of the policy container (eval_asd.sh style: every variable explicit)."""
    env = {}
    for var in POLICY_ENV:
        spec = MAP["env"][var]
        if "build" in spec:
            env[var] = BUILDERS[spec["build"]](arm)
        elif "collect" in spec:
            v = (collect_paths or {}).get(var)
            if v is None and spec.get("presence") == "is_not_none":
                continue                 # legacy code tests `is not None`: an empty value would switch it ON
            env[var] = v if v is not None else spec.get("default", "")
        else:
            v = _mapped(spec, _from(arm, spec))
            env[var] = spec.get("default", "") if v is _ABSENT else str(v)
    return env


def client_values(arm, seeds):
    """the run_rollout-equivalent client settings for a block of consecutive seeds of ONE town."""
    m, p = arm["method"], arm["protocol"]
    return {"episodes": len(seeds), "seed_base": seeds[0]["seed"], "town": seeds[0]["town"],
            "steps": p["sim"]["episode_steps"], "vis_raycast": p["perception"]["occlusion"] == "raycast",
            "hires": p["frontend"]["opening"]["hires_k"], "expert_control": m["control"]["expert_control"],
            "predict_road": m["control"]["predict_road"], "road_oracle": m["control"]["road_oracle"],
            "retarget_tick": p["sim"]["retarget_tick"]}


def client_argv(vals, infra_vals):
    argv = ["--policy-host", infra_vals["policy_host"], "--policy-port", str(infra_vals["policy_port"]),
            "--episodes", str(vals["episodes"]), "--seed-base", str(vals["seed_base"]), "--steps", str(vals["steps"]),
            "--town", vals["town"], "--out", infra_vals["out"]]
    for k in ("vis_raycast", "expert_control", "predict_road", "road_oracle"):
        if vals[k]:
            argv.append("--" + k.replace("_", "-"))
    if vals["hires"] != 1:
        argv += ["--hires", str(vals["hires"])]
    if vals["retarget_tick"] is not None:
        argv += ["--retarget-tick", str(vals["retarget_tick"])]
    return argv


def collect_paths_for(R, arm_name, tag="w0"):
    """collector → legacy env var with a path under the arm's results dir (absolute host path = policy-container path)."""
    base = f"{R['infra']['paths']['repo']}/{R['infra']['results_root']}/{R['name']}/{arm_name}"
    c, out = set(R["collect"]), {}
    if "ticks" in c or "dam4sam_diag" in c:
        out["ASD_LOG"] = f"{base}/logs/asd_{tag}.jsonl"
    if "dam4sam_diag" in c:
        out["DAM4SAM_DIAG"] = "1"
    if "tah_dagger" in c:
        out["TAH_DAGGER_DIR"] = f"{base}/collect/tah_dagger"
    if "dagger" in c:
        out["DAGGER_DIR"] = f"{base}/collect/dagger"
    if "opening_snapshots" in c:
        out["OPENING_DUMP"] = f"{base}/collect/opening_snapshots"
    if "actor_labels" in c:
        out["DUMP_LABELS"] = f"/workspace/{R['infra']['results_root'].split('carla_uav_tracking/', 1)[1]}/{R['name']}/" \
                             f"{arm_name}/collect/actor_labels"
    return out


# method fields the legacy policy does not read (it hard-codes the value): any other value cannot run on the legacy runtime
LEGACY_HARDCODED = {("identity", "tah", "lang_reanchor_thr"): 0.6, ("identity", "dam4sam", "reinit_cooldown"): 3}


def hardcoded_problems(arm):
    """→ settings of a resolved arm the legacy runtime would silently ignore (legacy_map.yaml constants)."""
    out = []
    for path, want in LEGACY_HARDCODED.items():
        v = arm["method"]
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None and v != want:
            out.append(f"method.{'.'.join(path)}={v!r} (the legacy policy hard-codes {want!r})")
    if arm["method"]["control"].get("absent_where", "ear") != "ear":
        out.append("method.control.absent_where (the legacy policy keeps the EAR sample; acot runtime only)")
    if (arm["method"]["identity"].get("tah") or {}).get("feature", "dinov2_crop") != "dinov2_crop":
        out.append("method.identity.tah.feature (DINOv3 crops; acot runtime only)")
    if (arm["method"]["identity"].get("tah") or {}).get("memory_write", "own") != "own":
        out.append("method.identity.tah.memory_write (acot runtime only)")
    if arm["protocol"]["frontend"]["opening"].get("shared"):
        out.append("protocol.frontend.opening.shared (shared opening; acot runtime only)")
    for k in ("gate_commit", "force_conf"):
        if not arm["method"]["filter"].get(k, True):
            out.append(f"method.filter.{k}=false (not implemented by the legacy policy; acot runtime only)")
    if arm["method"]["filter"].get("inherit"):
        out.append("method.filter.inherit=true (evidence inheritance; acot runtime only)")
    if any((arm["protocol"]["sim"].get("env_determinism") or {}).values()):
        out.append("protocol.sim.env_determinism (acot CARLA client only)")
    if arm["protocol"]["sim"].get("policy_rng", "unseeded") != "unseeded":
        out.append("protocol.sim.policy_rng (the legacy policy does not reseed its sampling RNG; acot runtime only)")
    if arm["protocol"]["frontend"]["opening"].get("handoff", "legacy") != "legacy":
        out.append("protocol.frontend.opening.handoff (the legacy policy seeds assoc / dam4sam once; acot runtime only)")
    if (arm["protocol"]["perception"].get("tracker") or {}).get("assign", "greedy") != "greedy":
        out.append("protocol.perception.tracker.assign (the legacy policy image has no scipy → greedy; acot runtime only)")
    return out


def translate(R, arm_name, worker=None, seeds=None, collect_paths=None):
    """resolved experiment + arm → {policy_argv, policy_env, client: {values, argv}} (worker = infra placement)."""
    arm = R["arms"][arm_name]
    infra = R["infra"]
    seeds = seeds or R["seeds"]
    if collect_paths is None:
        collect_paths = collect_paths_for(R, arm_name)
    w = worker or {"container": f"{infra['docker']['policy_container_prefix']}{R['name']}-{arm_name}",
                   "carla_port": 2000, "tm_port": infra["tm_ports"]["base"], "gpu": infra["policy_gpus"][0]}
    vals = client_values(arm, seeds)
    out = f"/workspace/{infra['results_root'].split('carla_uav_tracking/', 1)[1]}/{R['name']}/{arm_name}/legacy.json"
    return {"arm": arm_name, "config_hash": arm["config_hash"], "worker": w,
            "policy_argv": policy_argv(arm, infra), "policy_env": policy_env(arm, collect_paths),
            "pip_preinstall": infra.get("pip_preinstall"),
            "client": {"values": vals, "argv": client_argv(vals, {"policy_host": w["container"],
                                                                  "policy_port": infra["policy_port"], "out": out}),
                       "env": {"CARLA_PORT": str(w["carla_port"]), "CARLA_TM_PORT": str(w["tm_port"]),
                               "DUMP_LABELS": (collect_paths or {}).get("DUMP_LABELS", "")}}}


# ------------------------------------------------------------------------------------------------- semantic normaliser
def _json_kwargs(s):
    s = (s or "").strip()
    return json.loads(s) if s else {}


def _collect(e, present=None):
    """collection channels switched on by the env. DAGGER_DIR is tested with `is not None` in policy.py, so it is ON
    whenever the variable EXISTS (even empty); the others are truthiness tests."""
    on = [k for k in ("ASD_LOG", "OPENING_DUMP", "TAH_DAGGER_DIR") if e.get(k)]
    if ("DAGGER_DIR" in present) if present is not None else ("DAGGER_DIR" in e):
        on.append("DAGGER_DIR")
    if e.get("DAM4SAM_DIAG") == "1" and e.get("ASD_LOG"):
        on.append("DAM4SAM_DIAG")
    return sorted(on)


def semantic(policy_argv, policy_env, client_argv, client_env):
    """what the legacy code actually uses (scientific + diagnostic + collection presence); infra excluded."""
    a = parse(POLICY_SERVER, policy_argv)
    e = {k: v for k, v in (policy_env or {}).items()}
    out = {"policy": {k: v for k, v in a.items() if k not in ("device", "port", "dam4sam_host", "dam4sam_port",
                                                               "asd_cfg")}}
    if a["asd"] != "off":
        kw = dict(CTOR["ASD"])
        kw.update(_json_kwargs(a["asd_cfg"] or e.get("ASD_CFG")))
        kw.pop("mode", None)
        kw.pop("log_path", None)
        out["asd_kwargs"] = kw
        out["v1_selfreset"] = e.get("V1_SELFRESET", "0") == "1"
    if a["tracklets"] == "tracker":
        kw = dict(CTOR["TrackletAssigner"])
        kw.update(_json_kwargs(e.get("TRK_CFG")))
        out["tracker_kwargs"] = kw
    module = e.get("OPENING", "head") or "head"
    module = module if module in ("zoom", "attr") else "head"
    out["opening"] = {"module": module}
    if module != "head":
        cfg = _json_kwargs(e.get("OPENING_CFG"))
        pops = {k: cfg.pop(k, CTOR["OPENING_CFG"][k]) for k in list(CTOR["OPENING_CFG"])}
        scorer = {k: v for k, v in CTOR["AttrScorer" if module == "attr" else "ZoomVerifier"].items()
                  if k not in ("model", "processor", "device")}
        scorer.update(cfg)
        out["opening"].update(pops=pops, scorer=scorer)
    out["cold"] = {"window": int(e.get("COLD_WINDOW", "0") or 0), "lambda": float(e.get("COLD_LAMBDA", "0.9") or 0.9),
                   "single": e.get("COLD_SINGLE", "0") == "1"}
    out["dam4sam_every_frame"] = e.get("DAM4SAM_EVERY_FRAME") == "1"
    oa = e.get("ORACLE_ABSENT", "0")
    out["oracle_absent"] = oa if oa in ("1", "2") else "0"
    out["collect"] = _collect(e)
    c = parse(RUN_ROLLOUT, client_argv, env={"CARLA_PORT": (client_env or {}).get("CARLA_PORT", "2012")})
    out["client"] = {k: c[k] for k in ("episodes", "seed_base", "steps", "town", "expert_control", "road_oracle",
                                       "vis_raycast", "retarget_tick", "hires")}
    out["client"]["predict_road"] = c["predict_road"] or c["road_oracle"]
    if (client_env or {}).get("DUMP_LABELS"):
        out["collect"].append("DUMP_LABELS")
    return out


def semantic_of(inv):
    return semantic(inv["policy_argv"], inv["policy_env"], inv["client"]["argv"], inv["client"]["env"])


def dam4sam_argv(arm, infra):
    svc = infra["services"]["dam4sam"]["instances"][0]
    argv = ["--model", arm["method"]["identity"]["dam4sam"]["model"], "--port", str(svc["port"])]
    parse(DAM4SAM_SERVICE, argv)
    return argv


def shell_join(argv):
    return " ".join(shlex.quote(x) for x in argv)


def semantic_from_echo(echo, client_argv, client_env):
    """server_echo.json (captured constructor kwargs + derived Policy attributes) → the same structure as semantic(), so
    the EFFECTIVE configuration inside the policy process can be compared with the translator's (L1)."""
    cap = echo["captured"]
    P = cap["Policy"][0]
    kw = dict(P["kwargs"])
    a = parse(POLICY_SERVER, echo["argv"])
    pol = {k: v for k, v in a.items() if k not in ("device", "port", "dam4sam_host", "dam4sam_port", "asd_cfg")}
    for k in list(pol):
        if k in kw:
            pol[k] = kw[k]
    pol["ckpt"] = os.path.relpath(P["args"][0], echo["repo"])           # policy_server joins --ckpt with the repo
    if kw.get("tid_ckpt"):
        pol["tid_ckpt"] = os.path.relpath(kw["tid_ckpt"], echo["repo"])
    vlm = cap.get("OnlineVLM", [{}])[0].get("kwargs", {})
    pol["language_mode"] = vlm.get("language_mode", pol["language_mode"])
    out = {"policy": pol}
    if pol["asd"] != "off":
        k2 = dict(CTOR["ASD"])
        k2.update(cap["ASD"][0]["kwargs"])
        k2.pop("mode", None)
        k2.pop("log_path", None)
        out["asd_kwargs"] = k2
        out["v1_selfreset"] = bool(echo["policy_attrs"].get("v1_selfreset", False))
    if pol["tracklets"] == "tracker":
        k3 = dict(CTOR["TrackletAssigner"])
        k3.update(cap["TrackletAssigner"][0]["kwargs"])
        out["tracker_kwargs"] = k3
    at = echo["policy_attrs"]
    if "AttrScorer" in cap or "ZoomVerifier" in cap:
        module = "attr" if "AttrScorer" in cap else "zoom"
        cls = "AttrScorer" if module == "attr" else "ZoomVerifier"
        scorer = {k: v for k, v in CTOR[cls].items() if k not in ("model", "processor", "device")}
        scorer.update({k: v for k, v in cap[cls][0]["kwargs"].items() if k not in ("model", "processor", "device")})
        pops = {"window": at["_open_window"], "agg": at["_open_agg"], "final_reset": at["_open_final_reset"],
                "reset_asd": at["_open_reset_asd"], "force_final": at["_open_force_final"], "seed_asd": at["_open_seed_asd"]}
        out["opening"] = {"module": module, "pops": pops, "scorer": scorer}
    else:
        out["opening"] = {"module": "head"}
    out["cold"] = {"window": at["cold_window"], "lambda": at["cold_lambda"], "single": at["cold_single"]}
    out["dam4sam_every_frame"] = bool(at.get("_dam4sam_every_frame", False))
    oa = echo["env"].get("ORACLE_ABSENT", "0")
    out["oracle_absent"] = oa if oa in ("1", "2") else "0"
    e = echo["env"]
    out["collect"] = _collect(e, echo.get("env_set"))
    c = parse(RUN_ROLLOUT, client_argv, env={"CARLA_PORT": (client_env or {}).get("CARLA_PORT", "2012")})
    out["client"] = {k: c[k] for k in ("episodes", "seed_base", "steps", "town", "expert_control", "road_oracle",
                                       "vis_raycast", "retarget_tick", "hires")}
    out["client"]["predict_road"] = c["predict_road"] or c["road_oracle"]
    if (client_env or {}).get("DUMP_LABELS"):
        out["collect"].append("DUMP_LABELS")
    return out
