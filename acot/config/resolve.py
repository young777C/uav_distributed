"""Resolution of an experiment into fully explicit per-arm configurations (framework §5.1): protocol (frozen, by
name@version) ⊕ method (extends chain rooted at _base) ⊕ arm override (ownership rules) ⊕ CLI overrides; inactive head /
filter sections dropped; consistency + compatibility checks; GT-channel classification; canonical config hash."""
import copy
import hashlib
import json
import os

import yaml

from acot import CONFIGS, ROOT
from acot.config import schema as S
from acot.config.load import deep_merge, flatten, get_dotted, load_chain, read_yaml, set_dotted
from acot.config.seeds import SeedRegistry

METHOD_TOPS = ("control", "identity", "filter", "oracle_absent", "diagnostics", "accept_provenance_gaps")
ABLATION_SCOPE = {"opening": ("frontend.opening",), "frontend": ("frontend",), "perception": ("perception",)}
GT_EX_SOURCES = ("cv", "cv_gated", "zerovel", "zerovel_gated", "road", "oracle")


class ResolveError(ValueError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------------- protocol / method
def load_protocol(ref):
    """'name@version' → validated protocol dict (protocols may `extends:` another protocol file)."""
    if "@" not in ref:
        raise ResolveError(f"protocol reference {ref!r} must be name@version")
    d, chain = load_chain(ref, os.path.join(CONFIGS, "protocols"))
    d = dict(d)
    d.pop("extends", None)
    S.validate(d, S.PROTOCOL, f"protocol[{ref}]")
    if f"{d['name']}@{d['version']}" != ref:
        raise ResolveError(f"protocol file {ref}.yaml declares {d['name']}@{d['version']}")
    return d


def derived_privileged(proto):
    p = proto["perception"]
    out = ["cand_box_gt_projection", "cand_depth_gt", "task_vehicle_set"]
    if p["occlusion"] == "raycast":
        out.append("occlusion_raycast_gt")
    if p["identity_keys"] == "sim":
        out.append("identity_keys_sim_actor_ids")
    out.append("where_world_gt_matched_box" if p["where_localisation"] == "gt_matched_box" else "where_world_gt_committed_id")
    sh = proto["frontend"]["opening"].get("shared")
    if sh:
        out.append("first_frame_target_box" if sh.get("gt_target") else "shared_opening_actor")
    return out


def check_protocol(proto, where):
    p = proto["perception"]
    matched = p["identity_keys"] == "tracker" or p["occlusion"] == "raycast"     # policy._committed_world branch
    want = "gt_matched_box" if matched else "gt_committed_id"
    if p["where_localisation"] != want:
        raise ResolveError(f"{where}: where_localisation must be {want} for identity_keys={p['identity_keys']}, "
                           f"occlusion={p['occlusion']} (that is what policy._committed_world does)")
    if p["identity_keys"] == "tracker" and "tracker" not in p:
        raise ResolveError(f"{where}: identity_keys=tracker needs perception.tracker")
    if p["identity_keys"] == "sim":
        p.pop("tracker", None)                                                    # unused
    o = proto["frontend"]["opening"]
    if o["module"] == "attr" and "attr" not in o:
        raise ResolveError(f"{where}: opening.module=attr needs opening.attr")
    if o["module"] == "zoom" and "zoom" not in o:
        raise ResolveError(f"{where}: opening.module=zoom needs opening.zoom")
    for k in ("attr", "zoom"):
        if o["module"] != k:
            o.pop(k, None)
    if o["module"] == "head" and o["window"] != 1:
        raise ResolveError(f"{where}: the head opening has window 1 (OPENING_CFG is ignored for module=head)")
    if sorted(proto["privileged_perception"]) != sorted(derived_privileged(proto)):
        raise ResolveError(f"{where}: privileged_perception {proto['privileged_perception']} ≠ derived "
                           f"{derived_privileged(proto)}")


def load_method(name):
    d, chain = load_chain(name, os.path.join(CONFIGS, "methods"))
    if chain[0] != "_base":
        raise ResolveError(f"method {name!r} must extend _base (chain {chain})")
    d = dict(d)
    d.pop("extends", None)
    return d, chain


def _user_keys(chain):
    """dotted keys set by the method chain EXCLUDING _base (to detect settings in sections the head does not read)."""
    keys = set()
    for n in chain[1:]:
        keys |= set(flatten({k: v for k, v in read_yaml(os.path.join(CONFIGS, "methods", f"{n}.yaml")).items()
                             if k not in ("name", "extends", "description")}))
    return keys


def normalize_method(m, user_keys, where):
    """drop identity sections the head does not read and filter fields when the filter is off; enforce required fields."""
    ident = m["identity"]
    active = S.HEAD_SECTIONS[ident["head"]]
    for sec in ("tah", "reid", "assoc", "dam4sam", "attrbind", "oavat", "deepsort", "ocsort", "oavat_full"):
        if sec in active:
            if sec not in ident:
                raise ResolveError(f"{where}: head {ident['head']} needs identity.{sec}")
            continue
        stray = sorted(k for k in user_keys if k.startswith(f"identity.{sec}."))
        if stray:
            raise ResolveError(f"{where}: {stray} set but head {ident['head']!r} does not read identity.{sec}")
        ident.pop(sec, None)
    if "tah" in active and not ident["tah"].get("ckpt"):
        raise ResolveError(f"{where}: identity.tah.ckpt must be set explicitly (no default checkpoint)")
    if "attrbind" in active and not ident["attrbind"].get("ckpt"):
        raise ResolveError(f"{where}: identity.attrbind.ckpt must be set")
    f = m["filter"]
    if f["mode"] == "off":
        m["filter"] = {"mode": "off"}            # (a parent's filter settings are switched off with it)
    else:
        if f["mode"] == "filter" and not f.get("filter_kind"):
            raise ResolveError(f"{where}: filter.mode=filter needs filter.filter_kind (the ASD class default is the "
                               f"bayes ablation)")
        if f["mode"] == "learned" and not f.get("learned_ckpt"):
            raise ResolveError(f"{where}: filter.mode=learned needs filter.learned_ckpt")
    return m


def classify_gt(proto, m):
    """the GT channels a method uses at decision time (framework §7.1) → sorted list (empty = deployable)."""
    c, i = m["control"], m["identity"]
    uses = set()
    if i["memory_seed"] == "gt":
        uses.add("gallery_seed_gt")
    if i["reanchor"]["mode"] == "oracle":
        uses.add("reanchor_oracle")
    if i["warmup_oracle_s"] > 0:
        uses.add("warmup_oracle")
    if i["event_override"]:
        uses.add("event_override")
    if i["head"] == "oracle":
        uses.add("oracle_head")
    if i["head"] == "attrbind":
        uses.add("attrbind_gt_attrs")
    if m["oracle_absent"] != "off":
        uses.add(f"oracle_absent_{m['oracle_absent']}")
    if c["ex_source"] in GT_EX_SOURCES:
        uses.add("ex_source_gt")
    for k in ("expert_control", "predict_road", "road_oracle"):
        if c[k]:
            uses.add(k)
    if proto["sim"]["retarget_tick"] is not None:
        uses.add("retarget")
    return sorted(uses)


# ---------------------------------------------------------------------------------------------------------------- checks
def model_config(proto):
    return read_yaml(os.path.join(ROOT, proto["frontend"]["vlm"]["model_config"]))


def check_compat(proto, where):
    """the policy reads camera + candidate geometry from the model config (train/config_v5.yaml image:), the env from
    carla_uav_tracking/config/default.yaml → both must agree with the protocol."""
    img = model_config(proto).get("image", {})
    cam, cand = proto["camera"], proto["perception"]["candidates"]
    want = {"fov_deg": cam["fov_deg"], "width": cam["width"], "height": cam["height"],
            "vehicle_size_m": cand["vehicle_size_m"], "box_min_px": cand["box_min_px"], "box_max_px": cand["box_max_px"]}
    bad = {k: (img.get(k), v) for k, v in want.items() if img.get(k) is None or float(img[k]) != float(v)}
    if bad:
        raise ResolveError(f"{where}: model_config {proto['frontend']['vlm']['model_config']} image ≠ protocol: "
                           + ", ".join(f"{k} {a} vs {b}" for k, (a, b) in bad.items()))
    env = read_yaml(os.path.join(ROOT, "carla_uav_tracking", "config", "default.yaml")).get("environment", {})
    if float(env.get("max_lost_seconds", -1)) != float(proto["sim"]["max_lost_s"]):
        raise ResolveError(f"{where}: carla_uav_tracking/config/default.yaml max_lost_seconds "
                           f"{env.get('max_lost_seconds')} ≠ protocol sim.max_lost_s {proto['sim']['max_lost_s']}")
    for k in ("model_config", "stage2_ckpt"):
        if not os.path.exists(os.path.join(ROOT, proto["frontend"]["vlm"][k])):
            raise ResolveError(f"{where}: frontend.vlm.{k} {proto['frontend']['vlm'][k]} does not exist")


def artefact_refs(proto, m):
    """(config path, reference, compat wanted by the protocol / method) for every artefact an arm uses (B2, §10.2)."""
    cam, crop, op, vlm = proto["camera"], proto["frontend"]["crop_features"], proto["frontend"]["opening"], \
        proto["frontend"]["vlm"]
    refs = [("frontend.vlm.stage2_ckpt", vlm["stage2_ckpt"],
             {"fov_deg": cam["fov_deg"], "language_mode": vlm["language_mode"], "ablate": m["control"]["ablate"]})]
    if op["module"] == "attr":
        for k in ("ckpt_lin", "ckpt_ft"):
            refs.append((f"frontend.opening.attr.{k}", op["attr"][k], {"fov_deg": cam["fov_deg"], "hires_k": op["hires_k"]}))
    tah = m["identity"].get("tah")
    if tah and tah.get("ckpt"):
        refs.append(("identity.tah.ckpt", tah["ckpt"],
                     {"fov_deg": cam["fov_deg"], "tick_s": proto["sim"]["s2_period"] / cam["fps"],
                      # 1006 backbone trial: the feature the TAH head is fed (shared crop DINOv2 unless dinov3_crop)
                      "feature_id": "dinov3_vitb16_lvd1689m_cls768_c224_p1.3"
                      if tah.get("feature") == "dinov3_crop" else crop["id"], "depth_scale": crop["depth_scale"]}))
    ab = m["identity"].get("attrbind")
    if ab and ab.get("ckpt"):
        refs.append(("identity.attrbind.ckpt", ab["ckpt"], {"fov_deg": cam["fov_deg"]}))
    return refs


def check_artefacts(proto, m, where):
    from acot import artefacts as A
    recs, errs = A.check_refs(artefact_refs(proto, m), accepted=m.get("accept_provenance_gaps") or {})
    if errs:
        raise ResolveError(f"{where}: " + "; ".join(errs))
    return recs


def check_method_files(m, where):
    for key in ("identity.tah.ckpt", "identity.attrbind.ckpt", "filter.learned_ckpt", "identity.oavat.checkpoint"):
        v = get_dotted(m, key, None)
        if v and not os.path.exists(os.path.join(ROOT, v)):
            raise ResolveError(f"{where}: {key} {v} does not exist")


# ------------------------------------------------------------------------------------------------------------------ arms
def resolve_arm(protocol, name, arm, cli_overrides=()):
    where = f"arm {name!r}"
    S.validate(arm, S.ARM, where)
    method, chain = load_method(arm["method"])
    user = _user_keys(chain)
    proto = copy.deepcopy(protocol)
    overrides = dict(arm.get("override") or {})
    overrides.update(dict(cli_overrides))
    ablation = arm.get("ablation")
    applied = {}
    for key, val in overrides.items():
        top = key.split(".")[0]
        if top in METHOD_TOPS:
            set_dotted(method, key, val)
            user.add(key)
        elif top in ("frontend", "perception"):
            scope = ABLATION_SCOPE.get(ablation, ())
            if not any(key == s or key.startswith(s + ".") for s in scope):
                raise ResolveError(f"{where}: overriding protocol field {key} needs `ablation:` covering it "
                                   f"(have {ablation!r}; opening → frontend.opening.*, frontend → frontend.*, "
                                   f"perception → perception.*)")
            set_dotted(proto, key, val)
        else:
            raise ResolveError(f"{where}: {key} is a protocol invariant (camera / sim / scorer / privileged_perception) "
                               f"— change the protocol version instead")
        applied[key] = val
    if ablation == "perception" or any(k.startswith("perception.") or k.startswith("frontend.opening.shared")
                                       for k in applied):
        proto["privileged_perception"] = derived_privileged(proto)
    sh = proto["frontend"]["opening"].get("shared")
    if sh and not sh.get("gt_target") and not (sh.get("file") and os.path.exists(os.path.join(ROOT, sh["file"]))):
        raise ResolveError(f"{where}: shared opening file {sh.get('file')} does not exist")
    if sh and sh.get("gt_target") and sh.get("file"):
        raise ResolveError(f"{where}: shared opening: gt_target and file are exclusive")
    S.validate(method, S.METHOD, where + ".method")
    S.validate(proto, S.PROTOCOL, where + ".protocol")
    check_protocol(proto, where)
    method = normalize_method(method, user, where)
    method["name"] = arm["method"]
    uses = classify_gt(proto, method)
    declared = sorted(set(method["diagnostics"]))
    if uses != declared:
        missing, extra = sorted(set(uses) - set(declared)), sorted(set(declared) - set(uses))
        raise ResolveError(f"{where}: diagnostics must list exactly the GT channels used — undeclared {missing}, "
                           f"unused {extra}")
    check_compat(proto, where)
    check_method_files(method, where)
    arts = check_artefacts(proto, method, where)
    role = arm.get("role") or ("diagnostic" if uses else ("ablation" if ablation else "main"))
    if uses and role != "diagnostic":
        raise ResolveError(f"{where}: uses GT channels {uses} → role must be diagnostic")
    sci = {"protocol": proto, "method": method}
    if "repeat" in arm:
        sci["repeat"] = int(arm["repeat"])
    kgap = [k for k in (method.get("accept_provenance_gaps") or {})] + \
        [f"{r['path']}: {r['artefact']} {k}={h!r}" for r in arts for k, h, _ in r["waived"]]
    ids = {r["path"]: r["artefact"] for r in arts}             # content-derived → replacing a file changes the hash
    return {"name": name, "role": role, "ablation": ablation, "method_chain": chain, "override": applied,
            "uses_gt": uses, "provenance_gaps": kgap, "artefacts": arts,
            "config_hash": config_hash(dict(sci, artefacts=ids)), **sci}


def resolve_experiment(path, cli_overrides=(), allow_test=False, arms=None):
    exp = read_yaml(path)
    S.validate(exp, S.EXPERIMENT, f"experiment[{os.path.basename(path)}]")
    protocol = load_protocol(exp["protocol"])
    check_protocol(protocol, f"protocol {exp['protocol']}")
    seeds, purpose = SeedRegistry().select(exp["seeds"], allow_test=allow_test, prereg_ref=exp.get("prereg_ref"))
    infra = read_yaml(os.path.join(CONFIGS, "infra", "default.yaml"))
    infra = deep_merge(infra, exp.get("infra") or {})
    S.validate(infra, S.INFRA, "infra")
    cli_sci = [(k, v) for k, v in cli_overrides if not k.startswith("infra.")]
    for k, v in cli_overrides:
        if k.startswith("infra."):
            set_dotted(infra, k[len("infra."):], v)
    names = list(exp["arms"]) if arms is None else arms
    unknown = [a for a in names if a not in exp["arms"]]
    if unknown:
        raise ResolveError(f"unknown arms {unknown} (have {list(exp['arms'])})")
    R = {a: resolve_arm(protocol, a, exp["arms"][a], cli_sci) for a in names}
    out = {"name": exp["name"], "source": os.path.relpath(path, ROOT), "protocol_ref": exp["protocol"],
           "prereg_ref": exp.get("prereg_ref"), "ad_hoc": bool(cli_sci),
           "seeds": [{"town": t, "seed": s} for t, s in seeds], "seed_purpose": purpose,
           "completeness": exp.get("completeness") or {"require": "all_complete", "allow_skipped_reasons": [],
                                                       "pairing": "strict"},
           "retry": exp.get("retry") or {"infra_attempts": 2},
           "collect": exp.get("collect") or ["ticks", "trk", "gt_access", "scenario"],
           "comparisons": exp.get("comparisons") or [], "infra": infra, "arms": R}
    if exp.get("schedule"):
        out["schedule"] = dict(exp["schedule"])              # 1006 G10: randomised dynamic dispatch
    out["seeds_hash"] = config_hash(out["seeds"])
    return out


def diff_vs_base(arm):
    """dotted differences of an arm's method from _base (after normalisation of _base with the same head)."""
    base = read_yaml(os.path.join(CONFIGS, "methods", "_base.yaml"))
    fb, fm = flatten({k: v for k, v in base.items() if k != "name"}), flatten(
        {k: v for k, v in arm["method"].items() if k != "name"})
    out = {}
    for k in sorted(set(fb) | set(fm)):
        if k.startswith("identity.") and k.split(".")[1] in ("tah", "reid", "assoc", "dam4sam", "attrbind", "deepsort",
                                                             "ocsort") and k not in fm:
            continue                                     # section dropped (head does not read it)
        if k.startswith("filter.") and k not in fm:
            continue
        if fb.get(k, "<absent>") != fm.get(k, "<absent>"):
            out[k] = (fb.get(k, "<absent>"), fm.get(k, "<absent>"))
    return out


def dump_yaml(obj):
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=120)
