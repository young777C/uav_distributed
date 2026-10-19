"""What the acot runtime implements (framework §14 P2) — a light module (no torch) so resolve / preflight can refuse an
arm on the acot runtime before any container starts. Everything else runs on the legacy runtime (infra.runtime)."""
HEADS = ("tah", "assoc", "dam4sam", "xattn", "reid", "oracle", "oavat", "deepsort", "ocsort", "sticky")
COLLECTORS = ("ticks", "trk", "gt_access", "scenario", "record_act", "tah_dagger")
CROP_FEATURES = {"dino_s14_lvd142m_cls384_c224_p1.3": "vit_small_patch14_dinov2.lvd142m"}


def perception(protocol):
    pc, out = protocol["perception"], []
    want = {"candidates.source": (pc["candidates"]["source"], "sim_projection"),
            "identity_keys": (pc["identity_keys"], "tracker"),
            "where_localisation": (pc["where_localisation"], "gt_matched_box"), "noise": (pc["noise"], "none")}
    out += [f"perception.{k}={v!r} (acot runtime implements {w!r})" for k, (v, w) in want.items() if v != w]
    if pc["occlusion"] not in ("raycast", "none"):
        out.append(f"perception.occlusion={pc['occlusion']!r}")
    return out


def frontend(protocol):
    fr, out = protocol["frontend"], []
    if fr["crop_features"]["id"] not in CROP_FEATURES:
        out.append(f"frontend.crop_features.id={fr['crop_features']['id']!r}")
    if float(fr["crop_features"]["depth_scale"]) != 100.0:
        out.append("frontend.crop_features.depth_scale≠100 (TAH position input is depth/100)")
    if fr["opening"]["module"] not in ("attr", "head"):
        out.append(f"frontend.opening.module={fr['opening']['module']!r} (acot runtime implements attr | head)")
    return out


def method(protocol, method):
    C, I, F = method["control"], method["identity"], method["filter"]
    want = {"control.ex_source": (C["ex_source"], "track"),
            "identity.reanchor.mode": (I["reanchor"]["mode"], "off"), "identity.cold.window": (I["cold"]["window"], 0),
            "identity.event_override": (I["event_override"], False),
            "identity.warmup_oracle_s": (float(I["warmup_oracle_s"]), 0.0),
            "identity.conf_consensus": (float(I["conf_consensus"]), 0.0),
            "control.expert_control": (C["expert_control"], False), "control.predict_road": (C["predict_road"], False),
            "control.road_oracle": (C["road_oracle"], False), "filter.v1_selfreset": (F.get("v1_selfreset", False), False)}
    out = [f"{k}={v!r} (acot runtime implements {w!r})" for k, (v, w) in want.items() if v != w]
    if I["memory_seed"] != "committed" and not (I["memory_seed"] == "gt" and I["head"] == "tah"):
        out.append(f"identity.memory_seed={I['memory_seed']!r} (acot runtime: committed, or gt for head tah = U2)")
    if I["head"] not in HEADS:
        out.append(f"identity.head={I['head']!r} (acot runtime implements {HEADS})")
    if F["mode"] not in ("off", "filter"):
        out.append(f"filter.mode={F['mode']!r} (acot runtime implements off | filter)")
    if method["oracle_absent"] != "off" and F["mode"] != "filter":
        out.append("oracle_absent needs filter.mode=filter (legacy: evaluated inside the ASD step)")
    if protocol["sim"].get("retarget_tick") is not None:
        out.append("sim.retarget_tick (dynamic retarget) is not implemented")
    return out


def unsupported(protocol, method_, collect=()):
    """→ list of reasons the acot runtime cannot run this arm (empty = supported)."""
    out = perception(protocol) + frontend(protocol) + method(protocol, method_)
    extra = sorted(set(collect) - set(COLLECTORS))
    if extra:
        out.append(f"collectors {extra} (acot runtime implements {COLLECTORS})")
    return out
