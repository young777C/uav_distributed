"""Strict schemas of the acot configuration objects (framework §5). A schema is a nested dict whose leaves are `T` type
specs (or `Opt(...)` for keys that may be absent); validate() rejects unknown keys, missing keys, wrong types and values
outside `choices`. Ownership (§5.1): the PROTOCOL holds the invariants shared by every arm, the METHOD every per-arm
policy knob (control / identity / filter / diagnostics)."""


class SchemaError(ValueError):
    pass


class T:
    def __init__(self, kind, choices=None, null=False, item=None):
        self.kind, self.choices, self.null, self.item = kind, choices, null, item

    def check(self, v, path):
        if v is None:
            if not self.null:
                raise SchemaError(f"{path}: must not be null")
            return
        k = self.kind
        if k is float:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        elif k is int:
            ok = isinstance(v, int) and not isinstance(v, bool)
        else:
            ok = isinstance(v, k)
        if not ok:
            raise SchemaError(f"{path}: expected {k.__name__}, got {type(v).__name__} ({v!r})")
        if self.choices is not None and v not in self.choices:
            raise SchemaError(f"{path}: {v!r} not in {list(self.choices)}")
        if self.item is not None:
            for i, x in enumerate(v):
                check_value(x, self.item, f"{path}[{i}]")


class Opt:
    """key may be absent (the wrapped spec applies when present)."""
    def __init__(self, spec):
        self.spec = spec


def enum(*choices, null=False):
    return T(str, choices=choices, null=null)


def opt(t):
    return T(t.kind, t.choices, True, t.item)


def listof(item):
    return T(list, item=item)


FLOAT, INT, STR, BOOL, DICT = T(float), T(int), T(str), T(bool), T(dict)


def check_value(v, spec, path):
    if isinstance(spec, Opt):
        spec = spec.spec
    if isinstance(spec, dict):
        validate(v, spec, path)
    else:
        spec.check(v, path)


def validate(obj, schema, path="$"):
    if not isinstance(obj, dict):
        raise SchemaError(f"{path}: expected a mapping, got {type(obj).__name__}")
    for k in obj:
        if k not in schema:
            raise SchemaError(f"{path}.{k}: unknown key (allowed: {sorted(schema)})")
    for k, spec in schema.items():
        if k not in obj:
            if isinstance(spec, Opt):
                continue
            raise SchemaError(f"{path}.{k}: missing")
        check_value(obj[k], spec, f"{path}.{k}")
    return obj


# ---------------------------------------------------------------------------------------------------------------- protocol
PRIVILEGED = ("cand_box_gt_projection", "cand_depth_gt", "occlusion_raycast_gt", "task_vehicle_set",
              "where_world_gt_matched_box", "identity_keys_sim_actor_ids", "where_world_gt_committed_id",
              "shared_opening_actor")
PROTOCOL = {
    "name": STR, "version": INT, "extends": Opt(STR), "aliases": Opt(listof(STR)), "description": Opt(STR),
    "camera": {"fov_deg": FLOAT, "width": INT, "height": INT, "fps": INT},
    "sim": {"s2_period": INT, "episode_steps": INT, "scenario": STR, "max_lost_s": FLOAT,   # town comes from the seed set
            "warmup": {"expert_ticks": INT, "min_target_speed_mps": FLOAT}, "retarget_tick": opt(INT),
            # policy sampling RNG (1006 §3.5): unseeded (absent; legacy) | per_step = torch reseeded before every
            # EAR / DiT sample from H(scene seed, repeat, stream, step) → common random numbers across arms
            "policy_rng": Opt(enum("unseeded", "per_step")),
            # 1006 CARLA determinism (t0_determinism: identical runs differ already at step 0): sync_setup = the world
            # is switched to synchronous mode (fixed dt) BEFORE the scene is spawned, so no physics runs on wall-clock
            # time during setup; reload_world = client.reload_world() before every seed (no physics / TM state carried
            # over from the previous episode). Absent = false (legacy behaviour).
            "env_determinism": Opt({"sync_setup": BOOL, "reload_world": BOOL})},
    "perception": {
        "candidates": {"source": enum("sim_projection"), "set": enum("task_vehicles"), "vehicle_size_m": FLOAT,
                       "box_min_px": FLOAT, "box_max_px": FLOAT},
        "occlusion": enum("none", "raycast"),
        "identity_keys": enum("tracker", "sim"),
        "tracker": Opt({"max_age": INT, "gate_px": FLOAT, "gate_box": FLOAT, "w_app": FLOAT, "min_cos": FLOAT,
                        "vel_beta": FLOAT, "app_alpha": FLOAT,
                        "assign": Opt(enum("greedy", "hungarian"))}),   # absent = greedy (every deploy-v2 run so far)
        "where_localisation": enum("gt_matched_box", "gt_committed_id"),
        "noise": enum("none")},
    "frontend": {
        "vlm": {"model_config": STR, "stage2_ckpt": STR, "language_mode": enum("neutral", "none", "full")},
        "crop_features": {"id": STR, "depth_scale": FLOAT},
        "opening": {"module": enum("head", "zoom", "attr"), "hires_k": INT, "window": INT,
                    "agg": enum("sum", "mean", "last"), "final_reset": BOOL, "reset_asd": BOOL, "force_final": BOOL,
                    "seed_asd": FLOAT,
                    # identity-head hand-over during the opening window (1006 F-c): legacy (absent) = only the TAH
                    # memory is reset at the non-final opening ticks, assoc / dam4sam are initialised once at the first
                    # tick; uniform = EVERY head is reset and re-seeded from the current accumulated opening pick at
                    # every non-final opening tick (TAH behaviour unchanged)
                    "handoff": Opt(enum("legacy", "uniform")),
                    # 1006 §3.4 shared opening (mechanism tables): reference actor per opening tick, from a reference
                    # run (acot/analysis/shared_opening.py); privileged input `shared_opening_actor`
                    "shared": Opt({"file": STR}),
                    "attr": Opt({"ckpt_lin": STR, "ckpt_ft": STR, "wm": FLOAT, "wc": FLOAT, "wq": FLOAT, "ft": FLOAT}),
                    "zoom": Opt({"scale": FLOAT, "R": INT, "spotlight": BOOL})}},
    "scorer": {"commit_k": INT, "t_fail_s": FLOAT, "central_frac": FLOAT, "primary": enum("committed", "lenient", "strict")},
    "privileged_perception": listof(enum(*PRIVILEGED)),
}

# ------------------------------------------------------------------------------------------------------------------ method
HEADS = ("xattn", "attrbind", "reid", "oracle", "assoc", "dam4sam", "tah", "fusion", "sticky", "oavat", "deepsort",
         "ocsort")
# head → identity sub-sections it reads (the parameter-usage matrix; inactive sections are dropped at resolve time)
HEAD_SECTIONS = {"xattn": (), "attrbind": ("attrbind",), "reid": ("reid",), "oracle": (), "assoc": ("assoc",),
                 "dam4sam": ("dam4sam",), "tah": ("tah",), "fusion": ("tah", "dam4sam"), "sticky": (),
                 "oavat": ("oavat",), "deepsort": ("deepsort",), "ocsort": ("ocsort",)}
FILTER_FIELDS = {"filter_kind": enum("bayes", "resid", "tempered", null=True), "resid_decay": FLOAT,
                 "absent_hold": BOOL, "absent_max": INT, "conf_window": INT, "gate_memory": BOOL, "gate_tah": BOOL,
                 "gate_dam": BOOL, "naive": BOOL, "alpha": opt(DICT), "q_lo": FLOAT, "q_hi": FLOAT, "eps_switch": FLOAT,
                 "anchor_mode": enum("first", "spec"), "k_anchor": INT, "beta_lang": FLOAT, "lang_ema": FLOAT, "tau": FLOAT,
                 "kappa": FLOAT, "h": FLOAT, "refractory": INT, "learned_ckpt": opt(STR), "v1_selfreset": BOOL,
                 # 1006 §5.2 exploratory switches (acot runtime only; absent = true = current behaviour):
                 # gate_commit=false → PGIF still gates / undoes the memory write, but the pick and the commitment
                 #   follow the memory's own pick; force_conf=false → a PGIF re-pick is not committed directly with
                 #   c = 1: c = the PGIF evidence top1 − top2 margin over the visible keys, then the normal commitment rule
                 "gate_commit": BOOL, "force_conf": BOOL,
                 # 1007 opt (acot runtime only; absent = off): evidence inheritance across tracker-id changes
                 "inherit": BOOL, "inherit_gate_px": FLOAT, "inherit_px_per_tick": FLOAT}
DIAGNOSTICS = ("gallery_seed_gt", "reanchor_oracle", "warmup_oracle", "event_override", "oracle_absent_abstain",
               "oracle_absent_hygiene", "oracle_head", "attrbind_gt_attrs", "ex_source_gt", "road_oracle",
               "predict_road", "expert_control", "retarget")
METHOD = {
    "name": STR, "extends": Opt(STR), "description": Opt(STR),
    "control": {"ex_source": enum("ear", "cv", "cv_gated", "zerovel", "zerovel_gated", "road", "track"),
                "frame_gain": FLOAT, "commit_k": INT, "ablate": enum("ear", "iar", null=True), "shuffle_seed": INT,
                "expert_control": BOOL, "predict_road": BOOL, "road_oracle": BOOL,
                # WHERE when the committed key has no box this tick: ear = keep the EAR sample (absent = ear, legacy);
                # cv = constant-velocity extrapolation of the committed target's last confident position
                "absent_where": Opt(enum("ear", "cv"))},
    "identity": {
        "head": enum(*HEADS),
        "memory_seed": enum("committed", "gt"),
        "conf_tau": FLOAT, "conf_consensus": FLOAT, "warmup_oracle_s": FLOAT, "event_override": BOOL,
        "reanchor": {"mode": enum("off", "lang", "oracle", "anchor"), "patience": INT, "tavg": FLOAT},
        "cold": {"window": INT, "lambda": FLOAT, "single": BOOL},
        # feature (1006 backbone control): dinov2_crop (absent; the shared crop-DINOv2 feature) | dinov3_crop (the same
        # crops through DINOv3 ViT-B/16, CLS; the OA-VAT checkpoint) — TAHRelM's parameters do not depend on the dim
        "tah": Opt({"ckpt": opt(STR), "lang_reanchor_thr": FLOAT, "feature": Opt(enum("dinov2_crop", "dinov3_crop")),
                     # 1007 opt: what the TAHRelM memory learns each tick — its own pick (default, legacy) or the
                     # COMMITTED identity's box (as the re-ID gallery; own pick only while cold / uncommitted)
                     "memory_write": Opt(enum("own", "committed"))}),
        "reid": Opt({"feature": enum("vlmpool", "dinov2"), "bank": INT, "topm": INT, "tavg": FLOAT, "motion": FLOAT,
                     "motion_gate_px": FLOAT}),
        "assoc": Opt({"mode": enum("motion", "deepsort"), "gate_px": FLOAT, "lambda": FLOAT}),
        "dam4sam": Opt({"model": enum("sam21pp-L", "sam21pp-B", "sam21pp-S", "sam21pp-T"), "gate_px": FLOAT,
                        "every_frame": BOOL, "reinit_cooldown": INT, "service": STR}),
        "attrbind": Opt({"ckpt": opt(STR)}),
        # 1006 §5.6 OA-VAT identity adaptation (acot/runtime/head_oavat.py); frame-unit repo defaults → S2 ticks
        "oavat": Opt({"model": STR, "checkpoint": STR, "match_thresh": FLOAT, "reacq_thresh": FLOAT, "iou_pred": FLOAT,
                      "fullframe_after": INT, "max_predict_ticks": INT, "enhance_every": INT, "enhance_alpha": FLOAT,
                      "history": INT}),
        # 1016 DeepSORT / OC-SORT identity adaptations (acot/runtime/head_mot.py; official trackers; every_frame: one
        # step per camera frame, else per S2 tick)
        "deepsort": Opt({"every_frame": BOOL, "max_cosine_distance": FLOAT, "nn_budget": opt(INT), "max_iou_distance": FLOAT, "max_age": INT,
                         "n_init": INT}),
        "ocsort": Opt({"every_frame": BOOL, "det_thresh": FLOAT, "max_age": INT, "min_hits": INT, "iou_threshold": FLOAT, "delta_t": INT,
                       "asso_func": enum("iou", "giou", "ciou", "diou", "ct_dist"), "inertia": FLOAT,
                       "use_byte": BOOL})},
    "filter": dict({"mode": enum("off", "observe", "rule", "learned", "filter")},
                   **{k: Opt(v) for k, v in FILTER_FIELDS.items()}),
    "oracle_absent": enum("off", "abstain", "hygiene"),
    "diagnostics": listof(enum(*DIAGNOSTICS)),
    "accept_provenance_gaps": Opt(DICT),
}

# -------------------------------------------------------------------------------------------------------------- experiment
COLLECTORS = ("ticks", "trk", "gt_access", "scenario", "tah_dagger", "dagger", "opening_snapshots", "actor_labels",
              "dam4sam_diag", "record_act")
EXPERIMENT = {
    "name": STR, "description": Opt(STR), "protocol": STR, "prereg_ref": Opt(STR),
    "seeds": {"set": STR, "subset": Opt(listof(listof(INT))), "expect": Opt(INT)},
    "completeness": Opt({"require": enum("all_complete", "allow_skipped"), "allow_skipped_reasons": listof(STR),
                         "pairing": enum("strict", "intersection")}),
    "retry": Opt({"infra_attempts": INT}),
    # 1006 §3.6: randomised dynamic scheduling — (arm, block) units shuffled with `seed` into one shared queue; an
    # infra-failed (seed, arm) is retried on a DIFFERENT worker, up to `max_infra_attempts` attempts in total
    "schedule": Opt({"block": INT, "seed": INT, "max_infra_attempts": INT}),
    "arms": DICT,
    "collect": Opt(listof(enum(*COLLECTORS))),
    "comparisons": Opt(listof(DICT)),
    "infra": Opt(DICT),
}
ARM = {"method": STR, "ablation": Opt(enum("opening", "frontend", "perception")), "override": Opt(DICT),
       "role": Opt(enum("main", "ablation", "baseline", "diagnostic", "legacy")), "description": Opt(STR),
       "repeat": Opt(INT)}          # 1006 §3.5: independent repeat id (policy RNG salt); absent = 0

# ------------------------------------------------------------------------------------------------------------------- seeds
SEED_BLOCK = {"town": STR, "ranges": Opt(listof(listof(INT))), "list": Opt(listof(INT)), "list_file": Opt(STR),
              "exclude": Opt(listof(INT))}
SEEDSET = {"purpose": enum("train_data", "train", "dev", "heldout", "test"), "blocks": listof(DICT),
           "locked": Opt(BOOL), "description": Opt(STR)}

# ------------------------------------------------------------------------------------------------------------------- infra
INFRA = {
    "docker": {"network": STR, "carla_container": STR, "policy_image": STR, "policy_container_prefix": STR,
               "carla_mounts": DICT, "shm_size": STR, "policy_image_acot": Opt(STR)},
    "runtime": Opt(enum("legacy", "acot")),
    "python": {"carla_client": STR, "carla_client_version": STR, "policy": STR, "policy_version": STR},
    "results_root": STR,
    "carla_servers": listof(DICT),
    "tm_ports": {"base": INT, "per_worker": INT, "per_restart": INT},
    "policy_gpus": listof(INT),
    "reserved": {"carla_ports": listof(listof(INT)), "gpus": listof(INT), "containers": listof(STR)},
    "services": DICT,
    "watchdog": {"first_min": FLOAT, "stall_min": FLOAT, "max_restarts": INT},
    "paths": {"repo": STR, "data": STR, "hf_cache": STR},
    "threads": {"omp": INT},
    "policy_port": INT,
    "pip_preinstall": opt(STR),
    "record_act_ticks": Opt(INT),
}
