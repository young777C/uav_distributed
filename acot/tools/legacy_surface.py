"""Generate the LEGACY CONFIGURATION SURFACE of the ACoT-UAV code (1004 code-framework; Codex R1 MAJOR-1 / R2 MAJOR-2).

Discovery is glob-based over every in-scope directory (SCOPES); every top-level directory and root-level .py/.sh is
either scanned or listed in EXCLUDES with a reason (an UNCLASSIFIED top-level entry fails the coverage test), so
omissions are auditable. Sections: argparse flags + sys.argv use (every in-scope .py), constructor kwargs / JSON keys,
env-var reads/writes, env→policy wire fields, launcher scripts (python entry points, docker ops, every --flag token,
behaviour env assignments, ${VAR:-…} knobs, positional args) and curated behaviour constants.
The `acot` translator must map every row (acot/tests/test_legacy_surface.py re-runs this generator and fails on an
unmapped or stale row); from P2 on, unknown legacy env / JSON keys raise.
  python3 acot/tools/legacy_surface.py > acot_note/1004_legacy_surface.md     (repo root; stdlib only)

classes: scientific (method/protocol behaviour; hashed) · diagnostic (GT channel → arm is diagnostic) · privileged (GT
perception allowed by the sim-proxy protocol, identical for all arms) · collection (logging / data collection) ·
artefact (identity or location of an input/output artefact or result — managed by the artefact store / scheduler) ·
infra (host placement: devices, ports, hosts, shards, threads; not hashed) · injected (built by code) · offline
(recipe / prep parameter) · datagen (data-generation parameter) · analysis (analysis-script option; replaced by the
acot analysis registry) · test (test tooling) · other-owner (owned by another session; not migrated here)."""
import ast
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = "carla_uav_tracking/rollout"

SCOPES = ["carla_uav_tracking/**/*.py", "train/**/*.py", "acot_probe/**/*.py", "runs/**/*.py", "iter/**/*.py",
          "paper2_analyze.py"]
SH_SCOPES = ["carla_uav_tracking/**/*.sh", "train/**/*.sh", "runs/**/*.sh", "iter/**/*.sh"]
EXCLUDES = [  # (glob, reason) — everything under the repo root that is NOT scanned
    ("runs/pydeps/**", "vendored third-party (timm)"),
    ("external/**", "vendored third-party baselines (DAM4SAM, JointNLT); invoked only via dam4sam_service.py"),
    ("src/**", "other project in this repo (uav-distributed / Paper1Lite)"),
    ("scripts/**", "other project (Paper1Lite)"),
    ("tests/**", "other project (Paper1Lite)"),
    ("configs/**", "other project (Paper1Lite) — new framework YAMLs live under acot/configs/"),
    ("docs/**", "other project (Paper1Lite)"),
    ("acot_draft/**", "paper LaTeX + build scripts (no experiment behaviour)"),
    ("**/*ticker*.sh", "progress display only"),
    ("runs/asd/codex_review.sh", "review tooling"),
    ("**/__pycache__/**", "bytecode"),
    ("acot/**", "the new framework itself (not legacy surface)"),
    ("carla_uav_tracking/acot_client/**", "the new framework's CARLA-side client (not legacy surface)"),
    ("acot_note/**", "notes, Codex reviews, figure / report scripts (read results only; no experiment behaviour)"),
    ("paper_draft/**", "paper template (no code)"),
    ("related-survey/**", "literature notes (no code)"),
    ("result/**", "other project's result data (no code)"),
    (".github/**", "other project's CI workflow (Paper1Lite)"),
    (".humanize/**", "tool state of the humanize plugin"),
    (".git/**", "git metadata"),
    (".pytest_cache/**", "pytest cache"),
]
SKIP_SH = [g for g, _ in EXCLUDES if g.endswith(".sh")]

RUNTIME = {"policy_server", "run_rollout", "env", "policy", "remote_policy", "bridge", "metrics", "candidates",
           "tracklets", "asd", "opening_zoom", "target_state", "road_predictor", "online_vlm", "dam4sam_service"}
OFFLINE_ROLLOUT = {"opening_cls", "opening_ft", "opening_score", "opening_ens"}
TEST_PREFIX = ("smoke_", "parity_", "test_", "sanity_", "toy_")


def category(rel):
    base = os.path.splitext(os.path.basename(rel))[0]
    if rel.startswith("runs/data_v6/"):
        return "other-owner"
    if base.startswith(TEST_PREFIX):
        return "test"
    if rel.startswith(R + "/"):
        if base in RUNTIME:
            return "runtime"
        return "offline" if base in OFFLINE_ROLLOUT else "analysis"
    if rel.startswith(("carla_uav_tracking/scene/",)):
        return "runtime"
    if rel.startswith(("carla_uav_tracking/scripts/", "carla_uav_tracking/recording/", "iter/")) or \
            rel == "runs/tah_v90/b_fill.py":
        return "datagen"
    if rel.startswith(("train/", "acot_probe/")):
        return "offline"
    return "analysis"


SIGNATURES = [  # (file, class, method, which legacy channel feeds it)
    (f"{R}/policy.py", "Policy", "__init__", "Policy kwarg (policy_server flags)"),
    (f"{R}/asd.py", "ASD", "__init__", "ASD_CFG / --asd-cfg"),
    (f"{R}/tracklets.py", "TrackletAssigner", "__init__", "TRK_CFG"),
    (f"{R}/opening_zoom.py", "AttrScorer", "__init__", "OPENING_CFG (OPENING=attr)"),
    (f"{R}/opening_zoom.py", "ZoomVerifier", "__init__", "OPENING_CFG (OPENING=zoom)"),
    (f"{R}/metrics.py", "RolloutScorer", "__init__", "scorer (run_rollout passes fps, max_lost_s only)"),
]

DIAG = {"--gallery-seed", "--warmup-oracle-s", "--event-override", "--retarget-tick", "--expert-control",
        "--road-oracle", "--predict-road", "ORACLE_ABSENT", "gallery_seed", "warmup_oracle_s", "event_override"}
DIAG_VALUE = {"--reanchor": "diagnostic iff value=oracle", "reanchor": "diagnostic iff value=oracle",
              "--tid-head": "diagnostic iff value in {oracle, attrbind}",
              "tid_head": "diagnostic iff value in {oracle, attrbind}",
              "--ex-source": "diagnostic iff value != track", "ex_source": "diagnostic iff value != track",
              "--tracklets": "GT identity keys (diagnostic / deploy_v1 only) iff value=sim",
              "tracklets": "GT identity keys (diagnostic / deploy_v1 only) iff value=sim"}
COLLECT = {"ASD_LOG", "TAH_DAGGER_DIR", "DAGGER_DIR", "OPENING_DUMP", "DUMP_LABELS", "DAM4SAM_DIAG", "log_path"}
INFRA = {"CARLA_PORT", "CARLA_TM_PORT", "CARLA_HOST", "CARLA_ROOT", "HOSTNAME", "HF_CACHE", "NET", "SRV", "GPU", "GPUS",
         "NTHR", "TM_PORT", "STALL_MIN", "FIRST_MIN", "START_PART", "DATA", "REPO", "OMP_NUM_THREADS",
         "MKL_NUM_THREADS", "NVIDIA_DRIVER_CAPABILITIES", "device", "dam4sam_host", "dam4sam_port"}
INFRA_FLAG = re.compile(r"^--(device|port|host|policy-host|policy-port|carla-port|dam4sam-host|dam4sam-port|tm-port|"
                        r"shard|nshard|num-shards|workers|num-workers|gpu|gpus|threads|max-size-gb)$")
ARTEFACT_FLAG = re.compile(r"^--(out|output|out-dir|outdir|save[\w-]*|ckpt[\w-]*|cache[\w-]*|init[\w-]*|resume|"
                           r"load[\w-]*|report|dump[\w-]*|jprefix|qwen-cache|feats[\w-]*|progress-file|summary-jsonl|"
                           r"provenance|log[\w-]*|glob|split[\w-]*|manifest[\w-]*|data[\w-]*|episodes-glob)$")
TEST = {"SMOKE_TOWN", "SMOKE_STEPS", "PARITY_EP", "PARITY_DEVICE"}
INJECTED = {"model", "processor", "ckpt_path", "cfg", "vlm"}

WIRE = [
    ("act.rgb", "S1 camera frame", "deployable"),
    ("act.language", "instruction", "deployable"),
    ("act.proprio", "ego state", "deployable"),
    ("act.step", "tick index", "deployable"),
    ("act.rgb_hi", "k× opening camera at ticks 0/6/12 (--hires)", "deployable (protocol frontend)"),
    ("act.retarget_event", "scripted re-target flag (--retarget-tick)", "diagnostic"),
    ("act.target_color / act.target_bp", "GT target colour / blueprint (always sent today)",
     "diagnostic (attrbind) / collection label; blanked for every other arm from P1"),
    ("act.gt_candidates[0] tpos", "GT target world position → build_candidates (target box), _committed_world "
     "(matched-box localisation), estimator.update (ex_source != track)", "privileged (box / matched-box) · diagnostic (WHERE)"),
    ("act.gt_candidates[1] dstates", "GT distractor states → build_candidates, _committed_world", "privileged"),
    ("act.gt_candidates[2] campose", "ego camera pose → projection, frame_gain", "deployable (ego odometry)"),
    ("act.gt_candidates[3] road_pred_world", "road-graph prediction from the TRUE target (--predict-road / --road-oracle)", "diagnostic"),
    ("act.gt_candidates[4] vis", "ray-cast occlusion flags (t_vis, d_vis) (--vis-raycast)", "privileged (occlusion filter only)"),
    ("cand.idx (tracklets=sim)", "simulator actor index used as identity key (target = −1)", "GT identity → deploy_v1 / diagnostic only"),
    ("cset.is_target / cset.true_idx / cand.gt_idx", "GT labels created inside build_candidates (policy process)", "scoring / collection only"),
    ("reply.cand_lite", "is_target, idx, depth, true_idx → scorer; **gt_idx NOT sent** (known gap: tracker-keyed "
     "distractors are scored by tracker id, metrics.py:29-32)", "scoring"),
    ("reply.pred_slot / action / search_mode", "policy outputs", "deployable"),
    ("reset", "carries NO seed today (logs aligned by order); P1 adds a log-only seed", "—"),
]
CONSTANTS = [
    ("env_fov", f"{R}/env.py", "camera fov_deg=90 (run_rollout does not pass it)", "scientific"),
    ("model_image", "train/config_v5.yaml", "image.fov_deg 90 / width,height 512 / vehicle_size_m 4.0 / box_min_px 10 / box_max_px 140 → policy candidates", "scientific"),
    ("cfg_v5_shared_dirs", "train/config_v5.yaml", "context_cache, stage2_context_cache, ckpt_dir, stage2_ckpt_dir SHARED with config_v5_fov90.yaml", "offline (conflict)"),
    ("cfg_v5_split_missing", "train/config_v5.yaml", "split_manifest → missing file → dataset.py naive slice, test split empty", "offline (conflict)"),
    ("rollout_fps_res", f"{R}/run_rollout.py", "fps=10, resolution=(512,512); scorer gets fps + max_lost_s only (commit_k NOT bound to --commit-k)", "scientific"),
    ("max_lost_18", "carla_uav_tracking/config/default.yaml", "environment.max_lost_seconds 18 (scorer truncation; RolloutScorer default 5)", "scientific"),
    ("scorer_defaults", f"{R}/metrics.py", "RolloutScorer commit_k=5, t_fail 2 s, central_frac 0.3", "scientific"),
    ("opening_hires_ticks", f"{R}/env.py", "opening hi-res frames at steps 0/6/12; hi-res sensor destroyed at step>=12", "scientific"),
    ("warmup_stationary", f"{R}/env.py", "expert warm-up 90 ticks; abort 'target stationary' if max speed < 1.2 m/s; spawn failure → skipped ('setup: …')", "scientific"),
    ("shuffle_seed", f"{R}/policy.py", "shuffle_seed=0 (not exposed): candidate-shuffle RNG re-seeded at every reset()", "scientific"),
    ("tah_lang_reanchor", f"{R}/policy.py", "TAH lang re-anchor threshold 0.60 (not conf_tau)", "scientific"),
    ("dam4sam_cooldown", f"{R}/policy.py", "DAM4SAM re-init cooldown 3 ticks", "scientific"),
    ("crop_dino", f"{R}/policy.py", "crop-DINO vit_small_patch14_dinov2.lvd142m 224 px / pad 1.3; reid gallery momentum 0.9; pos depth/100", "scientific"),
    ("eval_asd_fixed", f"{R}/eval_asd.sh", "fixed: NET=cyh-carla-net, CONFIG=train/config_v5.yaml, --ckpt runs/stage2_v5/stage2_best.pt, --language-mode neutral, --ex-source track, --port 5555, image acot-uav-train:cu118, `pip install timm==1.0.29` at container start", "scientific / infra"),
    ("launch_wd_infra", "runs/asd/iter/launch_wd.sh", "ADAPTER map CARLA port → -graphicsadapter; TM port = tm + 10·(part−1); 5 restarts max; merges *_p<k>.json[.partial]", "infra"),
    ("tah_cache_meta", "train/tah_cache.py", "DINO crop 224/1.3, depth/100, stride 2 (0.2 s); saves only d_feat/order/fov", "offline"),
    ("tah_train_compat", "train/tah_train.py", "compat check = d_feat only; val = slice of the FIRST cache", "offline"),
    ("backbone_kv_skip", "train/backbone_kv.py", "skips any existing <stem>.npz without config/hash check", "offline"),
    ("attr_ckpt_paths", f"{R}/opening_zoom.py", "AttrScorer: lin ckpt default eval_out/opening_attr_cls.pt, ft ckpt HARD-CODED eval_out/opening_attr_ft.pt; DINO scales from ckpt + flip TTA", "scientific"),
    ("tick_align_order", f"{R}/tick_align.py", "1 tick = 0.6 s (assumes S2=6); logs → seeds by ORDER of non-skipped episodes", "analysis"),
    ("opening_cl_partial", f"{R}/opening_cl.py", "sorted(glob('*.json*'))[-1] → prefers .json.partial over .json; _p[0-9] globs miss part ≥10", "analysis"),
    ("iter_gate_precedence", f"{R}/iter_gate.py", "skipped/unscored dropped; first occurrence per seed wins; final > partial per part", "analysis"),
    ("postprocess_box_min", "carla_uav_tracking/recording/postprocess.py", "box_min 8 px (config 10)", "datagen"),
    ("labels_336", "train/labels.py", "image size 336 (stale)", "offline"),
    ("cyh_carla_python", "cyh-carla container", "`python` = 3.8.0 (CARLA client), `python3` = 3.6.9; mounts ONLY carla_uav_tracking → /workspace and carla_data → /data", "infra"),
    ("dam4sam_single_client", "dam4sam-svc container", "single-client sequential accept loop (dam4sam_service.py:55-58); currently started by hand", "infra"),
]


def rel(p):
    return os.path.relpath(p, ROOT)


def excluded(r):
    import fnmatch
    return any(fnmatch.fnmatch(r, g) for g, _ in EXCLUDES)


def files(scopes):
    out = set()
    for g in scopes:
        for p in glob.glob(os.path.join(ROOT, g), recursive=True):
            r = rel(p)
            if os.path.isfile(p) and not excluded(r):
                out.add(r)
    return sorted(out)


def lit(node):
    if node is None:
        return "REQUIRED"
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"


def classify(name, cat="runtime"):
    if name in TEST:
        return "test"
    if name in INFRA or (name.startswith("--") and INFRA_FLAG.match(name)):
        return "infra"
    if cat in ("test", "other-owner", "analysis"):
        return cat
    if name in INJECTED:
        return "injected"
    if name in COLLECT:
        return "collection"
    if name in DIAG:
        return "diagnostic"
    if name in DIAG_VALUE:
        return DIAG_VALUE[name]
    if cat in ("offline", "datagen"):
        return "artefact" if (name.startswith("--") and ARTEFACT_FLAG.match(name)) else cat
    if name == "--out":
        return "artefact"
    return "scientific"


def argparse_rows(pyfiles):
    rows, argv_rows, broken = [], [], []
    for f in pyfiles:
        src = open(os.path.join(ROOT, f), errors="ignore").read()
        cat = category(f)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            broken.append((f, str(e)))
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument":
                names = [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                kw = {k.arg: k.value for k in n.keywords}
                if "default" in kw:
                    d = lit(kw["default"])
                elif "action" in kw and lit(kw["action"]) == "'store_true'":
                    d = "False"
                else:
                    d = "—"
                flag = next((x for x in names if x.startswith("--")), names[0] if names else "?")
                rows.append((f"{f}:{n.lineno}", cat, flag, d, classify(flag, cat)))
        idx = sorted(set(re.findall(r"sys\.argv\[(\d+)", src)))
        if idx:
            argv_rows.append((f, cat, ", ".join(f"argv[{i}]" for i in idx)))
    return rows, argv_rows, broken


def signature_rows():
    rows = []
    for f, cname, meth, src in SIGNATURES:
        for n in ast.walk(ast.parse(open(os.path.join(ROOT, f)).read())):
            if isinstance(n, ast.ClassDef) and n.name == cname:
                for m in n.body:
                    if isinstance(m, ast.FunctionDef) and m.name == meth:
                        pos = m.args.args[1:]
                        pdef = [None] * (len(pos) - len(m.args.defaults)) + list(m.args.defaults)
                        for a, d in list(zip(pos, pdef)) + list(zip(m.args.kwonlyargs, m.args.kw_defaults)):
                            rows.append((f"{f}:{m.lineno}", src, a.arg, lit(d), classify(a.arg), cname))
    for i, line in enumerate(open(os.path.join(ROOT, f"{R}/policy.py")), 1):
        for k, d in re.findall(r'ocfg\.pop\("(\w+)",\s*([^)]+)\)', line):
            rows.append((f"{R}/policy.py:{i}", "OPENING_CFG (popped by Policy)", k, d.strip(), "scientific", "OPENING_CFG"))
    return rows


def env_rows(pyfiles):
    rows, dyn = {}, []
    rd = re.compile(r"""os\.(?:environ\.get|getenv)\(\s*['"](\w+)['"]\s*(?:,\s*([^)]*))?\)|os\.environ\[\s*['"](\w+)['"]\s*\](?!\s*=[^=])""")
    wr = re.compile(r"""os\.environ\[\s*['"](\w+)['"]\s*\]\s*=[^=]|os\.environ\.setdefault\(\s*['"](\w+)['"]""")
    dw = re.compile(r"""os\.environ\[\s*([A-Za-z_]\w*)\s*\]\s*=[^=]""")
    for f in pyfiles:
        cat = category(f)
        for i, line in enumerate(open(os.path.join(ROOT, f), errors="ignore"), 1):
            for m in rd.finditer(line):
                var = m.group(1) or m.group(3)
                rows.setdefault(var, []).append((f"{f}:{i}", "read", (m.group(2) or "").strip() or "—", cat))
            for m in wr.finditer(line):
                var = m.group(1) or m.group(2)
                rows.setdefault(var, []).append((f"{f}:{i}", "WRITE", "—", cat))
            for m in dw.finditer(line):
                dyn.append((f"{f}:{i}", m.group(1), cat))
    return rows, dyn


def launcher_rows(shfiles):
    rows = []
    for f in shfiles:
        txt = open(os.path.join(ROOT, f), errors="ignore").read()
        me = os.path.basename(f)
        entry = sorted(set(m for m in re.findall(r"(?:python[0-9.]*|bash|sh)\s+(?:-u\s+)?([\w./$-]+\.(?:py|sh))", txt)
                           if os.path.basename(m) != me))
        mods = sorted(set(re.findall(r"python[0-9.]*\s+-m\s+([\w.]+)", txt)))
        docker = [d for d in ("docker run", "docker exec", "docker rm", "docker start", "docker compose", "docker network")
                  if d in txt]
        flags = sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", txt)))
        envs = sorted(set(re.findall(r"\b(ASD_CFG|TRK_CFG|OPENING_CFG|OPENING|ORACLE_ABSENT|TAH_DAGGER_DIR|DAGGER_DIR|"
                                     r"COLD_WINDOW|COLD_LAMBDA|COLD_SINGLE|V1_SELFRESET|DAM4SAM_EVERY_FRAME|DAM4SAM_DIAG|"
                                     r"OPENING_DUMP|DUMP_LABELS|HIRES|VIS_RAYCAST|S2_PERIOD|SEED_BASE|EPISODES|STEPS|TOWN|"
                                     r"HEAD_ARGS|CONFIG)=(?![\"']?\$\{\1\b)", txt)))
        knobs = sorted(set(re.findall(r"\$\{([A-Za-z_]\w*)(?::?[-?+=])", txt)))
        pos = sorted(set(re.findall(r"\$\{?([1-9])\b", txt)))
        rows.append((f, ", ".join(entry + [f"-m {m}" for m in mods]) or "—", ", ".join(docker) or "—",
                     " ".join(flags) or "—", " ".join(envs) or "—", " ".join(knobs) or "—",
                     " ".join(f"${x}" for x in pos) or "—"))
    return rows


def top_level():
    """every top-level directory and root-level .py/.sh → ('scanned' | 'excluded: <reason>' | 'UNCLASSIFIED')."""
    scope_roots = {g.split("/")[0] for g in SCOPES + SH_SCOPES}
    excl_roots = {g.split("/")[0]: r for g, r in EXCLUDES if g.endswith("/**") and g.count("/") == 1}
    out = []
    for e in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, e)
        if not (os.path.isdir(p) or e.endswith((".py", ".sh"))):
            continue
        if e in scope_roots:
            partial = [g for g, _ in EXCLUDES if g.startswith(e + "/")]
            status = "scanned" + (f" (minus {', '.join(partial)})" if partial else "")
        elif e in excl_roots:
            status = f"excluded: {excl_roots[e]}"
        else:
            status = "UNCLASSIFIED"
        out.append((e + ("/" if os.path.isdir(p) else ""), status))
    return out


def collect():
    """machine-readable surface with STABLE keys (no line numbers) — used by acot/tests/test_legacy_surface.py."""
    pyf, shf = files(SCOPES), [f for f in files(SH_SCOPES) if not excluded(f)]
    a, argv_rows, broken = argparse_rows(pyf)
    e, dyn = env_rows(pyf)
    return {
        "top_level": top_level(), "broken": broken,
        "argparse": [{"key": f"{loc.rsplit(':', 1)[0]}:{flag}", "file": loc.rsplit(':', 1)[0], "category": cat,
                      "flag": flag, "default": d, "class": c} for loc, cat, flag, d, c in a],
        "argv": [{"key": f, "category": cat, "index": idx} for f, cat, idx in argv_rows],
        "signatures": [{"key": f"{cls}:{k}", "source": src, "field": k, "default": d, "class": c}
                       for loc, src, k, d, c, cls in signature_rows()],
        "env": {var: sorted({classify(var, c) for _, _, _, c in rows}) for var, rows in e.items()},
        "env_dynamic": [{"key": loc.rsplit(":", 1)[0], "var": key, "category": cat} for loc, key, cat in dyn],
        "wire": [{"key": k, "class": c} for k, _, c in WIRE],
        "launchers": [{"key": r[0]} for r in launcher_rows(shf)],
        "constants": [{"key": cid, "file": f, "class": c} for cid, f, _, c in CONSTANTS],
    }


def main():
    w = sys.stdout.write
    pyf, shf = files(SCOPES), [f for f in files(SH_SCOPES) if not excluded(f)]
    tl = top_level()
    a, argv_rows, broken = argparse_rows(pyf)
    s = signature_rows()
    e, dyn = env_rows(pyf)
    L = launcher_rows(shf)
    w("# 1004 · 旧配置面全清单(自动生成，勿手改)\n\n")
    w("> 生成命令:`python3 acot/tools/legacy_surface.py > acot_note/1004_legacy_surface.md`(只依赖标准库)。"
      "`acot/tests/test_legacy_surface.py` 会重新运行生成器，并与 `acot/config/legacy_map.yaml` 双向核对:"
      "清单中每一行都必须有映射(映射到新字段，或明确标为 infra / collection / artefact / analysis / test / other-owner / rejected),"
      "映射表中也不能有清单里已不存在的条目。P2 起，未知的旧环境变量或 JSON 字段直接报错。\n>\n"
      "> 分类：**scientific** 方法或协议行为，计入哈希;**diagnostic** 真值通道，所在臂标为 diagnostic;"
      "**privileged** 仿真代理协议允许的理想化感知，对所有臂相同;**collection** 只做记录或采集;"
      "**artefact** 输入、输出产物或结果的身份和位置，由产物库和调度器管理;**infra** 主机侧资源(设备、端口、主机、分片、线程),不计入哈希;"
      "**injected** 由代码构造;**offline** 训练配方或缓存参数;**datagen** 数据生成参数;**analysis** 分析脚本选项，由新的分析注册表取代;"
      "**test** 测试工具;**other-owner** 归另一会话所有，本框架不迁移。\n\n")
    w(f"## 0. 扫描范围\n\n- Python:{len(pyf)} 个文件(`" + "`、`".join(SCOPES) + "`)\n"
      f"- Shell:{len(shf)} 个文件(`" + "`、`".join(SH_SCOPES) + "`)\n\n| 排除 | 原因 |\n|---|---|\n")
    for g, why in EXCLUDES:
        w(f"| `{g}` | {why} |\n")
    w("\n| 顶层项 | 状态 |\n|---|---|\n" + "".join(f"| `{e}` | {st} |\n" for e, st in tl))
    unc = [e for e, st in tl if st == "UNCLASSIFIED"]
    w(f"\n未归类的顶层项:{', '.join(unc) if unc else '无'}(非空时 test_legacy_surface 失败)\n")
    if broken:
        w("\n**无法解析的文件**(需人工检查):" + "; ".join(f"`{f}` ({m})" for f, m in broken) + "\n")
    cats = {}
    for f in pyf:
        cats.setdefault(category(f), []).append(os.path.basename(f))
    w("\n| 文件类别 | 文件数 |\n|---|---|\n" + "".join(f"| {c} | {len(v)} |\n" for c, v in sorted(cats.items())))
    w(f"\n## 1. argparse 参数({len(a)} 个)与 sys.argv 位置参数({len(argv_rows)} 个文件)\n\n"
      "| 位置 | 文件类别 | 参数 | 默认值 | 分类 |\n|---|---|---|---|---|\n")
    for loc, cat, flag, d, c in a:
        w(f"| `{loc}` | {cat} | `{flag}` | `{d}` | {c} |\n")
    w("\n| 文件 | 文件类别 | sys.argv 下标 |\n|---|---|---|\n")
    for f, cat, idx in argv_rows:
        w(f"| `{f}` | {cat} | {idx} |\n")
    w(f"\n## 2. 构造函数参数与 JSON 配置字段({len(s)} 个)\n\n"
      "Policy 的 kwarg 由 policy_server 的同名 flag 传入;ASD、TrackletAssigner、AttrScorer、ZoomVerifier 的 kwarg 由对应的 JSON 字符串展开传入"
      "(`**json.loads(...)`),未知字段在运行时才报 TypeError;OPENING_CFG 先被 Policy pop 掉 6 个字段。\n\n"
      "| 位置 | 来源通道 | 字段 | 默认值 | 分类 |\n|---|---|---|---|---|\n")
    for loc, src, k, d, c, _ in s:
        w(f"| `{loc}` | {src} | `{k}` | `{d}` | {c} |\n")
    w(f"\n## 3. 环境变量读写({len(e)} 个变量 + {len(dyn)} 处动态键写入)\n\n| 变量 | 位置(读/写) | 默认值 | 分类 |\n|---|---|---|---|\n")
    for var in sorted(e):
        locs = "<br>".join(f"`{l}` {rw}" for l, rw, _, _ in e[var])
        d = next((x for _, rw, x, _ in e[var] if rw == "read"), "—")
        cs = sorted({classify(var, c) for _, _, _, c in e[var]})
        w(f"| `{var}` | {locs} | `{d}` | {' / '.join(cs)} |\n")
    for loc, key, cat in dyn:
        w(f"| `<动态键 {key}>` | `{loc}` WRITE | `—` | {cat} |\n")
    w(f"\n## 4. env → policy 线协议字段({len(WIRE)} 项，人工整理，依据 `remote_policy.py`、`policy_server.py`、`env.py`、`policy.py`、`candidates.py`)\n\n"
      "| 字段 | 内容 | 分类 |\n|---|---|---|\n")
    for k, what, c in WIRE:
        w(f"| `{k}` | {what} | {c} |\n")
    w(f"\n## 5. 启动脚本({len(L)} 个)\n\n"
      "静态提取，作为交叉核对;权威的调用参数由 P0b 的 docker shim 实际捕获(见框架文档 §9 L0)。每个脚本列出：调用的入口、docker 操作、"
      "出现的全部 `--flag`、写死的行为环境变量赋值(不含 `VAR=${VAR:-默认}` 这种自身默认)、`${VAR:-…}` 可调参数、位置参数。\n\n"
      "| 脚本 | 入口 | docker | 出现的 flag | 写死的行为环境变量 | 可调参数 | 位置参数 |\n|---|---|---|---|---|---|---|\n")
    for row in L:
        w("| `" + row[0] + "` | " + " | ".join(r.replace("|", "\\|") for r in row[1:]) + " |\n")
    w(f"\n## 6. 行为常量与已知冲突({len(CONSTANTS)} 项，人工整理;实现时复核行号)\n\n| id · 位置 | 内容 | 分类 |\n|---|---|---|\n")
    for cid, f, what, c in CONSTANTS:
        w(f"| `{cid}` · `{f}` | {what} | {c} |\n")


if __name__ == "__main__":
    main()
