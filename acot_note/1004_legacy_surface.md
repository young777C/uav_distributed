# 1004 · 旧配置面全清单(自动生成，勿手改)

> 生成命令:`python3 acot/tools/legacy_surface.py > acot_note/1004_legacy_surface.md`(只依赖标准库)。`acot/tests/test_legacy_surface.py` 会重新运行生成器，并与 `acot/config/legacy_map.yaml` 双向核对:清单中每一行都必须有映射(映射到新字段，或明确标为 infra / collection / artefact / analysis / test / other-owner / rejected),映射表中也不能有清单里已不存在的条目。P2 起，未知的旧环境变量或 JSON 字段直接报错。
>
> 分类：**scientific** 方法或协议行为，计入哈希;**diagnostic** 真值通道，所在臂标为 diagnostic;**privileged** 仿真代理协议允许的理想化感知，对所有臂相同;**collection** 只做记录或采集;**artefact** 输入、输出产物或结果的身份和位置，由产物库和调度器管理;**infra** 主机侧资源(设备、端口、主机、分片、线程),不计入哈希;**injected** 由代码构造;**offline** 训练配方或缓存参数;**datagen** 数据生成参数;**analysis** 分析脚本选项，由新的分析注册表取代;**test** 测试工具;**other-owner** 归另一会话所有，本框架不迁移。

## 0. 扫描范围

- Python:177 个文件(`carla_uav_tracking/**/*.py`、`train/**/*.py`、`acot_probe/**/*.py`、`runs/**/*.py`、`iter/**/*.py`、`paper2_analyze.py`)
- Shell:100 个文件(`carla_uav_tracking/**/*.sh`、`train/**/*.sh`、`runs/**/*.sh`、`iter/**/*.sh`)

| 排除 | 原因 |
|---|---|
| `runs/pydeps/**` | vendored third-party (timm) |
| `external/**` | vendored third-party baselines (DAM4SAM, JointNLT); invoked only via dam4sam_service.py |
| `src/**` | other project in this repo (uav-distributed / Paper1Lite) |
| `scripts/**` | other project (Paper1Lite) |
| `tests/**` | other project (Paper1Lite) |
| `configs/**` | other project (Paper1Lite) — new framework YAMLs live under acot/configs/ |
| `docs/**` | other project (Paper1Lite) |
| `acot_draft/**` | paper LaTeX + build scripts (no experiment behaviour) |
| `**/*ticker*.sh` | progress display only |
| `runs/asd/codex_review.sh` | review tooling |
| `**/__pycache__/**` | bytecode |
| `acot/**` | the new framework itself (not legacy surface) |
| `carla_uav_tracking/acot_client/**` | the new framework's CARLA-side client (not legacy surface) |
| `acot_note/**` | notes, Codex reviews, figure / report scripts (read results only; no experiment behaviour) |
| `paper_draft/**` | paper template (no code) |
| `related-survey/**` | literature notes (no code) |
| `result/**` | other project's result data (no code) |
| `.github/**` | other project's CI workflow (Paper1Lite) |
| `.humanize/**` | tool state of the humanize plugin |
| `.git/**` | git metadata |
| `.pytest_cache/**` | pytest cache |

| 顶层项 | 状态 |
|---|---|
| `.git/` | excluded: git metadata |
| `.github/` | excluded: other project's CI workflow (Paper1Lite) |
| `.humanize/` | excluded: tool state of the humanize plugin |
| `acot/` | excluded: the new framework itself (not legacy surface) |
| `acot_draft/` | excluded: paper LaTeX + build scripts (no experiment behaviour) |
| `acot_note/` | excluded: notes, Codex reviews, figure / report scripts (read results only; no experiment behaviour) |
| `acot_probe/` | scanned |
| `carla_uav_tracking/` | scanned (minus carla_uav_tracking/acot_client/**) |
| `configs/` | excluded: other project (Paper1Lite) — new framework YAMLs live under acot/configs/ |
| `docs/` | excluded: other project (Paper1Lite) |
| `external/` | excluded: vendored third-party baselines (DAM4SAM, JointNLT); invoked only via dam4sam_service.py |
| `iter/` | scanned |
| `paper2_analyze.py` | scanned |
| `paper_draft/` | excluded: paper template (no code) |
| `related-survey/` | excluded: literature notes (no code) |
| `result/` | excluded: other project's result data (no code) |
| `runs/` | scanned (minus runs/pydeps/**, runs/asd/codex_review.sh) |
| `scripts/` | excluded: other project (Paper1Lite) |
| `src/` | excluded: other project in this repo (uav-distributed / Paper1Lite) |
| `tests/` | excluded: other project (Paper1Lite) |
| `train/` | scanned |

未归类的顶层项:无(非空时 test_legacy_surface 失败)

| 文件类别 | 文件数 |
|---|---|
| analysis | 68 |
| datagen | 23 |
| offline | 50 |
| other-owner | 3 |
| runtime | 20 |
| test | 13 |

## 1. argparse 参数(394 个)与 sys.argv 位置参数(14 个文件)

| 位置 | 文件类别 | 参数 | 默认值 | 分类 |
|---|---|---|---|---|
| `acot_probe/run_probe.py:40` | offline | `--config` | `—` | offline |
| `acot_probe/run_probe.py:41` | offline | `--only` | `None` | offline |
| `acot_probe/run_probe.py:42` | offline | `--skip-cache` | `False` | offline |
| `acot_probe/visualize.py:57` | offline | `--config` | `—` | offline |
| `acot_probe/visualize.py:58` | offline | `--episode` | `—` | offline |
| `acot_probe/visualize.py:59` | offline | `--frames` | `8` | offline |
| `acot_probe/visualize.py:60` | offline | `--out` | `'runs/proj_check.png'` | artefact |
| `carla_uav_tracking/rollout/asd_stats.py:43` | analysis | `pairs` | `—` | analysis |
| `carla_uav_tracking/rollout/asd_stats.py:43` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/compare_runs.py:39` | analysis | `run_a` | `—` | analysis |
| `carla_uav_tracking/rollout/compare_runs.py:39` | analysis | `run_b` | `—` | analysis |
| `carla_uav_tracking/rollout/compare_runs.py:40` | analysis | `--label-a` | `'A'` | analysis |
| `carla_uav_tracking/rollout/compare_runs.py:40` | analysis | `--label-b` | `'B'` | analysis |
| `carla_uav_tracking/rollout/dam4sam_service.py:46` | runtime | `--model` | `'sam21pp-L'` | scientific |
| `carla_uav_tracking/rollout/dam4sam_service.py:47` | runtime | `--port` | `5601` | infra |
| `carla_uav_tracking/rollout/dam_diag.py:87` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/deploy_compare.py:57` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:242` | analysis | `logs` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:242` | analysis | `--llr` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:242` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:243` | analysis | `logs` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:243` | analysis | `--out` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:244` | analysis | `--fa` | `0.5` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:245` | analysis | `logs` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:246` | analysis | `logs` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:246` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:247` | analysis | `logs` | `—` | analysis |
| `carla_uav_tracking/rollout/drift_analysis.py:247` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/iter_gate.py:127` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/iter_mech.py:105` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/opening_cl.py:56` | analysis | `pairs` | `—` | analysis |
| `carla_uav_tracking/rollout/opening_cl.py:56` | analysis | `--stem` | `[]` | analysis |
| `carla_uav_tracking/rollout/opening_cl.py:57` | analysis | `--lock` | `12` | analysis |
| `carla_uav_tracking/rollout/opening_cls.py:98` | offline | `--out` | `—` | artefact |
| `carla_uav_tracking/rollout/opening_cls.py:98` | offline | `--device` | `'cuda:0'` | infra |
| `carla_uav_tracking/rollout/opening_cls.py:99` | offline | `--reuse-feats` | `False` | offline |
| `carla_uav_tracking/rollout/opening_cls.py:99` | offline | `--qwen-cache` | `''` | artefact |
| `carla_uav_tracking/rollout/opening_cls.py:100` | offline | `--no-qwen` | `False` | offline |
| `carla_uav_tracking/rollout/opening_ft.py:87` | offline | `--out` | `—` | artefact |
| `carla_uav_tracking/rollout/opening_ft.py:87` | offline | `--device` | `'cuda:0'` | infra |
| `carla_uav_tracking/rollout/opening_ft.py:88` | offline | `--blocks` | `4` | offline |
| `carla_uav_tracking/rollout/opening_ft.py:88` | offline | `--epochs` | `8` | offline |
| `carla_uav_tracking/rollout/opening_ft.py:89` | offline | `--qwen-cache` | `''` | artefact |
| `carla_uav_tracking/rollout/opening_ft.py:89` | offline | `--full` | `False` | offline |
| `carla_uav_tracking/rollout/opening_score.py:123` | offline | `--out` | `—` | artefact |
| `carla_uav_tracking/rollout/opening_score.py:123` | offline | `--max-eps` | `0` | offline |
| `carla_uav_tracking/rollout/opening_score.py:124` | offline | `--R` | `336` | offline |
| `carla_uav_tracking/rollout/opening_score.py:124` | offline | `--device` | `'cuda:0'` | infra |
| `carla_uav_tracking/rollout/opening_score.py:125` | offline | `--fast` | `False` | offline |
| `carla_uav_tracking/rollout/opening_score.py:126` | offline | `--dump` | `'npz'` | artefact |
| `carla_uav_tracking/rollout/opening_score.py:127` | offline | `--jprefix` | `'open_dump_'` | artefact |
| `carla_uav_tracking/rollout/opening_score.py:128` | offline | `--shard` | `0` | infra |
| `carla_uav_tracking/rollout/opening_score.py:128` | offline | `--nshard` | `1` | infra |
| `carla_uav_tracking/rollout/opening_score.py:129` | offline | `--report` | `—` | artefact |
| `carla_uav_tracking/rollout/pgif_replay.py:115` | analysis | `logs` | `—` | analysis |
| `carla_uav_tracking/rollout/pgif_replay.py:115` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/pgif_report.py:69` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/pgif_three_metrics.py:106` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/pgif_track_metrics.py:74` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/rollout/policy_server.py:98` | runtime | `--config` | `'train/config_v5.yaml'` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:99` | runtime | `--ckpt` | `'runs/stage2_v5/stage2_best.pt'` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:100` | runtime | `--language-mode` | `'neutral'` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:101` | runtime | `--device` | `'cuda:0'` | infra |
| `carla_uav_tracking/rollout/policy_server.py:102` | runtime | `--s2-period` | `1` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:103` | runtime | `--ablate` | `None` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:105` | runtime | `--ex-source` | `'ear'` | diagnostic iff value != track |
| `carla_uav_tracking/rollout/policy_server.py:111` | runtime | `--commit-k` | `5` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:112` | runtime | `--tid-head` | `'xattn'` | diagnostic iff value in {oracle, attrbind} |
| `carla_uav_tracking/rollout/policy_server.py:116` | runtime | `--tid-ckpt` | `None` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:117` | runtime | `--reid-feature` | `'vlmpool'` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:120` | runtime | `--reid-bank` | `1` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:121` | runtime | `--reid-topm` | `1` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:122` | runtime | `--warmup-oracle-s` | `0.0` | diagnostic |
| `carla_uav_tracking/rollout/policy_server.py:125` | runtime | `--frame-gain` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:128` | runtime | `--conf-tau` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:131` | runtime | `--reid-tavg` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:134` | runtime | `--conf-consensus` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:138` | runtime | `--reid-motion` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:142` | runtime | `--reid-motion-gate-px` | `160.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:143` | runtime | `--gallery-seed` | `'gt'` | diagnostic |
| `carla_uav_tracking/rollout/policy_server.py:146` | runtime | `--reanchor` | `'off'` | diagnostic iff value=oracle |
| `carla_uav_tracking/rollout/policy_server.py:150` | runtime | `--reanchor-patience` | `5` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:152` | runtime | `--reanchor-tavg` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:155` | runtime | `--assoc-mode` | `'deepsort'` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:158` | runtime | `--assoc-gate-px` | `160.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:159` | runtime | `--assoc-lambda` | `1.0` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:160` | runtime | `--dam4sam-host` | `'dam4sam-svc'` | infra |
| `carla_uav_tracking/rollout/policy_server.py:161` | runtime | `--tah-ckpt` | `'runs/tah_rel.pt'` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:162` | runtime | `--dam4sam-port` | `5601` | infra |
| `carla_uav_tracking/rollout/policy_server.py:163` | runtime | `--asd` | `'off'` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:165` | runtime | `--asd-cfg` | `''` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:166` | runtime | `--tracklets` | `'sim'` | GT identity keys (diagnostic / deploy_v1 only) iff value=sim |
| `carla_uav_tracking/rollout/policy_server.py:169` | runtime | `--occl-cands` | `False` | scientific |
| `carla_uav_tracking/rollout/policy_server.py:172` | runtime | `--event-override` | `False` | diagnostic |
| `carla_uav_tracking/rollout/policy_server.py:176` | runtime | `--port` | `5555` | infra |
| `carla_uav_tracking/rollout/probe_adapter.py:112` | analysis | `--cache` | `'runs/tah_cache.pt'` | analysis |
| `carla_uav_tracking/rollout/probe_adapter.py:113` | analysis | `--K` | `30` | analysis |
| `carla_uav_tracking/rollout/probe_adapter.py:113` | analysis | `--maxc` | `10` | analysis |
| `carla_uav_tracking/rollout/probe_adapter.py:114` | analysis | `--device` | `'cuda:0'` | infra |
| `carla_uav_tracking/rollout/probe_target_conditioning.py:77` | analysis | `--cache` | `'runs/tah_cache.pt'` | analysis |
| `carla_uav_tracking/rollout/probe_target_conditioning.py:78` | analysis | `--max-eps` | `0` | analysis |
| `carla_uav_tracking/rollout/run_rollout.py:82` | runtime | `--policy-host` | `'acot-policy-server'` | infra |
| `carla_uav_tracking/rollout/run_rollout.py:83` | runtime | `--policy-port` | `5555` | infra |
| `carla_uav_tracking/rollout/run_rollout.py:84` | runtime | `--carla-port` | `int(os.environ.get('CARLA_PORT', '2012'))` | infra |
| `carla_uav_tracking/rollout/run_rollout.py:85` | runtime | `--episodes` | `5` | scientific |
| `carla_uav_tracking/rollout/run_rollout.py:86` | runtime | `--seed-base` | `90000` | scientific |
| `carla_uav_tracking/rollout/run_rollout.py:87` | runtime | `--steps` | `1500` | scientific |
| `carla_uav_tracking/rollout/run_rollout.py:88` | runtime | `--town` | `None` | scientific |
| `carla_uav_tracking/rollout/run_rollout.py:89` | runtime | `--out` | `None` | artefact |
| `carla_uav_tracking/rollout/run_rollout.py:90` | runtime | `--expert-control` | `False` | diagnostic |
| `carla_uav_tracking/rollout/run_rollout.py:93` | runtime | `--predict-road` | `False` | diagnostic |
| `carla_uav_tracking/rollout/run_rollout.py:97` | runtime | `--road-oracle` | `False` | diagnostic |
| `carla_uav_tracking/rollout/run_rollout.py:101` | runtime | `--vis-raycast` | `False` | scientific |
| `carla_uav_tracking/rollout/run_rollout.py:104` | runtime | `--retarget-tick` | `None` | diagnostic |
| `carla_uav_tracking/rollout/run_rollout.py:109` | runtime | `--hires` | `1` | scientific |
| `carla_uav_tracking/rollout/smoke_iter.py:54` | test | `--dam4sam-port` | `5604` | infra |
| `carla_uav_tracking/rollout/trk_tune.py:54` | analysis | `logs` | `—` | analysis |
| `carla_uav_tracking/rollout/trk_tune.py:54` | analysis | `--top` | `10` | analysis |
| `carla_uav_tracking/rollout/v2_analysis.py:35` | analysis | `--json` | `—` | analysis |
| `carla_uav_tracking/scripts/annotate_road_pred.py:113` | datagen | `--dir` | `—` | datagen |
| `carla_uav_tracking/scripts/annotate_road_pred.py:114` | datagen | `--host` | `'127.0.0.1'` | infra |
| `carla_uav_tracking/scripts/annotate_road_pred.py:115` | datagen | `--port` | `2001` | infra |
| `carla_uav_tracking/scripts/generate_batch.py:28` | datagen | `--num-episodes` | `100` | datagen |
| `carla_uav_tracking/scripts/generate_batch.py:29` | datagen | `--config` | `None` | datagen |
| `carla_uav_tracking/scripts/generate_batch.py:30` | datagen | `--host` | `'127.0.0.1'` | infra |
| `carla_uav_tracking/scripts/generate_batch.py:31` | datagen | `--port` | `2000` | infra |
| `carla_uav_tracking/scripts/generate_batch.py:32` | datagen | `--output` | `'/data'` | artefact |
| `carla_uav_tracking/scripts/generate_batch.py:34` | datagen | `--start-id` | `0` | datagen |
| `carla_uav_tracking/scripts/generate_batch.py:35` | datagen | `--seed` | `42` | datagen |
| `carla_uav_tracking/scripts/generate_batch.py:37` | datagen | `--max-size-gb` | `45.0` | infra |
| `carla_uav_tracking/scripts/generate_batch.py:39` | datagen | `--postprocess` | `True` | datagen |
| `carla_uav_tracking/scripts/generate_batch.py:41` | datagen | `--town` | `None` | datagen |
| `carla_uav_tracking/scripts/generate_batch.py:42` | datagen | `--subset` | `''` | datagen |
| `carla_uav_tracking/scripts/generate_batch.py:43` | datagen | `--progress-file` | `None` | artefact |
| `carla_uav_tracking/scripts/generate_batch.py:44` | datagen | `--summary-jsonl` | `None` | artefact |
| `carla_uav_tracking/scripts/generate_batch.py:45` | datagen | `--provenance` | `None` | artefact |
| `carla_uav_tracking/scripts/generate_single.py:44` | datagen | `--config` | `None` | datagen |
| `carla_uav_tracking/scripts/generate_single.py:46` | datagen | `--host` | `'localhost'` | infra |
| `carla_uav_tracking/scripts/generate_single.py:48` | datagen | `--port` | `2000` | infra |
| `carla_uav_tracking/scripts/generate_single.py:50` | datagen | `--output` | `'./data/raw'` | artefact |
| `carla_uav_tracking/scripts/generate_single.py:52` | datagen | `--episode-id` | `0` | datagen |
| `carla_uav_tracking/scripts/generate_single.py:54` | datagen | `--max-steps` | `None` | datagen |
| `carla_uav_tracking/scripts/h5_to_mp4.py:112` | datagen | `--src` | `—` | datagen |
| `carla_uav_tracking/scripts/h5_to_mp4.py:113` | datagen | `--out` | `—` | artefact |
| `carla_uav_tracking/scripts/h5_to_mp4.py:114` | datagen | `--fps` | `10` | datagen |
| `carla_uav_tracking/scripts/h5_to_mp4.py:115` | datagen | `--qscale` | `23` | datagen |
| `carla_uav_tracking/scripts/h5_to_mp4.py:116` | datagen | `--workers` | `max(1, (os.cpu_count() or 2) // 2)` | infra |
| `carla_uav_tracking/scripts/h5_to_mp4.py:117` | datagen | `--limit` | `0` | datagen |
| `carla_uav_tracking/scripts/h5_to_mp4.py:118` | datagen | `--boxes` | `False` | datagen |
| `carla_uav_tracking/scripts/launch_parallel.py:22` | datagen | `--num-servers` | `4` | datagen |
| `carla_uav_tracking/scripts/launch_parallel.py:23` | datagen | `--episodes-per-server` | `50` | datagen |
| `carla_uav_tracking/scripts/launch_parallel.py:24` | datagen | `--base-port` | `2000` | datagen |
| `carla_uav_tracking/scripts/launch_parallel.py:25` | datagen | `--config` | `None` | datagen |
| `carla_uav_tracking/scripts/launch_parallel.py:26` | datagen | `--output` | `'./data/raw'` | artefact |
| `carla_uav_tracking/scripts/launch_parallel.py:27` | datagen | `--gpus` | `'0,1,2,3'` | infra |
| `carla_uav_tracking/scripts/qc_report.py:121` | datagen | `--data` | `—` | artefact |
| `carla_uav_tracking/scripts/qc_report.py:122` | datagen | `--previews` | `False` | datagen |
| `carla_uav_tracking/scripts/qc_v6.py:152` | datagen | `root` | `—` | datagen |
| `carla_uav_tracking/scripts/qc_v6.py:152` | datagen | `--split-out` | `—` | artefact |
| `carla_uav_tracking/scripts/qc_v6.py:152` | datagen | `--viz` | `0` | datagen |
| `carla_uav_tracking/scripts/qc_v6.py:153` | datagen | `--reject-report` | `False` | datagen |
| `carla_uav_tracking/scripts/render_video.py:81` | datagen | `--data` | `—` | artefact |
| `carla_uav_tracking/scripts/render_video.py:82` | datagen | `--out` | `None` | artefact |
| `carla_uav_tracking/scripts/render_video.py:83` | datagen | `--fps` | `15` | datagen |
| `carla_uav_tracking/scripts/render_video.py:84` | datagen | `--raw` | `False` | datagen |
| `carla_uav_tracking/scripts/render_video.py:85` | datagen | `--max-frames` | `0` | datagen |
| `carla_uav_tracking/scripts/visualize_episode.py:19` | datagen | `--episode` | `—` | datagen |
| `carla_uav_tracking/scripts/visualize_episode.py:20` | datagen | `--fps` | `10` | datagen |
| `carla_uav_tracking/scripts/visualize_episode.py:21` | datagen | `--output` | `None` | artefact |
| `iter/dashboard_tui.py:165` | datagen | `--once` | `False` | datagen |
| `iter/log_iter.py:57` | datagen | `--type` | `—` | datagen |
| `iter/log_iter.py:59` | datagen | `--agent` | `'data'` | datagen |
| `iter/log_iter.py:60` | datagen | `--round` | `0` | datagen |
| `iter/log_iter.py:61` | datagen | `--version` | `''` | datagen |
| `iter/log_iter.py:62` | datagen | `--metrics` | `''` | datagen |
| `iter/log_iter.py:63` | datagen | `--gates` | `''` | datagen |
| `iter/log_iter.py:64` | datagen | `--config` | `''` | datagen |
| `iter/log_iter.py:65` | datagen | `--artifact` | `''` | datagen |
| `iter/log_iter.py:66` | datagen | `--note` | `''` | datagen |
| `iter/log_iter.py:67` | datagen | `--ts` | `''` | datagen |
| `iter/review_sample.py:195` | datagen | `--dir` | `—` | datagen |
| `iter/review_sample.py:196` | datagen | `--config` | `'train/config.yaml'` | datagen |
| `iter/review_sample.py:197` | datagen | `--out` | `''` | artefact |
| `iter/review_sample.py:198` | datagen | `--round` | `0` | datagen |
| `runs/abl_allactor/ablate.py:180` | analysis | `--steps` | `1200` | analysis |
| `runs/abl_allactor/ablate.py:180` | analysis | `--workers` | `40` | infra |
| `runs/abl_allactor/ablate.py:181` | analysis | `--variants` | `'target,all,balanced,all_cut'` | analysis |
| `runs/abl_allactor/ablate.py:181` | analysis | `--seeds` | `'0,1,2'` | analysis |
| `runs/abl_allactor/ablate.py:182` | analysis | `--smoke` | `False` | analysis |
| `runs/abl_allactor/ablate2.py:234` | analysis | `--arch` | `—` | analysis |
| `runs/abl_allactor/ablate2.py:235` | analysis | `--steps` | `2400` | analysis |
| `runs/abl_allactor/ablate2.py:235` | analysis | `--workers` | `40` | infra |
| `runs/abl_allactor/ablate2.py:236` | analysis | `--variants` | `'target,all,balanced,all_cut'` | analysis |
| `runs/abl_allactor/ablate2.py:236` | analysis | `--seeds` | `'0,1,2'` | analysis |
| `runs/abl_allactor/ablate2.py:237` | analysis | `--smoke` | `0` | analysis |
| `runs/abl_allactor/ablate3.py:274` | analysis | `--plan` | `'all'` | analysis |
| `runs/abl_allactor/ablate3.py:275` | analysis | `--steps` | `2400` | analysis |
| `runs/abl_allactor/ablate3.py:275` | analysis | `--workers` | `64` | infra |
| `runs/abl_allactor/ablate3.py:276` | analysis | `--seeds` | `'0,1'` | analysis |
| `runs/abl_allactor/ablate3.py:277` | analysis | `--smoke` | `''` | analysis |
| `runs/abl_allactor/build_cache_v5.py:31` | analysis | `--shard` | `0` | infra |
| `runs/abl_allactor/build_cache_v5.py:31` | analysis | `--nshards` | `1` | analysis |
| `runs/abl_allactor/build_cache_v5.py:32` | analysis | `--stride` | `2` | analysis |
| `runs/abl_allactor/build_cache_v5.py:32` | analysis | `--out` | `—` | analysis |
| `runs/abl_allactor/build_cache_v5.py:33` | analysis | `--device` | `'cuda:0'` | infra |
| `runs/abl_allactor/eval_post.py:235` | analysis | `--workers` | `48` | infra |
| `runs/abl_allactor/eval_post.py:235` | analysis | `--seeds` | `'0,1'` | analysis |
| `runs/abl_allactor/eval_post.py:236` | analysis | `--smoke` | `''` | analysis |
| `runs/data_v6/orchestrate.py:101` | other-owner | `--plan` | `—` | other-owner |
| `runs/data_v6/orchestrate.py:102` | other-owner | `--out` | `—` | other-owner |
| `runs/data_v6/orchestrate.py:103` | other-owner | `--slots` | `SLOTS` | other-owner |
| `runs/data_v6/orchestrate.py:104` | other-owner | `--chunk` | `5` | other-owner |
| `runs/data_v6/orchestrate.py:105` | other-owner | `--seed-base` | `0` | other-owner |
| `runs/data_v6/orchestrate.py:106` | other-owner | `--min-free-gb` | `250.0` | other-owner |
| `runs/data_v6/orchestrate.py:107` | other-owner | `--dry-run` | `False` | other-owner |
| `runs/data_v6/orchestrate.py:108` | other-owner | `--resume` | `False` | other-owner |
| `train/audit_bearing.py:45` | offline | `--config` | `'train/config.yaml'` | offline |
| `train/backbone_kv.py:145` | offline | `--config` | `—` | offline |
| `train/backbone_kv.py:146` | offline | `--layers` | `None` | offline |
| `train/backbone_kv.py:148` | offline | `--shard` | `0` | infra |
| `train/backbone_kv.py:149` | offline | `--num-shards` | `1` | infra |
| `train/backbone_kv.py:150` | offline | `--device` | `None` | infra |
| `train/baseline_deadreckon.py:96` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/baseline_deadreckon.py:97` | offline | `--ckpt` | `None` | artefact |
| `train/baseline_deadreckon.py:98` | offline | `--device` | `'cuda:2'` | infra |
| `train/baseline_deadreckon.py:99` | offline | `--split` | `'val'` | artefact |
| `train/baseline_deadreckon.py:100` | offline | `--long-thresh` | `3.0` | offline |
| `train/baseline_deadreckon.py:101` | offline | `--turn-deg` | `30.0` | offline |
| `train/bc_center_train.py:178` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/bc_center_train.py:179` | offline | `--ckpt` | `'runs/stage2_v5/stage2_best.pt'` | artefact |
| `train/bc_center_train.py:180` | offline | `--stage` | `'all'` | offline |
| `train/bc_center_train.py:181` | offline | `--cache` | `'runs/bc_center_cache.pt'` | artefact |
| `train/bc_center_train.py:182` | offline | `--save-ckpt` | `'runs/stage2_bc_center.pt'` | artefact |
| `train/bc_center_train.py:183` | offline | `--episodes` | `999` | offline |
| `train/bc_center_train.py:184` | offline | `--stride` | `3` | offline |
| `train/bc_center_train.py:185` | offline | `--cap` | `150` | offline |
| `train/bc_center_train.py:186` | offline | `--epochs` | `15` | offline |
| `train/bc_center_train.py:187` | offline | `--lr` | `0.0001` | offline |
| `train/bc_center_train.py:188` | offline | `--last-layers` | `4` | offline |
| `train/bc_center_train.py:189` | offline | `--yaw-cap-deg` | `20.0` | offline |
| `train/bc_center_train.py:190` | offline | `--center-blend` | `1.0` | offline |
| `train/bc_center_train.py:192` | offline | `--device` | `'cuda:0'` | infra |
| `train/benchmark_leak_probe.py:114` | offline | `--config` | `'train/config.yaml'` | offline |
| `train/benchmark_leak_probe.py:115` | offline | `--split` | `'val'` | artefact |
| `train/check_data_ready.py:55` | offline | `--dir` | `'/nvidia/hque/data/carla_data/mvp'` | offline |
| `train/check_data_ready.py:56` | offline | `--stable-min` | `20.0` | offline |
| `train/dagger_train.py:104` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/dagger_train.py:105` | offline | `--dagger-dir` | `'runs/dagger_v5'` | offline |
| `train/dagger_train.py:106` | offline | `--warm` | `'runs/tid_attrbind.pt'` | offline |
| `train/dagger_train.py:107` | offline | `--device` | `'cuda:2'` | infra |
| `train/dagger_train.py:108` | offline | `--epochs` | `20` | offline |
| `train/dagger_train.py:109` | offline | `--lr` | `0.0003` | offline |
| `train/dagger_train.py:110` | offline | `--offline-n` | `6000` | offline |
| `train/dagger_train.py:111` | offline | `--out` | `'runs/tid_attrbind_dagger.pt'` | artefact |
| `train/eval_stage1.py:22` | offline | `--config` | `—` | offline |
| `train/eval_stage1.py:23` | offline | `--ckpt` | `'runs/stage1_ear/stage1_ear_best.pt'` | artefact |
| `train/eval_stage1.py:24` | offline | `--split` | `'val'` | artefact |
| `train/fullframe_probe.py:19` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/fullframe_probe.py:20` | offline | `--device` | `'cuda:2'` | infra |
| `train/fullframe_probe.py:21` | offline | `--ntr` | `5000` | offline |
| `train/fullframe_probe.py:22` | offline | `--nte` | `3000` | offline |
| `train/fullframe_probe.py:23` | offline | `--recrop` | `False` | offline |
| `train/fullframe_probe.py:24` | offline | `--random` | `False` | offline |
| `train/grounding_attr_probe.py:34` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/grounding_attr_probe.py:35` | offline | `--ncap` | `12000` | offline |
| `train/grounding_head_compare.py:35` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/grounding_head_compare.py:36` | offline | `--xattn` | `'runs/stage2_v5/stage2_best.pt'` | offline |
| `train/grounding_head_compare.py:37` | offline | `--attrbind` | `'runs/tid_attrbind.pt'` | offline |
| `train/grounding_head_compare.py:38` | offline | `--device` | `'cpu'` | infra |
| `train/grounding_probe.py:31` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/grounding_probe.py:32` | offline | `--ckpt` | `'runs/stage2_v5/stage2_best.pt'` | artefact |
| `train/grounding_probe.py:33` | offline | `--device` | `'cuda:2'` | infra |
| `train/grounding_probe.py:34` | offline | `--split` | `'val'` | artefact |
| `train/grounding_probe_midepisode.py:85` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/grounding_probe_midepisode.py:86` | offline | `--ckpt` | `'runs/stage2_v5/stage2_best.pt'` | artefact |
| `train/grounding_probe_midepisode.py:87` | offline | `--device` | `'cuda:0'` | infra |
| `train/grounding_probe_midepisode.py:88` | offline | `--split` | `'val'` | artefact |
| `train/grounding_probe_midepisode.py:89` | offline | `--batch-size` | `64` | offline |
| `train/grounding_probe_midepisode.py:90` | offline | `--workers` | `4` | infra |
| `train/grounding_train.py:130` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/grounding_train.py:131` | offline | `--ckpt` | `None` | artefact |
| `train/grounding_train.py:132` | offline | `--from-run` | `None` | offline |
| `train/grounding_train.py:135` | offline | `--device` | `'cuda:2'` | infra |
| `train/grounding_train.py:136` | offline | `--loss` | `'ce'` | offline |
| `train/grounding_train.py:138` | offline | `--epochs` | `None` | offline |
| `train/grounding_train.py:139` | offline | `--lr` | `None` | offline |
| `train/grounding_train.py:140` | offline | `--wd` | `None` | offline |
| `train/grounding_train.py:141` | offline | `--margin` | `2.0` | offline |
| `train/grounding_train.py:142` | offline | `--hard-w` | `3.0` | offline |
| `train/grounding_train.py:143` | offline | `--head` | `'xattn'` | offline |
| `train/grounding_train.py:145` | offline | `--tid-d` | `None` | offline |
| `train/grounding_train.py:146` | offline | `--tid-heads` | `4` | offline |
| `train/grounding_train.py:147` | offline | `--dropout` | `0.1` | offline |
| `train/grounding_train.py:148` | offline | `--tol` | `0.03` | offline |
| `train/grounding_train.py:149` | offline | `--strict` | `False` | offline |
| `train/grounding_train.py:150` | offline | `--out` | `None` | artefact |
| `train/intercept_eval.py:28` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/intercept_eval.py:29` | offline | `--ckpt` | `'runs/stage2_v5/stage2_best.pt'` | artefact |
| `train/intercept_eval.py:30` | offline | `--device` | `'cuda:2'` | infra |
| `train/make_split.py:89` | offline | `--config` | `—` | offline |
| `train/qwen_lora_grounding.py:188` | offline | `--rank` | `16` | offline |
| `train/qwen_lora_grounding.py:189` | offline | `--lora-lo` | `16` | offline |
| `train/qwen_lora_grounding.py:190` | offline | `--alpha` | `32` | offline |
| `train/qwen_lora_grounding.py:191` | offline | `--stride` | `6` | offline |
| `train/qwen_lora_grounding.py:192` | offline | `--eval-stride` | `9` | offline |
| `train/qwen_lora_grounding.py:193` | offline | `--epochs` | `3` | offline |
| `train/qwen_lora_grounding.py:194` | offline | `--accum` | `8` | offline |
| `train/qwen_lora_grounding.py:195` | offline | `--lr-lora` | `0.0001` | offline |
| `train/qwen_lora_grounding.py:196` | offline | `--lr-head` | `0.001` | offline |
| `train/qwen_lora_grounding.py:197` | offline | `--eval-max` | `1500` | offline |
| `train/qwen_lora_grounding.py:198` | offline | `--max-train` | `0` | offline |
| `train/qwen_lora_grounding.py:199` | offline | `--out` | `—` | artefact |
| `train/qwen_lora_grounding.py:200` | offline | `--device` | `'cuda:0'` | infra |
| `train/qwen_lora_paired_eval.py:57` | offline | `--out` | `—` | artefact |
| `train/qwen_lora_paired_eval.py:57` | offline | `--device` | `'cuda:0'` | infra |
| `train/recrop_probe.py:22` | offline | `--config` | `'train/config_v4.yaml'` | offline |
| `train/recrop_probe.py:23` | offline | `--device` | `'cuda:2'` | infra |
| `train/recrop_probe.py:24` | offline | `--ntr` | `4000` | offline |
| `train/recrop_probe.py:25` | offline | `--nte` | `2500` | offline |
| `train/recrop_probe.py:26` | offline | `--res` | `224` | offline |
| `train/reid_metric_train.py:171` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/reid_metric_train.py:172` | offline | `--stage` | `'all'` | offline |
| `train/reid_metric_train.py:173` | offline | `--cache` | `'runs/reid_metric_cache.npz'` | artefact |
| `train/reid_metric_train.py:174` | offline | `--episodes` | `999` | offline |
| `train/reid_metric_train.py:175` | offline | `--stride` | `5` | offline |
| `train/reid_metric_train.py:176` | offline | `--device` | `'cuda:0'` | infra |
| `train/reid_metric_train.py:177` | offline | `--epochs` | `30` | offline |
| `train/reid_metric_train.py:178` | offline | `--dproj` | `128` | offline |
| `train/reid_metric_train.py:179` | offline | `--tau` | `0.1` | offline |
| `train/reid_metric_train.py:180` | offline | `--lr` | `0.001` | offline |
| `train/reid_metric_train.py:181` | offline | `--p-inst` | `32` | offline |
| `train/reid_metric_train.py:182` | offline | `--k-view` | `4` | offline |
| `train/reid_metric_train.py:183` | offline | `--steps` | `100` | offline |
| `train/reid_metric_train.py:184` | offline | `--bank` | `2` | offline |
| `train/reid_metric_train.py:185` | offline | `--topm` | `2` | offline |
| `train/reid_resolution_probe.py:114` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/reid_resolution_probe.py:115` | offline | `--split` | `'val'` | artefact |
| `train/reid_resolution_probe.py:116` | offline | `--max-pixels` | `0` | offline |
| `train/reid_resolution_probe.py:117` | offline | `--feature` | `'vlmpool'` | offline |
| `train/reid_resolution_probe.py:121` | offline | `--dino-model` | `'vit_small_patch14_dinov2.lvd142m'` | offline |
| `train/reid_resolution_probe.py:122` | offline | `--crop-size` | `224` | offline |
| `train/reid_resolution_probe.py:123` | offline | `--crop-pad` | `1.3` | offline |
| `train/reid_resolution_probe.py:124` | offline | `--bank` | `1` | offline |
| `train/reid_resolution_probe.py:125` | offline | `--topm` | `1` | offline |
| `train/reid_resolution_probe.py:126` | offline | `--div-tau` | `0.9` | offline |
| `train/reid_resolution_probe.py:127` | offline | `--episodes` | `6` | offline |
| `train/reid_resolution_probe.py:128` | offline | `--stride` | `5` | offline |
| `train/reid_resolution_probe.py:129` | offline | `--device` | `'cuda:0'` | infra |
| `train/rl_e0_center.py:285` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/rl_e0_center.py:286` | offline | `--ckpt` | `'runs/stage2_v5/stage2_best.pt'` | artefact |
| `train/rl_e0_center.py:287` | offline | `--cache` | `'runs/bc_center_cache2.pt'` | artefact |
| `train/rl_e0_center.py:288` | offline | `--stage` | `'all'` | offline |
| `train/rl_e0_center.py:289` | offline | `--save-ckpt` | `'runs/stage2_rl_e0.pt'` | artefact |
| `train/rl_e0_center.py:290` | offline | `--episodes` | `999` | offline |
| `train/rl_e0_center.py:291` | offline | `--stride` | `4` | offline |
| `train/rl_e0_center.py:292` | offline | `--cap` | `100` | offline |
| `train/rl_e0_center.py:293` | offline | `--data-h` | `None` | artefact |
| `train/rl_e0_center.py:294` | offline | `--roll-h` | `6` | offline |
| `train/rl_e0_center.py:295` | offline | `--epochs` | `15` | offline |
| `train/rl_e0_center.py:296` | offline | `--lr` | `5e-05` | offline |
| `train/rl_e0_center.py:297` | offline | `--last-layers` | `4` | offline |
| `train/rl_e0_center.py:298` | offline | `--sigma-frac` | `0.25` | offline |
| `train/rl_e0_center.py:299` | offline | `--zex-noise` | `0.05` | offline |
| `train/rl_e0_center.py:300` | offline | `--batch` | `12` | offline |
| `train/rl_e0_center.py:301` | offline | `--sample-steps` | `0` | offline |
| `train/rl_e0_center.py:302` | offline | `--max-frames` | `0` | offline |
| `train/rl_e0_center.py:303` | offline | `--device` | `'cuda:0'` | infra |
| `train/stage1_ear.py:105` | offline | `--config` | `—` | offline |
| `train/stage2.py:244` | offline | `--config` | `—` | offline |
| `train/stage2.py:245` | offline | `--seed` | `None` | offline |
| `train/stage2.py:246` | offline | `--ckpt-dir` | `None` | artefact |
| `train/stage2.py:247` | offline | `--device` | `None` | infra |
| `train/tah_cache.py:27` | offline | `--config` | `'train/config_v5.yaml'` | offline |
| `train/tah_cache.py:28` | offline | `--out` | `'runs/tah_cache.pt'` | artefact |
| `train/tah_cache.py:29` | offline | `--stride` | `2` | offline |
| `train/tah_cache.py:30` | offline | `--device` | `'cuda:0'` | infra |
| `train/tah_cache.py:31` | offline | `--shard` | `0` | infra |
| `train/tah_cache.py:31` | offline | `--nshard` | `1` | infra |
| `train/tah_merge.py:9` | offline | `--glob` | `—` | artefact |
| `train/tah_merge.py:9` | offline | `--out` | `—` | artefact |
| `train/tah_train.py:56` | offline | `--cache` | `['runs/tah_cache.pt']` | artefact |
| `train/tah_train.py:60` | offline | `--seed` | `None` | offline |
| `train/tah_train.py:61` | offline | `--save-last` | `False` | artefact |
| `train/tah_train.py:63` | offline | `--out` | `'runs/tah.pt'` | artefact |
| `train/tah_train.py:64` | offline | `--epochs` | `40` | offline |
| `train/tah_train.py:65` | offline | `--lr` | `0.0003` | offline |
| `train/tah_train.py:66` | offline | `--val-frac` | `0.15` | offline |
| `train/tah_train.py:67` | offline | `--rel` | `False` | offline |
| `train/tah_train.py:68` | offline | `--motion` | `False` | offline |
| `train/tah_train.py:69` | offline | `--aug` | `False` | offline |
| `train/tah_train.py:70` | offline | `--fnoise` | `0.05` | offline |
| `train/tah_train.py:71` | offline | `--pjit` | `0.02` | offline |
| `train/tah_train.py:72` | offline | `--wd` | `0.0001` | offline |
| `train/tah_train.py:73` | offline | `--device` | `'cuda:0'` | infra |

| 文件 | 文件类别 | sys.argv 下标 |
|---|---|---|
| `carla_uav_tracking/rollout/continuous_metrics.py` | analysis | argv[1] |
| `carla_uav_tracking/rollout/opening_ds.py` | analysis | argv[1] |
| `carla_uav_tracking/rollout/opening_pair.py` | analysis | argv[1] |
| `carla_uav_tracking/rollout/trk_quality.py` | analysis | argv[1] |
| `carla_uav_tracking/scripts/_measure_occ.py` | datagen | argv[1] |
| `carla_uav_tracking/scripts/_occ_montage.py` | datagen | argv[1], argv[2] |
| `carla_uav_tracking/scripts/add_occ_labels.py` | datagen | argv[1] |
| `carla_uav_tracking/scripts/probe_occluders.py` | datagen | argv[1], argv[2] |
| `paper2_analyze.py` | analysis | argv[1] |
| `runs/data_v6/probe_instseg.py` | other-owner | argv[1], argv[2] |
| `runs/data_v6/probe_spawnfilter.py` | other-owner | argv[1], argv[2] |
| `runs/opening_dump/ocl_rate.py` | analysis | argv[1], argv[2] |
| `runs/opening_dump/oracle_cmp.py` | analysis | argv[1] |
| `train/mmap_cache.py` | offline | argv[1] |

## 2. 构造函数参数与 JSON 配置字段(92 个)

Policy 的 kwarg 由 policy_server 的同名 flag 传入;ASD、TrackletAssigner、AttrScorer、ZoomVerifier 的 kwarg 由对应的 JSON 字符串展开传入(`**json.loads(...)`),未知字段在运行时才报 TypeError;OPENING_CFG 先被 Policy pop 掉 6 个字段。

| 位置 | 来源通道 | 字段 | 默认值 | 分类 |
|---|---|---|---|---|
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `ckpt_path` | `REQUIRED` | injected |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `cfg` | `REQUIRED` | injected |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `vlm` | `REQUIRED` | injected |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `device` | `'cuda:0'` | infra |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `s2_period` | `1` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `shuffle_seed` | `0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `ablate` | `None` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `ex_source` | `'ear'` | diagnostic iff value != track |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `commit_k` | `5` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `tid_head` | `'xattn'` | diagnostic iff value in {oracle, attrbind} |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `tid_ckpt` | `None` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reid_feature` | `'vlmpool'` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reid_bank` | `1` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reid_topm` | `1` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `conf_tau` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `conf_consensus` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `frame_gain` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `warmup_oracle_s` | `0.0` | diagnostic |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reid_tavg` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reid_motion` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reid_motion_gate_px` | `160.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `gallery_seed` | `'gt'` | diagnostic |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reanchor` | `'off'` | diagnostic iff value=oracle |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reanchor_patience` | `5` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `reanchor_tavg` | `0.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `assoc_mode` | `'deepsort'` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `assoc_gate_px` | `160.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `assoc_lambda` | `1.0` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `dam4sam_host` | `'dam4sam-svc'` | infra |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `dam4sam_port` | `5601` | infra |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `tah_ckpt` | `'runs/tah_rel.pt'` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `asd` | `'off'` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `asd_cfg` | `''` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `tracklets` | `'sim'` | GT identity keys (diagnostic / deploy_v1 only) iff value=sim |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `occl_cands` | `False` | scientific |
| `carla_uav_tracking/rollout/policy.py:99` | Policy kwarg (policy_server flags) | `event_override` | `False` | diagnostic |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `mode` | `'rule'` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `k_anchor` | `3` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `beta_lang` | `0.25` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `lang_ema` | `0.7` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `tau` | `0.1` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `kappa` | `0.25` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `h` | `2.0` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `refractory` | `3` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `learned_ckpt` | `None` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `log_path` | `None` | collection |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `anchor_mode` | `'first'` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `eps_switch` | `0.01` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `q_lo` | `0.2` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `q_hi` | `0.7` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `conf_window` | `30` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `gate_memory` | `True` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `filter_kind` | `'bayes'` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `resid_decay` | `0.98` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `alpha` | `None` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `naive` | `False` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `gate_tah` | `True` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `gate_dam` | `True` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `absent_hold` | `False` | scientific |
| `carla_uav_tracking/rollout/asd.py:24` | ASD_CFG / --asd-cfg | `absent_max` | `8` | scientific |
| `carla_uav_tracking/rollout/tracklets.py:28` | TRK_CFG | `max_age` | `5` | scientific |
| `carla_uav_tracking/rollout/tracklets.py:28` | TRK_CFG | `gate_px` | `100.0` | scientific |
| `carla_uav_tracking/rollout/tracklets.py:28` | TRK_CFG | `gate_box` | `3.0` | scientific |
| `carla_uav_tracking/rollout/tracklets.py:28` | TRK_CFG | `w_app` | `0.5` | scientific |
| `carla_uav_tracking/rollout/tracklets.py:28` | TRK_CFG | `min_cos` | `0.3` | scientific |
| `carla_uav_tracking/rollout/tracklets.py:28` | TRK_CFG | `vel_beta` | `0.5` | scientific |
| `carla_uav_tracking/rollout/tracklets.py:28` | TRK_CFG | `app_alpha` | `0.3` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `model` | `REQUIRED` | injected |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `processor` | `REQUIRED` | injected |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `device` | `REQUIRED` | infra |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `ckpt` | `''` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `wm` | `1.0` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `wc` | `1.0` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `wq` | `0.0` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:116` | OPENING_CFG (OPENING=attr) | `ft` | `0.0` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:46` | OPENING_CFG (OPENING=zoom) | `model` | `REQUIRED` | injected |
| `carla_uav_tracking/rollout/opening_zoom.py:46` | OPENING_CFG (OPENING=zoom) | `processor` | `REQUIRED` | injected |
| `carla_uav_tracking/rollout/opening_zoom.py:46` | OPENING_CFG (OPENING=zoom) | `device` | `REQUIRED` | infra |
| `carla_uav_tracking/rollout/opening_zoom.py:46` | OPENING_CFG (OPENING=zoom) | `scale` | `3.0` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:46` | OPENING_CFG (OPENING=zoom) | `R` | `336` | scientific |
| `carla_uav_tracking/rollout/opening_zoom.py:46` | OPENING_CFG (OPENING=zoom) | `spotlight` | `True` | scientific |
| `carla_uav_tracking/rollout/metrics.py:36` | scorer (run_rollout passes fps, max_lost_s only) | `fps` | `10` | scientific |
| `carla_uav_tracking/rollout/metrics.py:36` | scorer (run_rollout passes fps, max_lost_s only) | `commit_k` | `5` | scientific |
| `carla_uav_tracking/rollout/metrics.py:36` | scorer (run_rollout passes fps, max_lost_s only) | `t_fail_s` | `2.0` | scientific |
| `carla_uav_tracking/rollout/metrics.py:36` | scorer (run_rollout passes fps, max_lost_s only) | `max_lost_s` | `5.0` | scientific |
| `carla_uav_tracking/rollout/metrics.py:36` | scorer (run_rollout passes fps, max_lost_s only) | `central_frac` | `0.3` | scientific |
| `carla_uav_tracking/rollout/policy.py:196` | OPENING_CFG (popped by Policy) | `window` | `1` | scientific |
| `carla_uav_tracking/rollout/policy.py:204` | OPENING_CFG (popped by Policy) | `agg` | `"sum"` | scientific |
| `carla_uav_tracking/rollout/policy.py:204` | OPENING_CFG (popped by Policy) | `final_reset` | `False` | scientific |
| `carla_uav_tracking/rollout/policy.py:205` | OPENING_CFG (popped by Policy) | `reset_asd` | `False` | scientific |
| `carla_uav_tracking/rollout/policy.py:209` | OPENING_CFG (popped by Policy) | `force_final` | `False` | scientific |
| `carla_uav_tracking/rollout/policy.py:209` | OPENING_CFG (popped by Policy) | `seed_asd` | `0.0` | scientific |

## 3. 环境变量读写(25 个变量 + 1 处动态键写入)

| 变量 | 位置(读/写) | 默认值 | 分类 |
|---|---|---|---|
| `ASD_CFG` | `carla_uav_tracking/rollout/policy.py:244` read<br>`carla_uav_tracking/rollout/policy.py:245` read | `—` | scientific |
| `ASD_LOG` | `carla_uav_tracking/rollout/policy.py:186` read<br>`carla_uav_tracking/rollout/policy.py:187` read<br>`carla_uav_tracking/rollout/policy.py:243` read<br>`carla_uav_tracking/rollout/policy.py:245` read<br>`carla_uav_tracking/rollout/policy.py:255` read<br>`carla_uav_tracking/rollout/policy.py:256` read<br>`carla_uav_tracking/rollout/policy.py:257` read<br>`carla_uav_tracking/rollout/smoke_iter.py:27` WRITE | `—` | collection / test |
| `CARLA_HOST` | `carla_uav_tracking/carla_quick_test.py:22` read | `"127.0.0.1"` | infra |
| `CARLA_PORT` | `carla_uav_tracking/carla_quick_test.py:23` read<br>`carla_uav_tracking/rollout/run_rollout.py:84` read<br>`carla_uav_tracking/rollout/smoke_expert_loop.py:75` read | `"2000"` | infra |
| `CARLA_ROOT` | `carla_uav_tracking/scripts/launch_parallel.py:16` read | `"/opt/carla"` | infra |
| `CARLA_TM_PORT` | `carla_uav_tracking/scene/scene_manager.py:24` read<br>`carla_uav_tracking/targets/vehicle_target.py:101` read | `"8000"` | infra |
| `COLD_LAMBDA` | `carla_uav_tracking/rollout/policy.py:234` read | `"0.9"` | scientific |
| `COLD_SINGLE` | `carla_uav_tracking/rollout/policy.py:235` read | `"0"` | scientific |
| `COLD_WINDOW` | `carla_uav_tracking/rollout/policy.py:233` read | `"0"` | scientific |
| `DAGGER_DIR` | `carla_uav_tracking/rollout/policy.py:158` read | `—` | collection |
| `DAM4SAM_DIAG` | `carla_uav_tracking/rollout/policy.py:255` read | `—` | collection |
| `DAM4SAM_EVERY_FRAME` | `carla_uav_tracking/rollout/policy.py:251` read | `—` | scientific |
| `DUMP_LABELS` | `carla_uav_tracking/rollout/env.py:177` read<br>`carla_uav_tracking/rollout/env.py:181` read | `—` | collection |
| `HOSTNAME` | `carla_uav_tracking/rollout/policy.py:355` read<br>`carla_uav_tracking/rollout/policy.py:734` read | `'x'` | infra |
| `OPENING` | `carla_uav_tracking/rollout/policy.py:193` read<br>`carla_uav_tracking/rollout/policy.py:210` read<br>`carla_uav_tracking/rollout/policy.py:212` read | `"head"` | scientific |
| `OPENING_CFG` | `carla_uav_tracking/rollout/policy.py:196` read<br>`carla_uav_tracking/rollout/policy.py:212` read | `—` | scientific |
| `OPENING_DUMP` | `carla_uav_tracking/rollout/policy.py:729` read<br>`carla_uav_tracking/rollout/policy.py:732` read | `—` | collection |
| `ORACLE_ABSENT` | `carla_uav_tracking/rollout/policy.py:1002` read<br>`carla_uav_tracking/rollout/policy.py:1013` read | `"0"` | diagnostic |
| `PARITY_DEVICE` | `carla_uav_tracking/rollout/parity_online_vlm.py:48` read | `cfg["backbone"].get("device", "cuda:0"` | test |
| `PARITY_EP` | `carla_uav_tracking/rollout/parity_online_vlm.py:34` read | `"episode_000000"` | test |
| `SMOKE_STEPS` | `carla_uav_tracking/rollout/smoke_expert_loop.py:76` read | `"200"` | test |
| `SMOKE_TOWN` | `carla_uav_tracking/rollout/smoke_expert_loop.py:83` read | `—` | test |
| `TAH_DAGGER_DIR` | `carla_uav_tracking/rollout/policy.py:161` read | `—` | collection |
| `TRK_CFG` | `carla_uav_tracking/rollout/policy.py:184` read<br>`carla_uav_tracking/rollout/policy.py:185` read | `—` | scientific |
| `V1_SELFRESET` | `carla_uav_tracking/rollout/policy.py:236` read | `"0"` | scientific |
| `<动态键 k>` | `carla_uav_tracking/rollout/smoke_iter.py:26` WRITE | `—` | test |

## 4. env → policy 线协议字段(17 项，人工整理，依据 `remote_policy.py`、`policy_server.py`、`env.py`、`policy.py`、`candidates.py`)

| 字段 | 内容 | 分类 |
|---|---|---|
| `act.rgb` | S1 camera frame | deployable |
| `act.language` | instruction | deployable |
| `act.proprio` | ego state | deployable |
| `act.step` | tick index | deployable |
| `act.rgb_hi` | k× opening camera at ticks 0/6/12 (--hires) | deployable (protocol frontend) |
| `act.retarget_event` | scripted re-target flag (--retarget-tick) | diagnostic |
| `act.target_color / act.target_bp` | GT target colour / blueprint (always sent today) | diagnostic (attrbind) / collection label; blanked for every other arm from P1 |
| `act.gt_candidates[0] tpos` | GT target world position → build_candidates (target box), _committed_world (matched-box localisation), estimator.update (ex_source != track) | privileged (box / matched-box) · diagnostic (WHERE) |
| `act.gt_candidates[1] dstates` | GT distractor states → build_candidates, _committed_world | privileged |
| `act.gt_candidates[2] campose` | ego camera pose → projection, frame_gain | deployable (ego odometry) |
| `act.gt_candidates[3] road_pred_world` | road-graph prediction from the TRUE target (--predict-road / --road-oracle) | diagnostic |
| `act.gt_candidates[4] vis` | ray-cast occlusion flags (t_vis, d_vis) (--vis-raycast) | privileged (occlusion filter only) |
| `cand.idx (tracklets=sim)` | simulator actor index used as identity key (target = −1) | GT identity → deploy_v1 / diagnostic only |
| `cset.is_target / cset.true_idx / cand.gt_idx` | GT labels created inside build_candidates (policy process) | scoring / collection only |
| `reply.cand_lite` | is_target, idx, depth, true_idx → scorer; **gt_idx NOT sent** (known gap: tracker-keyed distractors are scored by tracker id, metrics.py:29-32) | scoring |
| `reply.pred_slot / action / search_mode` | policy outputs | deployable |
| `reset` | carries NO seed today (logs aligned by order); P1 adds a log-only seed | — |

## 5. 启动脚本(100 个)

静态提取，作为交叉核对;权威的调用参数由 P0b 的 docker shim 实际捕获(见框架文档 §9 L0)。每个脚本列出：调用的入口、docker 操作、出现的全部 `--flag`、写死的行为环境变量赋值(不含 `VAR=${VAR:-默认}` 这种自身默认)、`${VAR:-…}` 可调参数、位置参数。

| 脚本 | 入口 | docker | 出现的 flag | 写死的行为环境变量 | 可调参数 | 位置参数 |
|---|---|---|---|---|---|---|
| `carla_uav_tracking/rollout/a1_finish.sh` | /workspace/rollout/continuous_metrics.py, carla_uav_tracking/rollout/eval_tah.sh, train/tah_train.py | docker run, docker exec, docker rm | --aug --cache --device --epochs --gpus --motion --out --rm --wd | EPISODES SEED_BASE STEPS | — | — |
| `carla_uav_tracking/rollout/a1_post.sh` | acot_note/figs/regen_arm_figs.py, carla_uav_tracking/rollout/control_stability.py, carla_uav_tracking/rollout/scenario_split.py | — | — | — | — | — |
| `carla_uav_tracking/rollout/b1_finish.sh` | /tmp/eval_tahdag.sh, train/tah_train.py | docker run, docker rm | --aug --cache --device --epochs --gpus --out --rel --rm --wd | EPISODES SEED_BASE STEPS | — | — |
| `carla_uav_tracking/rollout/collect_tahdag.sh` | /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gallery-seed --gpus --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --tah-ckpt --tid-head --town | TAH_DAGGER_DIR | EPISODES GPU HF_CACHE SEED_BASE STEPS | — |
| `carla_uav_tracking/rollout/deploy_eval_chain.sh` | /workspace/rollout/continuous_metrics.py, carla_uav_tracking/rollout/eval_tah.sh, train/docker/build.sh | docker exec, docker rm | — | EPISODES SEED_BASE STEPS | — | — |
| `carla_uav_tracking/rollout/eval_ablation.sh` | /workspace/rollout/compare_runs.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ablate --ckpt --config --episodes --format --gpus --label-a --label-b --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --town | EPISODES S2_PERIOD STEPS TOWN | ABLATE EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN ablate | $1 $2 $3 $4 $5 |
| `carla_uav_tracking/rollout/eval_asd.sh` | /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --config --episodes --ex-source --format --gpus --hires --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --tid-head --town --vis-raycast | CONFIG HEAD_ARGS | ASD_CFG ASD_LOG CARLA_PORT COLD_LAMBDA COLD_SINGLE COLD_WINDOW DAM4SAM_DIAG DAM4SAM_EVERY_FRAME DUMP_LABELS EPISODES GPU HEAD_ARGS HF_CACHE HIRES NTHR OPENING OPENING_CFG OPENING_DUMP ORACLE_ABSENT OUT_NAME S2_PERIOD SEED_BASE SRV STEPS TAH_DAGGER_DIR TM_PORT TOWN TRK_CFG V1_SELFRESET VIS_RAYCAST | — |
| `carla_uav_tracking/rollout/eval_attrbind.sh` | /workspace/rollout/compare_runs.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --config --device --episodes --ex-source --format --gpus --label-a --label-b --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --tid-ckpt --tid-head --town | EPISODES SEED_BASE STEPS | ATTR_CKPT CKPT CONFIG EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 $3 $4 |
| `carla_uav_tracking/rollout/eval_bc_center.sh` | /workspace/rollout/compare_runs.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gpus --label-a --label-b --language-mode --name --network --out --policy-host --policy-port --port --reid-bank --reid-feature --s2-period --seed-base --shm-size --steps --tid-head --town | EPISODES SEED_BASE STEPS | ARMS CONFIG CONF_TAU EPISODES GPU HF_CACHE REIDBANK S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 $3 |
| `carla_uav_tracking/rollout/eval_conf_consensus.sh` | /workspace/rollout/continuous_metrics.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-consensus --conf-tau --config --episodes --ex-source --format --gpus --language-mode --name --network --out --policy-host --policy-port --port --reid-bank --reid-feature --reid-motion --reid-tavg --reid-topm --s2-period --seed-base --shm-size --steps --tid-head --town | CONFIG EPISODES SEED_BASE STEPS | CONS EPISODES GPU HF_CACHE RMOT S2_PERIOD SEED_BASE STEPS TOWN | — |
| `carla_uav_tracking/rollout/eval_ex_source.sh` | /workspace/rollout/compare_runs.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --config --episodes --ex-source --format --gpus --label-a --label-b --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --town | EPISODES S2_PERIOD SEED_BASE STEPS TOWN | ARMS CKPT CONFIG EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 |
| `carla_uav_tracking/rollout/eval_ext_assoc.sh` | /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --assoc-mode --ckpt --conf-tau --config --episodes --ex-source --format --gpus --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --tid-head --town | EPISODES SEED_BASE STEPS | CONFIG EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 $3 |
| `carla_uav_tracking/rollout/eval_ext_dam4sam.sh` | /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --dam4sam-host --dam4sam-port --episodes --ex-source --format --gpus --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --tid-head --town | EPISODES SEED_BASE STEPS | CONFIG EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | — |
| `carla_uav_tracking/rollout/eval_ext_rerun.sh` | /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --assoc-mode --ckpt --conf-tau --config --dam4sam-host --dam4sam-port --episodes --ex-source --format --gpus --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --tid-head --town | CONFIG EPISODES SEED_BASE STEPS TOWN | HF_CACHE | $1 $2 |
| `carla_uav_tracking/rollout/eval_h0.sh` | /workspace/rollout/compare_runs.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --config --episodes --format --gpus --label-a --label-b --language-mode --name --network --out --policy-host --policy-port --port --s2-period --seed-base --shm-size --steps --town | EPISODES S2_PERIOD STEPS TOWN | EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 $3 $4 $5 |
| `carla_uav_tracking/rollout/eval_paper_batch.sh` | /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/continuous_metrics.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gpus --language-mode --name --network --out --policy-host --policy-port --port --reid-bank --reid-feature --reid-tavg --reid-topm --s2-period --seed-base --shm-size --steps --tid-head --town | EPISODES SEED_BASE STEPS | CONFIG EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 $3 $4 |
| `carla_uav_tracking/rollout/eval_reanchor.sh` | /workspace/rollout/continuous_metrics.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gallery-seed --gpus --language-mode --name --network --out --policy-host --policy-port --port --reanchor --reanchor-patience --reanchor-tavg --reid-bank --reid-feature --reid-tavg --reid-topm --s2-period --seed-base --shm-size --steps --tid-head --town | EPISODES SEED_BASE STEPS | ARMS CKPT CONFIG EPISODES GPU HF_CACHE PATIENCE RTAVG S2_PERIOD SEED_BASE SRV STEPS TAVG TOWN | $1 $2 $3 $4 $5 |
| `carla_uav_tracking/rollout/eval_reid_temporal.sh` | /workspace/rollout/compare_runs.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gpus --label-a --label-b --language-mode --name --network --out --policy-host --policy-port --port --reid-bank --reid-feature --reid-tavg --reid-topm --s2-period --seed-base --shm-size --steps --tid-head --town | EPISODES SEED_BASE STEPS | CKPT CONFIG EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE SKIP_BASE STEPS TAVG TAVGS TOWN | $1 $2 $3 |
| `carla_uav_tracking/rollout/eval_reidmotion.sh` | /workspace/rollout/continuous_metrics.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gpus --language-mode --name --network --out --policy-host --policy-port --port --reid-bank --reid-feature --reid-motion --reid-tavg --reid-topm --s2-period --seed-base --shm-size --steps --tid-head --town | CONFIG EPISODES SEED_BASE STEPS | EPISODES GPU HF_CACHE RMOT S2_PERIOD SEED_BASE STEPS TOWN | — |
| `carla_uav_tracking/rollout/eval_reidmotion_sweep.sh` | /workspace/rollout/continuous_metrics.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gpus --language-mode --name --network --out --policy-host --policy-port --port --reid-bank --reid-feature --reid-motion --reid-tavg --reid-topm --s2-period --seed-base --shm-size --steps --tid-head --town | CONFIG | EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 |
| `carla_uav_tracking/rollout/eval_road.sh` | /workspace/rollout/compare_runs.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --config --episodes --ex-source --format --gpus --label-a --label-b --language-mode --name --network --out --policy-host --policy-port --port --predict-road --road-oracle --s2-period --seed-base --shm-size --steps --town | EPISODES S2_PERIOD SEED_BASE STEPS TOWN | ARMS CKPT CONFIG EPISODES GPU HF_CACHE S2_PERIOD SEED_BASE STEPS TOWN | $1 $2 |
| `carla_uav_tracking/rollout/eval_tah.sh` | /workspace/rollout/continuous_metrics.py, /workspace/rollout/run_rollout.py, carla_uav_tracking/rollout/policy_server.py | docker run, docker exec, docker rm | --ckpt --conf-tau --config --episodes --ex-source --format --gallery-seed --gpus --language-mode --name --network --out --policy-host --policy-port --port --reanchor --reanchor-patience --s2-period --seed-base --shm-size --steps --tah-ckpt --tid-head --town | CONFIG EPISODES SEED_BASE STEPS | CONF EPISODES GPU GSEED HF_CACHE OUT_NAME REANCHOR RPAT S2_PERIOD SEED_BASE STEPS TAH_CKPT TOWN | — |
| `carla_uav_tracking/rollout/handoff_reanchor.sh` | carla_uav_tracking/rollout/eval_reanchor.sh | docker exec, docker rm | --format --reanchor-patience | EPISODES SEED_BASE STEPS | FIRED | $1 |
| `carla_uav_tracking/rollout/run_h1h2_pipeline.sh` | carla_uav_tracking/rollout/eval_ablation.sh | — | --format | EPISODES S2_PERIOD SEED_BASE STEPS TOWN | GPU | — |
| `carla_uav_tracking/rollout/run_langtavg_rescue.sh` | carla_uav_tracking/rollout/eval_reanchor.sh | docker exec, docker rm | --format --reanchor-tavg | EPISODES SEED_BASE STEPS | FIRED RTAVG | $1 |
| `carla_uav_tracking/scripts/gen_seed_town_list.sh` | scripts/generate_batch.py | — | --config --max-size-gb --num-episodes --output --port --seed --start-id --town | — | CARLA_ROOT | $1 $2 $3 $4 $5 $6 |
| `carla_uav_tracking/scripts/pull_videos_mac.sh` | — | — | --progress | — | — | $1 $2 $3 |
| `carla_uav_tracking/scripts/run_job_v6.sh` | scripts/generate_batch.py | — | --config --max-size-gb --num-episodes --output --port --progress-file --provenance --seed --start-id --subset --summary-jsonl --town | TOWN | CARLA_ROOT | $1 $2 $3 $4 $5 $6 $7 $8 $9 |
| `carla_uav_tracking/scripts/run_resilient.sh` | scripts/generate_batch.py | — | --config --max-size-gb --num-episodes --output --port --seed --start-id | — | CARLA_ROOT | $1 $2 $3 $4 $5 $6 $7 |
| `iter/gen_parallel.sh` | scripts/run_resilient.sh | docker exec | — | — | — | $1 $2 $3 $4 |
| `iter/gen_sample.sh` | scripts/run_resilient.sh, -m iter.review_sample | docker exec | --agent --artifact --config --dir --gates --metrics --note --out --round --type --version | — | — | $1 $2 $3 $4 $5 |
| `runs/abl_allactor/mon3.sh` | — | — | --pla | — | — | $1 |
| `runs/abl_allactor/mon_post.sh` | — | — | --worker | — | — | $1 |
| `runs/asd/after_s1.sh` | $R/eval_asd.sh | — | --asd --conf-tau --gallery-seed --reanchor --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE | — | — |
| `runs/asd/chains.sh` | $R/eval_asd.sh | — | --asd --conf-tau --gallery-seed --reanchor --reid-bank --reid-feature --reid-tavg --reid-topm --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE TOWN | — | $1 $2 $3 $4 $5 $6 $7 $8 $9 |
| `runs/asd/deploy_all/build_dam4sam.sh` | — | docker run, docker exec, docker rm | --base --gpus --index-url --name --network --shm-size | — | — | — |
| `runs/asd/deploy_all/chains.sh` | $PWD/$R/dam4sam_service.py, $R/eval_asd.sh | docker exec | --asd --assoc-mode --base --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --model --port --reanchor --reid-bank --reid-feature --reid-motion --reid-tavg --reid-topm --tah-ckpt --tid-head | ASD_CFG EPISODES HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 |
| `runs/asd/deploy_all/rerun_dam4sam.sh` | $R/eval_asd.sh | — | --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tid-head | ASD_CFG EPISODES HEAD_ARGS SEED_BASE | — | — |
| `runs/asd/deploy_v2/chains_v2.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | VIS_RAYCAST | — | $1 |
| `runs/asd/deploy_v2/chains_v2h.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | TRK_CFG VIS_RAYCAST | — | $1 |
| `runs/asd/deploy_v2/chains_v2t.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | TRK_CFG VIS_RAYCAST | — | $1 |
| `runs/asd/deploy_v2/resume_v2.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES SEED_BASE VIS_RAYCAST | START_PART | $1 |
| `runs/asd/iter/chains_dam_diag.sh` | $L/launch_wd.sh | — | --assoc-gate-px --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tid-head | DAM4SAM_DIAG | — | $1 |
| `runs/asd/iter/chains_g11.sh` | $L/launch_wd.sh | — | --assoc-gate-px --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tid-head | DAM4SAM_DIAG DAM4SAM_EVERY_FRAME | — | — |
| `runs/asd/iter/chains_r2.sh` | $L/launch.sh | — | --asd --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tah-ckpt --tid-head | COLD_LAMBDA COLD_WINDOW | — | $1 |
| `runs/asd/iter/chains_r2_s4b.sh` | $L/launch.sh | — | --asd --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tid-head | EPISODES SEED_BASE | — | — |
| `runs/asd/iter/chains_r3.sh` | $L/launch_wd.sh | — | --asd --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tah-ckpt --tid-head | COLD_LAMBDA COLD_WINDOW EPISODES SEED_BASE | — | $1 |
| `runs/asd/iter/launch.sh` | $R/eval_asd.sh | — | — | ASD_CFG HEAD_ARGS TOWN | EPISODES SEED_BASE | $1 $2 $3 $4 $5 $6 $7 |
| `runs/asd/iter/launch_wd.sh` | $R/eval_asd.sh | docker exec, docker rm | — | ASD_CFG EPISODES HEAD_ARGS SEED_BASE TOWN | EPISODES FIRST_MIN SEED_BASE STALL_MIN START_PART | $1 $2 $3 $4 $5 $6 $7 |
| `runs/asd/iter/queue1.sh` | $L/launch.sh | — | --asd --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tah-ckpt --tid-head | COLD_LAMBDA COLD_WINDOW | — | $1 |
| `runs/asd/iter/queue2.sh` | $L/launch.sh | — | --asd --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tah-ckpt --tid-head | COLD_LAMBDA COLD_SINGLE COLD_WINDOW V1_SELFRESET | — | $1 |
| `runs/asd/iter/queue3.sh` | $L/launch.sh | — | --asd --conf-tau --dam4sam-host --dam4sam-port --gallery-seed --reanchor --tah-ckpt --tid-head | COLD_LAMBDA COLD_WINDOW | — | $1 |
| `runs/asd/iter/stop_hung.sh` | — | docker exec, docker rm | --ppid | — | — | — |
| `runs/asd/launch_c.sh` | $R/eval_asd.sh | — | --asd --conf-tau --gallery-seed --reanchor --reid-bank --reid-feature --reid-tavg --reid-topm --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 $8 |
| `runs/asd/launch_f.sh` | $R/eval_asd.sh | — | --asd --conf-tau --gallery-seed --reanchor --reid-bank --reid-feature --reid-tavg --reid-topm --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 $8 |
| `runs/asd/launch_g.sh` | $R/eval_asd.sh | docker exec | --asd --conf-tau --gallery-seed --reanchor --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 $8 |
| `runs/asd/launch_gt.sh` | $R/eval_asd.sh | docker exec | --asd --conf-tau --gallery-seed --reanchor --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 $8 $9 |
| `runs/asd/relaunch_b.sh` | $R/eval_asd.sh | docker exec, docker rm | --asd --conf-tau --gallery-seed --reanchor --reid-bank --reid-feature --reid-tavg --reid-topm --tah-ckpt --tid-head | HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 |
| `runs/asd/restart_split.sh` | $R/eval_asd.sh | docker exec, docker rm | --asd --conf-tau --gallery-seed --reanchor --reid-bank --reid-feature --reid-tavg --reid-topm --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 $8 |
| `runs/asd/round3.sh` | $R/eval_asd.sh | docker exec, docker rm | --asd --conf-tau --gallery-seed --reanchor --reid-bank --reid-feature --reid-tavg --reid-topm --tah-ckpt --tid-head | ASD_CFG HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 $8 |
| `runs/asd/sticky_check/run.sh` | $R/eval_asd.sh | — | --asd --conf-tau --gallery-seed --reanchor --tah-ckpt --tid-head | ASD_CFG EPISODES HEAD_ARGS SEED_BASE | — | $1 $2 $3 $4 $5 $6 $7 $8 |
| `runs/opening_dump/chains_attr.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 |
| `runs/opening_dump/chains_v3.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 |
| `runs/opening_dump/chains_zo.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 |
| `runs/opening_dump/f_oracle_absent.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG ORACLE_ABSENT SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 |
| `runs/opening_dump/f_oracle_hyg.sh` | $L/launch_wd.sh | docker exec | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG ORACLE_ABSENT SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 $5 |
| `runs/opening_dump/f_v4.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 |
| `runs/opening_dump/launch_fix.sh` | run_openval.sh | — | — | — | — | $1 $2 |
| `runs/opening_dump/launch_v3.sh` | run_openval.sh | — | — | — | — | $1 $2 $3 |
| `runs/opening_dump/launch_v3p.sh` | run_openval.sh | — | — | — | — | $1 $2 $3 |
| `runs/opening_dump/launch_v4.sh` | $L/launch_wd.sh, run_openval.sh | docker exec | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 |
| `runs/opening_dump/redo_attr3.sh` | run_openval.sh | docker exec | — | — | — | — |
| `runs/opening_dump/run_dump.sh` | carla_uav_tracking/rollout/eval_asd.sh | — | --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head | EPISODES HEAD_ARGS OPENING_DUMP SEED_BASE STEPS VIS_RAYCAST | — | $1 $2 $3 $4 $5 |
| `runs/opening_dump/run_dumph.sh` | carla_uav_tracking/rollout/eval_asd.sh | — | --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head | EPISODES HEAD_ARGS HIRES OPENING_DUMP SEED_BASE STEPS VIS_RAYCAST | — | $1 $2 $3 $4 $5 |
| `runs/opening_dump/run_dumpt.sh` | carla_uav_tracking/rollout/eval_asd.sh | — | --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head | DUMP_LABELS EPISODES HEAD_ARGS HIRES OPENING_DUMP SEED_BASE STEPS VIS_RAYCAST | — | $1 $2 $3 $4 $5 |
| `runs/opening_dump/run_openval.sh` | carla_uav_tracking/rollout/eval_asd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | ASD_CFG EPISODES HEAD_ARGS HIRES OPENING OPENING_CFG SEED_BASE STEPS TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 $5 $6 $7 $8 |
| `runs/opening_dump/smoke_v3.sh` | run_openval.sh | — | — | — | — | — |
| `runs/opening_dump/start_base.sh` | run_openval.sh | docker exec | — | OPENING | — | $1 $2 |
| `runs/opening_dump/wait_f_v3.sh` | chains_v3.sh | — | --format | — | — | — |
| `runs/tah_v90/collect_dagger.sh` | $L/launch_wd.sh | — | --asd --conf-tau --gallery-seed --occl-cands --reanchor --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TAH_DAGGER_DIR TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 $5 |
| `runs/tah_v90/gen_fov90.sh` | scripts/run_resilient.sh | docker exec | — | — | — | $1 $2 $3 |
| `runs/tah_v90/gen_fov90b.sh` | scripts/gen_seed_town_list.sh | docker exec | — | — | — | $1 $2 $3 |
| `runs/tah_v90/pipeline.sh` | $L/launch_wd.sh, carla_uav_tracking/rollout/opening_ds.py, opening_ds.py, train/tah_cache.py, train/tah_merge.py, train/tah_train.py | docker run | --asd --aug --cache --conf-tau --config --device --epochs --gallery-seed --glob --gpus --motion --network --nshard --occl-cands --out --reanchor --rm --seed --shard --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 |
| `runs/tah_v90/pipeline_v2.sh` | $L/launch_wd.sh, carla_uav_tracking/rollout/opening_ds.py, opening_ds.py, train/tah_cache.py, train/tah_merge.py, train/tah_train.py | docker run | --asd --aug --cache --conf-tau --config --device --epochs --gallery-seed --glob --gpus --motion --network --nshard --occl-cands --out --reanchor --rm --seed --shard --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 |
| `runs/tah_v90/pipeline_v3.sh` | $L/launch_wd.sh, opening_ds.py, train/tah_train.py | docker run | --asd --aug --cache --conf-tau --device --epochs --gallery-seed --gpus --motion --network --occl-cands --out --reanchor --rm --save-last --seed --tah-ckpt --tid-head --tracklets | EPISODES HIRES OPENING OPENING_CFG SEED_BASE TRK_CFG VIS_RAYCAST | — | $1 $2 $3 $4 $5 $6 $7 |
| `train/docker/build.sh` | $HERE/run.sh | — | — | — | — | — |
| `train/docker/run.sh` | -m train.stage1_ear | docker run, docker exec, docker rm | --config --gpus --name --rm --runtime --shm-size | — | DETACH HF_CACHE HF_TOKEN IMAGE NAME | — |
| `train/docker/run_logged.sh` | -m train.xxx | docker run, docker rm | --device --gpus --name --rm --shm-size | — | DETACH HF_CACHE HF_TOKEN IMAGE LOG NAME SHM | — |
| `train/nolang_exp.sh` | — | — | --ckpt-dir --config --device --layers --num-shards --seed --shard | — | — | $1 $2 |
| `train/nolang_recover.sh` | — | — | --ckpt-dir --config --device --seed | — | — | $1 $2 |
| `train/sanity_r3.sh` | — | — | --config --device --layers --num-shards --shard | — | — | — |
| `train/seed_exp.sh` | — | — | --ckpt-dir --config --device --seed | — | — | $1 $2 $3 $4 |
| `train/train_full.sh` | — | — | --config --device --layers --num-shards --shard | — | — | — |
| `train/train_v3_e1.sh` | — | — | --config --device --layers --num-shards --shard | — | — | $1 $2 $3 |
| `train/train_v3_stab.sh` | — | — | --ckpt-dir --config --device | — | — | — |
| `train/train_v3_stage2.sh` | — | — | --config --device | — | — | — |
| `train/train_v4_e1.sh` | — | — | --config --device --layers --num-shards --shard | — | — | $1 $2 $3 |
| `train/train_v5_e1.sh` | — | — | --config --device --layers --num-shards --shard | — | — | $1 $2 |
| `train/train_v5_fair.sh` | — | — | --ckpt --ckpt-dir --config --device --layers --num-shards --shard | — | — | — |
| `train/wolang_full.sh` | — | — | --config --device --layers --num-shards --shard | — | — | — |

## 6. 行为常量与已知冲突(26 项，人工整理;实现时复核行号)

| id · 位置 | 内容 | 分类 |
|---|---|---|
| `env_fov` · `carla_uav_tracking/rollout/env.py` | camera fov_deg=90 (run_rollout does not pass it) | scientific |
| `model_image` · `train/config_v5.yaml` | image.fov_deg 90 / width,height 512 / vehicle_size_m 4.0 / box_min_px 10 / box_max_px 140 → policy candidates | scientific |
| `cfg_v5_shared_dirs` · `train/config_v5.yaml` | context_cache, stage2_context_cache, ckpt_dir, stage2_ckpt_dir SHARED with config_v5_fov90.yaml | offline (conflict) |
| `cfg_v5_split_missing` · `train/config_v5.yaml` | split_manifest → missing file → dataset.py naive slice, test split empty | offline (conflict) |
| `rollout_fps_res` · `carla_uav_tracking/rollout/run_rollout.py` | fps=10, resolution=(512,512); scorer gets fps + max_lost_s only (commit_k NOT bound to --commit-k) | scientific |
| `max_lost_18` · `carla_uav_tracking/config/default.yaml` | environment.max_lost_seconds 18 (scorer truncation; RolloutScorer default 5) | scientific |
| `scorer_defaults` · `carla_uav_tracking/rollout/metrics.py` | RolloutScorer commit_k=5, t_fail 2 s, central_frac 0.3 | scientific |
| `opening_hires_ticks` · `carla_uav_tracking/rollout/env.py` | opening hi-res frames at steps 0/6/12; hi-res sensor destroyed at step>=12 | scientific |
| `warmup_stationary` · `carla_uav_tracking/rollout/env.py` | expert warm-up 90 ticks; abort 'target stationary' if max speed < 1.2 m/s; spawn failure → skipped ('setup: …') | scientific |
| `shuffle_seed` · `carla_uav_tracking/rollout/policy.py` | shuffle_seed=0 (not exposed): candidate-shuffle RNG re-seeded at every reset() | scientific |
| `tah_lang_reanchor` · `carla_uav_tracking/rollout/policy.py` | TAH lang re-anchor threshold 0.60 (not conf_tau) | scientific |
| `dam4sam_cooldown` · `carla_uav_tracking/rollout/policy.py` | DAM4SAM re-init cooldown 3 ticks | scientific |
| `crop_dino` · `carla_uav_tracking/rollout/policy.py` | crop-DINO vit_small_patch14_dinov2.lvd142m 224 px / pad 1.3; reid gallery momentum 0.9; pos depth/100 | scientific |
| `eval_asd_fixed` · `carla_uav_tracking/rollout/eval_asd.sh` | fixed: NET=cyh-carla-net, CONFIG=train/config_v5.yaml, --ckpt runs/stage2_v5/stage2_best.pt, --language-mode neutral, --ex-source track, --port 5555, image acot-uav-train:cu118, `pip install timm==1.0.29` at container start | scientific / infra |
| `launch_wd_infra` · `runs/asd/iter/launch_wd.sh` | ADAPTER map CARLA port → -graphicsadapter; TM port = tm + 10·(part−1); 5 restarts max; merges *_p<k>.json[.partial] | infra |
| `tah_cache_meta` · `train/tah_cache.py` | DINO crop 224/1.3, depth/100, stride 2 (0.2 s); saves only d_feat/order/fov | offline |
| `tah_train_compat` · `train/tah_train.py` | compat check = d_feat only; val = slice of the FIRST cache | offline |
| `backbone_kv_skip` · `train/backbone_kv.py` | skips any existing <stem>.npz without config/hash check | offline |
| `attr_ckpt_paths` · `carla_uav_tracking/rollout/opening_zoom.py` | AttrScorer: lin ckpt default eval_out/opening_attr_cls.pt, ft ckpt HARD-CODED eval_out/opening_attr_ft.pt; DINO scales from ckpt + flip TTA | scientific |
| `tick_align_order` · `carla_uav_tracking/rollout/tick_align.py` | 1 tick = 0.6 s (assumes S2=6); logs → seeds by ORDER of non-skipped episodes | analysis |
| `opening_cl_partial` · `carla_uav_tracking/rollout/opening_cl.py` | sorted(glob('*.json*'))[-1] → prefers .json.partial over .json; _p[0-9] globs miss part ≥10 | analysis |
| `iter_gate_precedence` · `carla_uav_tracking/rollout/iter_gate.py` | skipped/unscored dropped; first occurrence per seed wins; final > partial per part | analysis |
| `postprocess_box_min` · `carla_uav_tracking/recording/postprocess.py` | box_min 8 px (config 10) | datagen |
| `labels_336` · `train/labels.py` | image size 336 (stale) | offline |
| `cyh_carla_python` · `cyh-carla container` | `python` = 3.8.0 (CARLA client), `python3` = 3.6.9; mounts ONLY carla_uav_tracking → /workspace and carla_data → /data | infra |
| `dam4sam_single_client` · `dam4sam-svc container` | single-client sequential accept loop (dam4sam_service.py:55-58); currently started by hand | infra |
