# 1004 · 代码框架设计：离线训练与闭环验证的统一入口

> 版本:v0(2026-10-04,提交 Codex 审核前的初稿)。
> 目的：让数据生成、缓存、离线训练、闭环验证、分析都只有一个入口;不同方法和不同实验臂只在**配置**上不同，从而杜绝「配置不一致」。
> 依据：两份只读摸底(离线训练约 45 个入口;闭环每个臂由 36 个 flag、约 14 个环境变量、3 个 JSON 字符串和 shell 默认值拼出来)。本文只做设计，不改算法。

---

## 0. 目标与非目标

**目标**
1. **每个阶段只有一个入口**:`python -m acot {data,prep,train,eval,analyze,lint,show} <spec.yaml> [key=value ...]`。
2. **方法和实验臂都只是配置，不是脚本。** 同一个评测协议只定义一次，并带版本号;所有臂引用同一个协议对象，差异只能出现在方法配置或显式覆盖里。
3. **天然留痕**:每个产物(数据集、缓存、checkpoint、闭环结果、分析报告)都附带完整的已解析配置、配置哈希、输入哈希、git commit、未提交改动摘要、镜像 digest。
4. **入口处做兼容性检查**:相机 FOV、分辨率、tick 间隔、裁剪和特征提取器 ID 等，在 checkpoint 和协议、数据集之间必须一致，否则报错。正是这类检查的缺失，导致了 70°/90° 错位和 0.2 s/0.6 s 采样间隔不一致。
5. **显式隔离真值通道。** 协议要声明它允许的「理想化感知」;方法如果用了其他真值通道，自动标为 diagnostic,分析时不能和可部署臂放进同一张主表。
6. **种子集统一登记。** 训练、开发、留出、测试的种子集登记在同一处，重叠时报错;test 种子要加锁才能用。
7. **结果按(实验, 臂, 种子)存放。** 支持断点续跑和合并，tick 日志不再靠顺序对齐。
8. **迁移过程中不悄悄改变现有行为**:先用适配器驱动现有代码，靠一致性测试(parity)验证后，再重构内部。

**非目标**
- 不改 TAHRelM、PGIF、属性开局、Stage-2 等算法。
- 不重写 CARLA 环境。
- 不接管另一个会话的 v6 数据管线，只提供接口。

---

## 1. 现状问题(摘要，证据见两份摸底)

| # | 问题 | 例证 |
|---|---|---|
| 1 | 没有统一的协议对象 | 一个臂的配置 = `policy_server` 36 个 flag + 约 14 个环境变量(`ASD_LOG`、`TRK_CFG`、`OPENING*`、`COLD_*`、`ORACLE_ABSENT`、`TAH_DAGGER_DIR`…)+ `ASD_CFG`/`TRK_CFG`/`OPENING_CFG` 三个 JSON 字符串 + `eval_asd.sh` 默认值 + `run_rollout` flag。优先级是隐式的,`--asd-cfg` 会覆盖 `ASD_CFG` |
| 2 | 默认值不安全 | 服务端默认值是旧的 oracle 设定(`--gallery-seed gt`、`--ex-source ear`、`--s2-period 1`、`--tah-ckpt tah_rel.pt`),ASD 默认 `filter_kind=bayes`(本是消融)。`launch_f.sh` 传了空的 `ASD_CFG`,结果 `paper_asd_tah_filter` 实际跑的是 bayes 消融 |
| 3 | 结果没有留痕 | 结果 JSON 只有 episodes,不记录 head、ASD、开局、跟踪器配置、ckpt、town、git;policy 打印的配置随容器一起删掉。训练侧也没有任何 git hash;`stage2_best.pt`、`tah_relm_dag.pt` 没有记录配方 |
| 4 | 启动脚本靠复制粘贴 | 19 个独立的 `docker run` 块，外加约 40 个 chain 脚本。同一串最终协议 flag 原样出现在 9 个文件里，`TRK_CFG` 出现在 14 个，tempered JSON 出现在 27 个。协议的代次(V0 → deploy_v2 → final)只能从用了哪个脚本看出来 |
| 5 | 相机和几何参数没有唯一来源 | FOV、分辨率、框尺寸限制写在 YAML 里，另有约 6 处硬编码副本，没有一处和 h5 attrs 核对。projection、pseudo-box、crop、describe、DINO 加载各有 2–7 份实现，参数也不一致(如框最小值 8 px 和 10 px、crop 的 pad 和重采样方式) |
| 6 | 划分是隐式的 | v5 配置指向不存在的 `__none_v5_naive__.json`,于是 test 划分为空;运行一次 `make_split` 就会悄悄改掉所有 v5 下游的划分。TAH、DAgger、开局分类器、闭环各用自己的划分;TAH 的 val 其实是数据集 train 划分里的一段 |
| 7 | 配置互相冲突、已经过期 | 20 个 YAML 中有 15 个指向已删除的数据;`config_v5_fov90.yaml` 和 `config_v5.yaml` 共用输出目录，用它跑 stage2 会覆盖正在部署的 ckpt |
| 8 | 采样间隔不一致 | TAH 离线缓存每 2 帧取一帧(0.2 s),闭环和 DAgger 每个 S2 tick 是 0.6 s,TAHRelM 的速度和运动门特征尺度差 3 倍 |
| 9 | 分析脚本绑死文件名 | 臂名、REF 字典、特例和单位数 part 的 glob 都写死在代码里;tick 日志按顺序对齐的逻辑有 4 份副本;`.partial` 和 `.json` 谁优先，各脚本不一致;「correct」有两种定义(1 − inst 和 1 − sustained);bootstrap 有 8 份实现，RNG 种子和重抽样次数 B 各不相同 |
| 10 | 评分器和 policy 脱节 | 评分器的 `commit_k` 没有和 policy 的 `--commit-k` 绑定;`_cand_lite` 不传 `gt_idx`,跟踪器 ID 模式下干扰车按跟踪器 ID 计 |

---

## 2. 设计原则

1. **单一事实来源**:所有影响行为的量都来自已解析的配置对象。代码里只允许出现 schema 中声明为 `safe_default` 的默认值，其余字段必须显式给出。
2. **严格 schema**:配置里出现未知字段就报错;字段类型和取值范围都要校验。
3. **组合，而不是复制**:`extends:` 继承加深合并(deep merge)。实验臂只写自己和基准方法的差异;覆盖记录写进 manifest,并纳入哈希。
4. **产物按内容寻址**:同样的配置和同样的输入一定得到同一个产物 ID,不再静默复用过期缓存。
5. **先适配、后重构(strangler 模式)**:P0/P1 由翻译器把已解析配置转成现有代码需要的 flag 和环境变量;做完一致性测试，再让内部代码直接读取配置对象。
6. **CARLA 侧兼容 Python 3.8**:配置包只依赖 PyYAML 和标准库，CARLA 客户端容器(py3.8)和 policy 容器(py3.11)共用同一份配置包。

---

## 3. 包结构与 CLI

```
acot/                         新包(仓库根目录;只依赖标准库 + PyYAML;兼容 py3.8)
  cli.py / __main__.py        python -m acot <stage> <spec> [overrides]
  config/
    schema.py                 dataclass:Protocol / Method / SeedRegistry / DatasetSpec / PrepSpec / Recipe / Experiment / Infra
    load.py                   YAML 读取、extends 组合、深合并、key=value 覆盖、严格校验
    resolve.py                解析引用(protocol@ver、method、seedset、artefact:<id>)、兼容性检查、真值通道计算、规范化哈希
    legacy.py                 翻译器：已解析配置 → 现有 policy_server / run_rollout 的 flag + 环境变量(P0–P1 用)
  registry.py                 注册表:methods / heads / trainers / prep / metrics / collectors
  provenance.py               git sha、未提交 diff 的哈希与文件清单、镜像 digest、主机/GPU、时间、输入产物哈希 → manifest.json
  artefacts.py                产物库 runs/artefacts/<kind>/<name>@<hash12>/{manifest.json, ...};别名表;旧产物导入(标注缺失的留痕)
  data/
    dataset.py                数据集清单(每个文件的 sha256、attrs 中的 fov/分辨率/seed/town)、强制划分文件
    camera.py                 唯一的投影和 pseudo-box 实现(相机参数来自数据集 attrs 或协议)
    crops.py                  唯一的 crop 和特征提取器(带 ID,如 dino_s14_cls384_c224_p1.3、dino_s14_clsmean768_s1.5-2.0)
    language.py               唯一的 describe() 和颜色词表
  prep/                       缓存构建：vlm_ctx(backbone_kv)、tah_seq(tah_cache)、opening_snapshots
  train/                      训练器适配：tah_relm、opening_attr_lin、opening_attr_ft;stage2、grounding、lora 先以 legacy 方式导入
  eval/
    protocol.py / method.py   配置到运行参数的映射
    policy_entry.py           policy 服务入口:--resolved <path>(P2 起);P0/P1 用 legacy.py 的翻译结果启动现有 policy_server
    episode.py                CARLA 侧单个 episode 的执行入口(包装 run_rollout/env),按种子写记录
    scheduler.py              (臂, 种子)工作队列 + CARLA 工作池 + 看门狗 / 重试 / 跳过 + 容器生命周期
    collectors.py             采集器(tah_dagger、opening_snapshots、actor_labels、dam4sam_diag),由实验配置声明，不再用环境变量
  analysis/
    results.py                按(臂, 种子)读取结果;旧结果导入器(带协议标签)
    metrics.py                指标注册表：committed / lenient / strict / longC / longW / opening_lock / ...(唯一定义)
    stats.py                  配对 bootstrap、符号翻转检验、McNemar、Wilson、Holm(统一的 RNG 与 B 设定)
    report.py                 根据实验配置中的 comparisons 生成表格和图
configs/
  protocols/   deploy_v1@1.yaml(只用于复现旧数字)、deploy_v2@1.yaml(当前)
  methods/     ours_final、ours_no_pgif、pgif_bayes、pgif_resid、pgif_nogate、ext_dam4sam、ext_deepsort、ext_ocsort、ext_reid、lang_xattn、diag_oracle_id、diag_oracle_absent …
  seeds.yaml   唯一的种子登记表
  datasets/    mvp_full_v5_fov90.yaml、mvp_full_v5_70.yaml(legacy)…(含划分文件路径)
  prep/        tah_seq_v90.yaml …
  recipes/     tah_relm.yaml、opening_attr_lin.yaml、opening_attr_ft.yaml、stage2_v5.yaml(冻结，只导入)
  experiments/ 1004_dev_gate_*.yaml、1004_table{1..5}.yaml …
  infra/       workers.yaml(CARLA 服务器的端口、graphicsadapter、policy GPU、TM 端口基数;DAM4SAM 服务;镜像;为另一会话保留的端口段)
legacy/        迁移后的旧启动脚本和分析脚本(只读，仍可运行)
tests/         单元测试 + 一致性测试
```

**CLI**

| 命令 | 输入 | 产出 |
|---|---|---|
| `acot data gen datasets/<ds>.yaml` | 数据集配置(种子 → 城镇的显式清单、场景、相机) | 数据集目录 + `dataset_manifest.json` |
| `acot prep tah_seq prep/<x>.yaml` | 数据集清单 + 特征提取器 ID + 步长 | 产物 `tah_seq@<hash>` |
| `acot train recipes/<x>.yaml seed=0` | 配方(输入产物 ID、超参、划分、选择规则) | 产物 `ckpt@<hash>`(best、last、metrics、manifest) |
| `acot eval experiments/<x>.yaml [--arms ..] [--seeds ..]` | 实验(协议、种子集、臂、采集器、重复次数) | `runs/exp/<x>/<arm>/...` |
| `acot analyze experiments/<x>.yaml` | 同一实验配置中的 comparisons | `runs/exp/<x>/report.{md,json}` |
| `acot lint` / `acot show <spec>` | — | 校验全部配置;打印已解析配置、哈希和真值通道 |

---

## 4. 配置 schema(核心字段)

### 4.1 Protocol(带版本，冻结后不可修改，修改即升版本)

```yaml
# configs/protocols/deploy_v2@1.yaml
name: deploy_v2
version: 1
camera: {fov_deg: 90, width: 512, height: 512}
sim:
  fps: 10
  s2_period: 6                     # 1 tick = 0.6 s;开局 tick 和分析中的 t≥12 都由它推导
  episode_steps: 900
  towns: [Town05]
  scenario: mvp
  max_lost_s: 18                   # 目标连续出画超过 18 s 结束 episode(当前 default.yaml 的值)
  warmup: {expert_ticks: 90, min_target_speed: 1.2}
candidates:
  source: sim_projection           # 仿真检测替身
  set: task_vehicles               # 只含目标和场景指定的干扰车(背景车流不作为候选)
  occlusion: raycast
  identity_keys: tracker           # tracker | sim_ids(后者只用于 deploy_v1)
  tracker: {max_age: 8, w_app: 0.85, gate_px: 150, gate_box: 2.0, min_cos: 0.2}
  noise: none
frontend:                          # 所有臂共享的感知前端
  vlm: {stage2_ckpt: "artefact:stage2_v5", model_config: train/config_v5.yaml, language_mode: neutral}
  features: {id: dino_s14_cls384_c224_p1.3, depth_scale: 100}
  opening: {module: attr, window: 3, ft: 0.5, hires_k: 3,
            ckpts: {lin: "artefact:opening_attr_lin", ft: "artefact:opening_attr_ft"}}
control: {ex_source: track, commit_k: 5, frame_gain: 0, where_localisation: gt_matched_box}
scorer: {commit_k: 5, t_fail_s: 2.0, central_frac: 0.3, primary_correct: committed}
privileged_perception:             # 本协议明确允许的理想化输入(对所有臂相同)
  [cand_box_gt_projection, cand_depth_gt, occlusion_raycast_gt, task_vehicle_set, where_world_gt_matched_box]
```

### 4.2 Method(实验臂的方法部分)

```yaml
# configs/methods/ours_final.yaml
name: ours_final
identity:
  head: tah
  ckpt: "artefact:tah_relm_dag"     # 旧产物导入后的别名，manifest 中标注 provenance_gaps
  memory_seed: committed           # committed | gt(gt 属于真值通道)
  conf_tau_commit: 0.5             # 拆开 --conf-tau 的两种含义：提交门控
  reanchor: {mode: off}
filter: {kind: pgif, filter_kind: tempered, resid_decay: 0.98, conf_window: 30,
         absent_hold: true, absent_max: 8, gate_memory: true}
diagnostics: []                    # 非空即为 diagnostic 臂
```

```yaml
# configs/methods/ext_dam4sam.yaml
name: ext_dam4sam
identity: {head: dam4sam, gate_px: 160, conf_tau_commit: 0.05, update: s2, reinit_cooldown: 3, memory_seed: committed}
filter: {kind: none}
diagnostics: []
```

开局模块属于协议的 `frontend`,默认所有臂共享。只有在实验配置里显式标为 `ablation: opening` 的臂，才允许覆盖 `frontend.opening`;其他前端字段在臂内覆盖会报错。

### 4.3 种子登记(`configs/seeds.yaml`)

```yaml
seedsets:
  datagen_mvp_v5:   {ranges: [[42, 108]],     purpose: train_data}
  train_opening:    {ranges: [[96000, 96280]], purpose: train}
  train_dagger_v1:  {ranges: [[92000, 92024]], purpose: train}     # 以 collect_tahdag 的实际种子为准，迁移时核对
  train_dagger_v2:  {ranges: [[97000, 97105]], purpose: train}
  dev:              {ranges: [[91000, 91020]], purpose: dev}
  dev2:             {ranges: [[93000, 93100]], purpose: dev}
  heldout_opening:  {ranges: [[93100, 93660]], purpose: heldout}
  test:             {ranges: [[95000, 96000]], purpose: test, locked: true}
rules: {disjoint_across_purposes: true, test_requires: [--allow-test, prereg_ref]}
```

不同用途之间的种子重叠时，`acot lint` 报错。使用 `test` 必须加 `--allow-test`,并在实验配置里引用对应的预注册文档版本号。

### 4.4 Experiment

```yaml
# configs/experiments/1004_dev_gate_tahv90.yaml
name: 1004_dev_gate_tahv90
protocol: deploy_v2@1
seeds: dev
arms:
  v1_old:   {method: ours_final}
  v90_best: {method: ours_final, override: {identity.ckpt: "artefact:tah_relm_v90_s0"}}
  v90_last: {method: ours_final, override: {identity.ckpt: "artefact:tah_relm_v90_s0_last"}}
replicates: 1
collect: [ticks, trk]
comparisons:
  - {family: ckpt, a: v90_best, b: v1_old, metrics: [correct, longC, longW, correct_s], test: paired_bootstrap}
  - {family: ckpt, a: v90_last, b: v1_old, metrics: [correct, longC, longW, correct_s], test: paired_bootstrap}
```

### 4.5 DatasetSpec、PrepSpec、Recipe

- **DatasetSpec**:路径、`expected_camera`(必须与每个 h5 的 attrs 一致)、划分文件(必须存在，带内容哈希)、生成信息(种子 → 城镇清单、场景配置快照)。
- **PrepSpec**(例如 tah_seq):数据集 ID、`stride`、特征提取器 ID、`depth_scale`。产物 manifest 会记录推导出的 `tick_s` 和相机参数。
- **Recipe**:训练器名、输入产物 ID 列表(顺序有意义)、验证划分规则、超参、`seed`、`select: best|last|both`、`compat_targets`(声明这个 ckpt 准备在哪些协议下使用)。

---

## 5. 旧配置通道到新字段的完整映射(收编清单)

| 旧通道 | 新字段 |
|---|---|
| `--config` / `--ckpt` / `--language-mode` | `protocol.frontend.vlm.*` |
| `--s2-period` / `--ex-source` / `--commit-k` / `--frame-gain` | `protocol.sim.s2_period` / `protocol.control.*`;**`scorer.commit_k` 必须等于 `control.commit_k`,除非显式声明解耦** |
| `--tid-head` / `--tid-ckpt` / `--tah-ckpt` | `method.identity.head` / `.ckpt` |
| `--gallery-seed` | `method.identity.memory_seed`(`gt` 即真值通道) |
| `--conf-tau` | 拆成 `identity.conf_tau_commit`(提交门控)和 `identity.reid.reanchor_conf_tau`。翻译器在 P0/P1 把两者映射回同一个 flag,并校验二者相等 |
| `--reid-*` / `--reanchor*` / `--conf-consensus` | `method.identity.reid.*` / `.reanchor.*` |
| `--assoc-*` / `--dam4sam-*` | `method.identity.{assoc,dam4sam}.*`;服务地址放在 `infra` |
| `--warmup-oracle-s` / `--event-override` / `--retarget-tick` | `method.diagnostics` / `protocol.sim.retarget_tick`(只用于动态换目标实验) |
| `--asd` / `--asd-cfg` / `ASD_CFG` | `method.filter.*`(`kind` 等字段必须显式写出) |
| `--tracklets` / `--occl-cands` / `--vis-raycast` / `TRK_CFG` | `protocol.candidates.*` |
| `--hires` / `OPENING` / `OPENING_CFG` | `protocol.frontend.opening.*`(开局消融臂可覆盖) |
| `--steps` / `--town` / `--episodes` / `--seed-base` | `protocol.sim.*` / 实验的种子集(由调度器分配) |
| `--expert-control` / `--predict-road` / `--road-oracle` | `method.diagnostics` / `method.where.*` |
| `COLD_*` / `V1_SELFRESET` | `method.identity.cold.*` / `method.filter.v1_selfreset`(legacy) |
| `DAM4SAM_EVERY_FRAME` / `DAM4SAM_DIAG` | `method.identity.dam4sam.update` / 实验的 `collect: [dam4sam_diag]` |
| `ORACLE_ABSENT` | `method.diagnostics: [oracle_absent{mode: abstain\|hygiene}]` |
| `TAH_DAGGER_DIR` / `DAGGER_DIR` / `OPENING_DUMP` / `DUMP_LABELS` | 实验的 `collect: [tah_dagger, opening_snapshots, actor_labels]` |
| `ASD_LOG` / `HOSTNAME` 命名 | 结果布局 `ticks/<seed>.jsonl`、`trk/<seed>.jsonl` |
| `CARLA_PORT` / `CARLA_TM_PORT` / `--port` / GPU / `NTHR` | `infra`(不计入科学配置的哈希) |
| `env.py` 的 `fov_deg=90`、`config_v5.image.fov_deg` | `protocol.camera.fov_deg`。env、候选、policy 都从这里读,`resolve` 检查 `model_config` 与之相等 |
| 评分器的 `commit_k` / `t_fail` / `central_frac`,`max_lost_seconds` | `protocol.scorer.*` / `protocol.sim.max_lost_s` |
| 硬编码的 crop 224/1.3、depth/100 | `protocol.frontend.features`,并与 prep 产物的特征 ID 做兼容性检查 |
| TAH lang re-anchor 的 0.60 阈值、DAM4SAM 冷却 3 tick | `method.identity.*` |
| 开局 tick 0/6/12、1 tick = 0.6 s | 由 `s2_period` 和 `opening.window` 推导;分析一侧从结果 manifest 读取 |

---

## 6. 真值通道隔离

- **协议声明的理想化感知**:写在 `privileged_perception` 里，对所有臂相同。论文中要如实披露，标为「部署设定 v2 下可部署」。
- **diagnostic 通道**:`tid_head=oracle`、`memory_seed: gt`、`reanchor: oracle`、`warmup_oracle`、`event_override`、`oracle_absent`、`attrbind`(用到目标的真值属性)、`ex_source != track`、`road_oracle`、`expert_control`。
  - 只要用了其中任何一个，resolver 就会把它写进 `uses_gt: [...]`,并把这个臂标为 diagnostic。
  - 在 `report.py` 里，diagnostic 臂只能放在「上界 / 诊断」分组。
- **只用于记录的通道**(tick 日志里的 `true`、DAgger 标签、开局快照标签):P3 起，这些真值放在独立的 `ScoringInfo` 对象里，只传给日志器和评分器，policy 拿不到，从代码结构上杜绝误用。

---

## 7. 闭环运行时

- **解析**:`acot eval` 先解析实验里每个臂，得到 `resolved.yaml` 和 `config_hash`,然后做兼容性检查(见 §8)。
- **调度**:由 `scheduler.py` 统一调度，取代 `launch_wd.sh`、`eval_asd.sh` 和各种 chain 脚本。
  - 工作池来自 `infra/workers.yaml`,包括端口、graphicsadapter、policy GPU 和 TM 端口。TM 端口由调度器统一分配，不会冲突;为另一会话保留的端口段(2100–2190)不会被使用。
  - 工作单位是「臂 × 连续种子块」。每个 worker 为一个臂起一个 policy 容器，名为 `acot-<exp>-<arm>-<worker>`,依次跑完分到的种子。
- **结果按种子写入**:
  - `episodes/<seed>.json`:评分器输出，额外带上 arm、config_hash、protocol、seed、git、镜像 digest;
  - `ticks/<seed>.jsonl`:每条 tick 记录都带 seed。这要求 `reset` 消息携带种子，1004 的共享开局本来也需要这个能力。
- **续跑与跳过**:已有最终记录的种子直接跳过。同一种子连续失败 2 次，就记为 `skipped` 并写明原因(例如目标生成失败)。卡死检测和 CARLA 重启沿用 `launch_wd` 的逻辑，但 adapter 映射只从 `infra` 读取一处。
- **配置回显与核对**:policy 启动时打印已解析配置的哈希，调度器核对它与预期一致，不一致就中止。policy 的完整日志保存在 `logs/` 下。
- **采集器**(`tah_dagger`、`opening_snapshots`、`actor_labels`、`dam4sam_diag`):由实验配置声明，输出同样按种子落盘到 `runs/exp/<x>/<arm>/collect/<name>/<seed>.*`。
- **评分器修正**:`_cand_lite` 补传 `gt_idx`(P2)。

---

## 8. 离线训练

- **数据集清单**:用 `acot data manifest datasets/<ds>.yaml` 生成，内容包括每个文件的 sha256 和 attrs(fov、分辨率、seed、town)。
  - 如果某个文件的 FOV 和 `expected_camera` 不一致，直接报错。像现在 `mvp_full_v5` 这样「图像是 70° 但没有 fov attr」的数据集，必须在清单中显式写 `camera_override: {fov_deg: 70, evidence: runs/data_audit/fov_check_5ep}`。
  - 划分文件必须存在;不存在就报错，并给出生成命令，但不会自动生成。
- **缓存是产物**:`acot prep tah_seq` 的输出 ID 由(数据集清单哈希、步长、特征 ID、代码版本)共同决定，不会再静默复用旧缓存。合并 DAgger 数据时保留 `order`、`fov`、`tick_s` 等元数据。
- **训练**:`acot train recipes/tah_relm.yaml seed=0` 调用现有的 `tah_train` 函数(适配器)。产物包括 best/last ckpt、指标、manifest(配方、输入产物 ID、划分、选择规则、git),以及 `compat`(`fov_deg`、`tick_s`、特征 ID)。
- **兼容性检查**:在闭环解析阶段，用 ckpt 的 `compat` 对照协议:
  - `fov_deg` 必须相同;
  - 特征 ID 必须相同;
  - `tick_s` 必须相同。如果配方里有意混用不同 tick 间隔(例如复现 `tah_relm_dag` 时的 0.2 s 离线数据加 0.6 s DAgger),需要在配方里写明 `waivers: [{field: tick_s, reason: ...}]`,这一豁免会写进 manifest,并在报告中标注。
- **旧产物导入**:`acot artefacts import` 为 `stage2_best.pt`、`tah_relm_dag.pt`、`opening_attr_*.pt` 和 v90 的 ckpt 建立产物条目，并写明 `provenance_gaps`(例如「无配方」「源数据已删除」「离线部分 FOV 错位」)。
  - 原文件保留在原处，产物库里只放引用和哈希。
  - 开局 ckpt 从 `rollout/eval_out/` 迁到产物库，`opening_zoom.py` 改为按产物 ID 加载。
- **去重**:projection、pseudo-box、crop、describe、DINO 特征提取统一放到 `acot/data`,各脚本改为调用这一份实现。每个特征提取器都有 ID,训练端和部署端靠这个 ID 对齐。
- **YAML 清理**:指向已删除数据的配置移到 `legacy/configs/`。`config_v5_fov90.yaml` 改为 `extends: config_v5`,只覆盖数据路径，并且使用独立的输出目录(当前的冲突必须先修掉)。

---

## 9. 分析

- **统一入口**:`acot analyze` 只读实验配置，按(臂, 种子)配对，不再写死臂名。
- **指标统一定义**,集中在 `analysis/metrics.py`:
  - `correct`:承诺口径，即 1 − `mis_follow_sustained`,为主口径;
  - `correct_l`(宽松)和 `correct_s`(严格)作为辅助口径;
  - `longC`、`longW` 按评分器逐帧计算;
  - `opening_lock`:窗口结束后首个满足 t ≥ `opening_end_tick` 的 tick,由协议推导。
- **统计设定统一**:B = 20000,RNG 种子 = 实验名的哈希;配对 bootstrap、符号翻转检验、McNemar、Wilson 区间各只保留一份实现;同一 `family` 内的多重比较做 Holm 校正。
- **旧结果导入**:`acot results import-legacy` 为文档中引用过的旧臂建立索引，标注 `protocol_tag ∈ {gt_seeded, deploy_v1, deploy_v2_head, final_v2}`,并带上来源脚本。对应关系由人工维护在 `legacy/arm_map.yaml` 中，这样旧数字可以查询，但不会和新协议的结果混在一起。

---

## 10. 迁移计划(分阶段，每阶段有验收标准)

| 阶段 | 内容 | 验收 | 估时 |
|---|---|---|---|
| **P−1 基线** | 两个会话的未提交改动先提交到一个基线分支(需用户同意);定好文件所有权(`env.py`、`policy.py`、`scene_manager.py` 有两方在改) | 有可引用的基线 git sha | 0.5 h |
| **P0 配置 + 翻译器** | schema、load、resolve、种子登记、协议和方法 YAML(覆盖 `ours_final`、H-A、开局变体、各外部基线、各 diagnostic 臂);`legacy.py` 翻译器;`acot show` / `acot lint` | **黄金测试**:对每个现有的最终协议启动脚本(`chains_attr.sh`、`f_v4.sh`、`collect_dagger.sh`、`pipeline_v3.sh`、`chains_v3.sh`),翻译器输出的 flag 和环境变量与脚本里实际传入的完全一致;lint 能检出 FOV 冲突和种子重叠 | 1 天 |
| **P1 调度 + 结果 + 分析核心** | `scheduler.py`(仍通过翻译器调用现有的 `eval_asd` 路径)、按种子存放结果、manifest、`analysis` 核心库;旧结果导入器;`ticks` 中带 seed(只改 reset 消息) | **冒烟一致性**:用新入口在 dev 的 2 个种子上跑 `ours_final` 和 H-A,与旧启动方式对比，已解析配置和实际调用完全一致，tick 日志首个决策相同(CARLA 存在不确定性，不要求整条轨迹一致);`acot analyze` 复现 `arch_compare` 和 `opening_ds` 的现有数字(误差 1e-9 以内) | 1.5 天 |
| **P2 policy/env 直接读配置** | `policy_server --resolved`、`run_rollout --protocol`;不再使用行为环境变量(设置了就报错);去掉不安全的默认值;评分器修正(`gt_idx`、`commit_k` 绑定) | 同 P1 的冒烟一致性;单元测试覆盖「漏传字段就报错」 | 1 天 |
| **P3 离线统一** | 数据集清单和划分文件、`acot/data` 去重、prep/train 适配器、产物库、兼容性检查、开局 ckpt 迁移 | 用新入口重建 tah_seq,在 2 集上与现有缓存的张量哈希一致;tah_train 在相同种子下指标误差 ≤ 1e-6(GPU 不保证逐位一致时，报告实际差异);兼容性检查能拦下「70° 数据按 90° 投影」 | 2 天 |
| **P4 清理** | 旧启动脚本和分析脚本移到 `legacy/`;按组件拆分 `policy.py`(开局 / 身份 head / 滤波 / WHERE),由一致性测试保障 | 全部测试通过;文档更新 | 视情况而定 |

合计约 5–6 个工作日，不含 P4。每个阶段做完都可以单独使用，不会出现「做一半就跑不了」的状态。

---

## 11. 测试

- **单元测试**:schema 严格性(未知字段、缺少必填字段)、`extends` 和覆盖的合并语义、哈希稳定性(字段顺序和空白不影响哈希)、翻译器黄金测试、种子不重叠、兼容性检查、真值通道判定、指标与统计和现有实现的一致性。
- **一致性测试**:见 §10 各阶段的验收标准。
- **CI**:仓库没有 CI,用 `python -m acot lint && python -m pytest tests/ -q` 作为提交前检查(手动或 git hook)。

---

## 12. 风险与协调

| 风险 | 应对 |
|---|---|
| 两个会话同时修改 `env.py`、`policy.py`、`scene_manager.py` | P−1 先定所有权;P2/P4 中涉及这几个文件的修改，需与另一会话协调后在独立分支上进行 |
| CARLA 不确定(例如红绿灯相位依赖运行历史) | 一致性以「配置和调用完全一致」为主，轨迹层面只要求首个决策一致和冒烟通过 |
| 过度工程 | 只保留必要的抽象;P0–P1 不改动任何算法代码 |
| py3.8 兼容 | `acot/config`、`acot/eval/episode.py`、`acot/analysis/results.py` 只用 3.8 语法和标准库;测试在两个容器里各跑一遍 |
| 历史数字无法完全复现(源数据已删、缓存被覆盖) | 旧产物导入时如实写明 `provenance_gaps`;不追求复现 `stage2_v5` 的训练过程 |

---

## 13. 需要用户决定的事项

1. **P−1**:是否同意先把当前未提交的改动(包括另一会话的)提交到基线分支。
2. **开局模块的归属**:放在协议的 `frontend`,即所有臂默认共享(本稿采用这种做法);还是作为方法的一部分。
3. **test 加锁**:使用 test 种子是否强制要求 `--allow-test` 并引用预注册文档。
4. **实施顺序**:是否先只做 P0–P1(约 2.5 天)就开始 1004 的重跑，P2–P3 并行推进。
