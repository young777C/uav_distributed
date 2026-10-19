# 1004 · 代码框架设计：离线训练与闭环验证的统一入口

> **版本**:v1(2026-10-04),按 Codex R1 意见修订。R1 结论为 MAJOR-REVISIONS,共 6 条 major、4 条 minor,原文见 `acot_note/codex/1004_code_framework_R1.md`;v0 原稿存于 `acot_note/codex/1004_code_framework_v0.md`。逐条对照见附录 A。
> **配套文件**:
> - `acot_note/1004_legacy_surface.md`:旧配置面全清单，由 `tools/legacy_surface.py` 自动生成;
> - `tools/legacy_surface.py`:清单生成器，只依赖标准库。
>
> **范围**:只做设计，不改算法。实施按 §14 分阶段进行，每个阶段都有验收标准。

---

## 0. 一页摘要

**问题。** 现在一个闭环臂的实际配置散落在 6 个地方：
- 36 个 `policy_server` flag;
- 14 个 `run_rollout` flag;
- 16 个行为环境变量;
- 3 个 JSON 字符串(`ASD_CFG`、`TRK_CFG`、`OPENING_CFG`),展开后进入 4 个构造函数，共 43 个可配置字段;
- 启动脚本里写死的参数(73 个启动脚本);
- 代码里的硬编码常量。

离线侧也有对应的问题：缓存、划分和 checkpoint 都没有可校验的元数据。后果已经出现过多次：
- FOV 70°/90° 错位;
- 离线 0.2 s 与闭环 0.6 s 的 tick 间隔不一致;
- 空 `ASD_CFG` 让一个臂悄悄跑成了 bayes 消融;
- 两个 YAML 共用输出目录。

**方案。** 只保留一个入口:`python -m acot {data,prep,train,eval,analyze,lint,show}`。各类对象的角色如下：
- **协议**:带版本、冻结;
- **方法**:由严格 schema 校验的 YAML;
- **实验**:引用协议、方法和种子集;
- **基础设施**:单独成文件，不计入科学哈希。

迁移采用 strangler 模式：先让翻译器生成旧代码需要的参数，用一致性测试证明行为没变，再让内部直接读取配置对象。

**五个关键机制(对应 R1 的 6 条 major):**
1. **清单驱动的收编(M1)**:
   - 由脚本生成旧配置面全清单，共 118 个 flag、92 个构造/JSON 字段、24 个环境变量、16 个线协议字段、73 个启动脚本、25 个行为常量;
   - 每一行都必须在 `legacy_map.yaml` 里有去处，否则测试失败;
   - P2 起，未知的旧环境变量或 JSON 字段直接报错。
2. **机械化的真值隔离(M2)**:
   - P0 就把每个真值通道的分类固化下来;
   - P1 在线协议层把目标属性置空，并用调用点审计做「棘轮」:出现未经审核的新访问点就失败;
   - P2 把接口拆成 `PolicyInput` 和 `ScoringInfo`,方法代码在结构上拿不到真值字段。
3. **四级一致性验收(M3)**:
   - L0:调用参数逐字节一致(用 docker shim 捕获);
   - L1:对象级配置回显;
   - L2:旧结果导入后，指标与旧分析脚本逐位一致;
   - L3:实机冒烟，核对种子、场景指纹和跳过集合。

   轨迹层面允许 CARLA 的不确定性，但调用参数、种子和评分不允许漂移。
4. **全产物 manifest 与兼容矩阵(M4)**:覆盖 11 类离线产物，在训练和闭环解析两道边界上比较 fov、tick_s、特征 ID、depth 尺度、语言模式、划分哈希和数据集哈希。
5. **结果失败即关闭，基础设施可执行检查(M5、M6)**:
   - 结果按种子记录状态(complete / skipped / failed / partial),默认要求预注册的种子全部 complete;
   - 基础设施(网络、容器、GPU、端口、DAM4SAM 单连接服务、CARLA 客户端 Python 版本)在 preflight 中逐项检查。

**工期。** P0a–P3b 合计约 13.5 个工作日，不含可选的 P4。其中 P0a–P1b 约 7.5 天;做完这一段，1004 的重跑就能全部走新入口，内部仍是旧代码，但留痕完整。

---

## 1. 目标与非目标

**目标**
1. **入口唯一**:`python -m acot <stage> <spec.yaml> [key=value ...]`。各方法、各实验臂只在配置上不同。
2. **协议只定义一次**:带版本、冻结;所有臂引用同一个协议对象。
3. **天然留痕**:每个产物都附带以下信息：
   - 已解析配置和配置哈希;
   - 输入产物的 ID;
   - git commit 与未提交改动摘要;
   - 镜像 ID;
   - 基础设施快照。
4. **入口处做兼容性检查**:fov、tick_s、特征 ID、depth 尺度、语言模式、划分和数据集，任何一项不一致就报错。
5. **机械化隔离真值通道**:方法代码拿不到评分用的真值。凡是用了真值通道的臂，自动标为 diagnostic。
6. **种子统一登记**:用途之间不允许重叠;test 种子加锁;支持任意种子列表和替补种子。
7. **结果失败即关闭**:按(实验, 臂, 种子)存放，每个种子带状态;默认要求完整。
8. **迁移中不悄悄改变行为**:以 §9 的 L0–L3 一致性验收为准。

**非目标**
- 不改 TAHRelM、PGIF、属性开局、Stage-2 的算法。
- 不重写 CARLA 环境。
- 不接管另一会话的 v6 数据管线。

**措辞**:当前协议命名为 **`sim_proxy_v2`(仿真代理感知协议 v2)**,不再叫「可部署 v2」。
- 协议中仍由仿真提供的部分：候选框(GT 投影)、深度(GT)、遮挡(射线检测)、候选集(只含任务车辆)、所匹配框的世界坐标(用于 WHERE)。
- 论文中统一表述为「部署形态的闭环，感知由仿真代理提供」。
- `deploy_v2` 只作为旧结果标签的别名保留。

---

## 2. 现状问题(证据)

| # | 问题 | 例证 |
|---|---|---|
| 1 | 没有统一的协议对象 | 一个臂的行为由 flag、环境变量、JSON 字符串、shell 默认值、写死参数和代码常量共同决定(详见清单)。优先级是隐式的：`--asd-cfg` 非空时整体替换 `ASD_CFG`,两者不合并(`policy.py:243-245`) |
| 2 | 默认值不安全 | 服务端默认是旧的 oracle 设定:`--gallery-seed gt`、`--ex-source ear`、`--s2-period 1`、`--tah-ckpt runs/tah_rel.pt`。ASD 默认 `filter_kind=bayes`,而这本来是消融。`launch_f.sh` 传了空的 `ASD_CFG`,结果 `paper_asd_tah_filter` 实际跑的是 bayes |
| 3 | 结果没有留痕 | 结果 JSON 只有 episodes。policy 打印的配置随容器删除而丢失;训练侧也没有记录 git 和配方 |
| 4 | 启动脚本靠复制粘贴 | 共 73 个启动脚本(不含 ticker)。最终协议的那串 TAH flag 原样出现在 12 个脚本里,`TRK_CFG` 赋值出现在 13 个，tempered JSON 出现在 21 个 |
| 5 | 相机和几何参数没有唯一来源 | FOV、分辨率、框尺寸限制写在 YAML 里，另有约 6 处硬编码副本，没有一处和 h5 attrs 核对。投影、裁剪、`describe` 各有 2–7 份实现(框最小值有 8 px 和 10 px 两种) |
| 6 | 划分是隐式的 | `config_v5.yaml` 的 `split_manifest` 指向一个不存在的文件，于是 `dataset.py:25-39` 退回到朴素切片，test 划分为空 |
| 7 | 输出目录冲突 | `config_v5_fov90.yaml` 与 `config_v5.yaml` 共用 `context_cache`、`stage2_context_cache`、`ckpt_dir`、`stage2_ckpt_dir` 四个目录。`backbone_kv.py:63-72,106-116` 遇到已有文件直接跳过，不做任何校验 |
| 8 | tick 间隔不一致 | `tah_cache` 默认 stride 2(0.2 s),而闭环 S2 是 6 帧(0.6 s)。`tah_cache` 只保存 `d_feat`、`order`、`fov`;`tah_train` 只检查 `d_feat` |
| 9 | 分析脚本绑死文件名 | tick 日志按顺序对齐的逻辑有 4 份副本(reset 消息不带种子，跳过的集不触发 reset)。`opening_cl.py` 取 `sorted(glob('*.json*'))[-1]`,会优先读 `.partial`。bootstrap 有 8 份实现，参数各不相同 |
| 10 | 评分器和 policy 脱节 | 评分器的 `commit_k` 没有与 policy 的 `--commit-k` 绑定(`metrics.py:35-41`;`run_rollout.py` 只传了 `fps` 和 `max_lost_s`)。`_cand_lite` 不传 `gt_idx` |
| 11 | 真值走在线协议上，无隔离 | env 每个 tick 都把 `target_color`、`target_bp`、`gt_candidates=(tpos, dstates, campose, road_pred_world, vis)` 发给 policy(`remote_policy.py:27-34`,`env.py:291-299`);`is_target`、`true_idx` 在 policy 进程内由 `build_candidates` 生成 |
| 12 | 基础设施约束只写在注释里 | 网络名写死在脚本里。`cyh-carla` 中 `python` 是 3.8.0、`python3` 是 3.6.9,注释却还写着 py3.6。DAM4SAM 服务只接受单个连接、靠手动启动，当前已退出 |

---

## 3. 设计原则

1. **单一事实来源**:所有影响行为的量都来自已解析的配置对象。代码里只允许出现 schema 中声明为 `safe_default` 的默认值。
2. **严格 schema**:配置里出现未知字段就报错。当前没有在用的 head,不允许出现它的子配置。
3. **组合，而不是复制**:用 `extends` 加深合并;实验臂只写自己和基准方法的差异。覆盖记录写进 manifest,并纳入哈希。
4. **清单驱动**:旧配置面由脚本生成，翻译器和 schema 必须覆盖清单的每一行(§6)。
5. **机械化防线优先于约定**:真值隔离、兼容性检查、结果完整性都由代码和测试强制执行，不靠文档约定。
6. **失败即关闭**:缺少划分、产物 manifest 不一致、种子不完整、基础设施冲突时，一律报错;不做静默回退。
7. **先适配、后重构(strangler)**:P0/P1 期间不改算法代码，只允许改日志和回显;用 §9 的一致性验收证明行为没变。
8. **CARLA 侧兼容 py3.8**:`acot/config`、`acot/infra`、`acot/eval/episode.py`、`acot/analysis/results.py` 只用 3.8 语法和标准库(外加 PyYAML)。

---

## 4. 包结构与 CLI

```
acot/                                 新包(仓库根目录)
  __main__.py, cli.py                 python -m acot <stage> <spec> [overrides]
  config/
    schema.py                         dataclass:Protocol / Method / SeedRegistry / Experiment / Infra / DataGenPlan / DatasetSpec / PrepSpec / Recipe
    load.py                           YAML 读取、extends、深合并、key=value 覆盖、严格校验
    resolve.py                        解析引用(protocol@ver、method、seedset、artefact:<id>)、兼容性检查、真值通道判定、规范化哈希
    legacy_map.yaml                   旧配置面每一行 → 新字段或 infra / collection / test / rejected(§6)
    legacy.py                         翻译器：已解析配置 → 旧 policy_server / run_rollout 的 argv 和环境变量(P0–P1)
  infra/
    preflight.py                      网络、容器、镜像、GPU、端口、DAM4SAM 健康检查、CARLA 客户端 Python 版本(§13)
    lease.py                          CARLA 服务器、TM 端口段、policy GPU、DAM4SAM 实例的租约(文件锁)
  runtime/                            (P2 起)policy 进程内部
    io.py                             PolicyInput / Detection / ScoringInfo / OracleHandle(§7)
    perception.py                     SimProxyPerception:线协议消息 → (PolicyInput, ScoringInfo)
  eval/
    scheduler.py                      (臂, 种子)工作队列、worker 池、看门狗、重试、跳过、容器生命周期
    episode.py                        CARLA 侧单集执行(py3.8):把 run_rollout 的循环体提取为 run_one(seed)
    legacy_server_entry.py            P1:policy 进程包装入口(参数断言 + 对象级配置回显 + 真值访问审计)
    collectors.py                     tah_dagger / opening_snapshots / actor_labels / dam4sam_diag / ticks / trk
  artefacts.py                        产物库(内容寻址)、manifest、别名、旧产物导入(带 provenance_gaps)
  provenance.py                       git sha、未提交 diff 哈希、镜像 ID、主机、GPU、时间
  data/                               dataset manifest、split、camera(唯一投影)、crops(唯一裁剪和特征提取，带 feature_id)、language(唯一 describe)
  prep/                               vlm_ctx、tah_seq、merge、opening_snapshots
  train/                              tah_relm、opening_attr_lin、opening_attr_ft 适配器;stage2 先只导入旧产物
  analysis/
    results.py                        按(臂, 种子)读取结果并检查状态;旧结果导入器(§12)
    metrics.py / stats.py / report.py 指标和统计各只有一份实现;报告由实验配置驱动
configs/  protocols/ methods/ seeds.yaml experiments/ infra/ datagen/ datasets/ prep/ recipes/
tools/legacy_surface.py               旧配置面清单生成器(已完成)
legacy/                               迁移后的旧脚本(只读，仍可运行)
tests/                                单元测试、黄金测试、一致性测试(§15)
```

**CLI**

| 命令 | 作用 |
|---|---|
| `acot lint` | 校验全部配置:schema、种子重叠、FOV 冲突、输出目录冲突、缺失的划分、清单覆盖率 |
| `acot show <spec> [--arm a]` | 打印已解析配置、配置哈希、真值通道分类、兼容性检查结果，以及翻译后的旧命令 |
| `acot data gen datagen/<x>.yaml` | 按冻结计划生成数据(§11);也可加 `--dry-run` |
| `acot data manifest/split …` | 生成数据集 manifest;显式生成划分(不会自动生成) |
| `acot prep <kind> prep/<x>.yaml` | 构建缓存产物(vlm_ctx、tah_seq、merge、opening_snapshots) |
| `acot train recipes/<x>.yaml seed=0` | 训练，产出 ckpt 产物(best、last、metrics、manifest、compat) |
| `acot eval experiments/<x>.yaml [--arms] [--seeds] [--dry-run] [--allow-test]` | 闭环评测(§8) |
| `acot analyze experiments/<x>.yaml` | 生成报告(§12) |
| `acot results import-legacy legacy/arm_map.yaml` | 导入旧结果(§12.2) |
| `acot artefacts import/ls/verify` | 导入、列出、校验产物 |

---

## 5. 配置 schema

### 5.1 解析顺序、覆盖规则与哈希

1. **解析顺序**:协议(冻结)→ 方法(`extends` 链)→ 臂级覆盖 → 命令行 `key=value`。
2. **臂级覆盖的范围**:
   - 只能覆盖 `method.*`;
   - 覆盖 `protocol.frontend.*` 或 `protocol.perception.*` 时，该臂必须声明 `ablation: <component>`,该臂在报告中自动归入消融分组;
   - 覆盖协议的其他字段一律报错。
3. **命令行覆盖的范围**:默认只能改 `infra.*` 和种子选择。改科学字段必须加 `--ad-hoc`,该运行在 manifest 中标为 ad hoc,不能进入预注册的比较。
4. **哈希**:取已解析配置中科学部分(协议、方法、种子列表、采集器中会影响行为的字段)的规范化 JSON,计算 sha256 并取前 12 位作为 `config_hash`。
   - 规范化：键排序、无空白、浮点数用 repr、未激活 head 的子配置不出现。
   - `infra` 不计入哈希。
5. **两个 `commit_k`**:
   - `protocol.scorer.commit_k` 是**指标定义**,对所有臂固定;
   - `method.control.commit_k` 是 policy 的提交滞后参数，默认继承协议的值。
   - 方法改了它，resolve 会在 manifest 和报告中写明「policy K ≠ 指标 K」;指标仍按协议的 K 计算，保证各臂可比。

### 5.2 Protocol:`configs/protocols/sim_proxy_v2@1.yaml`

```yaml
name: sim_proxy_v2
version: 1                                  # 冻结;任何修改都要升版本
aliases: [deploy_v2]                        # 旧结果标签
camera: {fov_deg: 90, width: 512, height: 512, fps: 10}
sim:
  s2_period: 6                              # 1 个 S2 tick = 0.6 s;开局 tick 由它和 opening.window 推导
  episode_steps: 900
  town: Town05
  scenario: mvp
  max_lost_s: 18                            # 来自 default.yaml(RolloutScorer 的默认值 5 不再生效)
  warmup: {expert_ticks: 90, min_target_speed_mps: 1.2}
  retarget_tick: null                       # 非空 → 动态换目标的协议变体(diagnostic)
perception:                                 # 仿真代理感知，对所有臂相同
  candidates: {source: sim_projection, set: task_vehicles, vehicle_size_m: 4.0, box_min_px: 10, box_max_px: 140}
  occlusion: raycast                        # = --vis-raycast + --occl-cands
  identity_keys: tracker                    # = --tracklets tracker(sim 只用于复现 deploy_v1)
  tracker: {max_age: 8, gate_px: 150, gate_box: 2.0, w_app: 0.85, min_cos: 0.2, vel_beta: 0.5, app_alpha: 0.3}
  where_localisation: gt_matched_box
  noise: none
frontend:                                   # 所有臂共享的感知前端
  vlm: {model_config: train/config_v5.yaml, stage2_ckpt: "artefact:stage2_v5", language_mode: neutral}
  crop_features: {id: dino_s14_lvd142m_cls384_c224_p1.3, depth_scale: 100}
  opening:
    module: attr                            # head | zoom | attr
    hires_k: 3
    window: 3
    agg: sum
    final_reset: false
    reset_asd: false
    force_final: false
    seed_asd: 0.0
    attr: {ckpt_lin: "artefact:opening_attr_lin", ckpt_ft: "artefact:opening_attr_ft", wm: 1.0, wc: 1.0, wq: 0.0, ft: 0.5}
control: {ex_source: track, frame_gain: 0.0, commit_k: 5, shuffle_seed: 0, ablate: null}
scorer: {commit_k: 5, t_fail_s: 2.0, central_frac: 0.3, primary: committed}
privileged_perception: [cand_box_gt_projection, cand_depth_gt, occlusion_raycast_gt, task_vehicle_set, where_world_gt_matched_box]
```

### 5.3 Method

```yaml
# configs/methods/ours_final.yaml —— 所有字段显式写出(不依赖代码默认值)
name: ours_final
identity:
  head: tah
  tah: {ckpt: "artefact:tah_relm_dag", lang_reanchor_thr: 0.60}
  memory_seed: committed                    # --gallery-seed;取 gt 即为真值通道
  conf_tau: 0.5                             # 提交门控(head=reid 且开启重锚时，还兼作低置信阈值)
  conf_consensus: 0.0
  reanchor: {mode: off, patience: 5, tavg: 0.0}       # mode: off | lang | oracle(oracle 为真值通道)
  cold: {window: 0, lambda: 0.9, single: false}        # COLD_*,只用于复现 V2 实验
filter:                                     # ASD / PGIF;ASD 构造函数的全部字段
  mode: filter
  filter_kind: tempered
  resid_decay: 0.98
  absent_hold: true
  absent_max: 8
  conf_window: 30
  gate_memory: true
  gate_tah: true
  gate_dam: true
  naive: false
  alpha: null
  q_lo: 0.2
  q_hi: 0.7
  eps_switch: 0.01
  anchor_mode: first
  k_anchor: 3
  beta_lang: 0.25
  lang_ema: 0.7
  tau: 0.1
  kappa: 0.25
  h: 2.0
  refractory: 3
  learned_ckpt: null
  v1_selfreset: false
diagnostics: []                             # 非空 → diagnostic 臂(§7)
```

```yaml
# configs/methods/ext_dam4sam.yaml
name: ext_dam4sam
identity:
  head: dam4sam
  dam4sam: {gate_px: 160, every_frame: false, reinit_cooldown: 3, service: dam4sam}   # gate_px → --assoc-gate-px(DAM4SAM 复用该参数)
  memory_seed: committed
  conf_tau: 0.05
  reanchor: {mode: off}
filter: {mode: off}
diagnostics: []
```

方法 YAML 覆盖当前所有臂：
- 主方法 `ours_final`;
- H-A,即 `ours_final` 加开局消融 `module: head`;
- 开局变体 v2–v4;
- PGIF 的各个变体;
- 外部基线：DAM4SAM、DeepSORT/OC-SORT 关联、reid;
- 语言 head:xattn;
- 诊断臂:oracle_id、oracle_absent、gt_seeded、road_oracle、attrbind。

`tid_head` 每个取值实际读取哪些 kwarg,在 P0a 生成「参数使用矩阵」。例如 DAM4SAM 复用 `assoc_gate_px`。schema 按这个矩阵校验：只允许当前 head 用到的字段出现。

### 5.4 种子登记:`configs/seeds.yaml`

```yaml
seedsets:
  datagen_mvp_v5:   {list_file: runs/tah_v90/b_manifest.json#seeds, purpose: train_data}   # 任意列表(含替补 108+)
  train_dagger_v2:  {ranges: [[97000, 97105]], exclude: [97077], purpose: train}           # 97077:Town02 生成失败
  dev:              {ranges: [[91000, 91020]], purpose: dev}
  dev2:             {ranges: [[93000, 93100]], purpose: dev}
  heldout_opening:  {ranges: [[93100, 93660]], purpose: heldout}
  test:             {ranges: [[95000, 96000]], purpose: test, locked: true}
substitutes: {datagen_mvp_v5: {from_manifest: runs/tah_v90/b_manifest.json#substitutes}}
rules: {disjoint_across_purposes: true, test_requires: [--allow-test, prereg_ref]}
```

- 种子可以用 `ranges`、`list`、`list_file` 三种方式给出，也可以组合。
- 生成失败需要替补的种子，由替补映射显式记录。
- 实际种子以迁移时核对 `collect_dagger.sh` 等脚本的结果为准。
- 不同用途的种子集重叠时,`acot lint` 报错。
- 使用 test 种子必须加 `--allow-test`,并在实验配置中填写预注册文档的引用(`prereg_ref`)。

### 5.5 Experiment

```yaml
# configs/experiments/1004_final_table.yaml
name: 1004_final_table
protocol: sim_proxy_v2@1
seeds: {set: dev, expect: 20}
completeness: {require: all_complete, allow_skipped_reasons: [], pairing: strict}   # 失败即关闭(§12.1)
retry: {infra_attempts: 2, retry_skipped: 0}
arms:
  ours:        {method: ours_final}
  no_pgif:     {method: ours_final, override: {filter.mode: off}}
  pgif_bayes:  {method: ours_final, override: {filter.filter_kind: bayes}}
  open_head:   {method: ours_final, ablation: opening, override: {frontend.opening.module: head, frontend.opening.window: 1}}
  ext_dam4sam: {method: ext_dam4sam}
collect: [ticks, trk, gt_access, scenario]
comparisons:
  - {family: main, a: ours, b: [no_pgif, pgif_bayes, open_head, ext_dam4sam],
     metrics: [committed, longC, longW], test: paired_bootstrap, correction: holm}
```

### 5.6 Infra:`configs/infra/default.yaml`(不计入哈希;§13 详述)

```yaml
docker:
  network: cyh-carla-net              # 必须已存在;只检查，绝不创建
  carla_container: cyh-carla          # 只用 docker exec
  policy_image: acot-uav-train:cu118  # 运行时记录 image ID
  policy_container_prefix: acot-      # 只有 policy 容器允许 docker run(--network 为上面的网络)
python: {carla_client: python, carla_client_version: "3.8", policy: python, policy_version: "3.11"}
carla_servers:                        # 端口 → -graphicsadapter(原 launch_wd.sh 中的 ADAPTER 表)
  - {port: 2000, adapter: 0}
  - {port: 2010, adapter: 2}
  - {port: 2020, adapter: 3}
  - {port: 2030, adapter: 4}
  - {port: 2040, adapter: 5}
  - {port: 2050, adapter: 6}
  - {port: 2060, adapter: 7}
  - {port: 2070, adapter: 0}
  - {port: 2080, adapter: 3}
  - {port: 2090, adapter: 5}
  - {port: 2230, adapter: 0}
  - {port: 2250, adapter: 3}
tm_ports: {base: 9700, per_worker: 30, per_restart: 10}
policy_gpus: [1, 2, 4, 6]              # 示例;P0a 按实际占用核对
reserved: {carla_ports: [[2100, 2190]], gpus: [], containers: []}     # 其他会话占用的资源
services:
  dam4sam:
    instances: [{container: dam4sam-svc, port: 5601}]
    clients_per_instance: 1           # 服务按顺序 accept 单个连接(dam4sam_service.py:55-58)
    lifecycle: managed                # managed:docker start + docker exec 启动服务;external:只做健康检查
    start_cmd: "conda activate dam4sam && python carla_uav_tracking/rollout/dam4sam_service.py --model sam21pp-L --port {port}"
    health: {tcp_roundtrip: close, timeout_s: 10}
watchdog: {first_min: 25, stall_min: 12, max_restarts: 5}
paths: {repo: /nvidia/hque/code/cyhe/uav-acot-track, data: /nvidia/hque/data/carla_data, hf_cache: ~/.cache/huggingface}
threads: {omp: 6}
```

### 5.7 DataGenPlan、DatasetSpec、PrepSpec、Recipe

- **DataGenPlan**:见 §11。
- **DatasetSpec**:路径、`expected_camera`、划分文件(必须存在，带内容哈希)、`camera_override`(必须附证据)。
- **PrepSpec**:产物种类、输入产物 ID、各种类特有的参数(stride、feature_id、depth_scale…)。manifest 中记录推导出的 `tick_s` 和 camera。
- **Recipe**:训练器名;输入产物 ID 列表(顺序有意义);验证划分规则;超参;`seed`;`select: best | last | both`;`compat_targets`(声明准备在哪些协议下使用);`waivers`。

---

## 6. 旧配置面的收编(M1)

### 6.1 清单

`tools/legacy_surface.py` 通过 AST 和正则扫描代码，生成 `acot_note/1004_legacy_surface.md`,共 6 节:

| 节 | 内容 | 数量 |
|---|---|---|
| 1 | argparse 参数:`policy_server`、`run_rollout`、`dam4sam_service`、`train/*`、`opening_*`、`generate_batch` | 118 |
| 2 | 构造函数参数和 JSON 字段:`Policy`(36 个 kwarg,含未暴露的 `shuffle_seed`)、`ASD`(24)、`TrackletAssigner`(7)、`AttrScorer`、`ZoomVerifier`、`RolloutScorer`,以及 `OPENING_CFG` 中被 Policy pop 掉的 6 个字段 | 92 |
| 3 | 环境变量读写，另有 `smoke_iter.py` 中 1 处动态键写入 | 24 |
| 4 | env → policy 线协议字段(按真值分类) | 16 |
| 5 | 启动脚本：调用了什么、docker 操作、写死的行为参数和环境变量、可调参数、位置参数 | 73 |
| 6 | 行为常量和已知冲突 | 25 |

每一行都带一个分类:scientific、diagnostic、privileged、collection、infra、injected、test、offline。

### 6.2 `legacy_map.yaml` 与覆盖测试

```yaml
flags:
  policy_server:
    --tid-head:  method.identity.head
    --conf-tau:  method.identity.conf_tau
    --asd:       method.filter.mode
    --asd-cfg:   {json: method.filter, precedence: "非空时整体替换 ASD_CFG(不合并)"}
    --device:    infra.policy.device
    --commit-k:  method.control.commit_k
    # ... 清单第 1 节的每一行
  run_rollout:
    --vis-raycast: protocol.perception.occlusion          # raycast ⇔ True
    --hires:       protocol.frontend.opening.hires_k
    # ...
env:
  ASD_CFG:       {json: method.filter}
  ORACLE_ABSENT: {diagnostic: oracle_absent, values: {"1": abstain, "2": hygiene}}
  ASD_LOG:       {collect: ticks}
  HOSTNAME:      {infra: naming_only}
json:
  ASD:              {mode: "由 --asd 注入", log_path: "由 ASD_LOG 注入", k_anchor: method.filter.k_anchor}   # ...
  TrackletAssigner: {max_age: protocol.perception.tracker.max_age}                                       # ...
constants:
  policy.shuffle_seed: protocol.control.shuffle_seed
  metrics.RolloutScorer.commit_k: protocol.scorer.commit_k
  # ...
rejected: []          # 有意不迁移的通道;必须写明原因，并指向 legacy/ 下的复现方式
```

`tests/test_legacy_surface.py` 的规则:
1. 重新运行生成器;
2. 清单中每一行都必须在 `legacy_map.yaml` 中找到映射，否则失败;
3. `legacy_map.yaml` 中多出的条目(指向清单里已经不存在的行)也算失败。

于是，代码里新增一个 flag 或环境变量而没有登记，测试立刻变红。这样，「完整收编」从一句口号变成了可以测试的性质。

### 6.3 分组映射(要点，完整内容以 `legacy_map.yaml` 为准)

| 旧通道 | 新字段 |
|---|---|
| `--config`、`--ckpt`、`--language-mode` | `protocol.frontend.vlm.*` |
| `--s2-period`、`--ex-source`、`--frame-gain`、`--ablate`、`shuffle_seed` | `protocol.sim.s2_period`、`protocol.control.*`。`ex_source ≠ track` 属于 diagnostic;`ablate` 只用于 H1/H2 消融(配合对应 ckpt) |
| `--commit-k`;评分器的 `commit_k`、`t_fail`、`central_frac`;`max_lost_seconds` | `method.control.commit_k`、`protocol.scorer.*`、`protocol.sim.max_lost_s`(见 §5.1 第 5 条) |
| `--tid-head`、`--tid-ckpt`、`--tah-ckpt`;TAH 语言重锚阈值 0.60 | `method.identity.head`、`.<head>.ckpt`、`.tah.lang_reanchor_thr` |
| `--gallery-seed`、`--conf-tau`、`--conf-consensus`、`--warmup-oracle-s` | `method.identity.memory_seed`、`.conf_tau`、`.conf_consensus`、`diagnostics: [warmup_oracle]` |
| `--reid-feature/-bank/-topm/-tavg/-motion/-motion-gate-px` | `method.identity.reid.*` |
| `--reanchor`、`--reanchor-patience`、`--reanchor-tavg` | `method.identity.reanchor.{mode, patience, tavg}`;`mode=oracle` 属于 diagnostic |
| `--assoc-mode/-gate-px/-lambda` | `method.identity.assoc.*`;DAM4SAM 复用 gate,对应 `identity.dam4sam.gate_px` |
| `--dam4sam-host/-port`、`DAM4SAM_EVERY_FRAME`、冷却 3 tick | `infra.services.dam4sam` 租约、`identity.dam4sam.{every_frame, reinit_cooldown}` |
| `--asd`、`--asd-cfg`、`ASD_CFG`(24 个字段) | `method.filter.*`(全部显式写出;`log_path` 改由采集器决定) |
| `--tracklets`、`--occl-cands`、`--vis-raycast`、`TRK_CFG`(7 个字段) | `protocol.perception.*` |
| `--hires`、`OPENING`、`OPENING_CFG`(window、agg、final_reset、reset_asd、force_final、seed_asd,以及 ckpt、wm、wc、wq、ft) | `protocol.frontend.opening.*`(开局消融臂可覆盖);ft 的 ckpt 路径目前写死，在 P0/P1 中只校验它与 `ckpt_ft` 一致 |
| `--steps`、`--town`、`--episodes`、`--seed-base` | `protocol.sim.*`;种子由实验的种子集提供，调度器分配 |
| `--expert-control`、`--predict-road`、`--road-oracle`、`--retarget-tick`、`--event-override`、`ORACLE_ABSENT` | `method.diagnostics: [...]`;`retarget_tick` 属于协议变体 |
| `COLD_WINDOW/LAMBDA/SINGLE`、`V1_SELFRESET` | `method.identity.cold.*`、`method.filter.v1_selfreset`(只用于复现旧实验) |
| `ASD_LOG`、`TAH_DAGGER_DIR`、`DAGGER_DIR`、`OPENING_DUMP`、`DUMP_LABELS`、`DAM4SAM_DIAG` | 实验的 `collect: [ticks, tah_dagger, dagger, opening_snapshots, actor_labels, dam4sam_diag]` |
| `CARLA_PORT/TM_PORT/HOST/ROOT`、`--port`、`--policy-host/-port`、`--carla-port`、GPU、`NTHR`、`SRV`、`NET`、`HF_CACHE`、`STALL_MIN/FIRST_MIN/START_PART`、ADAPTER 表 | `infra.*`(调度器负责分配，不计入哈希) |
| `eval_asd.sh` 写死的参数:`--ckpt stage2_best.pt`、`--language-mode neutral`、`--ex-source track`、`CONFIG`、镜像、启动时 `pip install timm==1.0.29` | 协议字段 / `infra.docker.policy_image`。依赖改为在镜像中固定，启动时安装依赖属于 P1a 需要修掉的问题 |
| `SMOKE_*`、`PARITY_*` | test(测试工具，不迁移) |

### 6.4 隐式语义(翻译器必须保留，P2 起显式化)

1. `--asd-cfg` 非空时整体替换 `ASD_CFG`,两者不合并。
2. `mode` 和 `log_path` 由 flag 或环境变量注入。如果 JSON 里也写了这两个字段，会报 TypeError(参数重复)。
3. JSON 字段是直接展开进构造函数的(`**json.loads(...)`)。未知字段在**运行时**才报 TypeError,而且报错发生在 policy 容器里，容易被忽略。
4. `OPENING_CFG` 先被 Policy pop 掉 6 个字段，剩下的才传给 `AttrScorer` 或 `ZoomVerifier`。
5. `--conf-tau` 一个值有两种用途：提交门控(`policy.py:1071`);在 reid 重锚开启时，还兼作低置信阈值(`policy.py:951`)。
6. `reset()` 每个 episode 都会用 `shuffle_seed=0` 重新初始化候选打乱用的 RNG。因此额外调用一次 reset 不会改变 RNG 状态，这是 §8.4 的前提。
7. `run_rollout` 构造评分器时只传 `fps` 和 `max_lost_s`;`commit_k`、`t_fail`、`central_frac` 用的是类默认值。

### 6.5 未知字段的处理(按阶段)

| 阶段 | 规则 |
|---|---|
| P0 | 新 YAML 中的未知字段在 `load` 时就报错。翻译器只输出 `legacy_map.yaml` 中登记过的通道 |
| P1 | `legacy_server_entry.py` 在启动 policy 前比对实际的环境变量和 argv,与翻译器的输出逐字节一致才继续，否则中止 |
| P2 | policy 和 env 只读 `--resolved`。只要设置了任何旧的行为环境变量(清单第 3 节中 scientific、diagnostic、collection 类),就报错 |

---

## 7. 真值通道隔离(M2)

### 7.1 通道分类(清单第 4 节)

| 类别 | 通道 | 规则 |
|---|---|---|
| 可部署 | `rgb`、`rgb_hi`、`language`、`proprio`、`step`、`campose`(自身相机位姿) | 方法可用 |
| 协议允许的理想化感知 | `tpos`/`dstates` 只能经由 `build_candidates` 变成候选框;GT 深度;`vis` 射线遮挡;任务车辆集合;所匹配框的世界坐标 | 对所有臂相同，在 `privileged_perception` 中披露;方法只能看到由此**生成的框**,看不到原始状态 |
| diagnostic | `target_color`/`target_bp`(attrbind)、`road_pred_world`、`retarget_event`、`ex_source ≠ track` 下用 `tpos` 做 WHERE、`gallery_seed=gt`、`reanchor=oracle`、`warmup_oracle`、`event_override`、`ORACLE_ABSENT`、`tid_head=oracle` | 只有在方法里声明为 `diagnostics` 才能用;该臂标为 diagnostic,只能进入「上界/诊断」分组 |
| 只用于评分和采集 | `is_target`、`true_idx`、`gt_idx`;目标属性(作为采集标签) | 只能交给评分器和采集器 |

### 7.2 代码接口(P2 实现，P0 定稿)

```python
@dataclass(frozen=True)
class Detection:                 # 一个候选框(仿真代理或将来的真实检测器)
    key: int                     # 跟踪器给的 ID,不使用仿真 actor 编号
    u: float; v: float; w: float; h: float
    depth_m: float               # 理想化感知(协议披露)
    world_xyz: Optional[Tuple[float, float, float]]   # 理想化感知:where_localisation=gt_matched_box
    feat: np.ndarray             # 由图像计算的框内特征

@dataclass(frozen=True)
class PolicyInput:               # 方法能拿到的全部输入
    seed: int; step: int; rgb: np.ndarray; rgb_hi: Optional[np.ndarray]
    language: str; proprio: np.ndarray; campose: Tuple[float, ...]
    detections: Optional[Tuple[Detection, ...]]
    # 不含 is_target / true_idx / gt_idx / target_color / target_bp / tpos / dstates / road_pred_world / vis

@dataclass(frozen=True)
class ScoringInfo:               # 只交给评分器和采集器
    seed: int; step: int; true_idx: int; is_target: Tuple[bool, ...]; gt_idx: Tuple[int, ...]
    target_color: str; target_bp: str; vis: Optional[tuple]

class OracleHandle:              # 只为声明了 diagnostics 的臂构造;其他臂传 None
    def __init__(self, allowed: FrozenSet[str], info_fn, audit): ...
    def true_slot(self, purpose: str) -> int     # purpose 必须在 allowed 中;每次调用都记入 gt_access
```

`SimProxyPerception.process(msg) -> (PolicyInput, ScoringInfo)` 在 policy 进程中运行，是 `act()` 的第一步，因为它需要用到 VLM 的 grid 特征。方法组件(开局、身份 head、滤波、WHERE)的签名只接受 `PolicyInput`,以及可能为 None 的 `OracleHandle`。`_cand_lite` 改为由 `ScoringInfo` 生成，并补传 `gt_idx`。

### 7.3 分阶段落实

| 阶段 | 措施 | 机械化程度 |
|---|---|---|
| **P0** | resolve 根据清单分类计算 `uses_gt`。方法用到 diagnostic 通道，却没有在 `diagnostics` 里声明 → 报错;声明了的 → 臂标为 diagnostic。`report.py` 拒绝把 diagnostic 臂放进主表分组 | 配置层强制 |
| **P1** | ① **在线协议置空**:新的 CARLA 侧执行器(`episode.py`)为非 diagnostic 臂发送 `target_color=""`、`target_bp=""`、`road_pred_world=None`、`retarget_event=False`。只有方法声明了 attrbind、road_oracle 或 retarget,或者实验声明了需要这些标签的采集器(opening_snapshots、dagger)时才放行，放行情况记入 manifest。② **调用点审计(棘轮)**:`legacy_server_entry.py` 在加载旧 policy 之前，把 `CandidateSet.true_idx`、`Candidate.is_target`、`Candidate.gt_idx`、`Observation.target_color/target_bp` 替换成属性访问器，记录每次访问的调用点(文件、函数、规范化后的源码行)。每个 episode 结束时写出 `gt_access/<seed>.json`。可部署臂的访问点集合必须是 `acot/config/gt_sites.yaml` 中已审核白名单(每项注明 scoring、logging 或 collection)的子集，否则该种子记为 failed,原因 `gt_access_violation`。代码里出现新的访问点时，必须先经过审核登记才能通过 | 运行时强制(不改算法代码) |
| **P2** | `PolicyInput` 和 `ScoringInfo` 在结构上分开，方法组件拿不到真值字段;只有 diagnostic 臂才会得到 `OracleHandle` | 结构强制 |

P1 白名单的初始内容来自现有代码：`_cand_lite`(评分器)、DAgger 缓冲区、TAH DAgger、开局快照、ASD 和 trk 日志中的 `true` 与 `gt` 字段，以及 `_committed_world` 中用 `gt_idx` 查当前所匹配框的世界坐标。最后一项属于协议披露的 `where_world_gt_matched_box`,在审核表中归为理想化感知。

### 7.4 测试

| 测试 | 内容 |
|---|---|
| `test_policy_input_fields` | `PolicyInput` 和 `Detection` 的字段与禁止字段集合的交集为空 |
| `test_method_no_gt_attr` | 对方法组件模块做 AST 扫描：不能访问禁止字段，不能 import `ScoringInfo` |
| `test_diag_label` | 声明了任一 diagnostic 通道的臂一定标为 diagnostic;用了 diagnostic flag 却没有声明 → resolve 报错;diagnostic 臂进入主表 → 报告报错 |
| `test_wire_blanking` | 可部署臂的 act 消息中,`target_color`、`target_bp` 为空,`road_pred_world` 为 None |
| `test_gt_audit_ratchet` | 用合成调用点模拟一个新的 `true_idx` 读取，审计判为失败;对已登记的日志访问点，审计判为通过 |
| `test_oracle_handle` | 可部署臂得到的 `oracle` 为 None;diagnostic 臂调用一个没有声明的 purpose → 报错 |

---

## 8. 闭环运行时

### 8.1 流程

1. 解析实验，得到每个臂的 `resolved.yaml` 和 `config_hash`,并做兼容性检查(§10.2)。
2. 基础设施 preflight(§13.2)。
3. 调度器申请租约：CARLA 服务器、TM 端口段、policy GPU、DAM4SAM 实例。
4. 每个 worker 为一个臂起一个 policy 容器：
   - 用 `docker run --network cyh-carla-net`,容器名 `acot-<exp>-<arm>-<w>`;
   - 等待 `server_echo` 返回，并核对其中的哈希(§8.5)。
5. CARLA 侧通过 `docker exec cyh-carla python -m acot.eval.episode` 逐个种子执行，每个种子落盘一组文件(§8.3)。
6. 收尾：释放租约、删除容器、写入实验 manifest。

### 8.2 每个种子的状态机

| 状态 | 判定 | 后续处理 |
|---|---|---|
| `complete` | 评分器写出最终记录 | 进入分析 |
| `skipped` | 场景准备阶段被拒绝：目标静止、目标生成失败(spawn failed)等，原因来自 env | 默认不重试，因为重试会改变世界历史，影响可复现性。是否允许跳过由实验的 `completeness` 决定 |
| `failed` | 开局后出现异常、卡死，或真值访问越界;已用完 `infra_attempts` 次重试 | 每次重试前都重启该 CARLA 服务器(沿用 `launch_wd` 的逻辑);记录 traceback |
| `partial` | 进程在一集中途死掉，没有写出最终记录 | 只保留 tick 日志用于排查，永不进入分析 |

状态文件最后写入，并通过 tmp + rename 保证原子性。续跑时，已经 `complete` 或 `skipped` 的种子直接跳过。

### 8.3 结果布局

```
runs/exp/<exp>/
  experiment.resolved.yaml   manifest.json(git、未提交 diff 哈希、镜像 ID、infra 快照、开始和结束时间)
  <arm>/
    resolved.yaml   server_echo.json   client_echo.json
    seeds/<seed>/
      status.json             {status, reason, attempts: [...], worker, config_hash, t_start, t_end}
      episode.json            评分器输出(只在 complete 时存在)
      scenario.json           场景指纹:town、目标的 blueprint 和颜色、干扰车的 blueprint、生成位姿、warm-up 结束时的位姿
      ticks.jsonl  trk.jsonl  gt_access.json
      collect/<collector>/...
    logs/                     每次尝试的 policy 和 client 日志
  report/
```

### 8.4 reset 携带种子(P1,只改日志)

- `RemotePolicy.reset()` 改为发送 `{"cmd": "reset", "seed": s, "arm": a, "config_hash": h}`。旧版 server 会忽略这些多出来的字段，所以这一步向后兼容。
- `policy_server` 把 `seed` 交给 `policy.reset(seed=...)`,只保存起来供日志使用。
- ASD、trk、开局、DAgger 的日志都加上 `seed` 字段。
- env 调用 reset 的时机保持不变：跳过的集仍然不会触发 reset。由于日志已经按种子作键，不再依赖顺序对齐。
- 这项改动不影响任何决策，由 L0–L2 验证(§9)。

### 8.5 配置回显

- **policy 侧(P1)**:`legacy_server_entry.py --resolved R` 的执行步骤:
  1. 断言 argv 和环境变量与翻译器的输出逐字节一致;
  2. 在 `Policy`、`ASD`、`TrackletAssigner`、`AttrScorer`/`ZoomVerifier` 的 `__init__` 外包一层，记录实际收到的 kwargs;
  3. 用 runpy 执行旧的 `policy_server.py`;
  4. 写出 `server_echo.json`,内容为实际 kwargs 及其哈希。

  调度器用翻译器的 `expected_kwargs(resolved)` 计算期望哈希，两者不一致就中止。这样核对的是对象级的有效配置，比单纯比较 argv 更强。
- **CARLA 侧**:`episode.py` 写出 `client_echo.json`,内容包括 steps、town、vis_raycast、hires_k、评分器参数、种子列表。
- **P2 起**:policy 直接读取 `--resolved`,回显的就是已解析配置本身的哈希。

### 8.6 评分器参数

`episode.py` 构造 `RolloutScorer` 时，把 `fps`、`commit_k`、`t_fail_s`、`central_frac`、`max_lost_s` 全部显式传入，值取自 `protocol.scorer` 和 `protocol.sim`,不再依赖类默认值。

### 8.7 采集器

采集器由实验配置的 `collect` 声明，输出按种子落盘。需要标签的采集器(DAgger、开局快照)从 `ScoringInfo` 取标签：P1 期间通过 §7.3 的放行机制拿到，P2 起改为结构化获取。

---

## 9. 一致性验收(M3)

| 级别 | 内容 | 通过标准 | 允许的差异 |
|---|---|---|---|
| **L0 调用** | docker shim(一个放在 PATH 最前面、只记录 argv、环境变量和挂载并返回成功的假 `docker`)分别捕获两组命令：旧启动脚本实际发出的 `docker run/exec`,以及 `acot eval --dry-run` 的输出 | 科学类 flag 和环境变量逐字节相等(JSON 值解析后比较);infra 类只比较字段集合;覆盖范围包括 `chains_attr.sh`、`run_openval.sh`、`f_v4.sh`、`f_oracle_absent.sh`、`collect_dagger.sh`、`eval_ext_dam4sam.sh`、`eval_ext_assoc.sh`、`launch_wd.sh` 的续跑路径 | 容器名、端口、路径(infra) |
| **L1 回显** | `server_echo`、`client_echo` | 对象级 kwargs 哈希与 `expected_kwargs` 相等 | 无 |
| **L2 评分** | 用新导入器和新分析，重算已有的旧结果 | 与 `arch_compare.py`、`opening_ds.py`、`iter_gate.py`、`opening_cl.py` 的数字相差 ≤ 1e-12。按种子对齐的 tick 日志，与旧的按顺序对齐(`tick_align.aligned`)逐条相同 | 只允许 §12.2 中预先列明的旧脚本缺陷造成的差异，并且要逐条列出 |
| **L3 实机冒烟** | 在 dev 的 2 个种子上，对 `ours_final`、H-A、`ext_dam4sam` 各跑一次旧启动方式，再跑一次新入口 | 种子相同;`scenario.json` 中开局之前的字段相同;跳过集合相同(只有原因属于不确定类时才允许不同，并列出);配置回显一致;可部署臂的 `gt_access` 不越出白名单;评分器参数相同 | 轨迹和指标数值(CARLA 的交通灯相位等依赖服务器历史) |
| **L3+(可选)** | A/A:新、旧入口各在 dev 上跑 20 个种子 | 主指标的配对差异落在 bootstrap 95% 区间内，作为健全性检查 | — |

`scenario.json` 中哪些字段在给定种子下是确定的，在 P1a 中实测确认：同一种子连跑两次比较。不确定的字段不进入 L3 的比较，并在此表中注明。

---

## 10. 离线：产物、manifest 与兼容性(M4)

### 10.1 产物种类与 manifest 字段

| 产物 | 生成方式 | manifest 必含字段 |
|---|---|---|
| `dataset` | `acot data gen` / 导入 | 每个文件的 sha256 和大小;attrs 汇总(fov、w、h、fps、seed、town);计划哈希、场景配置哈希;camera(覆盖时附证据);生成器的 git |
| `split` | `acot data split` | 数据集哈希、规则、随机种子、各划分的列表、文件 sha256 |
| `vlm_ctx`(stage1/stage2 上下文缓存) | `acot prep vlm_ctx` | 数据集哈希、划分哈希;backbone 的 kind、model_id、revision;layers、ctx_grid;frame_stride(由此推导 tick_s)、max_frames、min/max_pixels;language_mode、prompt 模板哈希、dtype |
| `crop_feats`(DINO,单独产出或内嵌在 tah_seq 中) | prep | `feature_id` = 模型和权重哈希、crop 224、pad 1.3、重采样、归一化、池化方式、TTA;depth_scale |
| `tah_seq` | `acot prep tah_seq` | 数据集哈希、划分哈希、stride、`tick_s = stride / fps`、feature_id、depth_scale、camera、候选集规则、框规则(vehicle_size、box_min/max)、order、d_feat |
| `dagger` | `acot eval` + 采集器 tah_dagger | 实验、臂、config_hash;协议 ID;种子和状态;`tick_s = s2_period / fps`;feature_id、fov、identity_keys(tracker 或 sim);order |
| `merge` | `acot prep merge` | 有序的分片列表(每片带 ID 和哈希);分片之间兼容字段必须一致，否则需要豁免 |
| `opening_snapshots` | 采集器 | 实验、臂、config_hash;种子;hires_k;开局 tick;标签来源(ScoringInfo) |
| `opening_cls` / `opening_ft` | `acot train` | 快照 ID;feature_id(DINO 尺度 1.5/2.0 加翻转 TTA;ft 为 4 个 block、224);词表哈希(cls_m、cls_c);训练和留出用的种子集;配方 |
| `tah_ckpt` | `acot train tah_relm` | 输入产物 ID(有序);配方;选择规则;best_epoch;seed;`compat` = {fov、tick_s(或 tick_s 集合加豁免)、feature_id、depth_scale、d_feat、arch} |
| `stage2_ckpt` | 先只做旧产物导入 | 配置快照哈希、数据集哈希、划分哈希、ctx 缓存 ID、camera、language_mode、epoch、metrics |

### 10.2 兼容性矩阵(两道边界)

| 比较字段 | B1:训练时，输入产物之间 | B2:闭环解析时,ckpt 与协议 |
|---|---|---|
| `fov_deg`、分辨率 | ✓ | ✓ |
| `tick_s` | ✓ 混用需要豁免 | ✓ |
| `feature_id`(含 crop、pad、模型、TTA) | ✓ | ✓(与 `frontend.crop_features.id` 对照) |
| `depth_scale` | ✓ | ✓ |
| `language_mode`、prompt 哈希 | ✓(vlm_ctx) | ✓(stage2 与 `frontend.vlm`) |
| 划分哈希、数据集哈希 | ✓ 必须来自同一数据集清单，或已声明的拼接 | 只记录 |
| 词表哈希 | ✓(开局) | ✓(开局 ckpt 与协议的语言描述器) |

- 字段取值为 `unknown`(旧产物缺失元数据)时，不算匹配。使用这样的产物必须在臂或配方中声明 `accept_provenance_gaps: [<artefact>:<field>]`,声明会写入 manifest,并在报告的脚注中列出。
- 例如：最终模型的 `tah_relm_dag` 混用了离线 0.2 s 和 DAgger 0.6 s 的数据，离线部分还有 FOV 错位，这些都会作为已声明的缺口公开。

### 10.3 内容寻址与缓存复用

- 产物 ID = sha256(规范化的 manifest 输入部分)取前 12 位，产物存放在 `runs/artefacts/<kind>/<name>@<id>/`。
- 目录已存在时，其中的 manifest 必须与请求完全一致，否则报错「冲突或过期的产物」。不再有「文件存在就跳过」的逻辑，这一条修掉 `backbone_kv`。
- 分片先写到 `.shards/<k>`。合并步骤核对分片和 episode 都齐全之后，才写 manifest。没有 manifest 的产物视为不完整，下游拒绝使用。

### 10.4 立即修复清单(P0a 中由 lint 检出，P3a 修复)

1. `config_v5_fov90.yaml` 与 `config_v5.yaml` 共用 4 个输出目录。在 P3a 之前，lint 禁止用 fov90 配置跑 stage1、stage2 和 backbone_kv。
2. 缺失 `split_manifest` 时，代码会退回朴素切片。新框架中这种情况直接报错;旧配置由 lint 标出。
3. `tah_cache` 的元数据不全，`tah_train` 只检查 `d_feat`。两者改由 manifest 和兼容性检查(10.1、10.2)取代。
4. 开局 ckpt 从 `rollout/eval_out/` 迁入产物库,`AttrScorer` 改为按产物 ID 加载 lin 和 ft 两个 ckpt。

### 10.5 旧产物导入

`acot artefacts import` 为以下产物建立条目：
- `stage2_best.pt`、`tah_relm_dag.pt`;
- `opening_attr_{cls,ft}.pt`;
- `tah_relm_v90_*`、`off90.pt`、`dag_*.pt`。

每个条目都写明 `provenance_gaps`,例如「无配方」「源数据已删除」「离线部分 FOV 错位」「tick_s 混用」。原文件留在原处，产物库只保存引用和 sha256。

---

## 11. 数据生成(minor 4)

```yaml
# configs/datagen/mvp_full_v5_fov90.yaml
name: mvp_full_v5_fov90
scenario: carla_uav_tracking/config/scenarios/mvp.yaml       # 快照后计算哈希
camera: {fov_deg: 90, width: 512, height: 512, fps: 10}
plan: {source: from_dataset_manifest, ref: mvp_full_v5}       # 也可以用 explicit 列出 [{seed, town}]
retry: {same_seed_attempts: 2, substitute: {pool_start: 108, same_town: true, max_per_slot: 6}}
qc: [attrs_match_camera, target_moving, min_frames, review_contract]
output: {staging: /nvidia/hque/data/carla_data/.staging/mvp_full_v5_fov90, final: /nvidia/hque/data/carla_data/mvp_full_v5_fov90}
```

流程如下:
1. **冻结计划**:把「种子 → 城镇」的显式列表写成 `plan.json` 并计算哈希，生成过程中不可修改。
   - 这样可以避免按 `ep_id // 10` 选城镇导致的城镇分布塌缩。
2. **分配 worker**:由 infra 的租约分配 CARLA 端口和 TM 端口。
3. **逐集生成**:每集写入 staging 目录，先通过 QC。
4. **重试与替补**:
   - 同一种子先按原计划重试;
   - 仍然失败，就从替补池中取新种子补位，城镇保持不变;
   - 所有替补和未补上的空位都写入 manifest。
5. **入库**:全部通过 QC 后，原子性地移入正式目录，并写出数据集 manifest。
6. **划分**:由 `acot data split` 显式生成，不会自动生成。

这套流程把 `gen_seed_town_list.sh` 和 `b_fill.py` 的做法固化为规范，因为 CARLA 的世界状态不可复现，同一种子重新生成时可能被判为「目标静止」而跳过。

---

## 12. 结果与分析(M5)

### 12.1 失败即关闭

- **默认要求**:`completeness.require: all_complete`,即参与比较的每个臂，在每个预注册种子上都必须是 `complete`。不满足就报错，并打印缺失表(臂 × 种子 × 状态 × 原因)。
- **允许跳过**:只有实验显式声明了 `allow_skipped_reasons` 和 `pairing: intersection` 时，才按交集配对。报告表头写明 n,以及每个臂被剔除的种子和原因。
- **partial**:永远不进入分析。
- **替补种子**:评测中不会自动替补，否则会破坏配对。如果要替补，必须由种子登记表显式给出，并且所有臂使用同一组替补。

### 12.2 旧结果导入(确定性的优先级规则)

1. **按臂收集文件**:
   - 有合并文件 `paper_it_<name>.json` 就用它;
   - 否则按 part 编号的**数值**排序收集各个 part,不限于个位数;
   - 同一 part 中，最终 `.json` 优先于 `.json.partial`。
2. **跨 part 去重**:同一种子在多个 part 中出现时，取按 part 顺序第一条 complete 记录。如果多条记录内容不同，记入警告。
3. **状态与留痕**:
   - 带 `skipped` 标记或未评分的记录，映射为对应状态;
   - 每次导入都写出 `import_report.json`,列出所用文件、所做选择和警告;
   - 每个旧臂标注 `protocol_tag ∈ {gt_seeded, deploy_v1, deploy_v2_head, final_v2}` 和来源脚本;对应关系维护在 `legacy/arm_map.yaml` 中。
4. **旧 tick 日志(不带种子)**:导入器复现按顺序对齐的逻辑，并用步数(max t ≤ 评分器的步数)交叉校验，对齐结果标为 `by_order_verified` 或 `by_order_unverified`。

**已知的旧脚本缺陷**(L2 必须单独列出的差异来源):

| 脚本 | 缺陷 | 实际影响 |
|---|---|---|
| `opening_cl.py` | 取 `sorted(glob)[-1]`,会优先读 `.partial`;`_p[0-9]` 只匹配个位数 part | 实测 54 个同时有 final 和 partial 的 part,种子列表全部相同;没有 ≥10 的 part。所以现有数字不受影响，但写法脆弱 |
| `tick_align.aligned` | 按顺序对齐 | 由 `by_order_verified` 覆盖 |

### 12.3 指标与统计(每项只有一份实现)

- **指标**:
  - `committed` = 1 − `mis_follow_sustained`,为主口径;
  - `lenient`、`strict` 为辅助口径;
  - `longC`、`longW` 按评分器逐帧计算;
  - `opening_lock` 由协议推导，取 t ≥ 开局窗口结束 tick 之后、第一个满足条件的 tick。
- **统计**:
  - 只保留配对 bootstrap、符号翻转检验、McNemar、Wilson 区间各一份实现;
  - B = 20000,RNG 种子 = 实验名的哈希;
  - 同一 `family` 内的多重比较做 Holm 校正。
- **报告**:diagnostic 臂只能出现在 `family: diagnostic` 中;主表的标题自动注明「仿真代理感知:…」。

---

## 13. 基础设施(M6)

### 13.1 硬性约束(写入 schema,由 preflight 强制执行)

1. **网络**:只使用 `infra.docker.network`(`cyh-carla-net`),先用 `docker network inspect` 确认存在。代码中没有任何创建网络的路径;`docker network create` 和 compose 都列入禁用命令，lint 会扫描。
2. **容器**:
   - CARLA 只用 `docker exec cyh-carla`;
   - 只有 policy 容器允许 `docker run`,必须挂在上面的网络上，并以 `policy_container_prefix` 开头;
   - DAM4SAM 只用 `docker start` 和 `docker exec`;
   - 下载一律在宿主机上做。
3. **CARLA 客户端的 Python**:
   - 固定为 `cyh-carla` 中的 `python`(3.8.0)。`python3` 是 3.6.9,不能用,`env.py` 中的 `from __future__ import annotations` 需要 3.7 及以上;
   - preflight 运行 `docker exec cyh-carla python -c "import sys; assert sys.version_info[:2]==(3,8)"`。
4. **GPU**:
   - policy 只用 `policy_gpus` 中的卡，并跳过 `reserved` 中的卡;
   - preflight 读取 `nvidia-smi --query-compute-apps`,目标卡上有不属于本调度器的进程时发出警告，显存不足时报错。
5. **端口**:
   - CARLA 端口和 adapter 来自 `carla_servers`;
   - TM 端口段按 worker 和重启次数分配(`base + 30·w + 10·restart`),不会冲突;
   - preflight 检查端口是否被占用，以及是否落在 `reserved` 范围内。
6. **租约**:所有资源用文件锁登记在 `runs/infra/leases/`,并发的 `acot eval` 不会互相抢占。

### 13.2 preflight 清单(`acot eval` 开始前和 `acot lint --infra` 时执行)

1. 网络存在;`cyh-carla` 正在运行;镜像存在，并记录 image ID。
2. CARLA 客户端的 Python 版本，以及 `import acot.config, acot.eval.episode` 能在 `cyh-carla` 中成功执行(验证 py3.8 兼容)。
3. GPU 和端口(见上)。
4. 只要有臂用到 DAM4SAM:
   - 每个实例都要通过健康检查，方法是 TCP 连接加一次 `close` 往返;
   - lifecycle 为 managed 时，自动 start/exec 并等待服务就绪;
   - 每个实例同时只租给一个 worker,因为服务只接受单个连接。并行跑 DAM4SAM 臂需要启动多个实例(不同端口)。
5. 数据和 HF 缓存路径可读;结果目录可写。

### 13.3 py3.8 兼容检查

- `tests/test_py38.py` 用 `ast.parse(src, feature_version=(3, 8))` 解析 CARLA 侧导入闭包中的全部模块，包括 `acot/config/**`、`acot/infra/**`、`acot/eval/episode.py`、`acot/analysis/results.py`,以及 `rollout/{env,run_rollout,remote_policy,bridge,metrics,candidates}.py`。
- 语法检查发现不了运行时 API 的差异，所以 preflight 中真实 import 的那一步仍然必要。
- P4 再修正 `env.py`、`bridge.py` 中过时的 py3.6 注释。

---

## 14. 迁移计划与工期(minor 1)

| 阶段 | 内容 | 验收 | 估时 |
|---|---|---|---|
| **P−1 基线** | 两个会话的未提交改动先提交到一个基线分支(需用户同意);定好 `env.py`、`policy.py`、`scene_manager.py` 的文件所有权 | 有可引用的基线 git sha | 0.5 h |
| **P0a 清单与 schema** | `legacy_surface.py`(已完成)和 `legacy_map.yaml`;参数使用矩阵;schema、load、resolve、哈希;种子登记;infra schema;lint 检查 FOV 冲突、种子重叠、输出目录冲突、缺失划分 | `test_legacy_surface` 全部通过;lint 检出 §10.4 中已知的冲突;哈希稳定性测试通过 | 1.5 d |
| **P0b 翻译器与黄金测试** | `legacy.py`;`acot show` 和 `--dry-run`;docker shim;所有现有臂的方法 YAML | L0 对 §9 中列出的全部启动脚本逐字节一致 | 1.5 d |
| **P1a 调度与运行时** | 调度器、租约、preflight;按种子落盘的结果、状态机、重试和看门狗;reset 携带种子(只改日志);`server_echo`/`client_echo`;在线协议置空和真值审计;评分器参数显式化;去掉启动时的 pip install | L1 对每个臂都相等;L3 冒烟通过(覆盖 3 个臂) | 2.5 d |
| **P1b 分析与导入** | metrics、stats、report;旧结果导入器;`arm_map.yaml` | L2 与 4 个旧分析脚本的数字一致(差异只来自 §12.2 列出的缺陷) | 1.5 d |
| — | **到这里，1004 的重跑就能全部走新入口**(内部仍是旧代码，但留痕完整) | | 累计约 7.5 d |
| **P2 policy/env 直接读配置** | `--resolved`;`PolicyInput`/`ScoringInfo` 拆分和 `SimProxyPerception`;一旦设置旧的行为环境变量就报错;去掉不安全的默认值;`_cand_lite` 补传 `gt_idx` | 与 P1 路径对照，L0–L3 全部通过;§7.4 的测试全部通过 | 2.5 d |
| **P3a 产物库** | 产物库、manifest、兼容矩阵;旧产物导入;§10.4 的修复 | 兼容检查能拦下「70° 数据按 90° 用」和「0.2 s 与 0.6 s 混用但未豁免」 | 1.5 d |
| **P3b prep、train、datagen** | tah_seq、vlm_ctx、merge、opening 的适配器;tah_relm 和 opening_attr 的训练适配器;datagen 执行器;camera、crops、describe 去重 | 在 2 个 episode 上重建 tah_seq,张量哈希与现有缓存一致(同一 FOV 配置);tah_train 在相同种子下指标误差 ≤ 1e-6(如果 GPU 不保证逐位一致，报告实际差异);datagen 对 2 个种子 dry-run,计划和 QC 输出可复现 | 2.5 d |
| **P4(可选)** | 旧脚本移入 `legacy/`;按组件拆分 `policy.py`(开局 / 身份 head / 滤波 / WHERE);修正过时注释 | 一致性测试全部通过 | 3–5 d |

- P0a–P3b 合计约 13.5 个工作日，不含 P4。估时包括 CARLA 上的一致性运行，但不包括等待 GPU 的时间。
- 每个阶段完成后都可以单独使用。
- P2 和 P4 涉及另一会话也在修改的文件，需要协调后在独立分支上进行。

---

## 15. 测试清单

- **配置**:
  - schema 严格性(未知字段、缺失字段、当前 head 之外的子配置);
  - `extends` 和覆盖的语义;
  - 覆盖范围限制;
  - 哈希稳定性(键顺序和空白不影响哈希;infra 不计入)。
- **收编**:`test_legacy_surface`(双向覆盖)和翻译器黄金测试(L0)。
- **种子**:用途之间不重叠;test 种子加锁;任意列表和替补;`exclude`。
- **真值**:§7.4 的 6 项测试。
- **运行时**:状态机，用假 env 模拟跳过、失败、卡死、中途死亡;续跑;原子写入;租约冲突;preflight,用假的 docker 和 nvidia-smi。
- **离线**:manifest 完整性、内容寻址冲突检测、兼容矩阵(包括 `unknown` 和豁免)、分片完整性。
- **分析**:失败即关闭;交集配对的声明;旧结果导入的优先级;metrics 和 stats 与现有实现一致(L2)。
- **py3.8**:`test_py38` 加 preflight 中的真实 import。
- **执行**:仓库没有 CI。提交前手动运行，或用 git hook 运行 `python -m acot lint && python -m pytest tests/ -q`;需要 CARLA 的测试(L3)单独用 `-m carla` 标记。

---

## 16. 风险与协调

| 风险 | 应对 |
|---|---|
| 两个会话同时修改 `env.py`、`policy.py`、`scene_manager.py` | P−1 先定所有权;P1a 的修改只涉及日志和回显，并且集中在新文件中;P2 和 P4 在独立分支上做，合并前跑一致性测试 |
| CARLA 不确定性 | L3 只比较调用、种子、场景和评分;轨迹层面的比较放到可选的 L3+ |
| 调用点审计误报 | 白名单按「文件 + 函数 + 规范化源码行」作键;新的访问点必须审核登记(棘轮)。P2 的结构化拆分完成后，审计只保留为冗余检查 |
| 过度工程 | P0–P1 不改算法代码;抽象只服务于已经出现过的故障模式(§2 中的 12 条) |
| 历史数字无法完全复现 | 旧产物和旧结果导入时，如实写明缺口和对齐方式;不追求复现 `stage2_v5` 的训练过程 |
| 估时偏乐观 | P0 拆成 a、b 两段;P1 完成就是可用的里程碑;每个阶段结束时重新估时 |

---

## 17. 需要用户决定的事项

1. **P−1**:是否同意先把当前未提交的改动(包括另一会话的)提交到基线分支。
2. **开局模块的归属**:本稿把它放在协议的 `frontend`,所有臂默认共享，只有开局消融臂可以覆盖。
3. **test 加锁**:使用 test 种子是否强制要求 `--allow-test` 并引用预注册文档。
4. **实施顺序**:是否先做 P0a–P1b(约 7.5 天)再开始 1004 的重跑;还是重跑先用旧脚本进行，框架并行推进。
5. **P1 的日志改动**:reset 携带种子，ASD、trk、开局、DAgger 日志加 `seed` 字段。这些改动只影响日志，但会修改另一会话也在用的 `policy.py` 和 `remote_policy.py`。

---

## 附录 A:Codex R1 意见对照

| R1 | 意见 | v1 的处理 | 位置 |
|---|---|---|---|
| M1 | §5 的映射不完整 | 由脚本生成全清单(6 节，共 349 行);`legacy_map.yaml` 双向覆盖测试;补齐 R1 点名的全部 flag、JSON 字段、启动脚本参数、`shuffle_seed`、ADAPTER 表;写明隐式语义;P2 起未知字段报错 | §6、`1004_legacy_surface.md` |
| M2 | 真值隔离不够机械 | 通道分类表;`PolicyInput`/`Detection`/`ScoringInfo`/`OracleHandle` 接口(P0 定稿);P1 在线协议置空和调用点审计棘轮;P2 结构化拆分;6 项测试 | §7 |
| M3 | 一致性标准太弱 | L0 用 docker shim 做调用逐字节比较;L1 对象级回显;L2 旧结果的指标逐位一致加 tick 对齐一致;L3 比较种子、场景指纹、跳过集合、真值访问;reset 携带种子;评分器参数显式化和两个 `commit_k` 的语义 | §5.1、§8.4–8.6、§9 |
| M4 | 离线兼容元数据不足 | 11 类产物的 manifest 字段;两道边界的兼容矩阵;`unknown` 不算匹配;内容寻址;分片完整性;共用目录、划分回退、缓存跳过、`d_feat`-only 检查四项修复 | §10 |
| M5 | 结果完整性和 partial 规则 | 每个种子 4 种状态;默认失败即关闭;交集配对需显式声明;导入优先级确定;实测 54 个 part 的 final 与 partial 一致 | §8.2、§12 |
| M6 | 基础设施约束不可执行 | infra schema;网络只检查不创建;只有 policy 容器可以 `docker run`;CARLA 客户端用 `python`(3.8,`python3` 是 3.6.9);GPU 和端口租约;DAM4SAM 单连接实例的租约、健康检查和生命周期;py3.8 语法检查加真实 import | §5.6、§13 |
| m1 | 估时偏乐观 | P0 拆成 a、b;总估时从 5–6 天改为约 13.5 天;7.5 天时有可用里程碑 | §14 |
| m2 | 种子需支持任意列表 | 支持 `ranges`、`list`、`list_file`、`exclude` 和替补映射 | §5.4 |
| m3 | 「可部署 v2」措辞过强 | 改称 `sim_proxy_v2`(仿真代理感知协议);`deploy_v2` 只作别名 | §1、§5.2 |
| m4 | `acot data gen` 规格不足 | 冻结计划、重试和替补策略、端口和 TM 分配、QC、staging 后原子入库 | §11 |
