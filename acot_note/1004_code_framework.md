# 1004 · 代码框架设计：离线训练与闭环验证的统一入口

> **版本**:v3 定稿(2026-10-04)。**Codex 第 3 轮结论 APPROVE-WITH-MINOR,无 major,审核通过。**
> - R1 结论 MAJOR-REVISIONS(6 major、4 minor);
> - R2 结论 MAJOR-REVISIONS,R1 中 3 条 major 和全部 minor 判为 RESOLVED,新提出 3 条 major、3 条 minor;
> - R3 结论 APPROVE-WITH-MINOR,R2 的 3 条 major 和 3 条 minor 全部判为 RESOLVED,另提 2 条 minor 和 1 条建议。v3 已全部吸收(附录 C),这些小改动没有再送审。
>
> 审核原文在 `acot_note/codex/1004_code_framework_R{1,2,3}.md`,历史版本在 `acot_note/codex/1004_code_framework_v{0,1,2}.md`;逐条对照见附录 A(R1)、附录 B(R2)、附录 C(R3)。
>
> **配套文件**:
> - `acot_note/1004_legacy_surface.md`:旧配置面全清单，自动生成;
> - `acot/tools/legacy_surface.py`:清单生成器，只依赖标准库。
>
> **范围**:只做设计，不改算法。实施按 §14 分阶段进行，每个阶段都有验收标准。
>
> **实施状态(2026-10-04)**:P0a–P1b(§18)、P3a–P3b(§19)、P2(§20)均已完成并通过验收;P4 按「不改共享文件」处理(§21)。闭环默认使用 acot 运行时，其决策与旧代码逐位相同(§20.3)。旧代码一行未改，未提交 git,未执行消融或基线的闭环重跑;待用户决定的事项见 §21.2。

---

## 0. 一页摘要

**问题。** 现在一个闭环臂的实际配置散落在 6 个地方：
- 36 个 `policy_server` flag;
- 14 个 `run_rollout` flag;
- 16 个行为环境变量;
- 3 个 JSON 字符串(`ASD_CFG`、`TRK_CFG`、`OPENING_CFG`),展开后进入 4 个构造函数，共 43 个可配置字段;
- 约 100 个 shell 脚本里写死的参数;
- 代码里的硬编码常量。

离线侧也有对应的问题：缓存、划分和 checkpoint 都没有可校验的元数据。后果已经出现过多次：
- FOV 70°/90° 错位;
- 离线 0.2 s 与闭环 0.6 s 的 tick 间隔不一致;
- 空 `ASD_CFG` 让一个臂悄悄跑成了 bayes 消融;
- 两个 YAML 共用输出目录。

**方案。** 只保留一个入口:`python -m acot {data,prep,train,eval,analyze,lint,show}`。各类对象的分工如下：
- **协议**:带版本、冻结，只包含所有臂共享的仿真、感知代理、前端和评分设置;
- **方法**:包含所有因臂而异的 policy 行为参数，由严格 schema 校验;
- **实验**:引用协议、方法和种子集;
- **基础设施**:单独成文件，不计入科学哈希。

新框架的文件全部放在 `acot/` 下(仓库根目录的 `configs/`、`tests/`、`scripts/`、`src/` 属于另一个项目)。CARLA 侧的代码放在 `carla_uav_tracking/acot_client/`,因为 `cyh-carla` 容器只挂载了这个目录。

迁移采用 strangler 模式：先让翻译器生成旧代码需要的参数，用一致性测试证明行为没变，再让内部直接读取配置对象。

**五个关键机制:**
1. **清单驱动的收编**:
   - 全目录扫描(177 个 Python 文件、100 个 shell 脚本),生成旧配置面全清单，约 700 行;
   - 每个顶层目录和根目录脚本都必须归为「扫描」或「排除(附原因)」,出现未归类的顶层项，测试就失败;
   - 每一行都必须在 `legacy_map.yaml` 里有去处，否则测试失败;
   - P2 起，未知的旧环境变量或 JSON 字段直接报错。
2. **机械化的真值隔离**:
   - P0 在配置层固化通道分类;
   - P1 在线协议层置空目标属性，并用审计代理包住原始真值数组(`tpos`、`dstates`、`vis`)和真值标签属性，记录每个读取点。只要出现未审核的读取点就判失败(棘轮)，并用一个故意读 `tpos` 的阳性对照验证审计有效;
   - P2 把接口拆成 `PolicyInput` 和 `ScoringInfo`,方法代码在结构上拿不到真值。
3. **四级一致性验收**:
   - L0:调用参数逐字节一致(docker shim 捕获);
   - L1:对象级配置回显;
   - L2:旧结果导入后，指标与旧脚本逐位一致;
   - L3:实机冒烟，核对种子、场景指纹和跳过集合。
4. **全产物 manifest 与兼容矩阵**:覆盖 11 类离线产物，在训练时和闭环解析时两道边界上做比较。
5. **结果失败即关闭，基础设施可执行检查**:
   - 每个种子记录 4 种状态之一，默认要求完整;
   - 网络、容器挂载、Python 版本、GPU、端口、DAM4SAM 单连接服务都在 preflight 中检查。

**工期。** P0a–P3b 合计约 14 个工作日，不含可选的 P4。其中 P0a–P1b 约 8 天;做完这一段，1004 的重跑就能全部走新入口，内部仍是旧代码，但留痕完整。

---

## 1. 目标与非目标

**目标**
1. **入口唯一**:`python -m acot <stage> <spec.yaml> [key=value ...]`。各方法、各实验臂只在配置上不同。
2. **协议只定义一次**:带版本、冻结;所有臂引用同一个协议对象。
3. **天然留痕**:每个产物都附带以下信息：
   - 已解析配置和配置哈希;
   - 输入产物的 ID;
   - git commit 与未提交改动摘要;
   - 镜像 ID 和库版本;
   - 基础设施快照。
4. **入口处做兼容性检查**:fov、tick_s、特征 ID、depth 尺度、语言模式、划分和数据集，任何一项不一致就报错。
5. **机械化隔离真值通道**:方法代码拿不到评分用的真值。用到真值通道的臂，自动标为 diagnostic。
6. **种子统一登记**:用途之间不允许重叠;test 种子加锁;支持任意种子列表和替补种子。
7. **结果失败即关闭**:按(实验, 臂, 种子)存放，每个种子带状态;默认要求完整。
8. **迁移中不悄悄改变行为**:以 §9 的 L0–L3 一致性验收为准。

**非目标**
- 不改 TAHRelM、PGIF、属性开局、Stage-2 的算法。
- 不重写 CARLA 环境。
- 不接管另一会话的 v6 数据管线(`runs/data_v6/`,清单中标为 other-owner)。

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
| 4 | 启动脚本靠复制粘贴 | 范围内共 100 个 shell 脚本(不含 ticker)。最终协议的那串 TAH flag 原样出现在 12 个脚本里,`TRK_CFG` 赋值出现在 13 个，tempered JSON 出现在 21 个 |
| 5 | 相机和几何参数没有唯一来源 | FOV、分辨率、框尺寸限制写在 YAML 里，另有约 6 处硬编码副本，没有一处和 h5 attrs 核对。投影、裁剪、`describe` 各有 2–7 份实现(框最小值有 8 px 和 10 px 两种) |
| 6 | 划分是隐式的 | `config_v5.yaml` 的 `split_manifest` 指向一个不存在的文件，于是 `dataset.py:25-39` 退回到朴素切片，test 划分为空 |
| 7 | 输出目录冲突 | `config_v5_fov90.yaml` 与 `config_v5.yaml` 共用 `context_cache`、`stage2_context_cache`、`ckpt_dir`、`stage2_ckpt_dir` 四个目录。`backbone_kv.py:63-72,106-116` 遇到已有文件直接跳过，不做任何校验 |
| 8 | tick 间隔不一致 | `tah_cache` 默认 stride 2(0.2 s),而闭环 S2 是 6 帧(0.6 s)。`tah_cache` 只保存 `d_feat`、`order`、`fov`;`tah_train` 只检查 `d_feat` |
| 9 | 分析脚本绑死文件名 | tick 日志按顺序对齐的逻辑有 4 份副本(reset 消息不带种子，跳过的集不触发 reset)。`opening_cl.py` 取 `sorted(glob('*.json*'))[-1]`,会优先读 `.partial`。bootstrap 有 8 份实现，参数各不相同 |
| 10 | 评分器和 policy 脱节 | 评分器的 `commit_k` 没有与 policy 的 `--commit-k` 绑定(`metrics.py:35-41`;`run_rollout.py` 只传了 `fps` 和 `max_lost_s`)。`_cand_lite` 不传 `gt_idx`(`policy_server.py:38-46`),跟踪器 ID 模式下干扰车按跟踪器 ID 计分(`metrics.py:29-32`) |
| 11 | 真值走在线协议上，没有隔离 | env 每个 tick 都把 `target_color`、`target_bp`、`gt_candidates=(tpos, dstates, campose, road_pred_world, vis)` 发给 policy(`remote_policy.py:27-34`,`env.py:291-299`)。`is_target`、`true_idx` 在 policy 进程内由 `build_candidates` 生成;`tracklets=sim` 时，候选的 `idx` 本身就是仿真 actor 编号(目标恒为 −1) |
| 12 | 基础设施约束只写在注释里 | 网络名写死在脚本里。`cyh-carla` 中 `python` 是 3.8.0、`python3` 是 3.6.9,注释却还写着 py3.6;该容器只挂载 `carla_uav_tracking → /workspace` 和数据目录。DAM4SAM 服务只接受单个连接、靠手动启动，当前已退出 |

---

## 3. 设计原则

1. **单一事实来源**:所有影响行为的量都来自已解析的配置对象。代码里只允许出现 schema 中声明为 `safe_default` 的默认值。
2. **严格 schema**:配置里出现未知字段就报错。当前没有在用的 head,不允许出现它的子配置。
3. **组合，而不是复制**:所有方法都 `extends: _base`;实验臂只写与基准方法的差异。覆盖记录写进 manifest,并纳入哈希。
4. **归属清晰**:协议只放所有臂共享的不变量，方法放所有因臂而异的 policy 行为参数(§5.1)。
5. **清单驱动**:旧配置面由脚本全目录扫描生成，翻译器和 schema 必须覆盖清单的每一行(§6)。
6. **机械化防线优先于约定**:真值隔离、兼容性检查、结果完整性都由代码和测试强制执行;能力边界如实写明(§7.3)。
7. **失败即关闭**:缺少划分、产物 manifest 不一致、种子不完整、基础设施冲突时，一律报错;不做静默回退。
8. **先适配、后重构(strangler)**:P0/P1 期间不改算法代码，只允许改日志和回显;用 §9 的一致性验收证明行为没变。
9. **命名空间与运行位置**:

   | 位置 | 内容 | Python 与依赖 |
   |---|---|---|
   | `acot/`(仓库根目录) | 新框架主体 | 宿主机 py3.12(调度、分析、lint);policy 容器 py3.11(容器挂载整个仓库) |
   | `carla_uav_tracking/acot_client/` | CARLA 侧代码 | 必须兼容 py3.8;只依赖标准库、PyYAML(容器内为 6.0.3)和 carla |

   CARLA 侧只读取宿主机解析好的 `client.json`,不 import `acot`。

---

## 4. 包结构与 CLI

```
acot/                                 新框架(仓库根目录;避开另一项目的 configs/ tests/ scripts/ src/)
  __main__.py, cli.py                 python -m acot <stage> <spec> [overrides]
  config/
    schema.py                         dataclass:Protocol / Method / SeedRegistry / Experiment / Infra / DataGenPlan / DatasetSpec / PrepSpec / Recipe
    load.py                           YAML 读取、extends、深合并、key=value 覆盖、严格校验
    resolve.py                        解析引用(protocol@ver、method、seedset、artefact:<id>)、归属和覆盖检查、兼容性检查、真值通道判定、规范化哈希
    legacy_map.yaml                   清单的每一行 → 新字段或 infra / collection / artefact / analysis / test / other-owner / rejected(§6)
    gt_sites.yaml                     已审核的真值读取点白名单(§7.3)
    legacy.py                         翻译器：已解析配置 → 旧 policy_server / run_rollout 的 argv 和环境变量(P0–P1)
  infra/
    preflight.py                      网络、容器、挂载、镜像、GPU、端口、DAM4SAM 健康检查、CARLA 客户端 Python 版本(§13)
    lease.py                          CARLA 服务器、TM 端口段、policy GPU、DAM4SAM 实例的租约(文件锁)
  runtime/                            policy 进程内部
    audit.py                          (P1)AuditedArray / AuditedTuple / 标签属性审计;每个 episode 结束时写出 gt_access
    io.py                             (P2)PolicyInput / Detection / ScoringInfo / OracleHandle(§7)
    perception.py                     (P2)SimProxyPerception:线协议消息 → (PolicyInput, ScoringInfo)
  eval/
    scheduler.py                      (臂, 种子)工作队列、worker 池、看门狗、重试、跳过、容器生命周期
    legacy_server_entry.py            (P1)policy 进程包装入口：参数断言、对象级配置回显、真值审计
    collectors.py                     tah_dagger / opening_snapshots / actor_labels / dam4sam_diag / ticks / trk
  artefacts.py                        产物库(内容寻址)、manifest、别名、旧产物导入(带 provenance_gaps)
  provenance.py                       git sha、未提交 diff 哈希、镜像 ID、库版本、主机、GPU、时间
  data/                               dataset manifest、split、camera(唯一投影)、crops(唯一裁剪和特征提取，带 feature_id)、language(唯一 describe)
  prep/                               vlm_ctx、tah_seq、merge、opening_snapshots
  train/                              tah_relm、opening_attr_lin、opening_attr_ft 适配器;stage2 先只导入旧产物
  analysis/                           results(按(臂, 种子)读取并检查状态;旧结果导入)、metrics、stats、report
  configs/                            protocols/ methods/ seeds.yaml experiments/ infra/ datagen/ datasets/ prep/ recipes/
  tools/legacy_surface.py             旧配置面清单生成器(已完成)
  tests/                              单元测试、黄金测试、一致性测试(§15)
  legacy/                             迁移后的旧脚本(只读，仍可运行)
carla_uav_tracking/acot_client/       CARLA 侧(py3.8):episode.py(把 run_rollout 的循环体提取为 run_one)、wire.py(RemotePolicy 子类：置空、reset 带种子)、echo.py
carla_uav_tracking/rollout/eval_out/exp/   结果根目录;宿主机、policy 容器、cyh-carla(/workspace/rollout/eval_out/exp)三处可见
```

**CLI**

| 命令 | 作用 |
|---|---|
| `acot lint` | 校验全部配置:schema、归属和覆盖、种子重叠、FOV 冲突、输出目录冲突、缺失的划分、清单覆盖率 |
| `acot show <spec> [--arm a] [--diff]` | 打印已解析配置、配置哈希、相对 `_base` 的差异、真值通道分类、兼容性检查结果，以及翻译后的旧命令 |
| `acot data gen datagen/<x>.yaml [--dry-run]` | 按冻结计划生成数据(§11) |
| `acot data manifest/split …` | 生成数据集 manifest;显式生成划分(不会自动生成) |
| `acot prep <kind> prep/<x>.yaml` | 构建缓存产物(vlm_ctx、tah_seq、merge、opening_snapshots) |
| `acot train recipes/<x>.yaml seed=0` | 训练，产出 ckpt 产物(best、last、metrics、manifest、compat) |
| `acot eval experiments/<x>.yaml [--arms] [--seeds] [--dry-run] [--allow-test]` | 闭环评测(§8) |
| `acot analyze experiments/<x>.yaml` | 生成报告(§12) |
| `acot results import-legacy acot/legacy/arm_map.yaml` | 导入旧结果(§12.2) |
| `acot artefacts import/ls/verify` | 导入、列出、校验产物 |

---

## 5. 配置 schema

### 5.1 字段归属、覆盖规则与哈希(R2 M1)

**归属模型**:协议只放所有臂共享的不变量;凡是因臂而异的 policy 行为参数，一律放在方法里。

| 字段组 | 归属 | 臂级覆盖 |
|---|---|---|
| `camera`、`sim`(fps、s2_period、steps、town、scenario、max_lost_s、warmup、retarget_tick) | 协议 | **不允许**;修改即新协议版本 |
| `scorer`、`privileged_perception` | 协议 | **不允许** |
| `perception`(候选、遮挡、identity_keys、tracker、where_localisation) | 协议 | 只有声明了 `ablation: perception` 的臂可以覆盖 |
| `frontend`(vlm 的 config、ckpt、language_mode;crop_features;opening) | 协议 | 只有声明了 `ablation: frontend` 或 `ablation: opening` 的臂可以覆盖 |
| `control`(ex_source、frame_gain、commit_k、ablate、shuffle_seed) | 方法 | 允许 |
| `identity`、`filter` | 方法 | 允许 |
| `diagnostics` | 方法 | 允许(该臂自动标为 diagnostic) |

1. **解析顺序**:协议 → 方法(`extends` 链，根为 `_base`)→ 臂级 `override` → 命令行 `key=value`。
2. **`_base` 方法**:固定所有方法共享的默认值(`control`、`identity.memory_seed`、`filter.mode: off` 等)。`acot show --diff` 列出每个臂相对 `_base` 的全部差异，臂之间的不同点一目了然。
3. **覆盖范围**:
   - 臂级覆盖协议字段，按上表执行，越权即报错;
   - 声明了 `ablation` 的臂在报告中自动归入消融分组。
4. **命令行覆盖**:默认只能改 `infra.*` 和种子选择。改科学字段必须加 `--ad-hoc`,该运行在 manifest 中标为 ad hoc,不能进入预注册的比较。
5. **哈希**:`config_hash` = sha256(规范化 JSON)取前 12 位，规范化对象是协议、方法和种子列表。
   - 规范化方式：键排序、无空白、浮点数用 repr、不包含未激活 head 的子配置。
   - `infra` 和采集器声明不计入哈希。采集器不改变决策，这一点由 §7.3 的审计保证。
6. **两个 `commit_k`**:
   - `protocol.scorer.commit_k` 是**指标定义**,对所有臂固定;
   - `method.control.commit_k` 是 policy 的提交滞后参数，`_base` 中设为 5。
   - 方法改了它，resolve 会在 manifest 和报告中写明「policy K ≠ 指标 K」;指标仍按协议的 K 计算，保证各臂可比。
7. **`ablate` 需要配套的 ckpt**:
   - `method.control.ablate`(H1/H2 消融：把 EAR 或 IAR 的输出置零)要求配套使用带同样消融训练出来的 Stage-2 ckpt;
   - 所以该臂还必须声明 `ablation: frontend` 并覆盖 `frontend.vlm.stage2_ckpt`;
   - resolve 检查 ckpt manifest 中的 `ablate` 字段是否与之一致。

### 5.2 Protocol:`acot/configs/protocols/sim_proxy_v2@1.yaml`

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
  identity_keys: tracker                    # = --tracklets tracker;sim(actor 编号作键，目标恒为 −1)只允许出现在 deploy_v1 协议中
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
scorer: {commit_k: 5, t_fail_s: 2.0, central_frac: 0.3, primary: committed}
privileged_perception: [cand_box_gt_projection, cand_depth_gt, occlusion_raycast_gt, task_vehicle_set, where_world_gt_matched_box]
```

### 5.3 Method

```yaml
# acot/configs/methods/_base.yaml —— 所有方法共享的默认值;每个字段都显式写出
name: _base
control: {ex_source: track, frame_gain: 0.0, commit_k: 5, ablate: null, shuffle_seed: 0}
identity:
  memory_seed: committed                    # --gallery-seed;取 gt 即为真值通道
  conf_tau: 0.0
  conf_consensus: 0.0
  reanchor: {mode: off, patience: 5, tavg: 0.0}       # mode: off | lang | oracle(oracle 为真值通道)
  cold: {window: 0, lambda: 0.9, single: false}        # COLD_*,只用于复现 V2 实验
filter: {mode: off}
diagnostics: []
```

```yaml
# acot/configs/methods/ours_final.yaml
name: ours_final
extends: _base
identity:
  head: tah
  tah: {ckpt: "artefact:tah_relm_dag", lang_reanchor_thr: 0.60}
  conf_tau: 0.5                             # 提交门控(head=reid 且开启重锚时，还兼作低置信阈值)
filter:                                     # ASD / PGIF;ASD 构造函数的全部字段，显式写出
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
```

```yaml
# acot/configs/methods/ext_dam4sam.yaml
name: ext_dam4sam
extends: _base
identity:
  head: dam4sam
  dam4sam: {model: sam21pp-L, gate_px: 160, every_frame: false, reinit_cooldown: 3, service: dam4sam}   # gate_px → --assoc-gate-px(DAM4SAM 复用该参数)
  conf_tau: 0.05
```

方法 YAML 覆盖当前所有臂：
- 主方法 `ours_final`;
- H-A,即 `ours_final` 加开局消融 `module: head`;
- 开局变体 v2–v4;
- PGIF 的各个变体;
- 外部基线：DAM4SAM、DeepSORT/OC-SORT 关联、reid;
- 语言 head:xattn;
- 诊断臂:oracle_id、oracle_absent、gt_seeded、road_oracle、attrbind、ex_source 消融。

`tid_head` 每个取值实际读取哪些 kwarg,在 P0a 生成「参数使用矩阵」(例如 DAM4SAM 复用 `assoc_gate_px`)。schema 按这个矩阵校验：只允许当前 head 用到的字段出现。

### 5.4 种子登记:`acot/configs/seeds.yaml`

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

- 种子可以用 `ranges`、`list`、`list_file` 三种方式给出，也可以组合;另有 `exclude` 和替补映射。
- 实际种子以迁移时核对 `collect_dagger.sh` 等脚本的结果为准。
- 不同用途的种子集重叠时,`acot lint` 报错。
- 使用 test 种子必须加 `--allow-test`,并引用预注册文档(`prereg_ref`)。

### 5.5 Experiment

```yaml
# acot/configs/experiments/1004_final_table.yaml
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

### 5.6 Infra:`acot/configs/infra/default.yaml`(不计入哈希;§13 详述)

```yaml
docker:
  network: cyh-carla-net              # 必须已存在;只检查，绝不创建
  carla_container: cyh-carla          # 只用 docker exec;挂载:carla_uav_tracking → /workspace,carla_data → /data
  policy_image: acot-uav-train:cu118  # 运行时记录 image ID
  policy_container_prefix: acot-      # 只有 policy 容器允许 docker run(--network 为上面的网络)
python: {carla_client: python, carla_client_version: "3.8", policy: python, policy_version: "3.11"}
results_root: carla_uav_tracking/rollout/eval_out/exp     # 三处可见;runs/exp 为指向它的软链接
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
    start_cmd: "conda activate dam4sam && python carla_uav_tracking/rollout/dam4sam_service.py --model {model} --port {port}"   # {model} 取自臂的 identity.dam4sam.model
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

输出路径不由用户指定，而是由产物库按内容寻址给出。旧脚本里 `--out`、`--cache`、`--ckpt*` 这类参数，在清单中归为 artefact。

---

## 6. 旧配置面的收编(R1 M1、R2 M2)

### 6.1 清单(全目录扫描)

`acot/tools/legacy_surface.py` 用 glob 扫描所有在范围内的目录，通过 AST 和正则提取配置入口。

清单第 0 节对仓库的**每一个顶层目录和根目录脚本**做归类，结果只有三种：扫描、排除(附原因)、未归类。出现未归类项时，`test_legacy_surface` 失败。当前的排除项有：
- 第三方代码(`external/`、`runs/pydeps/`);
- 另一个项目的目录(`src/`、`scripts/`、`tests/`、`configs/`、`docs/`、`.github/`、`result/`);
- 新框架本身(`acot/`);
- 笔记、审核记录和只读结果的作图脚本(`acot_note/`);
- 论文相关(`acot_draft/`、`paper_draft/`、`related-survey/`);
- 工具状态(`.humanize/`、`.git/`);
- ticker 脚本和审核工具。

当前的清单规模:

| 节 | 内容 | 数量 |
|---|---|---|
| 0 | 扫描范围：177 个 Python 文件，按用途分为 runtime 20、offline 50、datagen 23、analysis 68、test 13、other-owner 3;100 个 shell 脚本;排除表 | — |
| 1 | 全部 argparse 参数，以及 14 个使用 `sys.argv` 位置参数的文件 | 394 |
| 2 | 构造函数参数和 JSON 字段:`Policy`(36 个 kwarg,含未暴露的 `shuffle_seed`)、`ASD`(24)、`TrackletAssigner`(7)、`AttrScorer`、`ZoomVerifier`、`RolloutScorer`,以及 `OPENING_CFG` 中被 Policy pop 掉的 6 个字段 | 92 |
| 3 | 环境变量读写，另有 1 处动态键写入 | 25 |
| 4 | env → policy 线协议字段，按真值分类;包括 `cand_lite` 缺少 `gt_idx` 这一已知缺口 | 17 |
| 5 | shell 脚本：调用的入口、docker 操作、出现的全部 `--flag`、写死的行为环境变量、可调参数、位置参数 | 100 |
| 6 | 行为常量和已知冲突 | 26 |

每一行都带一个分类:scientific、diagnostic、privileged、collection、artefact、infra、injected、offline、datagen、analysis、test、other-owner。

shell 脚本一节是静态提取，只作交叉核对。启动脚本实际发出的 argv 和环境变量，以 P0b 用 docker shim 捕获的结果为准(§9 L0)。

### 6.2 `legacy_map.yaml` 与覆盖测试

```yaml
flags:
  policy_server:
    --tid-head:   method.identity.head
    --conf-tau:   method.identity.conf_tau
    --asd:        method.filter.mode
    --asd-cfg:    {json: method.filter, precedence: "非空时整体替换 ASD_CFG(不合并)"}
    --ex-source:  method.control.ex_source        # ≠ track → diagnostic
    --frame-gain: method.control.frame_gain
    --commit-k:   method.control.commit_k
    --ablate:     method.control.ablate           # 需要配套 ckpt(§5.1 第 7 条)
    --device:     infra.policy.device
    # ... 清单第 1 节的每一行
  run_rollout:
    --vis-raycast: protocol.perception.occlusion  # raycast ⇔ True
    --hires:       protocol.frontend.opening.hires_k
    --out:         {artefact: results_path}
  train/tah_train.py:
    --cache: {artefact: recipe.inputs}
    --lr:    recipe.hparams.lr
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
  policy.shuffle_seed: method.control.shuffle_seed
  metrics.RolloutScorer.commit_k: protocol.scorer.commit_k
  # ...
categories:            # 整个文件按类别处理(analysis、test、other-owner 中的选项不迁移，由新分析注册表、测试工具或另一会话负责)
  analysis: {replaced_by: acot.analysis}
  test: {kept_as: test_tooling}
  other-owner: {owner: v6 data session}
rejected: []           # 有意不迁移的通道;必须写明原因，并指向 acot/legacy/ 下的复现方式
```

`acot/tests/test_legacy_surface.py` 的规则：
1. 重新运行生成器;
2. 清单中每一行都必须在 `legacy_map.yaml` 中找到映射(逐项映射，或按类别整体映射);
3. 映射表中指向清单里已不存在的条目，同样算失败;
4. 新增一个在范围内的文件、flag 或环境变量而不登记，测试立刻变红;新增一个顶层目录，则必须加入扫描范围或排除表。

### 6.3 分组映射(要点，完整内容以 `legacy_map.yaml` 为准)

| 旧通道 | 新字段 |
|---|---|
| `--config`、`--ckpt`、`--language-mode` | `protocol.frontend.vlm.*` |
| `--s2-period` | `protocol.sim.s2_period`(不允许臂级覆盖) |
| `--ex-source`、`--frame-gain`、`--commit-k`、`--ablate`、`shuffle_seed` | `method.control.*`(`_base` 中给出默认值)。`ex_source ≠ track` 属于 diagnostic;`ablate` 需要配套的 ckpt |
| 评分器的 `commit_k`、`t_fail`、`central_frac`;`max_lost_seconds` | `protocol.scorer.*`、`protocol.sim.max_lost_s` |
| `--tid-head`、`--tid-ckpt`、`--tah-ckpt`;TAH 语言重锚阈值 0.60 | `method.identity.head`、`.<head>.ckpt`、`.tah.lang_reanchor_thr` |
| `--gallery-seed`、`--conf-tau`、`--conf-consensus`、`--warmup-oracle-s` | `method.identity.memory_seed`、`.conf_tau`、`.conf_consensus`、`diagnostics: [warmup_oracle]` |
| `--reid-feature/-bank/-topm/-tavg/-motion/-motion-gate-px` | `method.identity.reid.*` |
| `--reanchor`、`--reanchor-patience`、`--reanchor-tavg` | `method.identity.reanchor.{mode, patience, tavg}`;`mode=oracle` 属于 diagnostic |
| `--assoc-mode/-gate-px/-lambda` | `method.identity.assoc.*`;DAM4SAM 复用 gate,对应 `identity.dam4sam.gate_px` |
| `--dam4sam-host/-port`、`DAM4SAM_EVERY_FRAME`、冷却 3 tick | `infra.services.dam4sam` 租约、`identity.dam4sam.{every_frame, reinit_cooldown}` |
| `dam4sam_service.py --model`(默认 sam21pp-L,另有 B/S/T) | `method.identity.dam4sam.model`(scientific,计入哈希)。managed 模式由调度器按该值启动服务;两种模式下 preflight 都读取服务进程的命令行核对 `--model`,并写入 manifest(§13.2) |
| `--asd`、`--asd-cfg`、`ASD_CFG`(24 个字段) | `method.filter.*`(全部显式写出;`log_path` 改由采集器决定) |
| `--tracklets`、`--occl-cands`、`--vis-raycast`、`TRK_CFG`(7 个字段) | `protocol.perception.*`;`tracklets=sim` 只允许出现在 deploy_v1 协议中 |
| `--hires`、`OPENING`、`OPENING_CFG`(window、agg、final_reset、reset_asd、force_final、seed_asd,以及 ckpt、wm、wc、wq、ft) | `protocol.frontend.opening.*`(开局消融臂可以覆盖);ft 的 ckpt 路径目前写死，P0/P1 只校验它与 `ckpt_ft` 一致 |
| `--steps`、`--town`、`--episodes`、`--seed-base` | `protocol.sim.*`;种子由实验的种子集提供，调度器负责分配 |
| `--expert-control`、`--predict-road`、`--road-oracle`、`--retarget-tick`、`--event-override`、`ORACLE_ABSENT` | `method.diagnostics: [...]`;`retarget_tick` 属于协议变体 |
| `COLD_WINDOW/LAMBDA/SINGLE`、`V1_SELFRESET` | `method.identity.cold.*`、`method.filter.v1_selfreset`(只用于复现旧实验) |
| `ASD_LOG`、`TAH_DAGGER_DIR`、`DAGGER_DIR`、`OPENING_DUMP`、`DUMP_LABELS`、`DAM4SAM_DIAG` | 实验的 `collect: [ticks, tah_dagger, dagger, opening_snapshots, actor_labels, dam4sam_diag]` |
| `CARLA_PORT/TM_PORT/HOST/ROOT`、`--port`、`--policy-host/-port`、`--carla-port`、GPU、`NTHR`、`SRV`、`NET`、`HF_CACHE`、`STALL_MIN/FIRST_MIN/START_PART`、ADAPTER 表 | `infra.*`(调度器负责分配，不计入哈希) |
| 训练、缓存和数据脚本里的 `--out`、`--cache`、`--ckpt*`、`--save*`、`--output`、`--glob` 等 | artefact:由产物库管理，体现在 recipe、prep spec 的输入产物 ID 和 manifest 中 |
| 训练、缓存和数据脚本里的其余参数(lr、epochs、stride、layers…) | `recipe.hparams.*` / `prep.*` / `datagen.*` |
| `eval_asd.sh` 写死的参数:`--ckpt stage2_best.pt`、`--language-mode neutral`、`--ex-source track`、`--port 5555`、`CONFIG`、镜像、启动时 `pip install timm==1.0.29` | 协议、方法和 infra 字段。P1 的翻译器原样保留启动时的 pip install(L0 逐字节比较);P2 改为在镜像中固定该版本(§14) |
| `SMOKE_*`、`PARITY_*`;analysis 类脚本的选项 | test;由 `acot.analysis` 取代 |

### 6.4 隐式语义(翻译器必须保留，P2 起显式化)

1. `--asd-cfg` 非空时整体替换 `ASD_CFG`,两者不合并。
2. `mode` 和 `log_path` 由 flag 或环境变量注入。JSON 里如果也写了这两个字段，会报 TypeError(参数重复)。
3. JSON 字段是直接展开进构造函数的(`**json.loads(...)`)。未知字段在**运行时**才报 TypeError,而且报错发生在 policy 容器里，容易被忽略。
4. `OPENING_CFG` 先被 Policy pop 掉 6 个字段，剩下的才传给 `AttrScorer` 或 `ZoomVerifier`。
5. `--conf-tau` 一个值有两种用途：提交门控(`policy.py:1071`);在 reid 重锚开启时，还兼作低置信阈值(`policy.py:951`)。
6. `reset()` 每个 episode 都会用固定的 `shuffle_seed`(=0)重新初始化候选打乱用的 RNG(`policy.py:286`)。
7. `run_rollout` 构造评分器时只传 `fps` 和 `max_lost_s`;`commit_k`、`t_fail`、`central_frac` 用的是类默认值。

### 6.5 未知字段的处理(按阶段)

| 阶段 | 规则 |
|---|---|
| P0 | 新 YAML 中的未知字段在 `load` 时就报错。翻译器只输出 `legacy_map.yaml` 中登记过的通道 |
| P1 | `legacy_server_entry.py` 在启动 policy 前比对实际的环境变量和 argv,与翻译器的输出逐字节一致才继续，否则中止 |
| P2 | policy 和 env 只读 `--resolved`。只要设置了任何旧的行为环境变量(清单第 3 节中 scientific、diagnostic、collection 类),就报错 |

---

## 7. 真值通道隔离(R1 M2、R2 M3)

### 7.1 通道分类(清单第 4 节)

| 类别 | 通道 | 规则 |
|---|---|---|
| 可部署 | `rgb`、`rgb_hi`、`language`、`proprio`、`step`、`campose`(自身相机位姿) | 方法可用 |
| 协议允许的理想化感知 | `tpos`、`dstates` 只能经由 `build_candidates` 变成候选框，以及 `_committed_world` 中「当前所匹配框的世界坐标」;`vis` 只用于遮挡过滤;GT 深度;任务车辆集合 | 对所有臂相同，在 `privileged_perception` 中披露;方法只能看到由此**生成的框**和匹配框的坐标，看不到原始状态 |
| diagnostic | `target_color`/`target_bp`(attrbind);`road_pred_world`;`retarget_event`;`ex_source ≠ track` 时用 `tpos` 更新 WHERE;`gallery_seed=gt`;`reanchor=oracle`;`warmup_oracle`;`event_override`;`ORACLE_ABSENT`;`tid_head=oracle`;`tracklets=sim`(用 actor 编号作身份键) | 只有在方法里声明为 `diagnostics` 才能用;该臂标为 diagnostic,只能进入「上界/诊断」分组 |
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

- `SimProxyPerception.process(msg) -> (PolicyInput, ScoringInfo)` 在 policy 进程中运行，作为 `act()` 的第一步，因为它要用到 VLM 的 grid 特征。
- 方法组件(开局、身份 head、滤波、WHERE)的签名只接受 `PolicyInput`,以及可能为 None 的 `OracleHandle`。
- `_cand_lite` 改为由 `ScoringInfo` 生成，并补传 `gt_idx`。

### 7.3 分阶段落实

**P0(配置层)**
- resolve 根据清单分类计算 `uses_gt`。
- 方法用了 diagnostic 通道，却没有在 `diagnostics` 里声明 → 报错;声明了 → 该臂标为 diagnostic。
- `report.py` 拒绝把 diagnostic 臂放进主表分组。

**P1(运行时审计，不改算法代码)**

全部由 `legacy_server_entry.py` 在加载旧 policy 之前安装，旧文件一行不改。

1. **在线协议置空**(CARLA 侧，`acot_client/wire.py`):对非 diagnostic 臂，发送前做如下处理：
   - `target_color`、`target_bp` 置为空串;
   - `gt_candidates[3]`(road_pred_world)置为 None;
   - `retarget_event` 置为 False。

   以下情况放行，放行情况记入 manifest:
   - 方法声明了 attrbind、road_oracle 或 retarget;
   - 实验声明了需要这些标签的采集器(opening_snapshots、dagger)。
2. **原始真值数组审计**(R2 M3):把 `Policy.act` 包一层，进入原函数之前替换 `gt_candidates` 中的 3 个槽位：
   - `tpos`(槽 0)和 `dstates`(槽 1)换成 `AuditedArray`。它实现 `__array__`、`__getitem__`、`__iter__`、`__len__`、`shape`/`ndim`/`dtype`、`__array_ufunc__`/`__array_function__`,把操作转交给内部数组，同时记录调用点。调用点取最近一个代理之外的 Python 栈帧，用(文件、函数、规范化后的源码行)作键。`np.asarray(tpos)` 会调用 `__array__`,这时的上一帧就是调用方。
   - `vis`(槽 4)换成 `AuditedTuple`,并递归包装其中的 `d_vis` 列表(`vis[1]`)。否则代码先解包一次，之后读取干扰车可见性就逃过了审计。
   - `campose`(槽 2)属于可部署通道，不审计。
3. **标签属性审计**:
   - `CandidateSet.true_idx`、`Candidate.is_target`、`Candidate.gt_idx`、`Observation.target_color/target_bp` 替换为带 setter 的属性，只记录读取;
   - `Candidate` 和 `CandidateSet` 都是普通 dataclass,没有 `__slots__`,生成的 `__init__` 通过赋值写入属性，所以替换可行;
   - `dataclasses` 模块内部的读取(repr、eq)归为 bookkeeping。
4. **白名单与判定**:
   - 每个 episode 结束时写出 `gt_access/<seed>.json`;
   - `acot/config/gt_sites.yaml` 中每个读取点都标明类别:privileged、scoring、logging、collection,或 `diagnostic:<名称>`;
   - 可部署臂出现白名单之外的读取点，或出现 `diagnostic:*` 类读取点，该种子记为 failed,原因 `gt_access_violation`;
   - 代码里新增读取点，必须先审核登记(棘轮)。

   初始白名单来自对现有代码的审查:

   | 真值 | 读取点 | 类别 |
   |---|---|---|
   | `tpos`、`dstates` | `candidates.build_candidates` | privileged:生成框 |
   | `tpos`、`dstates` | `Policy._committed_world` 的 tracker/occl 分支 | privileged:匹配框的世界坐标 |
   | `tpos` | `Policy.act` 中 `ex_source ≠ track` 分支的 `estimator.update(np.asarray(tpos))` | `diagnostic:ex_source` |
   | `vis` | `Policy.act` 的遮挡过滤 | privileged |
   | `is_target`、`true_idx` | `build_candidates`(计算 `true_idx`)、遮挡过滤 | bookkeeping |
   | 标签 | `_cand_lite` | scoring |
   | 标签 | DAgger 缓冲、TAH DAgger、开局快照 | collection |
   | 标签 | ASD 和 trk 日志 | logging |
   | 标签 | gallery_seed=gt、warmup oracle、event_override、reanchor oracle、ORACLE_ABSENT、attrbind 各自的读取点 | `diagnostic:<名称>` |

5. **P1a 中必须验证审计本身**:
   - **决策中性**:用 L3 冒烟时录下的 act 消息回放，比较 `Policy.act` 在包装与不包装两种情况下的输出。离散决策(pred_slot、search_mode、提交)必须完全相同;连续动作差异 ≤ 1e-6,即 GPU 本身的抖动范围。
   - **阳性对照**:一个只在测试中使用的方法变体，在可部署臂的决策路径里直接读取 `tpos`,审计必须判为 `gt_access_violation`。`dstates`、`vis` 和各个标签属性也各做一个阳性对照。
   - **能力边界如实写明**:如果某个槽位的代理无法做到决策中性(例如旧代码对它做了代理不支持的操作),该槽位在 P1 中标为「未机械隔离」,实验 manifest 写入 `gt_fence: {tpos: audited|unfenced, ...}`。这时报告中关于「可部署」的表述，必须以 P2 的结构化拆分为准，不能声称 P1 已经隔离。

**P2(结构强制)**
- `PolicyInput` 和 `ScoringInfo` 在结构上分开，方法组件拿不到真值字段，只有 diagnostic 臂才会得到 `OracleHandle`。
- P1 的审计保留为冗余检查。

### 7.4 测试

| 测试 | 内容 |
|---|---|
| `test_policy_input_fields` | `PolicyInput` 和 `Detection` 的字段与禁止字段集合的交集为空 |
| `test_method_no_gt_attr` | 对方法组件模块做 AST 扫描：不能访问禁止字段，不能 import `ScoringInfo` |
| `test_diag_label` | 声明了任一 diagnostic 通道的臂一定标为 diagnostic;用了 diagnostic flag 却没有声明 → resolve 报错;diagnostic 臂进入主表 → 报告报错 |
| `test_wire_blanking` | 可部署臂的 act 消息中,`target_color`、`target_bp` 为空,`road_pred_world` 为 None,`retarget_event` 为 False |
| `test_raw_gt_positive_control` | 在可部署臂中直接读取 `tpos`、`dstates`、`vis`(每个各一例),都必须判为 violation |
| `test_audit_decision_neutral` | 用录下的消息回放，包装与不包装两种情况下的决策完全一致 |
| `test_gt_audit_ratchet` | 未登记的新读取点判为失败;已登记的 logging 或 scoring 读取点判为通过 |
| `test_oracle_handle` | 可部署臂得到的 `oracle` 为 None;diagnostic 臂调用一个没有声明的 purpose → 报错 |

---

## 8. 闭环运行时

### 8.1 流程

1. 宿主机解析实验，得到每个臂的 `resolved.yaml`、`config_hash`,以及给 CARLA 侧用的扁平化 `client.json`;同时做兼容性检查(§10.2)。
2. 基础设施 preflight(§13.2)。
3. 调度器申请租约：CARLA 服务器、TM 端口段、policy GPU、DAM4SAM 实例。
4. 每个 worker 为一个臂起一个 policy 容器：
   - 用 `docker run --network cyh-carla-net`,容器名 `acot-<exp>-<arm>-<w>`;
   - 入口是 `legacy_server_entry.py`(P1)或 `--resolved`(P2);
   - 等待 `server_echo` 返回，并核对其中的哈希(§8.5)。
5. CARLA 侧通过以下命令逐个种子执行：
   ```
   docker exec cyh-carla python /workspace/acot_client/episode.py \
       --client /workspace/rollout/eval_out/exp/<exp>/<arm>/client.json --seed <s>
   ```
   每个种子落盘一组文件(§8.3)。
6. 收尾：释放租约、删除容器、写入实验 manifest。

### 8.2 每个种子的状态机

| 状态 | 判定 | 后续处理 |
|---|---|---|
| `complete` | 评分器写出最终记录 | 进入分析 |
| `skipped` | 场景准备阶段被 env 拒绝：目标静止(`target stationary`),或生成失败(`setup: …`) | 默认不重试，因为重试会改变世界历史，影响可复现性。是否允许跳过由实验的 `completeness` 决定 |
| `failed` | 开局后出现异常、卡死，或真值访问越界;已用完 `infra_attempts` 次重试 | 每次重试前都重启该 CARLA 服务器(沿用 `launch_wd` 的逻辑);记录 traceback |
| `partial` | 进程在一集中途死掉，没有写出最终记录 | 只保留 tick 日志用于排查，永不进入分析 |

状态文件最后写入，并通过 tmp + rename 保证原子性。续跑时，已经 `complete` 或 `skipped` 的种子直接跳过。

### 8.3 结果布局(根目录为 `infra.results_root`)

```
carla_uav_tracking/rollout/eval_out/exp/<exp>/
  experiment.resolved.yaml   manifest.json(git、未提交 diff 哈希、镜像 ID、infra 快照、开始和结束时间)
  <arm>/
    resolved.yaml   client.json   server_echo.json   client_echo.json
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

- `acot_client/wire.py` 中 `RemotePolicy` 子类的 `reset()` 发送 `{"cmd": "reset", "seed": s, "arm": a, "config_hash": h}`。旧版 server 会忽略多出来的字段，所以这一步向后兼容。
- `policy_server` 把 `seed` 交给 `policy.reset(seed=...)`。`seed` 只保存起来写日志，**不进入 `_rng`**:`_rng` 仍然由固定的 `shuffle_seed` 初始化(`policy.py:286`)。
- **不增加任何 reset 调用**:env 调用 reset 的时机保持不变，跳过的集仍然不触发 reset。
- ASD、trk、开局、DAgger 的日志都加上 `seed` 字段，日志按种子作键，不再依赖顺序对齐。
- 这项改动不影响决策，由 L0–L2 和 `test_audit_decision_neutral` 验证(§9)。

### 8.5 配置回显

- **policy 侧(P1)**:`legacy_server_entry.py --resolved R` 依次执行：
  1. 断言 argv 和环境变量与翻译器的输出逐字节一致;
  2. 在 `Policy`、`ASD`、`TrackletAssigner`、`AttrScorer`/`ZoomVerifier` 的 `__init__` 外包一层，记录实际收到的 kwargs;
  3. 安装 §7.3 的审计;
  4. 用 runpy 执行旧的 `policy_server.py`;
  5. 写出 `server_echo.json`,内容包括：
     - 实际收到的 kwargs 及其哈希;
     - 库版本(torch、timm、transformers、numpy);
     - GPU 型号和镜像 ID。

  调度器用翻译器的 `expected_kwargs(resolved)` 计算期望哈希，两者不一致就中止。这样核对的是对象级的有效配置，比单纯比较 argv 更强。
- **CARLA 侧**:`episode.py` 写出 `client_echo.json`,内容包括 steps、town、vis_raycast、hires_k、评分器参数、种子、置空规则。
- **P2 起**:policy 直接读取 `--resolved`,回显的就是已解析配置本身的哈希。

### 8.6 评分器参数

`episode.py` 构造 `RolloutScorer` 时，把 `fps`、`commit_k`、`t_fail_s`、`central_frac`、`max_lost_s` 全部显式传入，值取自 `protocol.scorer` 和 `protocol.sim`,不再依赖类默认值。

### 8.7 采集器

- 由实验配置的 `collect` 声明，输出按种子落盘。
- 需要标签的采集器(DAgger、开局快照)从 `ScoringInfo` 取标签：P1 期间通过 §7.3 的放行机制获得，P2 起改为结构化获取。

---

## 9. 一致性验收(R1 M3)

| 级别 | 内容 | 通过标准 | 允许的差异 |
|---|---|---|---|
| **L0 调用** | 用 docker shim 分别捕获旧启动脚本和 `acot eval --dry-run` 发出的全部 docker 调用(见下方 shim 说明) | 科学类 flag 和环境变量逐字节相等(JSON 值解析后比较);infra 类只比较字段集合。覆盖：`chains_attr.sh`、`run_openval.sh`、`f_v4.sh`、`f_oracle_absent.sh`、`collect_dagger.sh`、`eval_ext_dam4sam.sh`、`eval_ext_assoc.sh`、`launch_wd.sh` 的续跑路径 | 容器名、端口、路径(infra) |
| **L1 回显** | `server_echo`、`client_echo` | 对象级 kwargs 哈希与 `expected_kwargs` 相等;库版本与基线记录相同 | 无 |
| **L2 评分** | 用新导入器和新分析，重算已有的旧结果 | 与 `arch_compare.py`、`opening_ds.py`、`iter_gate.py`、`opening_cl.py` 的数字相差 ≤ 1e-12。按种子对齐的 tick 日志，与旧的按顺序对齐(`tick_align.aligned`)逐条相同 | 只允许 §12.2 中预先列明的旧脚本缺陷造成的差异，并且要逐条列出 |
| **L3 实机冒烟** | 在 dev 的 2 个种子上，对 `ours_final`、H-A、`ext_dam4sam` 各跑一次旧启动方式，再跑一次新入口 | 种子相同;`scenario.json` 中确定性字段相同;跳过集合相同(只有原因属于不确定类时才允许不同，并列出);配置回显一致;可部署臂的 `gt_access` 不越出白名单;评分器参数相同 | 轨迹和指标数值(CARLA 的交通灯相位等依赖服务器历史) |
| **L3+(可选)** | A/A:新、旧入口各在 dev 上跑 20 个种子 | 主指标的配对差异落在 bootstrap 95% 区间内，作为健全性检查 | — |

**docker shim 的做法**

在 PATH 最前面放一个假的 `docker` 和一个假的 `sleep`。启动脚本先复制到临时目录，把脚本中写死的仓库绝对路径替换为临时目录，再运行;捕获结束后把路径归一化回来。这样不会污染真实仓库里的 `progress.log` 和 `eval_out`。

shim 对各类调用的响应:

| 调用 | shim 的响应 |
|---|---|
| `docker run` | 记录 argv、环境变量和挂载，返回一个假容器 ID |
| `docker logs` | 输出 `listening` |
| `docker ps` | 输出当前「运行中」的假容器名 |
| `docker exec … run_rollout.py --out X` | 把 `/workspace/...` 映射回宿主机路径，写出含 N 个假 episode 的结果文件，让看门狗循环正常结束 |
| `docker exec … python -c socket…`(端口探测)、`pkill`、`CarlaUE4.sh`、`docker rm` | 记录并返回成功 |

`scenario.json` 中哪些字段在给定种子下是确定的，在 P1a 中实测确认：同一种子连跑两次比较。不确定的字段不进入 L3 的比较，并在此处注明。

---

## 10. 离线：产物、manifest 与兼容性(R1 M4)

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
| `stage2_ckpt` | 先只做旧产物导入 | 配置快照哈希、数据集哈希、划分哈希、ctx 缓存 ID、camera、language_mode、`ablate`、epoch、metrics |

### 10.2 兼容性矩阵(两道边界)

| 比较字段 | B1:训练时，输入产物之间 | B2:闭环解析时,ckpt 与协议、方法 |
|---|---|---|
| `fov_deg`、分辨率 | ✓ | ✓ |
| `tick_s` | ✓ 混用需要豁免 | ✓ |
| `feature_id`(含 crop、pad、模型、TTA) | ✓ | ✓(与 `frontend.crop_features.id` 对照) |
| `depth_scale` | ✓ | ✓ |
| `language_mode`、prompt 哈希 | ✓(vlm_ctx) | ✓(stage2 与 `frontend.vlm`) |
| `ablate` | — | ✓(stage2 与 `method.control.ablate`) |
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
2. 缺失 `split_manifest` 时代码退回朴素切片。新框架中这种情况直接报错;旧配置由 lint 标出。
3. `tah_cache` 的元数据不全，`tah_train` 只检查 `d_feat`。改由 manifest 和兼容性检查(§10.1、§10.2)取代。
4. 开局 ckpt 从 `rollout/eval_out/` 迁入产物库,`AttrScorer` 改为按产物 ID 加载 lin 和 ft 两个 ckpt。

### 10.5 旧产物导入

`acot artefacts import` 为以下产物建立条目：
- `stage2_best.pt`、`tah_relm_dag.pt`;
- `opening_attr_{cls,ft}.pt`;
- `tah_relm_v90_*`、`off90.pt`、`dag_*.pt`。

每个条目都写明 `provenance_gaps`,例如「无配方」「源数据已删除」「离线部分 FOV 错位」「tick_s 混用」。原文件留在原处，产物库只保存引用和 sha256。

---

## 11. 数据生成(R1 m4)

```yaml
# acot/configs/datagen/mvp_full_v5_fov90.yaml
name: mvp_full_v5_fov90
scenario: carla_uav_tracking/config/scenarios/mvp.yaml       # 快照后计算哈希
camera: {fov_deg: 90, width: 512, height: 512, fps: 10}
plan: {source: from_dataset_manifest, ref: mvp_full_v5}       # 也可以用 explicit 列出 [{seed, town}]
retry: {same_seed_attempts: 2, substitute: {pool_start: 108, same_town: true, max_per_slot: 6}}
qc: [attrs_match_camera, target_moving, min_frames, review_contract]
output: {staging: /nvidia/hque/data/carla_data/.staging/mvp_full_v5_fov90, final: /nvidia/hque/data/carla_data/mvp_full_v5_fov90}
```

流程如下:
1. **冻结计划**:把「种子 → 城镇」的显式列表写成 `plan.json` 并计算哈希，生成过程中不可修改。这样可以避免按 `ep_id // 10` 选城镇导致的城镇分布塌缩。
2. **分配 worker**:由 infra 的租约分配 CARLA 端口和 TM 端口。
3. **逐集生成**:每集写入 staging 目录，先通过 QC。staging 位于 `/data` 下，容器内外都可见。
4. **重试与替补**:
   - 同一种子先按原计划重试;
   - 仍然失败，就从替补池中取新种子补位，城镇保持不变;
   - 所有替补和未补上的空位都写入 manifest。
5. **入库**:全部通过 QC 后，原子性地移入正式目录，并写出数据集 manifest。
6. **划分**:由 `acot data split` 显式生成，不会自动生成。

这套流程把 `gen_seed_town_list.sh` 和 `b_fill.py` 的做法固化为规范。之所以需要重试和替补，是因为 CARLA 的世界状态不可复现，同一种子重新生成时可能被判为「目标静止」而跳过。

---

## 12. 结果与分析(R1 M5)

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
   - 每个旧臂标注 `protocol_tag ∈ {gt_seeded, deploy_v1, deploy_v2_head, final_v2}` 和来源脚本;对应关系维护在 `acot/legacy/arm_map.yaml` 中。
4. **旧 tick 日志(不带种子)**:导入器复现按顺序对齐的逻辑，并用步数(max t ≤ 评分器的步数)交叉校验，对齐结果标为 `by_order_verified` 或 `by_order_unverified`。

**已知的旧脚本缺陷**(L2 必须单独列出的差异来源):

| 脚本 | 缺陷 | 实际影响 |
|---|---|---|
| `opening_cl.py` | 取 `sorted(glob)[-1]`,会优先读 `.partial`;`_p[0-9]` 只匹配个位数 part | 实测 54 个同时有 final 和 partial 的 part,种子列表全部相同;没有 ≥10 的 part。所以现有数字不受影响，但写法脆弱 |
| `tick_align.aligned` | 按顺序对齐 | 由 `by_order_verified` 覆盖 |
| `_cand_lite` 不传 `gt_idx` | 跟踪器 ID 模式下，干扰车按跟踪器 ID 计分 | 只影响按干扰车身份细分的统计，主指标不受影响;导入时标注，P2 修复 |

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
- **报告**:
  - diagnostic 臂只能出现在 `family: diagnostic` 中;
  - 主表的标题自动注明「仿真代理感知：…」;
  - P1 期间的运行，在表注中写明 `gt_fence` 的状态。

---

## 13. 基础设施(R1 M6)

### 13.1 硬性约束(写入 schema,由 preflight 强制执行)

1. **网络**:只使用 `infra.docker.network`(`cyh-carla-net`),先用 `docker network inspect` 确认存在。代码中没有任何创建网络的路径;`docker network create` 和 compose 都列入禁用命令,lint 会扫描。
2. **容器**:
   - CARLA 只用 `docker exec cyh-carla`;
   - 只有 policy 容器允许 `docker run`,必须挂在上面的网络上，并以 `policy_container_prefix` 开头;
   - DAM4SAM 只用 `docker start` 和 `docker exec`;
   - 下载一律在宿主机上做。
3. **挂载**:
   - `cyh-carla` 只挂载 `carla_uav_tracking → /workspace` 和 `carla_data → /data`。所以 CARLA 侧代码、`client.json` 和结果根目录都必须位于 `carla_uav_tracking/` 之下;
   - policy 容器挂载整个仓库，路径与宿主机相同;
   - preflight 用 `docker inspect` 核对这两组挂载。
4. **CARLA 客户端的 Python**:
   - 固定用 `cyh-carla` 中的 `python`(3.8.0)。`python3` 是 3.6.9,不能用,`env.py` 中的 `from __future__ import annotations` 需要 3.7 及以上;
   - preflight 运行 `docker exec cyh-carla python -c "import sys; assert sys.version_info[:2]==(3,8)"`。
5. **GPU**:
   - policy 只用 `policy_gpus` 中的卡，并跳过 `reserved` 中的卡;
   - preflight 读取 `nvidia-smi --query-compute-apps`,目标卡上有不属于本调度器的进程时发出警告，显存不足时报错。
6. **端口**:
   - CARLA 端口和 adapter 来自 `carla_servers`;
   - TM 端口段按 worker 和重启次数分配(`base + 30·w + 10·restart`),不会冲突;
   - preflight 检查端口是否被占用，以及是否落在 `reserved` 范围内。
7. **租约**:所有资源用文件锁登记在 `runs/infra/leases/`,并发的 `acot eval` 不会互相抢占。

### 13.2 preflight 清单(`acot eval` 开始前，以及 `acot lint --infra` 时执行)

1. 网络存在;`cyh-carla` 正在运行，挂载与 schema 一致;镜像存在，并记录 image ID。
2. CARLA 客户端的 Python 版本正确;`docker exec cyh-carla python -c "import acot_client.episode"`(在 `/workspace` 下执行)成功，以此验证 py3.8 兼容。
3. GPU 和端口(见上)。
4. 只要有臂用到 DAM4SAM:
   - 每个实例都要通过健康检查，方法是 TCP 连接加一次 `close` 往返;
   - 用 `docker exec dam4sam-svc ps -eo args` 读取服务进程的命令行，核对 `--model` 与臂的 `identity.dam4sam.model` 一致，并写入 manifest;
   - lifecycle 为 managed 时，自动 start/exec 并等待服务就绪;
   - 每个实例同时只租给一个 worker,因为服务只接受单个连接。并行跑 DAM4SAM 臂需要启动多个实例(不同端口)。
5. 数据和 HF 缓存路径可读;结果根目录可写。

### 13.3 py3.8 兼容检查

- `acot/tests/test_py38.py` 用 `ast.parse(src, feature_version=(3, 8))` 解析以下模块：
  - `carla_uav_tracking/acot_client/**`;
  - 它所 import 的 `rollout/{env,run_rollout,remote_policy,bridge,metrics,candidates}.py`,以及 `scene/*`。
- 语法检查发现不了运行时 API 的差异，所以 preflight 中真实 import 的那一步仍然必要。
- P4 再修正 `env.py`、`bridge.py` 中过时的 py3.6 注释。

---

## 14. 迁移计划与工期

| 阶段 | 内容 | 验收 | 估时 |
|---|---|---|---|
| **P−1 基线** | 两个会话的未提交改动先提交到一个基线分支(需用户同意);定好 `env.py`、`policy.py`、`scene_manager.py` 的文件所有权 | 有可引用的基线 git sha | 0.5 h |
| **P0a 清单与 schema** | 清单生成器(已完成，全目录扫描)和 `legacy_map.yaml`;参数使用矩阵;schema、load、resolve(含归属和覆盖检查)、哈希;`_base` 与全部方法 YAML;种子登记;infra schema;lint 检查 FOV 冲突、种子重叠、输出目录冲突、缺失划分 | `test_legacy_surface` 全部通过;lint 检出 §10.4 中已知的冲突;哈希稳定性测试通过;归属越权的覆盖会被拒绝 | 1.5 d |
| **P0b 翻译器与黄金测试** | `legacy.py`;`acot show/--diff/--dry-run`;docker shim | L0 对 §9 中列出的全部启动脚本逐字节一致 | 1.5 d |
| **P1a 调度与运行时** | 调度器、租约、preflight;`acot_client`(episode、wire、echo);按种子落盘的结果、状态机、重试和看门狗;reset 携带种子(只改日志);`server_echo`/`client_echo`;在线协议置空;真值审计(数组代理和标签属性)及其决策中性验证和阳性对照;评分器参数显式化 | L1 对每个臂都相等;L3 冒烟通过(覆盖 3 个臂);`test_raw_gt_positive_control`、`test_audit_decision_neutral` 通过;`gt_fence` 状态写入 manifest | 3 d |
| **P1b 分析与导入** | metrics、stats、report;旧结果导入器;`arm_map.yaml` | L2 与 4 个旧分析脚本的数字一致(差异只来自 §12.2 列出的缺陷) | 1.5 d |
| — | **到这里，1004 的重跑就能全部走新入口**(内部仍是旧代码，但留痕完整) | | 累计约 8 d |
| **P2 policy/env 直接读配置** | `--resolved`;`PolicyInput`/`ScoringInfo` 拆分和 `SimProxyPerception`;一旦设置旧的行为环境变量就报错;去掉不安全的默认值;`_cand_lite` 补传 `gt_idx`;在镜像中固定 timm 版本(新 tag,记录 image ID),去掉启动时的 pip install | 与 P1 路径对照，L0–L3 全部通过;§7.4 的测试全部通过 | 2.5 d |
| **P3a 产物库** | 产物库、manifest、兼容矩阵;旧产物导入;§10.4 的修复 | 兼容检查能拦下「70° 数据按 90° 用」和「0.2 s 与 0.6 s 混用但未豁免」 | 1.5 d |
| **P3b prep、train、datagen** | tah_seq、vlm_ctx、merge、opening 的适配器;tah_relm 和 opening_attr 的训练适配器;datagen 执行器;camera、crops、describe 去重 | 在 2 个 episode 上重建 tah_seq,张量哈希与现有缓存一致(同一 FOV 配置);tah_train 在相同种子下指标误差 ≤ 1e-6(如果 GPU 不保证逐位一致，报告实际差异);datagen 对 2 个种子 dry-run,计划和 QC 输出可复现 | 2.5 d |
| **P4(可选)** | 旧脚本移入 `acot/legacy/`;按组件拆分 `policy.py`(开局 / 身份 head / 滤波 / WHERE);修正过时注释 | 一致性测试全部通过 | 3–5 d |

- P0a–P3b 合计约 14 个工作日，不含 P4。估时包括 CARLA 上的一致性运行，但不包括等待 GPU 的时间。
- 每个阶段完成后都可以单独使用。
- P2 和 P4 涉及另一会话也在修改的文件，需要协调后在独立分支上进行。

---

## 15. 测试清单(全部位于 `acot/tests/`)

- **配置**:
  - schema 严格性(未知字段、缺失字段、当前 head 之外的子配置);
  - `extends` 和覆盖的语义;
  - 归属与覆盖权限(§5.1 表);
  - 哈希稳定性(键顺序和空白不影响哈希;infra 不计入)。
- **收编**:`test_legacy_surface`(双向覆盖;新文件或新目录必须登记)和翻译器黄金测试(L0)。
- **种子**:用途之间不重叠;test 种子加锁;任意列表和替补;`exclude`。
- **真值**:§7.4 的 8 项测试。
- **运行时**:
  - 状态机，用假 env 模拟跳过、失败、卡死、中途死亡;
  - 续跑;原子写入;租约冲突;
  - preflight,用假的 docker 和 nvidia-smi。
- **离线**:manifest 完整性、内容寻址冲突检测、兼容矩阵(包括 `unknown` 和豁免)、分片完整性。
- **分析**:失败即关闭;交集配对的声明;旧结果导入的优先级;metrics 和 stats 与现有实现一致(L2)。
- **py3.8**:`test_py38`,加上 preflight 中在容器内的真实 import。
- **执行方式**:
  - 仓库没有 CI。提交前手动运行，或用 git hook 运行 `python -m acot lint && python -m pytest acot/tests -q`;
  - 需要 CARLA 的测试(L3)用 `-m carla` 单独标记。

---

## 16. 风险与协调

| 风险 | 应对 |
|---|---|
| 两个会话同时修改 `env.py`、`policy.py`、`scene_manager.py` | P−1 先定所有权;P1 的修改只涉及日志和回显，集中在新文件中;P2 和 P4 在独立分支上做，合并前跑一致性测试 |
| 仓库根目录与另一个项目共用 | 新框架只占用 `acot/` 和 `carla_uav_tracking/acot_client/`;清单的排除表列出了另一个项目的目录 |
| CARLA 不确定性 | L3 只比较调用、种子、场景和评分;轨迹层面的比较放到可选的 L3+ |
| 审计代理影响决策或误报 | P1a 用决策中性回放和阳性对照验证;做不到的槽位如实标为 unfenced;白名单按「文件 + 函数 + 规范化源码行」作键，新读取点必须审核(棘轮);P2 结构化拆分后，审计只作冗余检查 |
| 过度工程 | P0–P1 不改算法代码;抽象只服务于已经出现过的故障模式(§2 中的 12 条) |
| 历史数字无法完全复现 | 旧产物和旧结果导入时，如实写明缺口和对齐方式;不追求复现 `stage2_v5` 的训练过程 |
| 估时偏乐观 | P0 拆成 a、b 两段;P1 完成就是可用的里程碑;每个阶段结束时重新估时 |

---

## 17. 需要用户决定的事项

1. **P−1**:是否同意先把当前未提交的改动(包括另一会话的)提交到基线分支。
2. **开局模块的归属**:本稿把它放在协议的 `frontend`,所有臂默认共享，只有开局消融臂可以覆盖。
3. **test 加锁**:使用 test 种子是否强制要求 `--allow-test` 并引用预注册文档。
4. **实施顺序**:是否先做 P0a–P1b(约 8 天)再开始 1004 的重跑;还是重跑先用旧脚本进行，框架并行推进。
5. **P1 的日志改动**:reset 携带种子，ASD、trk、开局、DAgger 日志加 `seed` 字段。这些改动只影响日志，但会修改另一会话也在用的 `policy.py`、`policy_server.py` 和 `remote_policy.py`。

---

## 18. 实施记录(P0a–P1b,2026-10-04)

用户决定：先把框架做到能用，但**不执行**消融和基线的重跑;开局模块作为所有臂共享的前端。基线提交(P−1)和改动共享文件(§17 第 5 项)都未获授权，所以本阶段：
- **没有提交 git**;
- **旧代码一行未改**。reset 带种子、配置回显、真值审计，全部通过新的包装入口实现。

### 18.1 已实现的模块

| 位置 | 内容 |
|---|---|
| `acot/config/` | `schema.py`(严格 schema)、`load.py`(extends 与深合并)、`resolve.py`(归属与覆盖权限、真值通道判定、兼容检查、哈希)、`seeds.py`(种子登记;种子身份为 town 加 seed)、`legacy_map.yaml`(旧配置面映射，同时也是翻译表)、`legacy.py`(翻译器、语义归一化、回显比对)、`legacy_args.py`(从源码静态重建 argparse)、`coverage.py`、`gt_sites.yaml`(49 个已审核的真值读取点) |
| `acot/configs/` | 协议 3 个:`sim_proxy_v2@1`、`sim_proxy_v2_openval@1`(19 步开局验证)、`deploy_v1@1`(只用于复现旧数字);方法 13 个(`_base`、`ours_final` 及其消融、外部基线及其旧版复现、诊断臂);`seeds.yaml`;`infra/default.yaml`;实验(冒烟、录制、7 个旧脚本的复现件、1004 重跑草稿) |
| `acot/eval/` | `scheduler.py`(worker 租约、policy 容器、看门狗、CARLA 重启、重试、逐种子状态、真值审计判定、`audit-replay`)、`legacy_server_entry.py`(policy 侧包装入口)、`replay_neutral.py`(决策中性回放和阳性对照)、`dry_run.py`、`l3_compare.py` |
| `acot/runtime/audit.py` | `AuditedArray`、`AuditedSeq`(递归包装 `vis[1]`)、标签属性审计、读取点判定 |
| `acot/infra/` | `preflight.py`(网络、挂载、py3.8、客户端可导入、镜像、CARLA 端口、GPU、DAM4SAM 健康检查)、`lease.py`、`docker.py`(只有允许的 docker 操作) |
| `acot/analysis/` | `metrics.py`、`stats.py`(各一份实现)、`results.py`(新结果，`logmap` 精确对齐)、`legacy_import.py`(确定的导入优先级)、`report.py`(`acot analyze`,结果不完整就报错) |
| `carla_uav_tracking/acot_client/` | CARLA 侧执行器(py3.8):`episode.py`、`wire.py` |
| `acot/tools/legacy_surface.py` | 旧配置面清单生成器 |
| `acot/tests/` | 单元测试、覆盖测试、L0 黄金测试(含 docker shim)、L2 一致性测试 |

### 18.2 实施中定下的细节(相对设计稿的修订)

1. **城镇移出协议，由种子集给出**。种子身份是(town, seed),DAgger 采集这类跨城镇的种子集可以直接表达;重叠检查也按(town, seed)进行。
2. **协议支持 `extends`**。新增 `sim_proxy_v2_openval@1`;另有 `deploy_v1@1`,只用于复现旧结果。
3. **`control` 字段**:增加 `expert_control`、`predict_road`、`road_oracle`(`run_rollout` 的诊断开关);`oracle_absent` 作为方法字段。
4. **CARLA 侧沿用旧的世界历史语义**:一个进程连续跑一组同城镇的种子，开头只 `load_world` 一次，与 `run_rollout` 相同。
5. **`DAGGER_DIR` 的空值问题**(第一次冒烟时由真值审计发现):
   - 旧代码用 `is not None` 判断 `DAGGER_DIR`,翻译器最初传了空串，结果打开了 grounding DAgger 缓冲(只占内存，不影响决策);
   - 现在翻译器不再输出这个变量，归一化时也区分「空串」与「未设置」,并加了回归测试;
   - 第一次冒烟的结果改名留档为 `_smoke_ours_final_run1_daggerdir_bug`。
6. **TrafficManager 端口段跨运行租约**(两个运行并发时发现):并发的 `acot eval` 不会再分到同一个 TM 端口段。
7. **审计读取点的键**:列表推导式和生成器的帧归并到外层的命名函数，使 py3.11 和 py3.12 下的键一致。
8. **旧 ASD/trk 日志与种子的对应**:包装入口在每次 reset 时记录 `logmap`(ASD 的 ep 编号到种子),新结果因此精确对齐;旧结果仍按顺序对齐并做核验。被看门狗杀掉的那一集会在日志末尾多出一组，这种情况记为警告。
9. **P1 不改共享文件**,§17 第 5 项因此不再需要用户授权。

### 18.3 验收结果

| 级别 | 内容 | 结果 |
|---|---|---|
| 单元测试与覆盖测试 | schema、归属与覆盖权限、真值通道判定、哈希、种子登记与加锁、旧配置面双向覆盖、审计机制、py3.8 语法 | 全部通过(`cd acot && python3 -m pytest -q`) |
| L0 调用 | docker shim 截获 7 个旧启动脚本、12 个场景的全部调用:最终模型 2 个分块;开局验证 v1、head、v3、v4;v4 开局 5 个 worker;oracle-absent 5 个 worker;DAgger 采集 7 个 worker、4 个城镇;DAM4SAM 和 DeepSORT/OC-SORT 两个外部基线;看门狗续跑路径 | 与翻译结果在语义上逐项相等;另有两项测试，确认改动任一旋钮、`DAGGER_DIR` 设为空串，都会被检出 |
| L1 回显 | 冒烟中 policy 进程实际收到的构造参数、开局配置和环境变量 | `echo_check` 为 ok,diff 为空(config_hash 32a6db666b32) |
| L2 评分 | 重算旧结果 | 种子集合和逐种子指标与 `iter_gate.load` 一致;tick 对齐与 `opening_cl.load` 一致(每个臂都标为 by_order_verified);`arch_compare.py`、`opening_ds.py` 打印的每一个数字都被复现;`iter_gate` 的臂汇总表(含 bootstrap 区间)误差 ≤ 1e-12 |
| L3 实机 | 用 `ours_final` 跑 dev 91000–91001:新入口一次，旧启动方式(`eval_asd.sh` 加 `DUMP_LABELS`)一次 | 场景指纹(目标、语言、干扰车集合)一致;跳过集合一致;L1 一致;真值审计 0 违规、0 个未登记读取点 |
| 审计验证 | 回放 dev2 93000 录下的 120 条 act 消息 | 两次不装审计的回放完全一致;装审计与不装审计的决策逐一相同(覆盖 20 个 S2 tick,含一次 PGIF 纠偏);4 个阳性对照(在决策路径中读取 `tpos`、`dstates`、`vis`/`vis[1]`、`true_idx`)全部判为未登记读取点 |
| L3+ A/A | 同一配置的三份重复(dev 20 个种子):acot、旧入口、10-03 历史 | acot 减旧入口：四个指标都不显著(承诺正确 +0.046 [−0.083, +0.173],p = 0.50),不比旧入口对旧入口的差异更大，新入口可以直接替代。详见 `acot_note/1004_aa_noise.md` |

**需要注意的发现：同一种子的运行间差异很大。** 同一配置、同一种子跑三次，承诺正确率分别为：

| 种子 | 新入口第 1 次 | 新入口第 2 次 | 旧启动方式 |
|---|---|---|---|
| 91000 | 0.978 | 0.083 | 0.677 |
| 91001 | 0.053 | 0.467 | 0.139 |

- **原因判断**:
  - 回放在固定随机数种子时完全确定，审计也被证明不改变决策;
  - 所以差异来自两个不确定性来源：一是 CARLA 的世界状态(交通灯相位等随服务器历史变化);二是线上 DiT 采样没有按 episode 固定随机数种子。
- **对 1004 重跑设计的含义**:按种子配对能消掉的方差，比原先假设的小得多。建议三项：
  - 种子数增加到 60 个以上，或每个种子重复运行;
  - 先做一次 A/A(同一方法跑两遍)来测量噪声底;
  - 考虑在新的协议版本中按 episode 固定 policy 的随机数种子。

### 18.4 后续阶段

P3a、P3b 的实施记录见 §19,P2 见 §20,仍未完成的事项汇总在 §21。

## 19. 实施记录(P3a、P3b,2026-10-04)

约束同 §18:没有提交 git,旧代码一行未改，没有执行任何消融或基线重跑。

### 19.1 P3a 产物库

| 位置 | 内容 |
|---|---|
| `acot/artefacts.py` | 产物库：`runs/artefacts/<kind>/<name>@<id>/manifest.json` 加 `index.json`(文件路径到产物的反查表、别名表);12 种产物(§10.1);兼容矩阵 B1(`check_inputs`,训练时比较输入产物之间)和 B2(`check_refs`,解析时比较 ckpt 与协议、方法) |
| `acot/legacy/artefacts.yaml`、`import_artefacts.py` | 旧产物导入：原文件留在原处，产物库只存引用、sha256 和 `provenance_gaps` |
| `acot/configs/provenance_waivers.yaml` | 按产物登记的豁免，每条写明理由 |
| CLI | `acot artefacts import \| ls \| verify` |

实现中定下的细节：
1. **两种寻址方式**。生成的产物按输入内容寻址(同样的输入得到同样的 ID);导入的旧产物按文件内容寻址。同一 ID 下已有内容不同的 manifest 时直接报错，不会悄悄复用。
2. **指纹**。1 GiB 以内的文件计算完整 sha256;更大的文件和数据集计算部分指纹(大小加首尾各 4 MiB)。sha256 缓存的键是路径、大小和 `st_mtime_ns`(最初用整数秒，同一秒内的修改会漏掉，已修正)。
3. **配置里引用产物**,可以写 `artefact:<别名>`,也可以直接写一个已登记的文件路径。解析时由反查表找到对应的产物。
4. **`unknown` 不算兼容**。旧产物缺失的字段记为 `unknown`,比较时视为不匹配，必须豁免。豁免有两个途径：产物级(豁免文件)和臂级(方法的 `accept_provenance_gaps`)。两者都会写进 `acot show` 的输出和结果 manifest。
5. **没有 manifest 的目录视为不完整**。`artefacts ls` 把它们标为 INCOMPLETE(进行中或中止);下游不能引用，因为引用只能经由反查表或别名，而这两者都只在登记时写入。

从旧产物导入并登记了 20 个(下列);另有 P3b 验收时由 `acot train` 产出并登记的 2 个(`tah_relm_v90_s0_repro` 及其 `_last`):
- **数据集**:`mvp_full_v5`(相机按证据覆盖为 70°)、`mvp_full_v5_fov90`;
- **划分**:`split_v5_naive`、`split_v5_fov90_naive`(55/9)、`split_v5_fov90_town10hd`(46/8/10,Town10HD 留出);
- **tah_seq**:`tah_cache_v5`、`tah_cache_v5_fov90`;
- **dagger**:`tah_dagger_v1`、`tah_dagger_v1_reencoded`、`tah_dagger_v2`;
- **tah_ckpt**:`tah_relm_dag`(最终模型)、`tah_relm_v90_s{0,0_last,1,2}`;
- **stage2_ckpt**:`stage2_v5`。**更正(10-04 下午)**:它是 08-17 在**原始** v5 上训练的。原始 v5 实际按 90° 渲染，标签也是 90°,已在 09-04 的磁盘清理中被删除。现在磁盘上的 `mvp_full_v5` 是 09-04 按 70° 重新生成的另一批数据(`1003_data-check.md` §5.1)。上午导入时误记为「mvp_full_v5、70° 图像 / 90° 投影」,现已更正为 fov 90,并去掉对应的豁免。训练数据和 Qwen3-VL 特征缓存都已不在磁盘上，记为 provenance gap;
- **开局**:`opening_attr_lin`、`opening_attr_ft`、`opening_snapshots_train`(811 个文件)、`opening_snapshots_hi_eval`(352 个文件)。

`acot artefacts verify` 全部 OK。唯一的 INCOMPLETE 目录是 `tah_seq/tah_seq_v5_fov90@ea7b4d8bb15a`:P3b 验收时只构建了探针分片(2 个 episode),按设计没有 manifest。

§10.4 立即修复清单的处理：
1. **共用输出目录**:新管线的产出全部写在内容寻址的产物目录里，不再共用;旧配置仍由 lint 标出。
2. **缺失划分**:`acot prep` 必须指定一个已登记的划分，而且这个划分必须属于同一个数据集，否则拒绝;不再退回朴素切片。旧配置仍由 lint 标出。
3. **tah_cache 元数据不全**:改由 manifest 的兼容字段加 B1、B2 检查。
4. **开局 ckpt**:lin 和 ft 已登记为产物，B2 检查 fov 和 hires_k。acot 运行时(§20)按协议给出的显式路径加载这两个 ckpt;旧运行时的 `AttrScorer` 仍从固定路径读取 ft,但读到的是同一个文件。

验收:

| 检查项 | 结果 |
|---|---|
| 用 90° 的模型配置处理 70° 的数据 | `acot prep` 拒绝("70°-images-with-90°-boxes"),负对照配置是 `prep/tah_seq_v5_70_wrong.yaml` |
| 0.2 s 与 0.6 s 的输入混用 | 没有豁免时 B1 拒绝;复现配方 `tah_relm_v90_s0_repro` 声明了带理由的 `tick_s` 豁免后才放行 |
| 最终模型解析 | `tah_relm_dag` 的 fov(离线部分为 70° 图像、90° 投影)和 tick_s 缺口，只能经豁免或 `accept_provenance_gaps` 放行，并出现在 `acot show` 中。`stage2_v5` 的 fov 是 90°,与协议一致，无需豁免(见下方更正) |
| 内容寻址冲突、`unknown` 与混合值不算兼容、B2 需要豁免 | 单元测试 `test_artefacts.py` 通过 |

### 19.2 P3b 离线入口

| 位置 | 内容 |
|---|---|
| `acot/data/` | `datasets.py`(`make_split`:naive 和 town_heldout 两种规则)、`gen.py`(冻结计划、worker 命令、QC)、`qc.py`;去重后的唯一实现:`geometry.py`(投影、伪框、候选框)、`language.py`(指令解析)、`crops.py`(crop 特征参数) |
| `acot/prep/` | `tah_seq.py`(按显式划分、分片构建、合并)、`vlm_ctx.py`(Stage-1/2 上下文缓存写进产物目录)、`merge.py`(由实验采集的 TAH-DAgger 文件合并为 dagger 产物，兼容字段由实验的协议推出) |
| `acot/train/` | `tah_relm.py`(训练后登记 best 和 last 两个 ckpt;`--register-only` 用于宿主进程先于训练容器退出的情况)、`opening_attr.py` |
| `acot/jobs.py` | 在训练镜像中运行作业(只用 `cyh-carla-net`)、代码哈希、git head |
| CLI | `acot data split \| gen \| qc`、`acot prep`、`acot train` |

验收:

| 检查项 | 结果 |
|---|---|
| tah_seq 重建 | 用 `acot prep` 重建 2 个 episode(order 0 和 32,各 750 帧),张量与旧缓存 `off90.pt` **逐位相同** |
| tah_train 复现 | `acot train recipes/tah_relm_v90_s0_repro.yaml`:输入是 3 个已登记产物，种子 0。best ckpt 与 `runs/tah_v90/tah_relm_v90_s0.pt` **逐位相同**(best_epoch 2,val acc 0.99066,权重最大差 0);全部 60 个 epoch 的日志与旧日志 `runs/tah_v90/logs/train_s0.log` 逐行相同;last ckpt(第 60 个 epoch)也逐位相同 |
| datagen | `datagen/mvp_full_v5_fov90_replan.yaml` 的 dry-run:冻结计划(`plan.json` 的哈希一旦写下就不允许改动)和 worker 命令可以复现;QC 中，现有的 90° episode 通过,70° episode 在 `attrs_match_camera` 上被拦下 |
| 去重 | 伪框、离线候选框构建与运行时实现、指令解析(覆盖数据集中的全部指令)、crop 参数与 feature_id,都与各处旧副本逐项一致(`test_dedup.py`)。另外记录了一处已知差异：后处理脚本的 `box_min` 是 8 px,运行时是 10 px |

尚未在真实数据上跑过的：`vlm_ctx`(完整构建是长时间的 GPU 作业)和 `merge`(需要先有一次带 `tah_dagger` 采集的实验)。两者的代码和兼容检查都已就位。

## 20. 实施记录(P2,2026-10-04)

约束同 §18:没有提交 git,旧代码一行未改，没有执行任何消融或基线的闭环重跑。新运行时全部是新文件，只复用旧代码中不接触真值的库(VLM、Stage-2 模块、TAHRelM、TrackletAssigner、AttrScorer 的打分、TargetStateEstimator、`build_candidates`)。下文的决策一致性检查都是开环回放，不产生任何实验结果。

### 20.1 新运行时

| 位置 | 内容 |
|---|---|
| `acot/runtime/io.py` | `Detection`、`PolicyInput`、`ScoringInfo`、`OracleHandle`、`Decision`;禁止字段集合 `FORBIDDEN` |
| `acot/runtime/perception.py` | `SimProxyPerception`:候选框(GT 投影伪框，按 episode 随机数打乱)→ 遮挡过滤 → crop-DINOv2 特征 → 跟踪器重新编号 → `Detection` 和 `ScoringInfo` |
| `acot/runtime/frontend.py` | 所有臂共享的前端：VLM、Stage-2 模块、crop 特征提取器、开局打分器(lin 和 ft 两个 ckpt 都按协议给出的路径加载) |
| `acot/runtime/heads.py` | 身份 head:`TahHead`、`AssocHead`、`Dam4samHead`(xattn 即 grounding 的选择) |
| `acot/runtime/pgif.py` | PGIF(tempered、resid、bayes;absence hold;后验门控的记忆更新),没有任何标签参数 |
| `acot/runtime/policy.py` | `AcotPolicy`:开局 → 身份 head → 置信度(top1 − top2)→ PGIF → K 帧提交 → WHERE → DiT,步骤顺序和随机数的使用与旧 `Policy.act` 相同 |
| `acot/runtime/server.py` | `python -m acot.runtime.server --spec …`:只读已解析配置;设置了任何旧的行为环境变量就拒绝启动;重算 config_hash 必须与调度器一致;按种子写 `ticks.jsonl`、`trk.jsonl`、`gt_access.json`、`act_record.pkl`;回显配置哈希、镜像 ID、库版本 |
| `acot/runtime/support.py` | 新运行时实现了什么(不依赖 torch),解析和 preflight 用它在启动容器之前拒绝不支持的臂 |
| `acot/eval/scheduler.py` | `infra.runtime`(legacy、acot)选择运行时;`acot replay-parity`;`runtime_problems` 在启动前列出所选运行时跑不了的臂 |
| `acot/eval/replay_parity.py` | 同一容器里用同一个 VLM,把录下的 act 消息分别送进旧 `Policy` 和新运行时，逐 tick 比较 |
| `acot/docker/` | `Dockerfile.acot` 和 `build_acot_image.sh`:镜像 `acot-uav-train:cu118-acot`(sha256:08dcad02975f),timm 1.0.29 固定在镜像中。wheel 在宿主机下载并核对 sha256,构建时 `--network none` |

**支持范围**,即 1004 重跑需要的全部组合:
- **身份 head**:tah、assoc(deepsort、motion)、dam4sam、xattn,`memory_seed=committed`;
- **滤波**:off,或 PGIF(tempered、resid、bayes,可开 absence hold 和 gate_memory);
- **开局**:attr(window、agg、final_reset、reset_asd、force_final、seed_asd)或 head;
- **WHERE**:`ex_source=track`,另有 frame_gain、conf_tau、ablate;
- **诊断**:oracle_absent(abstain、hygiene)经 `OracleHandle`。

以下设置会被拒绝，这类臂用 `infra.runtime=legacy` 跑:
- reanchor、cold window、event_override、warmup oracle、reid / fusion / attrbind / sticky / oracle 这几种 head;
- `memory_seed=gt`、`ex_source≠track`;
- 动态换目标;
- 需要标签的采集器(dagger、tah_dagger、opening_snapshots)。

### 20.2 实施中定下的细节(相对设计稿的修订)

1. **`PolicyInput` 不含种子**(设计稿 §7.2 中有 `seed`)。episode 种子不是可部署的观测，方法拿到它就可能按种子记忆答案。种子只留在服务端，用于写日志。
2. **`Detection` 带两种特征**:`vlm_feat`(VLM 网格在框内的池化，供 grounding head 使用)和 `crop_feat`(crop-DINOv2)。crop 特征在感知中只算一次，跟踪器、TAH 和 assoc 共用。旧代码对同一组 crop 要算 2–3 次，结果相同。
3. **跟踪器固定为贪心匹配**。策略镜像里一直没有 scipy:启动时的 `pip install timm` 也不会装 scipy,已在容器内核实。所以 `TrackletAssigner` 在所有旧运行中都退回了贪心匹配，而不是文档里写的匈牙利算法。感知模块显式固定贪心匹配，不再取决于镜像里有没有 scipy;新镜像也刻意不加 scipy。
4. **`cand_lite` 补传 `gt_idx`**,评分器因此按仿真 actor 区分干扰车。已证明这对主指标没有影响：在 `commit_k ≤ s2_period`(5 ≤ 6)时，一次 S2 决策在 6 个 tick 内重复出现，任何与当前提交不同的选择都会在同一周期内完成切换，所以「提交是否等于目标」与干扰车的键无关(测试 `test_cand_lite_gt_idx_is_neutral…`,200 组随机序列)。受影响的只是按干扰车身份细分的统计，因此不另开协议版本。
5. **真值审计保留为冗余检查**。新运行时同样包装原始真值数组。可部署臂的原始真值只允许在 `acot/runtime/perception.py` 和 `rollout/candidates.py` 中读取，其他位置的读取记为违规;诊断臂的真值访问按用途计数，写入 `gt_access.json` 的 `oracle_calls`。
6. **一处旧行为与注释不符，新运行时照实复现**:deploy v2 下，已提交的身份这一 tick 没有匹配到框(被遮挡、出画、PGIF absent)时,`_committed_world` 返回 None,`act()` 会跳过整个 WHERE 更新,z_ex 保持 EAR 本 tick 的采样，而不是像注释说的那样「estimator coasts on CV」。这关系到论文对方法的描述：目标不可见期间，控制由 EAR 的路点采样驱动。
7. **旧运行时的硬编码常量**。`legacy_map.yaml` 原先声称解析会拒绝非默认的 `tah.lang_reanchor_thr`(0.6)和 `dam4sam.reinit_cooldown`(3),实际上没有任何检查，旧服务端会悄悄沿用写死的值。现在由 `legacy.hardcoded_problems` 检查,`acot eval` 和 `--dry-run` 都会拒绝这类臂;新运行时则直接读取这两个字段。
8. **两处框架自身的修复**:
   - `file_sha256` 的缓存以 mtime 作键，同样大小的文件在时间戳精度(约 4 ms)之内被改写时，缓存会给出旧值。最近 2 秒内修改过的文件现在一律重算，且不写入缓存;
   - `artefacts ls` 遇到没有 manifest 的目录会崩溃，现在把这类目录标为 INCOMPLETE。
9. **默认运行时改为 acot**(验收通过之后，见 §20.3)。`infra.runtime` 不计入配置哈希，同一配置在两种运行时下哈希相同、决策相同;每次运行的 manifest 记录所用的运行时和镜像 ID。复现旧运行的实验(`experiments/legacy/*`、`aa_noise`、`smoke_ours_final`、`smoke_replay`)显式固定为 legacy;acot 没有实现的臂由 `runtime_problems` 在启动前列出，改用 `infra.runtime=legacy`。1004 重跑草稿的 10 个臂全部可以在 acot 运行时上跑。

### 20.3 验收

| 级别 | 内容 | 结果 |
|---|---|---|
| §7.4 测试 | `test_policy_input_fields`、`test_method_no_gt_attr`(AST 扫描方法模块：不碰禁止字段、不导入 `ScoringInfo`、不读环境变量、`true_slot` 只能经 `oracle` 调用)、`test_oracle_handle`、`test_diag_label…`、`test_wire_blanking`、`test_gt_audit_ratchet`;另有 PGIF 移植与旧 ASD 滤波逐步相等(3 种 kind × absence hold 开/关，随机序列)、支持范围、拒绝旧环境变量、`gt_idx` 中性 | 全部通过(`acot/tests/test_runtime.py`;全套 82 项) |
| L0 调用 | 新运行时的 policy 容器命令 | 只在 `cyh-carla-net` 上;没有任何旧的行为环境变量;不做 pip install;使用 acot 镜像(`test_acot_runtime_invocation…`) |
| L1 回显 | 服务端回显的配置哈希与镜像 ID | `echo_check` 为 ok(config_hash 45f7e1bcedd2,镜像 sha256:08dcad02975f) |
| 决策一致性(开环) | 见下表 | 见下表 |
| L3 实机 | `smoke_acot_runtime`(新运行时)对比 `smoke_ours_final`(P1 路径),都是 dev 91000–91001 | 状态相同(都是 complete);场景指纹相同;客户端回显(评分器参数、环境开关、置空规则)相同;两边的真值访问都干净，新运行时的原始真值读取全部发生在感知模块中。承诺正确率 0.16 对 0.08、0.19 对 0.47,差异在 §18.3 测得的运行间噪声之内 |

**决策一致性**。把同一段录下的 act 消息分别送进旧 `Policy`(按翻译器的 argv 和环境变量构建)和新运行时(按已解析配置构建),两边共用同一个 VLM,每次 reset 都把 torch、numpy、random 的种子设为 0,然后逐 tick 比较:
- 选择的槽位;
- 搜索标志;
- 动作(容差 1e-5);
- S2 的身份 logits;
- 发给评分器的 `cand_lite`。

| 录制 | 覆盖情况 | 臂 | 结果 |
|---|---|---|---|
| dev2 93000,前 120 tick(P1 录制) | 20 个 S2 tick,含开局 3 个 tick;PGIF 纠偏 0–5 次(随臂不同) | `p2_parity` 的 19 个臂 + DAM4SAM 的 3 个臂 | 22/22 逐位相同 |
| dev 91000,402 tick(acot 运行时录制，第 401 步因目标丢失截断) | 67 个 S2 tick;遮挡后跟踪器 ID 互换;PGIF 纠偏最多 7 次、absent 最多 6 次;弃权 30–37 个 tick;DAM4SAM+PGIF 有 5 次 SAM2 重初始化 | 同上，19 + 3 个臂 | 22/22 逐位相同 |
| dev 91001,891 tick(acot 运行时整集录制) | 149 个 S2 tick;跟踪器出现过 18 个 key;PGIF absent 最多 71 次(oracle_absent),纠偏最多 15 次(bayes);弃权最多 121 个 tick | 1004 重跑的 9 个臂(ext_dam4sam 除外) | 9/9 逐位相同 |

- **19 个臂**:
  - 最终模型;
  - 滤波的变体：关闭、无 absence hold、bayes、resid、不做记忆门控;
  - 开局的变体:head、mean、last、v3(final_reset 加 reset_asd)、v4(force_final 加 seed_asd);
  - WHERE 的变体：frame_gain 0.5、conf_tau 0;
  - 基线:DeepSORT、OC-SORT、DeepSORT+PGIF、xattn;
  - 诊断:oracle_absent 的 abstain 和 hygiene。
- **DAM4SAM 的 3 个臂**:DAM4SAM、DAM4SAM+PGIF、DAM4SAM 逐帧。
- **合计 53 次比对，没有一处不一致**:动作最大差 0.0,即浮点逐位相同;选择、搜索标志、身份 logits、`cand_lite` 也全部相同。
- **这证明了什么**:在相同观测和相同随机数状态下，新运行时与旧代码做出完全相同的决策。闭环结果仍然受 CARLA 世界状态和未固定的 DiT 采样影响(§18.3),所以 L3 只比较场景、状态和配置，不比较指标数值。
- 结果文件：`eval_out/exp/p2_parity{,_dam4sam}/replay_parity_{93000,91000,91001}.json`。


## 21. P4 的处理与仍未完成的事项(2026-10-04)

### 21.1 P4(可选阶段)

P4 原计划三件事。因为没有获得改动共享文件的授权，做法调整如下:
1. **旧脚本移入 `acot/legacy/`**:没有移动。笔记引用了这些脚本，另一会话也在使用。改为在 `acot/legacy/README.md` 中给出旧入口到 acot 命令的对照，每个脚本的处置记在 `legacy_map.yaml` 中，由 lint 检查覆盖完整。
2. **按组件拆分 `policy.py`**:这件事已经由 P2 的新运行时完成。前端、感知、身份 head、PGIF、提交和 WHERE 分属独立的模块，旧文件保持原样。旧运行时继续服务 acot 运行时没有实现的臂。
3. **修正过时注释**:旧代码中与实际行为不符的三处注释，列在 `acot/legacy/README.md` 末尾，留给代码所有者修正(见 §20.2 第 3、6 条)。

### 21.2 仍未完成或待用户决定

| 事项 | 状态 |
|---|---|
| tah_train 全程复现 | **已完成**:60 个 epoch 的日志与旧日志逐行相同;best ckpt(第 2 个 epoch,val acc 0.99066)和 last ckpt(第 60 个 epoch,val acc 0.98666)都与旧 ckpt 逐位相同(权重最大差 0)。已登记为 `tah_relm_v90_s0_repro@d2dd97fd46aa` 和 `tah_relm_v90_s0_repro_last@9e7d4f2356e6`。宿主进程中途被终止，训练容器照常跑完，随后用 `--register-only` 完成登记(§20.2 之后`acot/jobs.py` 改为由容器自己写日志和退出码) |
| `vlm_ctx`、`merge` 适配器 | 代码和兼容检查已就位，尚未在真实数据上运行 |
| acot 运行时未实现的设置 | reanchor、cold window、event_override、warmup oracle、reid / fusion / attrbind / sticky / oracle 这几种 head、`memory_seed=gt`、`ex_source≠track`、动态换目标;需要标签的采集器(dagger、tah_dagger、opening_snapshots)。这些用 `infra.runtime=legacy` |
| 待用户决定 | P−1 基线提交;1004 重跑的种子数和重复次数(噪声底见 §18.3 和 `1004_aa_noise.md`);是否新开一个按 episode 固定 policy 随机数种子的协议版本，并重做 A/A;以下两项保持默认 —— test 种子保持锁定，消融和基线重跑不执行 |

## 附录 A:Codex R1 意见对照

| R1 | 意见 | 处理 | 位置 | R2 判定 |
|---|---|---|---|---|
| M1 | §5 的映射不完整 | 自动生成全清单;`legacy_map.yaml` 双向覆盖测试;写明隐式语义;P2 起未知字段报错 | §6 | PARTIALLY,剩余问题见附录 B 的 R2-M2 |
| M2 | 真值隔离不够机械 | 通道分类;接口定稿;在线协议置空;调用点审计;结构化拆分 | §7 | PARTIALLY,剩余问题见附录 B 的 R2-M3 |
| M3 | 一致性标准太弱 | L0–L3;reset 带种子;评分器参数显式化 | §8.4–8.6、§9 | PARTIALLY,剩余问题见附录 B 的 R2-M1 |
| M4 | 离线兼容元数据不足 | 11 类产物 manifest;兼容矩阵;内容寻址;4 项修复 | §10 | RESOLVED |
| M5 | 结果完整性和 partial 规则 | 4 种状态;失败即关闭;确定的导入优先级 | §8.2、§12 | RESOLVED |
| M6 | 基础设施约束不可执行 | infra schema;preflight;租约;DAM4SAM;py3.8 | §5.6、§13 | RESOLVED |
| m1 | 估时偏乐观 | P0 拆成 a、b;里程碑 | §14 | RESOLVED |
| m2 | 种子需支持任意列表 | ranges、list、list_file、exclude、替补 | §5.4 | RESOLVED |
| m3 | 措辞过强 | 改称 `sim_proxy_v2` | §1、§5.2 | RESOLVED |
| m4 | 数据生成规格不足 | 冻结计划、重试替补、租约、QC、staging | §11 | RESOLVED |

## 附录 B:Codex R2 意见对照

| R2 | 意见 | v2 的处理 | 位置 |
|---|---|---|---|
| M1 | 协议与方法的字段归属不一致(`control` 两处出现;`commit_k` 的位置) | 明确归属模型：协议只放共享不变量;`control`(ex_source、frame_gain、commit_k、ablate、shuffle_seed)整体移入方法，由 `_base` 给出默认值;按归属表定义臂级覆盖权限;`legacy_map`、示例和哈希规则同步修改;`ablate` 与 ckpt 的配套检查 | §5.1–5.3、§6.2–6.3、§10.2 |
| M2 | 清单不够全(只扫了固定的文件列表;启动脚本漏掉写死的 flag) | 改为 glob 全目录扫描(177 个 Python 文件、100 个 shell 脚本),排除项逐一写明原因;`sys.argv` 位置参数入表;启动脚本列出全部 `--flag`;权威 argv 以 docker shim 捕获为准;新增文件或目录必须登记 | §6.1–6.2、清单第 0、1、5 节 |
| M3 | P1 审计没有覆盖原始真值数组 `tpos`/`dstates` | `AuditedArray`(tpos、dstates)和 `AuditedTuple`(vis)代理，记录读取点;按读取点类别建立白名单;决策中性回放和逐项阳性对照;做不到的槽位如实标为 unfenced,此时「可部署」的表述以 P2 为准 | §7.1、§7.3、§7.4 |
| m1 | 清单应标出 `cand_lite` 缺少 `gt_idx` | 清单第 4 节和 §12.2 中已标出 | 清单、§12.2 |
| m2 | reset 带种子的措辞 | 写明:seed 只写日志、不进入 `_rng`、不增加 reset 调用 | §8.4 |
| m3 | 输出路径不应归为 infra | 新增 artefact 类;infra 只指主机侧资源 | 清单分类、§5.7、§6.3 |
| 自查 | `cyh-carla` 只挂载 `carla_uav_tracking`;仓库根目录的 `configs/`、`tests/` 等属于另一个项目 | CARLA 侧代码放在 `carla_uav_tracking/acot_client/`,只读 `client.json`;结果根目录对三处都可见;新框架全部放在 `acot/` 下 | §3 第 9 条、§4、§8、§13.1 |
| 自查 | 启动时的 pip install | P1 原样保留(L0 比较),P2 改为在镜像中固定版本;`server_echo` 记录库版本 | §6.3、§8.5、§14 |

## 附录 C:Codex R3 意见对照(R3 结论 APPROVE-WITH-MINOR,无 major)

| R3 | 意见 | v3 的处理 | 位置 |
|---|---|---|---|
| R2 各项 | M1–M3、m1–m3 | 全部判为 RESOLVED | — |
| m1 | 「未扫描的都列出」说法过满：`acot_note/figs/*.py` 既没扫描也不在排除表里 | 生成器新增顶层归类：每个顶层目录和根目录脚本都必须是扫描或排除(附原因),出现未归类项测试失败;补充 `acot/`、`acot_note/`、`paper_draft/`、`related-survey/`、`result/`、`.github/`、`.humanize/` 的排除原因 | §6.1、清单第 0 节 |
| m2 | `dam4sam_service.py --model` 已进清单，但 §6.3 没有映射 | 映射到 `method.identity.dam4sam.model`(计入哈希);managed 模式下按该值启动服务;preflight 读取服务进程的命令行核对，并写入 manifest | §5.3、§5.6、§6.3、§13.2 |
| 建议 | `AuditedTuple` 应递归包装 `vis[1]` | 已写入 | §7.3 |
