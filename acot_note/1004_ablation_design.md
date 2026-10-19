# 1004 · 最终模型实验与消融设计(预注册稿 · v4.1 · **已冻结**)

> 2026-10-02 · 目标:CCF-A 会议(CoRL / ICRA / IROS 级别以上)或 SCI 一区机器人期刊(T-RO / RA-L / IJRR)。
> 状态:**v4.1,codex 第 4 轮审计 NO MAJOR(`codex/1004_ablation_R4.md`),本预注册稿冻结,进入 T0**。本文件审计的是**设计是否完整、T0 验收门槛是否充分**;代码实现在 T0 结束时按第 12 节门槛另行审计。T0 门槛全部通过并经审计前,不启动任何 test 运行。
> 关联:最终模型结构 `figs/fig_architecture_v1.png`(2026-10-04 起;原 `fig_architecture.png` 已更名为 legacy `fig_architecture_legacy_pgif.png`);方法 `1001_innovation-design.md`;迭代与口径修正 `1003_iteration_roadmap.md`;审计记录 `codex/1004_ablation_R*.md`。

---

## 0. 原则

1. **论文只呈现一个最终模型:TAHRelM + PGIF。** 所有消融都从最终模型出发,每次只改一个变量(leave-one-out),不按研究过程逐步搭建。
2. **研究过程不进正文主表,但负结果要披露**:
   - 启发式阶梯、ASD、1003 的 V1–V3 只在附录出现。
   - 正文「局限」一节用一两句话说明:融合、延迟承诺等方案在 dev 上未带来改善,最终只冻结 TAHRelM + PGIF 进入全新的 test。
3. **每个论文主张对应一组实验,每组实验有预注册的方向性预测。**
4. **所有正文数字来自同一份冻结代码、同一个批次、全新的 test 种子。**
5. **区分两类实验**:
   - **部署实验**:自然开局,代表独立部署表现,包括表 1 和表 4(b)。
   - **机制实验**:共享开局,用共同随机数隔离组件效应,**不代表独立部署表现**,包括表 2、表 3、表 4(a)、表 5。

## 1. 冻结的最终模型

| 部件 | 冻结内容 |
|---|---|
| 主干 | Qwen3-VL-4B、IAR、crop-DINOv2、DiT(全部冻结,沿用 ACoT,不是本文贡献) |
| WHICH | TAHRelM:7 维特征 → MLP 7·64·64·1,外加 3 个标量(记忆 EMA 门、速度 EMA、软运动门半径),共 4,740 参数 |
| PGIF | `--asd filter`,`filter_kind=tempered`,λ = 0.98(`resid_decay`),秩窗口 W = 30,0 个可训练参数 |
| 设定 | deployable:`--gallery-seed committed --reanchor off`,冷启动用语言 grounding 选车 |

**TAHRelM 训练配方要重新记录。**
- 现有 ckpt `runs/tah_relm_dag.pt` 只保存了 `arch: relm`,没有保存完整的训练命令。
- 正式实验前,用 `train/tah_train.py --motion --aug` 以固定且写入日志的配方,重训 Full 的 3 个训练种子(s0/s1/s2)。
  - 数据:离线 64 ep 缓存 `runs/tah_cache.pt` + DAgger 23 ep 缓存 `runs/tah_cache_dag.pt`。现有 `--cache` 只读一个文件,T0 中改为可接收多个缓存并合并。
  - 选模规则:**沿用现有 `tah_train.py` 的行为,保存固定验证划分上的最佳 val checkpoint**。验证划分为 `--val-frac 0.15`,划分随机种子固定,与训练种子分开。所有变体使用同一划分和同一规则。
  - 训练命令、数据文件哈希、划分种子、训练种子、选中的 epoch 写入 ckpt 和日志。
- 先在 dev 上确认 s0 与现有 ckpt 的闭环表现等价,即配对差的 CI 包含 0。**论文中的 Ours 预先指定为 s0**;3 个种子的稳定性在表 3 报告。

**冻结与记录。**
- 正式运行前打 git tag `paper-final-v1`。
- 每个结果 json 记录以下元数据:
  - git commit 与 dirty diff 的哈希;
  - policy server、rollout 的完整命令行;
  - `ASD_CFG`、`OPENING_FILE` 等环境变量;
  - CARLA 端口与 GPU、policy GPU、DAM4SAM 服务端口;
  - 种子范围、批次 ID。
- 之后如需改代码,整批重跑,不混用批次。

## 2. 主张 → 实验

| 论文主张 | 实验 | 表 | 类型 |
|---|---|---|---|
| C1 问题:闭环里自更新的身份记忆会形成确认偏差;瓶颈在身份(WHICH) | 上界分解(真值身份、真值刷新记忆) | 表 1 灰色诊断行 | 部署 |
| C2 TAHRelM:可学的车辆无关相对关联 | TAHRelM 内部消融 | 表 3 | 机制 |
| C3 PGIF:回火的置信加权 tracklet 证据 + 后验门控 | 表 1 的 w/o PGIF(部署)+ 表 2 的组件消融(机制) | 表 1、表 2 | 部署 + 机制 |
| C4 通用性 | 换记忆(机制)、换地图(部署) | 表 4 | 机制 + 部署 |
| 边界 | 关联噪声、检测噪声、换控制器 | 表 5 | 机制 |

## 3. 评测协议

### 3.1 种子划分(防止测试集复用与泄漏)
- **dev**:Town05 91000–91019(研究过程中被反复使用,只能作 dev)以及 93000–93019(新增)。所有超参选择、基线调参、试运行、代码冒烟只在 dev 上进行。
- **排除**:92000–92023 是 DAgger 训练数据,任何评测都不用。离线 64 ep 训练数据来自数据生成阶段的独立种子,不与 9x000 段重叠(正式运行前核对并写入附录)。
- **test**:Town05 95000–95079(n = 80);Town03 与 Town10HD 各 95000–95039(n = 40)。这些种子此前从未运行过。正式批次之前不在 test 上做任何运行或查看。

### 3.2 指标(主指标在 scorer 层定义)
- **身份输出的三个层次**:
  - 记忆原始选择 s^M;
  - PGIF 输出 ŝ;
  - 每 tick 输出 `pred_slot`:开启 PGIF 时为 ŝ,关闭时为 s^M。唯一例外是表 2 的 `gate_commit=false` 变体:PGIF 开启,但 `pred_slot` 与承诺跟随 s^M,只有记忆按 ŝ 更新。
- **两层承诺**:
  - policy 内部的承诺 `_committed_idx` 决定飞行,即 WHERE 跟谁。
  - scorer 对每 tick 的 `pred_slot` 再做 K = 5 滞回,得到 `scorer_committed_key`。所有臂使用相同的 K。
- **主指标:正确锁定比例** = 1 − `mis_follow_sustained`,即目标在画面内的帧中,`scorer_committed_key` 为真目标的比例。弃权帧沿用上一次承诺。
- **次指标**:最长连续正确锁定(`max_correct_run_s`)、最长锁错(`max_wrong_run_s`),都基于 `scorer_committed_key`。
- **主表同时报告**:弃权率,即目标在画面内、但 `pred_slot = −1` 的帧比例。表注给出宽松口径(弃权算对)与严格口径(弃权算错)的数值。DAM4SAM 的宽松口径不放到附录里藏起来。
- **口径已知细节**:episode 开头尚无承诺时的弃权帧不计为错,对所有臂同样偏宽。

### 3.3 开局控制(最大的有效性威胁)
- **问题**:每集的正确比例几乎只有 0 和 1 两种结果(V0 的 19 集里 8 集 ≈ 0、11 集 ≈ 1),主要由首帧语言选车决定。
  - 少数种子的首帧选择在不同运行间会翻转。在 dev 的 8 个种子、4 次运行里,翻转率约为每臂 6–12%。
  - 同配置 V0 两个批次分别为 0.613 与 0.576;V3 − V0 = −0.21,主要来自开局锁对数不同(7/19 对 11/19)。
- **参考开局(独立于所有被比较的臂)**:
  - 正式批次第一步,对每个 test 种子跑一次短的「语言开局参考过程」。它使用和所有臂相同的语言 grounding 冷启动,只运行到首个决策 tick 后结束(STEPS ≈ 12)。
  - 记录首个决策 tick 的语言选择,以 **actor 身份** 保存:`{seed: actor_idx}`,其中 `actor_idx = Candidate.idx`,目标为 −1,干扰车为其编号。不保存 slot,因为每帧候选顺序会打乱。
  - 同时记录这次选择是否为真目标,仅供离线分层使用。
  - 参考过程独立跑两遍,报告两遍之间的开局翻转率,作为语言冷启动的可复现性。
- **自然开局**(部署实验,表 1、表 4(b)):每个臂自己做语言冷启动。
- **共享开局**(机制实验,表 2、表 3、表 4(a)、表 5):
  - **注入语义**:在冷启动的首个决策 tick,**在 PGIF 之前**,按 `Candidate.idx` 找到参考 actor 当前所在的 slot `s_ref`,把本 tick 的记忆原始选择替换为 `s_ref`。随后以下各步全部由它驱动:
    - (i)记忆写入:TAHRelM 的首次 `update`、reid 的首个 gallery 模板、assoc 的初始轨迹状态、DAM4SAM 的初始化框,都取 `s_ref` 的候选;
    - (ii)PGIF 的首个证据投票记给该 actor;
    - (iii)本 tick 的 `pred_slot`;
    - (iv)policy 承诺 `_committed_idx`。
  - 日志里这 4 项都记录 actor_idx,用来验证它们一致。
  - 如果参考 actor 在首个 tick 不可见,退回到本臂自己的语言选择,并标记 `opening_fallback`。各臂报告 fallback 次数。
  - 从第二个决策 tick 起,各臂完全按自身机制运行。
- **分层报告**:
  - 统一的分层变量是**参考开局是否正确**。这是种子级协变量,在任何被比较的臂运行之前就已确定,不受处理影响,所以不构成对撞偏倚。
  - 各臂自己的开局正确率只做描述,不用于分组,也不做因果解释。
  - 开局正确层对应 PGIF 的主张(抗漂移);开局错误层(锁错后能否恢复)作为局限如实报告。
- **前提**:所有臂(包括外部基线)都写逐 tick 日志(见第 12 节)。

### 3.4 噪声底线(A/A 检验)
- **自然 A/A**:Ours 在表 1 中用不同的 TM 端口和 CARLA 槽位跑两遍,估计部署噪声。
- **共享 A/A**:表 2 的 Full 跑两遍,估计机制实验的残余噪声。
- 两个 A/A 都报告配对差的 95% CI。若某个组件效应小于对应 A/A 差的 CI 宽度,正文注明。

### 3.5 样本量(按双峰结果做模拟功效分析)
- **模型**:每个种子先抽开局是否正确(p ≈ 0.55)。
  - 开局正确时,以概率 h 保持锁定(正确比例约 0.9–1.0),否则漂移(约 0–0.6)。
  - 开局错误时,以概率 r 恢复(约 0.3–0.9),否则约 0–0.1。
  - Full:h = 0.95,r = 0.05。消融臂通过降低 h 产生总差 Δ。
  - 自然开局下,每臂相对参考开局独立翻转,翻转率 10%。
  - 同一种子内两臂的保持事件相互独立,这是保守假设。
- **功效**(配对符号翻转检验的大样本形式,双侧,每格 20,000 次模拟;α = 0.0125 对应 4 个比较 Holm 的最严一步):

| n | Δ | 共享开局 α=0.05 / 0.0125 | 自然开局 α=0.05 / 0.0125 |
|---|---|---|---|
| 60 | 0.15 | 0.95 / 0.83 | 0.71 / 0.49 |
| 60 | 0.20 | 0.99 / 0.97 | 0.92 / 0.79 |
| 80 | 0.15 | 0.98 / 0.94 | 0.84 / 0.65 |
| 80 | 0.20 | 1.00 / 1.00 | 0.97 / 0.91 |

- **样本量选择**:
  - 机制表(共享开局)用 n = 60,对 Δ ≥ 0.15 的功效 ≥ 0.83。
  - 部署主表(表 1)用 n = 80。
  - 已知效应量:dev 上 Ours 相对 w/o PGIF 约 +0.26,相对 DAM4SAM 约 +0.21。在 n = 80、两项主比较的设定下,功效约 0.85 以上。
  - 新地图(自然开局)每张 n = 40,两图合并为 80。
- **悲观敏感性**(`python power_sim.py --sens`,开局正确率 p ∈ {0.45, 0.55},翻转率 ∈ {0.10, 0.15},Full 保持率 ∈ {0.85, 0.95}):

| 设定 | Δ = 0.15 | Δ = 0.20 |
|---|---|---|
| 表 1 自然开局,n = 80,α = 0.025(2 个比较 Holm 的最严一步) | 0.65–0.81 | **0.92–0.98** |
| 机制表共享开局,n = 60,α = 0.0125(4 个比较) | 0.73–0.83 | **0.94–0.97** |

- **最小可检出效应**(悲观参数下功效 ≥ 0.8):表 1 约为 0.20,机制表约为 0.15–0.17。dev 上的预期效应(+0.26、+0.21)都在 0.20 以上;**小于 0.15 的效应,本设计不保证能检出**,正文据实说明。
- 模拟使用的检验与正式分析相同:配对符号翻转置换检验,取其大样本形式 z = Σd / √Σd²。
- n 在启动前固定,不做中途查看后的增减。脚本 `rollout/power_sim.py`。

### 3.6 批次与调度
- 每个臂按 10 个种子一块切分,(臂, 块)作业随机打乱后分配到 4 个 CARLA 槽位,不让一个臂连续跑完再跑下一个。随机种子写入调度日志。
- 每个(臂, 块)使用唯一的 TrafficManager 端口。

## 4. 表 1 · 主对比(部署实验:Town05 test,n = 80,自然开局)

> **所有 deployable 臂的命令行都必须显式包含 `DEP = --gallery-seed committed --reanchor off`**。`policy_server.py` 的默认值是 `--gallery-seed gt`(:139),漏写会让 re-ID / assoc / DAM4SAM / TAHRelM 的记忆悄悄用真值刷新,获得隐藏特权。下表和第 15 节中,「+ DEP」表示追加这两个参数。只有灰色上界行允许不带 DEP,由 G10 强制检查。

| 臂 | 配置 | 说明 |
|---|---|---|
| **Ours:TAHRelM(s0)+ PGIF** | `--tid-head tah --tah-ckpt <s0> --conf-tau 0.5 --asd filter --asd-cfg '{"filter_kind":"tempered","resid_decay":0.98}'` + DEP | 跑两遍(自然 A/A) |
| **TAHRelM w/o PGIF** | `--tid-head tah --tah-ckpt <s0> --conf-tau 0.5 --asd off` + DEP | 主自消融(部署层面的 PGIF 效应) |
| OC-SORT-style 关联 | `--tid-head assoc --assoc-mode motion --assoc-gate-px <dev> --conf-tau <dev>` + DEP | 外部:仅运动关联 |
| DeepSORT-style 关联 | `--tid-head assoc --assoc-mode deepsort --assoc-gate-px <dev> --assoc-lambda <dev> --conf-tau <dev>` + DEP | 外部:运动 + 外观关联 |
| DAM4SAM(CVPR'25) | `--tid-head dam4sam --dam4sam-host dam4sam-svc --dam4sam-port 560x --assoc-gate-px <dev> --conf-tau <dev>` + DEP | 外部 SOTA 记忆跟踪器;报告弃权率 |
| DINOv2 外观 gallery re-ID | `--tid-head reid --reid-feature dinov2 --reid-bank 2 --reid-topm 2 --reid-tavg <dev> --conf-tau <dev>` + DEP | 通用外观记忆 |
| 单帧语言 grounding | `--tid-head xattn` + DEP | 不带记忆的 VLM 选车 |
| 诊断上界(灰) | `--tid-head oracle`(真值身份);TAHRelM `--gallery-seed gt`(记忆每步用真值刷新) | 只说明还剩多少空间,不参与比较 |

- **命名与定位**:OC-SORT-style / DeepSORT-style 是在本系统的理想候选框、持久 tracklet 和相同控制链路下复现的关联策略,作为机制对照,不声称是原论文的完整 SOTA 复现。正文明确写出。
- **相同输入**:外部方法与我们使用相同的候选框、语言冷启动、WHERE/HOW 链路,全部 deployable。
- **基线调参**:所有可调方法在 dev 上使用下列预注册网格,取 dev 主指标最好的一组。

  | 方法 | 网格 |
  |---|---|
  | OC-SORT-style | 关联门 `--assoc-gate-px` {120, 160, 240} × 承诺置信阈值 `--conf-tau` {0.05, 0.2} |
  | DeepSORT-style | 关联门 {120, 160, 240} × 外观权重 `--assoc-lambda` {0.5, 1, 2} × `--conf-tau` {0.05, 0.2} |
  | DINOv2 re-ID | 时序 EMA `--reid-tavg` {0, 0.4} × `--conf-tau` {0.05, 0.2} |
  | DAM4SAM | 关联门 {160, 240, 320} px(2026-10-02 已在 dev 91000–19 上跑完,conf_tau = 0.05)× `--conf-tau` {0.05, 0.2} |
  | Ours | λ、W 不调,按先验固定;`--conf-tau 0.5` 在 dev 开发阶段确定。我们在 dev 上经历过多轮开发,这一点在附录如实说明;外部方法因此获得网格调参作为补偿 |

- **DAM4SAM 额外报告**:关联门、弃权率及弃权原因构成(空掩码 / 门外 / 未初始化)、初始化时机、服务版本(SAM2.1-L)、典型失败案例。
- **预注册主比较族(Holm,2 个)**。选择理由:
  - `w/o PGIF` 是本文核心机制在部署层面的自消融;
  - DAM4SAM 是 dev 上最强的外部记忆跟踪器(承诺口径),也是最常被引用的 SOTA。
  - 其余外部方法是风格对照或简单基线,作为次要结果。这一选择在看 test 结果之前确定。
  - H-M1:Ours > TAHRelM w/o PGIF;
  - H-M2:Ours > DAM4SAM。
  - 二者都基于主指标。其余外部基线的比较报告效应量与 CI,作为次要结果。

## 5. 表 2 · PGIF 组件消融(机制实验:Town05 test,n = 60,共享开局)

| 臂 | 实现 | 检验 | 预注册预测 |
|---|---|---|---|
| **Full** | 冻结配置 | — | 跑两遍(共享 A/A) |
| w/o PGIF | `--asd off`,记忆用自己的选择原始自更新 | 命题 1:确认偏差 | Full 更高;差异集中在参考开局正确层 |
| w/o 回火 | `filter_kind=bayes`,逐 tick 似然相乘 | 命题 2a | Full 更高;未回火贝叶斯被漂移记忆俘获 |
| w/o 置信加权 | `filter_kind=resid`,等权驻留 | 命题 2b | Full 更高 |
| w/o 记忆门控 | `gate_memory=false`:记忆按 s^M 自更新,`pred_slot` 与 policy 承诺跟随 ŝ | 门控必须作用于记忆 | Full 更高 |
| w/o 强制置信 | **新开关** `force_conf=false`:PGIF 触发时不把 `_pred_conf` 设为 1,改用 PGIF 后验在可见 tracklet 上的 top1−top2 margin 过 `conf_tau` | 检验强制置信是否是增益来源(§5.5 公平性审计 G) | 探索性,不预设方向 |
| w/o 承诺门控 | **新开关** `gate_commit=false`:记忆按 ŝ 更新,`pred_slot` 与 policy 承诺跟随 s^M | 拆开 PGIF 的两个输出 | 探索性,不预设方向 |

- 每个变体只改一个配置项,其余与 Full 完全相同。
- **预注册假设**:H-A1 至 H-A4 为 Full 在主指标上高于前四个变体(Holm,4 个比较);第五个变体只做描述。
- **机理量**(tick 日志,真值只用于离线打分):每臂报告门控改写率,以及改写中拉向真目标与拉离真目标的比例。

## 6. 表 3 · TAHRelM 内部消融(机制实验:Town05 test,n = 40 × 3 个训练种子,共享开局,PGIF 保持开启)

**单变量的实现方式:特征掩码。** 网络结构和参数量完全不变(MLP 输入仍为 7 维),只把指定的特征列在训练和推理时都置 0。训练数据、优化器、学习率、epoch 数、增广、验证划分和选模规则全部相同。选模规则与第 1 节一致:固定验证划分上的最佳 val checkpoint(现有 `tah_train.py` 的行为)。

| 变体 | 置 0 的特征(共 7 维:cos, relp_x, relp_y, depth, rank, dmot, msoft) | 检验 |
|---|---|---|
| w/o 软运动门 | msoft | 可学运动门是否必要 |
| w/o 相对运动(**特征组**消融) | relp_x, relp_y, dmot, msoft | 运动一致性整体是否必要;是一组特征,不是单个特征 |
| w/o 外观关系(**特征组**消融) | cos, rank | 外观记忆相似度是否必要 |
| w/o DAgger(**训练数据**消融) | 不置零;训练数据只用离线 64 ep(样本量更少,分布不同) | 部署分布数据是否必要;改变的是训练数据,不是模型 |
| 手工规则(非单变量,单列) | 相同 7 个特征,按先验符号等权线性打分,不训练(0 参数) | 学习 vs 手工设计;不计入 Holm 族 |

- **统计层级**:训练种子是模型层面的随机效应,episode 是环境层面的随机效应。
  - Full 与 4 个训练变体各训 3 个种子(s0–s2),每个模型在同一组 40 个 test 种子上闭环。
  - 采用两层 bootstrap:外层重采样训练种子,内层按 episode 配对重采样。不把 3 × 40 当作 120 个独立样本。
- **离线结果**:每个模型报告验证集 top-1 准确率(3 个种子的均值 ± 标准差)和参数量(全部 4,740)。
- **预注册假设与判定**:H-T1 至 H-T4 为 Full 在主指标上高于前四个变体。**判定依据是两层 bootstrap 的 Bonferroni 校正 CI**(4 个比较,双侧 98.75% CI):下界 > 0 才算成立。p 值仅作描述,不用于判定。
- 只有 3 个训练种子,训练稳定性的结论写得保守:只报告种子间范围,不做种子层面的显著性结论。

## 7. 表 4 · 通用性

**(a)跨记忆(机制实验:Town05 test,n = 60,共享开局)**:PGIF 包在另外两种记忆上,各自比较加与不加。两臂都用共享开局。
- DINOv2 gallery re-ID ± PGIF:预测加 PGIF 后更高,依据是其自置信 AUROC 为 0.82(1001 §2)。
- 表 4(a)的两个预测只报效应量与 95% CI,**不作显著性判定**,也不进入任何 Holm 族。
- DAM4SAM ± PGIF:**预测无显著增益**。依据是 SAM2 预测 IoU 对选对目标的 AUROC 只有 0.53(1003),而命题 2b 的置信加权需要有信息的置信度。这是一个可证伪的反向预测,不论结果如何都报告。

- **外部基线也使用理想 tracklet ID**(对应 §5.5 公平性审计 A):OC-SORT-style 与 DeepSORT-style 的关联改为直接继承 `Candidate.idx`(不再自行做运动或外观匹配),即与 PGIF 享有同一关联层。用来区分「增益来自 PGIF 的证据机制」与「来自理想关联」。只报效应量与 CI。

**(b)跨地图(部署实验:Town03、Town10HD test,各 n = 40,自然开局)**:Ours、TAHRelM w/o PGIF、DAM4SAM 三臂;每张地图同样先跑参考开局,用于分层。
- 预注册 H-G(Holm,2 个):两图合并(按地图分层的配对 bootstrap)时,Ours 在主指标上高于 w/o PGIF 与 DAM4SAM。单张地图只做描述。

## 8. 表 5 · 假设边界(机制实验:Town05 test,n = 60,共享开局;Full 与 w/o PGIF 两臂)

| 扰动 | 档位 | 注入位置 |
|---|---|---|
| 关联噪声:tracklet ID 互换(**优先在相邻、外观相似的车辆之间**)与碎片化,每 tick 概率 p;**必做** | p = 0.02、0.05 | `candidates.py::build_candidates` 输出之后、policy 使用之前;按种子可复现 |
| 检测噪声:框中心与尺寸抖动 σ,漏检率 m | σ = 框尺寸的 5%,m = 10% | 同上 |
| 换控制器(可选,T3) | DiT → 跟随 policy 承诺目标的 PID | `policy.py` 的 HOW 输出 |

- 不做方向性假设。报告增益随扰动强度的衰减,写明在哪个强度下增益消失。
- 噪声只作用于身份链路的输入;scorer 仍用无噪声的真值打分。

## 9. 超参敏感度(dev 种子 93000–93019,n = 20,共享开局)

- λ ∈ {0.95, 0.98, 0.99},W ∈ {10, 30, 100},每次只改一个,其余保持冻结值。
- 只画曲线,不用于选参,用来检验 1001 中按先验设定的 λ、W 是否稳健。

## 10. 统计分析计划

1. **分析单元与检验**:episode,按种子配对。
   - **p 值**:配对符号翻转置换检验,双侧。对每个种子的配对差 d_i 随机翻转符号,重复 100,000 次;统计量为 Σd_i。合并多地图时在地图内翻转。
   - 表 3 例外:只有 3 个训练种子,episode 层的置换检验不能充分反映训练种子这一层的随机效应。因此表 3 以两层 bootstrap(外层训练种子、内层 episode)的 Bonferroni 校正 CI 作为判定依据。描述性 p 值是先对 3 个种子取平均得到每个 episode 的配对差、再做翻转,并注明它是「以这 3 个训练种子为条件」的检验。
   - **置信区间**:配对 bootstrap 95% CI,10,000 次重采样;多地图按地图分层;表 3 用两层 bootstrap(外层训练种子,内层 episode)。
   - Holm 校正作用在置换检验的 p 值上;bootstrap 只给 CI。
2. **多重比较**:每张表的预注册主比较为一个族,使用 Holm–Bonferroni。
   - 表 1:H-M1–H-M2;
   - 表 2:H-A1–H-A4;
   - 表 3:H-T1–H-T4(按两层 bootstrap 的 Bonferroni 校正 CI 判定,见上);
   - 表 4(b):H-G 的 2 个比较。
   - 次指标、分层结果、描述性比较只报效应量与 CI,不作显著性结论。
3. **报告内容**:每个比较报告以下各项:
   - 均值差、95% CI、置换检验 p 值及 Holm 校正后的 p 值;
   - 参考开局正确 / 错误两层的条件均值与差值;
   - 每臂自己的开局锁对数(仅描述)、弃权率、fallback 次数。
4. **排除规则**:
   - 只排除 scorer 判为跳过的 episode(目标静止),这是场景级判定,应在各臂间一致。一张表内取所有臂共同计分种子的交集,并报告各臂的跳过列表是否一致。
   - 基础设施故障(仿真卡死、服务崩溃)导致的 episode 用同一种子补跑,保留原失败日志。
   - 不因任何结果排除 episode。
5. **冻结与变更**:本文件审计通过后冻结。任何修改都在文末以带日期的变更记录写明原因,且必须在查看对应 test 结果之前完成。

## 11. 不进入正文主表的内容

| 内容 | 去向 |
|---|---|
| 启发式阶梯 l4 / l5 / borrow#1 / TAH+DAgger | 附录「开发过程」一张表,使用 dev 种子结果并明确标注 |
| ASD 锚纠偏、V1–V3 融合、V2 延迟承诺 | 附录一段话 + 机理一句;正文「局限」用一两句话披露 |
| gt-seeded 的旧对比(09-06) | 不引用 |
| 口径修正 | 正文一句话 + 主表弃权率列 + 附录三口径对照 |
| `fig_architecture_legacy_pgif.png`(原 `fig_architecture.png`)中的 PoC 数字(n = 19,旧批次) | 删去,或在 test 完成后换成 test 数字 |

## 12. 需要新增或修改的代码(test 前全部在 dev 上完成并验证)

1. **`gate_commit` 开关**(`asd.py` + `policy.py`)。现状是 PGIF 触发时总会改写 `_committed_idx` 和 `pred_slot`。新开关为 false 时只改记忆更新,`pred_slot` 与 `_committed_idx` 跟随 s^M。
2. **共享开局**:
   - `run_rollout.py` / `RemotePolicy.reset()` 发送 `{"cmd": "reset", "seed": s}`;`policy_server.py` 的 reset 接收种子并传给 policy。
   - policy 读取 `OPENING_FILE`,在首个决策 tick 按 `Candidate.idx` 查当前 slot 并承诺;找不到时退回,并在日志里标记 `opening_fallback`。
   - 参考开局过程:`run_rollout.py --steps 12`,输出 `{seed: actor_idx, is_target}`。
3. **通用逐 tick 日志**:所有 tid head(tah / assoc / dam4sam / reid / xattn / oracle)每个决策 tick 记录以下字段:
   - `raw_mem_slot`、`pgif_slot`、`pred_slot`、`policy_committed_idx`;
   - 每个 slot 对应的 actor_idx、置信度、是否弃权及原因;
   - 真值 slot,仅供离线打分,任何决策都不读取。
   - scorer 在每帧额外记录 `scorer_committed_key`。
4. **`train/tah.py` + `train/tah_train.py`**:`--feat-mask`(列索引)开关;手工规则打分器;`--seed`;训练配方(命令、数据、epoch、选模规则)写入 ckpt 和日志。
5. **噪声注入**:`candidates.py` 增加 ID 互换 / 碎片化、框抖动 / 漏检,随机数按(种子, tick)确定。
6. **结果元数据**:写入第 1 节列出的全部字段。
7. **作业调度器**:按(臂, 10 种子块)随机分配作业到槽位,并记录调度种子。
8. (可选)PID 跟随控制器。

### T0 验收门槛(全部通过并经 codex 审计后,才允许启动 test)
| # | 门槛 | 验证方式 |
|---|---|---|
| G1 | 新开关全部关闭时,5 个 dev 种子的每 tick `pred_slot` 与 scorer 输出与冻结版本逐位一致;若仿真本身有非确定性,则差异不超过冻结版本自身两次运行之间的差异(A/A) | 对比脚本 + 日志 |
| G2 | 共享开局:每个机制臂类型(tah ± PGIF、reid、assoc、dam4sam)在 dev 上,首个 tick 的记忆原始选择、记忆写入、PGIF 首个证据、policy 承诺都指向同一个参考 actor_idx;fallback 被正确记录 | 日志检查脚本 |
| G3 | `gate_commit=false` 时 `pred_slot` 与 `_committed_idx` 跟随 s^M,记忆按 ŝ 更新;`gate_memory=false` 时相反 | 单元测试 + dev 日志 |
| G4 | TAH 重训:`--feat-mask`、`--seed`、多缓存合并、最佳 val 选模、配方写入 ckpt 全部可复现(同种子重训两次权重一致,或 val 指标一致) | 重训两次对比 |
| G5 | 重训 Full(s0)与冻结 ckpt 在 dev 上闭环等价(配对差 CI 包含 0) | dev 闭环 |
| G6 | 噪声注入按(种子, tick)可复现;关闭时与 G1 一致 | 单元测试 |
| G7 | 一次 dev 批次的 JSON 元数据、通用 tick 日志(所有 tid head)、`scorer_committed_key`、弃权率、fallback 都能由分析脚本复算出表格 | 端到端 dry run |
| G8 | 分析脚本(置换检验、bootstrap、Holm、分层、两层 bootstrap)冻结并打 tag | git tag |
| G9 | 外部基线 dev 网格跑完并按预注册规则选定参数;**每个 test 臂的完整最终命令行**(含 DEP、ckpt、ASD_CFG、`<dev>` 占位的实际取值)写入本文件附录并随 tag 冻结 | dev 结果 + 文件 diff |
| G11 | **DAM4SAM 逐帧更新对照**(dev 91000–19):每帧把图像送入 SAM2,身份决策仍在 S2 tick 读取。若空掩码率或承诺口径正确比例显著改善,则 test 中 DAM4SAM 采用逐帧更新(以对基线有利的一方为准) | dev 结果 |
| G12 | 带 `error` 的 episode(仿真超时等)由启动脚本自动判为失败并同种子补跑,`progress.log` 计入 fail | 启动脚本单元测试 |
| G13 | **遮挡可见性**:env 用射线或语义分割计算目标是否真正可见,写入 scorer(`target_visible`)与逐 tick 日志;主表补充「可见帧 / 视锥内被遮挡帧」分开的正确比例和弃权率(§5.5 公平性审计 4.5) | dev 冒烟 + 抽样人工核对 |
| G10 | **防隐藏特权**:启动器拒绝启动不带 `--gallery-seed committed --reanchor off` 的臂(灰色上界臂在白名单中显式标注);`gallery_seed`、`reanchor`、`tid_head` 写入结果元数据;分析脚本在汇总前逐臂断言,不符即报错 | 启动器单元测试 + 一次故意漏写的负例 |

## 13. 算力与排期

按每集约 3.75 分钟、4 个 CARLA 槽位并行估算(参考开局过程每种子约 0.5 分钟):

| 档位 | 内容 | episode 数 | 墙钟时间 |
|---|---|---|---|
| T0 | 代码改动、逐位一致检查、Full 重训(3 种子)与 dev 等价性、基线 dev 网格(31 组新设定 × 20;DAM4SAM 已跑 3 组)、敏感度曲线(4 × 20)、参考开局 ×2 | 约 820 + 参考 | 约 13 h |
| T1(必须) | 表 1:10 臂 × 80;表 2:7 臂 × 60 | 1,220 | 约 19 h |
| T2 | 表 4(a):4 × 60;表 4(b):3 臂 × 2 图 × 40;表 5:6 × 60 | 840 | 约 13 h |
| T3 | 表 3:(Full + 4 变体)× 3 训练种子 × 40 + 手工规则 × 40;可选换控制器 2 × 60 | 640 + 120 | 约 12 h |

- 合计约 57 h 墙钟时间,约 2.5 天。
- T1 完成即可得出部署主结论和 PGIF 机制结论;T2、T3 不改变 T1 的预注册判定。
- 如能稳定增加第 5 个 CARLA 槽位,总时长约缩短 20%。

## 14. 风险与预案

| 风险 | 预案 |
|---|---|
| 参考开局准确率偏离自然水平,或 fallback 多 | 报告二者;机制表的结论只在「给定开局」意义下成立,部署结论以表 1 为准 |
| test 上 Ours 相对 DAM4SAM 不显著 | 如实报告;主张收窄为「参考开局正确条件下的抗漂移」(分层)+ 机理 |
| 重训的 Full(s0)与冻结 ckpt 在 dev 上不等价 | 以重训版本为准(配方可复现),附录报告两者差异 |
| CARLA 卡死或主机重启 | watchdog 续跑;每集增量保存;同种子补跑 |
| 外部方法在 dev 调参后仍明显异常 | 附录报告其 dev 结果与失败原因,不在 test 上二次调参 |

## 15. 模型与实现代码索引(冻结前的现状;行号以 2026-10-02 工作区为准)

> 路径约定:`R/` = `carla_uav_tracking/rollout/`,`T/` = `train/`。标「待实现」的项属于 T0。

### 15.1 所有臂共用的闭环链路
| 环节 | 实现 | 说明 |
|---|---|---|
| 启动脚本 | `R/eval_asd.sh` | 每个臂一个 policy 容器(`acot-uav-train:cu118`,`cyh-carla-net`)+ 一个 CARLA 端口。固定参数:`--config train/config_v5.yaml --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period 6 --ex-source track`;臂特有参数由 `HEAD_ARGS` 传入 |
| 断点续跑 | `runs/asd/iter/launch_wd.sh` | 12 分钟无新 episode 视为卡死,重启 CARLA 后从下一个种子续跑,各段合并 |
| 仿真客户端 | `R/run_rollout.py`(`--town` 并入场景城镇列表,每集增量保存 `.partial`);`scene/scene_manager.py`(每集清场时 `set_autopilot(False, _tm_port())`,2026-10-02 修复跨 worker TM 卡死) | python 3.8,`cyh-carla` 容器 |
| 每集循环 | `R/env.py::StudentRolloutEnv.run`(:113;每集开头 `policy.reset()` :145;逐帧 `scorer.update` :210) | |
| 远程策略 | `R/remote_policy.py::RemotePolicy`(reset / act / close);`R/policy_server.py`(:64 处理 `reset` / `act` / `close` 命令) | reset 目前**不传种子**,共享开局需要改(待实现) |
| 候选集 | `R/candidates.py::build_candidates`(:44):由真值车辆位置投影出理想框,带持久 `Candidate.idx`(目标为 −1),**每帧顺序打乱**;`true_idx` 只给 scorer 和离线日志用 | 「候选框为仿真理想输入」的来源;噪声注入点(待实现) |
| 主干 | `R/policy.py::load_policy_modules`(:71,从 stage-2 ckpt 重建 EAR / IAR / DiT / 语言 grounding 头);`R/policy.py::Policy.act`(:547):S2 慢流每 6 帧跑 VLM → EAR → IAR → 身份,S1 快流每帧跑 DiT | 全部冻结 |
| 语言冷启动 | `T/dit.py::TargetIDHead`(:85,候选对 VLM 上下文做 cross-attention);`R/policy.py:583–585` 得到 `lang_slot` | 所有用语言初始化的臂共用 |
| 承诺与 WHERE | `R/policy.py::_update_commit`(:501,K 帧滞回)、`_committed_world`(:528,被承诺车辆的世界坐标)、`estimator.update`(:876) | 「车辆定位为仿真理想输入」的来源 |
| 打分 | `R/metrics.py::RolloutScorer`(:33;`update` :83;scorer 层 K = 5 滞回 :92–107;`mis_follow_sustained` :190;`max_wrong_run_s` / `max_correct_run_s` :195–196) | 主指标来源;`scorer_committed_key` 逐帧落盘(待实现) |

### 15.2 最终模型:TAHRelM + PGIF
| 部件 | 实现 | 说明 |
|---|---|---|
| TAHRelM 模型 | `T/tah.py::TAHRelM`(:88–140)。特征 `x = [cos, relp_x, relp_y, depth, rank, dmot, msoft]`(:121);MLP 7·64·64·1(:102);可学标量 `gate` / `vbeta` / `mgate`(:99–101);记忆更新 `update`(:127,外观 EMA + 速度 EMA) | 4,740 参数 |
| TAHRelM 训练 | `T/tah_train.py`:`--motion --aug`,`--cache`(:56,目前单文件),验证划分 `--val-frac`(:60、:73),保存最佳 val(:85–91) | 多缓存合并、`--feat-mask`、`--seed`、配方写入 ckpt 均待实现 |
| 部署分支 | `R/policy.py:651–736`:crop-DINOv2 特征 → `TAHRelM.step` → s^M(:665);committed 模式冷启动用 `lang_slot`;记忆写入源 `us = _pred_slot`(:728),写入前快照 `_tah_mem_prev`(:732) | `--tid-head tah --tah-ckpt runs/tah_relm_dag.pt --conf-tau 0.5 --gallery-seed committed --reanchor off` |
| PGIF 证据 | `R/asd.py::ASD._filter_step`(:190–208):`L_k ← 0.98·L_k + (0.5 + 秩(conf; 最近 30 个))·1[k = s^M]`;ŝ = 当前可见 tracklet 中证据最大者;`fire = (ŝ ≠ s^M)` | `--asd filter --asd-cfg '{"filter_kind":"tempered","resid_decay":0.98}'`;W 即 `conf_window = 30`(:27) |
| PGIF 门控 | `R/policy.py:793–843`:调用 `ASD.step`(:819,置信度 = TAH 的 top1−top2 margin 的 sigmoid);fire 时执行 `_asd_gate_memory(ŝ)`(:334–352,用 ŝ 的特征从 `_tah_mem_prev` 重做本 tick 的记忆写入),并改写 `pred_slot` 与 `_committed_idx`(:832–837) | `gate_memory` 由 ASD 配置控制;`gate_commit` 待实现 |

### 15.3 内部消融的实现状态
| 变体 | 实现方式 | 状态 |
|---|---|---|
| w/o PGIF | `--asd off` | 已有 |
| w/o 回火 | `"filter_kind":"bayes"`(`R/asd.py:209–232`,粘滞 HMM 逐 tick 似然相乘) | 已有 |
| w/o 置信加权 | `"filter_kind":"resid"`(:200,w = 1) | 已有 |
| w/o 记忆门控 | `"gate_memory":false`(:51;`R/policy.py:832` 跳过 `_asd_gate_memory`) | 已有 |
| w/o 承诺门控 | `"gate_commit":false`:fire 时只做 `_asd_gate_memory`,不改 `pred_slot` / `_committed_idx` | **待实现** |
| TAHRelM 特征掩码(软运动门 / 相对运动组 / 外观关系组) | `T/tah.py:121` 处对 `x` 的指定列置 0,训练与推理一致;ckpt 记录掩码 | **待实现** |
| w/o DAgger | `T/tah_train.py --cache runs/tah_cache.pt`(只用离线 64 ep) | 已有(需配合新配方) |
| 手工规则 | 同 7 个特征,按先验符号等权线性打分,不训练 | **待实现** |
| 超参敏感度 | `"resid_decay"` ∈ {0.95, 0.99};`"conf_window"` ∈ {10, 100} | 已有(ASD 配置项) |
| 共享开局 | reset 传种子 → `OPENING_FILE` → 首个决策 tick 在 PGIF 之前替换 s^M | **待实现** |
| 关联 / 检测噪声 | `R/candidates.py::build_candidates` 输出之后注入 | **待实现** |

### 15.4 外部基线与诊断上界
| 臂 | 实现 | 部署配置(dev 网格见第 4 节) |
|---|---|---|
| OC-SORT-style | `R/policy.py::_assoc_select`(:399–444),`assoc_mode="motion"`:图像平面匀速预测 + 关联门;committed 模式下轨迹状态用自己关联到的车更新,冷启动用 `lang_slot` | `--tid-head assoc --assoc-mode motion --conf-tau 0.05` + DEP(dev 现值) |
| DeepSORT-style | 同上,`assoc_mode="deepsort"`:运动 + crop-DINOv2 外观余弦(EMA 模板),权重 `--assoc-lambda` | `--tid-head assoc --assoc-mode deepsort --conf-tau 0.05` + DEP(dev 现值) |
| DAM4SAM | `R/policy.py::_dam4sam_select`(:445–485):冷启动用 `lang_slot` 的框初始化 SAM2,之后每个 S2 tick 跟踪 → 掩码转框 → 匹配关联门内最近候选,否则弃权;诊断日志 `_dam_log`(:486,`DAM4SAM_DIAG=1`)。服务 `R/dam4sam_service.py`(init / track / close,返回框与预测 IoU),跟踪器 `external/DAM4SAM/dam4sam_tracker.py::DAM4SAMTracker.track`(:178),模型 SAM2.1-L,容器 `dam4sam-svc`(conda env `dam4sam`,GPU6,端口 5601–5604) | `--tid-head dam4sam --dam4sam-host dam4sam-svc --dam4sam-port 560x --conf-tau 0.05 --assoc-gate-px <dev 选定>` + DEP |
| DINOv2 外观 gallery re-ID | `R/policy.py::_reid_select`(:295–320):K 视角 gallery、top-m 均值匹配(`T/reid_resolution_probe.py` 的 `bank_score` / `bank_update`);committed 模式下 gallery 只存承诺身份的裁剪 | `--tid-head reid --reid-feature dinov2 --reid-bank 2 --reid-topm 2 --conf-tau 0.05` + DEP(dev 现值) |
| 单帧语言 grounding | `T/dit.py::TargetIDHead`(:85),每 tick 取 argmax,无记忆 | `--tid-head xattn` + DEP |
| 真值身份上界(灰) | `R/policy.py:743–749`:每 tick 直接选 `true_idx` | `--tid-head oracle` |
| 真值刷新记忆上界(灰) | `R/policy.py:728`:`--gallery-seed gt` 时记忆写入源为 `true_idx` | `--tid-head tah --gallery-seed gt` |
| PGIF 包外部记忆(表 4a) | reid:`--asd filter`,门控通过「只存承诺身份」实现(`_asd_gate_memory` 对 reid 不做额外操作,:334 注释);DAM4SAM:`_asd_gate_memory` 用 ŝ 的框重新初始化 SAM2,冷却 3 个 tick(:339–347),置信度 = SAM2 预测 IoU(:818) | 同各自基线配置 + `--asd filter` + DEP |

### 15.5 分析与统计脚本
| 脚本 | 用途 |
|---|---|
| `R/iter_gate.py` | 三种正确比例口径 + 最长连续正确 / 最长锁错,配对 bootstrap;多段结果自动合并 |
| `R/iter_mech.py` | tick 日志机理量:各源准确率、误差相关、门控改写、开局分层 |
| `R/deploy_compare.py` → `acot_note/figs/fig_deploy_compare.py` / `update_s55_table.py` | 1001 §5.5 主表与对比图 |
| `R/dam_diag.py` | DAM4SAM 弃权原因与关联门扫描 |
| `R/power_sim.py` | 双峰结果模型下的功效模拟(`--sens` 为悲观敏感性) |
| `R/tick_align.py` | tick 日志与 scorer 种子对齐(跳过的 episode 不触发 reset) |
| 置换检验 + Holm + 两层 bootstrap 的正式分析脚本 | **待实现**,T0 结束时冻结并打 tag(G8) |

## 审计记录
| 轮次 | 结论 | 主要问题 | 文件 |
|---|---|---|---|
| R1 | MAJOR(12 + 7 MINOR) | 主表缺部署自消融;共享开局定位;分层对撞偏倚;双峰功效;训练种子层级;非单变量;基线调参;弃权口径;代码与算力低估 | `codex/1004_ablation_R1.md` |
| R2 | MAJOR(可开始 T0,不可开始 test) | 共享开局注入语义;重训配方与代码冲突;p 值定义;悲观功效;主比较族理由 | `codex/1004_ablation_R2.md` |
| R3 | MAJOR(3 条) | 选模规则矛盾;外部基线隐藏 GT 默认;表 3 p 值偏乐观 | `codex/1004_ablation_R3.md` |
| R4 | **NO MAJOR** | 3 条 MINOR,已修 | `codex/1004_ablation_R4.md` |

> test 启动条件:T0 的 G1–G10 全部通过,并经 codex 对实现另行审计(重点 G2 / G7 / G9 / G10)。

## 变更记录
- v1(2026-10-02):初稿。
- v4.6(2026-10-03 05:15,deployable v2 的 dev 验证,未涉及 test;详见 `1004_s55_fairness_audit.md` §8):
  - **deployable 的定义需要升级**:身份链路改用跟踪器 ID,加入遮挡感知候选,WHERE 只通过可见框定位。理想 ID 和不考虑遮挡的候选降为诊断上界。
  - 跟踪器参数在 dev 回放上调参后冻结:`TRK_CFG = {max_age 8, w_app 0.85, gate_px 150, gate_box 2, min_cos 0.2}`。
  - PGIF 新增 `absent_hold`(不可见假设),在 dev 上方向为正但不显著。
  - 在这一设定下,原有 PGIF 的增益消失。**原预注册的表 1 与表 2 需要在新设定下重新确定主张,并经审计后再进入 test。**
- v4.5(2026-10-03 02:30,G11 完成,dev 91000–19,n = 19,未涉及 test):DAM4SAM 逐帧更新(`damdiag_ef_g160`)与每 6 帧一次(`damdiag_g160`)配对比较:
  - 承诺口径 −0.06 [−0.24, +0.13],严格口径 −0.00 [−0.05, +0.05],最长锁错 +7.2 s [−2.0, +16.8],都不显著;
  - 逐帧统计的空掩码仍有 69%(6,735 / 9,695 帧);给出选择时的准确率由 0.59 降到 0.48。
  - **按 G11 规则(以对基线有利的一方为准),test 维持每 6 帧一次,关联门 160 px。** 降帧不是 DAM4SAM 弃权的主因。
- v4.4(2026-10-03 00:10):新增 G13,加入遮挡可见性标志。候选集与 `target_present` 原先只按视锥判断,不考虑遮挡。
- v4.3(2026-10-03,§5.5 公平性审计 `1004_s55_fairness_audit.md` 的修补,未涉及 test):
  1. 表 2 新增 w/o 强制置信(审计 G)。
  2. 表 4(a)新增「外部基线也使用理想 tracklet ID」(审计 A)。
  3. 表 5 的关联噪声改为必做,并优先在相邻相似车辆间互换(审计 A、I)。
  4. 新增 T0 门槛 G11(DAM4SAM 逐帧更新对照,审计 C)、G12(故障 episode 判失败并补跑,审计 F)。
- v4.2(2026-10-02 23:45,G9 部分结果,未涉及 test):DAM4SAM 关联门在 dev 91000–19 上的扫描完成(`rollout/dam_diag.py` → `eval_out/dam_diag.json`,`conf_tau = 0.05`)。按预注册规则(dev 主指标最高)**选定 160 px**:承诺口径 160 / 240 / 320 px = 0.447 / 0.416 / 0.355;320 − 160 = −0.09 [−0.19, −0.01]。`conf_tau` 网格待 T0 完成。
- v4.1(2026-10-02):codex R4 **NO MAJOR**,冻结。按 R4 的 MINOR 修订:
  1. `power_sim.py` 默认每格 20,000 次模拟,并加 `--sims` 参数。
  2. 第 3.2 节注明 `gate_commit=false` 是 `pred_slot` 定义的例外。
  3. 表 4(a)只报效应量与 CI,不作显著性判定。
  - 冻结之后的任何修改,只能以带日期的变更记录追加,并且必须在查看对应 test 结果之前完成。
- v4(2026-10-02):按 codex R3(`codex/1004_ablation_R3.md`,MAJOR,3 条)修订:
  1. 表 3 选模规则统一为最佳 val,与第 1 节和现有代码一致。
  2. 所有 deployable 臂的命令行显式包含 `--gallery-seed committed --reanchor off`(服务端默认为 `gt`,会带来隐藏特权);G9 改为冻结每个 test 臂的完整最终命令;新增 G10,由启动器与分析脚本双重强制检查。
  3. 表 3 改以两层 bootstrap 的 Bonferroni 校正 CI 判定,p 值仅作描述。
  4. MINOR:功效表用 20,000 次模拟重算。
- v3.1(2026-10-02,用户要求):新增第 15 节「模型与实现代码索引」(最终模型、内部消融、外部基线、分析脚本的代码位置与实现状态);按实际部署配置修正 re-ID 基线参数;外部基线网格统一加入 `conf_tau`;T0 算力相应更新。
- v3(2026-10-02):按 codex R2(`codex/1004_ablation_R2.md`,MAJOR;判定可开始 T0、不可开始 test)修订:
  1. 精确定义共享开局的注入语义:在 PGIF 之前替换记忆原始选择,并驱动记忆写入、首个证据、`pred_slot` 与承诺,4 项都记录日志。
  2. TAH 重训配方与代码对齐:沿用最佳 val 选模,固定划分种子;多缓存合并列入 T0。
  3. p 值改用配对符号翻转置换检验,bootstrap 只给 CI。
  4. 功效模拟改用同一检验,并补悲观敏感性表与最小可检出效应。
  5. 写明主比较族的选择理由。
  6. 新增 T0 验收门槛 G1–G9,代码实现在门槛处另行审计。
  7. 表 3 标注特征组消融与训练数据消融;训练稳定性结论写得保守;外部基线定位为机制对照。
- v2(2026-10-02):按 codex R1(`codex/1004_ablation_R1.md`,MAJOR)修订,逐条对应如下:
  1. 表 1 加入自然开局的 TAHRelM w/o PGIF。
  2. 共享开局降格为机制实验。
  3. 开局按 actor_idx 存取。
  4. 分层变量改为独立参考开局。
  5. 改用双峰模型的模拟功效分析,表 1 的 n 提高到 80。
  6. 表 3 采用两层 bootstrap,3 个训练种子。
  7. TAHRelM 消融改用特征掩码,做到单变量;手工规则单列。
  8. 基线调参改为预注册网格,基线命名为 -style。
  9. 弃权率进入主表。
  10. 补全代码清单,加入逐位一致检查。
  11. 明确两层承诺与日志字段。
  12. 重算算力。
  13. MINOR:训练入口、元数据、双 A/A、批次交错、排除规则、负结果披露、架构图数字。

- **2026-10-04(引用更新，不改实验设计)**:最终模型结构图改为 `figs/fig_architecture_v1.png`,内容为部署设定 v2、属性开局 v1、PGIF absent-hold、TAHRelM `tah_relm_dag.pt`,并披露理想化感知输入。原 `fig_architecture.png` 更名为 legacy `fig_architecture_legacy_pgif.*`,对应旧的理想 ID 设定;更早的仅 TAHRelM 图更名为 `fig_architecture_legacy_tahrelm.*`。开局修复和协议变更对 T0 的影响，分别见 `1004_opening_fix.md` 与 `1004_tahrelm_retrain_diag.md`。
