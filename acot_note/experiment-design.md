# ACoT-UAV-Track 完整实验设计方案

> 2026-07-29 · 实验设计 agent 产出 · 配套：`acot-uav-design.md`(§6/§7 主张与假设)、`training-plan.md`(阶段/门槛)、`dataset-card.md`(MVP 数据)、`learning_diary_0729.md`(已得发现)
> 目标数据：`/nvidia/hque/data/carla_data/mvp`(60 集 / ~100K 帧 / 5 地图, test_town=Town05)
> 训练侧现状：`train/stage2.py` 全损失已通;Qwen3-VL-4B layer 24 缓存(`ctx_cache_ml_neutral`);6-run seed 对比(`seed_exp.sh`)进行中

---

## 0. 设计原则(先立规矩,再排实验)

1. **实验服务于主张,不服务于跑分。** 本工作唯一可辩护的新颖性是 §6.3 的四轴交集(空中·车辆·语言消歧·闭环)。所有实验的第一职责是**证明语言 load-bearing**;SR/act_mse 只是配角。若语言必要性证伪,再高的 SR 也无贡献 → 这条决定了实验**排序**(§2 gate 先行)。
2. **离线指标 ≠ 闭环能力。** 现有 mis_follow / act_mse 都是**逐样本离线**评的(冻结缓存 + 有标签)。而"具身闭环跟踪"的主张只有 **CARLA on-policy rollout** 能兑现。设计必须两条腿:离线(便宜、可大批消融)+ 闭环(贵、少量、兑现主张)。见 §3、§7。
3. **诚实定位写进实验。** 机制迁移自 ACoT-VLA,不是发明。实验对照要能把"迁移适配的增量"单独量出来(EAR/IAR/语言/课程各自净贡献),而不是只报一个总分。
4. **MVP 与可发表分层。** MVP(100K 帧/5 图但每图少)只能给**初步**证据 + 打通管线;统计显著消融、强跨地图泛化、SOTA 规模对比留给 scale-up(~300K)。每条结论都要标注"MVP 可交付 / 需 scale"。
5. **每个数字带不确定度。** 测试集只有 6 集 → 单点数值噪声大。一律 **3 seed × (per-episode bootstrap CI)**,报 mean±std 且做**配对**比较(§8)。

---

## 1. 主张 → 假设 → 实验(逻辑主干)

| 编号 | 主张 | 可证伪假设 | 关键实验 | 主指标 | 类型 |
|---|---|---|---|---|---|
| **H0** | 任务需外部指定,且语言唯一表达细粒度区分(核心生死线;重构见 §2.0) | 去掉语言,**在去相关+连续性断裂条件下**消歧显著变差 | w/o-Language + 外观/运动-only + 反事实语言(LGS) + **连续性断裂再捕获** | **Mis-follow ↑ / LGS / 断裂后增幅** | 离线为主 + 闭环抽验 |
| **H1** | EAR 显式 waypoint 推理有独立贡献 | 去掉 EAR,遮挡/急转场景变差 | ACoT vs w/o-EAR | 遮挡恢复率、minADE/FDE、Δaction on 机动帧 | 离线 + 闭环 |
| **H2** | IAR 隐式先验有独立贡献 | 去掉 IAR,机动/即将遮挡预警丢失 | ACoT vs w/o-IAR | 急转/急停帧提前量、遮挡恢复率 | 离线 + 闭环 |
| **H3** | 动作空间 CoT > 语言 CoT | 自建 Language-CoT 头精度更差 | ACoT vs Language-CoT 头(ECoT 风格) | minADE/FDE(3D 轨迹)、Mis-follow | 离线 + 闭环 |
| **H4** | 中性语言 > 含时变从句语言(已初步得) | 恢复时变从句,mis_follow 回升 | 方案A 消融(已做, 需 3-seed 坐实) | Mis-follow 3.5%→1.75% | 离线 |
| **H5** | 完整系统 ≥ 头号竞品同场景基线 | UAV-Track VLA 风格基线不弱于 ours | 竞品复现 / π0.5 微调 | SR、Mis-follow(闭环) | 闭环 |
| **H6(可选)** | Stage-3 RL 治 BC 三病 | RL 后不优于 BC 或 reward hacking | GRPO 精调 vs Stage-2 BC | 闭环 SR、恢复率、无 hacking | 闭环 |

> **优先级**:H0 是门(不过则全盘停);H4 已近完成;H1/H2/H3 是方法贡献主体;H5 是外部可比性;H6 后置可选。

---

## 2. 生死线实验:语言必要性 gate(**必须最先做,先于所有消融**)

**为什么排第一**:数据卡显示 look-alike 共视仅 **13.6%** 帧、mean occlusion **0.045**、当前 mis_follow 已低到 **1.75%**。这三点合起来是**红色风险**:任务可能太易,模型可以靠"跟最居中/最近那辆"刷分而**根本不读语言**。若如此,H1–H5 全部无意义。所以先用一组廉价离线实验判定任务是否 well-posed。

### 2.0 H0 的可辩护性与重构(文献证据,deep-research 2026-07-29,22/25 断言过 3 票对抗验证)

**H0 不会自动崩,也不会自动安全——它锁在数据构造上,且威胁比"VA vs 语言"更精确。**

**(i) 前提被强证据支持:VLA 普遍无视语言、走捷径**(peer-reviewed:*Shortcut Learning in Generalist Robot Policies* @CoRL 2025;*VLABench* @ICCV 2025)
- π0.5 在**矛盾指令**下仍 96.2% 成功 → Linguistic Grounding Score 仅 **1.2**(几乎没读语言);控制掉抓取成功率后 VLA 选对语义目标≈**随机**(4 选 25%/10 选 10%)。→ 这是**弹药**(证明"走捷径"是公认难题),但也**警告**:观察到"去语言掉点"**不等于**证明语言必要——掉点也可能只是**别的**捷径没了。

**(ii) 语言何时才 load-bearing:必须"去相关",且有量化效应**
- **CAST**(2508.13446):有相似干扰车时标准 VLA **31.7% → 反事实增强 58.3%(1.84×)**;**无干扰时"指令条件化根本没必要"**。→ **效应量参照:decorrelate 后语言消融应看到 ~26 个点差距**;你 MVP 的位置捷径(memory 记 position-only ~3–5% mis_follow)正是"没解耦"的教科书案例。

**(iii) 最危险的反例:TrackVLA++ 用空间推理消歧,不是语言**(头号竞品,2510.07134)
- EVT-Bench **DT 干扰 split** 上,增益全来自**非语言模块**:Polar-CoT(极坐标空间推理)**+6.0%**、门控时序记忆 TIM **+2.8%**,**无语言消融**。→ **视觉相似车消歧可由空间连续性+记忆扛下**,不必靠语言。

**(iv) NL-tracking 里语言相对 bbox 是"互补"非"必要"**(最弱环;TNL2K@CVPR'21 / JointNLT@CVPR'23 / QueryNLT@CVPR'24)
- 语言定位为"补 bbox 歧义 + 帮重捕获",联合融合 +2.5% AUC,**但无干净的"给了 bbox 后语言仍必要"消融**;甚至有"bbox 扛了大部分定位信号"的说法。→ **所以设计里绝不能给持续首帧 bbox 身份锚**(见 §9 R3 + 设计文档 §2.1/§3.1 待删)。

**⚠️ 新增威胁:闭环跟踪的第二条捷径 = 时空连续性**
即使**初始**位置去相关,闭环里模型只要第 1 帧锁对目标,之后可纯靠**视觉/运动连续性**跟到底,**整段不再读语言**(正是 TrackVLA++ 机制)。→ **天真的"闭环 SR 有无语言"消融可能不掉点**(吃连续性红利)。**必须制造连续性断裂事件**(遮挡/车辆交叉/出画重入),使 re-acquisition **只能靠语言**。

**H0 重构(推荐,小步不弃):** 从"语言必要"收窄为——
> **任务需要外部目标指定;在视觉相似候选中,细粒度语言表达了 click/bbox/GPS 无法表达的属性/关系区分 → 语言是唯一具此表达力的指定方式。**

配套用领域标准方法学证明(有 peer-reviewed 先例):**反事实指令 + LGS 指标(正常 SR − 矛盾 SR)+ no-language VA 基线**(LIBERO-CF 的 CAG 分支 / VLABench Track 4:仅换语言 → 微调 VLA 掉 31–45%)。

**升级的 H0 成立判据(在原两条上加两条,写死):**
```
H0 成立 ⟺ (a) 无持续首帧 bbox 身份锚(语言-only 指定)
         ∧ (b) 目标身份 ⊥ 位置/几何(去位置捷径)
         ∧ (c) 目标身份 ⊥ 时空连续性 ← 【新增】必有连续性断裂事件,re-acquisition 只能靠语言
         ∧ (d) ≥2 辆"语言可分、但外观/运动不可分"的干扰车共视足够
现状: (a)实现满足/文档违反 · (b)不满足(有位置捷径) · (c)未设计 · (d)偏低(14%) → 全可修
```

### 2.1 五联证据(全部离线,复用现有缓存,1–2 天可出)

| 证据 | 做法 | 通过判据 | 失败则 |
|---|---|---|---|
| **E1 w/o-Language 消融** | 训练/评测时把语言 token 置空(或换成固定"track the vehicle") | Mis-follow **显著上升**(目标 ≥ 2× 且 CI 不重叠) | 语言无净贡献 → 任务太易 |
| **E2 外观/运动-only 基线** | 完全不喂语言,只给视觉+proprio 训一个头 | 其 Mis-follow **明显高于**完整模型 | 外观即可消歧 → 须加难(§2.3) |
| **E3 反事实语言切换 + LGS** | 同帧把指令改指另一辆车,看 target-id/动作是否切换;报 **LGS=正常 SR − 矛盾指令 SR** | 切换成功率高(≥70%)且 **LGS 显著 >0** | 模型没"听懂"语言(LGS≈0 = 走捷径) |
| **E4 难例子集分层** | 只在"≥2 相似车共视"帧上单独报 mis_follow | 该子集 mis_follow 明显高于全集,语言消融在此增益最大 | 难例太少 |
| **E5 连续性断裂再捕获** ★新增 | 只在遮挡/交叉/出画重入**之后的重捕获帧**上报有无语言的 mis_follow | **断裂后**去语言 mis_follow 大涨(连续性失效→只剩语言) | 若断裂后也不掉 → 数据无"语言唯一可分"事件,须加难 |

> **实施**:E1/E2 改 `backbone_kv.neutralize_language` 到"完全空语言"档 + 重建缓存;E3 同帧多 prompt 前向(只推理)+ 记 LGS;E4 用 `distractors` co-visibility 切片;**E5 用 `annotation/{occluded,off_screen}` + `distractors/positions` 交叉事件定位"断裂帧",取其后 N 帧评测**——这是把"语言 load-bearing"和"时空连续性捷径"分开的关键。

### 2.2 判据与分叉(写死,别临场松动)
- **全过** → 任务 well-posed,进入 §5 主实验。
- **E1/E2/E5 不过(语言不必要)** → **不要硬推下一阶段**。回数据侧加难:提高 look-alike 共视率(数据卡 §6.2:离屏平滑回收)、目标不恒居中、更多并行/交叉、**制造连续性断裂事件**。此为 **A1c③ 场景重设计**,MVP 决定留到 scale-up,但**若 gate 不过则必须提前做**。
- **E3 LGS≈0 但 E1 过** → 语言进了梯度但泛化弱,查 target-id 头权重(`loss.target_id=0.2`)、课程 2c 是否到位。
- **兜底**:若数据一时改不动,按 §2.0 把 H0 主张**收窄为"外部指定 + 语言唯一表达力"**,而非"语言严格必要"——不丢核心卖点又难被打。

### 2.3 gate 与数据加难的耦合
E4 的难例只有 ~1.4 万帧、E5 的断裂帧更少 → 即使 gate 过,统计功效也弱。**建议**:gate 一旦确认 well-posed,立即并行启动 scale-up 数据生成(提共视率 + 造断裂事件 + 位置/连续性双去相关),别等主实验跑完。

---

## 3. 测量协议:离线 + 闭环两套(不能只报离线)

### 3.1 离线评测(便宜,跑所有消融)
- **在冻结缓存 + held-out 样本上**逐样本前向,算:act_mse、**minADE_K / minFDE@{2,4,6}s**(waypoint 轨迹误差,见 §3.4)、Mis-follow(target-id 头 argmax≠真值目标)、遮挡恢复相关代理量。
- 用途:H1–H4 全量消融、seed 复现、超参扫。
- **局限**:无复合误差、无策略自身分布 → **不能**单独支撑"闭环跟踪"主张。

### 3.2 闭环评测(贵,少量,兑现主张)——**当前缺口,必须补**
- **协议**:在 CARLA 中用训练好的策略做 **on-policy rollout**(S1 20Hz + S2 ~1Hz 双系统闭环),UAV 运动学模型驱动,跑固定 N 集(建议 test_town=Town05 上 ≥30 rollout/配置,×3 seed)。
- **指标**(§7.2 全套):Success Rate、Mis-follow(闭环下跟错车)、Occlusion Recovery Rate、Avg Tracking Frames、Search-Mode Efficiency、Cross-town SR drop。
- **关键**:闭环里 mis_follow 才是真消歧证据(离线是逐帧分类,闭环是"整段有没有跟丢/跟错")。
- **工程前置**:需要一个 rollout harness(策略 ← CARLA 观测闭环)。`carla_uav_tracking` 有专家 PID 闭环,但**学生策略闭环 loop 是否存在需确认**——若无,这是 MVP→可发表的必建组件(见 §10 风险)。

### 3.3 staleness 一致性(训练=推理)
闭环时 S1 用的是最多 ~20 帧前的 S2 缓存。训练已有 `staleness_p=0.3 / noise=0.2`(设计原则5)。**评测也要在闭环下验证 staleness 鲁棒**:扫 staleness_p∈{0,0.3,0.6} 看闭环 SR 退化曲线,证明 staleness augmentation 有效。

### 3.4 指标族目录(跟踪类任务标准指标 + 本工作专属)

> 跟踪不是单一任务,指标按子任务分五族。本工作横跨"被动感知精度 + 主动闭环控制 + 轨迹预测",须**混搭**才能同时和两类基线(§5.4)可比。**命名一律采用领域标准**(如 ADE/FDE 而非 waypoint MSE),提升可比性与审稿可信度。

**族 A｜被动 SOT 精度**(对标 ORTrack/TCMLTrack/JointNLT/CiteTracker;OTB/LaSOT/TNL2K 通用)
- **Success / AUC**:预测框-GT 的 IoU≥阈值成功率对[0,1]积分的曲线下面积(SOT 头号指标)。
- **Precision@20px**:中心误差≤20px 的帧比例;**Normalized Precision**:按框尺寸归一化(跨尺度更公平)。
- **OP50 / OP75**:IoU≥0.5/0.75 的重叠精度。(若用 VOT 协议才报 EAO/Accuracy/Robustness。)

**族 B｜身份保持 MOT**(把 Mis-follow 学术化;不必全套)
- **IDF1**:身份 F1,"有没有一直跟对同一个";**ID switches**:身份跳变次数。
- (可选)**HOTA**(DetA×AssA 平衡,MOT 金标准)、MOTA/MOTP。
- → **本工作**:Mis-follow 等价于"跟错干扰车"的 identity-switch,报告时并列该等价表述。

**族 C｜轨迹/waypoint 预测**(EAR 头,Stage-1;nuScenes/Argoverse 标准)
- **ADE**(Average Displacement Error):全时域预测点与真值点平均 L2。
- **FDE**(Final Displacement Error):终点(如 +6s)L2。
- **minADE_K / minFDE_K**:flow-matching 多模态采样 → K 样本取最优(多模态轨迹标配)。
- **Miss Rate@dist**:终点误差>阈值(如 2m)比例。
- → **本工作 waypoint 一律用 minADE_K / minFDE@{2,4,6}s**;并列保留"EAR 归属正确率"(端点更近目标车而非干扰车)作语言-grounding 证据。

**族 D｜主动/具身闭环控制**(核心,对标 UAV-Track VLA / EVT-Bench;AD-VAT/TrackVLA 体系)
- **Success Rate (SR)**:episode 级跟到结束比例(§7.2 已有)。
- **Episode Length / Avg Tracking Frames**:丢目标前平均连续跟踪时长(已有)。
- **Following/Tracking Rate**:目标落在视野中心区/理想距离带的时间占比(建议补,比"在框内"严)。
- **Collision / Safety Rate**:碰撞或违反高度约束比例(UAV 必报,建议补)。
- **Occlusion Recovery Rate**、**Cross-town SR drop**(已有)。
- (Stage-3 RL)**Accumulated Reward**:逐帧跟踪奖励累积。

**族 E｜语言消歧专属**(H0,本工作区别于所有基线的地方,务必单列)
- **Mis-follow Rate / ID-correctness**(核心)、**反事实语言切换成功率**、**难例子集(≥2 相似车共视)Mis-follow**、**w/o-language Mis-follow 增幅**(=语言净贡献)。

**族 F｜效率/部署**(UAV 审稿人必问)
- **FPS/延迟**(S1 20Hz / S2 1Hz 分报)、参数量/显存、on-board 可行性。

**⚑ EVT-Bench 协议对齐**:TrackVLA++/CoMaTrack 报 **STT/DT/AT 三 split 上的 SR**(单目标 / 干扰 / 语言歧义)。→ **本工作把数据难度课程 2a/2b/2c 对齐成评测三档**(无干扰 / 相似干扰 / 反事实歧义语言),各报 SR + Mis-follow,即与其同构可比,又一箭双雕(课程即评测协议)。

**推荐报告集(按对标对象)**

| 对标 | 必报指标 |
|---|---|
| 被动基线(ORTrack/TCMLTrack/JointNLT) | Success(AUC)·Precision·Norm-Precision·FPS |
| 主动竞品(UAV-Track VLA/EVT-Bench) | **SR(STT/DT/AT 三档)**·Episode Length·Occlusion Recovery·Collision Rate |
| EAR 轨迹(Stage-1) | **minADE_K/minFDE@{2,4,6}s**·EAR 归属正确率 |
| 语言必要性(H0,独有贡献) | **Mis-follow(全集+难例)**·反事实切换率·w/o-lang 增幅 |
| 泛化 | Cross-town SR drop·跨车型/跨干扰数分层 SR |

---

## 4. 数据划分与 held-out 设计

| 划分 | MVP 现状 | 用途 | 备注 |
|---|---|---|---|
| train/val/test | ~47/? /6(`mvp_split.json`, val_frac 0.15) | 主训练/选点/终评 | test_town=**Town05** 留出 |
| **跨地图** held-out | 仅 Town05 一张留出(5 图各 ~10 集) | Cross-town SR drop(**粗**) | MVP 只能"粗"跨图;强泛化需 scale |
| **跨车型** held-out | 按 target_class 分层(car/moto/bike/scooter) | 泛化面 | 类别不均(car 75%),moto/bike 少 → 只报趋势 |
| **跨干扰配置** held-out | 按 num_distractors(4–8)分层 | 防"固定配置记忆" | MVP 可做分层报告 |

- **分层原则**:test 集按 `strategy × class × distance` 分层,避免 6 集偶然全落某档。
- **诚实声明**:MVP 的跨地图只能作**趋势性**证据;论文里 Cross-town SR drop 必须标"仅仿真内 + MVP 规模,强结论需 scale 到更多地图"。

---

## 5. 实验矩阵:基线 + 消融(主实验表)

### 5.1 对比方法(纵轴)

| # | 方法 | 角色 | 验证 | MVP 可行性 |
|---|---|---|---|---|
| M0 | **ACoT-UAV-Track (ours, 完整)** | 主模型 | — | ✅ |
| M1 | w/o-Language | 关键对照 | H0 | ✅ |
| M2 | 外观/运动-only(无语言从头训) | 关键对照 | H0 | ✅ |
| M3 | w/o-EAR | 消融 | H1 | ✅ |
| M4 | w/o-IAR | 消融 | H2 | ✅ |
| M5 | Language-CoT 头(ECoT 风格自建) | 范式对照 | H3 | ✅(需建头) |
| M6 | 含时变从句语言(baseline 语言) | 消融 | H4 | ✅(seed_exp 已在跑) |
| M7 | π0.5 微调(通用 VLA,无 UAV 特化) | 通用基线 | H5 | 🟡(需接管线) |
| M8 | **UAV-Track VLA**(空中·CARLA·主动,同任务竞品) | 头号竞品 | H5 | 🟡(预印本,代码/环境待评估) |
| M8b | **TrackVLA++ / CoMaTrack**(EVT-Bench DT/AT 消歧协议) | 消歧对照 | H0/H5 | 🟡(地面→空中需适配;主要借其**评测协议**) |
| M9(passive) | **ORTrack(CVPR'25)** / **TCMLTrack(SciRep'25)** / **UAVNLT(2024)** | 已发表被动基线 | H0/H5 | ✅(已发表·可复现;感知层对照) |
| M9b(lang) | **JointNLT(CVPR'23)** / **CiteTracker(ICCV'23)** | 语言消歧对照 | H0 | ✅(已发表·开源) |
| M10 | ACoT (Stage-3 RL) | 上限 | H6 | 🟡(可选,后置) |

> 详见 §5.4 基线来源与发表状态。**分工**:M8/M8b(预印本 VLA)是**同类主动跟踪的头号可比对象**,但代码/环境未定 → 可能只能"引用 + 定性切割"而非跑分;M9/M9b(已发表)是**可直接复现**的感知层/语言消歧对照,兜住 R7。

### 5.2 消融维度(横轴,复用 M0 训练脚手架)

除对比方法外,在 M0 上做以下**单因子消融**(每次只动一个):
- **课程**:2a→2c 全课程 vs 无课程(直接混训) → 验证原则3"不上课程学捷径"。
- **staleness**:p∈{0, 0.3, 0.6} → 验证原则5。
- **target-id 损失权重**:`loss.target_id`∈{0, 0.1, 0.2, 0.4} → 验证语言进梯度的支点强度。
- **课程稳定性修复(C)**:a_floor{0.15,0.25} × ramp{0.9,1.2} × grad_clip{off,1.0} → 坐实 9.3 后段崩溃的根因与修复(已部分在 config_ac 中)。

### 5.3 主实验表(填充目标)

| 方法 | act_mse↓ | minFDE@6s↓ | Mis-follow↓(全集) | Mis-follow↓(难例) | 反事实切换↑ | SR↑(STT/DT/AT) | 遮挡恢复↑ |
|---|---|---|---|---|---|---|---|
| M0 完整 | (base 参考 0.034) | ? | (~1.75%) | ? | ? | ? / ? / ? | ? |
| M1 w/o-Lang | — | — | **应显著↑** | **应大幅↑** | — | ↓(DT/AT 尤甚) | — |
| M3 w/o-EAR | | **应↑** | | | | | **应↓** |
| M4 w/o-IAR | | | | | | | 机动帧应↓ |
| M5 Lang-CoT | | **应↑** | | | | | |
| M6 时变语言 | 0.049 | | 3.5% | | | | |
| ... | | | | | | | |

> 每格 = 3 seed 的 mean±std;带 bootstrap CI;与 M0 做配对显著性(§8)。指标定义见 §3.4;**SR 按 STT/DT/AT 三档报**(=课程 2a/2b/2c,对齐 EVT-Bench)。

### 5.4 外部基线来源与发表状态(deep-research 2026-07-29 核实,3 票对抗验证)

> **关键格局发现**:**没有任何一个主动 VLA 跟踪工作(含头号竞品 UAV-Track VLA 自己)实现显式"视觉相似干扰车消歧"机制** → 直接支撑 §6.3 收窄新颖性。最接近的现有消歧评测是 EVT-Bench 的 **DT(distracted)/AT(ambiguity)** split → **建议把该协议移植到空中·车辆场景,作为本工作 benchmark 贡献**(回答"消歧怎么评")。

**第 1 类:主动/具身语言引导 VLA 跟踪(全为 arXiv 预印本 → 引用/适配,非可直接复现)**

| 方法 | arXiv | 域 | 关键指标 | 作基线的定位/限制 |
|---|---|---|---|---|
| **UAV-Track VLA** | 2604.02241 (2026.04, v2) | 空中/CARLA | 25-step 连续飞控;自比 π0/π0.5/WALL-OSS/ACT | **头号对口竞品**;无消歧机制 → 差异点=干扰消歧;须查代码可复现性 |
| **TrackVLA++** | 2510.07134 (2025.10) | 地面/室内 | EVT-Bench DT +5.1%(ego)/+12%(multi-cam);Polar-CoT+TIM | 消歧对照最相关;借其 **DT/AT 协议** |
| **CoMaTrack** | 2603.22846 (2026.03) | Habitat 室内 | EVT-Bench STT 92.1/DT 74.2/AT 57.5 | DT/AT 直接映射消歧;DT 仅微超 TrackVLA++(74.0→74.2) |
| **TrackVLA** | 2505.23189 (2025.05) | 地面/室内 | 10FPS 闭环,anchor-diffusion | ⚠️**发表状态存疑**(疑已被会议接收,验证器反驳"仍为预印本")→ **需人工核实**,若已发表可升为已发表基线 |
| **CognitiveDrone-R1** | 2503.01378 (2025.03) | 空中/UAV | CognitiveDroneBench base 59.6%→R1 77.2% | UAV-native 但非跟踪-消歧任务 → 作范式讨论 |

**第 2 类:被动 UAV / 语言引导视觉跟踪(均已同行评审 → 可直接复现/引用,兜 R7)**

| 方法 | venue | 任务 | 关键指标 | 开源 |
|---|---|---|---|---|
| **ORTrack** | **CVPR 2025** | UAV 航拍 SOT·抗遮挡 | 六 UAV 集实时 SOTA;ORTrack-D 蒸馏 | ✅ github.com/wuyou3474/ORTrack |
| **TCMLTrack** | **Sci. Reports 2025** | **UAV 语言引导**跟踪 | 六 UAV 集 acc 0.819/succ 0.654/61FPS;**自承认消歧失败** | — |
| **UAVNLT** | **Electronics(MDPI) 2024** | **UAV 语言引导车辆**跟踪 | 2000 序列/四城市/城市道路车辆 | ✅ 代码;⚠️数据集 coming soon |
| **JointNLT** | **CVPR 2023** | grounding+track 统一 | TNL2K/LaSOT/OTB99 | ✅ github.com/lizhou-cs/JointNLT |
| **CiteTracker** | **ICCV 2023** | 图文关联视觉跟踪 | — | ✅ github.com/NorahGreen/CiteTracker |
| **WebUAV-3M** | **IEEE TPAMI 2023** | 百万级 UAV SOT 基准(bbox+NL+audio) | 4500 视频/3.3M 帧/223 类 | ✅ |
| **TNL2K** | **CVPR 2021** | 语言消歧跟踪基准 | 2000 序列/1.24M 帧 | ✅ |
| 补充 | — | UAV123 榜首/UAV SOT SOTA | **LoRAT-g-378**、**CGTrack** | 供 passive 表填充 |

**与设计文档对账**:UAV-Track VLA(2604.02241)、TrackVLA++(2510.07134)号一致;⚠️ **DeTrack(2605.17451) / AerialMind(2511.21053) 本轮未独立验到 → 需单独核号**;CosFly-VLA 确认虚构、正确未出现。

---

## 6. 分阶段执行计划(接 training-plan §1,补实验门)

### Stage-0 前置门:骨干与 gate(**部分已推进,需收口**)
- **B1 骨干探针**:design §3.4 主张 PaliGemma-2-3B layer-16,但**实际训练用的是 Qwen3-VL-4B layer 24**(config_ac/ablA 均如此)。→ **必须收口二选一**:
  - 要么补跑 layer-16 探针,若 Qwen 在 24 层才 PASS 而 PaliGemma 在 16 层 PASS,则论文 §3.4 的"截断契合"论证要改成 layer-24 版本(否则设计文档与实验不自洽,审稿人会抓)。
  - **建议**:以实跑的 Qwen3-VL-4B layer 24 为准,回改 §3.4 论证;PaliGemma 作为"更省算力的备选"消融一次即可。
- **H0 gate(§2)**:先于一切主实验。

### Stage-1 EAR 预热
- 门槛:**FDE@6s < 10m**(即 config `gate_mse_6s_m: 10.0`,其语义=终点位移误差 FDE@6s)+ **EAR 预测目标车而非干扰车**(early 语言-grounding 检查:在难例子集上比对 EAR 端点与目标 vs 干扰车真值轨迹)。
- **实验产出**:**minADE_K / minFDE@{2,4,6}s** 曲线 + "EAR 归属正确率"(端点更近目标车的比例)。

### Stage-2 端到端 + 课程
- **正在跑**:6-run seed(baseline×3 + A+C×3),坐实 H4 的 2× + 修 9.3 崩溃。
- **补齐**:M1–M5 消融、§5.2 单因子消融。
- **门槛**:held-out Mis-follow 低 **且 w/o-Language 时显著上升**(=H0 在此阶段再验一次)。
- **稳定性纪律**(9.3 教训):best.pt 按 act_mse 选;所有消融共用同一选点判据;报告后段崩溃是否随 C 修复消失。

### Stage-3 RL(可选,后置)
- GRPO,CARLA on-policy;reward 含 **R_correct_id**(防崩成"跟任意车")。
- 门槛:闭环 SR 与 Mis-follow **均优于** Stage-2 BC,且**无 reward hacking**(专门查"恒跟最近车"退化)。
- 仅在闭环 harness 就绪 + MVP 主实验站稳后启动。

---

## 7. 闭环 CARLA 评测协议(兑现"具身"主张的核心,当前最大缺口)

```
配置: test_town=Town05, ≥30 rollout/方法, ×3 seed, 每 rollout ≤180s
观测闭环: RGB(20Hz) → S1(DiT, 取前 2 步执行) ; S2(EAR+IAR+VLM, ~1Hz 条件激活)
干扰: 复用数据生成的 look-alike 场景(同 spawn 逻辑, 不同 seed 保证未见)
记录: 每帧 目标可见/跟对哪辆车/search_mode/UAV-目标距离 → 算全套 §7.2 指标
```

- **必测配置**:M0 / M1(w/o-Lang) / M3(w/o-EAR) / M4(w/o-IAR) / M7 或 M8(外部基线)。消融的**离线结论必须在闭环上复现方向一致**,否则以闭环为准。
- **抗 reward-hacking / 捷径检查**:统计"目标是否恒居画面中心"和"是否总跟最近车",若高 → 捷径未除,回 §2.3 加难。
- **专家上限说明**(§4.2/数据卡§6.6):BC 上限 ≈ P-only PID;若闭环 SR 追平专家即达 BC 天花板,超越需 Stage-3。论文把价值定位在"语言消歧 + 遮挡恢复",而非纯跟踪精度。

---

## 8. 统计分析计划(小测试集必须做)

1. **重复**:每配置 **3 seed**(seed_exp 已是此结构);报 mean±std。
2. **不确定度**:测试仅 6 集 → 用 **per-episode bootstrap**(重采样 episode)给 95% CI;闭环用 per-rollout bootstrap。
3. **配对比较**:M0 vs Mk 在**相同 episode/rollout** 上配对 → 配对 t 或 Wilcoxon,报效应量(Δ + CI),而非只看均值差。
4. **多重比较**:一次报多个消融 → Holm-Bonferroni 校正,避免 6 集上假阳性。
5. **判据前置**:H0 的"2× 且 CI 不重叠"、Stage-1 的"<10m@6s" 等**在跑之前写死**,不事后挪。
6. **诚实标注**:凡 MVP 规模不足以显著的对比,明确写"趋势性,统计显著需 scale",不硬凑星号。

---

## 9. 风险登记与证伪条件(实验设计的免疫系统)

| 风险 | 触发信号 | 缓解 / 分叉 |
|---|---|---|
| **R1 任务太易/语言非必要(致命)** | E1/E2/E5 语言消融无差异;position-only ~3–5%(memory);难例仅 14% | gate 前置(§2/§2.0);**两条捷径都要堵**(位置+时空连续性);不过则先加难数据(去相关+造断裂事件),或按 §2.0 收窄 H0 主张,别硬推 |
| **R2 只有离线证据** | 无学生闭环 harness | §7 建 harness;闭环缺失则"具身闭环"主张站不住 |
| **R3 设计-实跑不自洽** | ①§3.4 写 PaliGemma-16,实跑 Qwen-24;②~~设计文档 §2.1/§3.1 首帧 bbox 输入~~ **✅已确认(2026-07-30):首帧 bbox 不提供目标身份信息 → 非身份锚,H0 判据(a)满足,此项解除** | ①§6 Stage-0 收口回改;②已澄清,建议在设计文档旁注"bbox 仅几何/非目标指定",免后人误读 |
| **R4 后段课程崩溃** | ~ep40 act_mse 0.05→1.0(9.3) | C 修复(a_floor/ramp/grad_clip);报修复前后对比 |
| **R5 遮挡证据弱(H2)** | occ_structural 仅 4.3% 帧 | H2 只作**初步**;强证据待 A1c③ scale 重生成 |
| **R6 跨地图不可信** | 5 图各 ~10 集 | Cross-town 只报趋势;标注 MVP 局限 |
| **R7 竞品复现不可得** | 第1类主动 VLA 全为预印本,UAV-Track VLA 代码/环境未定 | **双保险**(§5.4):主动竞品(M8/M8b)降级为"引用+定性切割",**可复现对照改用已发表被动/语言基线**(ORTrack CVPR'25 / JointNLT CVPR'23 / TCMLTrack SciRep'25)兜底 H0/H5;先核 TrackVLA 是否已发表、DeTrack/AerialMind 号是否正确 |
| **R8 专家上限封顶** | 闭环 SR 追平 PID 即停滞 | 价值定位到消歧/恢复;超越留 Stage-3 |

**全局证伪条件**:若 §2 gate 的 E1(w/o-Language)在难例子集上、**尤其是 E5 连续性断裂再捕获帧上** Mis-follow **不显著上升**(LGS≈0),则"语言 load-bearing"不成立、核心贡献 §6.3 落空 —— 此时应停下改数据(位置+连续性双去相关 + 造断裂事件)或按 §2.0 收窄主张,而非继续跑消融。**注意**:观察到"去语言掉点"是**必要非充分**条件——数据必须先排除位置捷径与时空连续性捷径,否则掉点可能来自别的信号消失(CAST/Shortcut-Learning 教训)。

---

## 10. MVP 可交付 vs 需 Scale-up(诚实边界)

**MVP(现有 100K 帧)能给**:
- ✅ 骨干选型结论(收口 Qwen-24)
- ✅ H0 语言必要性**初步**证据(E1–E4)
- ✅ H4 中性语言 2× 提升(3-seed 坐实)
- ✅ H1/H2/H3 消融的**方向性**结论 + EAR 收敛
- ✅ 少量闭环 rollout 演示(打通 harness 后)

**必须 scale 到 ~300K + 更多地图才能给**:
- ❌ 统计显著的完整消融(6 集测试功效不足)
- ❌ 强跨地图泛化(Cross-town SR drop 的可信数值)
- ❌ H2 遮挡先验的强证据(需 A1c③ 更多框内遮挡场景)
- ❌ 与 UAV-Track VLA(892K)同级的 SOTA 规模对比

**衔接动作**:H0 gate 一过,**立即并行启动 scale-up 数据生成**(优先提 look-alike 共视率 + 框内遮挡场景),不等 MVP 主实验跑完。

---

## 11. 执行顺序(建议排程)

```
第 0 步(进行中): seed_exp 6-run 完成 → 坐实 H4(2×)+ 验 C 修复(R4)         [离线, 已在跑]
第 1 步(最高优先): H0 gate 四联证据 E1–E4                                   [离线, ~1–2 天]
   ├─ 过 → 继续;  不过 → 停, 转数据加难(A1c③)
第 2 步: 骨干收口(§6 Stage-0)—— 认定 Qwen-24, 回改 §3.4 论证              [~0.5 天]
第 3 步: 主消融 M1–M5 + §5.2 单因子, 3-seed, 离线                          [离线, 批量]
第 4 步: 建学生闭环 rollout harness(R2 缺口)                              [工程]
第 5 步: 闭环评测 M0/M1/M3/M4/外部基线 + staleness 曲线 + 捷径检查          [闭环, 少量]
第 6 步: 统计汇总(§8)+ 主实验表填充 + MVP 结论定稿
第 7 步(并行/后置): scale-up 数据生成; (可选)Stage-3 RL                    [长线]
```

---

## 附:实验-代码索引

| 实验 | 主要代码/配置 | 备注 |
|---|---|---|
| H4 seed 对比 | `train/seed_exp.sh` + `config.yaml`/`config_ac.yaml` | baseline×3 vs A+C×3 |
| C 课程修复 | `train/curriculum.py`(a_floor/ramp) + `config_ac.yaml` | grad_clip 1.0 |
| H0 语言消融 | `train/backbone_kv.py::neutralize_language`(扩到"完全空语言"档) | 需重建一份缓存 |
| target-id | `train/stage2.py` L_target_id + `loss.target_id` | 语言进梯度支点 |
| Stage-1 门 | `train/stage1_ear.py` / `eval_stage1.py` / `gate_mse_6s_m` | FDE@6s<10m + 归属检查(指标见 §3.4 族C) |
| 难例切片 | `dataset-card.md` §5 co-visibility + `distractors` | ~14% 帧 |
| 闭环 harness | **待建**(参考 `carla_uav_tracking` 专家闭环) | R2 缺口 |
| 骨干探针 | `acot_probe/`(run_probe/projection) | §6 收口用 |
</content>
</invoke>
