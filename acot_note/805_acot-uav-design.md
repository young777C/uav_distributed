# ACoT-UAV-Track：动作思维链驱动的 UAV 具身跟踪 VLA 系统

> 完整技术设计文档 | 2026-07-24（2026-07-26 修订：修复引用与研究定位，依据 `positioning-analysis.md`）

---

## 一、研究背景与动机

### 1.1 从 D-EPA-RHP 到 VLA 的范式转变
****
原研究方向 D-EPA-RHP（GCS-UAV 双环分布式协同，通信退化巡检）存在三个结构性天花板：

| 天花板 | 根因 | 文献证据 |
|---|---|---|
| 离散动作空间上限低 | `A_U = {INS, TX, REC, SAFE, RET}` 5 个离散动作无法表达连续折中 | Ortner & Ryabko (2012): 离散化 regret 下界 Ω(T²/³) vs 连续 O(√T) |
| 手工概率公式表达力弱 | 8 标量几何平均无法捕捉通信-运动-能量的非线性交互 | 实验: C3-High 退化下概率估计完全失效 |
| VoI 硬阈值反馈脆弱 | 噪声驱动虚假触发 (Zeno behavior) | Scheres et al. (2023), D-EPA-RHP: C3 下反馈是净负贡献 |

**核心洞察**：GCS 对未来状态的预测能力被锁定在手写公式里。VLA 转向的本质是用神经网络在潜空间中学习这种预测，替代手工近似。

### 1.2 为什么选择 ACoT + UAV 跟踪

ACoT-VLA (Zhong et al., 2026) 的核心洞察："在动作空间中推理，而不是用语言描述动作。" 这一哲学在 UAV 跟踪中比桌面操作更适用：

- 跟踪天然是双层 hierarchy：慢层（预测目标去哪）→ 快层（保持目标在视野中心）
- EAR 的粗粒度 waypoint = 预测的 3D 拦截点，与 UAV 动作空间同构
- IAR 的隐式先验可以提取通信质量、遮挡风险、目标机动等"说不出来"的信息

**范式定位：predictive / world-model-augmented VLA（诚实落点，deep-research 2026-07-31）**

范式光谱上有三档确立的边界（NVIDIA WAM glossary + 3 篇 2026 survey 一致）：**纯 VLA**=反应式 obs→action、不建模世界动力学（π0/OpenVLA）；**World Model**=预测未来观测/状态但本身非策略（Dreamer/JEPA/Genie）；**World-Action Model(WAM)/world-model-augmented VLA**=**联合**预测未来世界状态 + 动作。

> **本工作 = 光谱轻量端的 world-model-augmented VLA**：骨干是标准 VLA（冻结 VLM + flow-matching 动作头，谱系承 ACoT-VLA/π0.5），但装了一个**目标中心的预测核 EAR**（显式预测目标未来 3D 轨迹/拦截点）。**"world-model-augmented VLA" 是已发表的确切术语**（DUST 2510.27607；同类 WorldVLA 2506.21539「action world model」、DriveWorld-VLA ICML 2026、UWM RSS 2025；survey 2605.00080 §3.5 列为公认活跃子线）。

**★ 核心洞察（可作卖点，但诚实定位）——跟踪任务的状态-动作同构**：一般任务里"世界状态预测（物体去哪）"与"动作（我该怎么动）"隔着语义-运动鸿沟；但在 UAV 拦截跟踪里，**我关心的世界状态 = 目标 3D 位置，我的动作目标 = 飞到该 3D 点，两者近乎同构**。故 EAR 的 3D 拦截点**同时是**世界状态预测**和**动作计划——ACoT"在动作空间推理"的本质，是利用该同构**把世界预测折叠进动作空间**，从而以 VLA 的成本吃到 world-model 式的预测-拦截红利。

**两条诚实边界（避免过度声称，写进 limitation/positioning）**：
1. **不是完整 WAM**：WAM 教科书判据是预测未来**观测/视频**（world dynamics）；EAR 只预测**目标状态**、无观测预测/imagined rollout → 只**部分**满足，故措辞用 "world-model-augmented"，**不自称 WAM/world model**。
2. **融合本身不新 + 同构洞察非全新原理**：world-model-augmented VLA 已有 WorldVLA/DUST；状态-动作同构未见直接先例（可作任务特定新 framing），但**呼应 goal-conditioned RL(state=goal) 与 pursuit-as-prediction** → novelty 押在**域（空中具身跟踪）+ 同构洞察的具体兑现（EAR 预测拦截）**，不 claim "首个融合"。（详见 `related-work-and-positioning.md` §4C。）

### 1.3 对标研究与定位

> 定位依据见 `positioning-analysis.md`（2026-07 领域扫描）。本工作是一篇 **新任务 + 新 benchmark + 方法迁移** 的贡献：核心动作思维链机制迁移自 ACoT-VLA，真正的新颖性在于一个此前无人占据的任务交集——**空中主动飞行 + 语言指定目标车辆 + 多辆视觉相似干扰车辆消歧 + 具身闭环跟踪**。

**头号竞品（必须精确切割）**：

| 方法 | arXiv | 做了什么 | 本工作 = + 什么 |
|---|---|---|---|
| **UAV-Track VLA** | 2604.02241 (2026.04) | 空中主动语言跟踪 VLA：π₀.₅/PaliGemma-3B + 时序压缩 + **显式 3D 定位辅助头** + flow-matching；~890K CARLA 帧；行人/车辆目标 | **+ 干扰物消歧**（它不强调相似干扰物）、**+ 语言 load-bearing 证据** |
| **TrackVLA++** | 2510.07134 (2025.10) | 地面四足：**Polar-CoT**（动作空间 CoT）+ **Target Identification Memory**，抗遮挡/抗相似干扰物 | **+ 空中 6-DoF、+ 车辆目标、+ 语言指定**（其为地面/人） |
| **DeTrack**（✅号已核） | 2605.17451 (2026.05, 预印本) | *A Benchmark and Altitude-Aware Dual World Model for Drone-embodied Tracking*：无人机**主动闭环** + **动态遮挡物**(车/人,非相似车) + 避障；11,368 轨迹；**纯视觉 world-model(AaDWorlds)，无语言** | **+ 语言消歧 + world-model-augmented VLA**（它撞"具身跟踪/world-model"两词但**无语言、无相似车消歧**，须切清；⚠️勿混 NeurIPS'24 被动 SOT 同名 DeTrack 2501.02467） |
| **AerialMind**（✅号已核,**升级**） | 2511.21053 → **AAAI-26 已发表**(Proc.AAAI 40(4):2805-2813) | *Towards Referring Multi-Object Tracking in UAV Scenarios*：首个 UAV **RMOT** benchmark；语言指定 referring 消歧(扩自 VisDrone/UAVDT,方法 HawkEyeTrack)；**被动感知，不控制无人机** | **+ 主动飞行闭环控制**（peer-reviewed 强竞品,与 UAVNLT 是**两篇不同工作**） |

**推理范式对标（本工作所在的"动作空间 CoT"分支）**：

| 推理范式 | 代表 | 中间"思维"形式 | 局限 |
|---|---|---|---|
| 语言 CoT | ECoT (2407.08693), CognitiveDrone-R1 (2503.01378) | 自然语言 plan | 语义-运动鸿沟：文字→3D 坐标需额外解码 |
| 隐空间 CoT | LaRA-VLA, HyT (2510.00600) | 抽象 latent | 可解释性弱 |
| **动作空间 CoT (ours，迁移自 ACoT-VLA 2601.11404)** | ACoT-VLA; Coarse-to-Control (2606.07107) | **动作空间粗轨迹（3D 拦截点）** | — |

> ⚠️ 早期草稿曾引用一个名为 "CosFly-VLA" 的"语言 CoT"基线——经核验该模型 **不存在**（真实工作为 CosFly-Track 数据集 2605.17776 与 CosFly CARLA 管线 2605.19120，均非 VLA）。已移除；"语言 CoT" 对标改用真实工作 ECoT / CognitiveDrone-R1。

---

## 二、问题定义

### 2.1 UAV 具身跟踪的形式化

```
给定:
  - 观测: 连续 RGB 帧序列 (I₁, ..., I_t)，336×336 @ 10Hz
  - 语言指令: "Track the {target_desc} among {N} other vehicles."
  - 本体感知: UAV 位置/速度/偏航/能量（从仿真器读取）
  - 目标指定: **仅由语言指令指定**（无首帧 bbox 身份锚）
    ↳ 这是 H0 语言 load-bearing 的前提：给持续 bbox 锚 → 可脱离语言跟踪 → 语言冗余
      （见 experiment-design §2.0 条件(a) / §9 R3）。目标身份由 target-id 头在候选中靠语言选出。

目标:
  在复杂城市环境中，从多辆行驶车辆中识别并持续跟踪指定目标，
  处理遮挡、机动和环境变化。

输出:
  每步控制动作: [Δx, Δy, Δz, Δyaw, search_mode]
  - search_mode ∈ [0,1]: 0=精确跟踪, 1=全区域搜索

约束:
  - UAV 保持在建筑之上（≥ 22m 高度）
  - 实时控制频率 ≥ 10 Hz
  - 跟踪成功率: 目标在视野中 > 80% 时间
```

### 2.2 多车辆语言指定跟踪

这是本文区别于现有工作的关键场景设计：

- 场景中包含 5-9 辆不同颜色/类型的车辆（1 目标 + 4-8 干扰车）
- 语言指令唯一指定目标：`"Track the orange Jeep Wrangler. Ignore the other vehicles."`
- VLA 必须同时完成：(1) 语言理解 → 识别目标，(2) 视觉辨识 → 在多辆车中定位目标，(3) 持续跟踪 → 不被干扰车分散

### 2.3 出画预测-拦截重捕获（aerial 差异化场景）

目标拐弯/遮挡/出画**短暂丢失**时,系统应由 EAR **预测目标未来 3D 位置并直飞拦截点抄近路重捕获**,而非结束 episode 或盲跟。无人机**无路网约束**可抄近路拦截,是地面跟踪做不到的差异化能力。长丢失(视觉连续性彻底断裂)时,重捕获**只能靠语言 + 预测** → 同时强化语言必要性(H0)与 EAR 价值(H1)。详见 §6.1 招牌场景、`data-fix-spec.md` §3。

---

## 三、模型架构

### 3.1 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    ACoT-UAV-Track                                 │
│                                                                  │
│  输入                                                             │
│  ├── 视觉: RGB 帧 (336×336) ── 无 target bbox 叠加（语言-only 指定）│
│  ├── 语言: "Track the orange Jeep Wrangler among 6 vehicles"     │
│  └── 本体: [px,py,pz, vx,vy,vz, yaw, energy]                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ VLM 骨干 (frozen, 截断到 layer 16)                          │  │
│  │ · 选型: PaliGemma-2-3B（主，详见 §3.4）                         │  │
│  │ · 输出: KV Cache → EAR + IAR 共享                            │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                      │
│         ┌─────────────────┴─────────────────┐                    │
│         ▼                                   ▼                    │
│  ┌──────────────┐                  ┌──────────────┐              │
│  │ EAR (显式)    │                  │ IAR (隐式)    │              │
│  │ 拦截点预测    │                  │ 跟踪先验提取  │              │
│  │              │                  │              │              │
│  │ 4 层 Transformer              │ 逐层 Query 投影│              │
│  │ d_model=256  │                  │ KV 下采样(64d)│              │
│  │ 4 attention  │                  │ Cross-Attn    │              │
│  │   heads      │                  │ ← VLM 各层 KV │              │
│  │              │                  │ Pooling       │              │
│  │ Self-Attn:   │                  │ → MLP(256d)   │              │
│  │  waypoint 间 │                  │              │              │
│  │  时序依赖    │                  │ → Zⁱᵐ (256d)  │              │
│  │              │                  │              │              │
│  │ Cross-Attn:  │                  │ 辅助 Loss:    │              │
│  │  ← VLM KV    │                  │ · 遮挡预测    │              │
│  │              │                  │ · 机动检测    │              │
│  │ Flow Matching│                  │              │              │
│  │ 噪声→K=3~5   │                  │              │              │
│  │ 3D waypoints │                  │              │              │
│  │ 每点:(x,y,z, │                  │              │              │
│  │   confidence)│                  │              │              │
│  │              │                  │              │              │
│  │ → Zᵉˣ (K×4)  │                  │              │              │
│  └──────┬───────┘                  └──────┬───────┘              │
│         │                                 │                      │
│         └─────────────┬───────────────────┘                      │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ AGP 融合 + DiT (32 层, d_model=512, 8 heads)               │  │
│  │                                                            │  │
│  │ Triple Cross-Attention (交替注入):                           │  │
│  │   ┌─ Q_action ↔ Zᵉˣ    — 粗 waypoint 引导 (每 2 层)        │  │
│  │   ├─ Q_action ↔ Zⁱᵐ    — 隐式先验注入 (每 2 层)            │  │
│  │   └─ Q_action ↔ VLM KV — 全局语义上下文 (每 4 层)           │  │
│  │                                                            │  │
│  │ + MLP: 本体感知 8d → action_embedding 512d                  │  │
│  │                                                            │  │
│  │ Flow Matching (4 步去噪)                                    │  │
│  │ → 16 步动作块 [Δx, Δy, Δz, Δyaw, search_mode]              │  │
│  │                                                            │  │
│  │ search_mode: 连续维度 [0,1]                                 │  │
│  │   0.0 = 精确跟踪（目标锁定）                                  │  │
│  │   0.3 = 轻微不确定性（扩大视野）                              │  │
│  │   0.7 = 预测轨迹搜索（按 EAR waypoint 飞）                   │  │
│  │   1.0 = 全区域扫描（目标丢失 > 5s）                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Total params: ~150M (可训练) + ~3B (frozen VLM)                 │
│                                                                  │
│  System 1 (DiT): 每控制周期运行, ~50ms, 20Hz                     │
│  System 2 (EAR+VLM): 条件激活, ~300ms, ~1Hz                      │
│    激活条件: waypoint 到达 / search_mode > 0.5 持续 3s /          │
│             目标丢失 > 2s / 安全事件                              │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 各模块参数表

| 模块 | 参数量 | 说明 |
|---|---|---|
| VLM 骨干 | ~3B (frozen) | **PaliGemma-2-3B** (SigLIP+Gemma-2B, 18层), 截断 layer 16；选型见 §3.4 |
| EAR | ~5M | 4 层 Transformer, d=256, 4 heads |
| IAR | ~3M | 逐层 query (4×256) + KV 下采样 + pooling |
| AGP + DiT | ~140M | 32 层 DiT, d=512, 8 heads |
| MLP (proprio) | ~1M | 8→512 |
| **总计可训练** | **~150M** | EAR + IAR + DiT + MLP |

### 3.3 推理管线

```
帧率: 20Hz (System 1 每控制周期运行)

每帧:
  1. 采集 RGB 帧 + 本体感知
  2. System 1 (DiT): 输入缓存的 Zᵉˣ + Zⁱᵐ + VLM KV + proprio
     → Flow Matching 4 步 → 16 步动作块
     → 取前 2 步执行 → 闭坏控制

System 2 激活时 (平均 ~1Hz):
  1. VLM 前向: 最近 5 帧 RGB + 文本指令 → KV Cache
  2. EAR: Flow Matching 去噪 → K 个 3D waypoint
  3. IAR: 逐层提取隐式先验 → Zⁱᵐ
  4. 更新缓存: Zᵉˣ, Zⁱᵐ, VLM KV Cache
```

### 3.4 VLM 骨干选型与验证

**主选 PaliGemma-2-3B（SigLIP + Gemma-2B，18 层 decoder）。** 决定性约束不是榜单分，而是本架构"**冻结 + 截断到 layer 16 + 读 per-layer KV**"：

| 模型 | decoder 层数 | layer 16 处 | 契合 |
|---|---|---|---|
| **PaliGemma-2-3B** | **18** | ~89% 深度 | ✅ grounding 已成形（靠重型 400M SigLIP 编码器 + 前缀双向注意力，非深层自回归） |
| Qwen2.5-VL-3B / Qwen3-VL-4B / InternVL3.5-4B | ~36 | ~44% 深度 | ⚠️ RefCOCO 高分是**末层**读出；16 层处细粒度定位未凝聚，分数不迁移到 layer-16 KV |

- **截断契合**：截断到 16 只对浅 decoder 有意义；深 decoder 在半程被掐断，享受不到其 grounding 优势还背算力。
- **谱系一致**：基座 ACoT-VLA（明确 18 层 SigLIP+Gemma-2B，建于 π0.5）与头号竞品 UAV-Track VLA 同栈 → **复现竞品基线最省事、可比性最强**。
- **原生 grounding**：`<locXXXX>` 检测 / `<segXXX>` 分割 token 开箱做 referring；用 `mix` 权重 + 448/896 分辨率应对远处小车。

**备选**（grounding 精度优先、且愿放宽截断到 ~20–24 层）：**Qwen3-VL-4B**（Apache-2.0，原生 2D+3D grounding）；InternVL3.5-4B（Apache-2.0）为同类平衡项。

**⚠️ 剔除原方案**：
- ~~Qwen2.5-VL-3B~~：① layer-16 丢大半 grounding；② **3B 版为非商用 Research License**（7B 才 Apache）——部署受限。仅留作研究基线。
- ~~SmolVLM-2B~~：**非 grounding 模型**（无原生 REC/bbox），与语言消歧核心任务不符。

**⚑ 选型验证 — layer-16 grounding 探针（训练前必做，~30 分钟）**：
```
对每个候选骨干:
  1. 冻结, 前向到 layer 16, 取该层特征 / KV
  2. 训一个轻量线性探针 / 单层注意力读出头, 只做一件事:
     从 layer-16 表征预测"被指目标车 bbox"或"目标 vs 干扰车"二分类
  3. 判据: 探针能从 layer-16 直接读出目标 → grounding 已就位 → 该骨干可用
     (PaliGemma 大概率过; Qwen 在 16 层大概率不过 → 需放宽截断到 ~20-24 层)
  副产物: 量化冻结骨干 grounding 上限, 作为 Stage 0 的前置门 (见 §4.3)
```

---

## 四、训练策略

### 4.1 数据需求

| 目标 | 数据量 | 存储 | 说明 |
|---|---|---|---|
| 概念验证 (MVP) | ~100K 帧 (55 集 × 3min) | ~15 GB | EAR 收敛，跟踪可演示 |
| 可发表 | ~300K 帧 (165 集 × 3min) | ~45 GB | 消融显著，对比有统计显著性 |
| 对标 SOTA | ~900K 帧 (500 集 × 3min) | ~140 GB | 与 UAV-Track VLA (892K) 同级 |

**当前目标**: ≥ 350K 帧 (~45 GB)，D 盘 65 GB 可用

**可选预训练数据（暖启动，非自建）**：grounding ← AerialMind + WebUAV-3M（空中语言定位）；飞行动作分布 ← CosFly（`AutelRobotics/CosFly`, apache-2.0, 行人域）；消歧任务设计借鉴 ← EVT-Bench。均只作初始化，域差大（见 `positioning-analysis.md`）。

### 4.2 训练总原则（由架构三约束决定）

1. **VLM 全程冻结 + 截断 layer 16 → 预计算并缓存 KV/特征**。全数据集 VLM 前向只跑一次存盘，训练循环不再前向 VLM（ACoT-VLA 证冻结不掉点；可训练部分仅 ~150M，缓存后单卡 batch 可开大）。
2. **专家是"特权信息蒸馏"，非"模仿 PID"**。PID 专家用了目标真值位置才能跟；学生只有 RGB+语言，天花板是"能否从语言+视觉推断正确目标"而非 PID 控制精度——正好把审稿人的"专家上限"质疑转成核心能力。
3. **难度课程 = 语言 load-bearing 的训练侧保证**（仿 EVT-Bench STT→DT→AT）：单目标 → 易区分干扰物 → 视觉相似干扰物。不上课程会先学会"跟最居中那辆"的捷径，语言学不进去。
4. **反事实语言编进训练**：同一场景只改指令指向 → 监督动作随之切换，逼模型真"读"语言。
5. **训练要模拟异步过期**：S1(20Hz) 推理时用的是最多 ~20 帧前的 S2 缓存；训练时若永远给新鲜 Z^ex/Z^im 会造成分布不匹配 → **staleness augmentation：随机复用过期 EAR/IAR 输出**。

### 4.3 分阶段训练管线（Stage 0–3）

> **前置门**：训练前先跑 §3.4 的 layer-16 grounding 探针，选定并冻结 VLM 骨干（PaliGemma-2-3B 主）。

```
═══════════════════════════════════════════════════════════════
阶段 0: 组件预训练 (可选, 数据紧时值得)
═══════════════════════════════════════════════════════════════
目标: 用可下载开源数据暖启动感知与动作分布
grounding: AerialMind + WebUAV-3M 预热 IAR/语言对齐 (空中·车辆·语言定位)
动作头:   CosFly (apache-2.0, 行人域) flow-matching 预热 EAR/DiT 飞行分布
边界:     域差大 (行人→车辆, 被动→主动), 仅作初始化; VLM 已带强先验, grounding 部分"免费"

═══════════════════════════════════════════════════════════════
阶段 1: EAR 预热 (50 epochs)
═══════════════════════════════════════════════════════════════
目标: EAR 学会从 VLM 特征 + 历史帧预测目标未来空间位置
冻结: VLM, IAR, DiT
训练: EAR
输入: 预计算缓存的 VLM KV (原则1) + 噪声 waypoint
监督: 从专家轨迹自动标注的未来目标位置
Loss: L_flow(W) = MSE(去噪后 waypoint, 真值 waypoint)
      + λ·时间一致性 loss (相邻 waypoint 速度连续)
数据: 全部 episode 的 waypoint 真值
门槛(gate): ① waypoint MSE < 10m @6s;
           ② EAR 预测的是"目标车"未来轨迹而非某辆干扰车 (early 语言-grounding 检查, 否则后续全白搭)

═══════════════════════════════════════════════════════════════
阶段 2: 端到端联合训练 (150 epochs)
═══════════════════════════════════════════════════════════════
目标: 联合训练 EAR + IAR + DiT，学会"语言消歧 + 从粗拦截点到精细跟踪动作"
冻结: VLM
训练: EAR, IAR, DiT, AGP, proprio-MLP
输入: 完整输入 (视觉 + 文本 + 本体); + staleness augmentation (原则5)
监督: 专家动作序列 (PID, 特权信息蒸馏, 原则2)  —— flow matching BC
Loss:
  L_total = L_action + 0.3·L_ear + 0.1·(L_visibility + L_maneuver)
          + 0.2·L_target_id     ← 【新增】辅助头: 分类"哪辆检测框是被指目标"
                                   使语言直接进入梯度, 是 load-bearing 的训练侧支点
  L_action: Flow Matching loss on action prediction
  L_ear: Flow Matching loss on waypoint prediction
  L_visibility: search_mode MSE (遮挡 → search_mode 增大)
  L_maneuver: 目标机动检测 BCE

课程 (必须按序, 每档过门再进下一档, 原则3):
  2a 单目标无干扰       → 学基础跟踪闭环
  2b 易区分干扰物(异色异型) → 学"不被带偏"
  2c 视觉相似干扰物 + 反事实语言 → 语言成为唯一区分信号
数据: ~350K 帧    优化: AdamW, lr=1e-4, batch=64, 4×A100
门槛: held-out 上 Mis-follow Rate 低, 且 w/o-Language 消融时 Mis-follow 显著上升
      (=语言有净贡献; 若不升 → 任务太易, 回 §5.1 加难, 别硬推下一阶段) [对齐 §7.4]

═══════════════════════════════════════════════════════════════
阶段 3: 闭环 RL 精调 (可选, 500 iterations)
═══════════════════════════════════════════════════════════════
目标: 治 BC 三病 (复合误差 / 专家次优遮挡恢复 / 无探索), 在策略自身分布上优化
冻结: VLM, EAR, IAR
训练: DiT 最后 4 层 (或 LoRA)
方法: GRPO (或 PPO), CARLA on-policy rollout
Reward:
  R_track: +1/帧 跟踪成功 (目标在画面中)
  R_correct_id: +/- 跟对/跟错被指目标   ← 【新增, 关键】否则 RL 会崩成
                "跟随任意车"来刷 R_track, 把语言消歧学没
  R_recover: +5 遮挡恢复 (重新锁定目标)
  R_search: -0.01 当 search_mode > 0.5 (惩罚长时间搜索)
  R_collision: -10 碰撞
  R_smooth: -0.001∥a_t - a_{t-1}∥
数据: CARLA 仿真器 rollout
门槛: SR 与 Mis-follow 均优于 Stage 2 BC; 检查无 reward hacking (如恒跟最近车)
```

### 4.4 验证门与数据划分（编进流程，不放最后）

- **三重 held-out**：留出 town（跨地图泛化 = §7.2 Cross-town SR drop）、留出车型、留出干扰物数量配置（4-8）——防"有限地图记忆"。
- **每阶段末跑缩减评测，任一门不过就地停**：Stage 1 查 waypoint 归属正确性；Stage 2 查 Mis-follow + w/o-language 消融 + 反事实语言切换成功率；Stage 3 查 SR/恢复率 + reward hacking。
- 全部与 §7.4「语言必要性验证」共用同一套指标与判据。

---

## 五、数据生成管线

### 5.1 场景设计

```
目标车辆 (1 辆):
  · 从 25 种民用车辆中随机选择 (CARLA CARS_ONLY)
  · CARLA autopilot 沿路网行驶 (含交通灯停车、路口转弯)
  · 速度: 5-15 m/s (autopilot 控制)

干扰车辆 (4-8 辆):
  · 关键设计: 其中 ≥2 辆与目标车"视觉高度相似"(同型号 / 近似颜色)——
    使"仅凭外观"无法唯一确定目标, 迫使模型依赖语言中的细粒度描述
    (颜色深浅 / 相对位置 / 朝向) 消歧  [让语言 load-bearing, 见 §7.4]
  · 同一条路 ±60m 范围内 spawn (确保在 UAV 视野内)
  · 同样使用 autopilot 行驶; 部分干扰车与目标并行 / 交叉行驶, 制造运动歧义
    (避免"跟随最居中 / 最近那辆"的捷径解)

环境 (宽路城市):
  · 地图: Town03, Town04, Town05 (6 车道主干道)
  · 天气: 白天, 无雾 (sun_altitude 15-70°, fog=0)
  · 时段: 昼 / 黄昏随机

UAV:
  · 起始位置: 目标后方 8-20m, 高度 22-32m, 侧偏 10-20m
  · 飞行: P-only PID visual servoing (高度 > 22m)
  · 相机: 单 RGB, 336×336, FOV 90°, pitch=-30°

APF 扰动 (产生恢复轨迹):
  · 3%/帧概率随机注入: 侧推(20-50m) / 落后(30-80m) / 过冲(20-40m) / 掉高(10-25m)
  · → 目标短暂丢失 → PID 恢复 → 训练数据含完整恢复序列
```

### 5.2 数据格式

```python
episode_{id}.h5:
  ├── rgb/              # (T, 336, 336, 3) uint8, gzip level 9
  ├── state/            # UAV 状态 (T × 10d)
  │   ├── t, uav_x, uav_y, uav_z, uav_vx, uav_vy, uav_vz
  │   ├── uav_yaw, uav_energy
  │   └── is_turning, perturbation, obstacle_avoid  # bool
  ├── target/           # 目标状态 (T × 8d)
  │   └── tx, ty, tz, tvx, tvy, tvz, tspeed, tyaw
  ├── action/           # 专家动作 (T × 4d)
  │   └── dx, dy, dz, dyaw
  ├── annotation/       # 自动标注
  │   ├── occlusion       # 遮挡等级 [0-1]
  │   ├── search_mode     # 搜索模式 [0-1]
  │   ├── waypoints        # EAR waypoint GT (T × 3 × 3)
  │   ├── waypoint_sigma   # waypoint 不确定性 (T × 3)
  │   └── perturbation_event  # bool
  └── attrs:
      ├── language       # 语言指令字符串
      ├── target_bp      # 目标车辆蓝图
      ├── town           # CARLA 地图名
      └── weather        # 天气参数
```

### 5.3 数据标注

| 标注 | 来源 | 方法 |
|---|---|---|
| occlusion [0-1] | 自动 | bbox 面积 + 画面外检测 |
| search_mode [0-1] | 自动 | occlusion × (1 + time_lost/5) |
| waypoint GT (3×3) | 自动 | 目标未来 2s/4s/6s 位置 (卡尔曼平滑) |
| perturbation_event | 实时 | 扰动注入器记录 |
| target_turning | 实时 | autopilot yaw 变化率 |
| language | 自动 | 模板生成器 (多层模板 + 目标描述) |

---

## 六、核心创新点

> **贡献层次（诚实定位）**：核心机制（EAR/IAR/AGP 动作思维链）**迁移自 ACoT-VLA (2601.11404)**，非本文发明；本文真正的贡献是 **§6.3 的新任务 + 新 benchmark（空中·车辆·语言消歧·具身闭环）** 以及把动作 CoT 适配到该场景。§6.1/§6.2 记录的是**迁移适配**与 UAV 特化，§6.4 为工程设计而非"涌现"发现。

### 6.1 迁移适配：动作空间推理用于空中跟踪 (EAR)

将 ACoT-VLA 的"在动作空间而非语言空间推理"迁移到 UAV 跟踪。与语言 CoT 基线对比：

```
语言 CoT (ECoT / CognitiveDrone-R1 风格):
  "目标上次在X, 速度Y, 方向Z, 可能出现在W"
  → 文字 → bbox 2D 坐标 → 3D 位置转换
  → 语义-运动鸿沟, 精度损失

动作 CoT (ACoT-UAV-Track):
  EAR 直接输出 3D 拦截点: [(x₁,y₁,z₁,c₁), (x₂,y₂,z₂,c₂), (x₃,y₃,z₃,c₃)]
  → 这些点 = 预测的目标未来位置
  → 与 UAV 动作空间同构 (都是 3D 空间坐标)
  → 无跨模态鸿沟
```

**与地面先例 TrackVLA++ (2510.07134) 的机制差异**（不能只说"它在地面"）：TrackVLA++ 的 Polar-CoT 输出**量化极坐标 token**（离散、2D 极坐标、单目标点）；本工作 EAR 输出**连续 3D 拦截点序列 + 置信度**并用 flow-matching 去噪，天然表达 6-DoF 空中拦截几何与多步时序。目标身份保持上，本工作依赖 IAR 隐式先验 + 语言消歧，而非 TrackVLA++ 的显式记忆模块 TIM。

**★ 招牌场景：出画预测-拦截重捕获（EAR 价值的核心兑现 + aerial 差异化）**
目标拐弯/被遮挡/出画而**短暂丢失**时,系统**不结束跟踪、也不盲跟**,而是由 EAR 预测目标未来 3D 位置,**直飞预测拦截点抄近路重捕获**。
- **为何是 aerial 独有**:无人机**无路网约束**,可沿直线抄近路飞向目标重现处;地面/竞品(TrackVLA++ 等)只能沿可行驶区域反应式跟随,做不到"抄近路拦截"。这把 EAR 的"预测 3D 拦截点"从锦上添花变成**不可替代的能力**(H1 的最强证据)。
- **机制(短丢失 <6s)**:闭环里 S2 在**最后可见帧**产出的 EAR waypoint(1/2/4/6s)延伸进丢失期,经 **staleness augmentation**(原则5)被 S1 复用为拦截目标 —— 现有设计本就对得上。
- **⚠️ 架构含义(长丢失 >6s)**:单帧输入 + proprio **无目标状态记忆**(proprio 是无人机自身状态),超出 EAR horizon 后无从预测。需**延长 EAR horizon** 或**显式传播目标状态估计**(以最后 waypoint 播种)或 IAR 加记忆;拦截不可靠时由 **search_mode + 语言重捕获** 兜底(§6.4)。
- **监督**:`target/t*` 出画期间仍有效 → 拦截 waypoint 与专家拦截动作**已可监督**;训练需把丢失帧的 action 从"一律 mask"改为"专家演示拦截则监督拦截"(见 `data-fix-spec.md` §3.2)。
- **数据/评测依赖**:需数据含长丢失(5-15s)+ 目标重现 + 拦截式专家演示(`data-fix-spec.md` §3);闭环新指标"重捕获成功率 vs 丢失时长"。

### 6.2 IAR 隐式先验提取（UAV 特化）

IAR 的逐层 KV-cache 查询机制**继承自 ACoT-VLA**；本文的适配在于提取 UAV 跟踪特有的、编码在 VLM 中间层表征里的三种"说不出来"的先验：

| 视觉线索 | 隐式先验 | 行为影响 |
|---|---|---|
| 目标走向建筑后方 | "即将被遮挡" | 提前爬升保持视线 |
| 目标突然加速/变向 | "机动意图" | 调整跟踪激进程度 |
| 多辆相似车辆出现 | "身份混淆风险" | 更依赖运动模型 |

### 6.3 多车辆语言指定消歧 + 长丢失预测重捕获 —— 核心贡献

**可辩护的新颖性（收窄后的精确主张）**：截至 2026-07，"**主动飞行 + 语言指定目标车辆 + 多辆视觉相似干扰车辆消歧 + 具身闭环**"这一交集无先例发表。**关键的"杀手场景"把这个交集从静态消歧推向动态**：

> **★ 长丢失预测-拦截重捕获**：目标拐弯/遮挡/出画导致**视觉连续性彻底断裂**,重现时画面里**同时有 ≥2 辆 look-alike 干扰车**——空间预测(EAR)能把无人机带到目标大致重现区域,但"这几辆相似车里哪辆才是目标"**只有语言能定**。此刻**位置捷径失效、视觉连续性失效、外观不可分**三者叠加,语言成为**唯一**可区分信号(H0 在此最强),同时兑现 aerial 预测拦截(H1)。

逐一切割（编号均已核验 2026-08-05，见 `related-work-and-positioning.md` §2/§3/§9）：

- **空中闭环不止一家,但都不做语言消歧**（措辞收口）：空中·主动·闭环现有**两家**——**UAV-Track VLA**(2604.02241,预印本,语言)与 **DeTrack**(2605.17451,预印本,**无语言**)。故 intro 不再写"空中闭环稀缺",改写"空中闭环已有两家、但**无一做语言指定的相似车消歧**"。
- **vs UAV-Track VLA (2604.02241)**：已做空中·主动·语言跟踪，但**不强调相似干扰物消歧**、更无长丢失语言重捕获；本工作补上该环 + 语言 load-bearing 证据（§7.4）。
- **vs DeTrack (2605.17451)** ⚠️**须切三点**：它做空中主动闭环 + 避障,但 ① **纯视觉 world-model(AaDWorlds)、无语言**;② "distractors"是**动态遮挡物(车/人)非视觉相似车**,不做消歧;③ 它撞了"drone-embodied tracking"和"world-model"两个词——本工作切割为 **world-model-augmented VLA + 语言消歧**(它是无语言 pure world-model)。**⚠️ 勿混 NeurIPS'24 同名被动 SOT "DeTrack" (2501.02467),引用须用全称+编号。**
- **vs AerialMind (2511.21053 → AAAI-26 已发表)** ⚠️**最接近的已发表语言消歧竞品**：它做 UAV 视角 **referring MOT**(语言指定相似目标消歧),**peer-reviewed**,但为**被动感知**(建于 VisDrone/UAVDT 录像),不控制无人机。差别**收窄到一条轴:主动飞行闭环 vs 被动感知**——须写得特别干净。(**AerialMind ≠ UAVNLT**,两篇不同工作;UAVNLT 为 Electronics'24。)
- **vs TrackVLA++ (2510.07134)**：抗干扰具身跟踪，重捕获**靠非语言空间连续性 + 显式记忆(TIM / Polar-CoT)**，且为**地面/人**。**在"连续性断裂 + look-alike 共视"的重捕获时刻,非语言空间记忆无法回答"哪辆才是目标",而本工作以语言 + 预测拦截可以**——最锐利差异化。

**★ 杀手场景的先例切割（deep-research 2026-07-30 锁定；详证与 cut 措辞见 `related-work-and-positioning.md` §4B/§4B.1）**：升级创新点=三能力**同时叠加**为真空白，但每个"零件"已被占据，故须逐件切割、novelty 押在**耦合**：

| 零件 | 已占据先例(venue) | 用语言? | 切割 |
|---|---|---|---|
| 预测未来位置+飞向拦截(抄近路) | Fast-Tracker(ICRA'21)、PN·ATPNG 制导(Springer'21–22) | ❌ 纯几何,身份已知 | 拦截机制不新→别押;它们是"保持在视野"非"完全丢失后 rendezvous" |
| FOV 离开-重入时刻消歧 | **DAM4SAM/SAM2.1++(CVPR'25)** | ❌ 外观 anchor 记忆 | 它**点名**该时刻却用外观解→语言消歧此刻空位 |
| 完全丢失后重检测 | GlobalTrack(AAAI'20)/SiamR-CNN(CVPR'20)/LTMU(CVPR'20) | ❌ 外观/运动 | 重识别只有外观先例 |
| 中途/丢失后用语言 | **TNL2K AdaSwitcher(CVPR'21)、Li'17(CVPR'17)** 全局重检测;Feng(WACV'20)、SNLT(CVPR'21) 逐帧;**QueryNLT(CVPR'24)** 连续消歧 | ⚠️ 有,但**均为外观驱动全局重定位/逐帧定位/连续消歧,非"丢失重现时刻 look-alike 唯一消歧"** | 逐一切割(见下正面弹药) |

- **🟢 正面弹药 DecoupleTNL (ICCV 2023)**：已发表地论证"语言分支**倾向最大化同类相似度**、干扰分离**交给视觉分支**"——**直接反证** pre-VLA NL-tracking 不把语言当 look-alike 消歧器,坐实本工作设定是新的。
- **cut 措辞(可直接改写)**：先前 NL-tracker 用语言做首帧指定与逐帧定位;两篇(TNL2K、Li'17)另用语言做**全局重检测**,但均靠外观相似度找回**同一模板**;DecoupleTNL 更表明语言干扰分离交给视觉。**无任何工作把细粒度语言用作"长丢失后重现、多辆共视 look-alike 中挑目标"的唯一消歧信号**(与 QueryNLT 区分:它只在连续跟踪消歧,从不在丢失重现时刻)。

**消歧证据精确到"重捕获时刻"（呼应数据侧修订，`data-fix-spec.md` §3/§4）**：语言必要性**不要求全局 look-alike 高共视**,而要求**断裂后的重捕获帧 ≥2 辆 look-alike 共视 + 足够多的重捕获事件**。全局共视率可放宽,消歧证据聚焦在最能体现"语言唯一可分"的时刻——避免退化成纯预测(那会与 TrackVLA++ 的非语言空间重捕获同质)。

> ⚠️ 不再声称"首个 UAV 语言跟踪 / 首个动作 CoT 跟踪 / 首个中途用语言"——这三者分别被 UAV-Track VLA、TrackVLA++、Feng/TNL2K 占据。主张严格限定在四轴交集 **+ 长丢失语言重捕获杀手场景（语言=丢失重现时刻的唯一消歧信号）**。残余未穷尽核验(GTI/CTRNL/CiteTracker 等)→ 论文写 "to our knowledge"。

### 6.4 search_mode 连续维度

用单一连续维度替代传统的离散模式切换 (TRACK→SEARCH→RECOVER)：

```
search_mode = f(EAR 置信度, IAR 遮挡先验, 目标丢失时长)

0.0 ────────────────────────────────────────── 1.0
精确跟踪    目标模糊      丢失 3s         丢失 10s      完全丢失
锁定目标    扩大视野     按预测轨迹飞   预测+螺旋扫描  全区域扫描
```

**关于监督方式的诚实说明**：search_mode 在训练中由手工标签 `occlusion × (1 + time_lost/5)`（§5.3）**显式监督**（L_visibility MSE），因此它**不是"涌现"信号**，而是一个可微、连续的回归目标。相对传统 TRACK→SEARCH→RECOVER **离散硬切换**的真正优势在于**连续可导、无阈值抖动（避免 Zeno 式虚假模式切换）**，而非"自发涌现"。若要主张涌现，需去掉显式监督并单独实验验证——列为未来工作。

---

## 七、实验设计

### 7.1 对比方法

| 方法 | 类型 | 验证什么 |
|---|---|---|
| **ACoT-UAV-Track (ours)** | 完整提案 | — |
| **UAV-Track VLA** (2604.02241) | 已发表 SOTA（头号竞品） | 空中主动语言跟踪基线；差异在干扰物消歧 |
| **DeTrack** (2605.17451, 若代码/环境可得) | 已发表 | 无人机闭环 + 移动干扰物（无语言）基线 |
| **π₀.₅ fine-tuned** | 通用 VLA 基线 | 无 UAV 特化、无推理 |
| **Language-CoT 头**（ECoT 风格，自建） | 推理范式对照 | 动作 CoT vs 语言 CoT（H3） |
| **外观/运动-only（无语言）基线** | **关键对照** | 证明语言 load-bearing：仅凭外观/运动能否消歧 |
| ACoT-UAV w/o EAR | 消融 | EAR 的显式推理贡献 |
| ACoT-UAV w/o IAR | 消融 | IAR 的隐式先验贡献 |
| ACoT-UAV w/o Language | 消融 | 语言引导的贡献（配合 §7.4） |
| **视觉重检测重捕获**（Siam R-CNN / DAM4SAM 风格,自建） | **杀手场景关键对照** | 证明重捕获时刻语言 > 纯外观重检测(H0 在重现时刻);非语言空间记忆无法在 look-alike 间选对 |
| **纯预测拦截无语言**（EAR-only 拦截,去 target-id 语言） | **杀手场景关键对照** | 证明抄近路到重现区后"哪辆是目标"必须靠语言;否则退化成 Fast-Tracker/PN 式几何拦截 |

> 已移除"CosFly-VLA (reproduced)"——该模型不存在，无法复现（见 §1.3）。TrackVLA++ 为地面工作，作为相关工作讨论；若作实验基线需说明其地面→空中的适配限制。
> **重捕获时刻的先例切割（must-cut，详见 §6.3 表 + `related-work-and-positioning.md` §4B/§4B.1）**：Fast-Tracker/PN(预测拦截,纯几何)、DAM4SAM(离开-重入,外观解)、Feng WACV'20(中途语言,逐帧)、TNL2K/Li'17(语言全局重检测,外观驱动)、QueryNLT(连续消歧,非丢失重现)、DecoupleTNL(ICCV'23,🟢正面弹药:语言不做同类消歧)。**新增两条杀手场景对照**(上表末二行)把"重捕获时刻语言唯一可分"做成可量化消融。

### 7.2 核心指标

| 指标 | 定义 |
|---|---|
| Success Rate | 目标保持在视野中直到 episode 结束的比例 |
| **Mis-follow Rate / ID-correctness** | **跟错到干扰车的比例（越低越好）——直接度量语言消歧能力，是语言贡献的核心证据** |
| Occlusion Recovery Rate | 遮挡后 N 秒内重新锁定的比例 |
| Avg Tracking Frames | 平均连续跟踪帧数 |
| Search Mode Efficiency | search_mode > 0.5 的时间占比 (越低越好) |
| Cross-town SR drop (sim) | 已见→未见 CARLA 地图的成功率下降（**仅仿真内泛化**，非真机） |

### 7.3 三个核心假设

```
H1: EAR 显式 waypoint 预测 > 纯隐式 VLM 特征
    → 消融: ACoT-UAV vs ACoT-UAV w/o EAR
    → 预期: 遮挡恢复场景下差异最显著

H2: IAR 隐式先验在遮挡/机动场景有独立贡献
    → 消融: ACoT-UAV vs ACoT-UAV w/o IAR
    → 预期: 目标急转/急停时 IAR 提供提前预警

H3: 动作空间 CoT > 语言 CoT
    → 对比: ACoT-UAV vs 自建 Language-CoT 头 (ECoT 风格, 文字→bbox→3D)
    → 预期: 3D waypoint 精度优于 2D bbox→3D 转换
```

### 7.4 语言必要性验证（针对最致命的归因风险）

**风险**：目标车与干扰车均由 autopilot 驱动、专家动作 (PID) 直接跟随目标——若干扰车易区分或目标恒居画面中心，模型可学成"跟随最居中 / 最近那辆"而**根本不解析语言**，则高 SR 无法归因于语言 / 推理，核心主张落空。

**验证协议（必须通过，否则核心贡献不成立）**：
1. **抗捷径的数据设计**（§5.1）：≥2 辆高相似干扰车 + 目标车不恒居中 + 与干扰车并行/交叉。
2. **外观/运动-only 基线**（§7.1）：去掉语言输入。若其 Mis-follow Rate 已很低 → 语言不必要 → 任务无效，须加难。
3. **w/o-Language 消融**：完整模型去语言，Mis-follow Rate 上升幅度 = 语言的净贡献。
4. **反事实语言测试**：同一帧、只把语言指令改指另一辆车，检验模型是否切换目标（真正"听懂"语言的强证据）。
5. **专家上限说明**（§4.2）：模仿 P-only PID 的策略上限 ≈ 该 PID；须经阶段 3 RL 精调证明超越，或明确模型价值在"语言消歧 + 遮挡恢复"而非纯跟踪精度。

---

## 八、项目文件结构

```
uav-dist-paper2-baseline/
├── acot-uav-design.md              ← 本文档
├── carla_uav_tracking/             ← 数据生成管线 (Python)
│   ├── config/default.yaml         ← 全局配置
│   ├── carla_uav/                  ← UAV 控制模块
│   │   ├── drone.py                ← KineticDrone
│   │   ├── pid_controller.py       ← PID visual servoing
│   │   ├── expert_policy.py        ← Expert + noise
│   │   ├── perturbation.py         ← APF 扰动注入
│   │   ├── obstacle_avoidance.py   ← 高度安全层
│   │   └── safety.py               ← 安全监控
│   ├── targets/                    ← 目标车辆
│   │   ├── vehicle_target.py       ← Autopilot 车辆
│   │   ├── route_follower.py       ← 路网导航 (备用)
│   │   └── maneuver_injector.py    ← 机动注入
│   ├── scene/                      ← 场景管理
│   │   ├── scene_manager.py        ← 演员生命周期
│   │   ├── language_generator.py   ← 多车辆语言生成
│   │   └── occlusion.py            ← 遮挡检测
│   ├── recording/                  ← 数据录制
│   │   ├── sensors.py              ← RGB/深度相机
│   │   ├── recorder.py             ← HDF5 同步录制
│   │   └── postprocess.py          ← 自动标注
│   ├── scripts/                    ← 入口脚本
│   │   ├── generate_single.py      ← 单集调试
│   │   ├── generate_batch.py       ← 批量生成
│   │   ├── launch_parallel.py      ← 多 GPU 并行
│   │   └── visualize_episode.py    ← 可视化
│   └── tests/                      ← 单元测试
│       ├── test_pid_tracking.py    ← PID 10/10 ✅
│       └── test_data_format.py     ← HDF5 格式验证
└── data/raw/                       ← (测试数据)
```

---

## 九、当前状态

| 模块 | 状态 |
|---|---|
| D-EPA-RHP 局限性分析 | ✅ 已完成 |
| UAV VLA 文献调研 (LIBERO-Plus, GR00T N1, etc.) | ✅ 已完成 |
| ACoT-UAV-Track 架构设计 | ✅ 已完成 |
| 数据生成管线 (多车辆 + autopilot + APF + 语言) | ✅ 已完成, 已验证 |
| CARLA 环境部署 (Win11, RTX 4070 Ti, WSL2) | ✅ 已完成 |
| 批量数据生成 | ⏳ 待启动 (目标 ~350K 帧 / 45 GB) |
| ACoT-UAV-Track 模型训练 | ⏳ 下一阶段 |
