# ACoT-UAV-Track 训练计划

> 2026-07-27 | 校准到 MVP 数据规模 | 配套：`acot-uav-design.md` §3.4/§4（同目录）、`../acot_probe/`、`../positioning-analysis.md`
> 训练数据：`/nvidia/hque/data/carla_data/mvp`（59 集 / 106,200 帧 / 22 GB）
> 训练机：8×A40（48 GB/卡）

---

## ⚑ 当前状态与主线计划（2026-08-18 更新）

**一句话**：语言必要性（H0）已在 mvp_full_v5 上**干净证成**;研究重心正式转向**核心贡献——目标出可视范围后,如何靠 VLA+WAM 预测-拦截重捕获**。语言识别是创新点之一,服务于重捕获时的"认领"。

### 两个创新点(语言服务于核心)
| | 创新点 | 状态 | 作用 |
|---|---|---|---|
| **① 语言识别** | 语言 grounding 消歧目标 | ✅ **已证(H0 干净)** | 出画重捕获时,在共视 look-alike 里认出是哪一辆 |
| **② VLA+WAM 出画重捕获**(核心) | 目标出画后预测其位置 + 直飞拦截重捕获 | 🟡 **有据待建**(2026-08-18 公平 no-WM 基线证成单帧无法出画预测,见下) | 空中差异化(无路网约束、可抄近路直飞),地面跟踪做不到 |

**咬合**:出画 → WAM 预测去向 → VLA 飞拦截点 → 重现时语言认对(而非"哪辆先出现算哪辆")。

### H0 干净证成(2026-08-16,取代旧的位置捷径叙事)
- 经**指标净化**(候选打乱修 argmax 平局漏洞)+ **多版本几何去相关**(v3 深度 / v4 居中 / **v5 分辨率 336→512**)后:E1 **有语言 ~58% vs 无语言 ~72%(≈chance),稳定 gap ~13pp**,有语言破 chance。
- **关键定位**:v4 卡 chance 的真因是**图像分辨率**(车 12px 身份不可读),非数据/头——由"同分布探针 make 55% vs by-episode 6%"锁定,512px(车 27px)一改即翻盘。全过程见 `transition-diary-801.md` §七、memory `acot-uav-language-necessity`。
- **稳定化四件套**(tid dropout+独立wd+余弦LR+best-on-mis_follow)+ **省磁盘配方**(layer24-only+fp16 缓存=86GB)已固化。

### 公平 no-WM 基线证成 WM 必要性(2026-08-18,建 WM 前的"先测量"闭环)
**先回答"原模型是否已到上限、要不要过早加 WM"**(用户方法论)。E1 的 EAR 欠训,故先**公平重训**(`train/train_v5_fair.sh`:Stage-1a EAR 预热 + 5 层 IAR + `supervise_intercept` → `runs/stage2_v5_fair`,mis_follow 0.582)再用 `train/intercept_eval.py` 测出画预测:

| 指标 | E1(欠训) | **公平** | 判读 |
|---|---|---|---|
| 可见 EAR FDE 1s/6s | 15.2/23.3m | **14.2/21.2m** | 仅好 1-2m,**仍不如 persist**(13.7m@6s) |
| 丢失 EAR FDE 1s/6s | 58.5/77.2m | **58.3/77.7m** | **几乎不变,训练救不回** |
| 拦截 cos 丢失(中位/>0.5) | 0.92/71% | 0.87/**66%** | 部分拦截靠 BC 惯性,34% 飞错 |
| mis_follow | 0.579 | **0.582** | ≈,不受 EAR/IAR 层数影响 |

**定论**:公平版 EAR ≈ 欠训版 → EAR 出画预测的失败是**架构性(单帧限制),非欠训** → **反应式单帧无法预测出画位置 → WM(路网世界建模,P2)有据**;继续 tune 原模型无用。语言(mis_follow)在 grounding 天花板,优化走 grounding 不走加层。`stage2_v5_fair` = 干净的 no-WM 离线基线(M0),WM 须超越之。详见 `transition-diary-801.md` §八、`experiment-design.md` §7.4 ★callout、memory `acot-uav-wm-roadgraph`。

### 已完成工作
| 模块 | 状态 | 位置 |
|---|---|---|
| VLA 架构(EAR/IAR/DiT/tid) | ✅ | `train/{ear,iar,dit}.py` |
| 训练管线 Stage-1/2 + 稳定化 | ✅ | `train/stage{1_ear,2}.py` |
| 语言必要性 H0 | ✅ 证成 | memory `acot-uav-language-necessity` |
| **公平 no-WM 基线**(证 WM 必要性) | ✅ 证成 | `train/train_v5_fair.sh` + `intercept_eval.py` → `runs/stage2_v5_fair` |
| 数据 mvp_full_v5(几何清+512+描述唯一+`distractors/similar`) | ✅ | `mvp_full_v5` |
| **mask 反转** `supervise_intercept`(遮挡帧监督拦截) | ✅ 代码就位,待长丢失数据激活 | `dataset_stage2.py` §3.2 |
| 长丢失数据 | 🟡 部分(≤16.5s,缺拦截演示+重现共视) | data-fix-spec §3 |
| **WAM 目标状态记忆** | 🔲 只设计未实现 | memory `acot-uav-predict-intercept` |
| **闭环 rollout harness** | 🟡 env.py 骨架已搭,最大缺口 | memory `acot-uav-harness` |
| Stage-3 RL | 🔲 设计完成未启动 | memory `acot-uav-stage3-rl`、Phase E |

### 未来主线:5-Phase(按依赖排,关键路径优先)
| Phase | 内容 | 归属 | 依赖 |
|---|---|---|---|
| **P1 长丢失+拦截数据** | 5-15s 长丢失事件 + 专家演示直飞拦截 + 目标重现 + 重现处 ≥2 look-alike 共视 | 数据侧 | 无(可立即) |
| **P2 WM 路网图预测器**(RoadGraphPredictor,**方案A**) | 反应式→世界建模。**不靠速度**(UAV 自身动/目标非匀速/瞬时速度无意义),靠**路网结构**预测目标未来位置:2D 自由预测→"1D 车道遍历+分叉选择"(PGP/DenseTNT 式 velocity-free)。**功能划分(方案A)**:WM 定**远端 where**(路网/遮挡)· **EAR 保留近端 where**(视觉/可见)+ 融合成 z_ex · 语言定 which · DiT 定 how(**EAR 不删,重定范围**)。**IAR 耦合**:occ 头→WM 可见/遮挡门控、man 头→WM 分叉 seed、z_im→DiT(**复用现有 IAR 头,不新训**)。**训练三阶段**:Stage-1a EAR 预热(已有)‖ Stage-1b WM 预热(新:车道遍历 BC + 走廊损失)→ Stage-2 联合(warmstart 两者,加 L_wm,EAR/WM 共适应)。分级 P2a 最小(CARLA 路网GT+单步遍历+走廊,验 H7)→P2b 多步遍历+latent→P2c 语言 which-way(H8)+视觉车道图。详见 `wm-roadgraph-design.md` / `wm_scheme_compare.html` | 架构(本 agent) | **依赖 P1**(BC 拦截演示)+ CARLA 车道图GT |
| **P3 闭环 harness** | 完成 env.py:感知→策略→动作→CARLA→下一帧(S1 20Hz/S2 1Hz);场景生成/reset/终止/特权GT奖励 | 工程(本 agent) | 无(可立即,与 P1 并行) |
| **P4 闭环评测 + 语言集成** | E5 断裂后重捕获(有vs无语言 mis_follow 大涨)、重捕获锁对率(硬失败)/收敛时间、SR vs 丢失时长 T 曲线 | 本 agent | 依赖 P2+P3 |
| **P5 Stage-3 闭环 RL**(可选/加分) | GRPO 微调 DiT 末层,reward=R_correct_id+**R_reacquire**;治 BC 分布漂移 | 本 agent | 依赖 P3+P4 |

**关键路径**:`P1 数据 + P3 harness(并行起步)→ P2 WAM(待 P1)→ P4 闭环评测(E5)→ [P5 RL]`;语言识别(已完成)在 P4 重捕获处集成。

**建议起点(两条并行)**:① P1 数据(数据 agent,见 data-fix-spec §11);② P3 harness(本 agent,env.py 收尾——不依赖数据、是评测+Stage-3 共同底座、当前最大缺口)。**P2 WAM 是核心创新落点,待 P1 数据到位。**

---

## 0. 数据体检结论（决定计划怎么排）

| 项 | 状态 | 影响 |
|---|---|---|
| 规模 | 59 集 / 106K 帧 / 每集 1800 帧(3min@10fps) / **2 地图**(Town03×29, Town04×30) | MVP 级：够 EAR 收敛 + 端到端可演示；**不够**强统计消融/跨地图泛化 |
| cam 位姿 / waypoint GT / 干扰车 3D | ✅ 59/59 齐、可用 | Stage 1 可直接用 |
| 相似干扰物 | ✅ 49/59：same_shape 16 + same_color 10 + same_class 23；distinct 10 | 语言 load-bearing 难例充足；可支撑课程 2a→2c |
| 目标类别 | car 36 / motorcycle 10 / bicycle 9 / scooter 4 | 多类，泛化面好 |
| 🔴 `annotation/{bbox,occlusion,search_mode}` | **被坏投影污染**（occlusion 假高均值 0.546；bbox 仅目标、无干扰车、写死 224²/忽略 pitch=-30°） | **Stage-2 的 L_visibility / L_target_id 依赖它 → 训练前必须重算** |
| waypoint GT (`annotation/waypoints`) | ✅ 用 3D 直接算，不经投影，未受污染 | Stage 1 可信 |
| `acot_probe/projection.py` | ✅ 真图验证正确（绿框稳落目标车，8/8 帧目标在画面内） | 作为重算标签的正确投影源 |

**结论**：数据主体可用；唯一地基问题是 **annotation 投影类标签被污染**，必须先修（Phase A1）。

---

## 1. 分阶段计划（按依赖/优先级排序）

### Phase A — 数据就绪（训练前必做；A1/A3 可在 CPU 完成）

- **A1【最高优先｜修标签】**
  用 `acot_probe/projection.py` 的正确投影替换 `carla_uav_tracking/recording/postprocess.py` 的 `_compute_bboxes` / `_compute_occlusions`：
  - 重算 **目标 + 全部干扰车** 的 2D bbox（当前仅目标）→ 供 Stage-2 `L_target_id`
  - 用正确投影（含 cam_pitch/cam_yaw、336²）重算 occlusion、search_mode
  - 在 59 集上重跑；验证 mean occlusion 回落到合理值、可视化抽查
  - waypoint GT 无需重算
- **A2 数据划分**
  按 episode 分 train/val/test（~47/6/6），按 strategy×class 分层。2 地图只能做**粗**跨地图检查（held-out 一张 town）；强泛化留给 scale-up。
- **A3 dataloader（当前不存在）**
  读 rgb + proprio(8d) + language + waypoint GT + 新 bbox + occlusion/search_mode；实现**冻结 VLM KV 预计算并缓存**（设计原则1），训练循环不再前向 VLM。

### Phase B — 骨干选型（§3.4 gate；现在即可跑，Stage 1 前必须完成）

- **B1**：`acot_probe` 在 ~15 集相似干扰子集上跑 **PaliGemma-2-3B vs Qwen3-VL-4B** 逐层探针（layers 8/12/16/20/24/last）→ 输出 backbone×layer 的 Target-Selection Accuracy + layer-16 PASS/WEAK 判定。
  - 前置：`pip install -r acot_probe/requirements.txt`；先 `python -m acot_probe.visualize` 抽查投影（本机已验证过 ep0）
  - 产出决定 EAR/IAR 读**哪个骨干、截断到哪层**；预期 PaliGemma 在 16 层过、Qwen 需 ~20-24
  - 运行：PaliGemma→cuda:0，Qwen→cuda:1（见 `acot_probe/config.yaml`）

### Phase C — Stage 1：EAR 预热（建模型 + 训练）

- 建 EAR（4 层 Transformer，d=256，flow-matching）+ 选定骨干的 per-layer KV 读取 + Stage-1 训练循环
- 冻结 VLM/IAR/DiT，仅训 EAR；监督 = waypoint GT（干净）
- **门槛**：waypoint MSE < 10m @6s；且 EAR 预测的是**目标车**未来轨迹而非干扰车（early 语言-grounding 检查）

### Phase D — Stage 2：端到端 + 难度课程

- 建 IAR + AGP/DiT(32层) + proprio-MLP + 全损失：
  `L_action + 0.3·L_ear + 0.1·(L_visibility+L_maneuver) + 0.2·L_target_id`
- **课程**（按 `strategy` 字段）：2a distinct → 2b same_class → 2c same_shape/same_color；每档过门再进
- **staleness augmentation**：随机复用过期 EAR/IAR 输出，匹配 S1(20Hz)/S2(1Hz) 异步推理
- 专家 = 特权信息蒸馏（PID 用目标真值，学生只有 RGB+语言）
- **门槛**：held-out 上 Mis-follow Rate 低，且 w/o-Language 消融时显著上升（=语言有净贡献；否则回 A/数据加难）
- **MVP 预期**：收敛 + 可演示 + **初步**消融；统计强消融/跨地图需 scale 到 ~300K + 更多地图

### Phase E — Stage 3：闭环 RL 精调（可选，后置）

**摘要**：冻结 VLM/EAR/IAR，训 DiT 末 4 层（或 LoRA）；GRPO，CARLA on-policy；Reward 含 **R_correct_id**（跟对/跟错被指目标），防 RL 崩成"跟任意车"；门槛：SR 与 Mis-follow 均优于 Stage 2 BC、无 reward hacking。

**定位**：这是**后训练的 RL fine-tuning 子阶段**（预训练=冻结的 Qwen3-VL；SFT/BC=Stage 1-2；RL 精调=本阶段），与 LLM 的 RLHF、机器人"BC 预训练→RL 精调"同构。**动机**：Stage 2 是离线 BC，存在**误差累积/分布漂移**（走到专家没演示的状态就步步偏）；闭环 RL 让策略**真的在 CARLA 里飞**（on-policy rollout），经历自己造成的状态分布，直接优化"有没有跟对目标"的真实目标。

#### E.1 观测空间 o_t（只用部署可得信息，与 BC 一致）
- 当前帧经**冻结 VLM** → `vlm_ctx (100,2560)` + `vlm_ctx_layers (5,100,2560)`；语言指令（目标身份，整段固定）；`cand_feats (N,2560)`（消歧）；`proprio (5)`（高度/俯仰+三轴速度）；可选 EAR 航点 / IAR 先验。
- ⚠️ **工程约束**：闭环状态是新的，**VLM 必须在线推理**（不能预计算）→ 冻结 VLM/EAR/IAR 只训 DiT 末层 + 双系统 S1(20Hz)/S2(1Hz) 异步，使在线可行。

#### E.2 动作空间 a_t（同 BC）
- 连续控制 `[dx,dy,dz,dyaw] + search_mode`，DiT 用 flow-matching **生成 16 步动作块**；有界。
- RL 需**随机策略**：flow-matching 采样天然给动作分布 → 采样多条供 GRPO 算相对优势。

#### E.3 奖励函数 r_t（加权和；用仿真器特权 GT 算，部署不用）
| 分量 | 含义 | 作用 |
|---|---|---|
| **★ R_correct_id** | 跟对被指目标 +，锁上干扰车重罚 | **防 reward hacking 崩成"跟任意车"（最大风险）** |
| R_track | 目标在画面内/居中/合适距离 | 跟踪主目标 |
| R_reacquire | 长丢失后成功重捕获 | 强化预测-拦截（旗舰场景，见 [[predict-intercept]]） |
| R_smooth / R_effort | 惩罚抖动/过大动作 | 平滑飞行 |
| R_safety | 撞击/出界惩罚 | 安全 |

#### E.4 算法与冻结
- **GRPO**：每状态采样一组 K 条 rollout，用"奖励−组内均值"作相对优势，**无需 critic**（契合扩散策略本就在采样）。
- **冻结 VLM/EAR/IAR，只训 DiT 末 4 层或 LoRA**：①省算力；②**别破坏 BC 学到的语言 grounding（tid 头）与几何（EAR）**；③防灾难性遗忘 —— RL 只精修"动作生成"最后一环。

#### E.5 门槛与依赖
- **Gate**：SR 与 mis_follow **均优于 Stage-2 BC**，且无 reward hacking（靠 R_correct_id + 持续监控 mis_follow 守住）。
- **依赖（硬前置）**：闭环 rollout harness（`experiment-design.md` §7「双重角色」note，**当前待建**）。**为什么是硬前置**：RL 的目标 = "策略在环境里飞出的轨迹分布"上的期望——没有环境在训练回路里就**没有轨迹/奖励/可优化对象**（on-policy 数据须现场 rollout 生成、状态分布由策略自身动作决定）。BC(Stage 1-2)读固定离线数据集**不需要** harness；RL(Stage 3)**必须**把环境放进训练回路。**先有 harness，才谈得上 Stage-3 训练与其闭环 gate**——这也是本阶段排最后、标"可选/后置"的原因。**MVP 靠 Stage-2 BC 即可交付**，Stage 3 为加分项。
- **三大难点**：①在线 VLM 推理成本；②reward 设计/hacking；③on-policy 样本效率。

---

## 2. MVP 能交付 / 不能交付

- ✅ 骨干选型结论；EAR 收敛；端到端跟踪可演示；语言必要性的**初步**证据（w/o-language 消融、反事实语言）
- ❌ 统计显著的完整消融（数据量偏小）；强跨地图泛化（仅 2 地图）；SOTA 规模对比
- → 打通 MVP 后按 `acot-uav-design.md` §4.1 scale 到 ~300K 帧 + Town05 等更多地图

---

## 3. 立即可动（不需 GPU）与建议顺序

| 步骤 | 需要 | 谁/在哪 |
|---|---|---|
| **A1 修标签 + 加干扰车 bbox**（CPU，可在本机重跑 59 集验证） | h5py/numpy | 现在做 |
| **A3 dataloader + Stage-1 EAR 骨架** | CPU 建码，训练需 GPU | 接 A1 |
| **B1 骨干探针** | 4×A40 + 模型下载 | 你在 A40 上跑 |

**建议：先做 A1（修标签，地基）→ 你并行在 A40 跑 B1（定骨干）→ 再 A3+C（dataloader+EAR）→ D → (E)。**

---

## 4. 关联文件

> 本计划位于 `acot_note/`；设计文档同目录，其余在上一级 `uav-acot-track/`。

- 设计文档：`acot-uav-design.md`（同目录；§3.4 骨干选型/探针、§4 训练策略）
- 骨干探针包：`../acot_probe/`（`run_probe.py` / `projection.py` / `README.md`）
- 待修 bug：`../carla_uav_tracking/recording/postprocess.py`（`_compute_bboxes` 坏投影、仅目标）
- 定位/竞品：`../positioning-analysis.md`
- 训练数据：`/nvidia/hque/data/carla_data/mvp`（绝对路径）
