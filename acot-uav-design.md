# ACoT-UAV-Track：动作思维链驱动的 UAV 具身跟踪 VLA 系统

> 完整技术设计文档 | 2026-07-24

---

## 一、研究背景与动机

### 1.1 从 D-EPA-RHP 到 VLA 的范式转变

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

### 1.3 对标研究

| 方法 | 推理范式 | 核心局限 |
|---|---|---|
| **UAV-Track VLA** (2026.04) | 隐式推理（VLM 特征 → DiT cross-attention） | 无显式预测，遮挡下纯黑盒 |
| **CosFly-VLA** (2026.07) | 语言 CoT（文字推理 + bbox 预测） | 语义-运动鸿沟，文字→坐标精度损失 |
| **ACoT-UAV-Track (ours)** | **动作 CoT（EAR waypoint + IAR 隐式先验）** | — |

---

## 二、问题定义

### 2.1 UAV 具身跟踪的形式化

```
给定:
  - 观测: 连续 RGB 帧序列 (I₁, ..., I_t)，336×336 @ 10Hz
  - 语言指令: "Track the {target_desc} among {N} other vehicles."
  - 本体感知: UAV 位置/速度/偏航/能量（从仿真器读取）
  - 初始目标位置: 首帧 bbox 标注

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

---

## 三、模型架构

### 3.1 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    ACoT-UAV-Track                                 │
│                                                                  │
│  输入                                                             │
│  ├── 视觉: 连续 5 帧 RGB (336×336) + 首帧 target bbox 叠加        │
│  ├── 语言: "Track the orange Jeep Wrangler among 6 vehicles"     │
│  └── 本体: [px,py,pz, vx,vy,vz, yaw, energy]                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ VLM 骨干 (frozen, 截断到 layer 16)                          │  │
│  │ · 选型: Qwen2.5-VL-3B 或 SmolVLM-2B                         │  │
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
| VLM 骨干 | ~3B (frozen) | Qwen2.5-VL-3B, 截断 layer 16 |
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

---

## 四、训练策略

### 4.1 数据需求

| 目标 | 数据量 | 存储 | 说明 |
|---|---|---|---|
| 概念验证 (MVP) | ~100K 帧 (55 集 × 3min) | ~15 GB | EAR 收敛，跟踪可演示 |
| 可发表 | ~300K 帧 (165 集 × 3min) | ~45 GB | 消融显著，对比有统计显著性 |
| 对标 SOTA | ~900K 帧 (500 集 × 3min) | ~140 GB | 与 UAV-Track VLA (892K) 同级 |

**当前目标**: ≥ 350K 帧 (~45 GB)，D 盘 65 GB 可用

### 4.2 三阶段训练管线

```
═══════════════════════════════════════════════════════════════
阶段 1: EAR 预热 (50 epochs)
═══════════════════════════════════════════════════════════════
目标: EAR 学会从 VLM 特征 + 历史帧预测目标未来空间位置
冻结: VLM, IAR, DiT
训练: EAR
输入: VLM KV Cache + 噪声 waypoint
监督: 从专家轨迹自动标注的未来目标位置
Loss: L_flow(W) = MSE(去噪后 waypoint, 真值 waypoint)
      + λ·时间一致性 loss (相邻 waypoint 速度连续)
数据: 全部 episode 的 waypoint 真值
期望: EAR 的 3D waypoint 预测误差 < 10m @ 6s 预测 horizon

═══════════════════════════════════════════════════════════════
阶段 2: 端到端联合训练 (150 epochs)
═══════════════════════════════════════════════════════════════
目标: 联合训练 EAR + IAR + DiT，学会从粗拦截点到精细跟踪动作
冻结: VLM
训练: EAR, IAR, DiT
输入: 完整输入 (视觉 + 文本 + 本体)
监督: 专家动作序列 (PID visual servoing)
Loss:
  L_total = L_action + 0.3·L_ear + 0.1·(L_visibility + L_maneuver)
  
  L_action: Flow Matching loss on action prediction
  L_ear: Flow Matching loss on waypoint prediction
  L_visibility: search_mode MSE (遮挡 → search_mode 增大)
  L_maneuver: 目标机动检测 BCE
数据: ~350K 帧
优化: AdamW, lr=1e-4, batch=64, 4×A100

═══════════════════════════════════════════════════════════════
阶段 3: 闭环 RL 精调 (可选, 500 iterations)
═══════════════════════════════════════════════════════════════
目标: 在策略自身诱导的分布上优化
冻结: VLM, EAR, IAR
训练: DiT 最后 4 层
方法: GRPO
Reward:
  R_track: +1/帧 跟踪成功 (目标在画面中)
  R_recover: +5 遮挡恢复 (重新锁定目标)
  R_search: -0.01 当 search_mode > 0.5 (惩罚长时间搜索)
  R_collision: -10 碰撞
  R_smooth: -0.001∥a_t - a_{t-1}∥
数据: CARLA 仿真器 rollout
```

---

## 五、数据生成管线

### 5.1 场景设计

```
目标车辆 (1 辆):
  · 从 25 种民用车辆中随机选择 (CARLA CARS_ONLY)
  · CARLA autopilot 沿路网行驶 (含交通灯停车、路口转弯)
  · 速度: 5-15 m/s (autopilot 控制)

干扰车辆 (4-8 辆):
  · 从其余车辆中随机选择 (视觉上与目标区分)
  · 同一条路 ±60m 范围内 spawn (确保在 UAV 视野内)
  · 同样使用 autopilot 行驶

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

### 6.1 动作空间推理替代语言推理 (EAR)

```
语言 CoT (CosFly-VLA):
  "目标上次在X, 速度Y, 方向Z, 可能出现在W"
  → 文字 → bbox 2D 坐标 → 3D 位置转换
  → 语义-运动鸿沟, 精度损失

动作 CoT (ACoT-UAV-Track):
  EAR 直接输出 3D 拦截点: [(x₁,y₁,z₁,c₁), (x₂,y₂,z₂,c₂), (x₃,y₃,z₃,c₃)]
  → 这些点 = 预测的目标未来位置
  → 与 UAV 动作空间同构 (都是 3D 空间坐标)
  → 无跨模态鸿沟
```

### 6.2 IAR 隐式先验提取

从 VLM 中间层表征中学习三种"说不出来"的先验：

| 视觉线索 | 隐式先验 | 行为影响 |
|---|---|---|
| 目标走向建筑后方 | "即将被遮挡" | 提前爬升保持视线 |
| 目标突然加速/变向 | "机动意图" | 调整跟踪激进程度 |
| 多辆相似车辆出现 | "身份混淆风险" | 更依赖运动模型 |

### 6.3 多车辆语言指定跟踪

首个在 UAV 跟踪中引入多干扰车辆 + 语言指定目标的 VLA 工作。验证了 VLA 的语言理解能力和视觉辨识能力在 UAV 平台的联合应用。

### 6.4 search_mode 连续维度

用单一连续维度替代传统的离散模式切换 (TRACK→SEARCH→RECOVER)：

```
search_mode = f(EAR 置信度, IAR 遮挡先验, 目标丢失时长)

0.0 ────────────────────────────────────────── 1.0
精确跟踪    目标模糊      丢失 3s         丢失 10s      完全丢失
锁定目标    扩大视野     按预测轨迹飞   预测+螺旋扫描  全区域扫描
```

无需硬阈值——search_mode 是从交叉注意力中涌现的连续信号。

---

## 七、实验设计

### 7.1 对比方法

| 方法 | 类型 | 验证什么 |
|---|---|---|
| **ACoT-UAV-Track (ours)** | 完整提案 | — |
| UAV-Track VLA | 已发表 SOTA | 隐式推理基线 |
| CosFly-VLA (reproduced) | 已发表 SOTA | 语言 CoT 基线 |
| π₀.₅ fine-tuned | 通用 VLA 基线 | 无 UAV 特化 |
| ACoT-UAV w/o EAR | 消融 | EAR 的显式推理贡献 |
| ACoT-UAV w/o IAR | 消融 | IAR 的隐式先验贡献 |
| ACoT-UAV w/o Language | 消融 | 语言引导的贡献 |

### 7.2 核心指标

| 指标 | 定义 |
|---|---|
| Success Rate | 目标保持在视野中直到 episode 结束的比例 |
| Occlusion Recovery Rate | 遮挡后 N 秒内重新锁定的比例 |
| Avg Tracking Frames | 平均连续跟踪帧数 |
| Search Mode Efficiency | search_mode > 0.5 的时间占比 (越低越好) |
| Zero-shot SR drop | 已见→未见场景的成功率下降幅度 |

### 7.3 三个核心假设

```
H1: EAR 显式 waypoint 预测 > 纯隐式 VLM 特征
    → 消融: ACoT-UAV vs ACoT-UAV w/o EAR
    → 预期: 遮挡恢复场景下差异最显著

H2: IAR 隐式先验在遮挡/机动场景有独立贡献
    → 消融: ACoT-UAV vs ACoT-UAV w/o IAR
    → 预期: 目标急转/急停时 IAR 提供提前预警

H3: 动作空间 CoT > 语言 CoT (CosFly-VLA)
    → 对比: ACoT-UAV vs CosFly-VLA (语言 CoT 头)
    → 预期: 3D waypoint 精度优于 2D bbox→3D 转换
```

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
