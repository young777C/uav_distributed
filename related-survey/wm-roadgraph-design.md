# WM 路网图预测器设计 · RoadGraphPredictor(替代 velocity-WAM)

> 2026-08-18 · 核心贡献②(出画预测-拦截重捕获)的世界模型落点。配套:`wam_architecture.html`、`training-plan.md` Phase P2、`related-work-and-positioning.md` §4D(SEP)、`experiment-design.md` §2.0c/H7/H8、memory `acot-uav-predict-intercept`。

## 0. TL;DR
目标出画后,**不靠速度外推**(UAV 自身在动→相对测速难;目标非匀速→瞬时速度无意义),而**靠路网结构预测目标未来可能出现的位置**。把"2D 自由空间预测"塌缩成"**1D 车道遍历 + 分叉选择**"——这正是 AV 轨迹预测 SOTA(PGP/DenseTNT)的 velocity-free 范式,搬到空中 VLA 是 novelty。

---

## 1. VLA vs WM 功能划分(架构梳理)

**两个系统按"语义角色 × 目标是否可见"分工:**

| | VLA(反应式感知-动作) | WM(预测式空间推理)= RoadGraphPredictor |
|---|---|---|
| 回答的问题 | **WHICH**(哪辆是目标)+ **HOW**(怎么飞) | **WHERE**(目标现在/将来在哪) |
| 何时主导 | 目标**可见**时 | 目标**丢失/出画**时 |
| 组成 | tid(语言认目标)+ EAR/IAR/DiT(动作) | 车道图编码 + 遍历策略 + 走廊预测 |
| 依赖 | 当前帧 VLM 特征 + 语言 | 最后可见位置 + **路网结构**(+ 语言意图) |
| 输出 | 目标身份 + UAV 动作块 | 目标未来位置分布 / 可达走廊 |

**咬合流程(时间轴):**
```
可见:  VLA 认对目标(tid/语言)+ 跟随;WM 用观测校正"目标在哪条车道"
丢失:  VLA 无目标可看 → WM 主导:沿路网遍历预测目标未来走廊 → 喂 VLA
        → VLA(EAR/DiT)飞向走廊拦截
重现:  VLA 的 tid+语言 在共视 look-alike 里【认对】目标 → WM 校正 → 回到可见
```

**关键澄清——EAR 角色的转变:**
- 现有 EAR 试图"从单帧猜目标未来位置",但单帧无速度、无记忆、无路网 → **这本是 WM 的活,EAR 做不好**;
- 重构后:**WM 负责"目标未来 WHERE"(路网强先验)**;**EAR/DiT 退回本职"UAV 动作生成"**(给定 WM 预测的目标走廊,规划怎么飞过去);
- 即:**未来位置预测 从 VLA 挪到 WM;VLA 专注 认领(WHICH)+ 动作(HOW)。**

**一句话**:**WM 定 where(目标去哪,路网驱动)· 语言定 which(哪辆是它,身份)· VLA 定 how(UAV 怎么飞)。** 三者正交,互不稀释(见 §7 的 H0 非放大 guard)。

---

## 2. 为什么用路网、不用速度(重构动机)

| 速度外推的问题(用户指出) | 路网预测如何规避 |
|---|---|
| UAV 自身运动 → 两运动物体多帧测速难 | 不测速;只需**最后可见位置 snap 到车道** |
| 目标非匀速 → 瞬时速度无意义 | 不用瞬时速度;沿**车道拓扑 rollout**(离散遍历) |
| 15s 恒速必然漂 | 路网把目标约束在**1D 车道流形**,15s 也只在有限走廊内 |
| 需精确未来点 | **不需要**:飞到**路网走廊 + 扫描**重捕获,纵向不确定性被吸收 |

---

## 3. 文献落点(velocity-free 路网预测是主流)

| 方法 | 贡献 | 我们借用 |
|---|---|---|
| **PGP · Lane-Graph Traversals**(Deo 2021)★ | 图编码 + **离散策略采样车道图路径遍历** + 沿路径解码;横向(分叉)由遍历策略、纵向由 latent。nuScenes SOTA | **核心方法学**:遍历=选走哪条路(无速度);latent=走多远 |
| **DenseTNT / GoRela / MTR** | 沿**可达车道**采样目标点、过滤不可达 → 目标=意图 | 可达走廊 / 目标点表示 |
| **VectorNet / LaneGCN** | 车道折线→图,注意力/图卷积聚合(LaneGCN 6s 1.90→1.35m) | 车道图编码骨干 |
| **Scene Informer**(遮挡推理) | 部分可观测下 Transformer 推理**被遮挡** agent 轨迹 | 直接对应"预测看不见的目标" |
| **Prior-Based Online Lane Graph(单目)/ MapTRv2** | 单图→BEV 车道图提取 | 推理时无外部地图也能推路网(空中俯视更易) |

引用见本轮 deep-research(arxiv 2106.15004 / DenseTNT ICCV21 / 2309.13893 / 2307.13344)。

---

## 4. RoadGraphPredictor 模块设计

### 4.1 输入 / 输出
```
输入:  last_seen_pose   最后可见目标 3D 位置 + 朝向(来自 tid 选中候选的框+深度反投影)
        lane_graph       局部车道图(节点=车道段/中心线折线, 边=前驱/后继/相邻)
        lang_intent      (可选)语言意图 embedding(如"右转")
        latent z         (可选)纵向变化采样
输出:  corridor          可达走廊 = 一组沿路的候选未来位置(近→远, 多分叉)
        或 target_wp      折算成 K 个未来目标位置(喂 EAR 的 waypoint 位)
        conf             各分支/位置的概率(不确定性 → 兜底)
```

### 4.2 车道图编码(VectorNet 式)
- 每条车道中心线 → 折线 → 向量集 → 局部子图;节点=向量,边=空序/拓扑;
- 注意力聚合 → 节点编码 `h_node`。训练用 CARLA `carla.Map` 拓扑 GT;推理可用 GT 或俯视车道检测。

### 4.3 目标锚定
- `last_seen_pose` → snap 到最近车道节点 + 朝向对齐 → **源节点 s0**。

### 4.4 遍历策略(velocity-free 核心)
- 从 s0 沿车道图**前向采样路径遍历**(PGP 式离散策略,行为克隆学"车会怎么走");
- **横向多模态**(路口选哪个分叉)由策略天然表达;**不涉及速度**。

### 4.5 可达走廊 / 位置分布
- 遍历路径 = 沿路 **1D 可达走廊**;
- 位置分布 = 走廊上近→远的采样(纵向"多远"用 latent z 粗略表达,或近/中/远分箱);
- **输出不是精确点,是走廊 + 概率** → 配合"飞到走廊+扫描"。

### 4.6 语言 which-way seed(H8,可选增强)
- 路口分叉处,`lang_intent`("右转")→ 偏置遍历策略的分支概率;
- 这是**语言的第二功能**(除身份 which 外的意图 which-way);必须过"意图冗余 guard"(仅在分叉歧义场景有净贡献)。

### 4.7 坐标系
- 车道图 + 预测走廊在**世界系**(目标独立于 UAV 运动);
- 喂 EAR 前用当前相机位姿转**相机系**(EAR 现用相机系航点)→ 天然 ego-motion 补偿。

---

## 5. 与 ACoT-VLA 的接口

**插入点不变**(观测→EAR 之间),内部从"速度外推"换成"路网遍历":
```
当前帧 → VLM → vlm_ctx ─┬─────────────────────────────→ tid(WHICH)
                        │
      last_seen + 路网 → [WM: RoadGraphPredictor] → corridor/target_wp (WHERE)
                        │
                        └─→ EAR(cond = vlm_ctx + corridor) → z_ex → DiT(HOW)→ 动作
```
- **EAR 的 waypoint 输入**:可见期用 EAR 反应式短时预测;丢失期由 **WM 的走廊**接管长时预测(或直接令 `z_ex = corridor`);
- WM 输出也可拼给 DiT / tid(重现处"目标应在此走廊"辅助认领);
- **EAR 本体不改**,只是 cond/z_ex 的来源在丢失期切到 WM。

---

## 6. 训练与监督
| 信号 | 监督 |
|---|---|
| 遍历策略 | 行为克隆:对齐目标**真实走过的车道路径**(CARLA GT 轨迹 snap 到车道) |
| 走廊/目标位置 | 对**目标 GT 3D**(丢失期仍有效)的可达位置分布损失(可达集内) |
| latent z(纵向) | 目标 GT 沿路进度;或 CVAE 式 |
| 拦截动作 | `supervise_intercept`(已实现):专家直飞拦截动作 |
- **需序列窗口**(覆盖 可见→丢失 过渡),但**不需精确速度**;遍历是拓扑量。

---

## 7. 风险与 gap
| 风险 | 缓解 |
|---|---|
| 纵向"多远"仍需粗略进度 | latent / 近-远分箱;飞到走廊+扫描吸收 |
| 路口分叉多模态 | 遍历策略天然多模态;语言 which-way seed;否则多走廊 |
| 车道图提取可靠性(视觉推时) | 训练/评测用 CARLA GT;论文范围充分 |
| 目标驶离路网 | 车基本在路上;边缘退回 search_mode |
| **H0 非放大 guard** | **SEP/WM identity-agnostic**(只用路网+位置,不用语言身份)→ 加 WM 后 w/o-lang 增幅**不缩小**(g2);语言仅管 which + which-way |

---

## 8. 数据需求(给数据 agent, Phase 1 附加)
- CARLA **车道图 GT**(`carla.Map` 拓扑/waypoint)随 episode 导出;
- 目标**逐帧 lane-anchor + heading**(snap 到车道);
- 长丢失段目标**真实走过的车道路径**(遍历策略监督);
- 其余同 data-fix-spec §11(长丢失+拦截+重现共视)。

---

## 9. 实验(对齐 experiment-design)
- **H7**:WM(SEP)在**长丢失 minFDE@{6,10}s** 显著优于反应式 EAR-only,且增益集中在**长丢失**(>6s);
- **g2 非放大 guard**:开启 WM 下重跑 E1/E5,w/o-language 增幅**不缩小**(证 where/which 分工);
- **H8 which-way**:三臂(无语言/仅消歧/消歧+意图)在**分叉歧义**场景,意图语言有净贡献 + 冗余 guard;
- **重捕获**:锁对率(硬失败)、收敛时间、按丢失时长分层。

---

## 10. 分级落地
```
P2a  RoadGraphPredictor 最小版:CARLA GT 车道图 + 单步遍历 + 走廊输出 → 接 EAR
     验证: 长丢失 minFDE 优于反应式(H7 最小可证)
P2b  PGP 式多步遍历策略 + latent(纵向)+ 多模态分叉
P2c  语言 which-way seed(H8)+ 视觉车道图提取(去 GT 依赖)
```
**先 P2a**:CARLA 有现成路网 GT,最小版即可验证"路网预测能否救回长丢失"——这是 WM 的最小可证命题。
