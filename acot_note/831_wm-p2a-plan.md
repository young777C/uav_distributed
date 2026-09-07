# WM P2a 最小实现落地计划

> 2026-08-31 · 项目核心创新(VLA+WM 出画预测)的最小落地 · 设计见 `related-survey/wm-roadgraph-design.md`、memory `acot-uav-wm-roadgraph`
> 方法论:**复刻 CV 基线的成功路径 —— 几何先行(不学习)证价值 → 再上闭环 → 最后才学习分支**。每步便宜可测。

---

## 0. 目标与假设(论文正向脊梁)

**核心假设 H7**:目标出画后,**沿车道结构预测(road-traversal)在"转向/路口"上显著优于匀速直线外推(CV)**。

**已测清的前提(诊断阶段给 WM 铺好的论证)**:
- CV 直路够用(10–23m)、**转向崩(47m)** → WM 的**价值区精确=转向/路口**;
- 反应式单帧无法预测出画(58–90m)→ 需要预测;
- 记忆+CV 先验闭环有效(track_s 22→63)→ 注入口(z_ex)已验证。

**WM 要超越的基线(已有)**:CV(`baseline_deadreckon` / `target_state.py`)。

---

## 1. 最小算法:road-traversal(几何版,零学习)

CV 的错:目标在弯道,CV 沿**上一刻速度直线**外推 → 冲出车道。
WM 的修:**把目标吸附到车道 + 沿车道弧长前进**(跟着路弯),而非直线。

```
输入:最后可见目标世界位 p0、世界速度 v0(estimator 已有)、CARLA 车道图
  wp0   = map.get_waypoint(p0)                 # 吸附到所在车道
  speed = |v0|
  for t in {1,2,4,6}s:
      arc   = speed * (Δloss + t)              # 沿车道弧长(含丢失已持续时长)
      wp_t  = traverse(wp0, arc)               # wp.next(arc);路口选分支
      p_pred(t) = wp_t.transform.location      # 车道上的预测点
  分支选择(P2a-0 几何): 选朝向最贴合 v0 的后继(最"直行"的续接)
```
→ 输出 (K,3) 预测世界点 → 投到当前相机系 /scale = **z_ex**(与 CV 先验同一个槽)。

**这是最小 WM**:只把"直线"换成"沿车道",不引入任何学习;若它就打赢 CV,说明**路网结构本身**是关键(和"CV 打赢 EAR"同一逻辑)。

---

## 2. 数据侧前置(data agent · A 类纯补标注,不重渲染)

用已存的 `target/tx,ty,tz` + CARLA 地图查询,**每帧补三样**(几秒/集 CPU):
| 字段 | 计算 | 用途 |
|---|---|---|
| `target_is_junction` | `map.get_waypoint(p).is_junction` | 转向分层(评测) |
| `target_lane_id` / `road_id` | `map.get_waypoint(p)` | 车道识别 |
| **`road_pred_{1,2,4,6}s`** (T,K,3) | 从**该帧目标位**按其速度沿车道 traverse(路口选最直行) | **离线 road-FDE 评测的预测源**(免 host 无 CARLA) |

> 说明:离线评测(`baseline_deadreckon`)在 host 跑、无 CARLA,所以 **road 预测必须由数据 agent 用 CARLA 预计算并存 h5**;闭环则由 env 侧在线算(有 `carla.Map`)。

**并行(B 类,新增集)**:P1 转向长丢失数据(5–15s 丢失**跨路口/弯道** + 目标重现 + 重现处 ≥2 look-alike),用于评测稳与后续训练。现有转向丢失帧仅 n=20–27,**不够**。

---

## 3. 模型侧落地(三级,便宜→贵)

### P2a-0 · 几何 road-FDE(离线,证 H7)— **最便宜、最先做**
- 给 `train/baseline_deadreckon.py` 加 **`road` 模式**:读 h5 的 `road_pred_*`,算 FDE,和 `cv`/`zerovel` 同表、按**丢失时长 × 转向**分层;
- **门 M2(核心)**:**转向丢失帧上 road-FDE < CV-FDE**(尤其 4/6s)。过 → H7 几何成立,WM 有据(无需学习)。
- 成本:低(数据 agent 出 `road_pred` 后,我加一个读取分支)。

### P2a-1 · 闭环注入(证转向重捕获)
- 加 `ex_source=road`:**env 侧**(有 `carla.Map` + 最后可见目标)在线算 road-traversal 预测世界点 → 经 wire 发给 policy → policy 投相机系 → z_ex(复用 `target_state.predict_wp` 的变换,只换预测源);
- **门 M3**:闭环**转向重捕获率** road > cv_gated > ear(按丢失时长×转向分层报);
- 成本:中(env 侧加 road 预测 + wire 字段 + policy 分支;≈ 之前 `track`/`cv_gated` 的接法)。

### P2a-2 · 学习分支/走廊(仅当几何不够)
- 路口分支从"最直行"启发式 → **学习分支选择器**:复用 **IAR maneuver 头 → 分支 seed**、或**语言 which-way**(H8);
- 走廊损失(预测车道中心线 vs CARLA `Waypoint` GT);Stage-1b WM 预热(车道遍历 BC + 走廊)。
- 成本:中-高(仅在 P2a-0/1 证明"结构有用但启发式分支不够"时才做)。

---

## 4. 评测指标(= 论文的正向结果,不依赖 SR)

| 层 | 指标 | 阶梯 |
|---|---|---|
| 离线 | 转向丢失帧 **minFDE@{4,6,10}s** | reactive-EAR > CV > **road-WM** |
| 闭环 | **转向重捕获成功率** / 收敛时间 / lock-error | ear < cv_gated < **road** |
| 分层 | 按 **丢失时长 × 转向/直行** | road 增益应**集中在长丢失+转向**(其它格 ≈ CV) |

> **关键**:论文报**"转向重捕获率"这类可动指标**,不被 SR=0 绑死(SR 作诚实的硬边界报)。road-WM 在转向上打赢 CV = 正向方法贡献。

---

## 4.5 结果 · M2 离线门 **已通过**(2026-08-31,`runs/ladder_road_0831_0818.log`)

数据 agent 交付 `annotation/{road_pred,target_is_junction,target_lane_id}`(66/66 集,QC:road 1s 预测 vs 真实中位 0.68m)。`baseline_deadreckon --ckpt stage2_v5_fair`(EAR 上界)出完整阶梯。

**⚠️ 诚实拆解(两条不同输入基准的轴,不可混成一条 ladder):**
- `road_pred` 由数据 agent 从**每帧目标真实位**算 → `road` 与 `cv_frame` **同为 oracle-当前**;
- M2 门(road vs cv_frame)是干净的**机制隔离**(跟车道 vs 直线),但是**上界**,非可部署增益。

**轴 A · 可部署 dead-reckon(从最后可见位)**:long_turn @6s ear 60 → cv 47.5 —— 记忆+速度 >> 反应式单帧,但转向都退化(最后可见位过时)。**cv→cv_frame 的暴跌(47→12.5)是 oracle-当前位效应,不是 WM。**

**轴 B · 机制隔离(都 oracle-当前,唯一差别=车道 vs 直线)= WM 的价值(H7)**:
| 分层 | n | cv_frame@6s | road@6s | Δ% | 
|---|---|---|---|---|
| loss_junction | 102 | 9.3 | **4.1** | −56% |
| vis_junction | 1064 | 11.4 | **7.7** | −32% |
| vis_turn(tyaw) | 278 | 22.9 | **13.1** | −43% |
| short_turn(tyaw) | 20 | 19.6 | **10.3** | −47% |
| long_turn(tyaw) | 27 | 12.5 | **10.1** | −19% |

→ 同起点下沿车道预测在转向/路口稳定优于直线 **20–56%,随视界增长**;直路 road≈cv_frame(无副作用);深度维尤其明显(short_turn road 深度 1.8 vs cv_frame 5.9 @6s)。tyaw 与 is_junction 两套分层互证。

**结论**:**H7 机制成立**(给定当前位,跟车道>走直线)。可部署的 road(从最后可见位吸附遍历)落在 cv(47)与 oracle-road(10)之间 → **P2a-1 闭环要证的正是这个可部署增益**。

## 4.6 结果 · M3 闭环门 **未过**(2026-08-31,`eval_out/road_{ear,cv_gated,road}.json`)

P2a-1 接通(`road_predictor.py` env 侧 lane-traversal → wire → z_ex;road_pred **确认全程在用**,19 集 4814 loss 帧,均值 253/集)。20 集 Town05 seed91000 配对:

| 指标 | ear | cv_gated | **road** |
|---|---|---|---|
| track_s | 24.3 | **64.7** | 49.2 ⬇ |
| reacquire_success | 0.42 | **0.55** | 0.46 ⬇ |
| convergence_s | 5.77 | 3.45 | **3.28** |
| SR | 0 | 0 | 0 |

**可部署 road 总体不如直线 CV**,且**用得越多越差**(高丢失集 Δtrack −20.7s vs 低 −13.6s),两次灾难(road 把无人机开到错误的路,track 崩到 5–12s、reacS 0.00)。

**机理**:CV 长直路错误="沿同一线走太远"(可恢复);road 路口**最直分支**选错="自信在另一条街"(不可恢复)。offline M2 优势是 **oracle-当前**;可部署 last-seen→长视界遍历使**分支选择**成瓶颈,恰在 road 本应发力的路口。

**结论**:**几何 road 结构只在给对分支时有用** → **P2a-2 学习分支 从"可选"变"必需"**;或先试**便宜的路口置信门**(弧内无路口才用 road,否则回退 CV)。阶梯:ear < road < **cv_gated**(CV 仍是闭环基线)。诚实价值:证明 offline 几何优势**不迁移**闭环,并定位根因=分支歧义。

## 4.7 结果 · P2a-2 正对照 **决定性负面 → which-way 头不值得建**(2026-08-31)

M3 未过后,先跑**正对照**再决定是否投学习分支(护栏:先定上界)。`--road-oracle`=env 用**真实当前位每帧刷新**锚定 road(去 last-seen 陈旧、弧最短、分支歧义最小),界定 road 的**价值上界**。4 臂 20 集 Town05 配对:

| 臂 | track_s | reacq | vs cv 显著? |
|---|---|---|---|
| ear | 24.3 | 0.42 | — |
| **cv_gated** | **64.7** | **0.55** | 基线 |
| road(可部署) | 49.2 | 0.46 | −15.5s **显著更差** |
| **road_oracle(完美锚点)** | 58.1 | 0.48 | −6.6s [−16.7,+2.4] **不显著≈CV** |

**结论**:即使完美锚点,road 也只 ≈CV、**并未超越** → 学习 which-way 头最多让可部署 road 逼近 oracle-road(58)→ **仍 ≤ cv_gated(65)**,**头的最好情况都打不过 CV** → **不该建**。

**闭合 WM-as-z_ex-prior 这条线**:road 的 offline FDE 优势(转向 road<CV,oracle-当前)**不迁移闭环**;近 oracle 的 WHERE 也不抬 track/reacq,因**闭环被 WHICH(grounding)卡,不是 WHERE**(SR=0、lock_wrong~0.74 四臂一致)。**CV 是更简单且更好的闭环 WHERE 先验。**

**诚实 caveat(未测)**:road 注入的是**用 EAR z_ex 训练的 DiT**(零样本/OOD);DiT **重训**(P2/P3 三阶段)后能否更好用 road z_ex 未测——但 CV 同为训练无关却更好,且近 oracle WHERE≈CV,故重训 DiT 抬 grounding-gated 的 SR 希望不大。

**重定向**:不建 which-way 头;把"offline 几何 WHERE 优势 ⇏ 闭环增益;SR 瓶颈是 WHICH 非 WHERE"作为诚实结论,SR 工作回到 grounding(在线鲁棒训练 / 分辨率-边缘特征,见 `research-bottlenecks.md ①`)。

## 5. 里程碑 / 门 · 关键路径

```
M0 数据: is_junction + road_pred + lane_id 补标注(data agent) —— 前置
   └─ M1 数据: P1 转向长丢失数据(新增集,并行)
M2 离线门(核心/便宜): road-FDE < CV-FDE @转向  ← 过则 H7 几何成立
   └─ M3 闭环门: ex_source=road 转向重捕获 > cv_gated
        └─ M4(可选): 学习分支/走廊(几何不够时)
```
**关键路径**:`M0 数据(road_pred)→ M2 离线几何门(便宜,先证价值)→ M3 闭环 → [M4 学习]`。
**先做 M0+M2**:最便宜地证"路网结构在转向上有用",再决定是否投 M3/M4。

---

## 6. 工作量与风险

| 项 | 成本 | 风险 |
|---|---|---|
| M0 road_pred 补标注 | 低(数据侧,几秒/集) | traverse 分支/弧长实现细节 |
| M2 离线 road 模式 | 低(加读取分支) | — |
| M3 闭环 ex_source=road | 中(env 预测 + wire + policy) | env 侧 CARLA 遍历、丢失时长弧长 |
| M4 学习分支 | 中-高 | 仅按需 |

**主风险**:① road_pred 的"选哪条分支"在 P2a-0 用启发式(最直行)——若目标常转弯到非直行分支,几何版可能不够 → 落到 M4 学习分支(但 M2 会先告诉我们够不够);② P1 转向数据未到前,评测样本少(n~27)→ 先小样本出方向,P1 到位再稳。

**诚实边界**:P2a 证的是"**WM 在转向重捕获上 > CV**"(正向方法),不是"SR 翻"。SR 仍受 grounding/取景约束(见 `research-bottlenecks.md`),但**转向重捕获的提升 + 语言认领 + 诊断分析**足以构成论文正向脊梁。

---

## 附:复用的现成件
- 注入口 z_ex + `ex_source` 门控:`rollout/policy.py`(track/cv_gated 已通);
- CV 基线 + 分层评测:`train/baseline_deadreckon.py`、`train/intercept_eval.py`;
- 估计器(last-seen + 速度):`rollout/target_state.py`;
- 两容器 wire:`rollout/{remote_policy,policy_server,env}.py`(gt_candidates 已在传,加 road_pred 同法);
- 转向分层字段:`dataset_stage2` 的 `target_px`/`target_central`(可加 `is_junction`)。
