# Stage-3 RL 设计 — 用 RL 修「控制取景」闭合 SR gap

> 2026-09-04 · 建立在 training-plan Phase E / experiment-design §H6 的 GRPO 骨架上,并被本轮 SR 诊断精确化(memory `acot-uav-reid-reframe`、`acot-uav-stage3-rl`)。

---

## 0. 诊断→目标(为什么是 RL、修什么)
- 真上界 **0.58**(D1 完美取景 / D2 完美认车都到);可部署 **0.105** 的 gap = **认车×控制的恶性级联**;
- 主导边 = **控制把目标框偏**(centering 0.02–0.11),而这是 **decorrelate off-center 去捷径设计**的产物(DiT 学会了偏心取景);
- 认车(reid)**不是瓶颈**:reid/warm-start 能改善认车但**不提 SR**(warm-start 甚至 0.125→0);frame-gain(盲目居中承诺目标)**放大认车错**。
- **∴ RL 目标 = 微调 DiT,使推理时把「reid 认定的真目标」框正**,靠**完整观测(视觉+z_ex+reid置信)学会条件化居中 + 视觉重检**,打破级联——这是固定增益 frame-gain 做不到的。

## 1. 能否突破 0.58(诚实)
- 0.58 = 可解 episode 比例;~42% 是**场景难度地板**(超长遮挡,oracle 也失败);
- RL 修取景 → 收回"级联本可解却丢的"→ **0.105 → ~0.4–0.58(逼近)**;
- **不超越 0.58**:那需修 42% 地板(WM 预测拦截 / 主动搜索 / 分辨率 / 易场景)。RL 的 R_reacquire 可能微顶,但地板是信息硬限。

## 2. 观测空间
复用 DiT 现有条件 + 新增"目标在画面哪 + 认车置信":
| 观测 | 维度 | 来源 | 作用 |
|---|---|---|---|
| z_ex | K×3 | reid 认定目标 CV 投相机系 | 去向先验 |
| z_im | n_im×d | IAR(冻结) | 视觉理解 |
| vlm_ctx | M×C | Qwen(冻结) | 场景/**视觉重检依据** |
| proprio | 5 | 无人机 | 本体 |
| **★ tgt_uv** | 3 | committed 目标投影 (u/W, v/H, depth/D) | **该把谁拉向中心** |
| **★ reid_conf** | 1 | reid top1−top2 margin | **低置信→保守/搜索** |
> 新增两项是 frame-gain 缺的条件信号:让策略学"高置信→居中、低→保持/搜索"。

## 3. 动作空间(不变)
DiT 现有输出:`(dx,dy,dz,dyaw)` 连续速度/偏航 + `search_mode`∈{0,1}。flow-matching 采样天然给随机策略(GRPO 采多条)。

## 4. 奖励(逐帧;GT 只进奖励不进观测→无作弊)
```
R = w_c·R_center + w_t·R_track + w_id·R_correct_id + w_r·R_reacquire − 惩罚
```
| 项 | 公式 | 作用 |
|---|---|---|
| **R_center** | 目标在框内: `exp(−‖(u,v)_真目标 − 中心‖² / σ²)`,σ≈0.25·W | **修取景核心**;奖励**真目标**居中 |
| R_track | 真目标在框 +1;出画 0 | 别丢 |
| **R_correct_id** | committed=真目标 +1;锁干扰车 **−2** | **防 hacking 崩成跟任意车(最大风险)** |
| R_reacquire | 丢失后 ≤T_reacq 内重捕获真目标 +B | 治长丢失 |
| 惩罚 | 出画≥5s、承诺错车≥2s 的持续时长 | 对齐 SR 判据 |
> **关键**:R_center 奖励**真目标(GT)居中**,不是"任意车居中" → 学的是"把**被指认的**那辆框正",**不重引入"跟最居中车"的捷径**。GT 不进观测 → 策略必须用可用输入(视觉/z_ex/reid)学会**从认车错误里视觉重检真目标**,这是打破级联的机制。

## 5. 去捷径 ↔ 居中 张力的解法(能成的关键)
```
训练数据 off-center(decorrelate) → VLM/reid 特征保持 identity load-bearing(H0 不破)
   ⊕
RL 只调控制(DiT) → 把 reid 认定的真目标居中,不碰数据/特征
```
两者**解耦**:去捷径=数据侧(身份可学);RL 居中=控制侧(推理框好);RL 居中的是 **identity-conditioned 真目标** ≠ identity-free "最居中车" → 不重引入捷径。这是 D1(专家居中真目标→0.58)的可部署化。

## 6. 训练设置
- **冻结** VLM/EAR/IAR;**只训 DiT 末 4 层或 LoRA**(保感知/身份,只调控制);
- **GRPO**:每状态采 K 条 rollout,`reward−组均值`作相对优势,无 critic(契合扩散采样);
- **reid 在环**:rollout 时 reid 提供 committed 身份 → z_ex/obs;奖励用 GT 核验;
- **课程**:短丢失/少干扰 → 长丢失/多 look-alike,提样本效率。

## 7. GRPO 伪代码
```python
for iter in range(N):
  s = env.reset()
  for t in steps:
    group = [dit.sample(obs_s) for _ in range(K)]        # K 条动作(flow 采样)
    rewards = [rollout_short(env.copy(), a) for a in group]  # 或单步 reward + bootstrap
    adv = [r - mean(rewards) for r in rewards]           # 组内相对优势(无 critic)
    loss = -sum(adv_k * logp(a_k | obs_s)) + β·KL(dit, dit_ref)  # KL 锚 BC 防漂移
    opt.step(loss)   # 只更新 DiT 末层/LoRA
```

## 8. 里程碑门
| M | 内容 | 门 |
|---|---|---|
| **E0(降风险,先做)** | 只加 R_center + LoRA 训 DiT 末层 | **centering↑(0.02→0.3+)且 mis 不升、SR 不降**(无 hacking) |
| E1 | +R_correct_id +R_reacquire 完整 reward | SR 与 mis_follow **均优于 BC**、无 hacking |
| E2 | 课程 + 调 K/β/权重 | 可部署 SR → 逼近 0.58 |

## 9. 硬前置与风险
1. **⚠️ harness 稳定性 = 硬阻塞**:RL 要连续数小时 on-policy rollout,而 cyh-carla 容器**每 ~20 分钟被 teardown 删除** → RL 跑不起来。**必须先解决**(独立 CARLA 部署 / 防清理 / 稳定机器 / DETACH 训练容器 + checkpoint)。
2. 在线 VLM 成本(每 S2 帧跑 Qwen,rollout 慢)→ 缓存/降频/小 VLM。
3. reward hacking → R_correct_id + 全程监控 mis_follow。
4. reid 置信信号可靠性 → 否则学不到"低置信保守"。

## 10. 关联
memory `acot-uav-reid-reframe`(诊断全链)、`acot-uav-stage3-rl`(原设计)、`acot-uav-harness`(rollout harness + teardown gotcha);Phase E(training-plan)、§H6(experiment-design)。
