# 外部 Baseline 对比 · 进度追踪

> 2026-09-05 起 · 用于 ACoT-UAV-Track 论文的外部对照工作(审计缺口 #4)· 配套 memory `acot-uav-paper-writing-kickoff`、`acot-uav-track-positioning` · 主结果/负结果 20ep batch 见 `runs/paper_batch.log`

---

## 0. 目标与原则

**为什么需要**:当前所有实验臂都是**同一 ckpt 的内部消融**,无外部方法对照 → benchmark 论文的常见审稿要求。补 ≥1 外部 baseline 强化说服力(冲顶会需要,D&B/CoRL/RA-L 加分)。

**核心原则(决定实验设计)**:harness 已支持切换 WHICH 模块(`policy.py tid_head` ∈ {xattn,reid,oracle,...})。**最严谨且可行的对比 = 把外部方法作为替代 `tid_head` 接入同一闭环(WHERE=CV、HOW=DiT 固定)**,只换身份机制 → apples-to-apples,隔离贡献。比对比整套异构系统更干净。

**评测协议**:与主 batch 同协议——20ep、seeds 91000-19、Town05、STEPS 900、s2=6、live CARLA;产出 `eval_out/paper_ext_<name>.json`;用 `rollout/continuous_metrics.py` 出客观指标表(正确跟踪占比 / 最长错跟时长 / 重捕获率&延迟 / id_switches / latch@Ts)。

---

## 1. ⚠️ 诚实警示(引用核验)

2026 年 UAV 预印本**须逐篇二次核验存在性/代码**;memory 已核出 **CosFly-VLA 是 fabricated → 禁用**。搜索摘要模型可能 confabulate,**未经核验不写进论文、不作 baseline**。

---

## 2. 候选 baseline 核验状态

| 方法 | venue | 代码 | 真实性 | 对我们的角色 | Tier |
|---|---|---|---|---|---|
| **OC-SORT / ByteTrack / DeepSORT** | ICCV/ECCV'21-22 | 成熟公开 | ✅ 确认 | **时序 EMA 的天然对照**(运动+外观关联) | 1 |
| **DAM4SAM** | CVPR'25 | [dam4sam](https://github.com/jovanavidenovic/dam4sam) | ✅ 确认 | SOTA **distractor-aware** 记忆跟踪(免训练 drop-in)→ 外观记忆强对照 | 1 |
| **iKUN / TransRMOT** | CVPR'23-24 | [RMOT](https://github.com/wudongming97/RMOT) / [TempRMOT](https://github.com/zyn213/TempRMOT) | ✅ 确认 | referring-MOT:语言型 WHICH 对照 | 1 |
| **JointNLT / UVLTrack** | CVPR'23 / AAAI'24 | [JointNLT](https://github.com/lizhou-cs/JointNLT) / [UVLTrack](https://github.com/OpenSpaceAI/UVLTrack) | ✅ 确认 | 现成 NL-tracker(被动感知)对照 | 1 |
| **TrackVLA / OmTrackVLA / TrackVLA++** | CoRL'25 | [TrackVLA](https://github.com/wsakobe/TrackVLA) / [OmTrackVLA](https://github.com/om-ai-lab/OmTrackVLA) | ✅ 确认(地面/人) | 全闭环 VLA 对照;迁空中 CARLA = 大工程 | 2 |
| **AerialMind / HETrack** | AAAI-26(2511.21053) | 待查 | ⚠️ 须核验 | 最接近的**已发表** UAV referring-MOT(被动) | 3 |
| UAV-Track VLA / DeTrack | 2026 预印本 | 待查 | ⚠️ 须核验 | UAV 域竞品 | 3 |
| ~~CosFly-VLA~~ | — | — | ❌ **fabricated 禁用** | — | — |

---

## 3. 推荐最小套件(性价比最优)——三点式 WHICH 对照

全部接入同一闭环(同 WHERE+HOW),配合内部阶梯(xattn→vlmpool→DINO→conf→temporal):

1. **标准时序关联(OC-SORT 式)** —— 时序 EMA 的**直接对照**;成本最低;**必做**。
2. **DAM4SAM** —— 证明"轻量 crop-DINOv2+时序EMA 对得起 SOTA 记忆跟踪";**强对照**。
3. **iKUN / JointNLT(语言型 WHICH)** —— 与 xattn 并列,支撑"单帧语言 grounding 不足→时序外观是杠杆"的**发现叙事**。

> TrackVLA 全闭环对照 = 冲顶会加码 / future work;AerialMind 核验后若可用 → 补为"已发表 UAV 竞品"。

---

## 4. 进度清单(勾选追踪)

**集成层**
- [ ] `tid_head=assoc`:OC-SORT 式运动+外观关联,接入 `policy.py`(与 reid_tavg 同插槽),`--assoc-*` 参数
- [ ] `tid_head=dam4sam`:DAM4SAM 作 tracker,给出 committed 目标(引入 repo+权重)
- [ ] `tid_head=refmot`:iKUN/TransRMOT referring 头,候选框+语言打分
- [ ] (可选)JointNLT/UVLTrack 被动感知对照(离线在 episode 帧上跑)
- [ ] (加码)TrackVLA/OmTrackVLA 全闭环适配调研

**核验层**
- [ ] AerialMind(2511.21053)代码/权重是否公开、许可、可跑
- [ ] UAV-Track VLA / DeTrack 逐篇核验(存在性/代码)

**评测层**
- [ ] ext baseline 各臂 20ep(seeds 91000-19)→ `eval_out/paper_ext_<name>.json`
- [ ] `continuous_metrics.py` 出"内部阶梯 + 外部 baseline"合并 WHICH 对比表
- [ ] 定稿数字写回 memory `acot-uav-paper-writing-kickoff`

---

## 5. 集成设计草案

**`tid_head=assoc`(OC-SORT 式,零外部依赖,先做)**:候选已有(`cset.cands`,稳定 actor idx + depth + 投影框)。维护目标轨迹的 Kalman 状态(图像位/尺度)+ 外观(crop-DINOv2 特征可复用);每 tick 用"运动门(IoU/马氏距) + 外观相似"匹配打分,匈牙利/贪心关联,committed = 关联到目标轨迹的候选。与 reid 公平:模板同样从 GT-可见目标建、SELECT 用过去、更新用当前(无重现泄漏)。→ 与时序 EMA 只差"关联规则(运动+外观 vs 纯外观时序积分)"。

**`tid_head=dam4sam`**:用 DAM4SAM 维护目标 mask/记忆;每 tick 把其预测目标框与候选框做 IoU/中心距匹配 → committed slot。需 SAM2 权重 + DAM4SAM repo(容器内 pip/clone)。

**`tid_head=refmot`**:iKUN 两阶段——候选 proposal(我们已有)+ 语言匹配打分;直接替换 tid_logits。

---

## 6. 参考链接

- RMOT/TransRMOT https://github.com/wudongming97/RMOT · TempRMOT https://github.com/zyn213/TempRMOT
- JointNLT https://github.com/lizhou-cs/JointNLT · UVLTrack https://github.com/OpenSpaceAI/UVLTrack · TNL2K toolkit https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit
- DAM4SAM https://github.com/jovanavidenovic/dam4sam
- TrackVLA https://github.com/wsakobe/TrackVLA · OmTrackVLA https://github.com/om-ai-lab/OmTrackVLA · TrackVLA++ https://pku-epic.github.io/TrackVLA-plus-plus-Web/
- AerialMind https://arxiv.org/abs/2511.21053
