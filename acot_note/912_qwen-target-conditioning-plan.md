# Qwen3-VL 弱项改进 · 目标条件化特征调制 · 尝试方案(2026-09-12)

> 承接：路线 B 已把手工 WHICH 栈蒸馏为 TAHRelM（`907_model-integration-907.md §0.5`，oracle-memory 关联质量全场最优；deployable 受 confident-drift 阻）。本方案针对**上游 vision 判别**——利用"离线不知目标、任务开始后知目标"的信息不对称，对通用模型做**测试时目标特化**。严格遵守纪律：**先 oracle 探针定 headroom，再建**。

## 0. 弱项定位（上轮确认）
Qwen3-VL 只做**粗粒度语言/in-context 条件化**，**不做 look-alike 实例级判别、不做目标在线特化**。我们系统判别实际跑在 **crop-DINOv2**（reid/TAH），Qwen 负责语言 grounding + 场景 context。

## 1. AWQ 类比的修正（为什么不是"放大权重"）
AWQ 的 salient-weight scaling 是**量化下保精度的等价变换**（放大权重↔缩小激活，全精度输出不变）；**全精度直接放大权重 = no-op 或破坏网络**。可迁移的只有"用输入/激活统计**识别** salient 通道"。全精度里正确的"强调"是**门控/调制特征**（FiLM/adapter/hypernet），不是缩放权重。→ 本方案 = **目标条件化特征调制**。谱系成熟（Siamese/DiMP/FiLM/adapter/LoRA），新在**冻结 VLM 的 look-alike 跟踪**设定。

## 2. 两条路（探针决定走哪条）
- **路 α**：让 Qwen 自己变实例判别（调制 Qwen vision 特征）——雄心，"修 Qwen"。
- **路 β**：在判别主力 **crop-DINOv2** 上做目标条件化调制——稳，增益直接落 WHICH。

## 3. Tier 0 · 探针（数小时，零/极少训练，决策级）

### 探针 B（先做，最决定性）：oracle 特征条件化 headroom
在**已缓存 crop-DINOv2 特征**（`runs/tah_cache.pt`，64ep）上，施加 **oracle 逐通道加权**（Fisher-style：用真标签算"目标 vs 干扰差异最大的通道"），量 look-alike 分离度增益。
- **对角 oracle = FiLM-γ 调制的上界**（FiLM 的 scale 就是逐通道对角加权）。
- 指标：目标-最难干扰的 cos margin；加权匹配的 pick 准确率（1−mis_follow）。
- **Go**：分离度显著涨 → 判别信息**存在但被平均掉** → 建 FiLM 调制值得。
- **No-go**：不涨 → **撞分辨率/特征天花板**（与 cos≈0.614、12→27px 旧结论一致）→ 别建调制，转**高分辨率渲染/更强 backbone**。
- 工具：`carla_uav_tracking/rollout/probe_target_conditioning.py`（离线、CPU、复用缓存）。

**结果（2026-09-12，64ep/41262帧）**：

| 方案 | in-sample oracle（乐观上界） | deployable（首30帧拟合→其余帧评估） |
|---|---|---|
| baseline（均匀 cosine） | 0.532 | 0.363 |
| diag（FiLM-γ） | **0.644 (+0.111)** | 0.388 (**+0.025**) |
| LDA（full adapter） | **0.878 (+0.346)** | 0.421 (**+0.058**) |

**解读（诚实）**：①**不是纯分辨率天花板**——in-sample LDA 0.878 证明判别信息**线性可分地存在**于 DINOv2 特征，被均匀 cosine 平均掉了；②**但朴素在线估计（首30帧拟合）几乎无效**（+0.025/+0.058）——短 warmup 过拟合起始姿态/光照/当时干扰，不泛化到后续；③**学习版（Tier-1 摊销 adapter）夹在 [+0.025 朴素, +0.346 上界] 之间、位置未知**。→ **idea 未被否（信息在），但"首帧朴素估计调制"走不通；判生死须测学习版摊销 adapter**。

### 探针 B-2（决定性）：learned 摊销 adapter 的 held-out headroom — **结果 GO（2026-09-12）**
离线训 target-conditioned adapter（target_emb=首30帧目标均值→逐通道 γ 或低秩），**跨 episode 划分**（train 51 / val 13，车辆不相交），测 val pick-accuracy。工具 `carla_uav_tracking/rollout/probe_adapter.py`。

| 方案 | train | **val** | val Δ |
|---|---|---|---|
| baseline（g=1 均匀 cosine） | 0.351 | 0.373 | — |
| **diag（FiLM-γ 学习）** | 0.474 | **0.452** | **+0.078** |
| lowrank adapter | 0.978 | 0.432 | +0.059（train 0.978≫val=**过拟合**） |

**结论 GO（但带保留）**：①**学习版 FiLM-γ 泛化到未见目标**（val +0.078），**且远超朴素逐-episode 拟合的 +0.025**（探针 B）——证明摊销学习比在线估计强，idea 在正确形态下有真 headroom；②**低秩 adapter 严重过拟合**（train 0.978/val 0.432,与 TAH v1 同病:51ep 车池窄)→ **先用对角 FiLM-γ,不上低秩**;③**但增益仍modest**（0.373→0.452,离 in-sample 上界 0.644 还远)——受训练目标池限;④这是 **per-frame WHICH-vision 离线增益,闭环 SR 转化未知**(我们反复的教训:单帧涨≠闭环涨),且**不解 confident-drift**。

### Tier-1（下一步,若上）：对角 FiLM-γ 接入 + 闭环
把 diag adapter 接到 vision→匹配路(调制特征→喂 TAHRelM cos-to-memory,与运动门正交),gt-seeded 闭环先验证,再 committed。**预期温和**(+0.078 单帧);若闭环 SR 无动→回到"单帧涨不移 SR"的老结论;若动→一个便宜、诊断驱动的 vision 增强。扩训练目标池可解低秩过拟合、逼近 0.644 上界。

### 探针 A（后做）：Qwen in-context 目标图
把首帧目标 crop 作参考图 + 当前帧多图喂 Qwen3-VL，对比"仅语言"下判别是否变好。
- **Go**：变好 → Qwen 自带条件化红利没用上，改 prompt/输入白捡（路 α 可行）；**No-go**：不变 → 走路 β。
- 需 OnlineVLM 多图路径（较重，探针 B 为正后再做）。

## 4. Tier 1 · 建轻量目标条件化调制（仅当 Tier 0 为正）
`target-conditioned FiLM/adapter head`：首帧目标 crop → 目标嵌入 → 逐通道 (γ,β) 调制候选 vision 特征再匹配。**离线在多目标上训**（学"给定目标→如何调制"，车辆不变）；**在线一次前向定调制、整集固定**（不在线反传，避首帧过拟合）；接入 TAHRelM 的 cos-to-memory（锐化外观信号，与运动门正交）。判据：离线分离度↑ → gt-seeded 闭环↑ → committed 闭环↑。

## 5. Tier 2 · 重型（仅当 Tier 1 不够 + Tier 0-B 显示信息在但需容量）
Qwen vision 的 LoRA/test-time 适配，或高分辨率目标-crop 专路。风险高，留最后。

## 6. 诚实边界
1. **提升的是 per-frame WHICH-vision，不直接解 confident-drift**（deployable 阻塞）；链条=调制↑单帧判别→降漂移发生率→间接帮 deployable；漂移检测仍独立。
2. **Tier 0-B 是生死门**：撞分辨率天花板则再精巧的调制也造不出可分性——**先探针，别直接建**。
3. 目标条件化与语言 grounding **互补**（语言给 which、外观调制给判别），非替代。

## 7. 产物 & 命令
- 探针 B：`carla_uav_tracking/rollout/probe_target_conditioning.py`（读 `runs/tah_cache.pt`）
- 探针 A：待建（OnlineVLM 多图）
- 当前状态见本文件 §3 结果小节（探针跑完回填）。
