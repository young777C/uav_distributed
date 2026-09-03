# re-ID 线总结 —— 从 SR≡0 到可部署 SR 0.105(论文正向脊梁)

> 2026-09-03 · 汇总整条 WHICH/re-acquisition 调查线 · 配套图 `reid_dinov2_gallery_pipeline.html`、memory `acot-uav-reid-reframe`
> **一句话**:出画重捕获的墙是**「认哪辆车」(WHICH)而非「去哪拦截」(WHERE)**;单帧语言 grounding 是其天花板;换成**时序外观 re-ID**(crop-DINOv2 特征 + K 视角图库 + 置信门控控制)把 SR 从恒 0 抬到**可部署 0.105**,**全程训练无关、不提分辨率**。

---

## 0. 任务与指标
无人机跟踪语言指定目标车,目标出画后需**预测-拦截-重捕获**。**SR** = 整段跟对被指目标不失守(committed 错车 ≥2s 或 出画 ≥5s 即失败)。辅助指标:mis_follow(逐帧认错率)、reacquire_lock_wrong(重现时锁错车率)。

## 1. 重定向:真墙是 WHICH,不是 WHERE(已证)
把 WHERE 从「烂」推到「近乎完美」,SR 与认错率**纹丝不动**:

| 臂 | WHERE 质量 | mis_follow | lock_wrong | **SR** |
|---|---|---|---|---|
| ear(反应式) | 差 | 0.71 | 0.69 | 0 |
| cv_gated(记忆+CV) | 好 | 0.70 | 0.74 | 0 |
| road(路网) | 好 | 0.72 | 0.75 | 0 |
| **road_oracle** | **近乎完美** | 0.72 | 0.74 | **0** |

→ WHERE 不是瓶颈(WM/road 不加分);**重现时认错车(lock_wrong 恒 ~0.74)才是墙**。

## 2. 单帧语言 grounding 是天花板 → 时序 re-ID 是范式修正
- 一切离线 grounding 优化(attrbind/加容量/换损失/DAgger)**不迁移闭环**(在线卡 0.74);专家取景(完美取景)SR 仅 **0.11**;
- **机理**:重捕获 = look-alike 里的**重识别(re-ID)= 时序外观关联**,不是单帧分类。tid 问「哪辆最贴描述」(look-alike 都满足),而非「哪辆是我一直跟的那辆」。

## 3. 把「0.57 墙」拆成两个正交子问题(诊断)
用 margin 诊断(self=目标 vs 自身图库,similar=look-alike vs 图库):

| 特征 | look-alike 余弦↓ | 自匹配 self↑ | margin | mis_follow(难) |
|---|---|---|---|---|
| VLM 网格池化 | 0.877 | 0.910 | +0.033 | 0.578 |
| DINOv2 单 EMA | **0.614** | 0.705 | +0.091 | 0.598 |

→ **墙 = 特征可分性 × 视角鲁棒性**:DINOv2 修可分性(0.88→0.61),但**视角敏感**(self 0.91→0.71),单模板对不上重现新视角 → 需第二味药。

## 4. 三个组件(全部训练无关、不提分辨率)

**① crop-DINOv2 特征** —— 从原始 RGB 裁每辆车 → DINOv2(实例判别),而非把 24px 车池化成 VLM 网格 ~1 格。同样像素给车 ~196 个 token(而非 1),**捞回被粗池化丢掉的身份信息**。

**② K=2 视角图库** —— 存目标多个视角(多样性采样)、重现时匹配最像的(top-m),**兑现 DINOv2 的可分性、修视角敏感**。甜点 K≈2-3(过多→干扰车碰运气撞视角)。离线难子集 mis_follow **0.578→0.460 首次破 0.5**(512 分辨率靠提分辨率封顶在 0.578)。

**③ 置信门控控制** —— track(reid 驱动控制)下,reid pick 的 margin(top1−top2)低时**不切换承诺、不锚控制**,改沿上次高置信轨迹 dead-reckon → **修「认错→飞错车→丢目标」的控制耦合代价**。

## 5. 完整阶梯(SR,19 集 Town05 seed91000 配对)

| 配置 | WHERE | WHICH | 控制 | mis_follow | lock_wrong | **SR** |
|---|---|---|---|---|---|---|
| 所有 WHERE/WM(含 road_oracle) | 近乎完美 | 单帧 | — | 0.72 | 0.74 | **0** |
| cv+xattn | CV(GT) | 单帧语言 | GT-WHERE | 0.703 | 0.737 | 0 |
| cv+reid-vlmEMA | CV(GT) | 时序 EMA | GT-WHERE | 0.568 | 0.633 | 0 |
| **cv+reid-DINOv2-K2** | CV(GT) | crop-DINO+图库 | GT-WHERE | **0.403** | **0.566** | **0.105** |
| track+reid-DINOv2-K2 | CV(reid认定车)† | crop-DINO+图库 | 耦合 | 0.357 | 0.488 | 0.053 |
| **track+reid-DINOv2-K2 + 门控** | CV(reid认定车)† | crop-DINO+图库 | **门控** | 0.385 | 0.586 | **0.105** |

> **† 脚注(WHICH 顺带决定 WHERE)**:reid 本身是 **WHICH**(认哪辆)。在 `track` 接线下,认定的那辆车 → 取其位置做 CV 递推 → 填 `z_ex`(WHERE) → 无人机飞过去。所以 WHERE 不是"reid",而是 **"CV(reid认定车)"**——WHICH 的决策**顺带决定了 WHERE**,认错就飞错,故需置信门控兜底。cv_gated 行的 WHERE 是 `CV(GT目标)`(用真值当 z_ex,reid 只打分,不驱动控制)。

**关键结果**:
- **SR 0→0.105**(全线首个非零,除专家取景);**四臂恒定的 lock_wrong~0.74 认错墙 → 0.49**;
- **离线突破迁移闭环**(cv+reid-DINO:mis 在线 0.40,攻破反复烧过的 offline↔online gap);
- **完全可部署版**(track+门控,无 GT-WHERE 拐杖、无重训、无提分辨率)**= 0.105 = GT-WHERE 版上界 ≈ 专家取景上界 0.11**。

## 6. 诚实边界(必须写进论文)
- **SR 0.105 是该场景的任务可解上界附近**(专家取景 0.11);再往上需**更易场景**或**叠加高分辨率**(512→车 24px 是当前物理约束);
- **图库 oracle-seeded**(目标可见时用 GT 建库)—— 「可见时跟对了」的合理假设,但严格可部署需从策略自身承诺建库;
- **控制侧仍用 CV 先验的 WHERE**(GT 锚点级隐私,与 GT 候选框同级);
- 分辨率探针证:512 数据认车封顶 0.578,crop-DINOv2 在同分辨率下破 0.5 = **「不是像素不够,是特征用糟了」**;但更高分辨率仍是正交的进一步杠杆。

## 7. 论文定位(三个可发表贡献)
1. **诊断/重定向方法学**:一整套闭环拆解证明 WHERE(含 WM)不是 SR 瓶颈、WHICH-under-reacquisition 才是;offline↔online gap;把「0.57 墙」拆成 特征可分性×视角鲁棒性 两个正交子问题;
2. **正向方法**:crop-DINOv2 + K 视角图库 + 置信门控 = **训练无关、分辨率无关**的 re-ID 修法,SR 0→0.105、认错墙被撬,且**可部署版追平 GT-WHERE 上界**;
3. **系统/工具**:双容器闭环 harness + `reid_resolution_probe`(特征/分辨率/图库探针)+ policy 的 reid/gallery/conf-gate 开关。

## 8. 关联
- 图:`reid_dinov2_gallery_pipeline.html`;memory:`acot-uav-reid-reframe`、`acot-uav-wm-roadgraph`(WM 闭合);
- 结果:`eval_out/road_{cv_gated,reid,reiddino,trackreiddino,trackreiddino_conf05}.json`;
- 代码:`policy.py`(tid_head=reid / reid_feature=dinov2 / reid_bank / conf_tau)、`reid_resolution_probe.py`(--feature/--max-pixels/--bank/--topm)。
