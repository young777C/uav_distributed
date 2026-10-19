# 1001 · PoC 可行性验证闭环报告(PGIF)

> 2026-10-01 · 配套 `1001_innovation-design.md`(v3+)· 审稿记录 `1001_codex_review_r{1,2,3}.md`
> 路径约定:`eval_out/` 均指 `carla_uav_tracking/rollout/eval_out/`;日志在 `runs/asd/`。最终稿中 max_wrong(错窗长度)**不作为主 claim**(多个设定 CI 含 0)。
> 证据等级:**[M] 已测**(附路径)· **[H] 假设/待测** · **[P] 部分(n 不足)**。所有闭环臂同协议:Town05、seeds 91000–19、STEPS 900、s2=6、live CARLA,n=19 计分(91004 不计分),同种子配对 bootstrap(`asd_stats.py`)。

---

## 1. Claim → 证据 → 状态 矩阵
| # | Claim | 证据(数据路径) | 状态 |
|---|---|---|---|
| C1 | deployable 断点存在:无纠偏的 committed 记忆 correct 低、错窗长 | `paper_asd_tah_observe`:correct **0.272**、max_wrong **15.7s**、SR 0(n=19)| **[M]** |
| C2 | 漂移检测是杠杆(oracle 上界) | `paper_ext_tahrelm_oracledrift` vs C1:correct +0.449 [+0.345,+0.551]、max_wrong −13.9s | **[M]** |
| C3 | 现有 GT-free 修补无效:低置信语言重锚、冻结外观锚 | lang:Δcorrect −0.009 [−0.17,+0.15];anchor-consec:Δ+0.031 [−0.07,+0.13](均不显著)| **[M]** |
| D1 | 记忆自置信逐 tick 弱有信息(非不可辨识) | 合并全部 observe 日志,episode bootstrap 95% CI:TAH(30 ep / 1565 tick)自置信 AUROC **0.710 [0.546,0.811]**,冻结外观锚相异度 **0.501 [0.421,0.622]**(=随机);reid(8 ep / 795 tick)自置信 **0.819 [0.792,0.862]**(`carla_uav_tracking/rollout/eval_out/auroc_{tah,reid}_observe.json`)| **[M]** |
| D2 | 外观/语言参考在错锁时刻≈随机指认目标 | 错锁 tick 命中率 0.25–0.28(TAH)/0.10–0.39(reid)vs 随机 0.32/0.36;锚告警→真目标 0.22/0.03(`runs/asd/partial1`)| **[M]** |
| D3 | tracklet 持续性携带漂移免疫信号 | 历史驻留 0.45/1.00、上一稳定身份 0.44/0.97;右→错切换时真目标可见 85–92% | **[M]** |
| T1 | 自更新 → 误差自相关 → 未回火(独立假设)的贝叶斯滤波会被俘获,比不纠偏更差 | `paper_asd_tah_bayes` vs 基线:correct 0.272→**0.074**,Δ**−0.198 [−0.336,−0.075]**;max_wrong +11.1s [+2.5,+20.6];门控把选择拉离真目标 0.343 > 拉回 0.165。toy 支撑(`carla_uav_tracking/rollout/eval_out/toy_drift.json`)| **[M]** |
| M1 | PGIF(回火证据累积 + 后验门控)显著提升 deployable 身份质量(TAH) | `paper_asd_tah_tempered` vs C1:correct 0.272→**0.613**,Δ**+0.341 [+0.173,+0.511]**;latch@5 0.16→0.58;SR 0→0.158(3/19,McNemar p=0.25,不显著);max_wrong 15.7→10.8s(CI 含 0);门控改写 65% 拉到真目标 / 11% 拉离 | **[M]** |
| M2 | 门控记忆更新是关键(非单纯平滑) | PGIF vs 去门控(同一回火滤波、只改承诺):Δcorrect **+0.217 [+0.039,+0.400]**,Δmax_wrong **−9.3s [−17.8,−1.2]**;去门控相对基线不显著(+0.124 [−0.045,+0.295])| **[M]** |
| M3 | 置信度加权有增益(vs 等权驻留启发式) | PGIF vs 等权驻留:Δcorrect **+0.237 [+0.058,+0.424]**,Δmax_wrong **−6.2s [−12.1,−1.0]**;等权驻留相对基线不显著(+0.104 [−0.071,+0.286])| **[M]** |
| G1 | 跨记忆:同一冻结超参用于启发式 reid gallery | vs **同会话** reid 基线 `paper_asd_reid_observe`(n=19):correct 0.400→0.510,Δ+0.110 [−0.025,+0.263](未显著);**latch@5 0.21→0.53,Δ+0.316 [+0.105,+0.526](显著)**;max_wrong 无改善(+0.7s,CI 含 0)。同会话基线与 9 月 `reid_deploy` 一致(Δ−0.016 [−0.099,+0.057])| **[M]** 部分:持续错锁显著减少,平均正确率未显著 |
| G2 | 跨场景:Town10HD(TAH,同冻结超参) | `paper_asd_tah_tempered_t10` vs `paper_asd_tah_observe_t10`(n=20):correct 0.356→0.456,Δ+0.100 [−0.086,+0.292](未显著);**latch@5 0.15→0.60,Δ+0.450 [+0.25,+0.65](显著)**;SR 0→0.150(p=0.25)| **[M]** 部分:同 G1 |
| G3 | 三个设定合并(分层配对 bootstrap,n=58,`carla_uav_tracking/rollout/eval_out/pgif_generalization.json`) | **Δcorrect +0.182 [+0.087,+0.280]**;**Δlatch@5 +0.397 [+0.259,+0.534]**;Δmax_wrong −1.5s [−6.1,+3.4](未显著)| **[M]**(次要分析,合并前未针对其调参)|
| L1 | 依赖持久 tracklet id;ID 噪声下增益衰减 | §3.1:p≤0.02/tick 保留约 3/4 增益,p=0.1 时消失(`carla_uav_tracking/rollout/eval_out/pgif_idnoise_tah.json`)| **[M]**(适用前提)|

## 2. 开环回放消融(observe 日志,GT 只用于打分)
(`pgif_replay.py`;TAH 8 ep / reid 3 ep,小样本,仅作机理参考,结论以闭环为准)
| 变体 | TAH acc | reid acc |
|---|---|---|
| 记忆原始选择 | 0.485 | 0.639 |
| PGIF-bayes(秩置信,未回火) | 0.546 | 0.819 |
| 常数 q HMM | 0.501 | 0.880 |
| 历史驻留(λ=0.98,等权) | 0.560 | 1.000 |
| 上一稳定身份 | 0.538 | 0.990 |
**读法**:开环下"持续性先验"本身是主要成分;各具体滤波器差异在小样本噪声内。开环无法反映门控对后续观测的反馈作用 → 这正是 M1/M2 必须闭环检验的原因。

## 3. 闭环主表(TAHRelM 记忆,Town05,deployable/GT-free 除上界两行;n=19 同种子配对;Δ 为相对"无纠偏"的配对 bootstrap 95% CI)
> **接口声明(codex R3 MAJOR-1)**:所有臂(含基线与外部对照)的候选 id `c.idx` 来自模拟器的**理想持久关联**(等价于完美 MOT tracklet 层);PGIF 的结论严格限定为"**在持久 tracklet 接口下**,如何抑制自更新身份记忆的确认偏差"。对非理想关联的敏感度见 §3.1。
| 臂 | correct | Δcorrect [CI] | max_wrong(s) | latch@5 | SR |
|---|---|---|---|---|---|
| 无纠偏(committed + 语言冷启动)`paper_asd_tah_observe` | 0.272 | — | 15.7 | 0.16 | 0.000 |
| 低置信语言重锚 `paper_ext_tahrelm_deploy` | 0.263 | −0.009 [−0.164,+0.146] | 13.3 | 0.21 | 0.000 |
| 冻结外观锚重锚(连续 3 tick)`paper_asd_tah_anchor_p3` | 0.303 | +0.031 [−0.072,+0.128] | 12.2 | 0.05 | 0.000 |
| 未回火贝叶斯滤波 + 门控 `paper_asd_tah_bayes` | 0.074 | **−0.198 [−0.336,−0.075]** | 26.9 | 0.11 | 0.000 |
| 等权驻留 + 门控 `paper_asd_tah_resid` | 0.376 | +0.104 [−0.071,+0.286] | 17.0 | 0.37 | 0.158 |
| PGIF 去门控(只改承诺)`paper_asd_tah_tempered_nogate` | 0.396 | +0.124 [−0.045,+0.295] | 20.1 | 0.21 | 0.053 |
| **PGIF(ours)** `paper_asd_tah_tempered` | **0.613** | **+0.341 [+0.173,+0.511]** | **10.8** | **0.58** | **0.158** |
| 上界:GT 漂移检测 `paper_ext_tahrelm_oracledrift` | 0.721 | +0.449 [+0.345,+0.551] | 1.9 | 1.00 | 0.105 |
| 上界:oracle 记忆 `paper_ext_tahrelm_gt` | 0.746 | +0.474 [+0.353,+0.585] | 1.9 | 1.00 | 0.316 |

**PGIF 对各消融的直接配对**:vs 去门控 Δcorrect +0.217 [+0.039,+0.400] / Δmax_wrong −9.3s [−17.8,−1.2];vs 等权驻留 +0.237 [+0.058,+0.424] / −6.2s [−12.1,−1.0];vs 未回火贝叶斯 +0.540 [+0.313,+0.753] / −16.1s [−24.8,−8.1]。
**读法**:三个组成部分(秩置信加权、回火、门控)缺一即失去显著增益,且"未回火"反而有害——与命题 2 的三条可检验预测一致。PGIF 收回 oracle-drift 上界相对基线增益的 **76%**(0.341/0.449)。max_wrong 均值仍 10.8s(长尾:少数 episode 冷启动锁错或漂移超过回火时间常数),SR 增益不显著——如实报告。
图:`figs/fig_pgif_main.png`(`figs/make_pgif_figs.py`)。数据:`carla_uav_tracking/rollout/eval_out/pgif_report.json`。
**冷启动分层**(按各臂自身日志首个可标注 tick 判定):PGIF 在冷启动正确子集 0.211→0.637(n=11)、冷启动错误子集 0.365→0.521(n=7)均提升——增益不来自删样本。**max_wrong** 均值 15.7→10.8s 下降但 CI 含 0(**未显著**);SR 0→0.158 McNemar p=0.25(**未显著**)。

### 3.1 对 tracklet 关联噪声的敏感度(开环回放,30 个 TAH observe episode / 1655 tick,`carla_uav_tracking/rollout/eval_out/pgif_idnoise_tah.json`)
向 PGIF 输入的 tracklet id 流注入每 tick 概率 p 的碎片化(新 id)与身份互换:
| p(/tick, tick=0.6s) | 记忆原始 | PGIF-tempered | 等权驻留 | 未回火 bayes |
|---|---|---|---|---|
| 0 | 0.332 | **0.402** (+7.0pp) | 0.407 | 0.329 |
| 0.005 | 0.332 | 0.391 (+5.9) | 0.395 | 0.321 |
| 0.01 | 0.332 | 0.385 (+5.3) | 0.386 | 0.322 |
| 0.02 | 0.332 | 0.384 (+5.2) | 0.381 | 0.322 |
| 0.05 | 0.332 | 0.354 (+2.2) | 0.349 | 0.316 |
| 0.1 | 0.332 | 0.339 (+0.7) | 0.342 | 0.316 |
**读法**:p≤0.02/tick(每个目标约每 30s 至多一次关联错误)时仍保留约 3/4 开环增益;p≥0.1 时增益消失 → PGIF 需要一个"中等质量"的 MOT 关联层,这是明确的适用前提。开环回放中回火与等权驻留持平(0.402 vs 0.407),二者的闭环差异(+0.237,§3)来自门控对后续观测的反馈——开环回放无法体现,这也是闭环消融必要的原因。

## 4. 闭合判据核对(预先登记,来自 codex R2 #5)
PGIF 视为闭合,当且仅当(同种子配对):
1. TAH:correct 显著高于无纠偏基线(0.272)与语言重锚(0.263),bootstrap CI 不含 0;
2. max_wrong 明显下降,向 oracle-drift(1.9s)靠拢;latch@5 上升;
3. 门控消融(nogate)显著差于 PGIF(证明不是单纯平滑);
4. 冷启动分层后提升仍在冷启动正确子集成立(不靠删样本);
5. 同一冻结超参在 reid 上不劣于其 deployable 基线。
**核对结果**:1 ✅(Δ+0.341,CI 不含 0;且 > 语言重锚);2 ⚠️ latch@5 ✅ 显著上升,max_wrong 均值下降但未显著;3 ✅(vs 去门控 +0.217,CI 不含 0);4 ✅(冷启动正确子集 0.211→0.637);5 ✅ 不劣(correct +0.110 未显著,latch@5 显著上升)。
**Codex 审稿轨迹**:R1 MAJOR(novelty/命题过强/闭环未闭)→ R2 MAJOR(parser bug/门控 vs 平滑/tracklet 依赖)→ R3 MAJOR(tracklet 接口/泛化待出)→ **R4 NO MAJOR ISSUES**(`1001_codex_review_r4.md`)。

## 5. 未闭合 / 局限
- 依赖模拟器持久 tracklet id(理想 MOT 关联);ID 噪声 ≥5%/tick 时增益消失(L1)。
- 冷启动锁错无法纠正。
- 每臂 n=19、单模拟器;部分 CARLA 实例(GPU3/4/6)渲染异常导致臂迁移,所有计入主表的臂均在正常实例上完整跑完 20 ep。
