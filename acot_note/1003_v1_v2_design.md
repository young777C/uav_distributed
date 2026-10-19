# 1003 · V1 / V2 设计文档(预注册;闭环前冻结)

> 2026-10-02 · 属于 `1003_iteration_workflow.md` 的 V1、V2。所有实验:deployable(身份决策不用真值;候选框与车辆定位为仿真理想输入),CARLA 在线闭环,Town05 seeds 91000–19(19 计分),三项客观指标(正确锁定比例、最长连续正确锁定、最长锁错),配对 bootstrap 95% CI。基线 V0 = TAHRelM + PGIF(`1001_innovation-design.md` v4)。

## 背景数据(deployable,`eval_out/deploy_compare.json`)
| | 正确锁定比例 | 最长连续正确(均值/中位) | 最长锁错(均值/中位) |
|---|---|---|---|
| DAM4SAM(SAM2.1-L + 干扰物感知记忆,CVPR'25) | 0.786 | 13.3 / 6.4 s | **26.7 / 21.6 s** |
| V0:TAHRelM + PGIF | 0.583 | 21.3 / 11.5 s | 15.4 / 0.9 s |

冷启动分层(V0):初始锁对时正确比例 0.96/0.90/0.80,初始锁错时接近 0(Town05 / Town10HD / reid),锁错开局占 7/19、11/20、7/19。TAHRelM 的冷启动目前是**单帧**语言选车(第 1 个决策步的 grounding argmax 直接写入记忆)。

---

## V1 · PGIF ∘ DAM4SAM:把后验门控推广到强外观记忆

**动机**:DAM4SAM 的长时间错锁(26.7 s)是 SAM2 记忆"用自己的预测掩码更新自己"造成的确认偏差,与 PGIF 在 TAHRelM 上处理的是同一机制(命题 1)。若 PGIF 的门控能在不同类型的记忆上起作用,则 C2"记忆无关的闭环身份门控"得到直接验证,同时有望保留 DAM4SAM 的高正确比例并压缩其错锁。

**方法(与 V0 的 PGIF 完全相同的规则与超参)**:
- 观测:DAM4SAM 每个决策步的选择 s^M = 与 SAM2 预测框中心最近的候选(160 px 门限内);置信度 c = SAM2 对所选掩码的**预测 IoU**(服务新增返回字段)。无匹配时该步不更新证据。
- 回火证据:L_k ← 0.98·L_k;L_{s^M} += 0.5 + rank(c; 最近 30 步)。
- 门控:ŝ = argmax_{可见 k} L_k;若 ŝ ≠ s^M → **重置钩子 = 用 ŝ 的候选框重新 init SAM2**(丢弃 RAM/DRM 记忆),控制承诺 ŝ。
- 预注册附加参数:重置冷却 3 个决策步(避免频繁 re-init 抹掉 DRM);只在 committed 模式启用。

**预注册假设(Town05,n=19 配对,V1 vs DAM4SAM)**:
- H1a:正确锁定比例**非劣**——Δ 的 95% CI 下界 > −0.05;
- H1b:最长锁错**显著降低**——Δ 的 95% CI 上界 < 0;
- H1c(探索性):最长连续正确锁定不下降。
- 失败解释预案:若 H1b 失败且冷启动锁错占主导 → 说明长尾来自初始身份错误而非漂移,转由 V2/V3 处理。

**与相关工作的差异**:DAM4SAM 自身的 DRM 用"备选掩码与所选掩码分歧"来**选择写入哪些帧**,仍以 SAM2 的当前预测作为身份;V1 在 tracklet 层引入独立于该记忆的、按误差相关时间回火的身份后验,并在后验与记忆冲突时**重置**记忆——处理的是"记忆已经锁到错误实例"之后的纠错,而非写入策略。

---

## V2 · 延迟承诺冷启动(sequential language commitment)

**动机**:长尾错锁主要来自开局锁错(P3);单帧语言选车准确率约 0.42(离线 xattn,mis 0.579),而 PGIF 会忠实维持初始身份。把"开局选谁"也变成一个带回火的序贯证据决策。

**方法**:
- 冷启动窗口:前 K = 5 个决策步(≈3 s)**不写入身份记忆**;每步对可见 tracklet 累积语言证据 G_k ← 0.9·G_k + (0.5 + rank(语言 margin; 窗口内)) · 1[k = 语言 argmax](与 PGIF 同形式的回火投票,λ_lang = 0.9 预注册)。
- 窗口内控制:承诺当前 argmax_k G_k(临时承诺,仅用于飞行;记忆不更新)。
- 窗口结束:承诺 k* = argmax_k G_k,用其当前截图初始化 TAHRelM 记忆;PGIF 证据以 L_{k*} = G_{k*} 初始化(保留开局的语言支持度)。
- 语言来源:沿用部署中的 Stage-2 grounding 头(冻结);LoRA r16 打分器作为可选 V2b(阶段 0 未显著,暂不预注册)。

**预注册假设(Town05,n=19 配对,V2 vs V0)**:
- H2a:开局锁错的 episode 数下降(描述性,配对比较各种子的开局是否锁对);
- H2b:最长锁错均值显著下降(Δ 的 95% CI 上界 < 0);
- H2c:正确锁定比例不下降(Δ 下界 > −0.05)。

**与相关工作的差异**:多目标跟踪中的"轨迹确认"(tentative → confirmed,如 DeepSORT 的 n_init)按检测连续性确认轨迹,不涉及"哪条轨迹是语言所指";指代多目标跟踪(RMOT/iKUN)逐帧做语言匹配。V2 把"语言指代"本身作为带回火的序贯证据决策,并把开局证据作为后续 PGIF 后验的先验。

---

## 统一实现清单
- `external/DAM4SAM/dam4sam_tracker.py`:输出所选掩码的预测 IoU(已改);`rollout/dam4sam_service.py` 返回 `iou`(已改)。
- `rollout/policy.py`:dam4sam 分支接入 ASD/PGIF(观测 = s^M、c = IoU;门控 = 重新 init);tah 分支冷启动窗口(V2)。
- `rollout/asd.py`:PGIF 证据初始化接口(V2)。
- 单测:合成序列上冷启动窗口选出语言证据最多的 tracklet;dam4sam 门控在冷却期内不重复 re-init。

---

## R1 审核后的修订(2026-10-02,第二轮预注册;codex 意见见 `codex/1003_R1a_R2a.md`)

### V1 修订
1. **机制对照(预注册,与 V1 同种子同时跑)**:
   - **V1-commit-only**:PGIF 只改写控制承诺,不重置 SAM2(`gate_memory=false`)→ 检验"重置记忆"是否必要;
   - **V1-selfreset**:在 V1 会触发的同一时刻,用 DAM4SAM **自己的选择**重新 init SAM2、不改变身份(`V1_SELFRESET=1`)→ 检验收益是否只是"re-init / 丢弃 DRM"本身。
   - 判定:V1 优于 V1-selfreset(最长锁错或正确比例,配对 CI)才说明收益来自"改到后验所指的身份",而非重置本身。
2. **冷启动分层(预注册)**:H1b 拆为 H1b-all(全部 episode)与 **H1b-drift(仅开局锁对的 episode)**;开局锁对与否由各臂 tick 日志首个可标注 tick 判定(`tick_align.py`)。V1 的主张限定为"抑制开局正确之后的漂移错锁";开局锁错交给 V2/V3。
3. **置信度有效性**:报告 SAM2 预测 IoU 对"所选为真目标"的 AUROC(episode bootstrap CI,observe 日志 = DAM4SAM 基线臂的 tick 日志)。若 AUROC < 0.55,则在 roadmap 中注明 PGIF 在 DAM4SAM 上退化为等权驻留,并以此解释结果。
4. 报告每 episode 重置次数、单 episode 耗时(`duration_seconds`);H1c 不作为通过门槛。

### V2 修订
1. **定位**:V2 是 PGIF 的**冷启动组件**,不单独作为贡献;整体贡献在 V3(融合)中评估。
2. **去除飞行混杂**:窗口内飞行改为与 V0 相同——每步跟随**单帧语言选择**(不再飞向临时 argmax);窗口内不写记忆;窗口结束时承诺 argmax_k G_k。
3. **对照(预注册)**:**V2-single@K**——窗口与飞行完全相同,但在第 K 步承诺**该步单帧语言选择**(不积累证据、无先验)→ V2 与它的差异只来自"序贯证据积累"。
4. **H2a 改为可检验**:开局正确率(第 K 步承诺是否为真目标)V2 vs V2-single@K,配对 bootstrap 95% CI;窗口内的临时承诺计入最长锁错(scorer 按承诺身份逐帧计)。
5. **先验截断**:PGIF 初始证据 L_k = min(G_k, 3.0)(≈ 3 票),防止错误开局获得过强先验;k* 只在可见 tracklet 中选(不可见则取可见中证据最多者)。
6. K = 5、λ_lang = 0.9 为先验设定(未在评测种子上调);敏感性列为后续。

### 闭环臂(Town05,seeds 91000–19,n=19)
| 臂 | 设定 |
|---|---|
| V1 | DAM4SAM + PGIF(重置 = 后验身份 re-init,冷却 3 步) |
| V1-commit-only | DAM4SAM + PGIF,只改承诺 |
| V1-selfreset | DAM4SAM,同时刻自身 re-init |
| V2 | TAHRelM + PGIF + 延迟承诺(K=5,λ=0.9,先验截断 3) |
| V2-single@K | 同 V2 窗口与飞行,第 K 步单帧承诺 |
| 对照(已有/在跑) | DAM4SAM、V0(TAHRelM+PGIF)、TAHRelM 无纠偏(`eval_out/paper_dep_*.json`) |
