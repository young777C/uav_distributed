---
name: paper1-v2-results-analysis
description: >
  Validate, analyze, visualize, and write Chapter 5 for the paper1_v2 C3 UAV–GCS
  experiments. Use only validated results_v2 data, answer RQ1–RQ4 with seed-level
  paired statistics, and generate publication-ready figures and LaTeX text.
disable-model-invocation: true
argument-hint: "[stage: validate|overall|ablation|timescale|runtime|chapter|all]"
---

# paper1_v2 第五章实验结果分析 Skill

## 1. 目标与适用范围

本 Skill 用于在全部正式实验完成后，对 `results_v2/` 中的数据进行完整性复核、统计建模、结果解释、图片与表格生成，并形成论文第五章“实验结果与分析”的 LaTeX 草稿。

本 Skill 只负责分析已经完成并通过完整性检查的实验，不负责修改算法、补造数据或重新定义研究问题。

第五章必须回答：

- **RQ1：** FDLC 在不同通信退化强度和任务–通信冲突等级下，是否优于代表性基线，其优势出现在哪些条件下？
- **RQ2：** 事件驱动更新和快环模式切换分别产生何种贡献？
- **RQ3：** 双时间尺度比 \(\rho\) 如何影响任务性能、重规划开销和事件补偿能力？
- **RQ4：** 默认配置下，快环与慢环计算是否满足在线周期要求？

分析原则：

1. 先验证数据，再建模，再绘图，最后写作；
2. 以研究问题组织结果，不按脚本或指标流水账组织；
3. 主要结论必须同时包含效应方向、效应量、置信区间和统计证据；
4. 必须报告不支持预期的结果和方法适用边界；
5. 不为追求篇幅完整而重复同一数据。

---

## 2. 强制前置条件

### 2.1 数据来源

只允许读取：

```text
results_v2/
```

禁止：

- 使用旧实验目录；
- 手工复制旧论文中的数值；
- 将 Pilot 数据并入正式统计；
- 将失败、缺字段或哈希不一致的 episode 纳入分析；
- 为填补表格而插值、补值或伪造结果。

分析开始前必须读取：

```text
results_v2/summaries/integrity_report.md
results_v2/summaries/episode_metrics.csv
results_v2/summaries/seed_summary.csv
results_v2/summaries/scene_summary.csv
results_v2/manifests/
results_v2/metadata/
```

若完整性报告未通过，立即停止。

### 2.2 实验协议一致性

第五章使用的实验规模应与第四章一致：

```text
10 seeds × 10 episodes
```

分析脚本必须从 manifest 和原始目录实际统计：

```text
n_unique_seeds
n_episodes_per_seed
n_successful_runs
n_failed_runs
n_missing_runs
```

若实际结果与第四章协议不一致，停止生成论文正文，并输出协议冲突报告。

不得通过“总 episode 数相同”替代独立统计单元一致性。例如：

```text
8 seeds × 20 episodes
```

与：

```text
10 seeds × 10 episodes
```

在统计意义上不等价。

### 2.3 名称统一

全文只使用：

```text
CDSL
WCDL
RHC-Inspection
CBCP
FDLC
Periodic Goal
Event-driven Goal
```

不再使用 `Full Coupling` 作为方法名称。

耦合消融中的完整配置统一写为：

```text
FDLC
```

禁止将 `FDLC` 写成 `FDCL`。

### 2.4 场景表述

场景因素统一为：

```text
通信退化强度：low / medium / high / severe
空间冲突等级：M0 / M1 / M2 / M3
```

本文只有一种复合通信场景，由距离衰减、局部链路退化和时变随机扰动共同构成。

禁止：

- 恢复 C1/C2 场景名称；
- 将结果解释为分别验证了“距离衰减鲁棒性”和“遮挡鲁棒性”；
- 将 M0–M3 与退化强度混为同一因素。

### 2.5 方案 B 的解释边界

必须保持：

- backlog 由快环管理；
- 数据交付事件用于任务完成状态确认；
- backlog 不进入慢环目标函数、约束或候选排序。

因此，不得写：

```text
backlog 直接改变慢环目标函数
backlog 直接提高候选点优先级
慢环显式优化数据积压
```

可以写：

```text
数据交付状态和交付事件更新全局任务边界，并触发后续规划。
```

### 2.6 新增脚本命名

所有新增分析、统计、绘图、表格和 LaTeX 生成脚本必须放入：

```text
scripts/paper1_v2/
```

文件名必须以：

```text
paper1_v2_
```

开头。

每个脚本文件头必须包含：

```python
"""
Paper: paper1_v2
Purpose: <用途>
Inputs: <输入>
Outputs: <输出>
"""
```

---

## 3. 第五章推荐结构

第五章按研究问题组织为以下五节：

```text
5 实验结果与分析
5.1 复合通信退化下的总体性能
    5.1.1 跨场景总体比较
    5.1.2 退化强度与空间冲突的影响
5.2 快慢环耦合机制的贡献
5.3 双时间尺度敏感性
5.4 计算实时性与部署可行性
5.5 本章小结
```

不要单独设置“指标逐项分析”小节。辅助指标应在回答相应研究问题时使用。

第五章建议控制为：

- 5.1：4–6 段；
- 5.2：3–5 段；
- 5.3：3–5 段；
- 5.4：2–4 段；
- 5.5：1–2 段。

详细局限性、外部有效性和未来真实系统部署问题留至讨论章节。

---

## 4. 分析数据构建

### 4.1 层级数据

保留三层数据：

```text
episode-level
seed-level
scene-level
```

正式统计以 seed 为独立单元。

对于每个：

```text
method/strategy + scene + rho + seed
```

先聚合其 10 个 episode。

输出：

```text
results_v2/summaries/paper1_v2_episode_analysis.csv
results_v2/summaries/paper1_v2_seed_analysis.csv
results_v2/summaries/paper1_v2_scene_analysis.csv
```

### 4.2 主指标与辅助指标

主指标：

```text
R_task
```

任务语义分解：

```text
R_cov
P_del|cov
delivery latency mean / median / p90
```

安全与任务闭合：

```text
R_home
R_oob
R_E
timeout rate
min return-energy margin
```

协同与开销：

```text
N_replan
replan_periodic
replan_event
event_replan_ratio
N_fb
target_switch_count
mode_switch_count
```

实时性：

```text
fast mean / p95 / p99 / max
slow mean / p95 / p99 / max
deadline-miss rate
solver timeout rate
solver failure rate
MIP gap
event-to-command latency
CPU
peak memory
```

### 4.3 指标解释约束

必须遵守：

- \(R_{\mathrm{task}}\) 是主要性能结论；
- \(R_{\mathrm{cov}}\) 与 \(P_{\mathrm{del}\mid\mathrm{cov}}\) 用于解释任务损失来源；
- 数据交付时延必须与完成率同时解释，避免“只完成少量容易任务”造成选择偏差；
- \(R_{\mathrm{home}}\) 不是越高越好，应与 \(R_{\mathrm{task}}\)、剩余能量和终止原因联合解释；
- \(R_E\) 是能量约束或返航阈值触发率，不是能量耗尽率；
- \(T_{\mathrm{deg}}\) 是诊断变量，无固定单调优劣；
- 重规划次数增加不能直接解释为目标抖动，除非 `target_switch_count` 同时增加；
- 事件触发占比升高不能自动解释为协同效果更好。

---

## 5. 统计分析总流程

### 5.1 跨场景总体模型

对总体方法比较拟合：

```text
Method × Intensity × Conflict + (1 | Seed)
```

根据指标分布选择模型：

| 指标 | 模型 |
|---|---|
| 完成数、覆盖数、交付数 | 二项 GLMM |
| 比例 | 优先使用原始分子/分母的二项 GLMM |
| 数据交付时延 | Gamma GLMM 或对数变换 LMM |
| 重规划、反馈、目标切换次数 | 负二项 GLMM |
| 返航、越界、超时 | 二项 GLMM |
| 快慢环计算耗时 | 对数变换 LMM |

至少输出：

```text
fixed effects
interaction effects
estimated marginal means
planned contrasts
95% CI
raw p
adjusted p
model family
convergence status
singularity status
```

不得仅根据均值曲线声称存在交互作用。

### 5.2 场景内辅助检验

每个场景：

1. 以 seed 为区组执行 Friedman 检验；
2. 总体差异显著后，执行预设 Wilcoxon 配对检验；
3. 比较族内使用 Holm 校正；
4. 报告均值差、BCa 95% CI、校正后 \(p\) 和秩二列相关系数。

总体方法对比只做：

```text
FDLC vs CDSL
FDLC vs WCDL
FDLC vs RHC-Inspection
FDLC vs CBCP
```

不得为追求显著结果执行所有方法的无计划两两比较。

### 5.3 耦合消融

跨场景模型：

```text
Strategy × Intensity × Conflict + (1 | Seed)
```

策略：

```text
Periodic Goal
Event-driven Goal
FDLC
```

预设比较：

```text
Event-driven Goal vs Periodic Goal
FDLC vs Event-driven Goal
FDLC vs Periodic Goal
```

该分析用于区分：

- 仅增加事件更新的贡献；
- 在事件更新基础上增加快环模式切换的增量贡献；
- 完整 FDLC 相对固定周期策略的总效应。

### 5.4 时间尺度分析

模型：

```text
Strategy × Rho × Scene + (1 | Seed)
```

其中：

```text
Strategy ∈ {Periodic Goal, FDLC}
Rho ∈ {20, 40, 80, 100}
```

\(\rho\) 作为分类变量。

重点比较：

```text
每个 rho 下：FDLC vs Periodic Goal
每个策略内：rho=40 vs rho=20/80/100
```

`log2(rho)` 与二次项只能用于探索性趋势描述，不作为主显著性结论。

### 5.5 Bootstrap

使用 seed 级配对 BCa bootstrap：

```text
5000 resamples
95% CI
```

同一次重采样中，不同方法必须使用相同 seed 索引。

禁止在 episode 层独立 bootstrap。

### 5.6 模型失败处理

若模型不收敛：

1. 保存完整警告；
2. 检查过度离散、完全分离和零膨胀；
3. 尝试与指标分布匹配的替代模型；
4. 不得静默删除交互项；
5. 若仍失败，报告失败并使用 seed 级非参数结果作为补充。

任何模型替换必须写入：

```text
results_v2/metadata/chapter5/model_decisions.md
```

---

## 6. 5.1 复合通信退化下的总体性能（回答 RQ1）

### 6.1.1 要回答的问题

必须回答：

1. 五种方法在全部 16 个场景中的总体排序如何？
2. FDLC 的平均优势和不确定性是多少？
3. FDLC 的优势是否随退化强度和空间冲突变化？
4. FDLC 在哪些场景中不占优或差异不明确？
5. 优势来自覆盖推进、覆盖后交付，还是二者共同作用？

### 6.1.2 分析顺序

第一段：报告跨场景总体结果。

包括：

- 各方法的估计边际均值；
- FDLC 相对四个基线的预设对比；
- 95% CI、校正后 \(p\) 和效应量；
- 方法主效应是否成立。

第二段：报告场景依赖性。

包括：

- `Method × Intensity`；
- `Method × Conflict`；
- `Method × Intensity × Conflict`；
- 不得只说“随着环境恶化，优势更明显”，必须用交互项或分层对比支持。

第三段：分析任务语义分解。

比较：

```text
R_task
R_cov
P_del|cov
delivery latency
```

明确区分：

- 到不了更多 POI；
- 覆盖后未完成数据交付；
- 完成更多困难任务导致交付时延上升。

第四段：报告边界与反例。

至少指出：

- FDLC 不显著优于最佳基线的场景；
- FDLC 均值低于某个基线的场景；
- 低冲突场景中附加反馈机制是否产生净收益。

第五段：直接回答 RQ1。

推荐结论结构：

```text
总体证据
+ 条件性优势
+ 主要机制指标
+ 适用边界
```

### 6.1.3 推荐图表

#### 图 3：总体方法性能

跨双栏图，两部分：

```text
(a) 五种方法 × 16 场景的 R_task 热力图
(b) FDLC 相对每个场景最优非 FDLC 基线的 ΔR_task 热力图
```

定义：

```text
ΔR_task =
R_task(FDLC) - max{R_task(CDSL), R_task(WCDL),
                   R_task(RHC-Inspection), R_task(CBCP)}
```

要求：

- 场景顺序固定为 low→severe、M0→M3；
- 面板 (b) 使用以 0 为中心的发散色标；
- 不以颜色深浅代替统计显著性；
- 必要时用简洁符号标记预设比较通过校正的场景；
- 不在热力图每个单元塞入过多小数。

#### 图 4：强度与冲突交互

两部分：

```text
(a) 各方法随退化强度变化的估计边际均值及 95% CI
(b) 各方法随冲突等级变化的估计边际均值及 95% CI
```

两幅图使用相同纵轴范围。

#### 表 7：总体统计结果

仅保留：

```text
method
estimated marginal mean R_task
95% CI
contrast vs FDLC
adjusted p
effect size
```

不在正文表格逐项列出 80 个场景–方法均值；完整数值放入补充材料。

---

## 7. 5.2 快慢环耦合机制的贡献（回答 RQ2）

### 7.1 要回答的问题

必须回答：

1. 事件驱动目标更新相对固定周期更新是否有独立贡献？
2. 快环模式切换在事件驱动更新基础上是否带来额外收益？
3. 两种机制的贡献是否随强度和冲突等级变化？
4. 收益是否伴随更多反馈或重规划开销？
5. 是否存在事件过密但任务收益不增加的场景？

### 7.2 分析顺序

第一段：报告三种策略的总体差异和预设对比。

第二段：分解增量贡献：

```text
Event-driven Goal - Periodic Goal
FDLC - Event-driven Goal
FDLC - Periodic Goal
```

第三段：分析强度和冲突交互。

重点关注 M2–M3 和 high–severe，但不得只选择有利场景而忽略其余场景。

第四段：结合：

```text
N_replan
event_replan_ratio
N_fb
target_switch_count
mode_switch_count
```

解释收益与开销。

第五段：直接回答 RQ2，并指出两种机制是互补还是一方主导。

### 7.3 禁止的推断

禁止：

- 将高重规划次数直接称为目标抖动；
- 将高事件占比直接称为更强适应性；
- 声称 backlog 直接进入慢环规划；
- 因 FDLC 优于 Periodic Goal 就断言所有事件触发都有效；
- 忽略 FDLC 相对 Event-driven Goal 不显著的场景。

### 7.4 推荐图表

#### 图 5：耦合消融效应

两部分：

```text
(a) Event-driven Goal 相对 Periodic Goal 的 ΔR_task
(b) FDLC 相对 Event-driven Goal 的 ΔR_task
```

均采用 4×4 场景热力图，发散色标以 0 为中心。

#### 表 8：耦合消融统计摘要

包括：

```text
planned contrast
overall mean difference
95% CI
adjusted p
effect size
high-stress subset difference
replan difference
feedback difference
```

高压力子集必须预先定义为：

```text
high-M2
high-M3
severe-M2
severe-M3
```

不得事后挑选场景。

---

## 8. 5.3 双时间尺度敏感性（回答 RQ3）

### 8.1 要回答的问题

必须回答：

1. \(\rho=40\) 是否在四个代表性场景中形成稳定折中？
2. 较小 \(\rho\) 是否主要增加重规划开销？
3. 较大 \(\rho\) 是否造成状态更新滞后？
4. 事件触发能否补偿较大的周期间隔？
5. 补偿作用在哪些条件下失效？

### 8.2 分析顺序

第一段：报告 `Strategy × Rho × Scene` 模型结果。

第二段：比较 FDLC 在四个 \(\rho\) 下的：

```text
R_task
delivery latency
N_replan
event_replan_ratio
```

第三段：在每个 \(\rho\) 下比较 FDLC 与 Periodic Goal。

第四段：结合目标切换次数和模式切换次数解释高频更新的执行代价。

第五段：回答 RQ3，并说明默认 \(\rho=40\) 是当前测试范围内的经验工作点，而不是普适最优值。

### 8.3 解释边界

只有在数据支持时才可写：

```text
较小 rho 增加规划频率但未带来任务收益。
较大 rho 增加全局状态滞后。
事件反馈能够部分补偿周期更新变慢。
```

若 `target_switch_count` 未增加，不得使用“目标抖动”解释。

若 FDLC 在某些 \(\rho\) 下低于 Periodic Goal，必须明确报告，并解释为事件反馈的适用边界，而不是隐藏该结果。

### 8.4 推荐图表

#### 图 6：时间尺度敏感性

采用 2×2 子图，对应四个代表场景：

```text
c3_medium_m2
c3_high_m2
c3_high_m3
c3_severe_m3
```

每个子图：

- 横轴：\(\rho\)；
- 纵轴：\(R_{\mathrm{task}}\)；
- 两条线：Periodic Goal、FDLC；
- 误差范围：seed-level 95% CI。

#### 表 9：性能–开销折中

按场景与策略列出：

```text
rho
R_task
delivery latency
N_replan
event_replan_ratio
target_switch_count
R_home
```

正文只保留代表性汇总；完整表放补充材料。

---

## 9. 5.4 计算实时性与部署可行性（回答 RQ4）

### 9.1 要回答的问题

必须回答：

1. 快环 p95、p99 和最大耗时是否低于 \(T_f\)？
2. 慢环 p95、p99 和最大耗时是否低于 \(T_s\)？
3. 是否存在 deadline miss、求解超时或求解失败？
4. 高冲突、高退化场景是否增加候选图规模或求解时间？
5. 事件发生到新任务指令下发的端到端延迟是多少？

### 9.2 分析顺序

第一段：报告快环计算时间相对预算的比例。

第二段：报告慢环计算时间、超时率、失败率和 MIP gap。

第三段：比较四个代表场景中的计算负载变化，并结合候选节点数、候选边数和实际视界解释。

第四段：回答 RQ4。结论应限定于：

```text
当前硬件
当前候选窗口
当前求解器设置
当前场景规模
```

不得由任务级仿真直接推断真实飞行平台已经完成部署验证。

### 9.3 推荐图表

#### 图 7：计算预算占用

分别展示快环和慢环：

```text
median / p95 / p99 / max
```

纵轴采用：

```text
computation time / loop budget
```

预算线为 1。

若跨越多个数量级，可使用对数坐标。

#### 表 10：实时性与资源统计

包括：

```text
scene
fast p95/p99/max
fast deadline-miss rate
slow p95/p99/max
solver timeout rate
solver failure rate
MIP gap
event-to-command latency
CPU
peak memory
```

---

## 10. 5.5 本章小结

本节只需 1–2 段，不重复全部数值。

按 RQ 顺序总结：

```text
RQ1：FDLC 是否有效，以及优势出现的条件
RQ2：事件更新与快环模式切换的增量作用
RQ3：rho 的性能–开销折中及事件补偿边界
RQ4：当前配置下的在线计算可行性
```

每个 RQ 用一句主结论和一句限定条件。

不要在本章小结引入新统计结果。

---

## 11. 结果写作规范

### 11.1 单段结构

结果段落优先采用：

```text
结论句
→ 关键数值与 CI
→ 统计证据
→ 辅助指标解释
→ 边界或限定
```

示例结构：

```text
FDLC 在高冲突场景中获得更高的有效任务完成率。
相对最优基线的平均差值为 ...，95% CI 为 ...，
Holm 校正后 p=...，效应量为 ...。
该差异主要伴随 ...，而不是 ...。
在 low-M0 中差异不明确，表明完整反馈机制的收益具有场景依赖性。
```

不得照抄示例中的占位内容。

### 11.2 统计措辞

只有校正后 \(p<0.05\) 时，才可使用：

```text
statistically significant
显著差异
```

否则使用：

```text
均值更高
呈上升趋势
差异未达到统计显著
置信区间跨越零
```

不得报告：

```text
p = 0.000
```

应写：

```text
p < 0.001
```

必须区分：

- 统计显著性；
- 效应量大小；
- 工程实际意义。

### 11.3 避免过度生成感

避免连续使用：

```text
结果表明
进一步说明
值得注意的是
显著提升
充分证明
```

优先直接写观察结果和证据。

不为每个表格单元写一句话。

每节只讨论：

- 主效应；
- 关键交互；
- 预设对比；
- 反例或边界。

### 11.4 因果与机制措辞

可以写：

```text
该结果与事件驱动更新减小状态滞后的设计目标一致。
```

谨慎写：

```text
差异可能与更及时的状态更新有关。
```

不得仅依据相关结果写：

```text
该机制导致了全部性能提升。
```

机制结论应由预设消融支持。

### 11.5 负结果

必须报告：

- FDLC 不占优的场景；
- 差异不显著的场景；
- 更高计算或反馈开销；
- 较大或较小 \(\rho\) 下的失效现象；
- 时延与完成率之间的权衡。

负结果用于界定适用范围，不应被删去。

---

## 12. 图片绘制规范

生成任何论文图前，必须读取并遵循：

```text
/home/yuhe/workspace/uav-distributed/Autonomous_Robots_图片样式与制作规范.md
```

若文件不存在、无法读取或内容冲突，停止绘图，不得自行猜测格式。

所有绘图脚本必须以：

```text
paper1_v2_
```

开头。

### 12.1 通用要求

- 主图优先输出 PDF 矢量格式；
- 位图仅用于热力图等确有必要的内容；
- 字体、字号、线宽、marker、图例顺序严格遵循规范文件；
- 同一方法在所有图中使用相同颜色、线型和 marker；
- 灰度打印后仍可区分；
- 不在图内放完整 caption；
- 不使用装饰性标题；
- 不使用 3D 柱状图、饼图或彩虹色图；
- 不显示未建模的 `No-fly zones`；
- 图中的误差范围必须是 seed-level 95% CI；
- 比例指标纵轴通常固定在 \([0,1]\)；
- 相关面板使用一致的坐标范围；
- 增益图使用以 0 为中心的发散色标；
- 热力图色标范围在相关面板间保持一致；
- 显著性符号只用于预设比较，并在 caption 中定义。

### 12.2 图像元数据

每张图同时生成：

```text
results_v2/metadata/chapter5/<figure_name>.json
```

至少记录：

```text
generator_script
git_commit
input_files
input_hashes
style_guide_path
style_guide_hash
created_at
figure_size
output_format
confidence_interval_definition
number_of_seeds
model_or_summary_source
```

### 12.3 推荐图片文件

```text
results_v2/figures/chapter5/
├── paper1_v2_fig3_overall_performance.pdf
├── paper1_v2_fig4_intensity_conflict_interactions.pdf
├── paper1_v2_fig5_coupling_ablation.pdf
├── paper1_v2_fig6_timescale_sensitivity.pdf
└── paper1_v2_fig7_runtime_budget.pdf
```

---

## 13. 表格设计规范

正文表格只保留回答 RQ 所需的汇总信息。

推荐：

```text
Table 7  Overall planned contrasts and marginal means
Table 8  Coupling-ablation planned contrasts
Table 9  Timescale performance–cost summary
Table 10 Runtime and resource summary
```

完整的：

```text
16 scenes × methods
all diagnostics
all pairwise tests
all model coefficients
```

放入补充材料。

表格要求：

- 均值与 CI 的小数位统一；
- \(p\) 值最多三位小数；
- 小于 0.001 写为 `<0.001`；
- 不用粗体代替统计检验；
- 最优均值可谨慎加粗，但必须在表注说明；
- 表格中的 `n` 表示 seed 数，不是 episode 数；
- caption 能独立说明指标、聚合层级和 CI 定义。

---

## 14. 自动生成脚本

至少生成或确认以下脚本：

```text
scripts/paper1_v2/
├── paper1_v2_validate_chapter5_inputs.py
├── paper1_v2_build_seed_analysis_dataset.py
├── paper1_v2_fit_overall_models.py
├── paper1_v2_run_scene_pairwise_tests.py
├── paper1_v2_analyze_coupling_ablation.py
├── paper1_v2_analyze_timescale.py
├── paper1_v2_analyze_runtime.py
├── paper1_v2_plot_overall_performance.py
├── paper1_v2_plot_interactions.py
├── paper1_v2_plot_coupling_ablation.py
├── paper1_v2_plot_timescale.py
├── paper1_v2_plot_runtime.py
├── paper1_v2_generate_chapter5_tables.py
└── paper1_v2_generate_chapter5_tex.py
```

优先拆分数据、模型、绘图与写作，禁止在一个脚本中混合所有逻辑。

---

## 15. 输出目录

```text
results_v2/
├── summaries/
│   ├── paper1_v2_episode_analysis.csv
│   ├── paper1_v2_seed_analysis.csv
│   ├── paper1_v2_scene_analysis.csv
│   └── paper1_v2_chapter5_claims.csv
├── models/chapter5/
│   ├── overall/
│   ├── ablation/
│   ├── timescale/
│   └── runtime/
├── tables/chapter5/
├── figures/chapter5/
├── metadata/chapter5/
└── text/chapter5/
    ├── paper1_v2_chapter5_draft.tex
    ├── paper1_v2_chapter5_evidence_map.md
    └── paper1_v2_chapter5_review_report.md
```

`paper1_v2_chapter5_claims.csv` 至少包含：

```text
claim_id
section
research_question
claim_text
metric
comparison
estimate
ci_lower
ci_upper
adjusted_p
effect_size
source_table
source_model
source_figure
status
```

每个正文定量结论必须能追溯到该表中的一行。

---

## 16. 证据映射

生成：

```text
paper1_v2_chapter5_evidence_map.md
```

结构：

| RQ | 结论 | 主指标 | 统计证据 | 图表 | 限定条件 |
|---|---|---|---|---|---|
| RQ1 | ... | R_task | ... | Fig. 3–4, Table 7 | ... |
| RQ2 | ... | R_task, N_replan | ... | Fig. 5, Table 8 | ... |
| RQ3 | ... | R_task, N_replan | ... | Fig. 6, Table 9 | ... |
| RQ4 | ... | p95/p99, miss rate | ... | Fig. 7, Table 10 | ... |

若某个 RQ 缺乏足够证据，必须标记为：

```text
insufficient evidence
```

不得用描述性均值强行补足。

---

## 17. 第五章 LaTeX 生成要求

输出：

```text
results_v2/text/chapter5/paper1_v2_chapter5_draft.tex
```

要求：

- 保留现有论文的符号和方法名称；
- 不重新介绍第三章已经定义的公式；
- 不重新解释第四章已有的实验设置；
- 每个小节以研究问题为主线；
- 图表首次出现后立即解释；
- 结果与机制解释相邻，但不把推测写成事实；
- 不写占位数值；
- 不产生与统计输出不一致的自然语言；
- 不在正文堆叠所有 p 值；
- 关键比较写入正文，其余放表格；
- `Full Coupling` 全部替换为 `FDLC`；
- 不出现 C1/C2；
- 不声称 backlog 进入慢环优化。

---

## 18. 质量审查

生成正文后执行四轮检查。

### 18.1 数值一致性

检查正文中的每个：

```text
mean
difference
percentage
CI
p
effect size
```

是否能在输出表或模型中定位。

### 18.2 研究问题覆盖

检查 RQ1–RQ4 是否均包含：

```text
直接答案
主要证据
统计不确定性
限制条件
```

### 18.3 过度解释检查

搜索并人工审查：

```text
prove
guarantee
always
universally
fully demonstrates
显著优于
充分证明
完全解决
```

### 18.4 冗余检查

删除：

- 对表中每个单元的逐项复述；
- 在多个小节重复出现的同一结论；
- 不回答任何 RQ 的诊断指标；
- 与第六章讨论重复的长篇局限性。

---

## 19. 最终验收标准

- [ ] 完整性报告通过；
- [ ] 实际协议与第四章一致；
- [ ] 统计以 seed 为独立单元；
- [ ] 所有 CI 均为 seed-level；
- [ ] 总体模型包含 Method × Intensity × Conflict；
- [ ] 消融模型使用 Periodic Goal、Event-driven Goal、FDLC；
- [ ] 时间尺度模型将 rho 作为分类变量；
- [ ] 预设比较经过 Holm 校正；
- [ ] 报告效应量和置信区间；
- [ ] 未将 episode 当作独立样本；
- [ ] 未使用旧实验数据；
- [ ] 未出现 Full Coupling；
- [ ] 未出现 C1/C2；
- [ ] 未声称 backlog 进入慢环优化；
- [ ] 所有新增脚本带 `paper1_v2_` 前缀；
- [ ] 所有图片遵循指定样式文件；
- [ ] 每张图均有元数据；
- [ ] 每个定量结论可追溯；
- [ ] RQ1–RQ4 均被直接回答；
- [ ] 报告不利结果和适用边界；
- [ ] 正文未逐项复述全部场景数值；
- [ ] LaTeX、表格、图片可由脚本重新生成。

---

## 20. 执行顺序

```text
1. 读取运行 Skill、第四章协议和完整性报告
2. 核对 seeds、episodes、场景、方法和 rho
3. 构建 episode/seed/scene 三层分析数据
4. 拟合总体混合效应模型
5. 执行场景内预设检验
6. 分析耦合消融
7. 分析时间尺度
8. 分析计算实时性
9. 生成证据映射
10. 读取图片样式规范
11. 生成论文图片和元数据
12. 生成正文表格与补充表格
13. 生成第五章 LaTeX 草稿
14. 执行数值一致性、RQ 覆盖、过度解释和冗余检查
15. 输出审查报告
```

任何一步失败，不得继续生成最终论文正文。
