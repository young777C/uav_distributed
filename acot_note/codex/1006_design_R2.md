总体结论：**MAJOR ISSUES**

v2 相比 R1 有实质进步：R1 的大多数公平性、外部基线命名、共享开局和 go/no-go 问题已经被设计层面吸收。按“这是 T0 实施蓝图与预注册草案”来审，不要求当前代码已实现是合理的。但设计本身仍有 2 个会影响确认性结论的主要问题，暂不建议冻结为 CVPR test 预注册。

**R1 逐项核实**
| R1 项 | 状态 | 简评 |
|---|---|---|
| MAJOR 1 主指标与 policy WHERE 承诺脱钩 | 已解决 | §4 改为读取 policy 内部承诺，旧 scorer 滞回降为对照口径；G6 要求实现与 golden test。 |
| MAJOR 2 TR90 / SR 未落地 | 已解决 | §4.2/§4.4/G6 已把 TR90、SR(τ)、AUC-SR 纳入 T0 必做；当前代码未实现但由 go/no-go 把关。 |
| MAJOR 3 “最强外部基线”定义不严谨 | 已解决 | §5.1 族 B 改为 Ours 对所有外部臂逐个 Holm，只有全部成立才写“优于所有”。 |
| MAJOR 4 OA-VAT / UAST 非完整 SOTA 复现 | 已解决 | §5.6 固定命名为 identity adaptation，并声明不是原系统复现；有一致性核查和排除规则。 |
| MAJOR 5 PGIF 功效 / MDE | 部分解决 | MDE 数值和族数基本正确，但 §7.3 用 A/A 估计 A/B 配对差 SD 的冻结规则仍不充分，见下方主要问题 1。 |
| MAJOR 6 共享开局不能隔离机制 | 已解决 | §3.4 改为开局窗口内统一身份、承诺、WHERE，fallback 全臂剔除；设计上可行。 |
| MAJOR 7 待实现项太多 | 已解决 | §10 已改成硬性 go/no-go 表，符合 T0 蓝图定位。 |
| MAJOR 8 重复与随机数定义不完整 | 已解决 | §3.5/§3.6 定义了 seed、repeat、stream、调度日志与服务器记录。 |
| MINOR 1 基线调参 | 已解决 | §6.1 加入原默认 + dev-B 小网格与敏感度附录。 |
| MINOR 2 代价指标 | 已解决 | §4.2 三层代价定义足够清楚。 |
| MINOR 3 `gate_commit` 表述不一致 | 已解决 | §5.2 作为探索臂列入，不进 Holm。 |
| MINOR 4 编号 / 顺序 | 已解决 | 章节与 T0 表已重排。 |

**剩余 / 新增问题**

**MAJOR 1：§7.3 的功效冻结规则把 A/A 噪声当成 A/B 配对差方差，可能低估 MDE。**  
位置：§3.5、§7.1–§7.4。  
问题：§7.3 写“用 T0 A/A 得到新口径下的配对差 SD”，但 A/A 只能估计同一 arm 重跑的技术噪声、服务器噪声和 repeat 噪声，不能估计 Full–ablation、Ours–baseline 的种子级 treatment heterogeneity。表 2 的 PGIF 效应本来约 +0.09，若用 A/A 低估方差，就可能错误地保留确认性 PGIF 假设。  
为什么重要：这是确认性统计设计问题，不是代码实现问题。若 test 后 PGIF 显著性边缘，审稿人会质疑样本量是在过低方差假设下冻结的。  
具体修改建议：§7.3 改为使用“保守方差冻结”：以历史 A/B 配对 SD、新口径 dev-B 小规模 A/B pilot、A/A 技术噪声三者中的保守上界来定 r。A/A 只用于估计 repeat/server 噪声，不单独决定 A/B MDE。若不跑 dev-B A/B pilot，则直接用 §7.1 的 0.38–0.41 作为冻结 SD 下限。

**MAJOR 2：§5.2 “w/o 回火”不是单变量消融，当前设计不能严格支撑“回火必要”。**  
位置：§5.2；代码 [pgif.py](/nvidia/hque/code/cyhe/uav-acot-track/acot/runtime/pgif.py:42)、[ours_pgif_bayes.yaml](/nvidia/hque/code/cyhe/uav-acot-track/acot/configs/methods/ours_pgif_bayes.yaml:1)。  
问题：v2 说 `filter_kind = bayes` 与 `w/o absent-hold` 比较“只差回火一个变量”。但代码里 `tempered/resid` 是衰减累积分数，`bayes` 是 sticky HMM posterior，涉及 `q_lo/q_hi/eps_switch`、归一化和似然形式变化；这不是仅移除 tempering。  
为什么重要：C3 是机制主张。若 confirmatory H-P3 显著，只能说明“tempered 累积实现优于当前 bayes 变体”，不能说明“回火”这一因素本身必要。  
具体修改建议：二选一：把 H-P3 改名为“tempered PGIF > bayes PGIF 变体”，不再声称单变量回火；或新增真正单变量臂，例如同一 additive update 中固定温度 / 固定权重版本，并把 H-P3 建在该对比上。

**MINOR 1：TR90 安全门槛表述偏弱，且适用范围略不一致。**  
位置：§4.2、§5.1、§8.4。  
问题：§4.2 写 TR90 差的 95% CI 上界 > 0 即“安全”。这等价于“没有显著变差”，但即使均值明显为负、CI 很宽也会通过。另 §4.2 说每个“Ours 更好”，§8.4 又像是所有 confirmatory 比较都要加 TR90 gate，表 2 的 Full-vs-ablation 是否适用不够明确。  
建议：定义每个确认性比较的方向为 left minus right；TR90 gate 明确适用于所有确认性“left better”结论。若想更像 safety criterion，建议改为非劣效门槛：TR90 95% CI 下界 > -δ，并预注册 δ。

**MINOR 2：表 1 / 算力估计的 arm 数需要精确化。**  
位置：§5.1、§9。  
问题：表 1 的“诊断上界”文字包含真值身份、真值刷新记忆、真值目标不在，可能是 1 行也可能是 3 个 arm；UAST 又是条件纳入。§9 写表 1 为 11 臂，只在某一种解释下成立。  
建议：在 §5.1 明确每个诊断上界是否为独立 arm；§9 用 “10/11/13 臂，取决于 UAST 与上界项” 或拆成固定 + 条件集数。

**MINOR 3：共享开局的 test-seed 参考运行应声明为实验干预而非开发运行。**  
位置：§3.4。  
问题：用 Ours 在 test 种子上先跑参考开局是可以接受的，因为机制表本来是特权共享开局；但需要更明确说明参考运行不用于调参、不查看主结果，只产生预注册的 `shared_opening_actor` 和 fallback 标记。  
建议：在 §3.4 加一句“参考开局输出在任何机制臂运行前固定，不用于模型选择或超参调整；其正确性只在分析阶段作为协变量报告”。

结论上：v2 已经解决 R1 的主体问题，但 §7.3 和 §5.2 这两处会直接影响确认性统计和机制解释。修掉后，我会倾向给 **MINOR ISSUES ONLY**。