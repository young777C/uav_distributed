总体结论：**MINOR ISSUES ONLY**

v3 已经充分处理 R2 的 2 项 MAJOR 和 3 项 MINOR。作为 **T0 实施蓝图与预注册草案**，设计本身已基本达到 CVPR 投稿的方法学标准；但仍需在 §10 go/no-go 全部通过后，才适合冻结为正式 test 预注册。

**R2 各项解决状态**

| R2 项 | 状态 | 复审意见 |
|---|---|---|
| MAJOR 1：A/A 不应单独决定样本量 | **已解决，余一处文字小歧义** | §7.3 已改为 dev-B A/B 预实验、历史 A/B 下限、A/A 技术噪声三者取最大，并有 r 冻结与降级规则。方向正确。 |
| MAJOR 2：w/o 回火不是单变量 | **已解决** | 核对 `acot/runtime/pgif.py`：在 `tempered/resid` 分支中，`resid_decay` 只控制历史 belief 衰减；若 `filter_kind` 保持 `tempered` 且仅设 `resid_decay = 1.0`，置信加权 `0.5 + rank(conf)`、absent-hold、gate_memory 均保留。因此这是回火/衰减的单变量消融。 |
| MINOR 1：TR90 门槛偏弱 | **已解决** | §4.2 与 §8.4 已改为非劣效：TR90 配对差 95% CI 下界 `> -0.02`，且适用于表 1、表 2、表 3(a) 的全部确认性比较。 |
| MINOR 2：臂数不精确 | **基本解决** | 表 1 已明确为 13 或 14 个独立臂；诊断上界 U1–U3 拆清楚。算力表还有小的重复数口径问题，见下。 |
| MINOR 3：参考开局定位 | **已解决** | §3.4 明确参考开局是实验干预，不用于调参/选型，输出在机制臂运行前固定。 |

**剩余或新增问题**

1. 位置：[§7.3](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1006_experiment_design_cvpr.md:263)

   问题：保守方差冻结规则已可接受，但 `r` 的选择还有一点可操作性歧义。§7.3 先用 dev-B A/B 每臂 `r = 2` 估计配对差 SD，随后又在 `r ∈ {2,3,4}` 中选择最小 r；但没有明确当候选 r 为 3 或 4 时，(a) 项的 A/B SD 如何外推，还是固定使用 r=2 的保守上界。

   为什么重要：这是预注册执行细节。若 T0 后从 r=2 改到 r=3/4，审稿人需要能复现冻结规则，而不是依赖解释。

   具体修改建议：在 §7.3 加一句：`SD_AB_UCB` 对所有候选 r 固定使用 dev-B r=2 均值差的 80% 上界，或明确写出从 r=2 外推到 r 的公式；同时定义 ρ 的估计式和 CI 口径。

2. 位置：[§9](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1006_experiment_design_cvpr.md:295)

   问题：算力估计中 T2 的表 4 似乎没有乘以 `r = 2`。§9 开头说按 `r = 2` 估计，但 T2 写 `表 4:6 × 60`，若也重复两次应为 `6 × 60 × 2`。T3 是否应用 r 也不清楚。

   为什么重要：不影响方法学有效性，但会影响资源排期与 go/no-go 执行可信度。

   具体修改建议：把 §9 拆成“确认性主表按 r 重复”和“低优先级/附录是否重复”的两列；对 r=2/3/4 给出自动更新后的总集数，UAST 条件纳入也单列。

3. 位置：[§5.4](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1006_experiment_design_cvpr.md:177)、[§9](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1006_experiment_design_cvpr.md:302)

   问题：表 4 的 “6 × 60” 条件划分不完全显式。文本列出关联噪声 `p ∈ {0.02, 0.05}` 与检测噪声，但没有明确 6 个条件是否为 `2 臂 × 3 噪声条件`，以及 clean 条件是否另算。

   为什么重要：边界实验虽非确认性主张，但预注册表应让实现者和复审者一眼知道要跑哪些配置。

   具体修改建议：在 §5.4 加一个小表：`clean / assoc p=.02 / assoc p=.05 / detection noise` 是否分别跑 Full 与 w/o PGIF；若 clean 已由表 2 提供，也注明“不重复跑，引用表 2 clean 条件”。

除此之外，§7.3 的保守方差原则、TR90 δ 与判定规则、表 2 的 Holm 家族和探索臂划分都已经清楚。当前版本不再有会阻止 T0 实施的 major 设计缺陷。