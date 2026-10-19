总体结论：**MAJOR ISSUES**

这份设计比上一版明显更接近可投稿实验，但目前还不能作为 CVPR test 预注册直接冻结。主要问题不是“方向错”，而是若按本文现在的定义启动 test，审稿人仍可质疑主指标是否真在量 WHERE 跟随、外部 SOTA 是否公平、PGIF 功效是否足够，以及 §7 中大量关键机制仍未实现/未验收。

**MAJOR 1**
位置：§4.1 主指标；代码 [policy.py](/nvidia/hque/code/cyhe/uav-acot-track/acot/runtime/policy.py:270)、[metrics.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/metrics.py:103)、[episode.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/acot_client/episode.py:73)

问题：scorer 的承诺滞回与 policy 真正用于 WHERE 的承诺不是同一个对象。policy 只在 S2 tick 更新承诺，`commit_k=5` 等于约 3 秒；scorer 每 10 Hz 帧更新，`commit_k=5` 只有 0.5 秒。PGIF gate 还会在 policy 内直接写 `self.committed`，进一步绕过 hysteresis。

为什么重要：主指标被解释为“无人机承诺跟随的是不是目标”，但实际统计的承诺可能不是控制链路正在跟随的承诺。CVPR 审稿人会认为身份指标与闭环控制行为脱钩。

具体修改建议：要么把主指标改为直接读取 policy trace 中的 internal `committed` key；要么明确 scorer 指标只是“输出身份经独立滞回后的身份正确率”，并把 WHERE/任务指标设为共同主指标。若继续用 scorer hysteresis，应把单位定义清楚，并与 S2 决策频率对齐，例如 `commit_k = 5 * s2_period` 或只在 S2 tick 计分。

**MAJOR 2**
位置：§4.1、§5.1、§9；代码 [metrics.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/metrics.py:203)、[acot/analysis/metrics.py](/nvidia/hque/code/cyhe/uav-acot-track/acot/analysis/metrics.py:5)、[report.py](/nvidia/hque/code/cyhe/uav-acot-track/acot/analysis/report.py:14)

问题：新增的 TR90 与 SR(τ) 在当前分析代码中还不存在。scorer 只有固定 `success`，没有 τ 曲线；分析报告只输出 `correct/correct_l/correct_s/longC/longW`。

为什么重要：文档正确指出主指标不惩罚更早丢失目标，但补救指标目前只是设计文字。若正文仍以现有报告生成，CVPR 审稿人会抓住“主指标选择性乐观”。

具体修改建议：T0 必须加入 TR90、SR(1/2/5/10)、AUC-SR 的单元测试与 golden 示例。建议把 TR90 设为 co-primary 或至少作为 H-M1/H-M2 的 confirmatory safety metric：Ours 在身份主指标显著时，TR90 不能显著变差。

**MAJOR 3**
位置：§5.1 H-M2、§6.1

问题：“Ours > 最强外部基线”的定义不严谨。H-M2 的 dev-B 候选只写 DAM4SAM、OA-VAT、DeepSORT，排除了表中 re-ID、OC-SORT、UAST、lang_xattn。这样 test 上若其他外部臂更强，“最强外部基线”这个 claim 会失效。

为什么重要：这是非常典型的审稿拒稿点：用 dev 选择一个子集中的“最强”，但正文表又展示更多外部基线。

具体修改建议：改成二选一：第一，H-M2 明确命名为“Ours > dev-B 预选外部基线”，不要叫最强；第二，更好的是对所有外部基线做 max-over-baselines 的闭合/置换检验，或把 H-M2 改为 Ours 对每个外部基线的 Holm 校正族。

**MAJOR 4**
位置：§5.7 OA-VAT/UAST 移植原则

问题：OA-VAT/UAST 被作为 CVPR’26 外部 SOTA，但设计只移植 WHICH 身份模块，替换检测器、候选集、飞行端/搜索端和部分目标指定方式。这个可以作为“身份模块适配”实验，但不宜作为完整外部 SOTA 比较支撑 H-M2。

为什么重要：主动跟踪论文的贡献常在搜索、规划、检测-跟踪耦合。只移植身份模块可能既不忠实，也可能因共享真值候选框而改变难度。审稿人会认为这是非官方、非等价复现。

具体修改建议：正文主表命名必须固定为“OA-VAT identity adaptation / UAST identity adaptation”。H-M2 不要依赖它们作为“外部 SOTA”结论，除非补充：原仓库默认输入的回放一致性、至少一个其原设定 benchmark sanity check、移植失败/缺权重的预注册排除规则。DeTrack 不纳入基本合理，但需要写成统一标准：无代码、身份机制非核心、目标指定不匹配，而不是只说“未找到”。

**MAJOR 5**
位置：§8 样本量与功效、§5.2

问题：PGIF 的预期效应约 +0.09，正好处在检出边缘；但 §8 的 MDE 表只列了 `α=0.0125(4 个比较)`，而表 2 实际有 5 个 Holm 假设。2 次重复、共同随机数、共享开局带来的降方差目前还是假设。

为什么重要：如果 test 后 PGIF 不显著，主张 C2/C3 会被迫收窄；如果显著性计划看起来事后调整，审稿人会质疑功效与预注册。

具体修改建议：按每个比较族分别重算 MDE：表1 3 个、表2 5 个、表3 2 个。T0 A/A 后必须在看 test 前冻结最终重复次数；若配对 SD 未降到预期，应预注册将 PGIF 部署 claim 降级为探索/机制证据，或增加 n/repeats。

**MAJOR 6**
位置：§3.4 共享开局、§5.2 机制实验、§7 F7

问题：共享开局只在开局窗口结束时注入参考 actor，并不保证各机制臂在前 12 步拥有相同物理状态、相同相机轨迹和相同候选可见性。若某臂在开局期间动作不同，tick 12 的候选集合和 fallback 就是处理后的结果。

为什么重要：机制实验的目标是隔离 PGIF/记忆组件；若开局前轨迹已分叉，组件效应会混入早期控制差异。

具体修改建议：机制实验应在开局窗口内统一控制与身份输出，或保存/恢复参考开局结束时的 simulator state；做不到时，至少强制所有机制臂在 tick 0/6/12 使用同一参考 actor 和同一 WHERE，之后才释放组件差异。fallback 不能只报告，需预注册处理规则，例如 fallback 种子全臂剔除或单独层分析。

**MAJOR 7**
位置：§7 代码检查、§11 T0；代码 [sim_proxy_v2@1.yaml](/nvidia/hque/code/cyhe/uav-acot-track/acot/configs/protocols/sim_proxy_v2@1.yaml:19)、[perception.py](/nvidia/hque/code/cyhe/uav-acot-track/acot/runtime/perception.py:23)、[scheduler.py](/nvidia/hque/code/cyhe/uav-acot-track/acot/eval/scheduler.py:126)

问题：§7 的关键待做项太多，且当前代码确实未实现：`sim_proxy_v2@2` 不存在；默认协议仍无 `assign: hungarian`；共享开局、统一 seed 钩子、`force_conf/gate_commit`、re-ID/oracle、OA-VAT/UAST、新指标都未接入。scheduler 当前仍按 arm 顺序规划，没有随机打乱；DAM4SAM 服务也总取第一个实例。

为什么重要：这不是普通工程债，而是影响公平性、可复现性和主表能否运行的实验条件。CVPR 方法学审计会要求 test 前这些都完成并冻结。

具体修改建议：把 §7 从“代码检查结果”改成硬性 go/no-go 表：每一项对应 commit hash、单元测试/回放 parity、配置哈希、失败时是否取消对应实验臂。尤其是调度随机化、DAM4SAM 多实例、`uses_gt` 断言、TR90/SR 分析必须在 test 前通过。

**MAJOR 8**
位置：§3.5、§3.6、§8

问题：2 次重复与共同随机数的定义不完整。文档说每集 reset 按 `(种子, 臂无关常数)` 设 torch/numpy RNG，但没有 repeat id/salt；如果两次重复使用同一 RNG salt，则重复不是独立重复，只是在 CARLA 非确定性上重复。scheduler 也尚未保证臂、种子块、服务器随机平衡。

为什么重要：重复次数用于降低方差和提高功效。如果重复的随机结构不清楚，方差估计可能偏乐观，p 值也会被质疑。

具体修改建议：预注册 `(seed, repeat, stream)` 的 RNG 派生规则，并在分析中以 seed 为主单元、repeat 取均值或混合模型。调度日志必须记录随机化顺序、服务器、GPU、TM 端口、重试历史；A/A 要覆盖 repeat salt 和服务器维度。

**MINOR 1**
位置：§6.1 基线调参

问题：外部基线只调 `conf_tau/gate_px/λ` 等小网格，而 Ours 经历多轮历史开发。文档已披露，但对 OA-VAT/UAST 这类复杂方法，小网格可能不足。

为什么重要：审稿人可能认为外部基线 under-tuned。

具体修改建议：给每个外部基线保留“原默认 + 小网格”的完整 dev-B 表，并报告超参敏感度；不要声称完整复现原论文性能。

**MINOR 2**
位置：§4.1 代价

问题：代价只写身份模块耗时/显存，不足以公平比较 DAM4SAM、OA-VAT、UAST 这类服务式或检测式方法。

为什么重要：CVPR 审稿人会问端到端 tick latency、GPU 占用、服务并发、是否含 crop/DINO/SAM/DINOv3。

具体修改建议：报告三层代价：identity-only、policy S2 tick end-to-end、整集 wall-clock/GPU memory。共享前端成本单列，外部服务成本单列。

**MINOR 3**
位置：§5.2、§7 F4

问题：`gate_commit` 在 1004 旧设计中是一个重要拆分臂，但 1006 表 2 删除了它，§7 F4 又仍写“需要的 `gate_commit`、`force_conf`”。表述不一致。

为什么重要：会让实现人员和审稿人不清楚 PGIF 的贡献到底拆到哪一层。

具体修改建议：若不做 w/o 承诺门控，就从 §7 F4/G4 删除 `gate_commit`；若保留，则加入表 2 并声明探索性、不进 Holm 主族。

**MINOR 4**
位置：§5.7、§5.6

问题：章节顺序是 5.7 在 5.6 前；T0 表中 G13 排在 G12 前。小问题，但预注册文档需要显得非常干净。

具体修改建议：重排编号，避免给人“临时拼接”的印象。

**总体建议**

这份文档可以作为 T0 实施蓝图，但还不应作为最终 CVPR test 预注册。最优先修正三件事：对齐“主指标承诺”与 policy WHERE 承诺；把 TR90/SR 和统计功效真正落到代码与预注册；把外部基线从“完整 SOTA 比较”降格或严格验证为“identity adaptation”。完成这些后，设计的可信度会明显上一个台阶。