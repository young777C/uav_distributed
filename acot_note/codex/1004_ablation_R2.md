**Verdict: MAJOR**

结论先说清楚：v2 作为预注册稿比 v1 明显强很多，R1 的大多数设计层面问题已回应；**可以开始 T0，也应该开始 T0**。但它还**不能进入 test 正式运行**。当前 MAJOR 主要不是“论文设计完全不成立”，而是统计口径和实现承诺还没有闭合，尤其共享开局、TAH 特征掩码/重训、bootstrap p 值、通用 tick 日志这些都会直接影响 test 结果解释。

**R1 MAJOR 逐条核对**

1. **已解决。** 表 1 已加入自然开局 `TAHRelM w/o PGIF`，能支撑部署层面的 PGIF 自消融。
2. **已解决。** v2 明确区分部署实验与共享开局机制实验，并写明共享开局不代表独立部署表现。
3. **设计已解决，代码未实现。** v2 改为按 `Candidate.idx` 存 actor 身份，方向正确；代码里候选确实每帧 shuffle，slot 不能用，[candidates.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/candidates.py:76)。但 reset 传 seed / `OPENING_FILE` 注入尚未实现。
4. **已解决。** 分层变量改成参考开局是否正确，避免用各臂自身行为做 post-treatment 分层。
5. **部分解决。** 已新增双峰功效模拟 [power_sim.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/power_sim.py:1)，表中数字基本可复现；但脚本用正态近似检验，不是正文统计计划的 bootstrap p 值，而且只模拟降低 `h`，未做 p、flip、r、批次相关性的敏感性分析。功效仍偏乐观。
6. **已解决但偏弱。** 表 3 明确训练种子为模型随机效应、episode 为环境随机效应，两层 bootstrap 正确；但只有 3 个训练种子，稳定性结论应写得保守。
7. **部分解决。** 特征掩码比 v1 好，结构/参数量不变的方向正确；但 `w/o 相对运动` 是一组运动特征块，不是严格单一特征；`w/o DAgger` 仍改变训练数据分布。当前代码也还没有 `--feat-mask`，TAHRelM 直接构造 7 维输入 [train/tah.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah.py:121)。
8. **基本解决。** 外部基线改名为 `-style`，并给 dev 网格，公平性显著改善。仍建议正文明确“外部基线是同一候选框/控制链路下的机制对照，不声称完整 SOTA 复现”。
9. **已解决。** DAM4SAM 弃权率和原因进入主表/主文。
10. **部分解决。** v2 已把缺失代码列入第 12 节，但代码仍缺：reset seed、共享开局、通用 tick log、结果元数据、噪声注入、`gate_commit`。例如 `RemotePolicy.reset()` 仍不传 seed，[remote_policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/remote_policy.py:23)；server reset 也不接 seed，[policy_server.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy_server.py:65)。
11. **设计已解决，代码未完全实现。** v2 澄清 policy commitment 与 scorer commitment；但 scorer 目前只内部维护 `_committed`，没有逐帧写 `scorer_committed_key`，[metrics.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/metrics.py:92)。
12. **已解决。** T1/T2/T3 episode 数现在基本自洽，T3 的 `(Full+4)×3×40 + 手工规则×40` 算作 640 是对的。

**R1 MINOR 逐条核对**

1. **已解决。** 第 12 节同时写 `train/tah.py` 与 `train/tah_train.py`。
2. **已解决。** 元数据字段已列出。
3. **已解决。** 自然 A/A 与共享 A/A 都有。
4. **已解决。** 已要求按 seed block / arm block 交错调度。
5. **已解决。** 排除规则包含场景级 skip、一致性检查、基础设施故障补跑并保留日志。
6. **已解决。** 负结果披露进入正文局限/附录。
7. **已解决。** 旧 n=19 图数字已要求删除或替换。

**v2 新引入 / 仍需修的关键问题**

1. **共享开局注入语义还不够精确。** 现在写“直接承诺参考 actor”，但机制实验真正需要固定的是首 tick 的 `s^M`、memory seed、PGIF 初始证据和 policy commitment。若只改 `_committed_idx`，而 TAH memory 仍按本臂语言选择更新，就不是真正共同开局。T0 实现时必须规定：首个决策 tick 在 PGIF 前把 raw memory pick 替换为参考 actor slot，并用它完成 memory update / PGIF evidence / pred_slot / commitment。
2. **TAH 重训配方与现有代码冲突。** v2 说固定取最后 epoch，但当前 `tah_train.py` 保存的是 best val checkpoint，[train/tah_train.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah_train.py:90)。v2 还说合并离线 64 ep + DAgger 23 ep，但当前 `--cache` 只读一个文件。这些必须在 T0 前改实。
3. **bootstrap p 值定义不完整。** v2 写“bootstrap 得到 Holm 校正后的 p 值”，但未定义 p 值算法。建议预注册为：配对符号置换/随机化检验给 p 值，bootstrap 只给 CI；或明确定义 bootstrap p 的计算方式。
4. **功效模拟仍偏乐观。** `h_full=0.95, r=0.05, flip=0.10` 固定，且没有参数不确定性；自然开局下 n=80、Δ=0.15、α=0.0125 功效只有约 0.69，低于主表稳健标准。建议补一个悲观敏感性表：`p∈{0.45,0.55}`, `flip∈{0.10,0.15}`, `h_full∈{0.85,0.95}`。
5. **主比较族可能被质疑为挑选。** 只把 `w/o PGIF` 与 DAM4SAM 放入表 1 主比较族是可辩护的，但需要一句预注册理由：一个是核心机制自消融，一个是最强外部记忆跟踪器；其他为次要机制/风格基线。
6. **代码目前无法支撑 test。** `gate_commit` 还不存在，PGIF fire 仍会改 `_pred_slot` 和 `_committed_idx`，[policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy.py:831)；候选噪声注入没有接口；结果 JSON 不写第 1 节元数据；逐 tick 通用日志也没有覆盖所有 tid head。

**是否可以进入 T0 / test**

可以开始 **T0**：目标就是把第 12 节这些代码和冒烟补齐。  
不能开始 **test**：必须先完成 T0，并在 dev 上通过至少这几项门槛：

- 新开关全关时，5 个 dev seed 的 `pred_slot` 与 scorer 输出逐位一致，或不超过冻结版本 A/A 噪声。
- 共享开局日志能证明首 tick 的 raw pick、memory seed、PGIF 输出、policy commitment 都来自同一个参考 `actor_idx`。
- `--feat-mask --seed`、合并 cache、last-epoch 保存、训练命令写 ckpt 全部可复现。
- bootstrap/置换检验分析脚本冻结。
- JSON 元数据、tick log、fallback、abstention、scorer commitment 都可从一次 dev batch 复算出来。

所以我的审稿判断是：**预注册方向已基本可接受，但当前仍是 MAJOR，原因是 test 前闭环实现与统计分析口径尚未达到可审计标准。**