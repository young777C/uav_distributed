verdict: **MAJOR**

v3.1 已经比 R2 后的版本更接近可冻结稿，但我不建议直接冻结并进入 T0。问题不是“代码尚未实现”，而是有几处预注册设计/配置仍然自相矛盾或门槛不足以防止 test 误跑。

**R2 问题核对**

1. 共享开局注入语义：**基本解决**。v3.1 明确要求在 PGIF 前替换 s^M，并驱动记忆写入、PGIF 首证据、`pred_slot`、policy commitment；G2 也能验。  
2. TAH 重训配方：**仍未完全解决**。第 1 节说沿用 best val checkpoint，[1004_ablation_design.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1004_ablation_design.md:34)；但第 6 节又写“固定取最后一个 epoch”，[1004_ablation_design.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1004_ablation_design.md:195)。代码当前是 best val，[tah_train.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah_train.py:85)。这是冻结前必须修的 MAJOR。
3. p 值定义：**解决**。已改为配对符号翻转置换检验，bootstrap 只给 CI。
4. 功效悲观敏感性：**基本解决**。`power_sim.py --sens` 存在且区间与文档大体一致；默认表有小幅 Monte Carlo 差异，例如 n=60/shared/Δ=.15/α=.0125 脚本约 0.83，文档写 0.86，不构成 MAJOR，但建议固定更大 sims 后更新表。
5. 主比较族理由：**解决**。w/o PGIF 与 DAM4SAM 的二元主比较族理由清楚。
6. 代码未实现：本轮不作为 MAJOR；v3.1 用 G1–G9 接住了大部分。

**代码索引抽查**

1. TAHRelM 特征索引基本准确。7 维 `[cos, relp_x, relp_y, depth, rank, dmot, msoft]` 在 [tah.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah.py:121)，参数量 4,740 的说法成立。
2. TAH 训练索引准确但状态未实现：当前 `--cache` 单文件、无 `--seed`、无 `--feat-mask`、无配方写入，best val 保存位于 [tah_train.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah_train.py:56) 和 [tah_train.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah_train.py:90)。
3. PGIF `_filter_step` 索引准确：tempered/resid/bayes 分支在 [asd.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/asd.py:190)。`gate_commit` 仍不存在，文档已列待实现。
4. policy TAH 分支、PGIF 门控索引基本准确：TAH 部署在 [policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy.py:651)，PGIF fire 后当前总会改 `pred_slot` 和 `_committed_idx`，在 [policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy.py:831)。
5. re-ID / assoc / DAM4SAM 代码支持 deployable committed 模式，但预注册表格配置不够安全。`policy_server.py` 默认 `--gallery-seed gt`，[policy_server.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy_server.py:139)；而表 1/15 的外部基线配置行没有显式写 `--gallery-seed committed --reanchor off`。实际 `runs/asd/deploy_all/chains.sh` 传了 `DEP="--gallery-seed committed --reanchor off"`，[chains.sh](/nvidia/hque/code/cyhe/uav-acot-track/runs/asd/deploy_all/chains.sh:6)，但冻结稿不能依赖脚本习惯。否则复现者按表格命令会给 re-ID/assoc/DAM4SAM 隐藏 GT 特权。
6. scorer 滞回实现存在，主指标来源准确，[metrics.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/metrics.py:92)；但 `scorer_committed_key` 尚未逐帧落盘，G7 已覆盖。
7. `RemotePolicy.reset()` 和 server reset 仍不传 seed，[remote_policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/remote_policy.py:23)、[policy_server.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy_server.py:65)。文档已列 T0 待实现，非本轮 MAJOR。

**v3.1 新问题**

1. **表 3 选模规则自相矛盾**：best val vs last epoch，必须统一为 best val，或统一改代码与 G4。当前应按 best val。
2. **外部基线命令有隐藏 GT 默认风险**：表 1、第 15 节配置行必须显式加 `--gallery-seed committed --reanchor off`，并把 G9 改成“选定参数和完整最终命令写入本文件”。否则 T0 不一定发现误用默认 `gt`。
3. **表 3 p 值仍偏乐观**：第 10 节把 3 个训练种子先平均成每个 episode 的差再做翻转，这给的是“3-seed 平均模型”的环境种子检验，不充分反映训练种子随机效应。建议表 3 主报告两层 bootstrap CI；p 值降为描述性，或改成训练种子层面的保守检验。

**判定**

当前 v3.1：**不能冻结**。  
修完上述 3 个文档/门槛问题后，可以进入 T0；无需等代码实现完成再修预注册稿。最关键的冻结前改动是：统一 TAH 选模规则、外部基线全量显式 committed 配置、补强 G9/表 3 统计口径。