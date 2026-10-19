verdict: **NO MAJOR**

可以冻结 v4 预注册稿并进入 T0。注意：这是“预注册设计可冻结、T0 门槛足够”的判定，不是允许启动 test；test 仍必须等 G1–G10 全部通过并再审计。

**R3 问题逐条核对**

1. **表 3 选模规则矛盾：已解决。**  
   v4 第 1 节、第 6 节、G4 都统一为“固定验证划分上的最佳 val checkpoint”。这与当前 [tah_train.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah_train.py:85) 的 `best val` 保存逻辑一致。未再发现“last epoch”残留。

2. **外部基线隐藏 GT 特权：已解决。**  
   v4 表 1 明确所有 deployable 臂都必须追加 `--gallery-seed committed --reanchor off`，第 15 节外部基线也用 `+ DEP`，并新增 G10。代码抽查确认风险真实存在：`policy_server.py` 默认仍是 `--gallery-seed gt`，见 [policy_server.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy_server.py:139)，所以 G10 是必要且充分的 T0 门槛。

3. **表 3 p 值偏乐观：已解决。**  
   v4 已把表 3 的主判定改为两层 bootstrap 的 Bonferroni 校正 CI，p 值仅作描述，并明确是“以这 3 个训练种子为条件”的 episode 层检验。这个口径足够保守，能避免把 `3 × 40` 当作 120 个独立样本。

4. **功效表 Monte Carlo 小差异：基本解决，MINOR。**  
   当前 `power_sim.py` 输出与 v4 主表/敏感性表大体一致。但文档写“每格 20,000 次模拟”，脚本默认仍是 `sims=4000`，且没有命令行 sims 参数。这不影响设计判定，但建议 T0 前把脚本默认或记录口径改一致。

**全文通读结论**

未发现新的 MAJOR 前后矛盾。部署/机制实验边界、共享开局注入语义、分层变量、排除规则、批次交错、A/A、元数据、日志字段、分析脚本冻结、隐藏特权防线都已经写清楚。

剩余只是 MINOR/执行提醒：

- 第 3.2 节“开启 PGIF 时 pred_slot 为 ŝ”最好补一句 `gate_commit=false` 是例外变体，否则和表 2 文字略有张力。
- 表 4(a) 有方向性预测，但第 10 节多重比较族只列表 4(b)；建议注明表 4(a) 只报效应量与 CI，不作显著性判定。
- T0 必须真的实现并审计 G10 的负例测试，因为服务端默认 `gt` 是高风险默认值。

**最终判定**

v4 可以冻结，进入 T0。  
不能启动 test；test 前必须通过 G1–G10，尤其是 G2/G7/G9/G10。