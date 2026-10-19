verdict: **MAJOR ISSUES**

**MAJOR**

1. **R1 未完全解决。** Novelty 叙事从 ASD 锚/CUSUM pivot 到 PGIF，是正确方向；命题1也已降格。但 R1 的“闭环未闭”和“memory-agnostic 有落差”仍在。更严重：`asd.py` 支持 `mode="filter"`，但日志显示 `policy_server.py --asd filter` 被 parser 拒绝，闭环 PGIF 目前甚至未真正跑通。

2. **PGIF novelty 仍不够 ACCEPT-READY。** 相对 MHT/identity management，本质是单目标 sticky Bayes/HMM over tracklet ids；相对自训练确认偏差，是在线伪标签平滑；相对 TLD，是把外观锚换成 tracklet-id prior。新意可成立为“闭环自更新记忆的 posterior-gated update wrapper”，但必须用实验证明“反馈门控更新”而非普通 temporal smoothing 才是关键。4 小时内最可行增强：同一 observe replay 加三消融：`rank-q PGIF` vs `constant-q HMM` vs `commit-only no memory-gate` vs `last-stable/longest-residence heuristic`；并闭环跑至少 TAH 5-10 ep 验证 parser 修复后 `gate update` 带来 max_wrong/latch 下降。

3. **命题2 逻辑闭环不严。** D1-D4 支持“置信度弱可用、锚失败、tracklet 持续性有信号”，但命题2 假设 B2 独立性是 PGIF 希望造成的结果，不是已证明前提；这里有循环论证风险。观测确实来自被门控的同一记忆，闭环后分布会被后验动作改变。建议改成“受控假设下的解释性模型”，删“指数纠错”强口吻，闭环用 max_wrong/latch 实证检验 B2 是否近似成立。

4. **tracklet id 依赖是当前方法最大外部假设。** 不致命，但如果只用模拟器真 id，对 CCF-A 方法向基本致命：等于把最难的关联层 oracle 化。处理：正文明确 PGIF 假设输入为 MOT persistent id；主结果必须至少有真实/算法 tracklet 层，或在 replay 中注入 ID switch/drop/missed detection，画性能随 switch rate ε 退化，并报告与 DeepSORT/OC-SORT ids 的闭环/回放结果。

5. **闭环闭合标准必须前置。** 至少同 seeds 报告 off / lang reanchor / anchor-ASD / PGIF / oracle-drift：correct、SR、max_wrong、latch@2/5、false gate、wrong gate。PGIF 才算闭合：TAH committed correct 明显高于 0.146 和 lang 0.263，max_wrong 从 18.7s 降到接近 oracle-drift 量级，latch@5 显著下降，且冷启动错锁分层后不靠删样本取胜。

**MINOR**

1. `asd.py` 顶部 docstring 仍是旧 ASD 锚叙事，和 v3 PGIF 不一致。  
2. `1001_poc-closure.md` 在文档中被引用但当前目录不存在。  
3. “无可学参数”应改为“无训练参数但有调参超参”。  
4. reid 的 `_asd_gate_memory` 实际不重做 gallery update，只覆盖 committed，应单独说明。  
5. D3 需说明“历史驻留”完全不使用 GT，否则会被审稿人质疑自指/泄漏。