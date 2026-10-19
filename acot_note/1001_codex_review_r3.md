verdict: **MAJOR ISSUES**

**R1/R2 MAJOR 复核**

- R1-1 novelty: **partially resolved**。从“冻结锚/CUSUM”转为 PGIF，且有回火、门控消融，方向成立；但仍依赖理想 tracklet id。
- R1-2 命题1过强: **resolved**。已降格为受限确认偏差机制。
- R1-3 命题2过强: **partially resolved**。已改“解释性模型”，但文中仍有“必须回火/缩短τ”等准因果语气。
- R1-4 闭环未闭: **mostly resolved for TAH**。TAH closed-loop 主表已能支撑 PoC；reid 同会话与 Town10HD 仍 pending。
- R1-5 memory-agnostic 落差: **partially resolved**。TAH gate update 接上；reid 是 committed identity 间接门控；tracklet-id 接口不是 memory-agnostic。
- R2-1 parser/闭环跑通: **resolved**。
- R2-2 证明门控不是 smoothing: **resolved for TAH**。nogate/resid/bayes 消融有效。
- R2-3 理论循环论证: **partially resolved**。
- R2-4 tracklet id 外部假设: **unresolved / current main MAJOR**。
- R2-5 闭合标准: **partially resolved**。TAH 满足 correct 与门控消融；max_wrong CI 不显著，reid/Town10 未闭。

**MAJOR**

1. **PGIF 主结果仍使用模拟器持久 actor/tracklet id，方法最关键输入近似 oracle。**  
   这对 CCF-A 方法向仍是 major：PGIF 解决的是“给定稳定 tracklet id 后如何抗确认偏差”，不是完整 deployable re-acquisition。L1 噪声回放有帮助，但只是开环且显示 5%/tick 即增益归零，反而凸显脆弱性。  
   最小修复：标题/贡献改为“under persistent-tracklet interface”；主表脚注写明 `c.idx` 为 simulator ideal association；把 L1 放进主表旁，列 p=0/0.01/0.05 的 acc 退化。若剩余时间够，优先用现有 observe 日志做一次离线 ID-switch/drop 扫描图，不必闭环。

2. **泛化闭环尚未达到“无 MAJOR”。**  
   TAH/Town05 结果强，但 reid 同会话基线、Town10HD 都未出；当前 reid 只是不显著正向，Town10 为 H。  
   最小修复：在文档中把“跨记忆/跨场景贡献”降为 pending，不放入核心贡献；核心 claim 仅写“TAHRelM single-scene PoC closed”。

**MINOR**

- `1001_innovation-design.md` 中“v3 贡献”应改 v4。
- “无训练参数”改“无训练参数，但有冻结超参 λ/W”。
- 补入冷启动分层数字：TAH tempered cold-right 0.211→0.637，cold-wrong 0.365→0.521，说明不靠删样本。
- 命题2措辞用“预测/解释”替代“必须/缩短τ”的强断言。
- 主表里 max_wrong 对 PGIF CI 含 0，应写“均值下降但未显著”。