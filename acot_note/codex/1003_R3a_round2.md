**V3 verdict: NO MAJOR**

1. **H3c 闭环污染: resolved。**  
   用 V3-ensemble 记录门控前错误相关、条件错误概率和错锁重叠，并把 V3 内相关性降为附录，基本消除了机理检验的自污染问题。

2. **关键对照不足: partially resolved。**  
   新增 ensemble 与 commit-only 正中要害，能检验“普通异构集成”与“融合只承诺不回灌”。gateTAH-only 也能观察 DAM 门控增益与确认偏差。但仍缺 gateDAM-only、固定/低置信 re-init、单源 PGIF 等完整拆解。因此足以支撑收窄后的核心主张，不足以支撑“每个子机制均必要”的强因果声明。

3. **α 启发式: resolved。**  
   主设定改三路等权，且报告 AUROC/缺席率，不在评测种子调参。ECE 缺失可作为 minor，不再构成 MAJOR。

4. **群体确认偏差: resolved。**  
   预注册门控前后分歧率、错误门控、SAM2 重置次数和错误门控占比，且有 gateTAH-only 对照，风险被可观测化。

5. **V2 选择偏差: resolved。**  
   同表报告 V3 与 V3+V2，避免按评测结果继承组件。

考虑 3 个并行闭环槽位、约 10 小时，当前 5 个 V3 臂是合理的最小集合；若已有 V0/DAM4SAM 基线可复用，它足以检验核心链条：V3 是否优于 DAM4SAM/V0，且是否优于普通 ensemble 与 commit-only。建议论文表述保持收窄：主张“融合后验回灌门控带来增益”，不要主张完整门控机制已被完全因果分解。