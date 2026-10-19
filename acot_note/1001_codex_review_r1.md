verdict: **MAJOR ISSUES**

**MAJOR**

1. **Novelty 目前不够 CCF-A 方法向。** ASD 是“冻结首帧/早期模板 + 语言分数 + CUSUM 重检测/重置”的组合，逐项都强烈接近 TLD/long-term tracking、SiamFC 固定模板、经典 CUSUM、MOT track management。修改建议：不要卖“冻结锚/CUSUM 新”，把贡献收窄为“闭环自更新 identity memory 的 drift-correction wrapper”，并必须用实验证明它区别于逐帧重检测、固定模板、低置信 reanchor、DeepSORT track management。

2. **命题1过强。** 对称 toy 模型只能说明特定 EMA+双峰同分布下 confidence 不可辨识；“任何仅为(m,x)函数的统计量 AUROC=0.5”太宽，且真实候选集、运动、历史、语言都破坏对称。修改建议：改成 limited proposition；正文用 AUROC 实测支撑“memory self-confidence 近随机，anchor divergence 更有效”。

3. **命题2不是当前形式的定理。** CUSUM 延迟/ARL 需要 iid或可控依赖、LLR校准、post-change 分布固定；闭环控制会改变观测分布。`W ≤ d_onset+h/D+r`也不是确定上界，只能是期望/高概率界。修改建议：降格为“under assumptions 的期望界”，列清假设，并报告 h-sweep 是否符合线性延迟/误报权衡。

4. **逻辑闭环未闭。** 设计稿主张 ASD 解决 deployable confident drift，但证据仍是 oracle-drift 上界、语言 reanchor 失败、合成单测；真实 GT-free ASD 闭环 pending。修改建议：在得到 `committed+ASD` 结果前，不能声称“证明最长错窗有界/直接约束 latch”。

5. **实现与“memory-agnostic”有落差。** `policy.py` 的 reset hook 只覆盖 TAH/ReID；ASD anchor 依赖早期模型选择，冷启动错则锚错。修改建议：明确接口约束、报告冷启动正确率，并分层统计“anchor correct 条件下/全量”的效果。

**必须补的最小 PoC**

1. TAHRelM committed: off / lang reanchor / ASD-rule / ASD-learned / oracle-drift，同 seeds，报告 correct、max_wrong、latch@2/5、SR、误报次数。  
2. AUROC: memory confidence vs ASD stats/LLR 检测 wrong lock，episode bootstrap CI。  
3. h-sweep: 至少 5 个 h，画检测延迟、误报、max_wrong/latch。  
4. 消融: frozen appearance only、language only、single-threshold、CUSUM、multi-hyp CUSUM。  
5. 泛化: LLR 在 TAH 训练，零样本到 ReID committed；另做密度/场景 split。

**MINOR**

1. 相关工作措辞要更谦虚：ASD 是新组合/新问题化，不是新检测范式。  
2. 推论“纠错信号必须来自记忆更新不触及变量”改为“应优先来自”。  
3. 合成单测只能放 sanity，不可作为核心实证。