第二轮审核结论：

**V1 MAJOR 复核**
1. **Resolved**：已加入 V1-commit-only 与 V1-selfreset，对“PGIF 后验改身份”vs“单纯 re-init/丢 DRM”做机制拆分；代码中 `V1_SELFRESET` 在同一 gate fire 时用 `_pred_slot` 重 init。
2. **Resolved**：已把 H1b 拆成 all 与 drift，并限定 V1 主张为“开局锁对后的漂移纠错”，避免把冷启动错锁归因给 DAM4SAM 漂移。
3. **Partially resolved**：加入 SAM2 predicted-IoU 的 AUROC 报告，能检验置信度是否有身份信息；但若 AUROC 低，仅解释为退化等权驻留，未把等权 PGIF 闭环对照列入主判据。降为 minor 风险。

**V1 verdict：NO MAJOR**。作为 DAM4SAM 上的 PGIF 泛化设计，目前对照与分层足以支撑第二轮继续实验。

**V2 MAJOR 复核**
1. **Resolved**：已明确 V2 是 PGIF 冷启动组件，不再作为独立一区贡献。
2. **Resolved**：飞行混杂已处理。文档要求窗口内按 V0 单帧语言飞行；代码中冷窗口 `_pred_slot = lang_slot`，`COLD_SINGLE` 只控制第 K 步承诺方式。
3. **Resolved**：H2a 改成可检验开局正确率；新增 V2-single@K；窗口内临时承诺仍按逐帧承诺身份计分；先验 `min(G,3.0)` 截断也已实现。

**V2 verdict：NO MAJOR**。作为组件设计合格，剩余 K/λ 敏感性属于 minor/后续实验问题。