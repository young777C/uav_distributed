# 1002 · 阶段 0:用 GT 身份监督 LoRA 微调 Qwen3-VL(离线)· 结果与判定

> 2026-10-02 · 结论:**未达到"有效"判据,不进入最终架构**(当时的 `figs/fig_architecture.png` 保持 TAHRelM + PGIF;该图 2026-10-04 更名为 legacy `fig_architecture_legacy_pgif.png`)。
> 模型细节图 `figs/fig_stage0_lora.png`;代码 `train/qwen_lora_grounding.py`(训练/评测)、`train/qwen_lora_paired_eval.py`(配对显著性);产物 `runs/qwen_lora/{r0,r16}/`、`runs/qwen_lora/paired_eval.json`。

## 1. 设计
- 目的:让 Qwen3-VL 第 24 层表征(部署时读出的特征层)具备 WHICH——在 look-alike 中按语言认出目标。
- **r16**:Qwen 解码层 16–23(0-based)的 q/k/v/o 加 LoRA(rank 16,alpha 32,dropout 0.05,262 万参数)+ grounding head(132 万);**r0(对照)**:Qwen 冻结,只训同一 head。r16 − r0 = LoRA 本身的贡献。
- 监督:GT 身份(仅离线训练使用),对画面内全部候选做交叉熵;前向截断到前 24 层,读出 = 第 23 层输出 = `hidden_states[24]`。
- 数据:`mvp_full_v5`,`train/splits/mvp_full_split.json` 中在盘的 episode:train 42 / val 7 / test 10(test 以 Town10HD 为主;val 7 个中 2 个、test 10 个中 1 个目标描述在训练集出现过)。
- 优化:AdamW(LoRA 1e-4,head 1e-3,wd 1e-2),50 步 warmup + cosine,梯度累积 8,裁剪 1.0,3 epoch;按 val 选 best。

## 2. 结果(选车准确率;配对、按 episode 聚类 bootstrap 95% CI)

| 口径 | r0 | r16 | Δ | 95% CI |
|---|---|---|---|---|
| val 总体(7 ep) | 0.558 | 0.650 | +0.092 | [−0.059, +0.239] |
| **test 总体(10 ep)** | 0.558 | 0.601 | **+0.043** | **[−0.061, +0.141]** |
| test look-alike | 0.529 | 0.583 | +0.054 | [−0.066, +0.163] |
| test 未见目标描述 | 0.579 | 0.596 | +0.016 | [−0.087, +0.119] |
| 语言替换后降幅(依赖语言程度) | 0.163 | 0.275 | **+0.112** | **[+0.015, +0.186]** |

- 训练曲线:r0 最佳 epoch 0,之后 val 下降(0.558→0.537);r16 最佳 epoch 1(0.650),epoch 2 回落(0.637);val→test 掉约 5 个点 → 过拟合存在(训练车池窄)。
- 逐 episode(test):3 个大幅提升(0.63→0.90、0.38→0.65、0.38→0.52),2 个明显变差(0.40→0.25、0.66→0.41),异质性大。

## 3. 判定
按事先登记的判据(test 上 r16 − r0 的 CI 不含 0 + 过拟合可控 + 语言依赖增强):
- 选车准确率提升**不显著**(所有口径 CI 含 0)→ **未达到有效**;
- 唯一显著的效应:LoRA **显著增强了对语言的依赖**(替换指令后接近随机)——方向正确,但准确率提升尚不稳定。

## 4. 下一步(若继续)
1. 扩训练车池/场景(数据生成),直接针对过拟合;
2. 扩大评测(纳入 deployable 闭环 rollout 帧),提高检出 ~4–5 点差异的统计功效;
3. 更小 rank / 更少层 / 早停与更强正则。
