**Verdict: MAJOR**

不是因为结果为负，而是因为当前稿件里还有三类会被 T-RO/RA-L/CoRL 审稿人抓住的硬伤：指标口径属于结果后修正的高风险区、预注册对照没有完整闭合、机理因果表述过强。

**MAJOR 1: 正确锁定比例改口径有原则依据，但 HARKing 风险没有完全消除**

代码核对结论：`弃权帧算正确` 对 `1 - mis_follow_inst` 属实。`metrics.py` 里只有 `pred_cand is not None and not target` 才计 instant wrong；DAM4SAM 在 `_dam4sam_select` 中遇到未初始化、空 mask、或 160 px 门内无候选会返回全 `-1e9`，DAM4SAM 分支随后 `pred_slot=-1`。因此 DAM4SAM 的宽松口径 0.786 确实主要是弃权伪影。

把主口径改为 `1 - mis_follow_sustained` 也有原则依据：它和 `max_correct_run_s` / `max_wrong_run_s` 都基于 committed identity，适合衡量“锁定”。但这是在看到 DAM4SAM 弃权现象后修正，不能写成“并非看结果后挑选”。更稳的写法是：

- “原预注册 endpoint 为宽松口径；实验后发现该口径对可弃权 tracker 不可比，因此将承诺口径作为修正主口径。”
- “所有预注册假设同时按原口径报告，修正口径用于最终论文主分析；任何‘预注册成功’只按原口径判定。”
- “三口径全部进主表或主文脚注，不只放 appendix。”
- 补一句 nuance：`mis_follow_sustained` 在已有承诺时弃权沿用上一承诺；若 episode 初期尚无承诺，弃权也不会被计为 sustained wrong，因此它不是“严格弃权沿用上一承诺”的完全等价实现。

当前 `1001` §指标口径和 §5.5 已经比旧版诚实很多，但“主口径改为承诺口径……并非看结果后挑选”建议改掉。

**MAJOR 2: 预注册门控判定没有完全闭合**

V1：预注册修订要求 `V1-selfreset`，但 roadmap 明确未跑。现在只能说“V1 相对 DAM4SAM / commit-only 未达门控”，不能说已完成 V1 机制判定。尤其“重置 SAM2 未显示额外收益”只能针对 `V1 vs V1-commit-only`，不能排除“re-init 本身/丢弃 DRM”的贡献。

V2：`V2-single@K` 未跑，因此 H2a “序贯证据积累提高开局正确率”无法正式检验。roadmap 机理里已有一句，但预注册判定表也应单列：`H2a: pending / not run`。

V3：五臂基本符合修订预注册，且机制对照失败的结论是诚实的。建议把“机制对照 ✗”拆开写：`V3 > ensemble` 失败，`V3 > commit-only` 失败，`V3 vs gateTAH-only` 仅描述，不作为通过门槛。

**MAJOR 3: “回灌门控使记忆误差同步 ρ_err 0.34→0.72”因果表述过强**

数据支持“V3 中两记忆错误高度相关”，也支持“V3-ensemble 中门控前误差相关较低”。但不能直接归因于“双记忆回灌门控”，因为：

- `V3-commit-only` 的 `ρ_err = 0.672`，没有门控记忆也已经很高。
- `V3-gateTAH-only` 的 `ρ_err = 0.723`，与 V3 几乎相同。
- `present_dam` 差异很大：ensemble 0.552，V3 0.768，commit-only 0.382，会影响相关系数的可比性。
- 各臂闭环飞行轨迹、episode 长度、DAM4SAM 弃权模式不同，ρ 不是同一观测分布上的纯机制量。

建议改成：

“无回火逐帧 ensemble 的门控前错误相关较低；一旦进入回火融合闭环，两个记忆的错误显著重合。双记忆回灌不是净增益，并会在开局锁错时固化错误身份。”

如果要主张“回灌导致同步”，需要补分析：按 common seeds、开局对/错分层、DAM present 匹配、每 episode ρ 或 Fisher-z bootstrap、gate 前后窗口对比，最好有固定轨迹/离线 replay 对照。

**MAJOR 4: 泛化与贡献措辞需要收窄**

泛化表述基本谨慎，但“融合/PGIF 家族优于 DAM4SAM”仍偏宽。数据只支持：

- V3 在 Town03 相对 DAM4SAM 承诺口径和最长锁错显著更好。
- V3 在 Town10HD 相对 DAM4SAM 承诺口径不显著，严格口径显著，最长锁错不优。
- 缺 V0 · Town10HD，不能说 V3 或 PGIF 家族泛化优于 V0。

建议写成：“修正口径下，V3 对 DAM4SAM 的跨地图结果方向较好，但证据不一致；尚不能支持 V3 优于 V0 或方法族整体泛化。”

**MINOR**

- “单帧语言准确率约 0.22–0.33”和“开局锁对 7–11/19 / 首帧约 50%”混在一起了。前者像 tick-wise `acc_lang`，后者是 opening accuracy。必须分开。
- `1001` §5.1 PGIF 正确比例 0.613，而 §5.5 / `deploy_compare.json` 是 0.576；需要解释是旧 tick-align 口径还是不同实验批次。
- V0 开局分层来自旧 `runs/asd/deploy_all`，而 1003 其他臂来自 `runs/asd/iter`；建议标注不是同批日志，避免 paired opening 分析被质疑。
- “PGIF 跨记忆可包装的弱证据”建议删或降级。V1 未过非劣、selfreset 未跑，作为负结果即可。
- “新增贡献足够 SCI 一区/CCF-A”目前不够。可写进论文的是有价值的负结果和评测修正，但若作为主线，还缺一个正向方法贡献或更大规模 benchmark 证据：更多 seeds、完整预注册对照、真实 detector/localizer 噪声、V0 Town10HD、DAM4SAM tick 日志、固定轨迹机理验证。

总体建议：这批结果很诚实，也有论文价值；但应把它定位为“指标修正 + 负结果机理诊断 + 下一步假设生成”，不要包装成 V1/V2/V3 的方法成功。