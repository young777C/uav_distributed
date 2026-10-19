# 1002 · 迭代升级路线 workflow 设计(未执行,待确认后启动)

> 2026-10-02 · 状态:**设计稿,PoC/deployable/泛化验证尚未运行**
> 触发问题:PGIF(`1001_innovation-design.md` v4,Codex R4 "NO MAJOR ISSUES")在正确锁定比例上仍弱于外部基线 DAM4SAM,最长锁错时间也不占优——创新点需要针对这两点继续迭代,12h 验证窗口内完成,每个版本须经 Codex review 判定创新点是否足够。
> 本文档只回答"怎么跑这个迭代",不产出新实验数据;候选方向与判据均来自已闭环文档的既有证据,未新增假设。

---

## 1. 现状诊断(复述已有证据,不重新实验)

来源:`1001_innovation-design.md` §5.5 / §5.1 / §8,`1001_poc-closure.md` §1。

| 症状 | 数据 | 根因(文档已诊断) |
|---|---|---|
| 正确锁定比例弱于 DAM4SAM | PGIF 0.583 [0.37,0.79] vs DAM4SAM 0.786 [0.65,0.90];PGIF−DAM4SAM = **−0.20 [−0.39,−0.03]**(显著) | DAM4SAM(SAM2-L 记忆)本身初始/粘滞能力强;PGIF 的增益上限由**冷启动语言初锁正确率**决定(L2) |
| 最长锁错偏高 | PGIF 均值 15.4s(中位 0.9s,长尾由冷启动锁错拉高) | 初锁错误时没有正确历史可回退,PGIF 会**坚持**错误身份(§5.1 分层:初锁错子集 Δ=−0.065~−0.156,方向为负) |
| DAM4SAM 自身弱点(可利用) | DAM4SAM 最长锁错均值 **26.7s**、中位 21.6s,全部方法中最长——典型"粘滞记忆":错了就一直错 | 与命题 1(自更新确认偏差)机理一致,§5.5 已建议"DAM4SAM 加 reset 钩子 + PGIF 包裹"作为下一实验 |

**结论**:两个弱点同源(冷启动),且文档已经给出一个具体、未验证的强候选实验(DAM4SAM+PGIF),不需要凭空想新方向。

---

## 2. workflow:复用仓库已有的 RLCR 工具链

```
Phase 0  /humanize:gen-idea      idea draft(N 个正交方向 + 仓库客观证据,只读探索,不改代码)
            │
Phase 1  /humanize:gen-plan      选定方向 → 实现 + 实验计划,任务打 coding/analyze 标签
            │
Phase 2  /humanize:refine-plan   QA ledger,剔除执行歧义,确认 12h 预算可行
            │
Phase 3  /humanize:start-rlcr-loop
            │   ├─ 每"轮"(round)= 一个创新点版本做完一次完整实现+实验
            │   ├─ 轮末写 round summary → Codex 审查(COMPLETE / 继续修)
            │   ├─ 每 N 轮一次 full alignment audit(--full-review-round,默认 5,可设更小)
            │   └─ 进入 Review Phase 后 `codex review` 做代码审查([P0-9])
            │
Phase 4  (人工收尾)                汇总版本演化表 + 写最终迭代升级路线 md
```

- Codex review 的"创新点是否足够"判据:复用 `1001_poc-closure.md` §4 已验证过的五条闭合判据模式(显著优于基线、门控/组件消融证明非平凡、冷启动分层不靠删样本、跨设定不劣化),外加本轮新增第六条:**正确锁定比例或最长锁错至少一项相对现有 PGIF 主表显著改善,且不以另一项显著变差为代价**。
- 每版本的 Codex 审稿记录沿用 `1001_codex_review_r{N}.md` 命名,版本演化表沿用 `1001_innovation-design.md` 附录 A 格式(版本 / 假设 / 被什么数据推翻 / 处理)。

---

## 3. Phase 0 候选方向(供 gen-idea 探索,非最终选择)

全部方向均有仓库内证据或文档已写明的下一步作为起点,不是凭空假设:

| # | 方向 | 直接证据来源 | 预期攻克的指标 |
|---|---|---|---|
| 1 | **DAM4SAM + PGIF 包裹**:给 DAM4SAM 加"用 ŝ 的框重新初始化"钩子,PGIF 包在外层 | `1001_innovation-design.md` §5.5 末尾原话建议;命题 1 机理(DAM4SAM 26.7s 锁错 = 同一种确认偏差) | 最长锁错(直接攻 DAM4SAM 的粘滞弱点);若正确锁定比例不掉,等于同时吃到两边优势 |
| 2 | **冷启动延迟承诺**:语言证据时序积分后再锁定,而非首帧即承诺 | §5.1 冷启动分层(初锁对 0.96/0.90/0.80 vs 初锁错 −0.065~−0.156);§9 下一步第2条;§8 L2 | 正确锁定比例(消除初锁错误的负向拖累) 与 最长锁错 |
| 3 | **重捕获窗口放大**:目标出画重新入画时对证据做短期快速放大 | 附录 B 负面发现:PGIF 在 Town10HD/reid 上重捕获成功率下降(0.31→0.17、0.55→0.15);§5.5 读法段落已提出此改进方向 | 次要:修复一个已知回归,不直接攻 DAM4SAM gap |
| 4 | **真实 MOT 关联替换理想 tracklet**:接入 OC-SORT/DeepSORT 关联层 | L1;§9 下一步第1条 | 写作完整性(审稿人必问项),不直接攻当前两个症状,优先级低于 1/2 |

**建议主路径**:方向 1(直接对标"弱于 DAM4SAM")为 Round 1 唯一目标;方向 2 作为 Round 2(若方向 1 提前验证完成或部分失败,视 12h 剩余时间并行/顺延);方向 3、4 不在本轮 12h 预算内,记入"下一步"而非本轮版本。

---

## 4. 12h 预算分配(草案,待 gen-plan 落实为可执行步骤)

| 时段 | 内容 | 产出 |
|---|---|---|
| 0.0–1.0h | gen-idea(方向 1 为主)+ gen-plan + refine-plan | 实现计划、任务标签、预先登记的闭合判据 |
| 1.0–2.5h | 实现:DAM4SAM reset 钩子 + PGIF 外层包裹接线(`policy.py` / `asd.py` 复用已有接口) | 代码改动,离线 smoke 测试(非 CARLA) |
| 2.5–7.0h | CARLA 在线闭环:Town05 同种子(91000–19,n=19),deployable 设定 | `eval_out/paper_asd_dam4sam_pgif.json` 等 |
| 7.0–8.0h | 统计检验(配对 bootstrap)、三项客观指标 + 消融(是否门控、是否回火) | round summary,提交 Codex review |
| 8.0–9.0h | Codex 审查轮次(若 MAJOR,修复后二次提交) | `1001_codex_review_r5.md`(或更高) |
| 9.0–11.5h | 若时间允许且方向 1 已闭合:方向 2(冷启动延迟承诺)最小验证(缩小种子数,n≈10 先验信号) | 版本 v5 草稿 + 初步显著性 |
| 11.5–12.0h | 收尾:版本演化表、最终迭代升级路线 md 定稿 | 本文档的 Phase 4 产出 |

**风险**:CARLA 在线闭环单臂约需 19 episode × 实时仿真,历史记录显示存在 GPU 实例渲染异常需要迁移(`1001_poc-closure.md` §5)。若方向 1 在 7h 内未跑完,放弃方向 2,把 7–12h 全部给方向 1 的充分统计功效与 Codex 多轮修复。

---

## 5. Phase 4 产出模板(最终迭代升级路线 md 的骨架)

实际执行后,`1002_iteration-roadmap-final.md`(或直接扩展本文件)应包含:

1. **版本演化表**(同 `1001_innovation-design.md` 附录 A 格式):
   | 版本 | 假设/方法 | 被什么数据证实或推翻 | 处理 |
2. **每版本 Codex 审稿轨迹**:R5(MAJOR/NO MAJOR)→ ... → 最终裁定
3. **贡献范围更新**(是否需要再次收窄,参照 §6 的收窄模式)
4. **与 DAM4SAM 对比的最终读法**(是否反转 −0.20 的 gap,或改写为"不可同时占优"的取舍型 claim)
5. **局限更新**(§8 L1–L4 是否需要新增 L5)

---

## 6. 启动方式(需用户确认后执行,未自动触发)

```bash
/humanize:gen-idea "DAM4SAM+PGIF 包裹:给 DAM4SAM 加 reset 钩子,PGIF 包在外层,目标是同时吃到 DAM4SAM 的高正确锁定比例与 PGIF 的短最长锁错" --n 4 --output acot_note/1002_idea_dam4sam_pgif.md
# 人工审阅 draft → 
/humanize:gen-plan --input acot_note/1002_idea_dam4sam_pgif.md --output acot_note/1002_plan_dam4sam_pgif.md
/humanize:refine-plan acot_note/1002_plan_dam4sam_pgif.md
/humanize:start-rlcr-loop acot_note/1002_plan_dam4sam_pgif.md --max 6 --full-review-round 2
```

本文档本身不触发任何上述命令。
