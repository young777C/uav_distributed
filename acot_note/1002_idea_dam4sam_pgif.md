# Socket-Bridge Reset Hook Lets PGIF Wrap DAM4SAM's SAM2-L Memory

## Original Idea

DAM4SAM+PGIF 包裹:给 DAM4SAM 加 reset 钩子(用 PGIF 的后验门控选择 ŝ 的框重新初始化 DAM4SAM 的 SAM2-L 记忆),PGIF 包裹在 DAM4SAM 外层。目标是同时吃到 DAM4SAM 的高正确锁定比例(0.786)与 PGIF 的短最长锁错(避免 DAM4SAM 当前 26.7s 均值/21.6s 中位的粘滞错锁),在 deployable 闭环设定下(CARLA 在线,Town05,同种子 91000-19,n=19,三项客观指标:正确锁定比例、最长连续正确锁定、最长锁错)验证。背景见 acot_note/1001_innovation-design.md §5.5 与 §7(DAM4SAM 对比)、acot_note/1001_poc-closure.md。PGIF 实现在 carla_uav_tracking/rollout/asd.py::ASD._filter_step 与 policy.py::_asd_gate_memory;DAM4SAM 接入在 policy.py 的 _dam4sam_select。

## Primary Direction: Socket-Bridge Reset Hook

### Rationale

这是 idea 的字面读法:复用 `_asd_gate_memory`/`_asd_reset_memory` 已有的"按 tid_head 分支"模式(tah/reid 已实现),给 `dam4sam` 补上同样的分支,通过已存在的 socket bridge 把 PGIF 选中的框重新灌回 DAM4SAM 的 SAM2-L 记忆——而不只是像现在这样,只在当前 tick 覆盖 `_pred_slot` 的"表面"选择。

### Approach Summary

在 `carla_uav_tracking/rollout/policy.py` 的 `_asd_gate_memory`/`_asd_reset_memory`(约第 318–337 行)新增 `tid_head == "dam4sam"` 分支:ASD 触发 gate/reset(`policy.py:707–729`)时,取 PGIF 选中槽位 `cset.cands[slot]`,按 `_dam4sam_select` 中已有的构框方式 `(u−w/2, v−h/2, w, h)`,通过现有的 `self._dam4sam_sock` 发送 `{"cmd": "init", "rgb": ..., "box": ...}`(`bridge.py` 的 `send_msg`/`recv_msg`)。服务端 `dam4sam_service.py` 的 `cmd == "init"` 处理分支已经调用 `tracker.initialize(...)`,等价于对 SAM2-L 记忆做一次真正的重新播种——**不需要新增协议动词**。唯一的非平凡改动是把 `_asd_gate_memory`/`_asd_reset_memory` 的签名从仅 `(slot)` 扩展为能拿到 `rgb`/`cset`(当前调用点在 `policy.py:719,725`),以及控制触发节奏(只在真正 fire 时重发 init,避免每 tick 都重置拖慢实时控制环)。

### Objective Evidence

- `carla_uav_tracking/rollout/policy.py:318-337` — `_asd_gate_memory`/`_asd_reset_memory` 已对 `tid_head == "tah"`、`"reid"` 实现同模式钩子;目前没有 `"dam4sam"` 分支(已用 grep 确认)。
- `carla_uav_tracking/rollout/policy.py:707-729` — ASD 触发 fire 时调用这两个钩子;对 `dam4sam` 今天两个分支都不命中,只覆盖该 tick 的 `_pred_slot`/`_committed_idx`,印证 idea 指出的确切缺口。
- `carla_uav_tracking/rollout/policy.py:386-413`(`_dam4sam_select`)— 已有从候选框构造 init box、经 `self._dam4sam_sock` 发送 `{"cmd": "init", "rgb": ..., "box": ...}` 并读取 `recv_msg(...).get("bbox")` 的先例,新 reset 钩子直接复用这套调用模式。
- `carla_uav_tracking/rollout/bridge.py` — 长度前缀 pickle 协议,payload 无关;不需要改协议,"init" 命令本身已经是一次完整的重置+重初始化。
- `carla_uav_tracking/rollout/dam4sam_service.py:62-67` — 服务端 `cmd == "init"` 调用 `tracker.initialize(img, None, bbox=...)`,证实中途重发 `init` 确实会重置 DAM4SAM 内部 SAM2-L 状态,不只是覆盖一次预测。服务运行在独立的 `dam4sam-svc` 容器/环境,只能经 socket bridge 访问——与"socket-bridge reset hook"的框架完全吻合。
- `carla_uav_tracking/rollout/eval_ext_dam4sam.sh` — 已有的 DAM4SAM 部署评测脚本(`--tid-head dam4sam --dam4sam-host dam4sam-svc --dam4sam-port 5601`,Town05,seeds 91000–19,n=19/20),与 idea 要求的评测协议完全一致。
- `acot_note/1001_innovation-design.md:195,228` — 原文已把这个实验列为下一步:"DAM4SAM 的长时间锁错正是命题 1 的自更新确认偏差;为其增加'用 ŝ 的框重新初始化'的重置钩子后,PGIF 可直接包在 DAM4SAM 外层"(§5.5)、"外部记忆(DAM4SAM)加 reset 钩子后接 PGIF,扩展跨记忆证据"(§9)——这不是凭空假设,是仓库笔记里已设计好的下一个实验。
- `acot_note/1001_poc-closure.md` — G1/G3 行记录了 PGIF 跨记忆(reid 启发式 gallery)泛化的既有证据,证明这套 reset-hook 机制此前已在另一种记忆类型上验证过可以工作。
- `carla_uav_tracking/rollout/asd.py` 约第 100–193 行 — `ASD.step` 返回 `{"fire","slot","kind":"gate"|...}`,gate 与 reset 式触发已经区分清楚,挂接点不含糊。

### Known Risks

- 需要把 `_asd_gate_memory(self, slot)` / `_asd_reset_memory(self, slot)` 的签名扩展为能拿到 `rgb`/`cset`(或提前存成实例状态),这是一个小但真实的签名改动,牵涉 `policy.py:719` 与 `725` 两处调用点。
- 每次 reset-fire 在同步的控制 tick 内整帧 RGB 往返 socket,如果 ASD 触发频繁,可能影响 deployable 闭环的实时 S2 节奏。
- 语义风险:服务端 `cmd:init` 原本只用于冷启动/episode 开始;中途重复 init 是否会和 DAM4SAM 内部的 Distractor-Resolving Memory(DRM)产生未验证的交互,目前没有把"软重播种"和"硬重初始化"区分开的独立命令——这个假设需要经验验证,不能只靠读代码确认。
- 任何对 `dam4sam_service.py` 或协议的改动都需要重启/重部署独立的 `dam4sam-svc` 服务(另一个 torch2.1/SAM2 环境),比纯 `policy.py` 改动多一层协调开销——不过本方向的设计已经通过复用现成的 `"init"` 命令,避免了协议改动本身。
- `_asd.step` 的 gate 路径(`policy.py:713`,`kind=="gate"`)目前只在 `gate_memory=True` 且 `self.gallery_seed == "committed"` 时调用 `_asd_gate_memory`;需要决定 DAM4SAM 是否同时要 gate 和 reset 两种钩子,还是只要 reset——这会影响实验设计对 idea 原话"PGIF 包裹在 DAM4SAM 外层"的忠实度。

## Alternative Directions Considered

### Alt-1: DAM4SAM-Native Confidence Signal
- Gist: DAM4SAM 的 SAM2 掩码解码器内部已经算出一个每 tick 的 IoU 预测置信度(`m_iou`,`external/DAM4SAM/dam4sam_tracker.py:212` 的 `track()` 中已提取,但最终 `out_dict = {'pred_mask': m}` 把它丢弃了)。把这个信号经 bridge 协议透传到 `policy.py`,替换掉目前 `tid_head=="dam4sam"` 时通用的 top1−top2 margin 代理置信度,让 PGIF 的秩置信加权(`asd.py::_filter_step`)吃到 DAM4SAM 自己的置信度,而不是候选框距离的间接代理。
- Objective Evidence:
  - `external/DAM4SAM/dam4sam_tracker.py:199-267` 的 `track()`:`m_iou` 已内部计算并用于 DRM 更新门控(`m_iou > 0.8`),但从未透传给调用方。
  - `carla_uav_tracking/rollout/policy.py:674-675`:`_pred_conf` 当前对所有 tid_head 用同一套 top1−top2 margin 逻辑,dam4sam 没有特殊处理。
  - `carla_uav_tracking/rollout/policy.py:587-602`(`tid_head=="tah"` 分支):已有"某个 head 直接设置 `self._pred_conf` = 模型原生置信度"的先例,可复用同样模式。
  - `acot_note/1001_innovation-design.md` D1 表(line 38)的自置信 AUROC 发现只覆盖 TAH/reid,没有 dam4sam 列——这个信号此前从未被验证过。
- Why not primary: confidence medium,且存在一个未解的根本性风险——`m_iou` 衡量的是"掩码质量",不是"身份正确性"。DAM4SAM 已知的粘滞错锁恰恰是"置信度高但身份错"的情形,`m_iou` 在那种情况下很可能仍然很高,作为 PGIF 需要的 `c_t` 价值存疑,需要先补一个类比 D1 的 AUROC 检验才能判断是否值得做,而不能直接拿来当主方向。

### Alt-2: PGIF as External Arbiter / No Internal Reset
- Gist: 不碰 DAM4SAM 内部记忆,让它按自己的节奏持续运行;同时并行跑第二个身份记忆(TAHRelM 或 reid),把两者的每 tick 候选选择都当作独立证据流喂给 PGIF 的信念累积(`asd.py::_filter_step` 的 belief dict 按持久 actor id 索引,天然可以接受多个来源的证据),由 PGIF 每 tick 仲裁"信哪个记忆"并提交控制承诺,而不是试图重写 DAM4SAM 的内部状态——完全不改动 DAM4SAM 的 socket/协议。
- Objective Evidence:
  - `carla_uav_tracking/rollout/asd.py:151-169`:belief 字典按 actor id 索引而非按记忆来源索引,理论上支持多来源证据的加法更新。
  - `carla_uav_tracking/rollout/policy.py:160-194`、`policy_server.py:108`:`__init__` 与 CLI `--tid-head` 目前都是严格互斥的单一字符串,双记忆并行是全新管线,不是现成开关。
  - `carla_uav_tracking/rollout/deploy_compare.py` 的 `ARMS` 列表目前是单机制臂的固定花名册,没有双记忆/仲裁臂的先例。
- Why not primary: confidence medium,且有一个结构性缺陷——这个方向完全不解决 DAM4SAM 自身的粘滞漂移(只能切换信任哪个记忆,无法缩短已经承诺给 DAM4SAM 的那段漂移),而且未被选中的那个记忆仍会按自己的输出自更新,重新引入 PGIF 本要切断的"自更新确认偏差"回路——与方向 1(直接重置 DAM4SAM 内部状态)相比,工程风险虽更低,但没有真正攻克目标症状(DAM4SAM 的 26.7s/21.6s 长锁错)。

### Alt-3: Evaluation-Harness Extension
- Gist: 把"跑通 dam4sam+pgif 这条臂"当作纯粹的 harness/插线问题:合并现有的 `eval_ext_dam4sam.sh`(DAM4SAM 专属 flag)与 `eval_asd.sh`(PGIF/ASD 通用参数化启动器),单个 `policy_server.py` 进程同时带 `--tid-head dam4sam ...` 与 `--asd filter --asd-cfg ...`;scorer(`metrics.py`)已经对 tid_head 无感知,不需要改;只需在 `deploy_compare.py` 的 `ARMS` 列表加一行,新臂就能自动进入已有的配对 bootstrap 统计与 `fig_deploy_compare.py` 出图逻辑。
- Objective Evidence:
  - `carla_uav_tracking/rollout/policy_server.py` argparse 已经允许 `--tid-head dam4sam` 与 `--asd filter` 同时存在,没有互斥限制。
  - `carla_uav_tracking/rollout/policy.py:164-165`(`_need_dino`)已经对任意 tid_head+asd 组合无条件生效。
  - `carla_uav_tracking/rollout/metrics.py` 全文无 tid_head/asd 分支,三项客观指标管线已经是 arm-agnostic 的。
  - 发现一个命名口径差异:`eval_ext_dam4sam.sh` 写 `paper_ext_dam4sam.json`,但 `deploy_compare.py::load()` 读 `paper_dep_{stem}.json`——新臂必须对齐后者的命名约定,否则会产出一个没人读的孤立文件。
  - `runs/asd/deploy_all/chains.sh`(§5.5 提到的批量编排脚本)在当前工作树中未找到,无法直接核实"插入第 11 条臂"的具体机制,只能作为空白如实报告。
- Why not primary: confidence 同样 high,但这个方向本质是方向 1(或任何其他方向)跑起来之后"必须顺带做"的配套工作,不是一个独立可选的算法创新路径——它不回答"怎么让 PGIF 真正控制 DAM4SAM 的记忆",只回答"跑通之后怎么评分"。选方向 1 为主路径时,方向 3 的插线清单(尤其是 `paper_ext_*` vs `paper_dep_*` 命名坑、`_dam4sam_inited` 在 episode 边界 vs ASD 触发的 reset 之间如何区分)应作为方向 1 实现计划里的必要步骤,而非替代它的另一个方向。

## Synthesis Notes

四个方向并非互斥:主方向(方向 1)解决"怎么让 PGIF 真正触碰 DAM4SAM 内部状态",方向 3(evaluation-harness)是方向 1 落地评测时必须完成的配套插线工作,二者应在同一个实现计划里顺序执行(先接线,再跑通评测)。方向 2(原生置信信号)的核心发现——`m_iou` 是掩码质量而非身份正确性——即使不作为主路径,也应该作为方向 1 实现后的一个**诊断动作**保留:在方向 1 跑通后,补一次"`m_iou` 在错锁 tick 上的 AUROC"检验(类比 `1001_innovation-design.md` 的 D1),如果发现 `m_iou` 确实对身份正确性有信息量,可以在后续版本里把它接入 PGIF 的置信加权,作为方向 1 的自然延伸而非另起炉灶。方向 2(外部仲裁器)揭示的"未被选中记忆仍会自更新、重新引入确认偏差"这个机理教训,应该写进方向 1 的实现计划的局限/风险登记里,作为选择"直接重置 DAM4SAM 内部状态"而非"双记忆并行仲裁"的论证依据之一。
