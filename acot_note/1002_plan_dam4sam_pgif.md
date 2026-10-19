# DAM4SAM Reset Hook for PGIF (Gate-Path, Rate-Limited)

## Goal Description

TAHRelM 与 PGIF 算法本身均保持冻结、不修改。本计划给 `carla_uav_tracking/rollout/policy.py` 的 PGIF 门控路径(`_asd_gate_memory`,对应 `ASD._filter_step` 的 `filter_kind="tempered"` 主路径,其触发 `kind` 恒为 `"gate"`,不会走 `_asd_reset_memory`)新增一个 `tid_head=="dam4sam"` 分支:当 PGIF 的门控判定当前应该相信的身份槽位与 DAM4SAM 自己的原始选择不同时,通过既有的 socket bridge(`bridge.py`)向 DAM4SAM 服务(`dam4sam_service.py`)重发已存在的 `{"cmd":"init",...}` 消息,强制 DAM4SAM 的 SAM2-L 内部记忆重新播种到 PGIF 选中的框——而不只是像现在这样仅在当前 tick 覆盖 `_pred_slot` 的表面选择。该钩子受 cooldown 速率限制与 socket 超时保护,失败时退化为保留 DAM4SAM 的原始选择,不强行覆盖控制承诺。

目标是在 deployable 闭环设定(CARLA 在线,Town05,seeds 91000–91019,三项客观指标:正确锁定比例↑、最长连续正确锁定↑、最长锁错↓)下,验证 PGIF 包裹后的 DAM4SAM 能否同时保留其高正确锁定比例(deployable 基线 0.786)、又显著缩短其粘滞错锁(基线最长锁错均值 26.7s/中位 21.6s),从而把"PGIF 正确锁定比例弱于 DAM4SAM"这一论文弱点,转化为"PGIF 可跨记忆泛化、能补齐 DAM4SAM 的粘滞错锁短板"的额外贡献点。TAHRelM+PGIF 的既有主结果(`1001_innovation-design.md` §5.1/§5.5)不受影响,DAM4SAM+PGIF 作为新增的第二条验证臂并入 §5.5 的部署对比表。

本计划经过两轮 Codex 收敛审查(见 `## Claude-Codex Deliberation`)与用户决策(见 `## Pending User Decisions`,全部已决议,无 PENDING 项)。

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: `_asd_gate_memory` 对 `tid_head=="dam4sam"` 新增结构化 reset 钩子,只挂在 PGIF 的 gate 路径(`kind=="gate"`),不扩展 `_asd_reset_memory`(legacy 路径维持 tah/reid 现状不变)。
  - Positive Tests (expected to PASS):
    - `tid_head="dam4sam"` 且 `asd_cfg.gate_memory=true` 时,一次 PGIF gate fire(`pgif_slot != raw_dam4sam_slot`)触发对 `self._dam4sam_sock` 的一次 `{"cmd":"init",...}` 发送,返回 `{"reset_sent":True,"reset_ok":True,"reason":"ok",...}`。
    - `tid_head="tah"` 或 `"reid"` 时,行为与改动前逐字节一致(回归测试:同种子同配置下 `eval_out/paper_asd_tah_tempered.json` 数值不变)。
  - Negative Tests (expected to FAIL / be rejected):
    - `tid_head="dam4sam"` 且 `asd_cfg.gate_memory=false`(commit-only 配置)时,即便 `pgif_slot != raw_dam4sam_slot`,也不得向 `self._dam4sam_sock` 发送任何消息。
    - 代码审查断言:`_asd_reset_memory` 中不存在任何 `tid_head=="dam4sam"` 的新分支(legacy reset 路径维持现状)。
  - AC-1.1: cooldown 速率限制(默认 K=10 个 S2 tick,先验写死,不依据本轮结果调参)。
    - Positive: 连续两次 gate fire 间隔 < K 个 S2 tick 时,第二次返回 `reason="cooldown_skip"`、`reset_sent=False`,且**不计入** reset 失败统计。
    - Negative: 间隔 ≥ K 个 tick 的 gate fire 不应被 cooldown 拦截(`reason != "cooldown_skip"`)。
  - AC-1.2: socket 超时与异常处理(默认 timeout=2s)。
    - Positive: 模拟 `socket.timeout`/`ConnectionError`/`EOFError` 时,钩子返回 `reset_sent=True(or False), reset_ok=False, reason∈{"timeout","conn_error"}`,控制环不卡死、不抛未捕获异常。
    - Negative: 正常响应(`bbox` 非空、合法)不得被误判为超时/失败。
  - AC-1.3: reset 失败时的回退语义。
    - Positive: `reset_ok=False` 的 tick,控制提交(`_committed_idx`/`_pred_slot`)保留 DAM4SAM 的 raw pick,不强行切到 PGIF slot;该 tick 的 fallback 原因写入事件日志。
    - Negative: `reset_ok=True` 的 tick,控制提交**仍然**切到 PGIF slot(维持当前 gate 的既有语义,不因新增失败处理而回退该行为)。
- AC-2: DAM4SAM reset box 的构造与边界保护。
  - Positive Tests: 候选槽位的框 `(u−w/2, v−h/2, w, h)` 经过 clamp 后,`0 ≤ x, 0 ≤ y, x+w ≤ 图像宽, y+h ≤ 图像高, w>0, h>0` 恒成立;对紧贴图像边缘的候选框,clamp 后仍发送合法 `init` 请求。
  - Negative Tests: 对越界框(如 `u` 为负、`w` 为 0)直接发送未 clamp 的框是不允许的;单测应断言未 clamp 的输入不会被直接送入 `send_msg`。
- AC-3: 每次 PGIF gate fire(不论 `tid_head` 是否为 dam4sam)写入结构化事件日志。
  - Positive Tests: 日志记录包含 `ep/step/raw_slot/raw_actor_id/pgif_slot/pgif_actor_id/reset_sent/reset_ok/reason/latency_ms/pgif_conf/dam4sam_m_iou` 字段;`dam4sam_m_iou` 本轮允许为空值(占位,见 Pending User Decisions 中关于协议改动的决议)。
  - Negative Tests: 非 dam4sam 的 gate fire(如 tah/reid)不应包含 `reset_sent`/`reset_ok`/`dam4sam_m_iou` 等 dam4sam 专属字段被置为非 null 的错误值(应为 `None`/缺省)。
- AC-4: ablation 矩阵可通过现有 `asd_cfg` JSON 机制跑通,不新增独立的 "hook mode" 控制面。
  - Positive Tests: 以下三档配置均可独立运行且互不干扰:
    1. `dam4sam` baseline:不传 `--asd`(或 `--asd off`)。
    2. `dam4sam_pgif_commit`:`--asd filter --asd-cfg '{"filter_kind":"tempered","resid_decay":0.98,"gate_memory":false}'`。
    3. `dam4sam_pgif`:`--asd filter --asd-cfg '{"filter_kind":"tempered","resid_decay":0.98,"gate_memory":true,"dam4sam_reset_cooldown_ticks":10,"dam4sam_socket_timeout_s":2.0}'`。
  - Negative Tests: 不存在额外的 `--dam4sam-hook-mode` 之类新 CLI 枚举;检查 `policy_server.py` argparse 差异,确认未新增该类参数。
- AC-5: 评测 harness 插线(`runs/asd/deploy_all/chains.sh`、`carla_uav_tracking/rollout/deploy_compare.py`)。
  - Positive Tests: `chains.sh` 在 `start_dam4sam` 之后新增 `dam4sam_pgif_commit` 与 `dam4sam_pgif` 两条 `run` 调用,产出 `eval_out/paper_dep_dam4sam_pgif_commit.json` 与 `eval_out/paper_dep_dam4sam_pgif.json`(命名严格遵循 `paper_dep_*` 约定,不使用 `paper_ext_*`)。
  - Negative Tests: `deploy_compare.py` 的输出中不得把 `paper_ext_dam4sam.json`(旧 gt-seeded/旧 scorer 口径)当作 deployable 主表数据源。
- AC-6: 显式配对比较矩阵与统计。
  - Positive Tests: `deploy_compare.py` 输出至少三组配对差值(配对 bootstrap 95% CI,只在各自共同 seeds 上配对):`dam4sam_pgif vs dam4sam`、`dam4sam_pgif vs dam4sam_pgif_commit`、`dam4sam_pgif vs tahrelm_pgif`;每条臂同时报告有效计分 episode 数。
  - Negative Tests: 不得把不同有效 episode 集合的臂直接按 index 配对(必须按 seed 取交集后配对)。
  - AC-6.1(硬性闭合判据,用户已确认为硬指标,不达标则本轮判定为未闭环):`dam4sam_pgif vs dam4sam` 的 `max_wrong_run_s` 配对差值 95% CI 不含 0 且为负(显著下降);同时 `correct_lock_ratio` 配对差值 95% CI 不得显著为负(即不显著下降,非劣或提升均可)。
    - Positive: 两个条件同时成立 → 本轮方向 1 实验判定为"闭合"。
    - Negative: 任一条件不成立 → 判定为"未闭环",如实记录为负结果(不得淡化或省略),比照 PGIF 历史版本演化(`1001_innovation-design.md` 附录 A)的诚实报告惯例。
- AC-7(诊断性,不阻塞 AC-6.1 的闭合判定):对已记录的 `pgif_conf`(当前仍是候选框距离 margin 代理,非 DAM4SAM 原生 `m_iou`)在错锁 tick 上做一次事后 AUROC 分析,作为是否值得在后续版本中接入 DAM4SAM 原生置信信号(Alt-1 方向)的决策依据。
  - Positive Tests: 产出一个 AUROC 数值及其 bootstrap CI,写入 round summary,不修改本轮任何实验的门控逻辑。
  - Negative Tests: 该分析不得反向影响 AC-1~AC-6 已完成实验的任何已记录结果(纯事后只读分析)。

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
实现包含 AC-1 至 AC-7 全部内容:结构化 reset 钩子(含 cooldown/超时/回退语义)、bbox clamp、完整事件日志、三档 ablation(baseline / commit-only / reset-hook)均在 Town05 seeds 91000–91019 上跑满 20 episodes(报告有效计分数,预期 n≈19-20)、`chains.sh`/`deploy_compare.py` 插线完成、显式三组配对比较产出、AC-7 诊断性 AUROC 分析完成,并把结果按 `1001_poc-closure.md` 的 claim→证据矩阵格式整理进 round summary。

### Lower Bound (Minimum Acceptable Scope)
实现至少包含:AC-1(含 AC-1.1/1.2/1.3)、AC-2、AC-3 的核心钩子与日志;AC-4 的 baseline 与 reset-hook 两档(commit-only 档可延后,但必须在 round summary 中明确标注为"本轮未跑,下一步");AC-5/AC-6 的插线与 `dam4sam_pgif vs dam4sam` 这一组配对比较必须产出,并给出 AC-6.1 的闭合/未闭环判定。AC-7 诊断分析可延后到下一轮,但若延后须在 round summary 中显式登记为待办,不得静默省略。

### Allowed Choices
- Can use:现有 `--asd filter` + `asd_cfg` JSON 机制(新增 `dam4sam_reset_cooldown_ticks`/`dam4sam_socket_timeout_s` 字段);现有 socket bridge 的 `{"cmd":"init",...}` 消息;`policy.py` 内部的状态扩展(如把 `rgb`/`cset` 存为实例属性,或扩展 `_asd_gate_memory` 签名)。
- Cannot use:**不得修改 `dam4sam_service.py` 的 wire 协议**(不新增 cmd 动词,不改消息字段),**不得重启/重新部署 `dam4sam-svc` 容器**(用户决议,DEC-2);**不得扩展 `_asd_reset_memory` 以支持 `tid_head=="dam4sam"`**(用户决议,DEC-3,legacy reset 路径维持现状);**不得新增独立的 "hook mode" CLI 控制面**(Codex round-2 DISAGREE 已采纳,复用现有 `asd_cfg` 机制);cooldown K 与 socket timeout 的默认值在实现时一次性写死为 K=10 tick、timeout=2s(用户决议,DEC-4),**不得在观测到本轮评测结果后回头调整这两个数值**(避免在评测种子上调参)。

> 本计划的核心技术路径(socket-bridge 复用 `init` 命令做硬重置、只挂 gate 不扩展 legacy reset、不改协议)已通过用户决议锁定为确定性设计,Upper/Lower Bound 的差异仅在于 ablation 矩阵的完整度与诊断分析是否本轮完成,不在于是否采用这条技术路径本身。

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

```
# policy.py::_asd_gate_memory(self, slot, cset=None, rgb=None, raw_slot=None) -> dict
if self.tid_head == "dam4sam":
    now_tick = self._s2_tick_counter
    if now_tick - self._dam4sam_last_reset_tick < self.dam4sam_reset_cooldown_ticks:
        return {"reset_sent": False, "reset_ok": False, "reason": "cooldown_skip", "latency_ms": 0.0}
    box = _clamp_bbox(_cand_to_box(cset.cands[slot]), image_w, image_h)
    t0 = time.monotonic()
    try:
        send_msg(self._dam4sam_sock, {"cmd": "init", "rgb": rgb, "box": box}, timeout=self.dam4sam_socket_timeout_s)
        resp = recv_msg(self._dam4sam_sock, timeout=self.dam4sam_socket_timeout_s)
        ok = resp is not None and resp.get("bbox") is not None
        self._dam4sam_last_reset_tick = now_tick
        return {"reset_sent": True, "reset_ok": ok, "reason": "ok" if ok else "bad_bbox",
                "latency_ms": (time.monotonic()-t0)*1000}
    except (socket.timeout, ConnectionError, EOFError) as e:
        reason = "timeout" if isinstance(e, socket.timeout) else "conn_error"
        return {"reset_sent": True, "reset_ok": False, "reason": reason,
                "latency_ms": (time.monotonic()-t0)*1000}
# 调用方(policy.py:707-729)按返回 dict 的 reset_ok 分支决定控制提交是否切到 pgif slot。
```

### Relevant References
- `carla_uav_tracking/rollout/policy.py:318-337`(`_asd_gate_memory`/`_asd_reset_memory`)— 现有 tah/reid 钩子实现模式,本计划的扩展点。
- `carla_uav_tracking/rollout/policy.py:386-413`(`_dam4sam_select`)— 构框与 socket 调用的既有先例,`_clamp_bbox`/box 构造应与此保持一致风格。
- `carla_uav_tracking/rollout/policy.py:707-729` — ASD gate/reset fire 的调用点,需要按新的结构化返回值调整控制提交分支。
- `carla_uav_tracking/rollout/asd.py:100-193`(`ASD.step`/`_filter_step`)— 确认 `filter_kind="tempered"` 的 fire 恒为 `kind=="gate"`,不需要改这个文件。
- `carla_uav_tracking/rollout/bridge.py` — `send_msg`/`recv_msg` 协议,本计划只复用不新增动词。
- `carla_uav_tracking/rollout/dam4sam_service.py:62-67` — 服务端 `cmd=="init"` 处理逻辑,确认行为但不修改此文件。
- `runs/asd/deploy_all/chains.sh` — 已验证存在,`tahrelm_pgif` 臂的调用模式是 `dam4sam_pgif`/`dam4sam_pgif_commit` 两条新 `run` 的直接模板。
- `carla_uav_tracking/rollout/deploy_compare.py` — `ARMS` 列表与配对统计逻辑,需要扩展为显式 pair matrix。
- `carla_uav_tracking/rollout/metrics.py` — 三项客观指标的 scorer,已验证 tid_head-agnostic,预期不需要改动。
- `acot_note/1001_poc-closure.md` — claim→证据矩阵与闭合判据的既有格式范例,round summary 应仿照其结构。

## Dependencies and Sequence

### Milestones
1. **核心钩子实现**(AC-1, AC-2, AC-3):`policy.py` 内 `_asd_gate_memory` 扩展、bbox clamp 辅助函数、结构化日志字段。
   - Phase A:签名扩展 + dam4sam 分支的核心逻辑(不含 cooldown/超时)。
   - Phase B:cooldown 计数器 + socket 超时/异常处理 + 回退语义。
   - Phase C:结构化日志落盘;非 CARLA 的本地 smoke(mock socket)验证 AC-1/AC-2/AC-3 的正负测试。
2. **评测 harness 插线**(AC-4, AC-5):依赖 Milestone 1 完成。
   - Step 1:`chains.sh` 新增两条 `run` 调用(`dam4sam_pgif_commit`、`dam4sam_pgif`)。
   - Step 2:`deploy_compare.py` 扩展显式 pair matrix 与有效 episode 数报告。
3. **闭环实验执行**(AC-6):依赖 Milestone 2 完成,且需要 `dam4sam-svc` 容器可用(复用现状,不改协议/不重启)。
   - Step 1:1-2 episode smoke 跑(确认事件日志与超时处理不卡死控制环,Codex round-2 OPTIONAL_IMPROVEMENTS 建议项)。
   - Step 2:Town05 seeds 91000–91019 三档 ablation 全量跑(baseline / commit-only / reset-hook)。
   - Step 3:统计与 AC-6.1 闭合判据核对。
4. **诊断分析与收尾**(AC-7):依赖 Milestone 3 产出的事件日志。
   - 对 `pgif_conf` 做错锁 tick 上的 AUROC 分析;整理 round summary(claim→证据矩阵格式)提交 Codex review。

<Describe relative dependencies between components, not time estimates>

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 扩展 `_asd_gate_memory` 签名(`slot, cset, rgb, raw_slot`),新增 `tid_head=="dam4sam"` 分支核心逻辑(发送 `init`、解析响应) | AC-1 | coding | - |
| task2 | 实现 bbox 构造 + clamp 辅助函数,接入 task1 | AC-2 | coding | task1 |
| task3 | 实现 cooldown 计数器(S2 tick 粒度,K=10) | AC-1.1 | coding | task1 |
| task4 | 实现 socket 超时(2s)与 `socket.timeout`/`ConnectionError`/`EOFError` 捕获,返回结构化 `reason` | AC-1.2 | coding | task1 |
| task5 | 调用方(`policy.py:707-729`)按返回值的 `reset_ok` 分支决定控制提交回退语义 | AC-1.3 | coding | task1, task4 |
| task6 | 新增结构化事件日志字段(`asd.py`/`policy.py` 日志写入路径) | AC-3 | coding | task1, task3, task4 |
| task7 | 本地 mock-socket smoke 测试:验证 AC-1/AC-1.1/AC-1.2/AC-1.3/AC-2/AC-3 的全部 Positive/Negative Tests(不依赖真实 CARLA/DAM4SAM 服务) | AC-1, AC-2, AC-3 | coding | task2, task3, task4, task5, task6 |
| task8 | `chains.sh` 新增 `dam4sam_pgif_commit`、`dam4sam_pgif` 两条 `run` 调用 | AC-4, AC-5 | coding | task7 |
| task9 | `deploy_compare.py` 扩展显式 pair matrix(三组配对)与有效 episode 数报告 | AC-6 | coding | task8 |
| task10 | 1-2 episode smoke 闭环跑(真实 `dam4sam-svc` + CARLA),确认事件日志与超时处理在真实环境下不卡死控制环 | AC-1, AC-4 | coding | task8 |
| task11 | Town05 seeds 91000–91019 三档 ablation 全量闭环跑(baseline / commit-only / reset-hook) | AC-6 | coding | task9, task10 |
| task12 | 统计三组配对差值(配对 bootstrap 95% CI),核对 AC-6.1 硬性闭合判据 | AC-6, AC-6.1 | coding | task11 |
| task13 | 对 `pgif_conf` 在错锁 tick 上做事后 AUROC 分析(诊断,不回头改门控) | AC-7 | analyze | task11 |
| task14 | 把 task12/task13 结果按 `1001_poc-closure.md` 的 claim→证据矩阵格式整理为 round summary,提交 Codex review | AC-6.1, AC-7 | analyze | task12, task13 |

## Claude-Codex Deliberation

### Agreements
- PGIF-tempered 的主路径触发 `kind=="gate"`,正确的挂接点是 `_asd_gate_memory`,不是 `_asd_reset_memory`;DAM4SAM 可以通过复用既有 `{"cmd":"init",...}` 消息做硬重置,不需要先改 `dam4sam_service.py` 协议。
- ablation 矩阵应包含:`dam4sam` baseline、`dam4sam+PGIF commit-only`(`gate_memory=false`)、`dam4sam+PGIF reset-hook`(`gate_memory=true`);oracle-reset 只作 GT diagnostic 上界,不进入 deployable 主结论。
- 评测口径统一用 `paper_dep_*` 命名(不用 `paper_ext_*`),seeds 91000–91019,Town05,配对统计只在共同 seeds 上进行,报告各臂有效计分 episode 数。
- DAM4SAM 原生 `m_iou` 不应作为本轮主线置信信号,只作诊断日志字段 + 事后 AUROC 检验保留,不阻塞主实验。
- `runs/asd/deploy_all/chains.sh` 确有其文件(`/nvidia/hque/code/cyhe/uav-acot-track/runs/asd/deploy_all/chains.sh`,Alt-3 草稿中"未找到"的记录已被 Codex 第一轮 TECHNICAL_GAPS 纠正并在本计划中采纳),新臂应仿照其中 `tahrelm_pgif` 行的调用模式接入。

### Resolved Disagreements
- **独立 "hook mode" 控制面 vs 复用现有 `asd_cfg` 机制**:Codex round-2 DISAGREE 认为不需要新增一套独立控制面,现有 `--asd filter` + `asd_cfg.gate_memory` 已经足够表达 baseline/commit-only/reset-hook 三档,只需新增 DAM4SAM 专属的 cooldown/timeout 两个 `asd_cfg` 字段。Claude 在 v2 候选计划中采纳,已体现为 AC-4 与本文档的 Allowed Choices。
- **失败回退语义的严格度**:Codex round-2 DISAGREE 指出"reset-hook socket timeout 时仍把 `_pred_slot` 改成 PGIF slot"不是真正的"保留 DAM4SAM 原始 pick"。Claude 在 v2 中采纳,改为钩子必须返回结构化状态(`reset_sent`/`reset_ok`/`reason`),调用方按状态显式分支(AC-1.3)。
- **`deploy_compare.py` 对比对象的隐式假设**:Codex round-2 DISAGREE 指出只加 `dam4sam_pgif` 到 `ARMS` 不会自动产出所需的多组对比。Claude 在 v2 中采纳,明确要求显式 pair matrix(三组配对,AC-6)。

### Convergence Status
- Final Status: `converged`(round 3 Codex 确认:无 REQUIRED_CHANGES,无高影响 DISAGREE;仅剩用户决策项,均已在 Pending User Decisions 中决议完毕)。

## Pending User Decisions

> 以下四项在 Codex 收敛循环中被标记为需要用户决策;全部已在 gen-plan 过程中由用户决议完毕,不再 PENDING,保留记录供追溯。

- DEC-1: 本轮实验的目标定位(影响是否允许修改 DAM4SAM 服务协议)。
  - Claude Position: 偏向最小 hook 验证创新点,复用现有协议,工程风险最低,12h 窗口内最稳妥。
  - Codex Position: 未给出强倾向,但在 UNRESOLVED 中列为需用户拍板的开放问题,隐含倾向于先用最小方案验证可行性。
  - Tradeoff Summary: 最小 hook 方案风险低、确定性强,但上限也更低(无法做 soft correction);最强系统方案上限更高但需要重启服务、协议改动,12h 内风险显著增加。
  - Decision Status: **用户决议:"最小 hook 验证创新点"**。
- DEC-2: 是否允许修改 `dam4sam_service.py` 协议并重启 `dam4sam-svc` 容器。
  - Claude Position: 不允许,只用现有 `init` 命令,与 DEC-1 的最小 hook 方向一致。
  - Codex Position: 同意不允许作为默认假设;若后续方向(如 Alt-1 的 `m_iou` 透传)需要,可在下一轮单独决议。
  - Tradeoff Summary: 不改协议 = 无需协调外部 torch2.1/SAM2 环境重部署,无服务中断风险;改协议 = 为后续 m_iou 透传等方向预留空间,但本轮用不到。
  - Decision Status: **用户决议:"不允许,只用现有 init 命令"**。
- DEC-3: legacy `_asd_reset_memory`(非 gate 路径)是否也要接 DAM4SAM 分支。
  - Claude Position: 不接,只接 gate,与 Codex 建议一致。
  - Codex Position: "我仍建议主线只接 gate,reset 只保兼容"(round-2 UNRESOLVED 原话)。
  - Tradeoff Summary: 只接 gate = 实现与测试面更小,且 PGIF-tempered 主路径本就只触发 gate;两者都接 = 增加实现/测试面,但对本轮主实验无增量价值(当前 `filter_kind="tempered"` 从不触发 reset 路径)。
  - Decision Status: **用户决议:"不接,只接 gate"**。
- DEC-4: DAM4SAM reset 的 cooldown(最小重置间隔)与 socket timeout 默认值。
  - Claude Position: K=10 S2 tick(≈6s)、timeout=2s,作为较保守的先验默认值。
  - Codex Position: "cooldown 默认 K 需要定一个先验值并冻结,不能看结果后调参"(round-2 REQUIRED_CHANGES),未给出具体数值倾向。
  - Tradeoff Summary: K=10/timeout=2s 更保守,不易频繁打断 DAM4SAM 自身的长时序优势;K=5/timeout=1s 更激进,纠偏更快但更容易把 DAM4SAM 的粘滞优势也一并打掉。
  - Decision Status: **用户决议:"K=10 tick(≈6s),timeout=2s"**。

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- 本计划新增/修改的代码应沿用仓库既有注释风格(见 `policy.py`/`asd.py` 现有函数的 docstring/行内注释),解释"为什么"而非复述代码;不要引入与本 plan 任务编号相关的术语(如不要写 `# AC-1.1: cooldown check`,应写 `# rate-limit DAM4SAM re-init to avoid clobbering its own temporal consistency`)。

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
