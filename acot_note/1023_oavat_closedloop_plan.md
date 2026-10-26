# 1023 OA-VAT 完整版闭环接入方案 v4（同感知；v4 = 加入遮挡感知规划器；送 codex R5 审查）

v1 / v2 存档于 `runs/oavat_full/codex/closedloop_plan_v1.md`、`closedloop_plan_v2.md`。codex R1（`plan_review_r1_final.md`，REVISE：3 BLOCKER / 14 MAJOR / 3 MINOR）的逐条处理见 §7；R2（`closedloop_r2_final.md`，REVISE：B1–B3 全部 RESOLVED，剩 2 MAJOR + 2 MINOR）的处理见 §7.1。

**已实现**（本轮一并审查）：
- `acot/runtime/oavat_closed.py`（`OavatClosedRuntime`）；
- `server.py` 中的 runtime 选择和 `oavat_frames` collector；
- `schema.py` / `support.py` / `resolve.py` 中的 head 登记；
- 方法配置 `ext_oavat_full_{g1,g2,g3,n0}.yaml`；
- 实验配置 `stress_tune_oavatcl_grid.yaml`、`stress_devS_selfflight_oavatcl.yaml`；
- 单测 `acot/tests/test_oavat_closed.py`（17 项）；
- 计分与选择 `runs/oavat_full/score_closedloop.py`，以及其合成失败路径测试 `test_score_closedloop.py`；
- 离线自检 `runs/oavat_full/closedloop_replay_check.py`。

## 0. 背景与目标

- **已完成：同感知开环 r5**（`runs/oavat_full/openloop_stress.py`；codex R5 PASS；结果见 `1023_stress_baselines.md` §2 †）。
  - R1 中的 B3（串行重捕获）、M2（几何读出偏向目标）、M3（结构化状态）、M8（开局与首 tick 计分）已在开环 r3–r5 修复。闭环沿用同一套实现。
- **目标**：在 dev-S（`stress_devS`，40 个种子，Town05/01，自主飞行）上，让 OA-VAT **用自己的 PID 控制器飞**，得到系统级的 TR90、M1、承诺正确和 M2。
  - 与已有自主飞行臂按种子配对比较：U1、re-ID、永不切换、Ours-CV、OA-VAT 身份模块版、DeepSORT、OC-SORT、Ours-CV（TAH 重训）。
- **定位**：这是"原系统迁移到我们场景"的系统级对照，不是同一飞行器下的身份对比。身份层面的对比由开环 r5 和模块版承担。

## 1. 运行路径（R1 M7：显式的原生控制器路径）

新增 `acot/runtime/oavat_closed.py`，其中 `OavatClosedRuntime` 的接口与 `server.Runtime` 相同（`reset` / `step` / `cand_lite` / `policy.committed` / `info` / `dets`）。

- **选择方式**：`server.main` 在 `method.identity.head == "oavat_full"` 时构建 `OavatClosedRuntime`，其余方法不变。
  - `schema.HEADS` 增加 `oavat_full`，配套 `identity.oavat_full` 块（§2 的参数全部写进 resolved 配置，计入 config hash）。
  - `support.method` 只对这个 head 放行，并拒绝 `expert_control=true`、`ablate`、`frame_gain` 等不适用的开关。
- **不加载的组件**：Qwen、Stage-2（EAR / IAR / DiT / grounding）、开局 attr 打分器。
- **保留的共享组件**：`SimProxyPerception`，即协议候选框、射线遮挡、DINOv2 裁剪特征加 `TrackletAssigner`，与其他臂相同的跟踪器 key。
  - 网格池化特征 `vlm_feat` 只给 grounding 用，这里传 dummy grid（与 `frame_detections` 的做法相同）。
  - 洗牌 RNG 流（`self.rng`）与其他臂一致，所以 key 分配规则与其他臂一致。
- **调度**：S2 tick 固定为 `step % 6 == 0`，与 EAR 状态无关，没有 `z_ex is None ⇒ 每帧 S2` 的分支。
- **客户端**：`expert_control=false`（自主飞行），与其他自主飞行臂的协议和客户端完全相同。
- **颜色通道（R1 M6）**：每帧只转换一次 `bgr = ascontiguousarray(rgb[..., ::-1])`。参考特征构建、GTDetector 注册和任务帧都使用这一份 BGR（与开环 r5 相同）。

### 1.1 每帧流程

1. **生成本帧协议框**：按 `frame_detections` 的规则生成（与 DeepSORT / OC-SORT 每帧框同一规则、同一真值通道等级），独立 RNG 流 `frame_rng`。结果注册给 GTDetector，替换 YOLOE。
2. **S2 tick**：运行 `perception.process`，得到本 tick 的 `dets` 和 key。
3. **开局（与开环 r5 相同，R1 M8）**：
   - 当 S2 tick 满足 t < 18 且目标在本 tick 的 key 中（等价于 `server.shared_slot(gt_target)`），**无条件重启** OA-VAT：
     - 参考特征用目标在本帧的协议框构建；
     - 本帧的首帧选择只提供目标框，相似度阈值仅在本帧放宽到 −1；
     - 承诺强制为目标 key，K5 状态清空。
   - 在第一个合格 tick 之前，帧 0 用 OA-VAT 自己的无参考启动。这是开环 r5 已登记的协议。
   - OA-VAT 在开局之后不再接收任何真值交接。
4. **`OavatFull.step(bgr)`**：得到本帧的 `cmd` 或 `None`，以及结构化事件：
   - `SELECTED`：ORTrack (re)initialize 成功后记录的框；
   - `CTRL`：`compute_action` 的输入，即控制所依据的点、面积和 `is_predict`；
   - capture 阶段。
   以上沿用开环 r5 的 hook。
5. **动作**：见 §2。
6. **承诺**：见 §3。

### 1.2 每局重置（R1 M9）

- `reset()` 依次执行：`oa.stop()`，若线程仍存活则抛错，当前 worker 判 INFRA 并重启容器，绝不复用旧任务；`reset_tracker`（ORTrack 的损失计数、状态、`frame_id`）；清空 `SELECTED`、`CTRL`、`WORKER_ERRORS`、PID 状态（已核对：`tracking/ortrack_core.py:75-86` 在每次进入跟踪循环时新建 PID 对象，无需额外处理）；`M.mission_counter = 0`。
- **RNG**：每局用 `torch.manual_seed`、`np.random.seed` 和 `random.seed` 设种子，种子为 `rng_base(seed, repeat) & 0x7FFFFFFF`（R2 #2：64 位哈希截成三个 RNG 都接受的范围）。perception 自己的洗牌流不变。
- `cudnn.deterministic=True`，`autocast` 与开环 r5 相同（主循环使用原设置，worker 内关闭）。
- **种子顺序无关性检验**：同一进程依次跑 A→B 与 B→A，对比每帧 `cmd` 和每 tick 的承诺（离线：在开环录制上回放，不经 CARLA；见 §6.2）。

## 2. 动作映射（R1 B1、M4、M5）

- **通道**：OA-VAT 的 `send_velocity(vx=0, vy, vz=0, yaw_rate)`。原版平面仿真同样只用 `[vy, yaw_rate]`（`DRONE/unreal_sim_controller.py`）。
  - `forward_command = cmd[1]`，`yaw_command = cmd[3]`。
  - `cmd[0]` 和 `cmd[2]` 若不为 0，直接报错（原版恒为 0）。
- **换算**（ψ 为无人机当前偏航角；位移由偏航决定，与相机 −50° 俯仰无关）：
  ```
  dx   = g_f · forward_command · 0.1 · cos ψ        # g_f: m/s per unit；0.1 s = 帧间隔（ctrl_dt = 0.1）
  dy   = g_f · forward_command · 0.1 · sin ψ
  dz   = 0                                          # 原版禁用垂直控制；保持绝对高度
  dyaw = g_ω · yaw_command · 0.1                    # g_ω: deg/s per unit
  ```
  - **符号**：原版 `error_x = cx − W/2 > 0`（目标在右）时 yaw PID 输出为正，表示右转。CARLA 中 yaw 增大即右转（UE 左手系，+y 在右）。因此 `g_ω > 0`。上线前用几何自检核实（§6.1），不靠回归。
- **没有指令的帧**（`cmd is None`：首帧检测、预测 / 重捕获中未发送指令、任务已结束）：动作为 `(0, 0, 0, 0)`，即悬停。
  - 依据是原版 `VIDEO/unrealcv_stream.read()`（:39-57） 每读一帧先执行一次零动作（stop）。丢失时原版也显式发送 `send_velocity(0, 0, 0, 0)`。
- **限幅**：原版已按 `ctrl_max_speed = 40` 单位对 `forward_command` 和 `yaw_command` 限幅。我们不额外增加共享动作限幅，所以不改协议。§2.1 给出最大物理量：
  - **专家包络**（R2 #1 更正）：实际生效值为 `xy_max = 12 m/s`、`yaw_rate_max = 30 °/s`（`carla_uav_tracking/config/default.yaml` 中的 `uav.expert_policy.velocity_limits`，stress_v1 不覆盖）。v2 引用的 15 / 60 是构造函数的回退默认值。
  - **网格 G1–G3 全部在此包络内**。
  - **N0 是唯一例外**：偏航 40 °/s，超过 30 °/s。它作为"原生单位参考"保留原值，在配置中单独声明上限 `max_yaw_dps: 40` 并披露。
  - 每个配置都声明 `max_speed_mps` / `max_yaw_dps`；runtime 启动时检查 `40·g ≤ 声明上限`，不满足就拒绝启动。
- **PID 参数**：原版默认值（`main.py:139-151`，`ctrl_dt = 0.1` 与我们的 10 Hz 一致），不调。
- **面积设定点（经验值）**：`target_area_ratio = 0.0028`。
  - 来源：开环 r5 的 114 局专家飞行录制中，OA-VAT **自己的 ORTrack Track 框**面积中位数为 735 px²，735 / 512² = 0.0028。
  - 分位数：P10 372，P90 1980 px²。
  - 这是专家取景下的经验统计，不是几何推导。原版默认 0.12 相当于 177 px 的方框，在我们的场景里会导致前进指令恒为饱和（R1 M4：81,257 / 81,539 帧为 +40）。
  - 容差 `area_tolerance_ratio = 0.20` 用原版默认值。
- **不做专家动作回归**（R1 M4）。

### 2.1 增益配置（预先登记，联合缩放；R1 M10、M11）

| 配置 | g_f (m/s/单位) | g_ω (°/s/单位) | 饱和 ±40 时的物理量 | 用途 |
|---|---|---|---|---|
| **N0 原生单位** | 0.01 | 1.0 | 0.4 m/s, 40 °/s | 固定换算参考：按 UnrealCV `set_move {angle} {velocity}`（`gym-unrealcv/.../tracking/interaction.py:149`）的通常单位 cm/s 与 °/s 解读（**假设的单位**，仓库只给出参数顺序）；**不调**，dev-S 上一并报告；偏航超专家包络，单独声明并披露 |
| G1 | 0.10 | 0.25 | 4 m/s, 10 °/s | 网格 |
| G2 | 0.20 | 0.50 | 8 m/s, 20 °/s | 网格 |
| G3 | 0.30 | 0.75 | 12 m/s, 30 °/s | 网格（= 专家包络） |

- 联合缩放共 3 组网格，外加 N0 参考，不做笛卡尔组合。不设回退网格：失败的配置照常参与选择。

## 3. 承诺与计分（R1 B2、M1、M3、M14）

- **飞行不依赖承诺**：OA-VAT 只根据自己的框飞。承诺只是计分读出，与其他臂一样只在 S2 tick 更新，帧间保持（sample-and-hold）。客户端照常对每帧计算 `pc_correct` / `tr90`（`info.committed_key`），`ticks.jsonl` 照常计算 M1。
- **S2 tick 的读出（与开环 r5 相同）**：
  - 只用**新鲜观测**：本帧的 `SELECTED`（重新初始化框），否则用本帧 Track 阶段的跟踪框。
  - 以框中心落入的协议候选框为准，按与中心的距离取最近，并列时取最小 key。这是与身份无关的几何规则，直接对应**本 tick 的** `Detection.key`。
  - 没有新鲜观测（Lost / Predict / Detect / 已结束）或没有命中时，**保留上一个 key**，不查它所记住车辆的后继。与其他臂的 `self.committed` 语义相同：key 被跟踪器转手时，按转手后的结果计分。
  - 读出用到的车辆标签只在计分中使用。
- **原生立即切换**（主读法）：OA-VAT 原版没有 K 票迟滞，本身就是系统差异，需要披露。
  - **敏感性分析**：用每 tick 的同一组读出，离线经共享 K = 5 规则得到 `committed_k5`，并报告其 M1 和 pc_tick（只影响计分，不影响飞行）。
  - **几何读出诊断**：同一读出规则作用在 Ours-CV 的 ticks 上（与开环 r5 相同）。
- **记录（R1 M3）**：每帧记录 `stage`、`track_bbox`、`predict_bbox`、`selected`、`ctrl`（控制点、面积、是否预测）、`cmd`、映射后的 `action`、`reseed`、`mission status` 和 `worker_errors`。每 tick 记录 `geo_key`、`n_hit`、`fresh`、`forced`。
  - 写入 `ticks.jsonl` 的扩展字段，以及 `oavat_frames.jsonl`（新增 collector `oavat_frames`，仅限此 head）。
- **计分口径（R1 M14、m2）**：
  - TR90 的分母固定为 900 帧，截断后的帧计为未跟踪；
  - M1 / M2 / pc 未定义的局不计入均值，并报告各自的局数；
  - 分别报告完成、截断、INFRA 和算法失败的计数；
  - 配对取交集，与 `selfflight_extra.py` 相同；
  - 重新捕获指标依赖各臂自己的轨迹，事件不是配对的，因此只作描述性报告（按种子聚类的 bootstrap），并给出删失事件数。

## 4. 网格与选择（R1 M10、M11）

- **调参种子（冻结）**：`stress_tune` 中 Town05 的 94000、94001、94003 和 Town01 的 94040、94041、94042。
  - 规则：两个 dev-S 城镇各取前 3 个录制状态为 complete 的种子。
  - 不用 Town10HD，dev-S 不用于调参，test 不触碰。
- **运行**：G1、G2、G3 各跑 6 局，共 18 局，自主飞行。N0 不参与选择，所以不在调参集上跑。
- **选择统计量**：按局平均的 TR90 最高者胜。
  - 并列（差 < 0.005）时比按局 M1；仍并列时取较小增益，更保守。
  - **INFRA**：每局共 3 次尝试（`max_infra_attempts: 3` 指总次数）。3 次后仍未解决的局会**暂停选择**，直到解决为止；不换种子，也不删除某个配置的局（R2 #3）。
  - **算法失败**（截断、任务结束后悬停）：照常计分，TR90 记为实际值。
  - **M1 未定义**：该配置在 M1 一级的并列比较中不能胜出，并列直接进入"取较小增益"。
  - 以上规则实现在 `score_closedloop.select`。
- **调参预算披露**：
  - OA-VAT 闭环：3 组配置 × 6 局，外加面积设定点的经验统计；
  - 其他臂：re-ID 有 dev-B 调参记录（`ext_reid.yaml:2`）；Ours-CV 的 `conf_tau` 等参数来自 dev-B；均未在 stress 上调参。
  - 论文中同时给出 N0（不调参）和选定配置的结果。

## 5. 评测

- **dev-S**：选定配置 G\* 和 N0 两个臂，各 40 局（94423 目标静止，按协议跳过）。
  - experiment 为 `stress_devS_selfflight_oavatcl.yaml`，protocol、seeds 和 schedule 与 `stress_devS_selfflight` 相同。
  - 报告 M1、按 tick 合并的 M1、pc、TR90、M2、K5 敏感性，以及描述性的重新捕获指标。
  - 与已有各臂按种子配对，做 bootstrap（10,000 次）。
- **资源（R1 M13）**：6 个 worker，GPU 0/1/2/3/5/6，每卡 1 个 CARLA + 1 个策略容器。不另开 GPU 任务，同期不跑其他评测。
  - CARLA 端口 2070、2260、2010、2250、2090、2050，加 flock，与之前的自主飞行相同。
  - 启动前检查 `nvidia-smi` 和其他用户的 CARLA（2100 / 2110），不触碰它们。

## 6. 上线前自检

1. **几何符号**（离线，单元测试）：在 ψ = 0°、90°、−90° 下，输入 `forward = +10` 时位移沿机头方向，`yaw = +10` 时右转。再用一个合成目标验证：目标在画面右侧 → yaw PID > 0 → 映射后的 dyaw > 0 → 重新投影后目标向画面中心移动（用 `acot_probe.projection`）。
2. **离线确定性**：在开环录制的 2 局上，以闭环运行路径的"读出 + 动作映射"跑新进程两次，以及 A→B / B→A 两种顺序，逐帧 `cmd` 和逐 tick 承诺须完全一致。闭环本身不承诺逐位一致（R1 M12）。
3. **计分对齐**：在开环录制上，用闭环路径的计分代码回放，每 tick 的 `committed` 须与开环 r5 的 `committed` 完全一致（同一读出规则）。
4. **key 对应的对抗测试**（R1 B2）：构造 key 删除、转手、重新出现和遮挡几种序列，确认只保留旧 key，不发生重新绑定。
5. **冒烟**：G2 跑 2 局（94000、94040）的闭环，确认 action 非零、飞行稳定（没有爆冲、没有原地打转）、`gt_access` 只有协议感知的读取。同一种子重复 1 次，报告轨迹和指标的差异。
6. **失败路径与计分口径**（R2 #4）：`runs/oavat_full/test_score_closedloop.py` 用合成数据检查以下几点：截断局的 TR90（分母固定 900）保留在 TR90 样本中；M1 未定义的局只从 M1 样本中排除；skipped 局不计；未解决的 INFRA 会暂停选择；并列规则正确；K5 敏感性正确。
7. **控制器检查**（R2 #4）：在单测中直接调用 OA-VAT 原版 `compute_drone_action_while_tracking`，验证以下几点，并经 G2 映射后在 −50° 下检查位移方向：
   - 小于设定点的框（目标远）→ 前进；
   - 大于设定点的框（目标近）→ 后退；
   - 在容差内 → 前进量为 0；
   - 目标在右 → 右转。
   另外还检查左侧目标 → 左转、`None` 和全零指令 → 悬停。
8. **codex 代码审查**：通过后才跑网格。

## 7. 对 R1 的逐条处理

| R1 | 处理 |
|---|---|
| B1 前进通道错误 | §2：`cmd[1]` 作为前进量，§6.1 几何自检 |
| B2 车辆重新绑定 | §3：只用新鲜观测对应本 tick 的 key，否则保留旧 key；§6.4 |
| B3 串行重捕获 | 开环 r3 已修（`_process_one` 在原入队点同步执行，`reacq_parallel=True`），闭环复用 |
| M1 承诺节拍 | §3：S2 采样并保持；原生立即切换作主读法，并披露；K5 作敏感性 |
| M2 并列时偏向目标 | 开环 r2 已改为与身份无关的规则（最近中心，并列取最小 key） |
| M3 状态记录 | 开环 r4 的结构化 hook，闭环多记 `cmd` 和 `action` |
| M4 专家回归 | 删除；单位来自原版仿真语义，符号来自几何，量级由预登记的网格决定 |
| M5 限幅 / 设定点 | §2：最大物理量全在专家包络内，不加共享限幅；设定点标明为经验统计 |
| M6 RGB/BGR | §1：只转换一次 |
| M7 expert_flown 借用 | §1：独立 Runtime，固定节拍，不加载 Qwen / Stage-2 |
| M8 开局 | 与开环 r5 相同（R5 PASS 的版本），所有 S2 tick 都计分 |
| M9 线程 / 状态 | §1.2 |
| M10 调参公平 | §4：冻结种子、统计量、并列和失败规则；披露预算；N0 参考 |
| M11 网格数量 | §2.1：3 组联合配置 + N0，共 18 局调参 |
| M12 确定性 | §6.2 离线逐位一致；闭环报告重复差异 |
| M13 资源 | §5：严格 6 卡；时间按实测 |
| M14 计分口径 | §3 末尾 |
| m1 | 不做回归，不适用 |
| m2 | §3：未定义值不计入均值，并报告局数 |
| m3 | 方案不再引用缺失章节；开环脚本示例在实现时修正 |

### 7.1 对 R2 的处理

| R2 | 处理 |
|---|---|
| #1 MAJOR 专家包络错误 | §2 更正为 12 m/s、30 °/s。网格改为 G1 (4, 10)、G2 (8, 20)、G3 (12, 30)。N0 偏航超出包络，单独声明并披露。runtime 检查声明上限，单测核对包络取值与 stress_v1 不覆盖 |
| #2 MAJOR NumPy 种子溢出 | §1.2：`rng_base(seed, repeat) & 0x7FFFFFFF`。§6.2 自检经过真实的 reset 和 OA 推理路径（`closedloop_replay_check.py`） |
| #3 MINOR INFRA 规则 | §4：每局共 3 次尝试；未解决则暂停选择，不换种子、不删局；M1 未定义时的并列规则已写明 |
| #4 MINOR 失败路径 / 取景检查 | §6.6、§6.7 |

### 7.2 对 R3 的处理（R3：R2 全部 RESOLVED；实现层 1 BLOCKER + 3 MINOR）

| R3 | 处理 |
|---|---|
| #1 BLOCKER 缺 `rt.oracle` | `OavatClosedRuntime.oracle = None`。新增 `runs/oavat_full/server_smoke.py`：经真实的 `python -m acot.runtime.server` 入口跑启动 → reset → 120 帧 act → reset（flush）→ close。结果 PASS：5 个文件齐全；gt_access 0 违规、0 次 oracle 调用；wire 与 ticks 中的 `committed` 一致，并与开环 r5 一致；`oavat_frames` 每帧一行 |
| #2 MINOR 算法失败计数 | `score_closedloop.episode` 取最后一个 tick 的 `oavat_ended`，区分 `n_algo_error`（任务抛错后悬停）和 `n_algo_ended`（结束或 FAILED 后悬停）。这些局仍计入所有指标样本，不算 INFRA。补了合成测试 |
| #3 MINOR cudnn | runtime 初始化时设 `cudnn.deterministic=True`、`benchmark=False`，§6.2 的 A→B / B→A 比较重跑 |
| #4 MINOR 读出精度 | 不复现 r5 的 0.1 px 序列化舍入，使用全精度检测框（两者都与身份无关）；在 `associate` 文档字符串和此处披露。"与 r5 完全一致"只针对 §6.3 已检验的局 |

实测吞吐：server 入口约 1.1 帧/s（120 帧，含预热，不含 CARLA），低于开环约 4 帧/s。冒烟时测量含 CARLA 的完整单局用时，再修正 §8。

## 8. 时间（墙钟，条件性）

| 阶段 | 预计 |
|---|---|
| 方案审查（本轮起 1–2 轮） | 2–4 h |
| 实现 Runtime / schema / 单测 | 4–6 h |
| §6 自检、冒烟、代码审查 | 3–5 h |
| 网格 18 局（6 worker，3 轮） | 0.5–1 h |
| dev-S：G\* 和 N0 共 80 局（6 worker，约 14 轮） | 2–4 h |
| 评分与文档 | 1–2 h |

每局时间开工前用冒烟实测。开环中每帧约 4 fps；闭环还要等 CARLA，按每局 6–12 min 估计。

## 9. 需披露的偏离与局限

1. **检测器换成协议框**：没有 mask，使用矩形回退。
2. **参考特征**来自 T1 开局的真值框；开局交接是强制的。
3. **DINOv3 vitb16**：README 的提取权重。
4. **线程同步化**：worker 零延迟。
5. **DP 救援关闭**：没有公开权重。
6. **控制器换算与调参**：N0 / G\* 都报告。
7. **面积设定点**为经验统计。
8. **立即切换**：没有 K 票。
9. **仿真等待推理**：没有实时延迟。
10. **特权输入差异**：OA-VAT 不使用框的世界坐标，Ours / re-ID 的 WHERE 使用。
11. **原控制器为平视 Tello 设计**：迁移到 −50° 俯视后的表现不代表原论文。
12. **E3 镜头移开**在自主飞行中不执行，与其他自主飞行臂一致。


## 10. v4：遮挡感知规划器（扩散策略）开启版（2026-10-25 用户决定）

**背景**：
- OA-VAT 论文标题是 *Instance-level Visual Active Tracking with Occlusion-Aware Planning*，规划器是论文的核心组件。
- 作者于 2026-10-20 在 HF `Harry2222/follow-dp-unrealcv` 公开了权重 `checkpoints/dp_unrealcv_20k_ep40.ckpt`（4.4 GB），README 同步更新（上游 `93745e3`）。
- 上游代码与我们的 `cea1f6b` **逐文件相同**：`main.py`、`dp/*`、`tracking/*`、`control/*`、`config.py` 的 sha256 一致。变的只有 README 和新放出的权重。
- 上游仍标注 "Optional"，`--dp_enable` 默认关闭。
- 之前关闭规划器的理由是"没有公开权重"，现在已不成立，所以闭环改为：**主表报规划器开启版，关闭版作消融**，N0 参考臂保留。

### 10.1 原版行为（代码事实）

- **触发**：跟丢后进入预测模式，`predict_frames_count >= kf_max_predict_frames`（默认 15 帧）时：
  - 若 `dp_enable`，调用 `dp_rescue`（`tracking/ortrack_core.py:396-445`）；
  - 否则返回 REDETECT。
  - 规划器只在跟丢后介入，正常跟踪时仍由 PID 控制。
- **补救循环**（`dp/dp_rescue.py`）：每个规划步依次执行：
  1. 读一帧，用 `build_dp_obs_from_frame`（缩放后的图像 + 归一化后的 KF 预测框）生成观测；
  2. `predict_action` 得到位置序列，差分后转成速度：`dp_action_to_tello_vel` 输出 `(vy, vz, yaw)`；yaw 在 dy < 0 时乘以 30，限幅 ±60；vy、vz 限幅 ±40；
  3. `send_velocity(0, vy, vz, yaw)`；
  4. 在 `dp_dt = 1 s` 内**按墙钟时间**读帧排空；
  5. 再读一帧，用 DINOv3 原型做全图匹配，命中即返回并重新初始化 ORTrack。
  - `dp_max_plans = 0` 表示不限次数，直到找回或视频结束。

### 10.2 适配（只有这些偏离，均需披露）

1. **依赖**：hydra-core 1.3.2、omegaconf 2.3.0、dill 0.3.8、**diffusers 0.25.0**、zarr 2.16.1、pandas 2.2.3 等装在独立目录 `runs/pydeps_oavat_dp`，不改镜像。
   - **diffusers 参照版本**（codex R5 #1）：规划器在 OA-VAT 主进程内运行，主环境 `follow`（`environment.yml` 与 `requirements.txt`）固定 diffusers 0.25.0，所以以 0.25.0 为准。diffusion_policy 子项目自带环境的 0.11.1 只用于训练。
   - 0.25.0 导入时引用 `huggingface_hub.cached_download`（镜像的 hub 1.31 已移除），推理路径不会调用，因此加了一个只用于导入的别名。
   - 实测：在同一 checkpoint 配置（100 步、cosine、fixed_small、clip、epsilon、16 步推理）下，0.25.0 与 0.30.3 的 `DDPMScheduler` 用固定输入和固定噪声逐步推理，结果**逐位相同**（最大差 0.0，timesteps 均为 [90, 84, …, 0]）。
   - `wandb` 只在训练 workspace 的 import 时用到，推理用不到，所以用一个空模块代替。
   - 加载后检查 `DP_CACHE['policy']` 不为空，否则拒绝启动，防止"规划器静默关闭"。
2. **仿真时钟**：`dp_rescue` 模块里的 `time` 换成 `SimClock`。每读一帧推进 0.1·(1+10⁻⁹) s，`sleep(s)` 推进 s。空操作版本在浮点舍入下可能死循环，已改掉；10⁻⁹ 的余量保证 10 帧一定达到 dp_dt = 1 s。时钟每局重置。
   - 一个规划步固定为 10 帧排空 + 1 帧匹配，与机器快慢无关。
   - 原版按墙钟计时，在同步仿真里每步读到的帧数会随负载变化，不可复现。
3. **指令保持**（这是我们为迁移做的选择，按 Tello 风格，原版 UnrealCV 并未规定，codex R5 #3）：规划步内只在开头 `send_velocity` 一次，之后的帧沿用该指令，直到下一条指令（Tello rc 语义）。
   - 原版 UnrealCV 的行为不同：`send_velocity` 执行一步环境，之后每次读帧都会下发 stop。
   - 实际保持时长：普通动作为 1.1 s 仿真时间（10 帧排空 + 1 帧匹配）；跨重新规划时为 1.2 s（再加 1 帧规划读帧）。实测指令起点为 40→51、106→118。
   - 依据是 `dp_action_to_tello_vel` 的设计：速度 = 位移 / dp_dt，即假定该速度持续 dp_dt。
   - 规划器以外的"无指令帧"仍然悬停（§2 不变）。
4. **丢弃 vz**：OA-VAT 自己的 UnrealCV 评测中跟踪体是平面体，`UnrealSimController.send_velocity` 只下发 `[vy, yaw_rate]`（`DRONE/unreal_sim_controller.py:57`），规划器在那里训练和评测时 vz 也不起作用。
   - 因此 dz = 0。原始 `cmd`（含 vz）照常写进 `oavat_frames`。
5. **偏航上限**：规划器的偏航限幅是 60 单位（不是 40）。配置中另外声明 `max_yaw_dps_dp`，runtime 检查 `60·g_ω ≤ max_yaw_dps_dp`。
   - G1：15 °/s；G2：30 °/s；G3：45 °/s。G3 会超过专家的 30 °/s，若 G\* = G3 需要单独披露。
6. **域差**：规划器在 UnrealCV（UrbanCity，行人目标，平视）上训练，迁移到 CARLA（车辆，−50° 俯视，约 27 px）有明显域差。照实写明，不重新训练。

### 10.2b 其他披露（codex R5）

- **补救不设上限**：沿用原版 `dp_max_plans = 0`，补救一旦开始，会一直替代 REDETECT，直到匹配成功；可能持续到本局结束。
  - 报告以下统计：触发次数、重新规划次数、补救占用的帧数、找回次数。
  - 冒烟实测：94098 离线 422 帧找回 0 次；CARLA 94000 两次运行均为 286 帧找回 0 次。
- **G\* 来自规划器关闭的网格**：开启版没有单独调增益，而这组增益同样影响补救期间的动作。在文档中写明"共用冻结增益的消融"。
- **规划器匹配只在其所在帧算新鲜观测**：补救成功时，本帧的匹配框（`DP_Found`）作为新鲜观测参与 S2 读出。下一帧用同一个框初始化 ORTrack，这一步只记为 `dp_init`，不再算新鲜观测（codex R6/R7）。

### 10.3 增益与评测设计

- **增益**：开启版沿用网格（规划器关闭）选出的 G\*。网格选的是 PID 换算；规划器只在丢失后介入，并通过同一换算执行。在文档中写明"G\* 是在规划器关闭时选出的"，不为开启版另跑网格。
- **dev-S 三个臂**：`oavatcl_dp`（G\* + 规划器，主表）、`oavatcl_sel`（G\*，规划器关闭，消融）、`oavatcl_n0`（N0，规划器关闭，固定换算参考），共 120 局。
- **开环敏感性**：`openloop_stress.py --dp` 生成 `stress_openloop_r5dp`（114 局）。
  - 专家飞行下规划器的动作不执行，但它的补救循环替代了 REDETECT 路径，且每个规划步后都会重新匹配，身份读出可能变化。
  - r5 仍是开环主结果，r5dp 作为敏感性分析报告。
- **自检**：
  - 规划器加载与暖机；
  - SimClock 下每个规划步正好 11 帧；
  - 指令保持与 vz 丢弃的单测；
  - 开启版 server 入口检查（`server_smoke.py --arm`）；
  - CARLA 冒烟 2 局：确认规划器至少触发一次，并报告触发次数和找回次数。
