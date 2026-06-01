# Paper1 §4.2.4 评估指标操作化说明

本文档说明仿真中**六项**论文指标与代码字段的对应关系及判据来源。实现入口：

- 回合聚合：`uavlab.paper1.metrics.paper_metrics.compute_paper_episode_metrics`
- 环境状态：`uavlab.paper1.sim.env.Paper1Env`（`covered` / `returned` / `poi_cov_time_s` / `poi_return_time_s`）
- 链路恢复：`uavlab.paper1.metrics.link_recovery.LinkRecoveryRecorder`
- 输出：`metrics.jsonl`（runner）、`summary.csv`（`scripts/sweep.py`）

## 双链路阈值（控制 vs 任务数据）

Paper1 按 `UAV_GCS_通信阈值设置说明.md` §9 区分两条链路，配置在 `configs/base.yaml` → `comm.control_link` / `comm.data_link`，解析为 `Paper1DualLinkThresholds`（`uavlab.paper1.comm.thresholds`）：

| 链路 | 丢包率 | 时延 / 回传 | 代码用途 |
|------|--------|-------------|----------|
| 控制/遥测 | ≤ 5% | ≤ 1 s | `link_loss_safe`（文档对照；快环 FSM 主用数据阈值） |
| 任务数据 | ≤ 20% | \(T^{ret}\) ≤ 10 s | 慢环 `comm_path_quality_mask`、`try_return_key`、FSM `link_loss_recover` |
| 弱通信/失联 | ≥ 50% | — | `link_drop_loss_p`、链路恢复事件起点 |

## 1. 空间覆盖率 \(R_{cov}\)（式 57）

\[
R_{cov}=\frac{1}{N}\sum_i c_i,\quad c_i\in\{0,1\}
\]

- **操作化**：POI \(i\) 进入覆盖半径且驻留满足 `poi_dwell_steps[i] >= poi_dwell_steps_required` 时置 `c_i=1`（`env.covered`）。
- **字段**：`R_cov`（主键）、`coverage_ratio`（别名）。

## 2. 任务有效完成率 \(R_{task}\)（式 58）

\[
R_{task}=\frac{1}{N}\sum_i e_i,\quad e_i=c_i\cdot r_i
\]

- **操作化**：
  - `r_i=1` 当且仅当 POI \(i\) 的关键数据已回传（`env.returned`）。
  - 覆盖时将该 POI 的 `key_bits` 记入 `poi_pending_bits` 并累加 `backlog_bits`。
  - `progress_key_return`（别名 `try_return_key`）按步以 \(b^{eff}\cdot\Delta t\) **分片传输**各 POI 的 `ReturnBuffer`（FCFS）；`remaining_bits→0` 时标记 `returned`；超过 `max_return_time_s` 未传完则该 POI 超时（`covered` 但非 `effective`）。
  - **何时调用** `progress_key_return` 由 `uavlab.paper1.runner.return_scheduling.should_attempt_key_return` 决定：`fast_upload_mode=policy` 时，目标 POI **空间覆盖完成**后每步可尝试（struct 公平比较）；`fast_upload_mode=fixed` 时按 `env.fixed_send_ratio` **确定性占空**门控（如 0.2 → 每 5 步 1 次），返航阶段（`goal_id is None`）不门控以便清空 backlog。耦合轴 periodic/event 使用 fixed；full 使用 policy FSM。
  - `env.effective` 与 `returned` 同步，供慢环/耦合包中的 `CompletionStatus.effective` 使用。
- **字段**：`R_task`（主键）、`effective_ratio`（别名）。

实现为 **per-POI 分片回传**（`uavlab.paper1.sim.return_queue`），与式 (10) 的传输时间语义一致，但不再对 backlog 总和做二元 bulk 判定。

## 3. 覆盖后失败率 \(R_{fail|cov}\)（式 59）

\[
R_{fail|cov}=\frac{\sum_i c_i(1-r_i)}{\sum_i c_i+\varepsilon}
\]

- **操作化**：`|covered \ returned| / |covered|`（\(\varepsilon=10^{-9}\)）。
- **字段**：`R_fail_given_cov`、`R_fail_cov`。
- **自检**：`R_task_check_product ≈ R_cov * (1 - R_fail_given_cov)`（`metrics()` 中输出）。

## 4. 关键数据平均回传时延 \(T_{ret}\)（式 61）

\[
T_{ret}=\frac{\sum_{i=1}^{N} c_i r_i\,(t_i^{ret}-t_i^{cov})}{\sum_{i=1}^{N} c_i r_i + \varepsilon}
\]

- **操作化**：
  - \(t_i^{cov}\)：POI \(i\) **首次**满足覆盖完成（`c_i=1`）的仿真墙钟时刻（`env.poi_cov_time_s[i]`）。
  - \(t_i^{ret}\)：POI \(i\) **首次**满足有效回传（`r_i=1`）的时刻（`env.poi_return_time_s[i]`）；与式 (12) 的 `return_success` 判定一致。
  - 仅对 \(e_i=c_i r_i=1\) 的 POI 计入分子/分母；无有效回传时 `T_ret_s = NaN`。
  - 当前为**批量回传**：一次 `try_return_key` 成功时，同一批 pending POI 共享同一 \(t_i^{ret}\)，但各 POI 的 \(t_i^{cov}\) 不同，故 \(t_i^{ret}-t_i^{cov}\) 仍按 POI 区分。
- **与式 (10) 区别**：式 (10) 的 \(T_i^{ret}\) 是**瞬时链路可行性**下的传输时间估计；式 (61) 的 \(T_{ret}\) 是**覆盖完成到回传完成**的墙钟滞后（越小越好）。
- **字段**：`T_ret_s`（主键）、`mean_key_return_delay_s`（别名）。

## 5. 链路退化恢复时延

从**链路突变**到**稳定恢复 INS 任务推进**的墙钟时间（秒）。

| 阶段 | 判据 | 配置键 |
|------|------|--------|
| 突变起点 | `loss_p` 由 `< link_drop_loss_p` 升至 `≥ link_drop_loss_p`（上升沿） | `comm.data_link.weak_loss_p` → `slow_loop.link_drop_loss_p`（默认 **0.50**，弱通信/失联区 §5） |
| 快环响应（记录用） | 突变后首次 `comm_mode ∈ {Srec, Ssafe, Stx}` | FSM 输出 |
| 恢复确认 | `loss_p ≤ link_loss_recover` 且 `comm_mode == Sins` 连续 **K** 步 | `comm.data_link.max_loss_p` → `link_loss_recover`（默认 **0.20**），`K=5` |
| 时延 | 突变起点时刻 → 恢复确认时刻 | `dt = 1/step_hz` |

- **字段**：`link_recovery_latency_s`（回合内事件均值，无事件为 `NaN`）、`link_recovery_event_count`。
- **说明**：多事件回合取算术平均；与论文“代表性一次退化—恢复”一致时可改为中位数或分位数（需在 sweep 后处理）。

## 6. 禁飞区暴露时间 \(T_{nf}\)（式 62）

\[
T_{nf}=\sum_t \mathbf{1}(p_t\in Z^{nf})\,\Delta t
\]

- **操作化**：每步 `in_nofly(pos)` 为真则 `nofly_dwell_s += dt`；进入次数记 `nofly_entry_count`。
- **字段**：`T_nf_s`（主键）、`nofly_dwell_s`（别名）。

## Sweep 汇总列（`summary.csv`）

除 legacy 的 `coverage_mean` / `effective_mean` 外，论文六指标在 `summary.csv` 中的均值/标准差列：

- `R_cov_mean`, `R_cov_std`
- `R_task_mean`, `R_task_std`
- `R_fail_given_cov_mean`, `R_fail_given_cov_std`
- `T_ret_s_mean`, `T_ret_s_std`
- `link_recovery_latency_s_mean`, `link_recovery_latency_s_std`
- `T_nf_s_mean`, `T_nf_s_std`

`link_recovery_latency_s` 与无有效回传回合的 `T_ret_s` 为 `NaN` 时，在 sweep 聚合中跳过（不参与均值）。
