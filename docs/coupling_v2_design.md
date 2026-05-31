# 耦合机制比较实验修改建议（coupling_axis_v2）

> **仓库实现状态（2026-05）**：本设计已在 coupling_v2 分支落地。配置见 `configs/experiments/paper1/system/coupling_*.yaml`；主 sweep：`configs/sweeps/paper1_axis_grid_coupling_v2.yaml`；设计背景见 [`coupling_axis_impl_and_first_test_issues.md`](coupling_axis_impl_and_first_test_issues.md)。

## 1. 当前问题判断

本轮 coupling first_test 的主要问题不是系统不稳定，而是三档耦合机制的对照强度不足。

当前实验中，`periodic_goal`、`event_driven_goal`、`full_coupling` 在覆盖率上差异很小，甚至 C2 中 `periodic_goal` 略高。这并不说明 Full Coupling 无效，而是说明当前实验设置下三者行为过于接近：

```text
periodic_goal 仍在高频周期重规划；
event_driven_goal 与 full_coupling 慢环触发机制过于相似；
fixed upload 过强，使 full 的快环 FSM 优势难以体现；
场景通信压力不足，覆盖后回传几乎全部成功。
```

因此，下一步修改目标不是继续调慢环目标函数权重，而是：

```text
拉大三档耦合机制的行为差异；
削弱 periodic_goal 的滚动纠偏能力；
让 full_coupling 的快环自适应与全量反馈真正发挥作用；
使用更能反映耦合机制价值的指标进行评估。
```

---

## 2. 三档耦合机制应体现的差异

建议重新明确三档耦合机制的功能边界。

| 档位 | 核心含义 | 慢环触发 | 快环能力 | 快→慢反馈 |
|---|---|---|---|---|
| PeriodicGoal | 低交互周期目标下发 | 低频周期更新 | 固定执行 / 固定上传 | 无任务事件反馈 |
| EventDrivenGoal | 事件驱动目标更新 | 周期 + 关键事件 | 固定上传，无完整 FSM | completion + link + safety |
| FullCoupling | 完整快慢环闭环 | hybrid + 事件分级 | FSM 自适应 | completion + link + safety + backlog + mode |

核心比较逻辑应为：

```text
PeriodicGoal：验证没有执行反馈时系统表现；
EventDrivenGoal：验证慢环事件反馈是否有价值；
FullCoupling：验证快环自适应 + 全量反馈是否进一步提升任务质量。
```

---

## 3. PeriodicGoal 修改建议

### 3.1 当前问题

当前 `periodic_goal` 虽然关闭快→慢事件反馈，但在 `periodic + repeat + Ts=40` 下仍然持续重规划，导致其并不是真正的弱耦合基线。

表现为：

```text
periodic_goal 仍然能高频滚动纠偏；
与 event/full 的轨迹和 POI 集合高度重合；
削弱了 FullCoupling 的可观测优势。
```

### 3.2 修改方向

建议将 `periodic_goal` 调整为真正的低交互基线。

推荐主方案：

```text
replan_trigger_policy = periodic
slow_interval_steps = 200 或 400
periodic_replan_scope = future_only 或 repeat_low_freq
enable_event_feedback = false
enable_fast_mode_switch = false
fast_upload_mode = fixed
fixed_send_ratio = 0.1 或 0.2
```

可选极端弱基线：

```text
periodic_replan_scope = init_only
```

### 3.3 预期效果

修改后，`periodic_goal` 应表现为：

```text
重规划次数显著低于 event/full；
无法根据覆盖完成、链路变化、安全事件及时修正；
在 C2 或回传压力场景中任务有效完成质量下降。
```

---

## 4. EventDrivenGoal 修改建议

### 4.1 当前问题

当前 `event_driven_goal` 与 `full_coupling` 在慢环侧过于相似，二者均使用 event replan，导致差异主要只剩 fixed upload 与 FSM 的区别。

### 4.2 修改方向

`event_driven_goal` 应突出“慢环接收事件反馈，但快环不具备完整自适应执行”的特点。

建议配置方向：

```text
enable_event_feedback = true
enable_fast_mode_switch = false
allow_mode_switching = false
fast_upload_mode = fixed
fixed_send_ratio = 0.1 或 0.2
replan_trigger_policy = hybrid
send_completion = true
send_link_stats = summary 或 full
send_safety_events = true
send_backlog = false
send_mode = false
```

### 4.3 事件处理规则

建议保留事件分级，但 EventDrivenGoal 不使用 backlog/mode 做细粒度执行闭环：

```text
Info：记录，不重规划；
Warning：下周期重规划；
Critical：立即重规划或返航。
```

### 4.4 预期效果

EventDrivenGoal 应相对 PeriodicGoal 体现：

```text
能更及时响应覆盖完成、安全事件和持续链路退化；
但由于快环无完整 FSM，自适应回传和局部恢复能力有限。
```

---

## 5. FullCoupling 修改建议

### 5.1 当前问题

当前 `full_coupling` 存在轻微 goal 振荡和 episode 方差，说明完整耦合仍需要继承 struct 轴 P0 稳定机制。

具体表现：

```text
C2 ep0 出现 A→B→A goal 振荡；
个别 episode 覆盖率明显下跌；
full 使用纯 event replan，未充分继承 hybrid + cooldown + goal lock。
```

### 5.2 修改方向

FullCoupling 不应追求更多重规划，而应追求更稳定、更有效的重规划。

建议配置方向：

```text
enable_event_feedback = true
enable_fast_mode_switch = true
allow_mode_switching = true
fast_upload_mode = policy
replan_trigger_policy = hybrid
send_completion = true
send_link_stats = full
send_safety_events = true
send_backlog = true
send_mode = true
event_levels = enabled
replan_cooldown_s = enabled
goal_lock = enabled
anti_thrashing = enabled
```

### 5.3 稳定性规则

必须保留以下机制：

```text
Warning 事件只触发下周期重规划；
Critical 事件才允许立即 interrupt；
普通周期更新不替换当前目标；
新目标收益不足时不切换；
近期失败/刚切换过的目标短期降权；
BACK / SAFE 状态下禁止重复触发同类 interrupt replan。
```

### 5.4 预期效果

FullCoupling 应体现：

```text
更短关键数据平均回传时延；
更低覆盖后失效率；
更快链路恢复；
更低或受控的禁飞区逗留时间；
重规划次数适中，不出现 goal 振荡。
```

---

## 6. fixed upload 强度修改建议

### 6.1 当前问题

当前 `periodic_goal` 和 `event_driven_goal` 使用：

```text
fixed_send_ratio = 0.5
```

该设置可能过强，使固定上传已经足以完成大部分数据回传，从而掩盖 FullCoupling 中 `S_tx/S_rec` 自适应回传的优势。

### 6.2 修改方向

建议将 fixed upload 削弱为：

```text
fixed_send_ratio = 0.1 或 0.2
```

或者采用：

```text
fixed upload 只能周期性尝试；
不根据链路状态自适应选择回传时机；
不使用 backlog/mode 反馈。
```

### 6.3 预期效果

削弱后，FullCoupling 的优势应体现在：

```text
pending-return 更少；
关键数据平均回传时延更短；
覆盖后失效率更低；
链路退化后恢复更快。
```

---

## 7. 场景压力修改建议

### 7.1 当前问题

当前 C1/C2 的覆盖率差距很小，C2 没有充分形成通信压力。三档均在约 57%–59% 覆盖平台竞争，差异接近噪声。

### 7.2 修改方向

建议增加或强化一个专门服务耦合轴的压力场景：

```text
C2-hard + G2/M2
```

强化方向：

```text
增加局部通信阴影强度；
增大任务点与强链路区错位；
提高关键数据大小 S_i；
降低可用带宽；
缩短 T_i_max；
收紧能量预算；
让部分 POI 覆盖容易但回传困难。
```

### 7.3 预期效果

耦合机制差异应在以下情形中被放大：

```text
覆盖后不一定能立即回传；
固定上传容易错过好链路窗口；
完整 FSM 能更快完成机会式回传；
事件反馈可避免继续深入弱链路区。
```

---

## 8. 评价指标修改建议

耦合轴不应只看 `R_cov`。`R_cov` 反映物理覆盖，但不能充分体现耦合机制对回传和恢复的价值。

建议主指标组合：

| 指标 | 作用 |
|---|---|
| R_task | 最终有效完成质量 |
| R_cov | 物理覆盖能力 |
| R_fail_given_cov | 覆盖后是否转化为有效完成 |
| T_ret | 关键数据平均回传时延 |
| link_recovery_latency | 链路退化恢复能力 |
| T_nf | 局部安全撤离能力 |
| mode_time_ratio | INS/TX/REC/SAFE/BACK 时间占比 |
| replan_reason_dist | 重规划是否由有效事件触发 |
| goal_switch_count | 目标切换稳定性 |
| remaining_energy | 资源消耗情况 |

解释优先级建议：

```text
先看 R_task；
再用 R_cov 与 R_fail_given_cov 分解失败来源；
再用 T_ret、link_recovery_latency、mode_time_ratio 解释耦合机制差异；
最后用 T_nf、remaining_energy 评估安全与资源代价。
```

---

## 9. 诊断日志增强建议

为判断耦合机制是否真正生效，建议补充记录：

```text
replan_reason
event_level
goal_switch_reason
mode_at_replan
pending_return_count
pending_return_age
T_ret_per_poi
link_recovery_latency
mode_duration: INS/TX/REC/SAFE/BACK
fixed_upload_attempt_count
policy_upload_attempt_count
upload_success_count
backlog_size
```

特别需要检查：

```text
periodic 是否仍高频重规划；
event 的 replan 是否主要由有效事件触发；
full 是否出现 goal 振荡；
full 的 S_tx/S_rec 是否真正减少 T_ret；
full 是否减少 pending-return 时间。
```

---

## 10. 建议的新一轮实验配置

### 10.1 coupling_v2 主实验

建议设置：

```text
cases = C2 + G2/M2
seeds >= 5
episodes >= 5
```

三档建议：

```text
PeriodicGoal:
  slow_interval_steps = 200 或 400
  fixed_send_ratio = 0.1 或 0.2
  enable_event_feedback = false
  enable_fast_mode_switch = false

EventDrivenGoal:
  replan_trigger_policy = hybrid
  fixed_send_ratio = 0.1 或 0.2
  enable_event_feedback = true
  enable_fast_mode_switch = false
  send_completion/link/safety = true
  send_backlog/mode = false

FullCoupling:
  replan_trigger_policy = hybrid
  fast_upload_mode = policy
  enable_event_feedback = true
  enable_fast_mode_switch = true
  send_completion/link/safety/backlog/mode = true
  cooldown + goal_lock + anti_thrashing = true
```

### 10.2 消融补充

建议额外做两个小消融，用于说明差异来源：

| 消融 | 目的 |
|---|---|
| fixed_send_ratio = 0.5 vs 0.2 | 证明 fixed upload 过强会掩盖 full 优势 |
| slow_interval_steps = 40 vs 200 | 证明高频 periodic replan 会削弱耦合轴差异 |

---

## 11. 预期结果

修改后不一定要求 FullCoupling 在 `R_cov` 上绝对最高，但应满足：

```text
FullCoupling 的 R_task 更高或更稳定；
FullCoupling 的 R_fail_given_cov 更低；
FullCoupling 的 T_ret 更短；
FullCoupling 的 link_recovery_latency 更短；
FullCoupling 的 goal 振荡不高于 event；
FullCoupling 的 T_nf 受控；
PeriodicGoal 的 replan 次数显著低于 event/full。
```

更合理的预期排序：

| 指标 | 预期趋势 |
|---|---|
| R_cov | full ≈ event ≥ periodic，或三者接近 |
| R_task | full ≥ event ≥ periodic |
| T_ret | full < event < periodic |
| R_fail_given_cov | full < event < periodic |
| link_recovery_latency | full < event < periodic |
| goal_switch_count | full 适中，不应爆炸 |
| remaining_energy | periodic 可能最高，但不代表最优 |

---

## 12. 论文表述建议

不要把耦合机制写成“FullCoupling 必然覆盖率最高”。建议改为：

```text
耦合机制主要影响任务执行过程中的信息闭环效率。
在通信压力较弱或周期重规划过于频繁时，不同耦合机制在覆盖率上的差异可能有限；
而在通信退化、回传延迟和局部风险更明显的场景中，
FullCoupling 应通过快环模式切换与慢环反馈重规划，
降低覆盖后失效率，缩短关键数据回传时延，
并提升任务有效完成质量。
```

核心论证从：

```text
FullCoupling 覆盖率最高
```

调整为：

```text
FullCoupling 在任务有效完成、数据回传时效、链路恢复和安全响应方面更优。
```

---

## 13. 给 Cursor 的修改摘要

```text
本轮 coupling_v2 修改目标是拉大三档耦合机制差异，而不是继续微调 planner 权重。

必须修改：
1. periodic_goal 不再使用 Ts=40 高频 repeat replan；改为 slow_interval_steps=200/400 或 init_only/future_only。
2. fixed_send_ratio 从 0.5 降到 0.1 或 0.2。
3. event_driven_goal 使用 hybrid replan，但不启用快环 FSM，不发送 backlog/mode。
4. full_coupling 使用 hybrid replan + event_levels + cooldown + goal_lock + anti_thrashing。
5. full_coupling 启用 policy upload、FSM 自适应、backlog/mode 全量反馈。
6. 增加 C2-hard 压力场景：提高回传压力，使 fixed upload 与 policy upload 差异可见。
7. 指标增加或重点输出：T_ret、pending_return_count、mode_time_ratio、replan_reason_dist、goal_switch_count。
8. 对比报告不要只看 R_cov，应以 R_task、R_fail_given_cov、T_ret、link_recovery_latency 作为耦合机制主要证据。
```

---

## 14. 一句话总结

当前 coupling first_test 说明系统健康，但三档对照不够锋利。下一步应削弱 PeriodicGoal 的高频滚动纠偏能力，降低 fixed upload 强度，让 FullCoupling 继承 hybrid + P0 稳定机制，并在更强回传压力场景下用 `R_task / T_ret / R_fail_given_cov / link_recovery_latency` 证明完整耦合的信息闭环价值。
