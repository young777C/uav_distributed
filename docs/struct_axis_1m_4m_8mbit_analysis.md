# Struct 轴：1 / 4 / 8 Mbit 数据量 sweep 分析报告

> **场景**：C2 / `g2_cluster_m2`（120 POI）  
> **Sweep**：`configs/sweeps/paper1_axis_grid_struct_diag_m2.yaml`  
> **通信 profile**：`configs/comm_profiles/c2.yaml`（覆盖 `data_chunk_bits`）  
> **Run 目录**：`runs/debug/struct_goal_trace/{1Mbit,4Mbit,8Mbit}/`  
> **条件**：seed=0，5 episodes，Ts=40  
> **生成日期**：2026-06-01

---

## 1. 实验设定

### 1.1 文件夹与 `data_chunk_bits` 对应关系

文件夹名对应 `comm.data_chunk_bits`（单位 **bit**，非 byte）：

| 文件夹 | `data_chunk_bits` | 约等于 | 单 POI 回传时间（ℓ≈0.15，~850 kbps 有效带宽） |
|--------|-------------------|--------|-----------------------------------------------|
| `1Mbit` | 1 000 000 | ~125 KB | ~1.3 s ✓ |
| `4Mbit` | 4 000 000 | ~0.5 MB | ~5.6 s ✓ |
| `8Mbit` | 8 000 000 | ~1 MB | ~**11.2 s ✗**（超过默认 `max_return_time_s=10`） |

实际生效值以各 run 的 `metrics.jsonl` 中 `pending_return_max_bits` 及 comm 配置为准。

### 1.2 对比架构

| 架构 | system YAML | 论文符号 |
|------|-------------|----------|
| CDSL | `struct_centralized_single_loop.yaml` | 中心式单环 |
| WCDL | `struct_decoupled_dual_loop.yaml` | 弱耦合双环 |
| FDLC | `struct_full_dual_loop_distributed.yaml` | 全双环分布式 |

三架构 **慢环 modeling 相同**（`comm_energy_aware_decision`）；差异来自 `StructAxisProfile`、耦合 f2s、快环 FSM。

---

## 2. 核心结论（Executive Summary）

| 维度 | 结论 |
|------|------|
| **1 Mbit** | 回传无约束，`R_fail\|cov=0`；差异来自 **探索策略**；**FDLC > WCDL > CDSL** |
| **4 Mbit** | CDSL/WCDL 与 1 Mbit **逐 episode 完全相同**；**仅 FDLC** 失稳（R_task≈22%，R_fail≈19%） |
| **8 Mbit** | 全体 R_cov 下降；**WCDL > CDSL ≫ FDLC**；FDLC R_task≈0.8%，60% 能量耗尽 |
| **结构排序** | 随数据量 **翻转**：1 Mbit 时 FDLC 最优，8 Mbit 时 FDLC 近乎失效 |
| **根因** | 批量 backlog 回传 + 回传阈值未标定 + FDLC 快环 FSM / 全 f2s 正反馈 |

**一句话**：1 Mbit 下 struct 轴测的是覆盖探索，不是回传协作；4 Mbit 起 FDLC 因 backlog 积压与 replan 振荡单独恶化；8 Mbit 下单 POI 在典型链路下已物理不可传，FDLC 双重崩溃。

---

## 3. 定量结果

### 3.1 三档 × 三架构（summary.csv 汇总）

| 数据量 | 架构 | R_cov | R_task | R_fail\|cov | term_return | 备注 |
|--------|------|-------|--------|-------------|-------------|------|
| **1 Mbit** | CDSL | 38.0% | 38.0% | 0 | 100% | 稳定 |
| **1 Mbit** | WCDL | 46.5% | 46.5% | 0 | 100% | 稳定 |
| **1 Mbit** | FDLC | **56.7%** | **56.7%** | 0 | 100% | 探索最积极 |
| **4 Mbit** | CDSL | 38.0% | 38.0% | 0 | 100% | 与 1 Mbit 相同 |
| **4 Mbit** | WCDL | 46.5% | 46.5% | 0 | 100% | 与 1 Mbit 相同 |
| **4 Mbit** | FDLC | 26.2%±15% | 22.2%±16% | **19.2%** | 100% | 高方差 |
| **8 Mbit** | CDSL | 28.5% | 28.5% | 0 | 100% | R_cov 下降 |
| **8 Mbit** | WCDL | **31.3%** | **31.3%** | 0 | 100% | 三架构最优 |
| **8 Mbit** | FDLC | **2.5%** | **0.8%** | **66.7%** | 0% | 60% 能量耗尽 / 40% 超时 |

### 3.2 FDLC 分 episode 明细（4 Mbit，说明 mosaic 偏差）

| Episode | R_cov | R_task | pending_max (bit) | 说明 |
|---------|-------|--------|-------------------|------|
| ep0 | 17.5% | 12.5% | 24 M | **mosaic 默认展示此局** |
| ep1 | 12.5% | 10.0% | 12 M | |
| ep2 | **48.3%** | **48.3%** | 8 M | 唯一健康局 |
| ep3 | 17.5% | 13.3% | 20 M | |
| ep4 | 35.0% | 26.7% | 40 M | 10 POI 积压 |

### 3.3 辅助指标

| 数据量 | 架构 | T_ret (s) | T_nf (s) | pending_max | link_recovery |
|--------|------|-----------|----------|-------------|---------------|
| 1 Mbit | CDSL | ~0.9 | ~175 | 1 M | ~3.6 次 |
| 1 Mbit | WCDL | ~0.9 | ~216 | 1 M | ~5.2 次 |
| 1 Mbit | FDLC | ~2.7 | **~365** | 2–3 M | ~5.2 次 |
| 8 Mbit | CDSL | ~6–11 | ~74–146 | 8 M | ~4.6 次 |
| 8 Mbit | WCDL | ~9–10 | ~71–159 | 8 M | ~4.6 次 |
| 8 Mbit | FDLC | ~0.8–3.0 | **0** | 16 M | 0 次 |

8 Mbit FDLC：`replan_count` 250–375，`goal_switch_count` 最高 123；5 ep 全部 R_cov=2.5%（仅 3 POI）。

---

## 4. 三架构实现差异（代码层）

差异分布在 **StructAxisProfile（慢环可行域）**、**耦合反馈（f2s）**、**快环 FSM** 三层；三份 `struct_*.yaml` 的 `modeling` 字段相同。

### 4.1 慢环：`StructAxisProfile`

实现：`src/uavlab/paper1/contracts/struct_profile.py`

| 维度 | CDSL | WCDL | FDLC |
|------|------|------|------|
| `control_cmd_min_ratio` | 0.95（最严） | 0.85 | 0.70（最松） |
| `control_loss_exposure_max` | 0.0 | 0.05 | 0.10 |
| `use_data_return_hard_mask` | **true** | false | false |
| `energy_plan_margin_frac` | 0.15 | 0.08 | **0.02** |
| `candidate_degrade_max_level` | 1 | 2 | 3 |

- FDLC 慢环可行 POI 集合最大 → 1 Mbit 下覆盖最高。
- FDLC 能量规划裕度仅 2% → 8 Mbit 下悬停 REC/TX 易触发能量耗尽。

### 4.2 耦合：f2s 与慢环触发

配置：`configs/experiments/paper1/system/struct_*.yaml`

| 维度 | CDSL | WCDL | FDLC |
|------|------|------|------|
| 慢环触发 | **仅 periodic** | hybrid | hybrid + link/control warning |
| f2s link/backlog/mode/safety | **无/弱** | 有 | **全量** |
| `on_post_cover_upload_stuck` | 无 | 无 | **有** |
| `use_comm_in_fast` | false | false | **true** |
| `use_energy_in_fast` | false | false | **true** |

### 4.3 快环：FSM

实现：`src/uavlab/paper1/loops/fast/fsm.py`

- **CDSL / WCDL**：`use_comm_in_fast=false`；覆盖后简单 TX，无 REC/BACK 占步。
- **FDLC**：覆盖后进完整 TX/REC FSM；链路差时长时间悬停 REC；能量低时 BACK。

### 4.4 回传：批量判定

实现：`src/uavlab/paper1/sim/env.py` → `try_return_key()`

对 **backlog 总和** 调用 `return_success()`，成功则一次性标记全部 pending POI 为 `returned`；无分片、无跨 step 传输进度。

- CDSL/WCDL：慢节奏 → backlog ≈ 1 POI → 易成功。
- FDLC：快覆盖 → 多 POI 积压（20–40 Mbit）→ bulk 判定失败 → `R_fail|cov` 飙升。

回传阈值（`configs/base.yaml`）：`data_link.max_loss_p=0.20`，`max_return_time_s=10.0 s`。  
带宽模型：`link_proxy_at` 设 `bw=1M×(1-ℓ)`，`effective_bandwidth_bps` 再乘 `(1-ℓ)`（双重损耗）。

---

## 5. 分档行为解读

### 5.1 1 Mbit — 通信透明，差异来自探索

- 三架构 `R_fail|cov=0`，`R_task = R_cov`。
- FDLC 57% > WCDL 47% > CDSL 38%：慢环可行域最宽 + 快环不拖探索。
- 轨迹：FDLC 探到右上 POI 簇；CDSL/WCDL 偏保守。
- FDLC `T_nf≈365 s`（约为 CDSL 2 倍）：探索更积极，禁飞区暴露更多。
- FDLC `pending_max=2–3 Mbit`（2–3 POI 积压），bulk 仍可传完。

**本档测的是路径探索，不是通信退化下的回传协作。**

### 5.2 4 Mbit — FDLC 进入失稳区

- **CDSL/WCDL 与 1 Mbit 逐 episode 完全相同**（R_cov、T_nf、轨迹一致）→ 1→4 Mbit 对二者 **无行为影响**（始终单 POI 回传，`pending_max=4 Mbit`）。
- **仅 FDLC 恶化**：bulk 失败 + `on_post_cover_upload_stuck` → replan 振荡。
- mosaic 若只看 ep0（17.5%），会严重低估 FDLC（均值 22%）；ep2 可达 48.3%。

### 5.3 8 Mbit — 全体降覆盖，FDLC 双重崩溃

- CDSL/WCDL：R_cov 降至 29%/31%；`T_ret` 升至 6–11 s；仍 `R_fail=0`（单 POI 等待 ℓ 极低窗口）。
- FDLC：仅 3 POI 覆盖；`pending_max=16 Mbit` 整局传不出；`T_nf=0`（困在 GCS 附近 TX/REC）。
- 终止：60% `energy_depleted`（`use_energy_in_fast` + 悬停），40% `time_limit`。
- 单 POI 8 Mbit 在 ℓ=0.15 时 `T_ret≈11.2 s > 10 s`，物理上已不可传。

---

## 6. 暴露的问题

### P0 — 机制性缺陷

| # | 问题 | 说明 |
|---|------|------|
| 1 | **批量回传放大 FDLC 劣势** | CDSL/WCDL backlog≈1 POI；FDLC 多 POI 积压 → bulk 失败 |
| 2 | **回传阈值与 payload 未标定** | 8 Mbit 典型链路下单 POI 不可传；1/4 Mbit 对 CDSL/WCDL 等价 |
| 3 | **FDLC 正反馈死锁** | upload_stuck → warning replan → goal 振荡 → 继续覆盖 → backlog 更大 |
| 4 | **spatial 先于 effective 换目标** | `on_goal_spatial_complete=true`：覆盖即换目标，不等回传完成 |

### P1 — 实验与指标

| # | 问题 | 说明 |
|---|------|------|
| 5 | **`R_fail=0` 误导** | CDSL/WCDL 靠少覆盖维持；不代表通信压力小 |
| 6 | **mosaic ep0 偏差** | 4 Mbit FDLC ep0 远差于 ep2 |
| 7 | **结构轴与快环 FSM 纠缠** | 无法单独归因「双环结构」vs「快环上传策略」 |

### P2 — 安全

| # | 问题 | 说明 |
|---|------|------|
| 8 | **禁飞区约束偏弱** | 1 Mbit FDLC `T_nf≈365 s`；轨迹频繁擦边禁飞区 |

---

## 7. 修改建议（按优先级）

### P0 — 实验标定

1. **Struct 主实验工作点**：**4 Mbit**（能拉开 FDLC `R_fail|cov`，CDSL/WCDL 仍可闭环）；8 Mbit 作 appendix stress case。
2. **报告字段**：同报 `R_cov`、`R_task`、`R_fail|cov`、`pending_return_max_bits`、`term_return_rate`；FDLC 分 episode 或 ≥5 seeds。
3. **标定 `max_return_time_s`**：按 payload 缩放，例如 `max_return_time_s = 2 + 1.5 × (data_chunk_bits / 1e6)`。

### P1 — 机制修复（FDLC 失稳）

4. **Per-POI 回传判定**：改 `try_return_key()`，对每个 pending POI 独立 `return_success`。
5. **Backlog 门控扩覆盖**：backlog 超软/硬阈值时禁止新 POI / 强制 return phase。
6. **解耦 spatial / effective**：FDLC 仅 `on_goal_effective_complete` 触发换目标，或 spatial 后必须 TX 成功。
7. **抑制 upload_stuck 振荡**：stuck 时 return phase 而非 replan；或 warning 加 cooldown。
8. **FDLC ablation 分支**：`use_comm_in_fast=false` 或 per-POI 回传，分离结构 vs FSM 归因。

### P2 — 模型层

9. **复核带宽双重损耗**是否 intentional。
10. **分片或跨 step 传输进度**（大 payload 更真实）。
11. **禁飞硬约束加强**（慢环路径 mask 或 `T_nf` 阈值触发 SAFE）。

---

## 8. 论文叙事风险

| 表述 | 是否成立 | 说明 |
|------|----------|------|
| 「FDLC 任务完成率最高」 | **仅 1 Mbit** | 8 Mbit 时 FDLC R_task≈0.8% |
| 「通信压力下 FDLC 更优」 | **否** | 8 Mbit 下 WCDL 最优 |
| 「R_fail\|cov 体现架构差异」 | **部分** | 主要惩罚 FDLC；CDSL/WCDL 靠降覆盖保 0 |
| 「结构轴随耦合增强单调变好」 | **否** | 排序随数据量翻转 |

**更稳妥表述**：

> 关键数据量较小时，结构差异体现为覆盖探索策略（FDLC 慢环可行域最宽）；当回传成为瓶颈时，弱双环 WCDL 在有效完成率与稳定性上优于全双环 FDLC。FDLC 的快环 FSM、批量回传与全量 f2s 正反馈，在大 payload 下导致 backlog 积压、重规划振荡与能量耗尽。

---

## 9. 数据量「相变」示意

```
数据量 ↑
─────────────────────────────────────────────────────────────
1 Mbit   │ 通信透明     │ FDLC 探索优势       │ 测的是覆盖率
4 Mbit   │ 开始约束     │ 仅 FDLC 失稳        │ CDSL/WCDL 不变
8 Mbit   │ 强约束       │ FDLC 崩溃           │ WCDL > CDSL
─────────────────────────────────────────────────────────────
```

---

## 10. 相关文件索引

| 用途 | 路径 |
|------|------|
| Struct 轴 profile | `src/uavlab/paper1/contracts/struct_profile.py` |
| 三架构 YAML | `configs/experiments/paper1/system/struct_*.yaml` |
| 批量回传 | `src/uavlab/paper1/sim/env.py` → `try_return_key()` |
| 回传判定 | `src/uavlab/related_models/key_data_return.py` |
| 快环 FSM | `src/uavlab/paper1/loops/fast/fsm.py` |
| upload_stuck | `src/uavlab/paper1/loops/slow/slow_loop.py` |
| 数据量配置 | `configs/comm_profiles/c2.yaml` → `comm.data_chunk_bits` |
| Struct sweep | `configs/sweeps/paper1_axis_grid_struct_diag_m2.yaml` |
| 1 Mbit 结果 | `runs/debug/struct_goal_trace/1Mbit/` |
| 4 Mbit 结果 | `runs/debug/struct_goal_trace/4Mbit/` |
| 8 Mbit 结果 | `runs/debug/struct_goal_trace/8Mbit/` |
| 历史问题梳理 | `docs/struct_axis_key_data_pressure_issues.md` |
| 指标操作化 | `docs/paper1_metrics_operationalization.md` |

---

## 11. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-06-01 | 初版：汇总 1 / 4 / 8 Mbit 三档 struct 对比实验的分析、机制成因与修改建议 |
