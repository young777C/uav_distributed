# Struct 轴关键数据量压力实验 — 问题梳理

> **实验背景**：在 C2 / `g2_cluster_m2` 场景下，通过调整 `comm.data_chunk_bits`（每 POI 关键数据量）观察 CDSL / WCDL / FDLC 三架构差异。  
> **Sweep 配置**：`configs/sweeps/paper1_axis_grid_struct_diag_m2.yaml`  
> **通信 profile**：`configs/comm_profiles/c2.yaml`  
> **主要 run 标签**（`runs/debug/struct_goal_trace/`）：
>
> | 标签 | `data_chunk_bits` | 约等于 |
> |------|-------------------|--------|
> | `fixed_send_ratio_0.2_key_reverce` | 8 000 | ~1 KB |
> | `fixed_send_ratio_0.2_key_05MB` | 4 000 000 | 4 Mbit ≈ 0.5 MB |
> | `fixed_send_ratio_0.2_key_1MB` | 8 000 000 | 8 Mbit ≈ 1 MB |
> | `fixed_send_ratio_0.2_key_2MB` | 8 000 000（同 1MB 档） | 8 Mbit ≈ 1 MB |
>
> **生成日期**：2026-05-22

---

## 1. 核心结论（Executive Summary）

| 维度 | 结论 |
|------|------|
| 小数据（8k bit） | 回传无约束，`R_fail\|cov=0`；结构差异来自 **探索策略**，FDLC 覆盖最高（~59%），**不能**体现通信退化 |
| 中压（0.5 MB / 4 Mbit） | 回传阈值开始生效；**仅 FDLC** 出现高方差与 `R_fail\|cov≈18%`；CDSL/WCDL 仍 100% 返航、零失败 |
| 高压（1 MB / 8 Mbit） | 全体 `R_cov` 下降；WCDL > CDSL > FDLC；FDLC **近乎失效**（R_task≈2.5%，R_fail≈67%） |
| 结构轴排序 | 随数据量 **翻转**：8k 时 FDLC > WCDL > CDSL；1 MB 时 **WCDL > CDSL ≫ FDLC** |
| 根因 | 回传物理阈值（10s / ℓ≤0.20）+ **批量 backlog 判定** + FDLC 快环 FSM / 全 f2s 正反馈 |

**一句话**：默认 8k bit 的 struct sweep **掩盖**了「覆盖后回传」矛盾；加大 `data_chunk_bits` 后暴露出 **FDLC 在大 payload 下的失稳与通信死锁**，而 CDSL/WCDL 靠「少覆盖、多等链路」维持 `R_fail=0`。

---

## 2. 定量结果摘要（C2 / g2_m2，seed=0，5 episodes）

### 2.1 三档 × 三架构

| 数据量 | 架构 | R_cov | R_task | R_fail\|cov | term_return | 备注 |
|--------|------|-------|--------|-------------|-------------|------|
| **8k** | CDSL | 38.0% | 38.0% | 0 | 100% | 稳定 |
| **8k** | WCDL | 46.5% | 46.5% | 0 | 100% | 稳定 |
| **8k** | FDLC | **58.7%** | **58.7%** | 0 | 100% | 探索最积极 |
| **0.5 MB** | CDSL | 38.0% | 38.0% | 0 | 100% | 与 8k 几乎相同 |
| **0.5 MB** | WCDL | 46.5% | 46.5% | 0 | 100% | 与 8k 几乎相同 |
| **0.5 MB** | FDLC | 25.8%±14% | 22.5%±16% | **17.7%** | **20%** | ep0 尚可，ep1–4 崩盘 |
| **1 MB** | CDSL | 28.5% | 28.5% | 0 | 100% | R_cov 下降 |
| **1 MB** | WCDL | **31.3%** | **31.3%** | 0 | 100% | 三架构最优 |
| **1 MB** | FDLC | **2.5%** | **2.5%** | **66.7%** | **0%** | 全 time_limit |

### 2.2 辅助指标（1 MB 档典型）

| 架构 | T_ret (s) | T_nf (s) | pending_max (bit) | link_recovery (s) |
|------|-----------|----------|-------------------|-------------------|
| CDSL | ~7 | 74–146 | 8M（单 POI） | ~67 |
| WCDL | ~10 | 71–159 | 8M | ~95 |
| FDLC | ~1.2 | **0** | 16M+ | 无有效恢复 |

---

## 3. 配置与单位问题

### 3.1 `data_chunk_bits` 单位为 bit，非 byte

- 配置路径：`comm.data_chunk_bits` → `Paper1SimConfig.key_bits_per_poi` → `Poi.key_bits`
- **2 MB（十进制）** 应写 `16_000_000.0`，不是 `2_000_000.0`
- **0.5 MB** 对应 `4_000_000.0` bit；**1 MB** 对应 `8_000_000.0` bit

### 3.2 修改 `config.py` fallback 不生效

`src/uavlab/paper1/sim/config.py` 第 124 行：

```python
key_bits_per_poi = float(comm.get("data_chunk_bits", 16_000_000.0))
```

仅当 YAML **未**设置 `data_chunk_bits` 时才用 fallback。实际生效值以 `resolved_config.json` / `metrics.jsonl` 中 `pending_return_max_bits` 为准。

### 3.3 run 标签与真实 bit 数不一致

部分目录名含 `2MB`，但 `resolved_config.json` 中为 `8_000_000`（1 MB 档）。分析时 **以 resolved 配置为准**，不以 tag 字面含义为准。

---

## 4. 回传机制导致的问题

### 4.1 批量回传判定（bulk return）

`Paper1Env.try_return_key()` 对 **backlog 总和** 调用 `return_success()`，成功则 **一次性** 标记全部 pending POI 为 `returned`：

- CDSL/WCDL：常在 backlog ≈ 1×`key_bits` 时尝试 → 易在链路窗口成功
- FDLC：多 POI 积压时 pending 可达 **20M–28M bit**（5–7 POI）→ 单次 `T_ret` 超 10s 上限 → 长期传不出去

**问题**：架构差异被 **积压形态** 放大，而非仅单 POI 大小。

### 4.2 回传可行性阈值过严

`return_success`（`key_data_return.py`）要求：

1. `loss_p ≤ data_link.max_loss_p`（默认 **0.20**）
2. \(T^{ret} = \tau + B / b^{eff} \leq\) `max_return_time_s`（默认 **10 s**）

其中 \(b^{eff} = b(1-\ell)^2\)，\(b \approx 1\,\text{Mbps}\cdot(1-\ell)\)（**损耗被计算两次**）。

| 单次回传 B | ℓ=0.15 | ℓ=0.20 | ℓ=0.23 |
|------------|--------|--------|--------|
| 8k | ✓ ~0.1s | ✓ | ✓ |
| 4M（0.5MB/POI） | ✓ ~5.6s | ✓ ~6.3s | ✗ |
| 8M（1MB/POI） | **✗ ~11.2s** | ✗ | ✗ |
| 16M（2MB/POI） | ✗ ~22s | ✗ | ✗ |

**问题**：

- 8k 时通信对任务 **完全透明**，struct 轴无法考察「通信退化下的回传」
- 1 MB 时 **单 POI 在典型好链路下已不可传**，但 episode 2000s 内 CDSL/WCDL 仍可通过 **等待 ℓ 极低窗口** 清零 backlog
- 阈值与带宽模型未与 `data_chunk_bits` 标定，导致「工作点」难以预测

### 4.3 瞬时成功/失败模型

回传无分片、无断点续传、无跨 step 传输进度；每 step 仅做二元可行性判定。

**问题**：大 payload 下「等好链路再传」的策略在 CDSL/WCDL 可行，但 FDLC 因持续扩覆盖 + FSM 占步，无法在时限内凑够有效传输窗口。

---

## 5. 三架构行为差异与机制成因

### 5.1 CDSL（单环中心式）

| 配置要点 | `use_comm_in_fast: false`；慢环 `periodic` only；无 f2s link/backlog |
|----------|------------------------------------------------------------------------|
| 8k / 0.5MB | R_cov ~38%，100% 返航，`mode_time_ratio: 100% Sins` |
| 1 MB | R_cov 降至 ~28%，仍 R_fail=0 |

**机制**：快环不参与 TX/REC FSM；覆盖节奏慢；backlog 常为单 POI 量级；episode 末回 GCS 前在好链路窗口清空 pending。

**暴露问题**：R_fail=0 **不表示**通信压力小，而是 **主动少覆盖** 以保证已覆盖必回传。

### 5.2 WCDL（弱双环）

| 配置要点 | 慢环 hybrid + f2s；快环 **仍** `use_comm_in_fast: false`（Stx 仅 ~2–3%） |
|----------|-----------------------------------------------------------------------------|
| 三档 | R_cov 始终 **高于 CDSL**；三档 R_fail=0；1 MB 时最优（~31%） |

**机制**：慢环通信评分选 POI + 简单快环执行；无大量 S_rec/S_tx 占步；在高压下 **探索与回传平衡最好**。

**暴露问题**：当前 Paper1Lite 下 **弱双环优于全双环**，与「FDLC 最优」论文叙事相反。

### 5.3 FDLC（全双环分布式）

| 配置要点 | `use_comm_in_fast: true`；全 f2s；hybrid + link warning + goal_lock |
|----------|-----------------------------------------------------------------------|
| 8k | R_cov 最高（~59%），探索优势 |
| 0.5 MB | 高方差：ep0 ~48% vs ep1–4 10%–33%；80% time_limit |
| 1 MB | R_cov ~2.5%，R_fail ~67%，T_nf=0，0% 返航终止 |

**机制（失稳链）**：

1. 快环 FSM 进入 S_tx / S_rec / Sback，占步 40%–75%
2. 多 POI 覆盖 → backlog 累加 → bulk 回传失败
3. `on_post_cover_upload_stuck` / link warning → 慢环 replan → goal 振荡（diag：backlog 16M、covered=3、effective=1，goal 100↔105 高频切换）
4. 传不出去仍继续触发 warning → **正反馈死锁**
5. episode 跑满 time_limit，已覆盖 POI 无法标记 returned → `R_fail|cov` 飙升

**暴露问题**：全耦合快环在大 payload 下 **负收益**；ep0 偶然成功会 **严重高估** FDLC（mosaic 若只看 ep0 会误导）。

---

## 6. 指标与实验设计问题

### 6.1 主指标在 8k 下失效

- `R_fail_given_cov` 全为 0 → 无法区分「覆盖后回传失败」
- `R_task = R_cov` → 结构对比实为 **覆盖率对比**
- 此前 struct 主表若用默认 8k，**高估 FDLC、低估通信矛盾**

### 6.2 `R_fail=0` 的误导性（CDSL/WCDL）

CDSL/WCDL 在 0.5–1 MB 下仍 R_fail=0，是因为：

- 降低扩覆盖速度（R_cov 从 38%/46% 降至 28%/31%）
- 在 2000s 任务时长内等待 ℓ≤0.15 的短窗口完成回传
- 不表示「通信压力已解决」

**建议同报**：`R_cov`、`R_task`、`R_fail|cov`、`term_return_rate`、`pending_return_max_bits`、`T_ret`。

### 6.3 结构轴与耦合轴纠缠

三架构 YAML 同时改变：

| 维度 | CDSL | WCDL | FDLC |
|------|------|------|------|
| 快环 comm FSM | 关 | 关 | **开** |
| f2s link/backlog | 弱/无 | 有 | **全** |
| 慢环 hybrid 事件 | periodic | hybrid（无 link drop） | **full hybrid** |

**问题**：数据量放大后的 FDLC 劣势，无法单独归因于「双环分布式结构」还是「快环 FSM + 批量回传 + 全 f2s」组合。

### 6.4 统计与可视化局限

- 当前 diag sweep：`seeds=[0]`，5 episodes；FDLC 0.5 MB 档 **高方差** 需多 seed 确认
- `plot_trajectory --episode N` 仍可能输出 `mosaic_*__ep0.png` 文件名 → 易误读失败局
- mosaic 默认 ep0 对 FDLC 0.5 MB **严重乐观**

### 6.5 `replan_count` 口径

- **metrics**：慢环 `should_replan` 总次数
- **diag**：仅 goal/sequence **变化** 的 slow_replan 事件

对比时需统一口径，避免与 periodic Ts 混淆。

---

## 7. 数据量「相变」与工作点

```
数据量 ↑
─────────────────────────────────────────────────────────────
8k          │ 通信透明     │ FDLC 探索优势      │ struct 轴失真
4M (0.5MB)  │ 开始约束     │ FDLC 失稳（高方差） │ WCDL/CDSL 仍稳
8M (1MB)    │ 强约束       │ FDLC 崩溃          │ 全体降 R_cov
16M (2MB)   │ 单 POI 不可传 │ 预期三架构均极差   │ （若跑需单独验证）
─────────────────────────────────────────────────────────────
```

**推荐 struct 主实验工作点**：**4 Mbit（0.5 MB）** — 能拉开 FDLC 的 `R_fail|cov`，且 CDSL/WCDL 仍可完成基本闭环；**1 MB** 可作为高压 stress case。

---

## 8. 论文叙事风险

| 表述 | 是否成立 | 说明 |
|------|----------|------|
| 「FDLC 任务完成率最高」 | **8k：是；≥0.5MB：否** | 1 MB 时 FDLC R_task≈2.5% |
| 「结构轴随耦合增强单调变好」 | **否** | 排序随数据量翻转 |
| 「通信退化下 FDLC 更优」 | **否** | 高压下 WCDL 最优 |
| 「R_fail\|cov 体现架构差异」 | **部分** | 主要惩罚 FDLC；CDSL/WCDL 靠降覆盖保 0 |

**更稳妥表述**：

> 在关键数据量较小时，结构差异主要体现在覆盖探索策略；当回传成为瓶颈时，弱双环（WCDL）在有效完成率与稳定性上优于全双环（FDLC），单环（CDSL）最保守。全耦合快环在大 backlog 下易陷入上传失败与重规划振荡。

---

## 9. 与 coupling_v2 实验的交叉问题

（参见 `docs/coupling_axis_impl_and_first_test_issues.md`、`docs/coupling_v2_design.md`）

| 问题 | 说明 |
|------|------|
| C2 基线 event ≈ full | 通信压力不足时 coupling 轴拉不开 |
| periodic Ts 设置 | `experiment.sweep.slow_interval_steps` 覆盖 sweep 默认 Ts |
| 大数据 + coupling | struct 与 coupling 实验应 **统一标定** `data_chunk_bits` 与 `max_return_time_s` |

---

## 10. 改进建议（按优先级）

### P0 — 实验标定

1. 在 `configs/comm_profiles/c2.yaml`（或 dedicated profile）**明确** `data_chunk_bits`，文档注明 bit 与 MB 换算
2. struct 主表改用 **4 Mbit**；1 MB 作 appendix stress
3. 跑 **≥5 seeds** 确认 FDLC 0.5 MB 方差

### P1 — 机制修复（FDLC）

1. backlog > 阈值时 **禁止扩覆盖** / 强制 return phase
2. 评估 **per-POI 回传判定** 替代 bulk sum
3. 抑制 `on_post_cover_upload_stuck` 导致的 goal 100↔105 振荡
4. FDLC ablation：`use_comm_in_fast=false` 分离「结构 vs 快环 FSM」

### P2 — 指标与报告

1. summary 增加 `pending_return_max_bits_mean`、`term_return_rate`（已有字段需入表）
2. 分 episode 报告 FDLC，避免只报 ep0
3. 统一 `replan_count` 口径说明

### P3 — 回传模型（若需更真实）

1. 复核 \(b^{eff}\) 双重损耗是否 intentional
2. 按 payload 标定 `max_return_time_s` 或引入分片传输
3. 考虑 `terminate_on_returned_home` 与 time_limit 对 R_fail 的影响

---

## 11. 相关文件索引

| 用途 | 路径 |
|------|------|
| 数据量配置 | `configs/comm_profiles/c2.yaml` → `comm.data_chunk_bits` |
| 解析到仿真 | `src/uavlab/paper1/sim/config.py` → `key_bits_per_poi` |
| 批量回传 | `src/uavlab/paper1/sim/env.py` → `try_return_key()` |
| 回传判定 | `src/uavlab/related_models/key_data_return.py` |
| 三架构 YAML | `configs/experiments/paper1/system/struct_*.yaml` |
| Struct sweep | `configs/sweeps/paper1_axis_grid_struct_diag_m2.yaml` |
| 8k 结果 | `runs/debug/struct_goal_trace/fixed_send_ratio_0.2_key_reverce/` |
| 0.5 MB 结果 | `runs/debug/struct_goal_trace/fixed_send_ratio_0.2_key_05MB/` |
| 1 MB 结果 | `runs/debug/struct_goal_trace/fixed_send_ratio_0.2_key_1MB/` |
| 指标操作化 | `docs/paper1_metrics_operationalization.md` |
| Struct refactor 问题 | `docs/struct_a_diff_v2_experiment_issues.md` |
| Coupling 轴问题 | `docs/coupling_axis_impl_and_first_test_issues.md` |

---

## 12. 变更记录

| 日期 | 说明 |
|------|------|
| 2026-05-22 | 初版：汇总 8k / 0.5MB / 1MB 三档 struct 对比实验暴露的配置、机制、指标与叙事问题 |
