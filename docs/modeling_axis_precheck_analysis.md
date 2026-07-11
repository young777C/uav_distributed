# Modeling 轴 Precheck 实验分析

> **控制变量**：`struct=fdlc`，`coupling=full_coupling`，仅变 `modeling`（§4.2.2 信息利用消融）  
> **主锚点**：`c2_g2_m2`（C2 阴影区 + G2 聚簇 + M2 高冲突）  
> **当前结论依据**：`runs/sweeps/modeling_axis_b1_E210`（B1 + A 层 + 3 seeds）  
> **历史对照**：`runs/sweeps/modling_axis_full_20260610_precheck`（首版 precheck，存在配置混杂）

---

## 0. 配置与代码修改摘要（B 层 + A 层）

在首版 precheck 之后，为对齐论文 §4.2.2、修复控制变量、并让能量「可观测」，仓库做了如下修改（**不影响 struct / coupling 轴**）。

### 0.1 信息利用语义门控（modeling 三档）

| 档位 | 慢环 | 快环 | `energy_constraint` |
|------|------|------|---------------------|
| `comm_aware_decision` | 通信适宜度 + 控制链路掩码 | `S_tx`/`S_rec` 链路自适应 | `none` |
| `energy_aware_decision` | 能量掩码 + tour 预算 | 仅 `S_back`；无链路自适应回传 | `full` |
| `comm_energy_aware_decision` | 通信 + per-POI 能量掩码 | 链路自适应 + `S_back` | **`return_home`**（B1） |

实现要点：

- `apply_modeling_semantics_to_struct_profile()`：`use_comm_in_slow=false` 时关闭 struct 层通信硬掩码与 `beta_ret`；`use_energy_in_slow=false` 时 `energy_plan_margin_frac=0`。
- `energy_aware` 关闭 `return_policy` backlog 门控（无通信驱动的慢环返航压力）。
- 慢环空窗 fallback：无能量约束时用 `nearest_uncovered_poi`，避免 comm 档隐式能量过滤。

相关文件：`src/uavlab/paper1/contracts/struct_profile.py`、`contract_config.py`、`loops/slow/planner.py`、`loops/slow/slow_loop.py`、`configs/experiments/paper1/system/modelling_*.yaml`、`tests/test_modeling_axis_control_vars.py`。

### 0.2 B0：modeling YAML `extends struct_full_dual_loop_distributed.yaml`

三份 `modelling_*.yaml` 改为 **diff on struct FDLC**（与 coupling 轴 `extends struct` 一致），sweep 组合为 `case + modelling_*.yaml` 即可带上：

- `return_policy.enable_backlog_gates: true`
- `env.fast_upload_mode: policy`、完整 f2s、hybrid replan

**struct / coupling 轴未改**：`struct_full_dual_loop_distributed.yaml` 仍为 `energy_constraint: full`；仅 modeling 轴 `comm_energy` YAML 覆盖为 `return_home`。

### 0.3 B1：联合档 `full` → `return_home`

论文式 (26) per-POI 能量掩码 \(M_i^E\) 保留；去掉 OR-Tools 全程 tour 总预算（式 34），缓解「能量只掐覆盖率、无增益」：

```yaml
# modelling_comm_energy_aware_decision.yaml
modeling:
  energy_constraint: return_home   # energy_hard=true, energy_budget=false
```

### 0.4 A 层：modeling 轴专用能量压力 case（E210）

| 项 | 说明 |
|----|------|
| 路径 | `configs/experiments/paper1/cases/modeling_energy_stress/` |
| 隔离 | **不在** `cases/*.yaml` glob 内 → struct sweep 仍为 8 case |
| Tier E210 | `E_0_Wh: 210`（原 263.2）、`P_fly_W: 440`（原 400） |
| Sweep | `configs/sweeps/paper1_axis_grid_modelling_energy_stress.yaml` |

### 0.5 未实施（留待全网格后按需）

- **B2**：`energy_objective` 软惩罚（MILP 偏好短距/低能耗路径）— B1 后 joint 已接近 comm，暂不启用。
- **preset** 中 `comm_energy` 仍为 `full`；struct/coupling 由 struct YAML 覆盖，modeling 由 modelling YAML 覆盖为 `return_home`。

---

## 1. 实验 sweep 一览

| Sweep | 标签 | 配置状态 | 规模 |
|-------|------|----------|------|
| `modling_axis_full_20260610_precheck` | 首版 | 无 B0/B1/A；无 struct extends | 2 case × 3 modeling × 1 seed × 5 ep |
| **`modeling_axis_b1_E210`** | **当前验收** | B0 + B1 + baseline/E210 × 3 seeds | 4 case × 3 modeling × 3 seeds × 5 ep |

Plan：`runs/sweeps/modeling_axis_b1_E210/plan.yaml`  
Runner：`uavlab.paper1.runner.run`  
数据来源：`runs/sweeps/modeling_axis_b1_E210/summary.csv`

---

## 2. 首版 precheck 结果（历史对照）

> 存在 §5.1 控制变量问题，**勿与 struct 轴或当前 B1 结果直接比绝对值**；仅作「改前」参照。

| Case | Modeling | R_task | 终止特征 |
|------|----------|--------|----------|
| c1_g2_m2 | comm | 0.690 | 100% OOB |
| c1_g2_m2 | comm+energy | 0.654 | 100% 返航 |
| c1_g2_m2 | energy | 0.270 | 100% OOB |
| c2_g2_m2 | comm | **0.710** | 100% OOB |
| c2_g2_m2 | comm+energy | **0.560**（σ=0.24） | 80% 返航 / 20% OOB |
| c2_g2_m2 | energy | 0.270 | 100% OOB |

首版排序：`comm (69–71%) > comm+energy (56–65%) >> energy (27%)`；C2 联合档 ep 分布 `0.67, 0.13, 0.68, 0.66, 0.66`（单 ep 崩溃至 13%）。

---

## 3. B1 + E210 主结果（`modeling_axis_b1_E210`）

### 3.1 C2 `c2_g2_m2`（3 seed 均值，主表）

| Case | Modeling | R_task | R_fail\|cov | T_ret (s) | T_nf (s) | remaining_energy | returned_home | energy_rate | oob_rate |
|------|----------|--------|-------------|-----------|----------|------------------|---------------|-------------|----------|
| baseline | comm | **0.712** | 0 | 12.3 | 20.0 | 0.439 | **0%** | 0% | **93%** |
| baseline | comm+energy | **0.666** | 0 | 12.2 | 16.9 | 0.202 | **100%** | 0% | 0% |
| baseline | energy | 0.270 | 0.036 | 13.2 | 10.5 | 0.637 | 0% | 0% | 100% |
| E210 | comm | **0.710** | 0 | 12.6 | 17.5 | **0.270** | 0% | 0% | 100% |
| E210 | comm+energy | **0.575** | 0 | 13.1 | 49.3 | **0** | 100% | **100%** | 0% |
| E210 | energy | **0.256** | 0 | 13.5 | 10.4 | 0 | 100% | **100%** | 0% |

**C2 baseline 排序**：

```text
comm (71%)  >  comm+energy (67%)  >>  energy (27%)
```

**C2 E210 排序**：

```text
comm (71%)  >  comm+energy (58%)  >>  energy (26%)
```

### 3.2 C1 `c1_g2_m2`（补充，不宜作主 pooled）

| Case | comm | comm+energy | energy |
|------|------|-------------|--------|
| baseline R_task（3 seed 均值） | ~0.697 | ~0.653 | 0.270 |
| E210 R_task（3 seed 均值） | ~0.639 | ~0.552 | 0.270 |

C1 E210 下 comm seed2 降至 0.588，方差较大；与 struct/coupling 轴一致，**主文用 C2**。

### 3.3 B1 相对首版 precheck 的改善（C2）

| 指标 | 首版 C2 comm+energy | B1 C2 comm+energy |
|------|---------------------|-------------------|
| R_task | 0.560（σ=0.24） | **0.666**（σ≈0.02） |
| vs comm 差距 | ~15 pp | **~4.5 pp** |
| 单 ep 崩溃 | 有（13%） | **无** |

B1（去掉 tour 总预算、保留 \(M_i^E\)）将联合档从「过强硬掐」拉回 **≥ comm×85%（~60.4%）** 的验收线，且稳定性明显改善。

### 3.4 A 层 E210 能量压力效应（C2）

| 对比 | comm | comm+energy | energy |
|------|------|-------------|--------|
| baseline → E210 `R_task` | 0.712 → 0.710（几乎不变） | 0.666 → **0.575**（−9 pp） | 0.270 → 0.256 |
| `remaining_energy` | 0.44 → **0.27** | 0.20 → **0** | 0.64 → **0** |
| 终止 | 仍 **OOB 主导** | **100% 电量耗尽** + 返航 | **100% 耗尽** + 返航（不再 OOB 早停） |

要点：

- **comm** 仍无视能量语义 → E210 下覆盖率不降，但余电下降、仍不返航 → 对照「无能量意识的高覆盖 + 不安全收束」。
- **comm+energy** 在 E210 下 `energy_rate=100%`，电量用尽后返航，`R_task` 降至 57% → 能量风险与联合收束可写进论文。
- **energy** 在 C2 E210 从 OOB 早停（~5200 步）变为满程耗尽返航（~9570 步），能量维度行为更可解释。

### 3.5 与 struct 轴 FDLC 对照（C2 baseline）

| 来源 | comm+energy R_task |
|------|-------------------|
| 首版 precheck（混杂配置） | 0.560 |
| **B1 modeling 轴** | **0.666** |
| struct 轴 FDLC（`full` 约束） | ~0.57–0.66 |

B1 下 modeling 轴联合档与 struct 轴 FDLC 同量级，但 **comm-only（71%）仍高于 struct FDLC**，因 comm 消融故意不使用能量约束。

---

## 4. 与论文叙事的对齐（基于 B1 + E210）

### 4.1 一致 / 可写

| 论点 | 证据（C2） |
|------|------------|
| 通信信息必要 | energy 档恒 ~27%，远低于 comm/joint |
| 有效完成 = 覆盖 + 回传 | comm/joint `R_fail\|cov=0`，`R_cov=R_task` |
| 联合 = 通信 + 能量底线 | baseline joint `returned_home=100%`，`remaining_energy` 低于 comm |
| 能量是约束与收束机制 | E210 下 joint/energy `energy_rate=100%`；comm 仍高覆盖但不返航 |
| B1 缓解「能量纯掐覆盖率」 | joint 0.666 vs 首版 0.560 |
| 场景依赖 | C2 上 comm vs energy 差距大于 C1 |

### 4.2 需谨慎 / 改写

| 原叙事 | 当前数据 |
|--------|----------|
| 「联合档最优」 | **不成立**；comm 仍略高于 joint（C2 baseline 约 4.5 pp） |
| 「联合在能量压力下更稳」 | E210 下 joint 掉幅大于 comm（−9 pp vs 持平） |
| comm 体现能量风险 | **不应写**；用「同场景 joint 耗尽而 comm 仍飞」作对照 |
| comm-only = 方法上界 | comm 高 `R_task` 伴随 **~100% OOB**、0% 返航 |

**推荐表述**：通信感知决策提升有效完成率；联合决策在能量底线与返航收束下略牺牲覆盖率；能量收紧时该 trade-off 更明显（E210 子表）。

---

## 5. 仍存在的问题与后续

| 问题 | 说明 | 建议 |
|------|------|------|
| comm 终止方式 | C2 ~93–100% OOB，高 `R_task` 含「越界前尽量覆盖」 | 主表同时报 `returned_home_rate`、`remaining_energy` |
| E210 对 joint 偏狠 | 57% + 100% 耗尽 | 可试 E220/E230 或略降 `P_fly` 增幅 |
| energy baseline ~27% | 无通信消融预期行为 | 不作「能量增益」主证据 |
| C1 方差 | E210 comm seed2 偏低 | 全网格 C2-only 或 C1 分表 |

---

## 6. 全网格验收 checklist

- [x] modeling YAML `extends struct_full_dual_loop_distributed.yaml`
- [x] B1：`comm_energy` → `energy_constraint: return_home`
- [x] A 层：E210 case 仅 modeling sweep 引用
- [x] 3 seeds precheck（`modeling_axis_b1_E210`）
- [x] C2 baseline：joint ≥ comm×85%，joint `returned_home` > comm
- [ ] 全网格：`paper1_axis_grid_modelling.yaml` 建议 **仅 C2**（可选 + C2_E210）
- [ ] 5–10 ep × 3 seeds 生产 sweep
- [ ] 暂不开 B2，除非全网格 joint 仍明显低于 comm

---

## 7. 一句话结论

**首版 precheck** 强支持「通信必要」，并暴露配置混杂与 `full` tour 预算过紧。**B1 + E210（`modeling_axis_b1_E210`）** 在修复 FDLC 控制变量后，C2 上形成可接受的 **comm (71%) > joint (67%) >> energy (27%)** 阶梯，且联合档具备 **100% 安全返航** 与 E210 下 **可观测的能量耗尽**；**不宜写联合全面优于通信单档**，宜写 **trade-off + 能量压力子实验**。可进入 **C2 全网格 modeling sweep**。

---

## 8. 关联路径

| 类型 | 路径 |
|------|------|
| 验收 sweep | `runs/sweeps/modeling_axis_b1_E210/` |
| 首版 sweep | `runs/sweeps/modling_axis_full_20260610_precheck/` |
| Modeling sweep（baseline） | `configs/sweeps/paper1_axis_grid_modelling.yaml` |
| Modeling + 能量压力 | `configs/sweeps/paper1_axis_grid_modelling_energy_stress.yaml` |
| 系统 YAML | `configs/experiments/paper1/system/modelling_*.yaml` |
| 压力 case | `configs/experiments/paper1/cases/modeling_energy_stress/` |
| 测试 | `tests/test_modeling_axis_control_vars.py`、`tests/test_modeling_energy_stress_cases.py` |
| Struct 对照 | `runs/sweeps/struct_axis_full_20260606/summary.csv` |
| 跨轴一致性 | `docs/experiment_consistency_check.md` |
