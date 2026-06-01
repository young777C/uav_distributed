# uav-distributed

无人机巡检任务级仿真平台（Paper1Lite）：在通信退化条件下，对比 **慢环路径规划** 与 **快环局部控制 / 数据回传** 的不同架构（CDSL / WCDL / FDLC）及建模、耦合消融。

- **远程仓库**：<https://github.com/young777C/uav_distributed>
- **详细运行说明**：[`docs/RUN_AND_VISUALIZE.md`](docs/RUN_AND_VISUALIZE.md)
- **实验设计**：[`docs/experimental_design_simulation_evaluation.md`](docs/experimental_design_simulation_evaluation.md)
- **文档索引**：[`docs/README.md`](docs/README.md)

---

## 1. 仓库定位

| 层级 | 说明 |
|------|------|
| **main** | 已通过单元测试与验收 checklist 的**稳定基线**；任何对比 / 消融论文结论必须能在此分支复现 |
| **feature 分支** | 进行中的代码或配置改动；**不得**直接 push 到 main |
| **experiment 分支** | 某次 sweep 的**结果快照**（summary、关键图）；与 main 代码版本一一对应，便于回溯 |

本仓库**不是**高保真飞控仿真（无 PX4/Gazebo 耦合）；目标是可复现、可测试的任务级决策对比。

---

## 2. 开发共识（必读）

以下约定适用于所有参与者（含 AI Agent），目的是：**实验可对比、代码不互相污染、main 始终可信**。

### 2.1 分支策略

```
main                         ← 仅合入已验证改动
 │
 ├── feat/<描述>              ← 功能 / 配置 / 测试开发
 ├── fix/<描述>               ← 紧急修复
 ├── ablation/<轴>-<变体>     ← 单轴消融（如 modelling-no-comm-constraint）
 └── exp/<sweep-tag>         ← 实验结果存档（可选，见 §2.4）
```

| 分支类型 | 命名示例 | 用途 |
|----------|----------|------|
| 功能 | `feat/struct-c2-degrade-ladder` | 改 `planner.py`、struct YAML、测试等 |
| 修复 | `fix/return-phase-trigger` | 单点 bugfix |
| 消融 | `ablation/coupling-event-only` | 只动耦合轴，不混入 struct 改动 |
| 实验 | `exp/validate-struct-baseline` | 完整 sweep 跑通后的结果分支 |

**规则：**

1. 从 **最新 main** 拉分支，开发完成后通过 **Pull Request** 合入 main（禁止 force-push main）。
2. **一个 PR 只做一件事**：要么 struct 轴、要么 coupling 轴、要么 modelling 轴；避免多轴混改导致无法归因。
3. 未通过 §2.3 验收 checklist 的改动**不得**合入 main，可留在 feature 分支继续迭代。
4. 实验分支从**已合入 main 的 commit** 拉出；分支名与 sweep `--tag` 保持一致，便于对照 `runs/debug/` 目录。

### 2.2 合入 main 的门槛

合入前必须在本地（或 CI）全部通过：

```bash
# 1. 单元测试（必做）
export PYTHONPATH=src
python -m pytest tests/ -q

# 2. Smoke：单 case、单 seed、单 episode（改动了 loops / planner / FSM 时必做）
python3 -m uavlab.paper1.runner.run_vis \
  --config configs/experiments/paper1/cases/c2_g2_m2.yaml \
  --system configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml \
  --episodes 1 --seed 0 --slow_interval_steps 40

# 3. 涉及 struct 轴对比时：跑 diag sweep 并检查 summary（见 §2.3）
```

GitHub Actions（`.github/workflows/ci.yml`）在 push / PR 时自动跑 `pytest`；**CI 绿是必要条件，不是充分条件**——struct / coupling 相关改动还需人工核对 sweep 指标。

### 2.3 验收 checklist（struct 轴示例）

完整 struct 验收可参考 [`cursor_document_analysis_for_goal_trace.md`](cursor_document_analysis_for_goal_trace.md) 末尾流程；核心通过标准：

| 步骤 | 通过标准 |
|------|----------|
| `pytest tests/` | 全部通过 |
| struct sweep | 生成 `summary.csv` 与三架构 `metrics.jsonl` |
| 低数据量基线 | `data_chunk_bits≈8000` 时 **FDLC ≥ WCDL ≥ CDSL**（`R_task` 等主指标） |
| 返航 | `returned_home_rate=1.0`，`term_return_rate=1.0` |
| 可视化 | mosaic 图生成，轨迹与 summary 数值一致 |

各实验轴有独立 sweep plan（`configs/sweeps/`），合入前至少跑对应轴的 **smoke / diag** 配置，不要一次跑全网格 unless 发 paper 定稿。

### 2.4 什么提交进 Git、什么留本地

| 内容 | 是否入库 | 说明 |
|------|----------|------|
| `src/`、`configs/`、`tests/`、`scripts/` | **是** | 代码与配置是唯一事实来源 |
| `docs/` | **是** | 设计、契约、运行说明 |
| `runs/**` | **默认否** | `.gitignore` 已忽略；体量大、易冲突 |
| `summary.csv`、关键 mosaic PNG | **可选** | 仅在 `exp/*` 分支按需 `-f` 添加，用于结果存档 |
| 完整 `metrics.jsonl`、wandb | **否** | 本地或网盘；仓库只保留可复现 config + 汇总 |

若需在 experiment 分支追踪结果，推荐只 track：

```text
runs/debug/<tag>/summary.csv
runs/debug/<tag>/plot_trajectory/mosaic_*.png
```

### 2.5 配置与实验轴约定

实验三维正交（详见 `configs/experiments/paper1/README.md`）：

| 轴 | 预设键 | 典型 system YAML |
|----|--------|------------------|
| **结构 struct** | `experiment.paper1.struct` → `cdsl` / `wcdl` / `fdlc` | `system/struct_*.yaml` |
| **建模 modelling** | `experiment.paper1.modeling` | `system/modelling_*.yaml` |
| **耦合 coupling** | `experiment.paper1.coupling` | `system/coupling_*.yaml` |

**共识：**

- 场景几何以 `configs/taskA/scene/*.yaml` 为权威；case YAML 只 `extends` comm profile + 指定 scene。
- 架构差异必须通过 **`struct_*.yaml` + `StructAxisProfile`** 体现，禁止在 case 里硬编码覆盖 struct 语义。
- `presets.py` 合并后仍会 re-apply 实验 YAML 中的 `paper1_loops`；新增 preset 时需确认不抹平三架构差异。
- 改 `planner.py` / `fast_loop.py` / `slow_loop.py` 时，同步更新或新增 `tests/test_paper1_*.py`。

### 2.6 提交信息

使用简短英文或中文均可，格式建议：

```text
feat(struct): add degrade ladder for empty candidate window
fix(fast-loop): control_link_lost timer triggers S_back
test(planner): CDSL subset WCDL subset FDLC feasible set
docs: add struct baseline validation commands
```

一次 commit 聚焦一个逻辑变更；实验结果 commit 与代码 commit **分开**。

### 2.7 AI Agent 协作约定

在 Cursor 等 Agent 中继续开发时：

1. 先读 **main 上最新代码** 与相关 `docs/`，勿假设未合入分支已生效。
2. 默认在 `feat/*` 分支工作；**未经用户明确要求不要 commit / push**。
3. 完成改动后给出：改了什么、跑了哪些命令、指标是否达 checklist。
4. 长对话记录可导出到 `docs/dev-log/`，但**以代码与 config 为准**，对话仅作背景参考。

---

## 3. 快速开始

### 3.1 环境

```bash
cd uav-distributed
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip pytest "ortools>=9.10"
export PYTHONPATH=src
python -m pytest tests/ -q
```

### 3.2 单次可视化运行

```bash
PYTHONPATH=src python3 -m uavlab.paper1.runner.run_vis \
  --config configs/experiments/paper1/cases/c2_g2_m2.yaml \
  --system configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml \
  --episodes 1 --seed 0 --slow_interval_steps 40
```

### 3.3 Struct 轴 sweep（诊断网格）

```bash
python3 scripts/sweep.py \
  --plan configs/sweeps/paper1_axis_grid_struct_diag_m2.yaml \
  --tag validate_struct_baseline
```

### 3.4 轨迹 mosaic

```bash
SWEEP=runs/debug/struct_goal_trace/validate_struct_baseline
python3 scripts/plot_trajectory.py --sweep_root "$SWEEP" --mosaic_publish
```

更多命令见 [`docs/RUN_AND_VISUALIZE.md`](docs/RUN_AND_VISUALIZE.md)。

---

## 4. 目录结构（简）

```text
configs/
  base.yaml                 # 全局默认
  experiments/paper1/       # system（轴）× cases（C×G×M）
  sweeps/                   # sweep plan
src/uavlab/
  paper1/loops/             # 慢环 / 快环 / FSM
  paper1/contracts/         # StructAxisProfile、合同配置
  tasks/taskA/              # 环境、训练入口（与 Paper1Lite 对齐）
scripts/                    # sweep、plot、analyze
tests/                      # 单元测试（合入门槛）
docs/                       # 设计与运行文档
runs/                       # 本地实验输出（默认不入库）
```

---

## 5. 常见问题

**Q：三种架构结果几乎一样？**  
A：检查三份 `struct_*.yaml` 的 `modeling` / `coupling` / `slow_loop.replan_trigger_policy` 是否被 preset 覆盖成相同；见 dev-log 中 struct 轴根因分析。

**Q：FDLC 在 C2 场景覆盖率为 0？**  
A：检查慢环空窗降级、`StructAxisProfile` 硬 mask 阈值；跑 `tests/test_paper1_struct_planner.py`。

**Q：main 与本地实验对不上？**  
A：确认 `resolved_config.yaml`、seed、sweep tag 与 git commit hash 一致；结果分支应注明对应 main commit。

---

## 6. 相关链接

| 资源 | 路径 |
|------|------|
| Baseline 切换与训练 | [`docs/experiments_baseline_guide.md`](docs/experiments_baseline_guide.md) |
| 快慢环设计 | [`docs/fast_slow_loop_design.md`](docs/fast_slow_loop_design.md) |
| 指标操作化 | [`docs/paper1_metrics_operationalization.md`](docs/paper1_metrics_operationalization.md) |
| Struct 开发记录（归档） | [`cursor_document_analysis_for_goal_trace.md`](cursor_document_analysis_for_goal_trace.md) |

---

**维护原则：`main` 慢而稳，实验在分支上快跑快存档；指标达标再合入，避免未验证代码污染对比基线。**
