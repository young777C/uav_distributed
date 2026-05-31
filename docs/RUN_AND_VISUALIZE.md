## Paper1Lite：运行与可视化说明

本文档说明如何在本仓库中运行 Paper1 轻量任务级仿真（Paper1Lite），并生成：
- sweep 扫描实验结果（`case × system × seed × Ts`）
- 汇总表 `summary.csv`
- POI 三色可视化（**已回传 / 已访问未回传 / 未访问**），可选叠加**障碍物圆**
- 返航信息（`dist_to_home_m` / `returned_home`）用于判断是否回到起点（GCS）
- 轨迹图（单 run / sweep 批量 / **按场景拼一张大图 mosaic**），可选通信场热力图

### 0. 运行前提

- **在仓库根目录执行命令**：`/home/yuhe/workspace/uav-comm-degradation`
- Python 需能找到 `src/` 下的包，所以命令都建议加：

```bash
PYTHONPATH=src
```

- **OR-Tools（必装，若走默认慢环 `heuristic`）**  
  默认 `configs/base.yaml` 与多数 `system/*.yaml` 使用 `slow_backend: heuristic`，慢环会在候选窗上调用 **CP-SAT**（`ortools.sat`）。若未安装，会出现：
  `ModuleNotFoundError: No module named 'ortools'` / `RuntimeError: OR-Tools is required...`。  
  **请在你实际用来跑 `python3 scripts/sweep.py` 的同一个环境里安装**（conda 环境不会自动用系统 `pip --user` 的包）：

```bash
# 任选其一（在已激活的 conda / venv 中执行）
python3 -m pip install "ortools>=9.10"
# 或：conda install -c conda-forge ortools
```

  也可使用仓库中的 `requirements-paper1.txt`：`python3 -m pip install -r requirements-paper1.txt`。

### 0.1 与当前代码对齐的配置要点（Paper1Lite）

下列项会影响仿真行为；sweep 的 `system/*.yaml` 会覆盖 `base` 中的 **`paper1_loops`** 片段（可选地仍可通过遗留键 `paper1_switchboard` 合并，见 `Paper1ContractConfig.from_cfg`）。

- **语义与信息开关（推荐）**：`paper1_loops.semantics` 下的 `structure`、`coupling_mechanism`、`use_comm_in_slow` 等（见 `configs/base.yaml` 注释）。
- **慢环建模（消融四开关）**：`paper1_loops.modeling` 推荐只维护  
  `comm_objective`、`comm_constraint`、`energy_objective`、`energy_constraint`（由 `Paper1ContractConfig.from_cfg` 解析为合同字段）。
- **慢环窗口（更新版流程）**：`paper1_loops.slow_loop` 中  
  `window_k`（基础尺度 \(K\)）、`prefetch_k_multiplier`（预取 \(|\widetilde W_t|=\min(\mathrm{mult}\cdot K,|U_t|)\)`，默认 3）、`horizon_h`、`path_samples`。
- **慢环后端**：仅支持 `slow_backend: heuristic`。`slow_policy` 为 `periodic_or_event_replan` 时在候选窗上调用 **OR-Tools** 短时域选序；`advance_on_poi_done` 时用窗口内最近邻等选点逻辑（见 `slow_loop.py`），二者与是否每步调用 CP-SAT 无简单一一对应。
- **快环上传模式**：`env.fast_upload_mode`：`policy` = 链路自适应 FSM（`Srec`/`Stx`）；`fixed` = 按 `env.fixed_send_ratio` 对**进入 `Stx`** 做占空比门控（消融固定上传强度）。

---

## 1) 单次运行（不 sweep）

### 1.1 使用默认 runner（只输出聚合指标）

`uavlab.paper1.runner.run` 会输出每个 episode 的聚合指标到 `metrics.jsonl`（含论文 §4.2.4 **六指标**字段；**不包含**逐 POI 的 `covered_ids`/`returned_ids` 列表，见 `run_vis`）。操作化说明见 [`docs/paper1_metrics_operationalization.md`](../../docs/paper1_metrics_operationalization.md)。

示例（跑 1 个 episode）：

```bash
PYTHONPATH=src python3 -m uavlab.paper1.runner.run \
  --config configs/experiments/paper1/cases/c1_g1.yaml \
  --episodes 1 \
  --seed 0 \
  --slow_interval_steps 40 \
  --run_dir /tmp/paper1lite_run \
  --metrics_jsonl /tmp/paper1lite_run/metrics.jsonl
```

输出目录 `/tmp/paper1lite_run/` 会包含：
- `resolved_config.json`：resolved 后的配置（用于追溯）
- `metrics.jsonl`：每个 episode 一行 JSON

### 1.2 使用可视化 runner（推荐：多输出 POI 状态）

`uavlab.paper1.runner.run_vis` 在 `metrics.jsonl` 里额外写出：
- `covered_ids`：访问到达（覆盖）POI 的 id 列表
- `returned_ids` / `effective_ids`：关键数据已回传 POI 的 id 列表（二者相同；`effective_ids` 为兼容字段）

示例：

```bash
PYTHONPATH=src python3 -m uavlab.paper1.runner.run_vis \
  --config configs/experiments/paper1/cases/c1_g1.yaml \
  --episodes 1 \
  --seed 0 \
  --slow_interval_steps 40 \
  --run_dir /tmp/paper1lite_runvis \
  --metrics_jsonl /tmp/paper1lite_runvis/metrics.jsonl
```

---

## 2) Sweep 扫描实验（case × system）

### 2.1 编辑 sweep plan

sweep plan 在 `configs/sweeps/*.yaml`。Paper1 **§6.3 三轴**已各有一份计划（均为 `case × system` 笛卡尔积）：

| 轴 | 计划文件 | `axis_grid.systems` 典型 glob |
|----|-----------|--------------------------------|
| §6.3.1 结构 | `configs/sweeps/paper1_axis_grid_struct.yaml` | `configs/experiments/paper1/system/struct_*.yaml` |
| §6.3.2 建模 | `configs/sweeps/paper1_axis_grid_modelling.yaml` | `configs/experiments/paper1/system/modelling_*.yaml` |
| §6.3.3 耦合 | `configs/sweeps/paper1_axis_grid_coupling.yaml` | `configs/experiments/paper1/system/coupling_*.yaml` |

关键字段：
- `axis_grid.cases`：环境 case 列表（通常来自 `configs/experiments/paper1/cases/*.yaml`）
- `axis_grid.systems`：系统轴 YAML（通常来自 `configs/experiments/paper1/system/*.yaml` 的 glob）
- `episodes`：每个组合重复跑多少个 episode
- `seeds`：随机种子列表
- `slow_interval_steps_list`：慢环更新间隔（step 为单位）
- `runner_module`：选择 runner；若 plan 中省略该字段，`scripts/sweep.py` 默认使用 **`uavlab.paper1.runner.run`**（Paper1Lite 指标 runner；已修正旧默认 `uavlab.tasks.paper1.runner.run`）。

正式统计时可提高 `episodes` 与 `seeds` 长度；当前仓库内默认多为 **烟测规模**（例如 `episodes: 1`）。

推荐如果后续要画 POI 三色图，则设置：
- **`runner_module: uavlab.paper1.runner.run_vis`**

若只关心指标、希望 sweep 更快，可改为 **`uavlab.paper1.runner.run`**（不写 `covered_ids` / `effective_ids` / 默认轨迹）。

### 2.2 运行 sweep

在仓库根目录执行：

```bash
PYTHONPATH=src python3 scripts/sweep.py \
  --plan configs/sweeps/paper1_axis_grid_struct.yaml \
  --tag demo_struct
```

输出目录默认在：
- `runs/sweeps/<tag>/`

例如：
- `runs/sweeps/demo_struct/`

每个组合会生成一个叶子目录：
- `runs/sweeps/<tag>/<case>__<system>/Ts40/seed0/`

该目录里会有：
- `combined_config.yaml`：sweep 自动生成的组合配置（`extends: [case, system]`）
- `resolved_config.json`：runner 解析后的配置快照
- `metrics.jsonl`：每个 episode 一行结果

---

## 3) 汇总结果：生成 summary.csv

sweep 跑完后，`scripts/sweep.py` 会在 sweep 根目录自动生成：
- `runs/sweeps/<tag>/summary.csv`

如果你只想对已经跑完的结果**重新生成汇总**（不重跑实验）：

```bash
PYTHONPATH=src python3 scripts/sweep.py \
  --plan configs/sweeps/paper1_axis_grid_struct.yaml \
  --tag demo_struct \
  --aggregate_only
```

### 3.1 summary.csv 主要字段

- `exp_name`：`<case>__<system>`
- `episodes`：该 run_dir 中统计到的 episode 行数
- **论文六指标（推荐）**：`R_cov_*`、`R_task_*`、`R_fail_given_cov_*`、`T_ret_s_*`、`link_recovery_latency_s_*`、`T_nf_s_*`
- **Legacy 别名**：`coverage_mean / coverage_std`、`effective_mean / effective_std`（与 `R_cov` / `R_task` 数值相同）
- `remaining_energy_mean / remaining_energy_std`
- `steps_mean`
- `dist_to_home_mean_m / dist_to_home_std_m`
- `returned_home_rate`：0~1，表示回到起点的比例

### 3.2 关于 episode 与 step

- **episode**：一次完整回合，从 `reset()` 到 `done()`
- **step**：episode 内的离散时间步，每一次 `env.step_fast(...)` 计为 1 step
- `steps_mean` 通常接近配置中的 `env.episode_steps`

---

## 4) POI 三色可视化（有效回传 / 访问未回传 / 未访问）

该可视化依赖 `metrics.jsonl` 中的：
- `covered_ids`
- `effective_ids`

因此请确保你的结果是用 **`uavlab.paper1.runner.run_vis`** 生成的。

### 4.1 对某个 run_dir 生成图

选取一个叶子目录（示例）：
- `runs/sweeps/demo_struct/c1_g1__struct_full_dual_loop_distributed/Ts40/seed0`

画 `episode=0`：

```bash
PYTHONPATH=src python3 -m uavlab.viz.plot_poi_outcomes \
  --run_dir runs/sweeps/demo_struct/c1_g1__struct_full_dual_loop_distributed/Ts40/seed0 \
  --episode 0
```

输出为：
- `<run_dir>/poi_outcomes_ep0.png`

### 4.2 叠加障碍物圆

如果场景来自 `configs/scenes/*.yaml`，通常会通过 `extends` 继承 `configs/scenes/base.yaml` 的 `obstacles_circles`。
绘图脚本已使用 resolved 加载方式解析 `extends`。

叠加障碍物：

```bash
PYTHONPATH=src python3 -m uavlab.viz.plot_poi_outcomes \
  --run_dir runs/sweeps/demo_struct/c1_g1__struct_full_dual_loop_distributed/Ts40/seed0 \
  --episode 0 \
  --show_obstacles
```

### 4.3 三种颜色含义

- **绿色**：visited & returned（`effective_ids`）
- **黄色**：visited, not returned（`covered_ids - effective_ids`）
- **灰色**：unvisited（不在 `covered_ids`）

---

## 5) 常见报错排查

### 5.1 `No module named 'uavlab'`

需要设置 `PYTHONPATH=src`，例如：

```bash
PYTHONPATH=src python3 -m uavlab.paper1.runner.run --help
```

### 5.2 `metrics.jsonl missing covered_ids/effective_ids`

说明该结果不是用 `run_vis` 生成的。解决：
- 重新用 `uavlab.paper1.runner.run_vis` 跑一遍该 `run_dir`（用其 `combined_config.yaml` 作为 `--config`）
- 或在 sweep plan 中把 `runner_module` 改为 `uavlab.paper1.runner.run_vis` 并重跑 sweep

### 5.3 同一个 tag 重跑后 summary 与终端输出不一致

历史原因可能是旧版本 `metrics.jsonl` 以追加模式写入，导致同目录多次运行混合统计。
当前 runner 已使用覆盖写，且 sweep 在每次运行前会清理旧 `metrics.jsonl`。

---

## 6) 说明：障碍物与碰撞

Paper1Lite 当前为任务级轻量仿真：
- **不会进行障碍物/碰撞检测**
- 即使场景 YAML 含 `obstacles_circles`，目前主要用于可视化展示与后续扩展

---

## 7) 轨迹可视化（Trajectory）

实现：`src/uavlab/viz/plot_trajectory.py`；入口：`python3 scripts/plot_trajectory.py`（脚本已把 `src/` 加入路径，一般可不加 `PYTHONPATH`）。

**前提**：叶子目录 `.../Ts*/seed0/` 内需有 `traj.jsonl`（`run_vis` 默认写）、`metrics.jsonl`、`resolved_config.json`。缺轨迹时用该目录的 `combined_config.yaml` 再跑一次 `run_vis` 并指定同一 `--run_dir`。

**单 run**（默认输出 `<run_dir>/trajectory_ep0.png`）：

```bash
python3 scripts/plot_trajectory.py --run_dir runs/sweeps/<tag>/<exp>/Ts40/seed0 --episode 0 --show_obstacles
```

**sweep 批量**（每 leaf 一张 PNG，默认 `--out_dir <sweep_root>/plot_trajectory`）：

```bash
python3 scripts/plot_trajectory.py --sweep_root runs/sweeps/<tag> --out_dir runs/sweeps/<tag>/plot_trajectory --episode 0 --show_obstacles
```

**按场景 mosaic**（同 `scene_file` basename 一张大图；leaf 父目录名需含 `__struct_` / `__modelling_` / `__coupling_` 才能拆成「case × 变体」网格）：

```bash
python3 scripts/plot_trajectory.py --sweep_root runs/sweeps/<tag> --out_dir runs/sweeps/<tag>/traj_mosaic --mosaic_publish
```

`--mosaic_publish` = `--mosaic_by_scene --show_obstacles --mosaic_comm_heatmap` + `quality` + `episode=0`；若已写 `--mosaic_comm_metric` 则保留你的取值。

其它常用项：`--show_blackholes`、`--heatmap_comm`（单图）、`--mosaic_subplot_bottom`（mosaic 底边留白）。详见 `python3 scripts/plot_trajectory.py --help`。
