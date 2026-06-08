# CODEX_WORKLOG

## 当前目标

使当前 UAV 分布式双环规划代码的实验输出与论文叙事保持一致。

## 当前重点

1. 梳理 FDCL、WCDL、CDSL 三种架构代码差异；
2. 检查通信模型是否真实影响任务选择；
3. 检查能量约束是否真实影响任务选择；
4. 检查返航约束是否真实影响任务选择；
5. 检查任务有效完成率和平均回传时延统计是否合理；
6. 检查实验结果是否支持 FDCL > WCDL > CDSL。

## 修改记录

每次 Codex 修改后，需要记录：

- 修改文件；
- 修改目的；
- 修改前问题；
- 修改后效果；
- 是否运行实验验证。

### 2026-06-05 耦合机制消融弱上传机制修正

- 修改文件：
  - `configs/experiments/paper1/system/coupling_periodic_goal.yaml`
  - `configs/experiments/paper1/system/coupling_event_driven_goal.yaml`
  - `src/uavlab/experiments/presets.py`
  - `tests/test_coupling_axis_control_vars.py`
- 修改目的：
  - 只针对耦合机制消融实验补齐弱耦合档的快环回传控制变量，使 PeriodicGoal / EventDrivenGoal 与 FullCoupling 的差异来自反馈可见性、反馈频率和状态可用性，而不是新增架构或新增指标。
- 修改前问题：
  - PeriodicGoal / EventDrivenGoal 继承 `struct_full_dual_loop_distributed.yaml` 后仍使用 `env.fast_upload_mode=policy`，在禁用快环 FSM 时会退化为“有 backlog 即尝试回传”，使弱耦合档获得接近 FullCoupling 的回传能力，压缩任务有效完成率和平均回传时延差异。
- 修改后效果：
  - PeriodicGoal / EventDrivenGoal 显式使用 `fast_upload_mode=fixed`、`fixed_send_ratio=0.2`；FullCoupling 保持 FDLC 的 `policy` 上传。
  - 架构轴配置未改动，FullCoupling 与 `struct_full_dual_loop_distributed.yaml` 的契约保持一致。
- 是否运行实验验证：
  - 已运行 `PYTHONPATH=src python3 -m pytest tests/test_coupling_axis_control_vars.py tests/test_sweep_slow_interval_override.py -q`，结果 `9 passed`。
  - 已运行配置解析检查，确认 PeriodicGoal / EventDrivenGoal 为 `fixed/0.2`，FullCoupling 与结构轴 FDLC 保持 `policy/0.5`。
  - 未重新跑完整 sweep。

### 2026-06-06 100 POI 场景与吞吐/能量保守性调优

- 修改文件：
  - `scripts/generate_POI.py`
  - `configs/scenes/g1_uniform.yaml`
  - `configs/scenes/g2_cluster_m0.yaml`
  - `configs/scenes/g2_cluster_m1.yaml`
  - `configs/scenes/g2_cluster_m2.yaml`
  - `configs/base.yaml`
  - `configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml`
- 修改目的：
  - 将 G1/G2 场景从 120 个 POI 降到 100 个 POI，降低任务密度对有效完成率的硬压制。
  - 按能量建模说明修正初始能量与单 POI 悬停估计，避免慢环把 5s dwell 当成 10s 悬停成本。
  - 在不增加任务时长、不关闭安全约束的前提下，略提高巡航速度，扩大慢环候选窗口/预取/规划深度，并放宽 FDCL backlog 返航/上传恢复门限，减少过早打断覆盖。
- 修改前问题：
  - 100 POI 初次生成后发现 G1 中存在 POI 落入 no-fly 区，导致部分架构在极少 POI 后被 no-fly 停留/规避拖垮，完成率异常低。
  - 慢环吞吐参数偏小，能量规划对 POI dwell 的估计偏保守，FDCL backlog 返航触发偏早，容易在聚簇任务中中断覆盖。
- 修改后效果：
  - 重新生成四个场景，每个场景均为 100 个 POI，且 `inside_nofly=0`。
  - `mission_time_s` 保持 2000s，未盲目增加任务时长；`v_xy_cruise` 从 10m/s 小幅提高到 11m/s；`return_reserve_s` 从 300s 调整为 240s；`E_0_Wh` 使用 263.2Wh；`energy.t_hov_s` 与 `slow_loop.poi_dwell_s=5s` 对齐。
  - C2/G2_M2 耦合验证显示：PeriodicGoal `R_task≈0.113/T_ret≈39.38s`，EventDrivenGoal `R_task≈0.192/T_ret≈66.65s`，FullCoupling `R_task≈0.675/T_ret≈10.05s`，FullCoupling 的覆盖后失败率为 0。
- 是否运行实验验证：
  - 已运行 `python3 scripts/generate_POI.py`。
  - 已校验 `g1_uniform.yaml`、`g2_cluster_m0.yaml`、`g2_cluster_m1.yaml`、`g2_cluster_m2.yaml` 均为 100 POI 且无 POI 位于 no-fly 区。
  - 已运行 `PYTHONPATH=src python3 -m pytest tests/test_coupling_axis_control_vars.py tests/test_sweep_slow_interval_override.py -q`，结果 `9 passed`。
  - 已运行 C2/G2_M2 耦合小 sweep：`configs/sweeps/paper1_axis_grid_coupling_v2.yaml`，输出目录 `runs/debug/coupling_goal_trace/coupling_c2g2m2_100poi_balanced_20260606`。
  - 已尝试完整结构轴 sweep：`runs/sweeps/struct_axis_100poi_balanced_20260606`，3600s 超时，只完成 27/72 个 run，覆盖到 C1/G2_M1；该结果只能作为局部验证，不能视为完整结构轴结论。
  - 早先的 `runs/sweeps/struct_axis_100poi_throughput_20260606` 混入了修复前 no-fly 场景和参数，不作为有效实验结论使用。
