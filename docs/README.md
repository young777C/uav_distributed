# 文档索引（uavlab / TaskA）

| 文档 | 说明 |
|------|------|
| [experimental_design_simulation_evaluation.md](experimental_design_simulation_evaluation.md) | 巡检任务、IMU/GPS/Camera/通信与观测契约；训练与 PX4+Gazebo 评价对齐 |
| [scene_experiment_world.md](scene_experiment_world.md) | 与 `scene.jpg` 一致的 250m 场景：POI/障碍/禁飞/通信退化/起降同点 |
| [uav_gcs_collaboration.md](uav_gcs_collaboration.md) | 起降点处 GCS；UAV 与 GCS 职责、快慢环分工、与代码对应 |
| [fast_slow_loop_design.md](fast_slow_loop_design.md) | 快环/慢环输入输出、数据回传策略 |
| [data_flow.md](data_flow.md) | 单步数据流与双链路；传感器级细节见实验设计文档 §2 |
| [dual_link_metrics_design.md](dual_link_metrics_design.md) | 控制链路与任务数据链路指标（公式.txt） |
| [model_selection_and_experiments.md](model_selection_and_experiments.md) | 全局/局部/回传模型选择切入点与实验设计（基线、消融、因子） |

**代码**：`TaskAEnv` 构造参数与 YAML 对齐统一由 `src/uavlab/tasks/taskA/env_from_config.py` 的 `build_env_kwargs` 生成（快环 train/eval、sweep、双环 run_hierarchical 共用）。
