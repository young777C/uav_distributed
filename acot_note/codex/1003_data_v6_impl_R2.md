verdict:修复后再生成

总评：R1 的工程性修复大多已经落地，但 v6.1 还没有通过干净 smoke 验收；当前最危险的不是标签字段缺失，而是“场景跑不出目标分布”。我复核 smoke2–4：用当前 `qc_v6.check_episode()` 读正式 `.h5`，没有一个正式 episode 通过当前 QC。全量 500 集现在开跑，极大概率会生成大量 reject/top-up，并且之后还要重来。

**MUST**

- 问题：G9 规格与实现不一致，且 smoke 证明目标仍大量静止。  
  证据：[1003_data-regen-v6-spec.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-regen-v6-spec.md:92) 写“运动帧占比 ≥50%”，但 [qc_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/qc_v6.py:17) 和 [qc_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/qc_v6.py:109) 实现为 40%。smoke2 的 `episode_101500/105500/110000` 运动占比分别约 0.32/0.34/0.14。  
  改法：先统一阈值；若论文希望“非静止数据”，按 50% 执行，并修 spawn/route/traffic，而不是降门槛。  
  验收：干净 smoke 每个训练 town 至少 1 集 pass，且运动占比 p10 ≥ 0.5。

- 问题：reacq 的 `hover` 丢失机制仍未产出有效重捕获。  
  证据：[v6_reacq.yaml](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/config/scenarios/v6_reacq.yaml:8) 配了 4 个 loss window、[recorder.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/recorder.py:226) 在窗口内悬停并写 `action_valid=0`，但 smoke2/3 `episode_202000.rejected.h5` 的 loss window 中目标仍 100% visible，`events/target_reentry` 为空。强制转弯只是 [scene_manager.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scene/scene_manager.py:1057) `set_route(["Left","Right"])`，CARLA 可能不按预期立即离开 FOV。  
  改法：loss window 开始后在线检测目标是否在 1–1.5s 内真正不可见；若没有，改用确定性相机/无人机 yaw hold、目标临时 lateral route/teleport 到合法 lane、或扩大初始相对运动。  
  验收：reacq 每集至少 1 次 `gap>=30 && lookalike>=1`，并报告 loss window 内 visible 比例。

- 问题：当前没有干净 occ smoke pass，不能证明 v6.1 的 occ-heavy 子集可生成。  
  证据：[v6_occ.yaml](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/config/scenarios/v6_occ.yaml:14) 已设 `bias_fraction: 1.0`，但 smoke2–4 里 occ 只有 `.tmp.h5` 或缺 postprocess；[qc_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/qc_v6.py:116) 的 occ 门槛尚未被干净样本证明能通过。  
  改法：单独跑 Town05、Town10HD、Town03 的 occ smoke，失败时调整 occluder anchor 或 oblique camera，而不是靠补录硬筛。  
  验收：每个 occ town 至少 2/3 pass，且 `target_struct_occ_frames >= 20` 或 `cause_occ >= 1`。

- 问题：拒收 + 补录会引入选择偏差，当前没有 reject ledger 作为论文证据。  
  证据：[run_job_v6.sh](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/run_job_v6.sh:35) 用私有编号段补录，[generate_batch.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/generate_batch.py:151) 单集 QC 不过即 `.rejected.h5`；但最终 split 只保留 good，[qc_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/qc_v6.py:153)。这会系统性排除楼宇峡谷、长遮挡、低运动样本。  
  改法：保留并汇总所有 rejected 的 subset/town/seed/fail reason/T/运动/遮挡/重入画统计，随数据卡报告。  
  验收：最终发布包含 `reject_report.json`；若任一 subset×town reject 率 > 50%，必须调场景或在论文中声明选择机制。

- 问题：补录预算 2N 在当前通过率下明显不稳。  
  证据：[run_job_v6.sh](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/run_job_v6.sh:41) 每块最多 2N top-up；smoke2–4 当前复核正式 pass 为 0，失败主因包括 G9、G8 reacq、短截断。  
  改法：先把 clean smoke pass 率拉到至少 70–80%，再跑 full；否则 2N 只是把失败延后。  
  验收：同一 commit、空目录 smoke5/6 连续两轮，各 subset×town pass 率达标。

- 问题：磁盘保护低估占用，会漏算 `.rejected.h5` 和 `.tmp.h5`。  
  证据：[generate_batch.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/generate_batch.py:137) 只统计 `episode_*.h5`，而 rejected 文件名是 `episode_*.rejected.h5`，tmp 也不算。  
  改法：磁盘保护按目录总 `episode_*` 或 `shutil.disk_usage` 实际剩余空间执行。  
  验收：制造 rejected/tmp 后，生成器仍能正确触发空间停止。

- 问题：分布改动足以影响与 Stage-2 原 v5 的可比性。  
  证据：目标/干扰车 100% 忽略红绿灯见 [scene_manager.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scene/scene_manager.py:189)，背景车降到 40 见 [v6_base.yaml](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/config/scenarios/v6_base.yaml:13)，目标类别 car-only 见 [v6_base.yaml](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/config/scenarios/v6_base.yaml:20)，随机机动关闭见 [v6_base.yaml](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/config/scenarios/v6_base.yaml:31)。  
  改法：attrs/数据卡显式写 `dataset_version=v6.1_controlled_day_caronly`、traffic light policy、ambient density、reject policy；论文不要称其为 v5 自然分布等价复刻。  
  验收：数据卡有 v5/v6.1 差异表；主实验以 v6.1 为新 benchmark，v5 只作历史参考或另保留小型原样校准集。

**SHOULD**

- R1 已修：原子提交基本到位。证据：[generate_batch.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/generate_batch.py:111) 写 `.tmp.h5`，[generate_batch.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/generate_batch.py:157) `os.replace` 到正式或 rejected。仍建议 kill-test 验收。
- R1 已修：并发锁与 resume 校验到位。证据：[orchestrate.py](/nvidia/hque/code/cyhe/uav-acot-track/runs/data_v6/orchestrate.py:110) `.lock`，[orchestrate.py](/nvidia/hque/code/cyhe/uav-acot-track/runs/data_v6/orchestrate.py:123) 校验 commit/dirty/plan。
- R1 已修：原始实例图和 3D 字段已补。证据：[recorder.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/recorder.py:428) 写 `seg/iid`/`seg/semantic`，[recorder.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/recorder.py:397) 写 `ids/role/extent/bbox_loc`，[recorder.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/recorder.py:423) 写 `yaw/pos`。
- R1 已修但需继续观察：近裁剪与时序关联已实现。证据：[labels_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/labels_v6.py:50) 近平面交点，[labels_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/labels_v6.py:126) 上帧 iid ×2，[qc_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scripts/qc_v6.py:92) G7。建议增加 per-actor dominant-iid 报告，不只查 target。

**NICE**

- 给 `qc_v6.py` 增加 `--include-rejected` 和失败原因聚合；现在主程序只扫正式 `episode_*.h5`，不利于选择偏差审计。
- 把 `config_snapshot` 之外的关键分布开关提升为直接 attrs，方便不用解析 YAML 就能筛选：`ignore_lights_pct`、`ambient_vehicles`、`target_class_policy`、`reject_qc_version`。
- 每轮 smoke 用新目录，避免像 smoke2–4 这样混入不同小修版本；否则 summary 里的 `status: ok` 会和当前 QC 复核结果不一致。