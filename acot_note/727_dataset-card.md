# ACoT-UAV-Track — MVP 数据卡 (Dataset Card)

> 版本: MVP v1 (A1+A1b+A1c②) · 生成日期 2026-07-27 · seed 3000
> 位置: 宿主机 `/nvidia/hque/data/carla_data/mvp/`（容器内 `/data/mvp/`）
> 生成管线: `uav-acot-track/carla_uav_tracking/`（`cyh-carla` 容器, CARLA 0.9.15 / UE4.26, `-quality-level=Epic`）

---

## 1. 概述

空中无人机 **语言指定 + 相似车辆消歧 + 闭环跟踪** 的模仿学习数据集。每集：一辆语言指定的 **目标车辆** + 若干 **视觉相似干扰车**（look-alike），无人机从后上方俯视跟踪，专家 P-only PID 视觉伺服提供动作。用于 ACoT-UAV-Track 的 Stage-1（EAR 预热）与 Stage-2（EAR+IAR+DiT 联合训练）。

**定位**: MVP / 概念验证档（design §4.1）。可跑通训练、验证 EAR 收敛、支撑消融管线；**非**可发表规模（可发表需 ~300K 帧）。

### 速览
| 项 | 值 |
|---|---|
| Episodes | **60** |
| 总帧数 | **100,603**（RGB 336×336, gzip-9 无损） |
| 大小 | ~21 GB |
| 单集时长 | 目标 180s@10fps=1800 帧；均值 **1676**（53 满长 / 7 遮挡截断, 最短 92） |
| 目标类别 | 仅车辆（汽车/摩托/自行车/电动）；**行人仅作背景** |
| 地图 | Town01/02/04/05/10HD |
| 目标在框率 | **99.36%** |

---

## 2. 生成配置

- **仿真器**: CARLA 0.9.15 / UE4.26, `-quality-level=Epic`（Low 会在 load_world 段错误）。
- **相机**: 单 RGB, 336×336, FOV 90°, pitch **−50°**（俯视,匹配 20–60m 高空的目标俯角）。
- **无人机**: 运动学模型（`set_transform`,无物理）；跟踪高度 = 距离档决定的 **20–60m**。
- **专家**: P-only PID 视觉伺服 + 每集随机风格（噪声/增益）；**机动注入器关闭**（原 raw-control 会致乱开,已改 TM 平滑版但默认关）。
- **背景交通**: 官方 `generate_traffic.py` 式批量 spawn —— **80 辆多类车**（含公交/卡车/摩托/自行车）+ **30 行人**，用 hybrid-physics + `respawn_dormant`（目标=hero）集中在目标周围 **半径 70m**。背景不标注。
- **look-alike 共视**: 仅"共位出生"（`covisibility.enabled=false`,瞬移回收关闭以保平顺）。
- **天气**: 白天,无雾(fog=0),随机 cloudiness/sun_altitude/wetness。
- **可复现**: 每集 `seed = 3000 + episode_id`,统一播种 `random/numpy/TrafficManager`。

---

## 3. HDF5 Schema（每集 `episode_XXXXXX.h5`）

```
├── rgb                     (T, 336, 336, 3) uint8, gzip-9        # UAV 相机
├── state/
│   ├── t, uav_x/y/z, uav_vx/vy/vz, uav_yaw, uav_energy          # 本体感知
│   ├── is_turning, perturbation, obstacle_avoid, maneuver        # 事件标志 (0/1)
│   ├── cam_x/y/z, cam_pitch, cam_yaw                             # 逐帧相机位姿(投影用)
│   └── occ_raycast          [0,1]                                # A1b 结构遮挡(见 §4)
├── target/                 tx,ty,tz, tvx,tvy,tvz, tspeed, tyaw   # 目标 3D 状态
├── action/                 dx, dy, dz, dyaw                      # 专家动作(世界系)
├── distractors/
│   ├── positions           (T, D, 6) [x,y,z,vx,vy,vz]           # 干扰车逐帧轨迹
│   └── bbox                 (T, D, 4) [u,v,w,h]                  # A1 干扰车 2D bbox(供 L_target_id)
├── annotation/
│   ├── bbox_u/v/w/h                                             # 目标 2D bbox(正确 336 投影)
│   ├── occlusion           [0,1]                                # 旧合并信号 = max(视图几何, occ_raycast)(兼容保留)
│   ├── occ_structural      [0,1]                                # A1c② 仅框内结构遮挡(IAR 遮挡头训练用)
│   ├── off_screen          {0,1}                                # A1c② 出画/相机后(独立信号,驱动搜索)
│   ├── occluded            {0,1}                                # A1c② "看不见"掩码=(occ_structural>=.5)|off_screen → 屏蔽动作损失
│   ├── search_mode         [0,1]                                # occlusion×(1+time_lost/5)
│   ├── waypoints           (T, 9) = (T,3,3) 未来 +2/4/6s 目标位置
│   └── waypoint_sigma      (T, 3)  不确定度
└── attrs:
    language, target_desc, target_bp, target_color, target_class,
    strategy, num_similar, distractors(json列表: id/bp/color/desc/similar/cls),
    seed, town, weather(json)
```
`bbox`/`positions` 中 `[-1,-1,0,0]` / NaN 表示该 actor 在相机后方或出画。

---

## 4. 标签语义

- **目标/干扰车 2D bbox**（A1）: 用 `acot_probe/projection.py` 的正确投影（含 cam_pitch/yaw + 真实 336²,与 CARLA 运行时 Δ=0 验证）重算；干扰车 bbox 为 **Stage-2 target-ID 损失**提供候选框。因仅存 3D 中心,bbox 为**距离缩放的伪方框**。
- **遮挡三信号（A1c② 分离,互不混叠)** —— 原 `occlusion=max(视图几何,occ_raycast)` 把"出画(0.5)"和"框内结构遮挡"合并了,会让 IAR 遮挡头学成"出画"。现拆成三个独立标签:
  - **`occ_structural` [0,1]**: 目标**在框内但被几何遮挡**(=`state/occ_raycast` 门控到在框内)。`occ_raycast` = 目标 bbox 8 角点向相机打射线被建筑/隧道/树/桥挡住的比例(`OcclusionDetector`,录制时实时算)。**IAR §6.2/§6.4 遮挡头应训练此信号。** 已可视化验证:高值帧目标确被墙/树/桥挡住。
  - **`off_screen` {0,1}**: 目标出画或在相机后(独立信号,驱动搜索/重捕获,**不是**结构遮挡)。经验证 occ_raycast 本已 99% 落在框内,故 `occ_structural` 与 `off_screen` **零重叠**。
  - **`occluded` {0,1}**: "看不见目标"掩码 =(`occ_structural`≥0.5)∨`off_screen`。**用于遮挡期屏蔽动作模仿损失**(专家用了策略看不到的特权真值);**EAR waypoint 损失不屏蔽** —— 让模型继续学"目标从哪出来"。
  - 旧 `occlusion` 字段保留(向后兼容)。
- **EAR 穿过遮挡(A1c①)**: `annotation/waypoints` 用特权目标轨迹算,**遮挡期依然有效**(占用帧 waypoint NaN=0);遮挡时 waypoint 指向目标沿路**前方**的重现点(例:ep24 遮挡帧,+6s 路点前移 19.2m),给无人机遮挡期的拦截点。
- **search_mode** = occlusion 派生（遮挡越久越大）。
- **waypoints**: 目标未来 +2/4/6s 位置（移动平均平滑）,EAR 监督用（穿过遮挡）。
- **language**: 富指代表达 `[动作] + the [颜色+型号] + [空间关系] + [运动意图] + [距离约束]`,纯英文,全部真值驱动（空间/距离来自几何,运动意图由轨迹后处理判定）。**无消歧从句**（作为冗余捷径已移除）。

---

## 5. 分布统计（全 60 集真实值）

**目标类别**: car 45 (75%) · bicycle 8 (13%) · motorcycle 5 (8%) · scooter 2 (3%)

**消歧策略**: same_color_diff_shape 23 · same_shape_diff_color 16 · distinct 6 · same_class(两轮) 15
（相似 look-alike 数 `num_similar`: 均值 2.2, 范围 0–3;`distractors` 每集 4–8 辆）

**地图**: Town01 20 · Town02 10 · Town04 10 · Town05 10 · Town10HD 10

**距离档**（由语言 "about N m" 统计）: close(≤25m) 14 · suitable(30–40m) 23 · long(≥45m) 23

**颜色**(18 色全覆盖): orange 7 · purple 6 · yellow/royal blue/black 各 4 · dark red/light blue/dark blue/gray/green/red/blue/silver/gold/dark green 各 3 · beige/white 2 · brown 1

**遮挡**: mean occlusion **0.045**;结构遮挡帧(occ_raycast>0.3) **4.32%**;完全遮挡帧(≥0.6) **2.20%**（密集/隧道图更高,开阔乡村近 0）

**look-alike 共视**: ≥2 相似车同时在框 **13.6%** 的帧（其余帧仍有 0–1 个相似车 + 大量未标注背景车）

**集长度**: 53 满长(1800) / 7 遮挡截断（最短 92 帧,截断保证数据无 >5s 长期丢失）

---

## 6. 已知局限 / 使用注意

1. **规模 = MVP 档**（~100K 帧）。够 Stage-1 收敛 + Stage-2 冒烟/消融;可发表需 ~300K（design §4.1）。
2. **look-alike 共视率 ~14%**: 因干扰车共位出生后自然散开（瞬移回收关闭以保平顺行驶）。消歧硬样本 ~1.4 万帧,可用但偏低;若需强化 §6.3 主张,可后续加"离屏平滑回收"。
3. **结构遮挡量级偏低**（`occ_structural`>0.3 仅 **4.3%** 帧;`off_screen` 0.6%;`occluded` 掩码 4.0%）: 空中俯视天生遮挡稀少 + 截断限制长遮挡;标签本身**真实且已与出画分离**,但 H2 强证据需要**更多框内遮挡场景** —— 这属 **A1c③(场景重设计)**,已决定**留到可发表级(~300K)重生成时做**,MVP 阶段不做。
4. **距离恒为档位内值**（close/suitable/long ≈ 15/30/45m）,由专家维持高度实现。
5. **背景车流不标注**（80 辆 + 30 行人只作视觉杂波）;偶有未标注的相似背景车（自然干扰,一般无害）。
6. **专家上限**: P-only PID 且录制时始终拿目标真值位置（`target_visible` 恒 True）;模仿上限 ≈ 该 PID,遮挡恢复主要来自扰动而非真实视觉丢失。
7. **行人不作目标**（仅背景干扰,不分类年龄/性别）。
8. `num_distractors` 未单独存 attr —— 干扰车数请从 `distractors`(json) 长度或 `distractors/positions` 的 D 维取。

---

## 7. 复现与来源

- **代码**: `carla_uav_tracking/{scene,recording,targets,carla_uav,config}`;配置 `config/default.yaml` + `config/scenarios/mvp.yaml`。
- **生成命令**（容器内）:
  ```
  bash scripts/run_resilient.sh config/scenarios/mvp.yaml /data/mvp 60 3000 50 2000 0
  ```
- **QC/可视化**: `scripts/qc_report.py`, `scripts/render_video.py`;抽查图 `/data/mvp/{bbcheck,occcheck}/`。
- **投影一致性**: postprocess 投影 == `acot_probe/projection.py` == CARLA 运行时（Δu=Δv=0 验证）。
- **seed 3000**: 同 seed 重跑得到相同场景/指令/目标/干扰车（A1b 在此基础上补齐了 `occ_raycast`）。

---

## 8. 预期用途（各训练阶段）

| 阶段 | 用到的标签 | 数据是否够 |
|---|---|---|
| Stage-1 EAR 预热 | `annotation/waypoints` | ✅ 够 |
| Stage-2 动作模仿 | `action/*` + `rgb` + `state` + `language` | ✅ 够 |
| Stage-2 target-ID (L_target_id) | `distractors/bbox` + `annotation/bbox_*` | ✅ 够(A1) |
| Stage-2 动作损失遮挡屏蔽 | `annotation/occluded`(掩码) | ✅ 数据侧就绪(A1c②);训练侧套用即可 |
| Stage-2 IAR 遮挡头 / §6.4 search_mode | `annotation/{occ_structural,off_screen,search_mode}` | 🟡 已分离(A1c②),结构遮挡够训练但量级偏低 |
| Stage-1 EAR 穿过遮挡 | `annotation/waypoints`(遮挡期有效)+`occluded` | ✅ 就绪(A1c①),遮挡期路点指向重现点 |
| §6.2 IAR "即将被遮挡"先验 / H2 强证据 | `occ_structural` | 🟡 有真实分离标签,H2 强证据待 A1c③(可发表级重生成) |
| Stage-3 RL | 无(仿真 rollout) | — |
