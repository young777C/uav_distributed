# Phase 1: FDLC × PX4+Gazebo 离线仿真 (Dry-Run) 验证结果

> 生成日期: 2026-06-26
> 分支: `codex-paper-align-001`
> 相关源码: `scripts/phase1_fdlc_px4_bridge.py`

---

## 1. 概述

FDLC (Feedback-Driven Loop Closure) 快环 FSM 已在两个场景上通过 `--dry-run` 模式完成离线验证。该模式不依赖 PX4 SITL，使用 `Paper1Env` 内部运动学模型仿真 UAV 运动，FDLC 决策逻辑与真机模式完全相同。

### 验证范围

| 验证项 | 小场景 (3 POI) | 大场景 (100 POI) |
|--------|:-------------:|:----------------:|
| FSM: Sins | ✅ | ✅ |
| FSM: Stx 触发 | ✅ | ⏳ 未到达 |
| FSM: Srec 触发 | ✅ | ✅ (1步) |
| FSM: Ssafe 触发 | N/A (无禁飞区) | **✅ 首次验证** |
| FSM: Sback 触发 | ⏳ 未触发 | ⏳ 未触发 |
| 通信距离衰减模型 | ✅ | ✅ |
| POI 覆盖检测 | ✅ | ✅ |
| Backlog 回传管理 | ✅ | ⏳ 未触发 |

---

## 2. 小场景 (phase1_px4)

### 配置

- **配置文件**: `configs/experiments/paper1/cases/phase1_px4.yaml`
- **系统配置**: `configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml`
- **场景**: `configs/scenes/phase1_px4.yaml`
  - 200m × 200m, GCS=(0,0), 3 POI: (50,0), (0,80), (-40,-40)
  - 无禁飞区, 无通信黑洞
- **通信**: 距离衰减 5%→50%, d₁=150m
- **参数**: 5Hz, v_cruise=5m/s, 120s

### 运行结果

```
运行目录: runs/debug/phase1_fdlc_dryrun_verify/
总步数: 630 | 时长: 119.9s
经停 POI: 0 → 1 → 2
```

#### 任务时间线

```
t=  0s  出发 → POI 0 (50,0)
t=  8s  POI 0 到达
t= 10s  → POI 1 (0,80)
t= 28s  POI 1 到达
t= 30s  → POI 2 (-40,-40)
t= 55s  POI 2 到达，backlog 注入 ~11.7Mbits → Stx 模式
t= 57s  链路不足 → Srec 恢复等待
t= 68s  backlog 清零 → Sins
t=68-120s 悬停等待
```

#### FSM 模式分布

| 模式 | 步数 | 占比 |
|------|------|:----:|
| Sins | 553 | 87.8% |
| Srec | 66 | 10.5% |
| Stx | 11 | 1.7% |

#### 关键观察

- **FSM 正确按优先级切换**: 覆盖后自动触发 pending-return (Stx) → 链路不足降级 Srec → 回传完成回到 Sins
- **链路质量随距离变化**: GCS 附近 loss=5% → 最远点 (56m) loss=22%

---

## 3. 大场景 (C1_G1)

### 配置

- **配置文件**: `configs/experiments/paper1/cases/c1_g1.yaml`
- **系统配置**: `configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml`
- **场景**: `configs/scenes/g1_uniform.yaml` (extends `configs/scenes/base.yaml`)
  - 2500m × 2500m, GCS=(1250,1250), 100 POI
  - 9 个禁飞区
- **通信**: 距离衰减 5%→40%, d₁=2500m
- **参数**: 5Hz, v_cruise=11m/s, 180s (仿真步长)

### 运行结果

```
运行目录: runs/debug/phase1_fdlc_dryrun_fullscale/
总步数: 966 | 时长: 179.9s
经停 POI: 0 → 1, 前往 POI 2 途中
```

#### 任务时间线

```
t=  0s  出发 → POI 0 (1865,1412)  [NE方向]
t= 54s  POI 0 到达 → POI 1 (2342,1632)
t=107s  POI 1 到达 → POI 2 (1416,417) [SW方向]
t=162s  进入禁飞区 #8 (2000,900, r=250m) → Ssafe 触发
t=163-180s  Ssafe 持续中...
```

#### FSM 模式分布

| 模式 | 步数 | 占比 |
|------|------|:----:|
| Sins | 863 | 89.3% |
| **Ssafe** | **102** | **10.6%** |
| Srec | 1 | 0.1% |

#### 链路质量统计

| 指标 | 值 |
|------|:--:|
| 平均 loss | 14.9% |
| 最小 loss | 5.0% (GCS) |
| 最大 loss | 21.0% (距 GCS 1143m) |

#### 关键观察: 禁飞区触发现象

UAV 从 POI 1 (2342,1632) 直线飞往 POI 2 (1416,417) 时，航路经过禁飞区 #8 (中心 2000,900, 半径 250m)。FSM 在 t=162s 正确检测并切换至 Ssafe。

**⚠️ 已知问题**: Ssafe 模式下 UAV 并未有效远离禁飞区中心（距离从 249m→181m，反而在靠近）。原因：
1. 桥接器 (`phase1_fdlc_px4_bridge.py`) 在 `cmd.vel_ne_cmd is None` 时会用 P 控制器向目标推，可能覆盖 Ssafe 的逃避速度
2. 桥接器使用简单的航点列表（无慢环 OR-Tools 路径规划），无法规划绕行禁飞区的路径

---

## 4. PX4 实连状态

### 现有运行记录

| 目录 | 类型 | 结果 |
|------|------|------|
| `runs/debug/phase1_fdlc_dryrun/` | dry-run (旧版) | ✅ `traj.jsonl` 正常 (84步) |
| `runs/debug/phase1_fdlc_dryrun_verify/` | **dry-run 小场景** | ✅ **本次验证: 630步完整FSM** |
| `runs/debug/phase1_fdlc_dryrun_fullscale/` | **dry-run 大场景** | ✅ **本次验证: 966步含Ssafe** |
| `runs/debug/phase1_fdlc_px4/` | PX4 SITL 实连 | ⚠️ 连接成功，但 UAV 未移动(642步原位) |
| `runs/debug/phase1_fdlc_px4_wsl/` | PX4 SITL 实连 (WSL2) | ⚠️ 同上 |
| `runs/debug/phase1_fdlc_px4_20260625_164153/` | PX4 SITL 实连 | ❌ 仅有 `resolved_config.json` |

### PX4 实连问题分析

从 `runs/debug/phase1_fdlc_px4/traj.jsonl` 数据看：
- PX4 SITL **成功发送遥测**（yaw≈1.52rad 来自真实姿态估计，pos=(0.01,-0.01) 来自真实 GPS）
- 桥接器 **持续发送 Offboard 速度指令** (8m/s 向东)
- **但 UAV 位置从未变化** → 说明 `arm_and_offboard()` 或 `takeoff()` 序列未生效

可能原因：
1. Offboard 模式进入失败（`COMMAND_ACK` 未验证）
2. 起飞前未正确建立 Offboard setpoint 流
3. MAVLink type_mask 或坐标系设置问题

---

## 5. PX4 桥接相关文件索引

### 入口脚本
- `scripts/phase1_fdlc_px4_bridge.py` — 主桥接器 (pymavlink)
- `scripts/fast_loop_px4_bridge.py` — 简化 MAVSDK 示例
- `scripts/test_px4_connect.py` — 连通性测试

### Docker 部署
- `scripts/docker-compose.yml` — Docker Compose 编排
- `scripts/docker_run_px4_sitl.sh` — PX4 SITL 容器启动
- `scripts/phase1_docker_bridge.sh` — 一站式 Docker+Bridge
- `scripts/Dockerfile.px4` — PX4 Docker 镜像定义

### 配置
- `configs/experiments/paper1/cases/phase1_px4.yaml` — PX4 实验 case
- `configs/scenes/phase1_px4.yaml` — PX4 测试场景
- `configs/comm_profiles/phase1_px4.yaml` — PX4 通信 Profile

### FDLC 核心 (被桥接器调用)
- `src/uavlab/paper1/loops/fast/fast_loop.py` — `FastLoop.step()` 快环决策
- `src/uavlab/paper1/loops/fast/fsm.py` — `select_comm_mode()` FSM 模式选择
- `src/uavlab/paper1/loops/fast/guidance.py` — 航点跟踪/安全撤离
- `src/uavlab/paper1/sim/env.py` — `Paper1Env` 任务级仿真环境 (dry-run 使用)
- `src/uavlab/paper1/contracts/contract_config.py` — 合同配置

---

## 6. 验证路线图

```
Phase 1: ✅ Dry-run 离线
  ├── ✅ FSM: Sins/Stx/Srec 在小场景上验证
  ├── ✅ FSM: Ssafe 在大场景上验证
  ├── ⏳ FSM: Sback 未触发 (需要长距飞行耗尽能量)
  └── ⏳ Stx/Srec 在大场景上未触发 (180s 不足以覆盖足够多 POI)

Phase 2: 🔄 PX4 SITL 在线调试
  ├── ❌ Offboard 模式进入 (需修复 arm/offboard 序列)
  ├── ❌ 起飞 + 位置控制闭环
  └── ⏳ 完整 FDLC 快环在 PX4 上运行

Phase 3: 🚁 全场景 Paper1Lite 仿真
  └── 使用 scripts/sweep.py (含慢环 OR-Tools 规划)
```

## 7. 常用命令备忘

```bash
# 小场景 dry-run
PYTHONPATH=src python3 scripts/phase1_fdlc_px4_bridge.py \
  --config configs/experiments/paper1/cases/phase1_px4.yaml \
  --system configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml \
  --duration_s 120 --dry-run

# 大场景 dry-run (C1_G1, 100 POI)
PYTHONPATH=src python3 scripts/phase1_fdlc_px4_bridge.py \
  --config configs/experiments/paper1/cases/c1_g1.yaml \
  --system configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml \
  --duration_s 180 --dry-run

# 连接 PX4 SITL (需先启动 PX4)
PYTHONPATH=src python3 scripts/phase1_fdlc_px4_bridge.py \
  --config configs/experiments/paper1/cases/phase1_px4.yaml

# 一键 Docker + Bridge
bash scripts/phase1_docker_bridge.sh --duration 120
```
