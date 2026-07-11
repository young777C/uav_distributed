# CLAUDE.md — uav-distributed

> UAV 巡检任务级仿真平台（Paper1 / Paper1V2）：通信退化条件下 CDSL / WCDL / FDLC 对比。

## 长期规则 (Standing Rules)

### 0. 实验 Skill 优先
所有实验运行必须首先读取并遵循：
```
/home/yuhe/workspace/uav-distributed/skills/C3_UAV_GCS_实验运行_SKILL.md
```

### 1. 脚本命名
- 所有为当前论文（Paper1V2 / C3）新增的实验、统计、校验、表格和绘图脚本，文件名必须使用 `paper1_v2_` 前缀。
- 统一放入 `scripts/paper1_v2/`。
- 每个新增脚本文件头必须包含：
  ```python
  """
  Paper: paper1_v2
  Purpose: <用途>
  Inputs: <输入目录或配置>
  Outputs: <输出目录或文件>
  """
  ```

### 2. 结果隔离
- 所有新实验结果只能写入 `results_v2/`。
- 禁止混入 `results/` 旧实验结果。
- 禁止从旧 `summary.csv` 复制数值。
- 不覆盖旧目录。

### 3. 图片规范
- 生成或修改论文图片前，必须读取并遵循：
  ```
  /home/yuhe/workspace/uav-distributed/Autonomous_Robots_图片样式与制作规范.md
  ```
- 优先输出 PDF 矢量图。
- 方法颜色、线型、marker 全文统一（参见图片规范 §5）。
- 误差棒为 seed 级 95% CI。

### 4. 名称统一
- 完整方法名称统一为 `FDLC`，**禁止**写成 `FDCL`。
- 方法列表：CDSL / WCDL / RHC-Inspection / CBCP / FDLC。
- 耦合策略：Periodic Goal / Event-driven Goal / Full Coupling。
- `Full Coupling = FDLC 完整配置`。

### 5. 统计规模
- 正式实验使用 **10 seeds × 10 episodes**。
- seed 是独立统计单元，episode 先在 seed 内聚合。
- 不得把 episode 当成独立样本进行显著性检验。

### 6. 结果复用
- FDLC、Full Coupling 和 rho=40 的完整配置必须复用同一批结果。
- 不得为同一配置重复运行并生成多组默认数值。

### 7. 保持方案 B
- backlog 由快环管理。
- backlog 不进入慢环目标函数、约束或候选排序。
- 结果解释不得声称 backlog 直接修改慢环目标函数。

### 8. 不修改核心算法
- 优先复用现有代码。
- 新增 runner / 封装脚本，不改变 sim、loops、planner 的行为。

### 9. 禁止方向
不引入：强化学习、深度学习、复杂神经网络、与当前论文无关的新系统架构。

### 10. 提交规范
- 一个 PR 只做一件事。
- 未经用户明确要求不要 commit / push。
- Commit 信息格式：`feat(...): ...` / `fix(...): ...`。

---

## 项目结构

```text
configs/
  base.yaml                # 全局默认
  comm_profiles/           # C1, C2, C3_{low,medium,high,severe}
  scenes/                  # g2_cluster_m{0,1,2,3}.yaml + g1_uniform.yaml
  experiments/paper1/
    system/                # struct_*.yaml, coupling_*.yaml, modelling_*.yaml
    cases/                 # c3_*.yaml (C3 case configs)
  sweeps/                  # sweep plans
src/uavlab/
  paper1/
    loops/slow/            # SlowLoop, planner
    loops/fast/            # FastLoop, policy
    contracts/             # StructAxisProfile, contract_config
    coupling/              # CouplingPolicy
    sim/                   # Env, config
    metrics/               # paper_metrics, link_recovery
    runner/                # run.py (单episode入口), run_vis.py
  experiments/presets.py   # 实验预设
scripts/
  sweep.py                 # 旧 sweep runner
  generate_POI.py          # 场景生成器
  paper1_v2/               # 新增 paper1_v2 脚本 (待创建)
results/                   # 旧实验结果 (只读)
results_v2/                # 新实验结果 (待创建)
tests/                     # pytest 单元测试
docs/                      # 设计文档
```

## 实验入口

```bash
# 单 episode 入口
PYTHONPATH=src python3 -m uavlab.paper1.runner.run \
  --config <case.yaml> --system <system.yaml> \
  --episodes 10 --seed 0 --slow_interval_steps 40 \
  --run_dir <dir> --metrics_jsonl <dir>/metrics.jsonl

# Sweep 入口 (旧)
PYTHONPATH=src python3 scripts/sweep.py --plan <plan.yaml> --tag <tag>
```

## C3 场景矩阵 (16 scenes)

```text
c3_low_m0       c3_low_m1       c3_low_m2       c3_low_m3
c3_medium_m0    c3_medium_m1    c3_medium_m2    c3_medium_m3
c3_high_m0      c3_high_m1      c3_high_m2      c3_high_m3
c3_severe_m0    c3_severe_m1    c3_severe_m2    c3_severe_m3
```

所有场景为 G2 聚簇分布，C3 = F1(距离衰减) + F2(局部阴影) + F3(时变抖动)。
