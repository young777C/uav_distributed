---
name: c3-uav-gcs-experiment-runner
description: >
  Run, resume, validate, aggregate, and statistically analyze the C3 UAV–GCS
  experiment matrix. Store all new outputs under results_v2 and never mix legacy results.
---

# C3 UAV–GCS 实验运行 Skill

## 1. 目标

重新建立统一、可复现、可配对统计的实验结果，替代旧实验数据。

统一复合通信场景：

\[
\mathrm{C3}
=
\text{距离均匀衰减}
+
\text{局部链路退化}
+
\text{时变随机抖动}.
\]

实验因素：

- 退化强度：`low`、`medium`、`high`、`severe`
- POI–弱链路区域冲突：`M0`、`M1`、`M2`、`M3`

共 16 个场景：

```text
c3_low_m0       c3_low_m1       c3_low_m2       c3_low_m3
c3_medium_m0    c3_medium_m1    c3_medium_m2    c3_medium_m3
c3_high_m0      c3_high_m1      c3_high_m2      c3_high_m3
c3_severe_m0    c3_severe_m1    c3_severe_m2    c3_severe_m3
```

本 Skill 完成：

1. 总体方法对比；
2. 耦合机制消融；
3. 双时间尺度敏感性；
4. 计算实时性与部署可行性；
5. 完整性检查、统计分析和论文图表输出。

---

## 2. 不可违反的规则

### 2.1 结果隔离

- 所有新结果仅写入 `results_v2/`
- 不使用旧结果参与统计
- 不覆盖旧目录
- 不复制旧表格数值
- 每次运行保存配置、日志、代码版本和环境信息

### 2.2 名称统一

方法：

```text
CDSL
WCDL
RHC-Inspection
CBCP
FDLC
```

禁止将 `FDLC` 写成 `FDCL`。

耦合策略：

```text
Periodic Goal
Event-driven Goal
Full Coupling
```

其中：

```text
Full Coupling = FDLC 完整配置
```

### 2.3 统计规模

默认：

```text
8 seeds × 20 episodes
seed = 0...7
episode_id = 0...19
```

统计时：

- 先在每个 `方法–场景–seed` 内对 20 个 episode 求均值
- seed 是独立统计单元
- 不得把 160 个 episode 当成 160 个独立样本

### 2.4 配对公平性

同一 `scene_id + seed + episode_id` 下，所有方法共享：

- 相同 POI
- 相同弱链路区域
- 相同 UAV/GCS 初始位置
- 相同通信场
- 相同时变扰动
- 相同能量模型
- 相同终止条件
- 相同底层运动与最低安全层

只允许方法自身决策机制不同。

### 2.5 两个场景因素必须独立

退化强度只改变通信参数，不改变几何与 POI。

冲突等级只改变 POI 与弱链路区域的空间重合，不改变退化参数。

同一 seed、同一 M 下，low–severe 必须复用相同空间结构。

### 2.6 保持方案 B

- backlog 由快环管理
- 数据交付事件用于完成状态确认
- backlog 不进入慢环目标函数、约束或候选排序
- 不得临时加入 backlog-aware 慢环优化
- 结果解释不得声称 backlog 直接修改慢环目标函数

### 2.7 图像生成规范

所有用于论文的场景图、结果图、统计图和实时性分析图，
必须在生成前读取并遵循以下规范文件：

```text
/home/yuhe/workspace/uav-distributed/Autonomous_Robots_图片样式与制作规范.md
````

该文件是论文图片生成的唯一统一规范。所有新建或修改的绘图脚本必须遵守其中关于以下内容的要求：

* 单栏图与跨双栏图尺寸；
* PDF/EPS 矢量输出；
* 字体、字号、线宽和 marker；
* 方法颜色、线型和图例顺序；
* 灰度可辨识性；
* 子图编号和共享图例；
* 误差棒与置信区间；
* 坐标轴、单位和 caption；
* 文件名和导出分辨率。

生成图片前必须先检查规范文件是否存在。若文件不存在、无法读取或规则存在冲突，应停止生成论文图片并报告问题，不得自行猜测图片样式。

所有论文主图优先输出为 PDF 矢量图；位图仅在热力图或实物照片等确有必要时使用，并按照规范文件设置分辨率。

### 2.8 新增脚本命名规范

实验运行过程中新增的脚本必须带有 `paper1_v2` 标识，以区别旧实验脚本和其他论文代码。

统一采用文件名前缀：

```text
paper1_v2_
```

示例：

```text
paper1_v2_run_overall.py
paper1_v2_run_coupling.py
paper1_v2_run_timescale.py
paper1_v2_profile_runtime.py
paper1_v2_validate_results.py
paper1_v2_aggregate_metrics.py
paper1_v2_statistical_analysis.py
paper1_v2_plot_scene_matrix.py
paper1_v2_plot_overall_results.py
paper1_v2_generate_tables.py
```

适用范围：

* 新增的实验 runner；
* 场景生成与校验脚本；
* 完整性检查脚本；
* 指标汇总脚本；
* 统计分析脚本；
* 图像生成脚本；
* LaTeX 表格生成脚本；
* 数据迁移或结果修复脚本。

已有且经过验证的核心算法脚本无需仅为满足命名规则而重命名。
若必须修改已有脚本，应优先保留原文件，并新建带
`paper1_v2_` 前缀的入口或封装脚本调用原实现，避免破坏旧实验流程。

所有新增脚本均应在文件头注明：

```python
"""
Paper: paper1_v2
Purpose: <脚本用途>
Inputs: <输入目录或配置>
Outputs: <输出目录或文件>
"""

```
```

---

## 3. 运行前检查

先定位：

- 单 episode 入口
- 场景生成器
- 方法配置
- seed 设置
- 指标模块
- 事件日志
- 求解器计时
- 结果输出逻辑

优先复用现有代码。若缺少矩阵 runner，可新增统一批量入口，但不得改变算法行为。

Runner 至少支持：

```text
--stage
--method
--strategy
--scene-id
--rho
--seed
--episode-id
--output-root
--resume
```

每次运行记录：

```text
git commit
working-tree state
Python version
dependency versions
OS
CPU
memory
solver name/version
config hash
code hash
```

---

## 4. 结果目录

```text
results_v2/
├── configs/
├── manifests/
├── raw/
│   ├── overall/
│   ├── coupling/
│   ├── timescale/
│   └── runtime/
├── logs/
│   ├── runs/
│   ├── failures.jsonl
│   └── warnings.jsonl
├── summaries/
│   ├── episode_metrics.csv
│   ├── seed_summary.csv
│   ├── scene_summary.csv
│   ├── statistical_tests.csv
│   └── integrity_report.md
├── tables/
├── figures/
└── metadata/
```

每个 episode 保存：

```text
config.json
metrics.json
events.jsonl
timing.json
termination.json
run.log
```

唯一运行目录：

```text
results_v2/raw/<stage>/<method_or_strategy>/<scene_id>/rho_<rho>/seed_<seed>/episode_<id>/
```

---

## 5. Pilot 与场景校准

### 5.1 场景图

生成：

```text
results_v2/figures/c3_scene_matrix.pdf
```

图中展示：

- 期望综合链路质量
- POI
- GCS/起点
- 局部弱链路区域
- M0–M3
- low–severe

若色条仅为 `1-loss`，caption 必须明确它不是综合链路质量。

### 5.2 强度单调性

对同一 `seed + M`，检查 low–severe：

- 平均丢包率不下降
- 平均时延不下降
- 平均带宽不升高
- 平均综合链路质量不升高

### 5.3 冲突单调性

记录：

\[
\eta_{\mathrm{weak}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf 1(p_i\in\mathcal B_{\mathrm{weak}}).
\]

M0–M3 应总体单调增加。

### 5.4 Pilot

运行：

```text
2 seeds × 2 episodes
```

场景：

```text
c3_low_m0
c3_medium_m2
c3_high_m3
c3_severe_m3
```

方法：

```text
RHC-Inspection
FDLC
```

验收：

- 无崩溃、空指标、重复 run_id
- 不出现所有方法全为 0 或全为 1
- severe-M3 不应初始即全部失败
- low-M0 不应大面积不可行
- 终止原因、计时、事件日志完整

Pilot 未通过时，不得启动正式实验。

---

## 6. 阶段一：总体方法对比

方法：

```text
CDSL
WCDL
RHC-Inspection
CBCP
FDLC
```

配置：

```text
16 scenes
rho = 40
8 seeds ×  20 episodes
```

总运行量：

```text
5 × 16 × 8 × 20 = 12800 episodes
```

该阶段同时用于：

- 总体方法对比
- 通信退化强度敏感性
- 冲突等级敏感性
- Method × intensity × conflict 交互分析

规则：

- FDLC、rho=40 是唯一默认完整方法结果
- 后续 Full Coupling、rho=40 必须复用该批数据
- 不得重复运行并生成另一组默认数值

输出：

```text
overall_episode_metrics.csv
overall_seed_summary.csv
overall_scene_summary.csv
overall_rtask.csv
overall_diagnostics.csv
```

---

## 7. 阶段二：耦合机制消融

策略：

```text
Periodic Goal
Event-driven Goal
Full Coupling
```

配置：

```text
16 scenes
rho = 40
8 seeds × 20 episodes
```

结果复用：

- Full Coupling 直接复用阶段一 FDLC
- 只新增 Periodic Goal 和 Event-driven Goal

新增运行量：

```text
2 × 16 × 8 × 20 = 5120 episodes
```

禁止：

- 在该阶段同时做全场景 rho 扫描
- 改变通信、能量或安全层
- 重跑另一批 Full Coupling 默认结果

---

## 8. 阶段三：双时间尺度敏感性

代表性场景：

```text
c3_medium_m2
c3_high_m2
c3_high_m3
c3_severe_m3
```

策略：

```text
Periodic Goal
Full Coupling
```

扫描：

```text
rho ∈ {20, 40, 80, 100}
```

由于不是等距或对数等距，统计时将 rho 视为分类变量。

结果复用：

- Full Coupling、rho=40 复用阶段一
- Periodic Goal、rho=40 复用阶段二
- 只新增 rho=20、80、100

新增运行量：

```text
2 × 3 × 4 × 8 × 20 = 3840 episodes
```

必须记录：

```text
R_task
R_cov
P_del|cov
delivery latency mean/median/p90
replan total/periodic/event
event replan ratio
feedback count
return-home rate
OOB rate
energy-trigger rate
target-switch count
mode-switch count
```

若当前没有 `target-switch count`，必须补充，避免把高频重规划直接解释为目标抖动。

---

## 9. 阶段四：计算实时性与部署可行性

仅分析：

```text
FDLC
rho = 40
```

场景：

```text
c3_low_m0
c3_medium_m1
c3_high_m2
c3_severe_m3
```

优先复用阶段一日志。若计时字段不完整，再做专门 profiling。

专门 profiling 要求：

- 固定硬件
- 关闭调试器
- 关闭高频终端输出
- 固定进程数
- 固定求解器设置
- 保存环境信息

快环记录：

```text
mean
p95
p99
max
deadline-miss rate
```

慢环记录：

```text
mean
p95
p99
max
timeout rate
failure rate
mean/max MIP gap
candidate-node count
candidate-edge count
actual horizon
event-to-command latency
```

资源记录：

```text
CPU
peak memory
replans per episode
feedbacks per episode
```

单回合墙钟时间只作为实现成本参考，不作为在线实时性的主要证据。

---

## 10. 必须记录的字段

### 身份字段

```text
run_id
stage
method
strategy
scene_id
intensity
conflict
rho
seed
episode_id
git_commit
config_hash
scene_hash
```

### 任务字段

```text
n_poi_total
n_covered
n_delivered
n_pending
r_cov
p_del_given_cov
r_task
delivery_latency_mean_s
delivery_latency_median_s
delivery_latency_p90_s
pending_peak
pending_time_integral
covered_count / total_poi_count
delivered_count / covered_count
returned_episode_count / total_episode_count
```

`pending_*` 只做状态诊断，不解释为慢环优化输入。

### 通信字段

```text
loss_mean
loss_p95
latency_mean_ms
latency_p95_ms
bandwidth_mean
link_quality_mean
weak_region_time_s
```

### 安全字段

```text
returned_home
oob_triggered
energy_triggered
timeout_triggered
termination_reason
final_energy
min_return_energy_margin
```

不得把 `energy_triggered` 命名为 `energy_termination`，除非确实因能量耗尽终止。

### 协同字段

```text
replan_total
replan_periodic
replan_event
event_replan_ratio
feedback_total
feedback_periodic
feedback_event
target_switch_count
mode_switch_count
```

### 计算字段

```text
fast_time_mean_ms
fast_time_p95_ms
fast_time_p99_ms
fast_time_max_ms
slow_time_mean_ms
slow_time_p95_ms
slow_time_p99_ms
slow_time_max_ms
solver_timeout_count
solver_failure_count
mip_gap_mean
mip_gap_max
episode_wall_time_s
cpu_mean_percent
memory_peak_mb
```

---

## 11. Resume 与失败处理

唯一键：

```text
stage + method/strategy + scene_id + rho + seed + episode_id + config_hash
```

`--resume` 规则：

- 已成功且校验通过：跳过
- 失败或文件不完整：重跑
- 不重复追加相同 episode
- 不覆盖成功原始数据

失败自动重试最多 2 次。

失败日志必须包含：

```text
run_id
exception_type
message
traceback
retry_count
config_path
log_path
```

连续失败后标记为 failed，不参与统计，不得静默忽略。

---

## 12. 完整性检查

预期新增运行量：

```text
阶段一：8000
阶段二：3200
阶段三：2400
总计：13600 episodes
```

检查：

- 缺失组合
- 重复唯一键
- NaN/Inf
- 指标越界
- scene_hash 是否正确配对
- 同一 seed 的 low–severe 是否复用几何
- FDLC/Full Coupling/rho=40 是否复用同一结果
- 配置哈希是否一致
- 终止原因是否冲突
- 单位是否一致
- \(R_{\mathrm{task}}\approx R_{\mathrm{cov}}P_{\mathrm{del}\mid\mathrm{cov}}\) 是否成立

输出：

```text
results_v2/summaries/integrity_report.md
```

完整性检查失败时，不得生成论文表格。

---
````markdown
## 13. 统计分析协议

### 13.1 独立统计单元

随机种子是独立统计单元。

对于每个“方法/策略–场景–seed”组合，先对同一 seed 下的
10 个 episode 进行聚合，再使用 seed 级结果开展统计分析。

禁止：

- 将 100 个 episode 视为 100 个独立样本；
- 在 episode 层直接进行显著性检验；
- 在 bootstrap 中分别打乱不同方法的 seed 对应关系。

### 13.2 跨场景总体分析

总体方法对比采用包含以下固定效应的混合效应模型：

```text
Method × Degradation intensity × Conflict level
````

随机种子作为随机截距：

```text
(1 | Seed)
```

重点检验：

* Method 主效应；
* Method × Degradation intensity；
* Method × Conflict level；
* Method × Degradation intensity × Conflict level。

根据指标类型选择模型：

| 指标            | 推荐模型                    |
| ------------- | ----------------------- |
| 任务完成数、覆盖数、交付数 | 二项广义线性混合模型              |
| 比例指标          | 优先基于原始分子/分母采用二项 GLMM    |
| 数据交付时延        | 对数变换后的 LMM 或 Gamma GLMM |
| 重规划次数、反馈次数    | 负二项 GLMM                |
| 返航、越界、超时      | 二项 GLMM                 |
| 快环与慢环耗时       | 对数变换后的 LMM              |

若模型无法稳定收敛，应记录原因，并采用 seed 级非参数分析作为补充，
不得静默更换统计方法。

### 13.3 单场景方法比较

对于每个场景：

1. 以 seed 为区组，对全部方法执行 Friedman 检验；
2. 仅当总体差异显著时，执行预设 Wilcoxon 配对符号秩检验；
3. 在当前场景的预设比较族内采用 Holm 方法校正；
4. 报告配对均值差、95% 置信区间、校正后 p 值和效应量。

总体方法对比的预设方法对为：

```text
FDLC vs CDSL
FDLC vs WCDL
FDLC vs RHC-Inspection
FDLC vs CBCP
```

不执行与研究问题无关的全部两两比较。

### 13.4 耦合机制消融

预设比较为：

```text
Full Coupling vs Event-driven Goal
Event-driven Goal vs Periodic Goal
Full Coupling vs Periodic Goal
```

跨场景主模型包含：

```text
Coupling strategy × Degradation intensity × Conflict level
```

随机种子作为随机截距。

### 13.5 双时间尺度敏感性

时间尺度比：

```text
rho ∈ {20, 40, 80, 100}
```

在主分析中作为分类变量处理。

混合效应模型包含：

```text
Strategy × Rho × Scene
```

随机种子作为随机截距。

重点比较：

* 每个 rho 下 Full Coupling vs Periodic Goal；
* 同一策略下 rho=40 vs 20、80、100；
* Strategy × Rho 交互效应。

`log2(rho)` 及其二次项仅可作为探索性趋势分析，
不得作为主要显著性结论。

### 13.6 Bootstrap 与效应量

置信区间采用 seed 级配对 bootstrap：

* 重采样次数：5000；
* 推荐使用 BCa 95% 置信区间；
* 每次重采样以 seed 为单位；
* 同一次重采样中所有方法使用相同 seed 索引；
* 不对 episode 单独重采样。

Wilcoxon 配对比较同时报告秩二列相关系数：

```text
rank-biserial correlation
```

不得只报告 `p < 0.05`。

最终统计结果至少包含：

```text
comparison
mean_difference
ci_lower
ci_upper
raw_p
holm_adjusted_p
effect_size
n_seeds
model_type
convergence_status
```

显著性水平统一设置为 0.05。

```
```


先在 `方法–场景–seed` 层聚合 episode。

### 总体方法对比

分析：

```text
Method × Degradation intensity × Conflict level
```

seed 作为随机截距。

预设比较：

```text
FDLC vs CDSL
FDLC vs WCDL
FDLC vs RHC-Inspection
FDLC vs CBCP
```

场景内：

1. Friedman 检验
2. 预设 Wilcoxon 配对检验
3. Holm 校正
4. 报告均值差、95% bootstrap CI、校正后 p、秩二列相关效应量

### 耦合消融

预设比较：

```text
Full vs Event-driven
Event-driven vs Periodic
Full vs Periodic
```

### 时间尺度

将 rho 视为分类变量，分析：

```text
Strategy × rho × scene
```

重点比较：

- 每个 rho 下 Full vs Periodic
- 每个策略下 rho=40 vs 20/80/100

不得只报告 `p < 0.05`。

---

## 14. 自动生成结果

表格：

```text
overall_method_comparison.csv/.tex
coupling_ablation.csv/.tex
degradation_sensitivity.csv
conflict_sensitivity.csv
timescale_sensitivity.csv
runtime_feasibility.csv
statistical_summary.csv
```

图片：

```text
c3_scene_matrix.pdf
overall_heatmap_rtask.pdf
degradation_sensitivity_rtask.pdf
conflict_sensitivity_rtask.pdf
coupling_ablation.pdf
timescale_sensitivity.pdf
runtime_budget.pdf
```

图片要求：

- PDF 矢量
- 方法颜色、线型、marker 全文统一
- 灰度可辨
- 误差棒为 seed-level 95% CI
- 不显示未建模的 No-fly zones
- 色条与实际链路质量定义一致

---

## 15. 论文结果映射

```text
总体对比：
五种方法 × 16 场景
回答“是否有效、何时有效”

敏感性：
直接复用总体对比数据
不重复运行

耦合消融：
Periodic / Event-driven / Full，固定 rho=40
回答“为什么有效”

时间尺度：
代表性场景，Periodic vs Full
回答“时间尺度如何影响性能、事件补偿边界在哪里”

实时性：
FDLC、rho=40
回答“默认配置能否在线运行”
```

不再进行大篇幅通信–能量消融。

---

## 16. 最终验收

- [ ] 16 场景通过校准
- [ ] low–severe 长期统计总体单调
- [ ] M0–M3 冲突指标总体单调
- [ ] 阶段一完成 8000 个有效 episode
- [ ] 阶段二新增完成 3200 个有效 episode
- [ ] 阶段三新增完成 2400 个有效 episode
- [ ] 无重复唯一键
- [ ] 无静默失败
- [ ] 所有方法使用配对场景与扰动
- [ ] FDLC、Full Coupling、rho=40 完全复用
- [ ] episode、seed、scene 三层汇总均生成
- [ ] 统计以 seed 为独立单元
- [ ] 输出 CI、校正后 p 和效应量
- [ ] 快慢环计时完整
- [ ] 完整性报告通过
- [ ] 表格可由 CSV 自动重建
- [ ] 配置、环境和代码版本可追溯
- [ ] 所有 episode 已先聚合到 seed 层；
- [ ] seed 被作为独立统计单元；
- [ ] 比例指标保留原始分子与分母；
- [ ] 总体分析包含 Method × Intensity × Conflict；
- [ ] Friedman 仅用于单场景辅助分析；
- [ ] Holm 校正仅作用于预先定义的比较族；
- [ ] bootstrap 在 seed 层进行配对重采样；
- [ ] rho 在主分析中作为分类变量；
- [ ] 各指标使用与其分布匹配的模型；
- [ ] 统计输出包含模型收敛状态；
- [ ] 报告置信区间、校正后 p 值和效应量。
---

## 17. 执行顺序

```text
1. 检查入口和指标
2. 固化配置与版本
3. 生成并校验 16 场景
4. 运行 Pilot
5. 运行总体方法对比
6. 完整性检查
7. 运行耦合消融
8. 完整性检查
9. 运行时间尺度扫描
10. 完整性检查
11. 复用或补跑实时性 profiling
12. 生成 seed 级汇总
13. 统计检验
14. 生成论文表格与图片
15. 输出最终完整性报告
```

任一阶段未通过完整性检查时，不得进入下一阶段。
