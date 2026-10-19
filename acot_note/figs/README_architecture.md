# 架构图索引(2026-10-12 更新)

| 文件 | 状态 | 内容 |
|---|---|---|
| `fig_architecture_v4.{png,pdf,svg}` · 脚本 `fig_architecture_v4.py` | **当前模型(2026-10-12 代码核实)** | 与 v3 同一模型(属性开局 v1 + TAHRelM + PGIF + 匈牙利跟踪器,WHERE 原规则),按代码核实更正 v3 的 5 处出入(虚线框 + 「v4 更正」):① 承诺门限 0.5 → 当前 Ours 的 0.05;② EAR / IAR / DiT / grounding 头是 Stage 2 离线训练过的;③ EAR 无 Stage-1 预热,但每个 S2 tick 都采样,至少约 60% 的 S2 tick 是 DiT 的 z_ex;④ DiT 训练时 z_ex = 真值航点,EAR → DiT 耦合只在部署时存在;⑤ grounding 头在 TAH 记忆为空时作初选。EAR / WHERE 占比由脚本从当前 Ours 的闭环 ticks 现算。核实记录 `runs/arch_check/codex_arch_r1.md`;v4 经 codex 复核 FIGURE MATCHES CODE(`runs/arch_check/codex_arch_v4.md`) |
| `fig_closed_loop_{overview,tick,methods}.{png,svg}` · 源 `.dot`(graphviz) | **闭环评测代码调用流程(2026-10-12)** | 不是模型架构图,而是实现路径图:A 总体编排(调度器 / CARLA 客户端 / 策略服务端 / DAM4SAM 服务 / 分析)、B 策略服务端每 tick 调用链(带 `文件:行号`)、C 方法配置 → 开关字段 → 代码分支。说明文档 `acot_note/1012_closed_loop_impl_map.md` |
| `fig_architecture_v3.{png,pdf,svg}` · 脚本 `fig_architecture_v2.py hung` | **已被 v4 取代**(2026-10-06 的当前模型图;10-12 核实与代码有 5 处出入,见 v4) | ours = hung:属性开局 v1 + TAHRelM + PGIF(tempered + absent-hold)+ 匈牙利跟踪器，WHERE 为原规则(承诺 ID 无框时沿用 EAR 航点)。按 acot 运行时代码绘制，底部为跟踪器离线对比与 2×2 闭环结果(1001 §5.5c) |
| `fig_arch_overview.*` · 脚本 `fig_arch_overview.py` | **总览(所有方法)** | 所有臂共享的部分(感知 + 匈牙利跟踪器、属性开局模块、冻结前端、置信度、承诺、WHERE、DiT)与各臂不同的「WHICH 槽位」;**两种开局设定对照带**(绿 = 自然开局 / 部署设定，表 1、3(a);紫 = 强制共享开局 / 机制设定，表 2、3(b)、4、5);下方 21 臂配置表(由 methods/*.yaml 解析，标出每臂所属设定与表、T0 状态) |
| `fig_arch_ablations.*` · 脚本 `fig_arch_ablations.py` | 消融 | 参照链(ours)+ 表 1 族 A(w/o PGIF、w/o 属性开局)、表 2 H-P1–P5、3 个探索臂、U3 诊断，只标出改动的模块;每格有设定标签(自然 / 共享),开局块按设定着色;证据为 T0 pilot(@2)数字，其余旧协议证据只说明方向 |
| `fig_arch_baselines.*` · 脚本 `fig_arch_baselines.py` | 外部基线 | 顶部两种开局设定在基线上的用法(表 1 族 B / 3(a) 自然开局;表 3(b) 跨记忆强制共享开局);lang_xattn、ext_deepsort、ext_ocsort、ext_reid、ext_dam4sam、ext_oavat 六个面板(含统一交接与 T0 调参网格);更新后的公平性说明 |
| `fig_architecture_v2.{png,pdf,svg}` · 脚本 `fig_architecture_v2.py`(默认参数) | **两处改动都打开(cv_hung)** | 按 acot 运行时代码绘制:属性开局 v1 + TAHRelM + PGIF(tempered + absent-hold),加两处改动——跟踪器匹配改为匈牙利、承诺目标无匹配框时 WHERE 匀速外推(= 实验臂 `ab_where_assign/cv_hung`)。补画了 EAR、置信度(top1 - top2 差值)与跟踪器 ID 的记账路径。底部为改动 ① 的离线回放数字和 2×2 闭环结果。结论：建议采用匈牙利(臂 hung),外推暂不并入(见 1001 §5.5c) |
| `fig_architecture_v1.{png,pdf,svg}` · 脚本 `fig_architecture_v1.py` | **10-03 最终模型** | 属性开局 v1(`OPENING=attr`、`HIRES=3`)+ TAHRelM(`runs/tah_relm_dag.pt`)+ PGIF(tempered + absent-hold),部署设定 v2(跟踪器 ID + 射线遮挡过滤)。图中披露两项理想化感知输入：候选集仅含任务车辆;WHERE 使用仿真真值世界坐标。底部参考面板为开局变体 v2–v4 的对比。数字由脚本从评测文件读取 |
| `fig_architecture_legacy_pgif.*`(原 `fig_architecture.*`) | **LEGACY** | TAHRelM + PGIF,旧评测设定：仿真理想 tracklet ID、不过滤遮挡、语言头开局、PGIF 无 absent-hold、WHERE 始终使用仿真世界坐标。结果表来自 10-01 PoC 批次(0.613 / 30.3 s;同配置 10-02 重跑为 0.576 / 21.3 s) |
| `fig_architecture_legacy_tahrelm.*`(原 `fig_architecture_v1_tahrelm.*`,即 09-07 时的 `fig_architecture`) | **LEGACY** | 仅 TAHRelM,无 PGIF,同为旧设定 |

两张 legacy 图上都有「LEGACY」水印和横幅。旧设定与新设定之间逐步演进的同种子对比见 `carla_uav_tracking/rollout/arch_compare.py`:S1 旧图模型 → S5 新图模型，并拆分出每一步的影响。

**fig_architecture_v1 与代码不符的三处(10-06 核对 acot 运行时时发现，图未改，以 v2 图为准)**:① 跟踪器标为「匈牙利」,实际所有 v1 运行都是贪心(策略镜像无 scipy);② WHERE 写「承诺 ID 不可见时按匀速估计滑行」,实际不覆盖 z_ex,沿用 EAR 航点;③ 图中没有 EAR 模块。

共享绘图函数在 `_archlib.py`(配色同 v1/v2;10-07 新增紫色 = 强制共享开局，与 `tag()` 标签)。三张新图中的「当前协议」= `sim_proxy_v2@3`(匈牙利跟踪器 + 统一开局交接 + CARLA 环境确定性);外部基线的调参网格在该协议下排队中，消融在 T0 只跑了 pilot(w/o PGIF、w/o 属性开局)。

**10-07 更新要点**:① 标出两种开局设定(1006 §3.4);② 外部基线的开局交接改为统一交接(此前图中写「只在第一个 S2 tick 读开局选择」,@2 起已改);③ 补 re-ID、OA-VAT 面板，删去 UAST(D2);④ 贝叶斯变体注明该分支没有缺席输出(`pgif.py`,配置里的 absent_hold 不生效);⑤ 探索臂 w/o 承诺门控 / w/o 强制置信已实现。
