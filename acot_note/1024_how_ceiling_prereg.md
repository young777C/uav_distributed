# 1024 HOW 上限阶梯：预注册稿(codex R1 后修订)

2026-10-24 起草，10-25 按 codex R1 修订。动机见 memory `acot-uav-how-ceiling-1024`。上游设计是 `1023_joint_which_where_workflow.md` §3 的 WHERE 第 0 步，本稿把它收窄为最小阶梯。**数据出来之前定稿，判定规则不再修改。**

## 0. 要回答的问题

身份完美(U1)时，自主飞行的 TR 只有 0.247;专家回放的近似值 TR90ˣ 约 0.76。这段飞行差距里，各有多少来自：
- 目标状态(WHERE);
- 航点的更新节奏;
- DiT 控制器本身?

现有的专家飞行同时换掉了信息(全知真值，10 Hz)、控制器(PID)和取景方式，压力场景里还执行了 E3,所以分不开。

**口径(codex R1 MAJOR 1)**:下面的差值是**按顺序叠加的干预差**,不是纯因果分解。R2 是**经验上的专家参照**,不是"只剩控制器"的上界:它还带着专家自己的取景方式、动作噪声和设定点接口。

## 1. 四个臂(dev-S 自主飞行，同种子配对，都不执行 E3)

| 臂 | 方法配置 | 目标状态 → z_ex | 控制器 |
|---|---|---|---|
| **R0** | `diag_oracle_head_cv` | 现行共享栈：承诺车有框时用它的真值坐标，缺席时用 CV 外推(同 `ours_cv`) | DiT |
| **R1** | `diag_oracle_head_gtwhere` | 每个 S2 tick(0.6 s)都用**目标**真值位置更新 CV 估计器，可见与否都一样;航点在 S2 tick 算好，沿用到下一个 S2 tick | DiT |
| **R1f** | `diag_oracle_head_gtwhere_f` | 同 R1,但**每帧**用当前位姿、当前时刻重新投影航点(`wp_every_frame`)。DiT 训练时看到的正是这种"当前帧航点" | DiT |
| **R2** | `diag_oracle_head_expert_noe3` | 全知真值，每帧更新 | 专家 PID(`expert_look_away: false`) |

身份都用 U1,承诺和计分照常运行。

**差值的含义**
- G_where = R1 − R0:目标状态，**也包括关联错误**。R1 飞向真目标，即使承诺被跟踪器转给了干扰车。
- G_cad = R1f − R1:航点更新节奏和参考系过时。这一段有可部署的修法：每帧重投影不需要真值。
- G_ref = R2 − R1f:与专家参照之间的剩余差距。它不只是 DiT 策略，还包括(codex R2):
  - 设定点接口(取景、站位);
  - 残余的状态和预测差异:R1f 的位置仍按 S2 采样、CV 外推最多 0.5 s,专家每帧拿当前真值;
  - 专家自己的动作噪声。
- G_how = R2 − R1 = G_cad + G_ref。
- 另报 total = R2 − R0。

**实验配置**:`how_ladder_devS.yaml`,protocol `sim_proxy_v2@4`,种子集 `stress_devS`(40 个，94423 目标静止按协议跳过，实际 39 局)。`collect` 含 `record_act`,`record_act_ticks: 1`,只记 step 0 的 act 消息。

## 2. 指标

- **主指标**:TR(`episode.json` 的 `tr90`)。按种子配对求差，10,000 次 bootstrap 求 95% CI,另报符号翻转 p。
- **完整名单(codex R2 MAJOR 1)**:判定只在登记名单(`experiment.resolved.json` 的 seeds)完整时进行。完整指每个种子要么在每个臂都 complete 且 `tr90` 为有限值，要么在所有臂都因 "target stationary" 跳过。其他情况(缺失、失败、格式错误、只在部分臂跳过)一律**不判定**,只能 `--peek` 查看，输出标明"不是判定"。描述性数据读不出来，不影响主数据。
- **描述性(不参与判定)**:M1、画面内时间占比、截断局数、与目标距离中位数、目标框尺寸(中位数 / 下四分位)、key 抖动率、出画归因(`flight_exit_attr.py` 口径)、开局 12 s 内"有框仍出画"的局数。

## 3. 判定规则(codex R1 MAJOR 2)

- **跑满全部配对种子，不提前停止**,不做中途检查。
- 每个差值分三档:
  - 点估计 ≥ 0.15 且 95% CI 下界 > 0.05 → **大**;
  - 95% CI 上界 < 0.10 → **小**;
  - 其余 → **不确定**。
- 解读(按顺序叠加的干预差，不做单一归因;codex R2 MAJOR 2):
  - **G_how 大** → 在目标状态已用真值之后，闭环控制栈仍有大段差距。论文把它写进上限分解表和局限一节，并作为"身份放在回放里评"的依据。至于差距在哪一段，看下面 G_cad 和 G_ref 的拆分。
  - **G_cad 大(或点估计 ≥ 0.05 且 CI 下界 > 0)** → 航点更新节奏是可修的一段。值得在可部署臂上试每帧重投影，那是另一个实验，要另行预注册。
  - **G_ref 大** → 剩余差距在 DiT 策略、设定点接口或残余状态差异上，本实验分不开这三者。CVPR 版不重训 DiT,除非用户另行决定。
  - **G_where 大** → 值得调整缺席时的 WHERE 和搜索策略(F 线)。F 线的价值只由 G_where(以及可部署的 G_cad)判断，**不由 G_how 推出**。
  - **G_how 小** → 控制栈不是主墙。
  - 不确定 → 只描述性报告，不据此追加投入。
- 阈值 0.15 / 0.10 沿用第 0 步。

## 4. 干净种子(codex R1 MAJOR 3,结果盲)

- 定义：读每个臂的 step 0 act 消息(`act_record.pkl` 中第一条 `cmd == "act"` 的消息，必须 `step == 0`,形状正确且都是有限值)。四个臂**两两比较**都满足以下条件，才算干净种子(codex R2 MINOR 1、2):
  - 目标、每辆干扰车、相机位置的每个坐标分量相差 ≤ 1 cm;
  - 相机 pitch 和 yaw 的环绕角差 ≤ 0.01°。
- 干净种子集只由这些字段决定，不看任何结果。
- 某臂缺少有效的 step 0 记录时，该种子只从干净子集里剔除，仍保留在主分析中。
- 全部配对种子和干净子集两个口径都报。判定以全部配对种子为准，干净子集作为敏感性分析。

## 5. 事先写下的预期(不参与判定)

零 GPU 预检(dev-S 已有日志):各臂约有 11–12/39 局，在开局 12 s 内"目标有框、承诺在目标上"时出画(此时航点已由真值驱动),而且基本是同一批种子。
- 预期 R1 救不回这部分。
- 如果原因是航点在 S2 之间过时，R1f 能救回一部分;R2 应能救回。
- 缺席时的 WHERE 决定出画后能否找回:U1(EAR)8/11 没有回来,Ours(CV)1/12。

## 6. 公平性与特权边界

- 四个臂都是诊断臂，不进主表的方法行，表里用灰色标出。
- R1/R1f 只通过 `OracleHandle.true_world("ex_source_gt")` 读目标真值。只有 `ex_source: oracle` 的臂由感知模块填 `ScoringInfo.target_world`,其他臂不增加任何真值读取。真值缺失或不是有限值时，该局直接报错失败，不静默退回 EAR/CV(codex R1 MINOR 1)。
- 现有 `stress_devS_selfflight/u1`(EAR,TR 0.247)可作参照，但不参与判定。

## 7. 代码改动(全部默认不生效)

| 文件 | 改动 |
|---|---|
| `acot/runtime/io.py` | `ScoringInfo.target_world`(默认 None);`OracleHandle.true_world(purpose)`,受声明约束并计数 |
| `acot/runtime/perception.py` | 构造参数 `target_world`(默认关);开启时每个 S2 tick 把 `tpos` 放进标签 |
| `acot/runtime/server.py` | 只有 `ex_source == oracle` 时开启上面这个参数 |
| `acot/runtime/policy.py` | `_where(inp, oracle)`:oracle 时用真值更新估计器并覆盖 z_ex,缺失则报错;`wp_every_frame` 时在非 S2 帧按当前位姿重新投影由估计器给出的航点 |
| `acot/runtime/support.py` | acot 运行时支持 `ex_source ∈ {track, oracle}`;`oracle` 和 `wp_every_frame` 都要求由 DiT 飞 |
| `acot/config/schema.py` / `resolve.py` | `ex_source` 新增 `oracle`,归入 GT 通道 `ex_source_gt`;可选键 `expert_look_away` 和 `wp_every_frame` |
| `acot/eval/scheduler.py`、`acot_client/episode.py`、`rollout/env.py` | 把 `expert_look_away`(默认 true)传到 env;为 false 时不执行 E3,保留 25 m 质心取景，允许按真值找回 |
| `acot/tests/test_how_ladder.py` | 新增 11 个测试 |

- 开发在会话临时目录的代码镜像里完成，没有动实时仓库。
- 配置哈希：补丁前后 54 个实验中 53 个逐臂一致，剩下一个前后都因缺 G\* 方法文件而报同样的错。
- **落地方式**(`runs/how_ladder/apply_and_launch.sh`,codex R1–R4):
  - 全程持有 `carla.lock`(fd 9),并把它直接交给启动的调度器，中途不释放;
  - 三方合并，冲突即中止;所有文件先暂存，写入前复核实时文件的哈希;
  - 先写成同目录的临时文件，再按"被调用方 → 调用方 → 新配置"的顺序逐个原子改名，整个安装只有几毫秒;
  - 测试在安装**之前**进行：在一份实时代码副本上叠加合并后的文件来跑，测试不通过就中止，实时仓库不动;
  - 只有改名失败时才回滚(同样按改名操作)。回滚时如果有进程正在读实时仓库，就推迟回滚。
- **剩余风险(codex R4 MAJOR,已知并接受)**:本项目有些启动入口不经过这把锁(`runs/stress/record_g3.sh`、`runs/mot_baselines/chain_grid.sh`、`runs/where_cv/chain_look.sh`、autotune 回放 worker)。对策：
  - 落补丁前和改名前各扫描一次，只要有本项目读实时仓库的 python 进程(调度器、策略服务器、CARLA 客户端、回放 worker)就中止;
  - 剩下的竞态窗口只有改名的几毫秒;
  - 不改共享 CLI 去强制加锁：现有的 `flock … acot eval` 包装再在内部加锁会自锁死锁。
  - 落补丁前先和其他会话打招呼。

## 8. 资源与排队

- OA-VAT 闭环网格占用 GPU 0/1/2/3/5/6(6 卡上限)。本实验在 `carla.lock` 上排队，用同样 6 个 worker,每卡 1 个 CARLA + 1 个策略。
- 156 局，按近期自主飞行实测每局约 25–30 min(专家臂更快),约 10–13 h。

## 9. 状态

- [x] 配置与代码(镜像)、单元测试 11/11
- [x] codex R1:MAJOR ISSUES,三条 MAJOR 和两条 MINOR 已按上文修订
- [x] codex R2:MAJOR ISSUES(代码通过;问题在分析脚本的名单校验、解读措辞和干净种子判定),已修订
- [x] 实时仓库的 `policy.py` 和 `schema.py` 在 10-24 23:55 被另一会话改过(DAM4SAM 逐帧推进、`oavat_full.dp`),与本补丁不重叠;启动脚本改为三方合并，有冲突即中止
- [x] 感知模块和策略新增属性改用 `getattr` 带默认值读取：有测试和回放工具用 `__new__` 构造对象，不经过 `__init__`
- [x] codex R3:MAJOR ISSUES(启动脚本锁、名单被过滤、隔离局),已修;代码与合并通过
- [x] codex R4:剩一条 MAJOR(不走锁的启动入口),按上面"剩余风险"处理
- [x] codex R5:剩一条 MAJOR(测试失败时的回滚路径),已改为先在副本上测试再安装
- [x] codex R6:**NO MAJOR ISSUES**;三条 MINOR 已修。演练(暂存 + 在副本上测试):80 个测试通过，实验可以解析
- [x] 完整测试：补丁前 4 个已有失败 / 181 个通过;补丁后同样这 4 个失败 / 192 个通过(新增 11 个)
- [ ] 补丁落到实时仓库(空闲时;先核对原文件哈希没被别人改过)
- [ ] 开跑 → 跑满 → 判定
