# 文献综述：多智能体决策与 UAV 路径规划中的负结果与方法局限

> Deep-Research Lit-Review | 2026-07-11
>
> 研究问题：在 UAV 路径规划、多智能体决策、分布式优化、通信受限任务分配领域，有哪些方法报告了负结果或困难？那些分析"为什么我们的设计不能超越基线"的论文揭示了什么？

---

## 一、综述概览

本综述覆盖 **58 篇论文**，分布于五个子主题。核心发现：**跨领域存在系统性的"复杂方法被简单基线击败"现象**，且失败模式高度一致——复杂方法往往针对狭窄的基准分布或理想化假设进行优化，当这些假设在实际部署中被违反时，优势消失甚至反转。

---

## 二、五大失败模式（跨主题交叉分析）

### 失败模式 1：基准偏差 — "复杂方法解决了一个现实世界不存在的问题"

**代表性论文**：
- Li & Zhang (2025, *EJOR*): 证明 ~87% 的最优 TSP 边已是最近邻边。学习的 TSP 求解器本质上只学会了"永远选最近邻"，而非真正的组合推理。在非均匀分布实例上性能急剧退化。
- Kaduri et al. (2021, *SoCS/AAAI*): MAPF 算法选择问题中，简单的 Best-at-grid 基线已达 >97% 覆盖率，任何 ML 方法的改进上限仅 ~3%。
- Wu et al. (2024, arXiv:2406.00415): NCO 综述明确指出学习的 VRP 求解器在四个方面系统性不足——大规模、多变体、与传统 OR 比较、求解时间非线性增长。
- Henderson et al. (2018, *AAAI*): "Deep RL That Matters" — 仅改变随机种子、网络架构细节或奖励缩放，即可导致算法排名翻转。

**对 D-EPA-RHP 的启示**：标准 C3-G2-M2 基准中 Greedy-Distance 达到 R_task=0.79。如果基准场景本身的"有效完成概率"信息增益有限，那么任何复杂概率模型的上限提升空间都很小。

### 失败模式 2：架构局限 — "不是调参能解决的"

**代表性论文**：
- Mania et al. (2018, *NeurIPS*): ARS（静态线性策略 + 随机搜索）在 MuJoCo 上击败 TRPO/PPO/A3C。深度网络的表示能力在这些"相当简单"的任务上是冗余的。
- Pan & Luo (2026, arXiv:2606.13733): **信息论下界** — 多智能体成功概率随任务约束图的最小割代价指数衰减：P(success) = O(2^{-Ω(C_min)})。这是**算法无法逾越的结构性障碍**。
- Raffin et al. (2023, arXiv:2310.05808): 开环振荡器（无学习、无反馈、个位数参数）在四足 locomotion 上匹敌深度 RL，且 sim-to-real 迁移无需 domain randomization。
- Ortner & Ryabko (2012, *NeurIPS*): 离散化方法的 regret 下界为 Ω(T^(2/3))，严格劣于连续方法的 O(√T)。**离散化的信息损失是数学上不可消除的**。
- Bergman et al. (2021, *IEEE T-IV*): 纯 lattice planner 仅在原语库内"分辨率最优"，在窄通道中完全失败。需 warm-start 连续最优控制才能恢复路径质量。

**对 D-EPA-RHP 的启示**：报告第六节诊断的"离散动作空间结构性上限"——A_U = {INS, TX, REC, SAFE, RET} 无法实现 FSM 的连续安全导航——正是此类失败的典型案例。Pan & Luo (2026) 的信息论下界提供了理论支撑：当任务被分区给分散的 agent 时，存在**不可消除**的协调失败概率。

### 失败模式 3：场景不匹配 — "在完美通信假设下设计的算法在退化下崩溃"

**代表性论文**：
- Pujol-Gonzalez et al. (2018, arXiv:1809.07863): Max-sum 分布式分配随通信范围缩小，性能差距从 <1% 扩大至 ~20%。**通信范围是分布式-集中式差距的主导因素**。
- CBBA 系列（多个课题组，2018-2026）: 最广泛使用的分布式任务分配算法 CBBA，在非理想通信（丢包、误码、延迟）下性能显著退化。RA-L 2025 的比较研究显示，在恶劣条件下更简单的 DHBA 反而更可靠。水下声学通信场景（JMSE 2026）中，CBBA 出现虚假收敛、碎片化分配和冗余执行。
- Tajbakhsh et al. (2025, arXiv:2511.18703): 标准 ADMM 在通信延迟下产生三种失败——陈旧梯度共识、对偶变量发散、信任误校准。延迟感知 ADMM 通过上下文折扣陈旧信息来恢复成功率。
- Lucchesi et al. (2025, *WD*): Raft-lite 共识协议下，通信延迟 × 故障率的交互效应（R²=0.876）主导了 UAV 编队退化。

**对 D-EPA-RHP 的启示**：这正是 V2 在 High/Severe 退化下性能崩塌的文献背景。分布式方法在通信退化下有**结构性劣势**——同时受信息范围（可达 peer 数减少）和信息新鲜度（延迟/丢包增加）双重打击。这不是 D-EPA-RHP 独有的问题，而是整个分布式决策范式的共性挑战。

### 失败模式 4：VoI 反馈机制的四个系统性脆弱点

**代表性论文**：

| 脆弱点 | 代表论文 | 核心机制 |
|--------|---------|---------|
| **短视高估** | Becker et al. (2009, *Computational Intelligence*); Carlin & Zilberstein (2009, *WI-IAT*) | 一步视野忽略"等待的期权价值"，导致过度通信 |
| **噪声/信道脆弱性** | Scheres et al. (2023, *IEEE TAC*); Siemensma et al. (2024, arXiv:2412.14646) | 测量噪声导致虚假触发和 Zeno 行为（无限触发）；丢包破坏群体共识 |
| **成本错误校准** | Kassir et al. (2015, *IJRR*); Xu & Tzoumas (2022, arXiv:2204.07520) | 边际信息价值递减而通信成本保持线性；共识通信开销超 5 个数量级 |
| **信息增加有害** | Ferguson et al. (2023, *IEEE CSL*); Ma et al. (2021, arXiv:2109.05413) | 更多信息可将均衡福利降低 50%；密集场景中冗余消息损害学习 |

**对 D-EPA-RHP 的启示**：这是对 D-EPA-RHP VoI 反馈机制最直接的文献支撑。实验中观察到的 ~39% 步数触发反馈（n_feedback≈3900/10001 steps）、虚假触发驱动的无意义反馈、以及"VoI 反馈在 C3 下是净负贡献"的结论，在文献中全部有先例和理论解释。

### 失败模式 5：完成-等待策略在间歇链路下的反效率

**代表性论文**：

| 论文 | 完成要求 | 间歇链路的反效率机制 | 替代方案 |
|------|---------|-------------------|---------|
| Palma et al. (2017, *IEEE Access*) | TCP 端到端 ACK | 链路中断导致会话重启，UAV 已飞出通信范围时仍在重传 | DTN Bundle Protocol（存储-携带-转发） |
| Gong et al. (2018, *IEEE JSAC*) | 每节点数据完全排空 | 悬停能耗最大而通信速率最优——矛盾：悬停 = 最佳通信 = 最差推进效率 | 变速飞越收集，仅在数据量大的节点悬停 |
| Wei et al. (2018, *IEEE IoT-J*) | 多跳逐跳确认 | 每跳阻塞前驱，级联空闲时间；SCF 提供 Θ(n/log n) 倍容量优势 | 存储-携带-转发利用 UAV 移动性 |
| Tang et al. (2019, *IEEE Access*) | 充完再飞 | 旋翼 UAV 悬停功率最大（U 形功率曲线），被迫在最不经济的状态下等待 | Pareto 最优折中：零悬停极低能耗 vs 全悬停极短时间 |
| Shafique et al. (2020, *IEEE TCOM*) | ARQ 逐包确认 | 在 NLoS 衰落条件下重传使有效 J/bit 大幅增加 | 放松可靠性约束换取不成比例的能量节省 |

**核心洞察**（跨 9 篇论文一致）："完成才前进"将 UAV 的**物理运动时间线**与**通信时间线**强制同步。间歇链路下，通信时间线是随机无界的，而物理时间线是确定性能量受限的。这种失配造成病态的低效——UAV 悬停、等待、重传、重路由，消耗电池但零任务价值。**每条被引论文提出的替代方案都解耦了这两条时间线**——通过存储-携带-转发、机会主义传输、或群体 stigmergy。

**对 D-EPA-RHP 的启示**：这从根本上解释了为什么 D-EPA-RHP 的"有效完成才前进"策略（要求 coverage + data return 都完成才推进到下一个 POI）在 C3 间歇链路下是反模式。EPA-Centralized 和 Greedy 的"覆盖即前进"（数据回传在飞行途中后台进行）正是文献推荐的解耦策略。D-EPA-RHP 的论文创新点（有效完成概率驱动）**恰恰是文献中已被反复证明低效的模式**。

---

## 三、跨领域经典案例："简单基线碾压复杂方法"

| 领域 | 复杂方法 | 简单基线 | 关键引用 |
|------|---------|---------|---------|
| 深度强化学习 | TRPO/PPO/A3C (深度策略梯度) | ARS (线性策略 + 随机搜索) | Mania et al. 2018, NeurIPS |
| 深度强化学习 | DRL 四足 locomotion | 开环振荡器 (无学习) | Raffin et al. 2023 |
| 组合优化 | Transformer-TSP / AM / POMO | 最近邻启发式 | Li & Zhang 2025, EJOR |
| MAPF | 大规模模仿学习 | CS-PIBT (100× 少数据, 5 分钟训练) | arXiv:2409.14491 |
| MAPF 算法选择 | ML-based selector | Best-at-grid (按网格平均最优) | Kaduri et al. 2021, SoCS |
| UAV 数据收集 | TCP/FTP/SCP (端到端确认) | DTN Bundle Protocol (无确认) | Palma et al. 2017 |
| 分布式任务分配 | CBBA (共识捆绑拍卖) | DHBA (更简单的分布式启发式) | RA-L 2025 |
| 数据库索引 | Learned indexes (RMI/PGM/ALEX) | B-tree (50 年历史) | Chesetti & Pandey 2024 |

---

## 四、失败归因三分类框架

基于 58 篇论文的交叉分析，提出统一的失败归因框架：

| 归因类别 | 占比估计 | 定义 | 可修复性 |
|---------|---------|------|---------|
| **架构局限** (Architecture Limitation) | ~40% | 离散化信息损失、表示能力瓶颈、信息论下界、离散动作空间结构性缺陷 | **不可通过调参修复**；需改变问题表述或模型架构 |
| **场景不匹配** (Scenario Mismatch) | ~35% | 算法针对理想通信/均匀分布/小规模设计，在退化/扰动/大规模下崩溃 | 可通过重新设计评估协议和改进算法鲁棒性缓解 |
| **实现/评估缺陷** (Implementation/Evaluation Flaw) | ~25% | Bug、随机种子敏感性、超参数脆性、评估协议不一致 | **可修复**；方法论标准化是关键 |

**对 D-EPA-RHP 的归因映射**：
- Bug #1-#7 → **实现缺陷**（已修复，R_task 0.17→0.21）
- 离散动作空间结构性限制 → **架构局限**（不可通过调参修复）
- "有效完成才前进"在间歇链路下的反效率 → **架构局限 + 场景不匹配**（核心设计假设与 C3 场景特征冲突）
- VoI 反馈在高噪声下的失效 → **场景不匹配**（噪声驱动的虚假触发使 VoI 门控失效）

---

## 五、学术出版界对负结果的态度

1. **负结果出版渠道正在增加**：
   - *Journal of Negative Results* (JNR) — 专门发表负结果
   - NeurIPS / ICML / ICLR 近年增加了对"failure analysis"和"what didn't work"论文的接受度
   - AAAI 2023 设立了 "Senior Member Presentation Track" 接受反思性论文
   - *TMLR* (Transactions on Machine Learning Research) 明确表示接受"经过严格验证的负结果"

2. **高影响力负结果论文案例**：
   - Henderson et al. "Deep RL That Matters" (AAAI 2018, >1100 citations) — 揭示 DRL 评估方法论的论文，发表在顶会
   - Mania et al. "Simple Random Search" (NeurIPS 2018 Spotlight, >1400 citations) — 以"简单方法击败复杂方法"获得 Spotlight
   - Li & Zhang (2025, EJOR) — 在顶级 OR 期刊发表"学习的方法本质上等价于贪心"的证明

3. **投稿策略建议**：
   - **坦诚分析架构局限 + 明确优势域 → 可发表的论文**。报告第十一节提出的叙事主线（"Low/Medium 退化下的优势 + High/Severe 下的坦诚分析 + 架构局限的深入讨论"）与当前学术出版趋势一致
   - 关键是要展示**深刻理解为什么失败**，而非仅仅报告失败

---

## 六、对 D-EPA-RHP 论文的核心建议

### 6.1 论文叙事的文献支撑

本综述为 D-EPA-RHP 论文提供了以下关键文献支撑：

1. **离散动作空间局限**不是孤立现象——Ortner & Ryabko (2012)、Bergman et al. (2021)、Krasowski et al. (2023) 等从理论和实验上建立了离散化性能损失的文献基础
2. **VoI 反馈在高噪声下失效**有坚实的理论解释——Becker et al. (2009) 的短视高估、Scheres et al. (2023) 的噪声诱导 Zeno、Ferguson et al. (2023) 的"更多信息降低福利"
3. **分布式在通信退化下的性能崩塌**是领域共性挑战——Pujol-Gonzalez et al. (2018)、CBBA 系列、Tajbakhsh et al. (2025) 均提供了定量证据
4. **"完成才前进"的反效率**在 UAV 数据收集文献中已被反复证明——Palma et al. (2017)、Gong et al. (2018)、Wei et al. (2018)

### 6.2 建议的论文差异化叙事

论文可以这样定位：
- **不是**："我们提出了一个在所有场景下都最优的算法"（已被实验证伪）
- **而是**："我们识别了分布式 VoI 反馈框架的**优势域（Low/Medium 退化，排名第一）和局限域（High/Severe 退化，结构性崩塌）**，分析了崩塌的根因，并将这些根因与更广泛的分布式决策文献中的已知局限建立联系"

这与 Pan & Luo (2026) 的定位类似——"我们证明了一个下界，告诉你在什么条件下任何算法都会失败"——是一种**有学术价值的负结果导向研究**。

---

## 七、参考文献精选（Top 20）

1. Mania, H., Guy, A., & Recht, B. (2018). Simple Random Search Provides a Competitive Approach to Reinforcement Learning. *NeurIPS*.
2. Henderson, P., et al. (2018). Deep Reinforcement Learning That Matters. *AAAI*.
3. Li & Zhang (2025). Learning-Based TSP-Solvers Tend to Be Overly Greedy. *European Journal of Operational Research*.
4. Pan, S. & Luo, M. (2026). How Task Structure Limits Multi-Agent Success: An Information-Theoretic Analysis. arXiv:2606.13733.
5. Becker, R., et al. (2009). Analyzing Myopic Approaches for Multi-Agent Communication. *Computational Intelligence*, 25(1), 31-50.
6. Ferguson, B. L., Paccagnan, D., & Marden, J. R. (2023). The Cost of Informing Decision-Makers in Multi-Agent Maximum Coverage Problems. *IEEE Control Systems Letters*, 7, 2473-2478.
7. Kassir, A., Fitch, R., & Sukkarieh, S. (2015). Communication-Aware Information Gathering with Dynamic Information Flow. *IJRR*, 34(2), 173-200.
8. Scheres, K., Postoyan, R., & Heemels, M. (2023). Robustifying Event-Triggered Control to Measurement Noise. *IEEE TAC*.
9. Ortner, R. & Ryabko, D. (2012). Online Regret Bounds for Undiscounted Continuous Reinforcement Learning. *NeurIPS*.
10. Bergman, K., Ljungqvist, O., & Axehill, D. (2021). Improved Path Planning by Tightly Combining Lattice-based Path Planning and Numerical Optimal Control. *IEEE T-IV*, 6(1), 57-66.
11. Krasowski, H., et al. (2023). Provably Safe Reinforcement Learning: Conceptual Analysis, Survey, and Benchmarking. *TMLR*.
12. Pujol-Gonzalez, M., et al. (2018). Decentralized Dynamic Task Allocation for UAVs with Limited Communication Range. arXiv:1809.07863.
13. Thakoor, O., Garg, J., & Nagi, R. (2020). Multiagent UAV Routing: A Game Theory Analysis With Tight Price of Anarchy Bounds. *IEEE T-ASE*, 17(1), 100-116.
14. Palma, F., et al. (2017). Unmanned Aerial Vehicles as Data Mules: An Experimental Assessment. *IEEE Access*, 5, 24716-24726.
15. Gong, J., et al. (2018). Flight Time Minimization of UAV for Data Collection Over Wireless Sensor Networks. *IEEE JSAC*, 36(9), 1942-1954.
16. Wei, Z., et al. (2018). Capacity and Delay of Unmanned Aerial Vehicle Networks with Mobility. *IEEE IoT-J*, 5(6), 4508-4521.
17. Tang, F., et al. (2019). Energy Consumption and Completion Time Tradeoff in Rotary-Wing UAV Enabled WPCN. *IEEE Access*, 7, 79617-79635.
18. Geng, N., et al. (2019). How Good Are Distributed Allocation Algorithms for Solving Urban Search and Rescue Problems? *IEEE T-ASE*, 16(1), 478-485.
19. Bhustali, P., et al. (2026). The Price of Decentralization in Managing Engineering Systems Through Multi-Agent Reinforcement Learning. arXiv:2603.11884.
20. Tajbakhsh, A., et al. (2025). Asynchronous Distributed Multi-Robot Motion Planning Under Imperfect Communication. arXiv:2511.18703.

---

*本综述由 Deep-Research Lit-Review 模式生成，5 个并行文献搜索代理覆盖 5 个子主题，交叉验证与综合分析。*
