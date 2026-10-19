**Verdict: MAJOR**

这份预注册稿方向是对的，尤其是把“开局彩票”作为首要威胁来控制，这一点很审稿友好。但按 T-RO / RA-L / IJRR / CoRL 标准，目前还不能直接开跑 test。主要问题不是方法本身，而是若干主张、统计口径和实现路径还没有完全闭合。

**MAJOR 问题**

1. **主表缺少最关键的自然部署自消融：TAHRelM w/o PGIF。**  
   表 1 现在有外部基线和诊断上界，但没有在自然开局下的 `TAHRelM` 无 PGIF。表 2 的共享开局消融不能替代表 1 的部署主结论。  
   建议：表 1 加 `TAHRelM w/o PGIF --asd off --gallery-seed committed --reanchor off`，作为 Ours 的主自消融；表 2 保留机制隔离。

2. **共享开局不能写成 deployable 结果，只能写成 common-random-number 机制实验。**  
   共享开局把参考运行的首个选择注入其他臂，这不是实际部署时可用的信息。它是合理的方差控制设计，但不能作为部署性能。  
   建议：明确表 2/3/5 是“机制实验，不代表独立部署表现”。表 1/跨地图自然开局才是部署结论。

3. **共享开局记录对象写错：不能存 slot，应存 actor/tracklet identity。**  
   代码里候选每帧会 shuffle，`Candidate.idx` 才是稳定身份；slot 只是当前帧顺序。预注册稿写“存为 tracklet ID”是对的，但“首个 tick 承诺该 ID”实现时必须按 `c.idx` 查回当前 slot。  
   建议：开局文件格式写成 `{seed: actor_idx}`，target 用 `-1`，distractor 用其 `idx`；若该 actor 首个 tick 不可见，退回本臂语言选择并记录。

4. **分层分析有 collider / post-treatment 风险。**  
   “按各臂自己的开局锁对/错分层”会把模型行为本身用于分组，容易被审稿人视为事后条件化。共享开局下问题小一些，自然开局下问题很大。  
   建议：主分层用“参考运行开局正确性”或“共享开局文件中的开局正确性”作为统一分层变量；各臂自己的开局正确率只描述，不做因果解释。

5. **统计功效计算对双峰 episode 结果过乐观。**  
   现在用 n=19 的配对差标准差估 n，但结果接近 0/1 双峰，且开局正确数支配均值。普通连续差功效计算不够。  
   建议：补一个基于 Bernoulli/两层混合模型的模拟功效：开局正确概率、开局对层均值、开局错层均值分别建模。报告 n=60 对 Δ=0.15/0.20 的检出概率，尤其是 Holm 后功效。

6. **Table 3 的训练种子统计层级不清。**  
   3 个训练种子 × 60 episode 不能简单当 180 个独立 episode。训练种子是模型随机效应，episode 是环境随机效应。  
   建议：要么每个训练种子都跑 Full 和每个变体并做双层 bootstrap；要么明确闭环只跑中位 seed，此时 Table 3 只能支持“代表性 seed”而非训练稳定性。

7. **TAHRelM 内部消融不是严格 one-variable。**  
   `w/o 学习`、`w/o 相对特征`、`w/o 软运动门`会同时改变特征、参数化、训练目标或容量，不完全是 leave-one-out。  
   建议：定义每个变体的参数量、输入维度、训练数据、训练 epoch、选择规则。`w/o 相对特征`最好拆成“只去掉运动相对特征”和“绝对外观替代”两个层级，否则 reviewer 会说不是单变量。

8. **外部基线调参预算不公平。**  
   Ours 已经过多轮 dev 探索，外部方法“最多 3 组超参”会被认为偏少，尤其 DAM4SAM 和 DeepSORT/OC-SORT-style。  
   建议：改成“所有基线在 dev 上使用预注册网格，网格列明”；或承认这些是“style baselines”，避免叫完整 OC-SORT / DeepSORT SOTA。DAM4SAM 至少报告 gate、abstention、init timing、服务版本和失败原因。

9. **DAM4SAM 弃权口径必须进主文，不应只做附录。**  
   1003 已证明宽松口径会严重高估 DAM4SAM。  
   建议：主表同时列 commitment 口径和 abstention rate；脚注明确宽松/严格口径。不要把 DAM4SAM 的宽松口径藏到附录。

10. **实现清单低估了改动量。**  
   现有代码里还缺：`gate_commit`、共享开局 seed 传递、非 ASD 臂通用 tick 日志、结果 json 的 git/命令/批次元数据、闭环噪声注入。  
   代码对应问题：`RemotePolicy.reset()` 现在不传 seed；`policy_server.py` reset 也不接 seed；`asd.py` 只有 `gate_memory`；`policy.py` 在 PGIF fire 后总会改 `_committed_idx`；`candidates.py` 目前无噪声注入。  
   建议：test 前必须先在 dev 上证明“所有新开关关闭时与冻结版本逐位一致”。

11. **scorer 承诺和 policy 承诺双重滞回需澄清。**  
   policy 中 PGIF fire 会强制 `_committed_idx`，但 scorer 又用自己的 K=5 hysteresis 计算 `mis_follow_sustained`。  
   建议：主指标定义为 scorer 层承诺；tick 日志另存 `raw_mem_slot / pgif_slot / policy_committed_idx / scorer_committed_key`，避免解释“承诺门控”时混淆。

12. **算力估算偏低，尤其 Table 3。**  
   Table 3 若 4 个变体 × 3 训练种子 × 60 episode，已经 720 episode，不是 360；若还要 Full 的 3 seed，对应更多。  
   建议：重算 T3，明确哪些臂只离线、哪些闭环。否则排期会被 reviewer/合作者认为不可信。

**MINOR 问题**

1. `train/tah.py` 只是模型定义，训练入口是 `train/tah_train.py`。预注册稿第 12 节应改成两处都要加开关。

2. 结果元数据建议包括：git hash、dirty diff hash、完整 policy/server/rollout 命令、ASD_CFG、环境变量、CARLA port、policy GPU、DAM4SAM service port、seed range、batch id。

3. A/A 检验建议自然开局和共享开局各做一次。自然 A/A 估部署噪声，共享 A/A 估机制表残余噪声。

4. 批次效应需要随机化/交错。不要一个臂连续跑完再跑另一个臂；应按 seed block 或 arm block 交错分配到 CARLA 槽位。

5. 排除规则应补充“基础设施错误 episode 同 seed 补跑，原失败日志保留；target stationary 是场景级 skip，需确认各臂一致”。

6. “研究过程不进正文”基本可以，但 1003 的负结果和口径修正属于材料事实。建议正文局限中简短披露“融合/延迟承诺等 dev 方案未改善，最终只冻结 TAHRelM+PGIF 进入新 test”。

7. 图 `fig_architecture.py` 里仍写 n=19 PoC 数字。最终论文图应移除旧结果或更新为 test 数字，避免预注册后混用 dev/PoC 证据。

**总体建议**

先修预注册稿再跑 test。最关键的三件事是：表 1 加自然开局 w/o PGIF；把共享开局降格为机制实验并实现为 actor id 注入；补齐日志/元数据/`gate_commit`/噪声注入的 dev 冒烟。修完这些后，这个设计才比较像能扛住 CCF-A / SCI 一区机器人审稿的预注册方案。