**verdict: 不存在严格意义作弊，但存在不公平。**

我没有发现 deployable §5.5 决策链路直接读 `true_idx/is_target/true_slot` 来选择目标的证据；但 §5.5 不能作为“同等工程预算、同等传感接口下 PGIF 优于外部基线”的强结论。它更像是：在理想候选框、理想持久 actor ID、理想被承诺车辆世界坐标、开发种子复用的设定下，TAHRelM+PGIF 在正确锁定比例和最长正确锁定上表现最好或接近最好。

**(1) 评价标准一致性**

MAJOR: 种子、地图、最大步数、主干和 deployable 参数基本一致。`chains.sh`/`rerun_dam4sam.sh` 显示 10 臂均为 Town05、`SEED_BASE=91000`、`EPISODES=20`、`STEPS=900`、`--gallery-seed committed --reanchor off`，且同一 `eval_asd.sh` 启动 `stage2_best.pt`、`config_v5.yaml`、`--language-mode neutral --s2-period 6 --ex-source track`。JSON 也显示每臂 20 集、计分 19 集，均跳过 91004。

MAJOR: 但“代码版本一致”无法严格确认。当前 [policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy.py:1) 和 [eval_asd.sh](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/eval_asd.sh:1) 的 mtime 晚于 `paper_dep_*.json` 生成时间；结果文件也没有记录 git hash / image digest / policy hash。只能说日志命令行一致，不能证明当前源码就是运行源码。

MAJOR: 参数不完全同等。作者 TAH 系臂用 `conf_tau=0.5`，多数外部/reid 基线用 `0.05`；PGIF 额外开 `--asd filter` 和 `resid_decay=0.98`。这是方法差异的一部分，但也是调参预算不对等的证据。

MINOR/MAJOR: 有 6 个 scored episode 带 simulator timeout `error`，但 `progress.log` 仍报 0 fail。这些异常分布在 xattn、borrow1、tahrelm、PGIF、OC-SORT、DeepSORT。剔除 error 后 PGIF 正确率约 0.608，不削弱主方向，但说明 §5.5 的统计口径不干净，尤其最长锁错长尾不能过读。

**(2) 真值泄漏与受益臂**

MAJOR: 候选框本身来自 GT 投影。[candidates.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/candidates.py:44) 用 target/distractor 世界位置生成候选，并在 `Candidate` 中携带 `is_target` 和 `idx`。这是理想检测器和理想关联，不是现实感知。

未发现直接标签泄漏: 当前代码中 `true_idx` 进入 deployable 决策的路径主要在关闭或未使用分支：`gallery_seed != committed`、`oracle` head、`reanchor=oracle`、DAgger 采集、日志/scorer。§5.5 启动参数关闭这些通道。`asd.py` 的 `true_slot` 只写日志，`_filter_step` 只用 `aidx/mem_slot/conf`。

MAJOR: `Candidate.idx` 是间接便利。目标恒为 `idx=-1`，干扰车为稳定编号；PGIF 在 [asd.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/asd.py:190) 按 actor id 积累证据，l5/borrow1 也用 id 做 EMA。OC-SORT/DeepSORT/DAM4SAM 本体不等价使用这个理想持久 ID。因此这不是“告诉哪辆是目标”，但确实给 PGIF/时序 reid 一个理想 tracklet 接口。

MAJOR: WHERE 使用真实世界坐标。[policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy.py:528) 根据已承诺 `idx` 直接返回目标或干扰车 GT world position。所有臂共享，但这仍是理想定位假设；谁 identity 更准，谁就更能从这个理想 WHERE 受益。

MINOR: 候选顺序没有固定目标槽。PGIF 日志里 target-present 1338 条均满足 `aidx[true_slot] == -1`，true slot 分布在 0–7，不是固定排序作弊。

**(3) 自查 A–E**

A 成立，MAJOR。PGIF 独享理想持久 ID 的核心判断正确。`id_persistence_audit.py` 数字可复现：重发 re-entry ID 后仍保留 76%/91%/68% 开环增益；闭环 146 次 corrective fire 中 8% 在重入后 3 秒内。但该分析只模拟“重入换 ID”，没有模拟可见期间 ID switch / 近邻互换；因此会低估真实 tracker 关联错误对 PGIF 的伤害。

B 成立，MAJOR。训练 DAgger 数据本身是 92000–92023，和 91000–91019 不重合；但 PGIF 形态、λ/W、TAHRelM、`conf_tau` 和结论选择明显复用了 91000–19 作为开发/展示集。外部基线没有同等 dev 网格。

C 成立但未量化，MAJOR for DAM4SAM claim。DAM4SAM 作为视频 tracker 只在 S2 tick 每 0.6s 更新，可能被不公平降帧；其高弃权率支持这个担忧，但需逐帧更新对照确认。

D 成立，MAJOR。承诺口径有合理性，但它是在看到 DAM4SAM 弃权后修正的。三口径必须并列报告：DAM4SAM 原口径约 0.786、承诺 0.367、严格 0.190；PGIF 三者约 0.58，差异很小。

E 基本成立，MAJOR for longest-wrong。n=19 单次运行不足以支撑最长锁错强结论。最长锁错受运行间方差和 simulator timeout 影响大；“PGIF 比 DAM4SAM 最长锁错少 11.3s”不应作为显著结论。

**(4) 其他偏向作者方法的问题**

MAJOR: PGIF fire 后强制 `_pred_conf = 1.0`，见 [policy.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/policy.py:831)。这让 `conf_tau=0.5` 必过，从而强制更新承诺/控制。它不是 GT 泄漏，但应作为 PGIF 机制的一部分做消融：不强制置信、或统一 `conf_tau` 网格。

MAJOR: `conf_tau` 不同且未对外部基线调参。TAH/PGIF 0.5，reid/SORT/DAM4SAM 0.05，不能称为完全同一评价标准下的最优公平比较。

MINOR/MAJOR: scorer 的承诺口径对“弃权型 tracker”解释敏感。它比原口径公平，但仍会让早期承诺质量支配后续弃权段；因此 DAM4SAM 必须同时报告 original/commit/strict。

MINOR: 当前结果文件不含启动参数、代码 hash、环境变量快照，只能从日志旁证；正式论文实验应固化 provenance。

**(5) 综合结论与补充实验**

§5.5 不应表述为“PGIF 公平地战胜 DAM4SAM/外部 SOTA”。建议表述为：

“在理想候选框、理想持久 tracklet ID 和理想定位接口下，TAHRelM+PGIF 在 Town05 seeds 91000–19 的 deployable 身份决策设定中，相比 TAHRelM 无纠偏及若干未调参基线，提高了承诺口径正确锁定比例和最长连续正确锁定；相对 DAM4SAM 的优势为边界或不显著，且受后验口径、DAM4SAM 降帧、开发种子复用和理想关联接口影响。”

必须补充：全新 test seeds；各 baseline dev 网格调参含 `conf_tau`；记录 git hash/docker digest；剔除或重跑 timeout episode；DAM4SAM 逐帧更新；PGIF 在 noisy ID / ID-switch / fragmented tracklet 下闭环评测；给 SORT/DAM4SAM/reid 同等理想 ID 接口或给 PGIF 接真实 tracker ID；PGIF `_pred_conf=1.0` 消融；多次重复/交错调度。