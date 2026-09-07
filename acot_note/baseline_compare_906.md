# 外部 Baseline 对比 · 进度追踪

> 2026-09-05 起 · 用于 ACoT-UAV-Track 论文的外部对照工作(审计缺口 #4)· 配套 memory `acot-uav-paper-writing-kickoff`、`acot-uav-track-positioning` · 主结果/负结果 20ep batch 见 `runs/paper_batch.log`

---

## 0. 目标与原则

**为什么需要**:当前所有实验臂都是**同一 ckpt 的内部消融**,无外部方法对照 → benchmark 论文的常见审稿要求。补 ≥1 外部 baseline 强化说服力(冲顶会需要,D&B/CoRL/RA-L 加分)。

**核心原则(决定实验设计)**:harness 已支持切换 WHICH 模块(`policy.py tid_head` ∈ {xattn,reid,oracle,...})。**最严谨且可行的对比 = 把外部方法作为替代 `tid_head` 接入同一闭环(WHERE=CV、HOW=DiT 固定)**,只换身份机制 → apples-to-apples,隔离贡献。比对比整套异构系统更干净。

**评测协议**:与主 batch 同协议——20ep、seeds 91000-19、Town05、STEPS 900、s2=6、live CARLA;产出 `eval_out/paper_ext_<name>.json`;用 `rollout/continuous_metrics.py` 出客观指标表(正确跟踪占比 / 最长错跟时长 / 重捕获率&延迟 / id_switches / latch@Ts)。

---

## 进展快照(2026-09-06)

**内部对照基线已定稿(外部 baseline 要比的对象)** —— 主 batch 完成:7 臂 20ep 同种子 91000-19、live CARLA、消除 run 方差,`eval_out/paper_*.json`。

- **主阶梯(l1_xattn→l5_temporal)客观指标定稿**(19 计分,`continuous_metrics.py`):track_correct_frac **0.251→0.519→0.653→0.671→0.738**(~2.9×);最长错跟时长(中位)**9.70→3.00s**;latch@5s **0.105→0.895**(8.5×);id_switches 13.7→…→9.32(**非单调**);q_reacq 0.448→…→0.471(**l5 退化**);SR 0→0.053→0.105→0.105→0.105(l3 起**场景地板**不动)。
- **负结果(l4 vs BC-b0.4 vs RL-E0)20ep 定稿**:SR 0.105→0→0;track_fraction 0.217→0.073→0.040(崩)。⚠️ RL 的 latch@5s=1.0 是**近零跟踪的假象**,须点破。
- 诚实 caveat 全部写回 memory `acot-uav-paper-writing-kickoff`(FINAL PINNED NUMBERS)。

**外部 baseline 进度(更新 2026-09-06 晚)**:
- ✅ **`tid_head=assoc`(DeepSORT/OC-SORT)**:实现+单测+提交;**20ep 评测运行中**(motion+deepsort,GPU2)。
- ✅ **`tid_head=dam4sam`(DAM4SAM SOTA distractor-aware SAM2 记忆)**:独立 env(GPU3)+ socket 服务 + tid_head 全通,smoke PASS,**链式排在 assoc 后自动跑 20ep**。env 配方见文末附录。
- ⏳ **iKUN / JointNLT**:已评估——**均为零样本域差**(JointNLT 训 LaSOT/TNL2K 通用物体、iKUN 训 Refer-KITTI 地面车),迁 24px 空中 look-alike 会差(是"现成方法不 transfer"的信息,但须标注 zero-shot 域差)。env 可行(JointNLT py3.7+cu113;iKUN 类似),每个是一块独立 env+服务+tid_head 集成。**边际价值低于 assoc/DAM4SAM**(与内部 xattn 语言基线重叠)。判定:assoc+DAM4SAM 已构成强外部套件;iKUN/JointNLT 视投稿目标可选。

---

## 结果对比图(2026-09-06,真实数据)

**图 1 · 外部 WHICH-module 对比**(19ep 同种子同配置,只换身份机制)。生成脚本 `acot_note/figs/make_baseline_figs.py`。

![外部 baseline 对比](figs/fig_external_baselines.png)

> 诚实读:我们的 reid/temporal **非最优**——assoc-deepsort SR 更高、DAM4SAM 正确率 0.94+id-switches 近零(但 16s 粘滞长错窗)。**但所有外观/关联法(0.67-0.94)≫ 单帧语言 xattn(0.25)→ WHICH 瓶颈 + "外观/时序关联是杠杆"被外部方法独立证实**。latch-SR 无分辨力再获坐实(DAM4SAM 0.94 正确却 SR 0.105)。

**图 2 · reanchor(paper2 Plan-A/甲)**(17ep 同种子配对):语言兜底重锚**证伪**。

![reanchor 对比](figs/fig_reanchor.png)

> 语言重锚不帮反略差(0.363→0.225);oracle 重锚有效(→0.741)→ 机制有 headroom 但**语言仲裁不可靠**(collapse 时单帧语言在 look-alike 中挑错,同天花板)。

**图 2b · 时序语言重锚"抢救"(甲的翻盘尝试,17ep 配对)**:甲**彻底证伪**。

![reanchor 抢救](figs/fig_reanchor_rescue.png)

> 时序 EMA **部分降噪**(单帧 0.225→时序 0.291,证明"病根=单帧太吵"的诊断对)**但仍低于无重锚基线(0.363,虚线)**→ 语言即便时序积分仍是太弱的仲裁。oracle 0.741 证明**机制有大 headroom → 杠杆是更强的重锚信号(运动共识,见"可借鉴设计排序"①),不是语言**。

**图 3 · 架构结构对比**(Ours vs OC-SORT/ByteTrack vs DeepSORT vs DAM4SAM,基于 `policy.py` 真实接入代码)。源 `acot_note/architecture_compare_baselines.html`。

![架构对比](figs/architecture_compare_baselines.png)

> 结构性结论:DeepSORT(运动+外观联合门控)SR 全场最高(0.211)→ **ours 的 reid 缺一个显式运动一致性门控**是最小可改动的杠杆;DAM4SAM 的 distractor-aware 记忆让 id_switches 近零但 16s 粘滞长错窗证明**"记忆越强、错锁定代价越大"**,给 ours 若要加干扰项抑制提了一个必须配退出机制的警示。

---

## 可借鉴设计 · 排序与预判(结合全迭代发现,2026-09-06)

> **排序原则(非按"识别多准",而按"是否攻真正瓶颈")**:全迭代反复证明 **单纯提升识别准确率不移 SR**(warm-start↑身份 SR 不动、时序reID↑身份 SR 0.105 不动、reanchor↑机制 SR 0 不动)——因为 SR 受 **①场景地板(D1/D2=0.58,~42% 不可恢复)+ ②identity×control 级联 + ③latch 指标钝化** 三重制约。故借鉴设计按 **"攻级联/控制耦合 × 避开粘滞失效 × 代价 × 证据强度"** 排,而非按识别精度。

| 排名 | 借鉴设计(来源) | 具体改动 | **预判(效果+置信度)** | 迭代证据依据 | 代价/风险 |
|---|---|---|---|---|---|
| **① 首选** | **运动一致性作为"共识信号"(借 DeepSORT)** | `_reid_select` 打分 = 外观余弦 **× 运动门/一致性**(复用 `_assoc_uv` CV 图像位预测 + 已算的 DINOv2)。不是单纯加门,而是**双弱信号共识**(运动 AND 外观须一致才高分) | **SR ~0.105→0.15-0.21、max_wrong_run_s 3.2→~2.7s(高置信)** | **DeepSORT 是唯一 SR 更高的外部法(0.211,2×ours)**;assoc_motion 最短错跟(2.69s);而纯外观的 ours 易被同色 look-alike 吸走。运动作第二共识信号=剪掉空间不一致的 look-alike | 低(同一进程、复用特征,改打分融合) |
| **② 首选** | **共识置信度调制控制(强化④,统一门控→`conf_tau`/`_track`)** | 用①的**融合(运动+外观)置信度**驱动 commit-fly 决策:双信号一致→信任身份驱动 z_ex;不一致→保持+搜索,不追 | **直击真正瓶颈(control 耦合半),可能比①单独更破级联;但受 0.58 地板封顶(中-高置信)** | **④ 的洞察 + D1/D2 级联**:身份准确率单独提升边际收益已低(SR 地板);杠杆是"何时信任身份飞",而非 tid 本身更准。conf-gate 历史已 0.053→0.105 | 低(改 `_track`/`conf_tau` 用融合分) |
| ③ 谨慎 | **distractor 负记忆(借 DAM4SAM②)+ 强制退出** | K-view gallery 对称扩展:维护"近期高频误匹配候选"负样本集,打分惩罚 | **id_switches↓,但无退出机制则 max_wrong_run_s 会变差(中置信,高风险)** | DAM4SAM id_switches 0.79(记"谁不是目标")**但 16s 粘滞**;时序EMA 也现 q_reacq 退化——"记忆越强错锁代价越大"反复出现;而退出机制(reanchor)语言版已**证伪** → 退出本身未解 | 中,**风险高**:须先有可靠退出,否则把 3.2s 做差 |
| ④ 避免 | 纯外观记忆加强(DAM4SAM 稠密记忆范式) | 换更强外观/稠密特征记忆 | **正确率↑但 SR 不动(高置信,不值)** | DAM4SAM 正确率 0.941 却 SR 仍 0.105;时序reID 同理。**SR 地板证明这条路封顶** | 高代价、低回报 |
| ⑤ 不单独借 | 纯运动关联(OC-SORT) | 只用运动 | 单独 SR 更差(0.053)| 密集同速 look-alike 运动歧义大;**只在与外观共识(①)时有价值** | — |

**元预判(合全迭代)**:最值得借的 = **把运动做成第二共识信号(①)+ 用融合置信度调制控制(②)**——理由:(a) DeepSORT 是唯一实测 SR 更高的外部法;(b) 它攻的正是级联的"控制耦合"半(别 commit-fly 到空间不一致的 look-alike),而非我们已封顶的"识别精度";(c) 代价最低(复用现有 `_assoc_uv`+DINOv2)。**诚实上界**:即便①②到位,仍受 0.58 地板/42% 场景 + latch 钝化制约,**预期 SR 增益温和(→~0.15-0.20),最好用连续指标(max_wrong_run_s / lock_wrong)读**。**避免**把力气花在纯外观/纯记忆加强(④)——SR 地板证明边际已尽。③(负记忆)诱人但**须先解退出机制**(语言退出已证伪),否则重蹈 DAM4SAM 16s 粘滞。

---

## 1. ⚠️ 诚实警示(引用核验)

2026 年 UAV 预印本**须逐篇二次核验存在性/代码**;memory 已核出 **CosFly-VLA 是 fabricated → 禁用**。搜索摘要模型可能 confabulate,**未经核验不写进论文、不作 baseline**。

---

## 2. 候选 baseline 核验状态

| 方法 | venue | 代码 | 真实性 | 对我们的角色 | Tier |
|---|---|---|---|---|---|
| **OC-SORT / ByteTrack / DeepSORT** | ICCV/ECCV'21-22 | 成熟公开 | ✅ 确认 | **时序 EMA 的天然对照**(运动+外观关联) | 1 |
| **DAM4SAM** | CVPR'25 | [dam4sam](https://github.com/jovanavidenovic/dam4sam) | ✅ 确认 | SOTA **distractor-aware** 记忆跟踪(免训练 drop-in)→ 外观记忆强对照 | 1 |
| **iKUN / TransRMOT** | CVPR'23-24 | [RMOT](https://github.com/wudongming97/RMOT) / [TempRMOT](https://github.com/zyn213/TempRMOT) | ✅ 确认 | referring-MOT:语言型 WHICH 对照 | 1 |
| **JointNLT / UVLTrack** | CVPR'23 / AAAI'24 | [JointNLT](https://github.com/lizhou-cs/JointNLT) / [UVLTrack](https://github.com/OpenSpaceAI/UVLTrack) | ✅ 确认 | 现成 NL-tracker(被动感知)对照 | 1 |
| **TrackVLA / OmTrackVLA / TrackVLA++** | CoRL'25 | [TrackVLA](https://github.com/wsakobe/TrackVLA) / [OmTrackVLA](https://github.com/om-ai-lab/OmTrackVLA) | ✅ 确认(地面/人) | 全闭环 VLA 对照;迁空中 CARLA = 大工程 | 2 |
| **AerialMind / HETrack** | AAAI-26(2511.21053) | 待查 | ⚠️ 须核验 | 最接近的**已发表** UAV referring-MOT(被动) | 3 |
| UAV-Track VLA / DeTrack | 2026 预印本 | 待查 | ⚠️ 须核验 | UAV 域竞品 | 3 |
| ~~CosFly-VLA~~ | — | — | ❌ **fabricated 禁用** | — | — |

---

## 3. 推荐最小套件(性价比最优)——三点式 WHICH 对照

全部接入同一闭环(同 WHERE+HOW),配合内部阶梯(xattn→vlmpool→DINO→conf→temporal):

1. **标准时序关联(OC-SORT 式)** —— 时序 EMA 的**直接对照**;成本最低;**必做**。
2. **DAM4SAM** —— 证明"轻量 crop-DINOv2+时序EMA 对得起 SOTA 记忆跟踪";**强对照**。
3. **iKUN / JointNLT(语言型 WHICH)** —— 与 xattn 并列,支撑"单帧语言 grounding 不足→时序外观是杠杆"的**发现叙事**。

> TrackVLA 全闭环对照 = 冲顶会加码 / future work;AerialMind 核验后若可用 → 补为"已发表 UAV 竞品"。

---

## 4. 进度清单(勾选追踪)

**集成层**
- [x] **`tid_head=assoc`**:DeepSORT/OC-SORT 式运动(CV 图像位预测+门)+外观(crop-DINOv2 EMA)关联,接入 `policy.py`+`policy_server.py`;`--assoc-mode {motion,deepsort} --assoc-gate-px --assoc-lambda`;motion-mode 逻辑单测通过(选最近/GT更新/出画不更新+门控);SELECT用过去态、UPDATE用GT(与reid同无泄漏特权)。**待评测**(GPU2 主 batch 占用中)
- [x] **`tid_head=dam4sam`**:DAM4SAM 独立 env+socket 服务+tid_head 全通,smoke PASS,20ep 评测链式排队
- [~] iKUN/JointNLT:**决策 A(2026-09-06)= 停在 assoc+DAM4SAM 强外部套件,iKUN/JointNLT 转 future work**。JointNLT 已 de-risk(权重可下、env 配方验证 py3.7+torch1.11+cu113),续做低风险但零样本域差/边际低;顶会再补。
- [ ] (可选)JointNLT/UVLTrack 被动感知对照(离线在 episode 帧上跑)
- [ ] (加码)TrackVLA/OmTrackVLA 全闭环适配调研

**核验层**
- [ ] AerialMind(2511.21053)代码/权重是否公开、许可、可跑
- [ ] UAV-Track VLA / DeTrack 逐篇核验(存在性/代码)

**评测层**
- [x] **内部对照阶梯定稿**(l1_xattn→l5_temporal + 负结果 BC/RL,20ep 同批次)`eval_out/paper_*.json`；两表 + 诚实 caveat 已写回 memory
- [ ] ext baseline(assoc 先)各臂 20ep(seeds 91000-19)→ `eval_out/paper_ext_<name>.json`
- [ ] `continuous_metrics.py` 出"内部阶梯 + 外部 baseline"合并 WHICH 对比表
- [ ] 定稿数字写回 memory `acot-uav-paper-writing-kickoff`

---

## 5. 集成设计草案

**`tid_head=assoc`(OC-SORT 式,零外部依赖,先做)**:候选已有(`cset.cands`,稳定 actor idx + depth + 投影框)。维护目标轨迹的 Kalman 状态(图像位/尺度)+ 外观(crop-DINOv2 特征可复用);每 tick 用"运动门(IoU/马氏距) + 外观相似"匹配打分,匈牙利/贪心关联,committed = 关联到目标轨迹的候选。与 reid 公平:模板同样从 GT-可见目标建、SELECT 用过去、更新用当前(无重现泄漏)。→ 与时序 EMA 只差"关联规则(运动+外观 vs 纯外观时序积分)"。

**`tid_head=dam4sam`**(接口已侦察:`dam4sam_tracker.py` 首帧 bbox 初始化 → 逐帧出 mask/box):首帧目标可见时用其框(`cset.cands[true_idx]` u,v,w,h)init;每 tick step 当前 RGB → 预测目标框 → 与候选框 IoU/中心距匹配 → committed slot。⚠️ **集成风险(实测,须单独环境)**:官方要 torch2.1+cu121+py3.10;**但本机 GPU 驱动 470 → cu121 跑不了(需≥525)** → 须改 `torch==2.1.0+cu118`+py3.10+SAM2(还需 `python setup.py build_ext --inplace` 编 `_C`)。方案 = **独立 cu118/torch2.1 容器跑 DAM4SAM socket 服务(GPU3)**,policy-server(cu118)经 bridge(复用 `rollout/bridge.py`)请求目标框 → tid_head=dam4sam 匹配候选。**进展**:repo 已 clone(`external/DAM4SAM`),SAM2.1 权重下载中(各~160MB)。**判定 = 驱动受限的高风险多小时集成**(cu118×torch2.1×SAM2 兼容 + _C 编译 + GPU 显存 + 桥)。

**`tid_head=refmot`**:iKUN 两阶段——候选 proposal(我们已有)+ 语言匹配打分;直接替换 tid_logits。

---

## 6. 参考链接

- RMOT/TransRMOT https://github.com/wudongming97/RMOT · TempRMOT https://github.com/zyn213/TempRMOT
- JointNLT https://github.com/lizhou-cs/JointNLT · UVLTrack https://github.com/OpenSpaceAI/UVLTrack · TNL2K toolkit https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit
- DAM4SAM https://github.com/jovanavidenovic/dam4sam
- TrackVLA https://github.com/wsakobe/TrackVLA · OmTrackVLA https://github.com/om-ai-lab/OmTrackVLA · TrackVLA++ https://pku-epic.github.io/TrackVLA-plus-plus-Web/
- AerialMind https://arxiv.org/abs/2511.21053

## 附:DAM4SAM env 配方(2026-09-06,实测打通)
持久容器 `dam4sam-svc`(基 `acot-uav-train:cu118`,GPU3,cyh-carla-net,`sleep infinity`)。独立 conda env:
```
mamba create -n dam4sam python=3.10.15 -y && conda activate dam4sam
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118   # cu118 绕开驱动470
pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 hydra-core iopath pillow tqdm omegaconf einops
pip install vot-toolkit==0.7.1 vot-trax==4.0.2      # dam4sam_tracker 需 vot.region(精确版本)
cd external/DAM4SAM && pip install -e .              # 装 sam2(bundled)
```
**踩坑**:① 官方 cu121 撞驱动470→改 cu118;② numpy 必须 <2(torch2.1 ABI),**opencv-python 5.0 会把 numpy 拉回 2.x→卸掉只留 headless 4.10**;③ dam4sam_tracker 用 PIL Image 不是 ndarray;④ vot 精确版 0.7.1/trax4.0.2。SAM2.1 权重 `checkpoints/download_ckpts.sh`(large ~900MB)。smoke:`initialize(PIL,None,bbox=(x,y,w,h))`→`track(PIL)`→`{'pred_mask'}` 正确跟随。
