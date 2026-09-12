# 模型整合(路线 B)· Model Integration — TAH 时序关联头

> 2026-09-07→09-11 · 目标:把"大乱炖"的启发式 WHICH 栈**收成一个可学习模块 TAH**,变工程管线为干净算法贡献。**权威系统梳理见 §0.5 实验账本**(B0→B0.1→B1→A1→deployable)。配套 memory `acot-uav-paper-writing-kickoff`、`acot-uav-reid-reframe`;架构图 `architecture_tah.html`;外部对照 `907_baseline_compare_906.md`。

---

## 0. 动机(为什么走 B)
当前系统 WHICH 堆了 crop-DINOv2 + K-view gallery + top-m + 时序EMA + 运动共识融合(borrow#1)+ 共识门控(borrow#2)+ conf-tau + K滞回……**大多免训练手工启发式,每个修一个失效**。读起来像**在 benchmark 上调参堆技巧的工程建模论文,不是干净 AI/算法贡献**(kitchen sink 反模式)。且实测**borrow#2 共识门反害**(SR 0.263→0.053,过度保守)——**"继续堆组件边际转负"印证该停止加法**。
→ **路线 B**:一个学习模块替代整个启发式栈。

---

## 0.5 ★B 路线完整实验账本(systematic,截至 2026-09-11)

> 本节是 B 路线所有实验的权威系统梳理(设置→结果→结论)。细节见 §1-9 + §5c/5d。**所有闭环 n=19 同种子(91000-19,除 reid_deploy n=17);gt-seeded=记忆每帧用 GT crop(oracle,所有对比臂同条件);committed=记忆从模型认定目标更新(GT-free,deployable)。**

| # | 实验 | 设置/关键改动 | 参数 | 离线 val | 闭环关键结果 | 结论 |
|---|---|---|---|---|---|---|
| **B0** | TAH v1 | 绝对候选嵌入 + GRU 记忆(注意力关联) | 724,737 | train 0.759 ≫ **val 0.521**(mis 0.479) | — | **过拟合**(开集 64ep 车池窄,绝对嵌入记住训练车);诊断=需车辆不变特征 |
| **B0.1** | TAHRel | **相对/车辆不变特征**[cos(候选,记忆)⊕相对位置⊕rank]+增广(置换/噪声/抖动) | **4,610** | train 0.905≈**val 0.909**(gap→0) | gt-seeded correct **0.681**、SR 0.053 | 过拟合**彻底解决**(离线碾压启发式);**但闭环<启发式=offline→online gap**(训练在专家取景、部署 off-center) |
| **B1** | TAHRel+**DAgger** | on-policy 重训:采集部署分布 23ep(seed 92000-23,与 eval 不相交)+离线 64ep 混合(87ep/44793帧,60ep) | 4,610 | — | gt-seeded SR 0.053→**0.158**、correct 0.681→0.716、latch@5s 0.68→0.95、mean持续 4.22→3.11 | **DAgger 补分布见效**(SR 3×、尾部阻尼);**仍<borrow#1(0.810/0.263)**;残余 gap 定位=**q_reacq(0.438,全场最低)** |
| — | 控制论 framing | 复用日志,**0 新 rollout** | — | — | max_wrong/q_reacq/idsw=闭环稳定性观测量 | offline→online=**有最优阻尼的正反馈回路**;一句话解释 borrow#1(甜点阻尼)vs borrow#2(过阻尼→崩) |
| — | 场景切分 | mis_by_N / central-off / reacq 桶,复用日志 | — | — | TAH+DAgger N=5 密度 0.778(最优)、Δ−0.046(最稳) | **学习外观关联在高干扰密度最优/最鲁棒**(运动类 DeepSORT 崩−0.27)→ 论文命题实证 |
| **A1** | TAHRelM+DAgger | **+可学软运动门** `msoft=exp(−‖relp‖/gate)`(borrow#1 运动共识的可学版,门半径可学)**+ EMA 速度**;7 特征;**OFAT(仅 arch 变 vs B1)** | **4,740** | **val 0.990**(mis 0.010) | **gt-seeded SR 0.316(1st/9)**、correct 0.746、max_wrong **1.90s**、latch@2/3/4/5s 全最优(.68/.79/.95/**1.00**)、N=5 **0.802**、off-center **0.818**、q_reacq 0.438→**0.590** | **oracle-memory 关联质量全场最优**(超手工栈 borrow#1 + 所有外部 tracker DeepSORT/DAM4SAM/OC-SORT);**离线 0.990 transfer**(运动共识=相对几何信号抗部署漂移) |
| **A1-dep** | TAHRelM committed(裸) | 去 oracle 记忆,GT-free,无语言 | 4,740 | — | SR 0.316→**0**、correct 0.746→**0.146**、max_wrong→**18.66s**、latch@5s→0.105 | **★SR 0.316 依赖 oracle 记忆种子**;deployable 崩(冷启动锁 candidate 0 + 记忆自我强化错锁);**< deployable reid 0.363** |
| **A1-dep2** | +语言冷启动+reanchor | committed + mem 空时用 lang_slot 冷启动 + reanchor=lang(low-conf streak→重置记忆+语言重选) | 4,740 | — | SR 0、correct 0.146→**0.263**、max_wrong 18.66→**13.30s**、latch@5s 0.105→**0.211** | **语言冷启动救初锁(~2×)但不充分**;**残余=confident drift**(自信跟错车,low-conf reanchor 抓不住);**仍< reid_deploy 0.363** |

### 分层结论(offline→online 的三层,逐层攻克)
1. **过拟合层(开集泛化)** ✅ 解决:相对/车辆不变特征(B0→B0.1),val 0.521→0.909,gap→0,4610 参。
2. **训练分布层(covariate shift)** ✅ 部分解决:DAgger on-policy(B1),SR 0.053→0.158;残余定位=恢复动力学。
3. **恢复动力学层(控制)** ✅ 解决(oracle-memory 下):可学软运动门+EMA(A1),q_reacq 0.438→0.590、gt-seeded SR→0.316 全场最优。
4. **memory-seeding 层(deployable)** ❌ **未解决**:gt-seeded 是上界;committed 崩(A1-dep);语言冷启动救 ~2×(A1-dep2)但 **confident drift 未解**,仍<deployable reid。

### 当前定论(诚实)
- ✅ **B 路线核心目标达成**:把手工 WHICH 栈(gallery+EMA+运动门+共识门+conf-tau+滞回)蒸馏为**一个 4740 参可学模块**,**oracle-memory 下关联质量全场最优**(超手工栈 + 所有外部 tracker)。干净算法贡献:诊断驱动 → 车辆不变可学相对特征 → 可学运动门。
- ❌ **未达成 deployable**:**SR 0.316 是 oracle-memory(gt-seeded)关联质量上界,非 deployable SR**。deployable(committed)SR=0;语言冷启动救 ~2×(correct→0.263)但 confident drift 未解,仍 < deployable reid(0.363)。
- **论文措辞铁律**:SR 0.316 必须标注 **gt-seeded / oracle-memory**(对比公平——所有臂同 gt-seeded);deployable 崩溃作**已刻画的开放问题/局限 + future work**,勿当 deployable SOTA。
- **下一杠杆(deployable)**:抓 confident drift 的漂移检测——**周期性语言复核**(每 K 秒用语言重验当前锁的车,不看置信度)或**记忆-语言锚点散度检测**,而非 low-conf reanchor。

---

## 1. TAH 设计(一个模块 = 原 5 个启发式)
`train/tah.py`,**0.72M 参数(724,737)**,唯一可训练;Qwen(4B)/DINOv2(vit-s 22M)/IAR/DiT 全冻结复用。
```
每帧候选 {crop-DINOv2 外观(384) ⊕ 图像位置(u/W,v/H,depth)}
  → enc(Linear→GELU→Linear, 256)                        [候选嵌入]
  → 注意力关联 score_i = ⟨q(记忆m), k(候选_i)⟩ / √d       [身份 logits]
  → softmax → committed;学习 conf(m) → 控制门
记忆 m ← GRUCell(committed目标嵌入, m)                     [递归更新]
```
**吃掉的启发式**:K-gallery+时序EMA = **GRU 记忆**;运动共识 = **位置作输入特征模型自学权重**;共识门 = **学习 conf 输出**。**零手工旋钮**(RMOT/conf-tau/gate-px 全消)。
**训练**(`train/tah_train.py`):监督关联,episode 序列 BPTT,GT 身份 teacher-force 记忆更新;off-screen 帧持记忆+跳损失。学的是**身份关联**(信号可学,DINOv2 已证 look-alike 可分)——非失败的 BC/RL 控制。

## 2. 数据 & 训练规模
- 缓存 `train/tah_cache.py` → `runs/tah_cache.pt`:**64 episodes / 42,637 frames**(mvp_full_v5,离线 crop-DINOv2 + 位置 + GT 身份,时序序列)。
- 训练:40 epochs,序列 BPTT,GPU 单卡。**参数 0.72M 极小**。

## 3. 当前结果(TAH v1)—— 学习中,过拟合是瓶颈
| epoch | train_acc | val_acc(mis) |
|---|---|---|
| 1 | 0.379 | 0.262 (0.738) |
| 10 | 0.558 | 0.343 (0.657) |
| 20 | 0.624 | 0.413 (0.587) |
| 30 | 0.720 | 0.477 (0.523) |

**读数**:val_acc **单调升**(0.26→0.48),**已进入启发式离线区间(mis 0.46-0.58)**;但 **train 0.72 ≫ val 0.48 = 明显过拟合**。→ **TAH 在学、逼近启发式,卡在泛化**(非无救)。对比锚点:启发式栈闭环 track_correct_frac 0.738(l5);离线 reid mis ~0.46-0.58。

## 4. 根因诊断(grounded)
TAH 学**绝对候选嵌入 + 记忆编码特定车辆外观** → 交叉熵在训练车过拟合(open-set 研究:"embeddings only locally accurate")→ 未见车泛化差。**启发式反而泛化,正因它相对**(候选特征 vs 存储目标的余弦,车辆无关)。**64 eps 车池窄 = 真数据瓶颈**(与 metric-learning marginal 同源)。

## 5. 泛化提升排序(下一步路线图)
| Tier | 方法 | impact | 成本 | 依据 |
|---|---|---|---|---|
| **①** | **相对/车辆不变特征**:输入改 cos(候选,记忆)⊕相对位置⊕相对邻居(GNN star-topology)⊕时序一致性,替代绝对嵌入 → 学"车辆不变门控"非"编码车" | **最高** | 低(改输入) | GNN-MOT relative features;open-set"绝对嵌入过拟合" |
| **②** | **增广促不变**:候选顺序置换/DINOv2 特征噪声dropout/时序stride抖动/位置抖动/难负样本加权/target-absent 注入 | 高 | 低 | 增广→不变特征+域泛化 |
| ③ | 正则:↑weight decay、dropout、**缩小 GRU 记忆容量**(占 0.39M 是过拟合主因)、早停、集成 | 中 | 低 | — |
| ④ | **更多/多样 episodes**(数据侧扩车型/场景) | 高 | 中(需数据 agent) | 直接开集修复 |
| ⑤ | 车辆不相交 train/val(量真 gap)、课程、对比辅助损失、meta-learning | 中 | 中 | few-shot open-set |

**元建议**:先 **①相对特征 + ②增广**(便宜、直击根因、文献支撑,训练 ~秒级迭代零成本)→ 预期 val 逼近/超启发式(相对函数车辆不变);仍受限 → ④扩数据。

## 5b. TAHRel Tier1+2 结果 + 闭环(2026-09-07)
**Tier1(相对特征)+Tier2(增广)彻底解决过拟合 + 离线飞跃**:TAHRel **4,610 参**(vs v1 72.5万,小 157×),val_acc 0.521→**0.909**(mis 0.479→0.091),train 0.905≈val 0.909(**gap 0.244→0**)。离线碾压启发式(reid 离线 mis 0.46-0.58)。
**但闭环 gt-seeded 输给启发式(offline→online gap)**:19ep 同种子 l5 / borrow#1 / TAH:track_correct_frac 0.738 / **0.810** / **0.681**;SR 0.105 / **0.263** / 0.053;max_wrong 3.18/3.07/**4.22**。**离线 0.909 → 闭环 correct 0.681**,且低于 borrow#1 与 l5。根因=**分布漂移**:TAH 训练在录制专家取景(crop居中),部署是 DiT off-center 取景(decorrelate 设计)→ 学到的特征加权失准。gt-seeded 已给 oracle 记忆(同 borrow#1)仍输 → 不是记忆、是**特征/位置分布漂移**(贯穿全项目的 offline↔online gap:BC-居中离线corr0.97→崩、reid 同病)。源 `eval_out/paper_ext_tah_gt.json`。
**判读**:offline-trained TAH **不 transfer**。offline 0.909+无过拟合证明"相对特征学习模块"信号/容量都在,唯一缺**部署分布数据** → 标准修法 **DAgger/on-policy 训练**(闭环采 crop+GT 重训)。B 未死,需 DAgger 定生死。

## 5c. ★B1 DAgger 终判(2026-09-08)—— B 退回 A
DAgger 采集 23ep 部署分布(seed 92000-23,与 eval 91000-19 不相交)→ 合并 offline(64ep)+DAgger 缓存(87ep/44793帧)→ 重训 TAHRel(`runs/tah_rel_dag.pt`,60ep `--rel --aug`)→ 闭环 gt-seeded(seed 91000-19)。
| 指标 | 旧TAH离线 | **TAH+DAgger** | l5 | borrow#1 |
|---|---|---|---|---|
| SR | 0.053 | **0.158**(3×) | 0.105 | **0.263** |
| track_correct | 0.681 | **0.716** | 0.738 | **0.810** |
| 持续 mean s | 4.22 | 3.11 | 3.18 | 3.07 |
| latch@5s | 0.684 | **0.947** | 0.895 | 0.789 |
| q_reacq | 0.490 | **0.438**↓ | 0.471 | 0.635 |

**判读**:DAgger **有效但不够**——SR 3×、灾难性长错窗基本消除(latch@5s 0.68→0.95),**证实 gap 一大块是分布性的**;但 tahdag track_correct 0.716 **仍 < l5 0.738 < borrow#1 0.810**,q_reacq 全场最低。**按 §6 判据 → B 不达标,退回 A**:borrow#1 作正文唯一有效升级,TAHRel 降为带机理负结果/消融。
**残余 gap 的定位(控制论视角,见 `907_research_reflection` §7-8)**:DAgger 补完分布后残差属**恢复动力学(控制)+ 取景不变性(表征)**,非分布——下一杠杆不是更多 DAgger,是补恢复/不变性。三视角实证为互补分层。源 `eval_out/paper_ext_tahdag_gt.json`、`eval_out/stability_control.json`、`figs/stability_plane.html`。

## 5d. ★★A1 终判(2026-09-08)—— B 路线全面胜出(SR 全场第 1)
B1 定位残余 gap = q_reacq(恢复动力学),A1 据此升级:`train/tah.py::TAHRelM`(4740 参)= TAHRel + **可学软运动门** `msoft=exp(−‖relp‖/gate)`(borrow#1 运动共识的可学版,门半径 softplus 可学)+ **EMA 速度**(可学衰减,跨丢失稳)+ 显式距离,共 7 车辆不变特征。同 offline+DAgger 缓存、同 60ep/aug(**OFAT,仅 arch 变**)。
**离线**:val 0.990(mis **0.010**,vs TAHRel 0.091)。**闭环 gt-seeded(n19)全场对比**:
| 指标 | l5 | borrow#1(旧最强) | TAH+DAgger | **TAHRelM(A1)** |
|---|---|---|---|---|
| **SR** | 0.105 | 0.263 | 0.158 | **0.316 ★1st/9** |
| track_correct | 0.738 | **0.810** | 0.716 | 0.746 |
| q_reacq | 0.471 | **0.635** | 0.438 | 0.590 |
| max_wrong mean | 3.18 | 3.07 | 3.11 | **1.90 ★** |
| latch@2/5s | .316/.895 | .474/.789 | .421/.947 | **.684/1.000 ★** |
| N=5 密度 | 0.750 | 0.738 | 0.778 | **0.802 ★** |
| off-center | 0.736 | 0.807 | 0.732 | **0.818 ★** |

**判读(oracle-memory 设定)**:在 gt-seeded 下 **TAHRelM 关联质量全场最优**——4740 参在 SR、error-persistence、latch 全谱、密度、off-center 均最优,超 borrow#1 与所有外部 tracker;离线 0.990 在此设定 transfer(TAHRel 0.909→0.681 崩;TAHRelM 无崩,因运动共识=相对几何信号)。**三视角闭合**:分布(DAgger)+控制(可学运动门)+表征(off-center 0.818)。

**★★deployable 复评修正(committed-seed,2026-09-11)**:去 oracle 记忆后 **TAHRelM 崩**:SR 0.316→**0.000**、track_correct 0.746→**0.146**、max_wrong 1.90→**18.66s**、latch@5s→**0.105**;且 **< deployable reid 0.363**。机理=**冷启动无语言引导(锁 candidate 0)+ 记忆自我强化错锁**。→ **gt-seeded 数是"oracle 记忆下的关联质量上界",非 deployable SR**;修正"全面胜出 deployable"的过度声称。deployable 需 **committed + 语言冷启动 + reanchor**(进行中)。源 `eval_out/paper_ext_tahrelm_{gt,committed}.json`。

## 6. 诚实判据(B 成 / 退 A)
- **B 成立**:TAH(+①②)关联准确率 ≥ 启发式栈(离线 + **闭环未见 seed** 验证)→ 一个 0.72M 模块替代乱炖,干净算法贡献;
- **B 走不通**:①②后仍显著低于启发式 + 过拟合不消(需大量新数据)→ **退回 A**(定位发现/研究型,组件降消融,borrow#1 运动共识作唯一有效升级留正文)。
- **上界**:仍受 DINOv2 特征判别力(look-alike cos 0.614)封顶——TAH 造不出特征里没有的可分性。

## 7. 产物 & 命令
- `train/tah.py`(模块)· `train/tah_cache.py`(数据缓存)· `train/tah_train.py`(训练)
- `runs/tah_cache.pt`(64ep/42637帧)· `runs/tah.pt`(best-val ckpt)· `runs/tah_train.log`
- 架构图 `acot_note/architecture_tah.html`
- 缓存:`docker run ... tah_cache.py --stride 2`;训练:`tah_train.py --cache runs/tah_cache.pt --epochs 40`
- **待接**:`tid_head=tah` 接入 `policy.py` → 闭环 eval vs 启发式栈(borrow#1 SR 0.263)。

## 8. 关联负结果(为何 B 而非继续 A 的加法)
- borrow#1 运动共识:✅ SR 翻倍 0.105→0.263(唯一有效借鉴,稳健甜点 RMOT≈0.3)。
- **borrow#2 共识门控:❌ SR 0.263→0.053**(过度保守)——加法边际转负,佐证"停止堆组件、转 B 提炼"。
- reanchor 甲(单帧+时序语言):❌ 彻底证伪(语言太弱仲裁)。
