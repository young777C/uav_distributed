# 模型整合(路线 B)· Model Integration — TAH 时序关联头

> 2026-09-07 · 目标:把"大乱炖"的启发式 WHICH 栈**收成一个可学习模块 TAH**,变工程管线为干净算法贡献。配套 memory `acot-uav-paper-writing-kickoff`、`acot-uav-reid-reframe`;架构图 `architecture_tah.html`;数据清单 `paper_data_inventory.md`;外部对照 `baseline_compare_906.md`。

---

## 0. 动机(为什么走 B)
当前系统 WHICH 堆了 crop-DINOv2 + K-view gallery + top-m + 时序EMA + 运动共识融合(borrow#1)+ 共识门控(borrow#2)+ conf-tau + K滞回……**大多免训练手工启发式,每个修一个失效**。读起来像**在 benchmark 上调参堆技巧的工程建模论文,不是干净 AI/算法贡献**(kitchen sink 反模式)。且实测**borrow#2 共识门反害**(SR 0.263→0.053,过度保守)——**"继续堆组件边际转负"印证该停止加法**。
→ **路线 B**:一个学习模块替代整个启发式栈。

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
