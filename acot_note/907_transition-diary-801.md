# 研究迭代总结 · Transition Diary

> 2026-07-29 起 · 2026-08-11 增补"研究点收敛(去发散)"(§五)· 2026-09-05 增补"瓶颈重构→re-ID→客观度量"(§九)· 用于近期工作汇报 · 配套:`acot-uav-design.md`、`experiment-design.md`、`data-fix-spec.md`、`world-model-survey.md`、`language-seeded-prediction-survey.md`、`learning_diary_0729.md`、memory `acot-uav-reid-reframe`/`acot-uav-stage3-rl`

## TL;DR(一句话迭代线)

我们从"**空中·语言指定·多相似车消歧跟踪**"的 VLA 出发,打通了完整训练管线;但在 MVP 严谨消融中发现:**纯目标跟踪(不含长时间丢失)靠视觉/几何已能很好求解,语言几乎无净贡献,且该问题在现有工作里已相对成熟**。据此,我们把问题**难度升级为"长时间画面丢失后的预测-重捕获"(源于真实需求)**——这一升级同时**恢复了语言的必要性**、**兑现了 EAR 预测的价值**,并暴露出**架构需从反应式 VLA 向具备世界建模能力的 WAM(World-Action Model)演进**。此后(08-05~11)又派生出三条探索线——**SEP 结构化环境先验 / world-model 方法学综述 / 语言 which-way 意图种子**;它们**不是三个平行新课题,而是同一主线的三个层**,§五做统一收敛。最后(08-18)在建 WM 之前先**公平测量原模型的预测上限**(§八):公平重训后 EAR 的出画预测**仍然失败且训练救不回**——坐实"反应式单帧模型无法预测出画位置",**为 WM(路网世界建模)提供了干净判据**。**(08-31→09-05,§九)** 闭环诊断进一步把 SR=0 的墙从 **WHERE(预测-拦截)** 移到 **WHICH(look-alike 身份重捕获)**:WHERE 被 CV 先验便宜地补上、WM 边际价值被压缩;沿 WHICH 做出**单调改进的客观指标阶梯**(单帧语言 grounding → crop-DINOv2 → K-view gallery → track → 置信门控 → **时序 EMA re-ID**),**正确跟踪时间占比翻倍(0.31→0.72)、最长错跟时长缩到 1/3.4**;两次"控制侧修取景"(BC-居中 / Stage-3 RL)被证伪(居中在身份下游、放大身份错);并确立**用客观连续指标(而非饱和的 latch-SR)衡量逐步改进**的度量方法学。

---

## 一、起点:最初的模型架构与实验设计

### 1.1 模型架构(VLA)
双系统混合 VLA,机制迁移自 ACoT-VLA:
- **冻结 VLM 骨干** Qwen3-VL-4B(图像+语言融合,layer-24 缓存);
- **EAR**(显式动作推理):flow-matching 输出粗粒度 3D waypoint(动作空间 CoT);
- **IAR**(隐式动作推理):逐层 KV 查询,提取遮挡/机动/混淆等"说不出来"的先验;
- **AGP/DiT**:三重 cross-attention 融合 → 动作块;**target-id 头**:语言在候选车中选目标;
- **双系统**:S1(DiT,~20Hz 反应)+ S2(VLM+EAR+IAR,~1Hz 审慎)。

### 1.2 核心主张与实验设计
- **核心主张**:四轴交集(**空中 + 车辆 + 语言消歧 + 具身闭环**)无先例。
- **实验主干**:H0(语言必要性,生死线 gate)→ H1(EAR)→ H2(IAR)→ H3(动作 CoT>语言 CoT)→ H4(中性语言)→ H5(竞品)。**H0 不过则全盘停**。
- **训练**:Stage 0-3(骨干探针→EAR 预热→端到端+课程→RL);MVP 数据 60 集 CARLA。

---

## 二、MVP 过程中发现的问题(按逻辑顺序)

### 2.1 [已解决] Stage-1 EAR 定位地板
初期 waypoint 误差近乎水平 ~14m。诊断为**定位地板**(连"目标当前在哪"都定不准),非预测难度。**修复**:世界系→相机系 waypoint + 注入 proprio(高度/俯仰/速度)。地板 14m→7m,误差曲线由**平**(定位坏)转为**随 horizon 上升**(定位已解、残差=预测不确定性)。

### 2.2 [核心发现] 语言在纯跟踪上**不 load-bearing**
正式 w/o-language 消融(3 seed,严谨):

| 指标(困难子集 ≥2 候选) | 有语言 | 无语言 | 结论 |
|---|---|---|---|
| Mis-follow(选错目标车率) | **6.16%±1.81** | **6.23%±1.12** | **统计无差异** |

**去掉语言几乎不掉点** → 模型靠**非语言捷径**就能选对目标。根因量化:

| 策略(困难子集) | Mis-follow |
|---|---|
| 随机 / 最居中 / 最近 | ~70% / 53% / 19% |
| **最优纯几何位置分类器**(无外观、无语言) | **~3-5%** |
| 有语言 / 无语言 神经模型 | ~6% |

**纯几何位置(u,v,深度+排名)就能把目标选对到 ~3-5%,比用了整个 VLM 的模型还低** → 目标位置被数据系统性泄漏,语言冗余。外观探针进一步确认:**外观不构成额外泄漏**(相似干扰车确实视觉难分,符合设计意图)。
> 工具已固化:`train/benchmark_leak_probe.py`(数据修复后一键复验捷径是否堵住)。

### 2.3 [补充证据] 纯跟踪+消歧的领域已相对成熟
- **VLA 普遍无视语言、走捷径**:π0.5 在矛盾指令下仍 96.2% 成功,Linguistic Grounding Score 仅 1.2(*Shortcut Learning* @CoRL 2025;*VLABench* @ICCV 2025)。
- **视觉相似车消歧可由空间推理+记忆扛下,不必靠语言**:TrackVLA++(2510.07134)在 EVT-Bench 干扰 split 的增益全来自**非语言**模块(Polar-CoT 空间推理 +6.0%、时序记忆 TIM +2.8%);CoMaTrack(2603.22846)同理。
- **语言仅在"去相关"后才 load-bearing**:CAST(2508.13446)显示,有相似干扰车时标准 VLA 31.7%→反事实增强 58.3%(1.84×),**无干扰时"指令条件化根本没必要"**。

**小结**:在"目标基本始终可见 + 位置可预测"的设定下,**视觉/几何已足够,语言无用武之地,且强竞品已做得很好**——继续在此设定上堆语言消歧,难有贡献、还与竞品同质。

---

## 三、研究点的迭代:从"纯语言消歧" → "长丢失预测-重捕获"

### 3.1 为什么升级问题(证据 + 真实需求,非凭空)
- **证据面**:2.2/2.3 表明纯跟踪消歧已被视觉/几何解决;
- **需求面(真实)**:观察 rollout 视频发现,目标拐弯/遮挡/出画会**长时间丢失**;现实中无人机**不应因丢失就结束跟踪**,而应**预测目标未来位置、直飞拦截重捕获**。当前数据丢失过短(实测最长 ~2.0s、仅 0-2% 帧),既未训练也无法评测该能力。

### 3.2 升级后为什么"起死回生"
把问题设为**长时间丢失后的重捕获**,一举恢复三点:
1. **语言重新变必要(H0)**:长丢失使**视觉连续性彻底断裂**;重现时若有 ≥2 辆 look-alike 共视,则**位置捷径失效 + 连续性失效 + 外观不可分**三者叠加,"哪辆才是目标"**只有语言能定**;
2. **EAR 预测变不可替代(H1)**:预测目标未来 3D 位置以引导拦截,是这套设计的用武之地;
3. **aerial 差异化**:无人机**无路网约束**可直飞抄近路拦截;竞品(TrackVLA++)的重捕获靠**非语言空间记忆**,在"断裂+look-alike"时刻**答不出"哪辆是目标"**——这是最锐利的切割。

> **关键提醒(避免走偏)**:长丢失恢复若重现时只有一辆车,退化为**纯预测**,与 TrackVLA++ 的非语言空间重捕获同质、丢卖点。故消歧证据必须**精确到"重捕获时刻有 look-alike 共视"**(已写入 design §6.3 / data-fix-spec §3-4)。

### 3.3 H0 判据的收窄(锁死在数据构造上)
H0 成立 ⟺ (a) 语言-only 指定(无 bbox 锚)∧ (b) 身份⊥位置 ∧ (c) 身份⊥时空连续性(须有断裂事件)∧ (d) 重捕获时刻 look-alike 共视足够。现状:(a) 已改文档满足、(b) 待数据去位置捷径、(c) 待造长丢失断裂、(d) 聚焦到重捕获时刻。

---

## 四、架构的迭代:VLA → WAM(World-Action Model)

### 4.1 为什么反应式 VLA 不够
长丢失预测暴露了当前架构的根本局限:**单帧输入 + proprio 没有"目标状态记忆"**(proprio 是无人机自身状态,不含目标)。目标出画后,模型看到的是一张**没有目标的图**,**无从预测目标去了哪**。现有机制只能靠 EAR 在最后可见帧的 waypoint(≤6s)+ staleness 复用扛**短**丢失;**长丢失(>6s)超出 horizon 即失效**。

### 4.2 需要补的能力:世界建模(WAM)
要在长时间不可见下持续预测目标,模型必须**维护并推演目标在世界中的状态/动力学**,而非纯粹反应当前观测——即从反应式 VLA 升级为**具备世界模型的 WAM**:
- **目标状态记忆/信念**:最后可见位置 + 速度/朝向,随时间传播;
- **动力学/世界模型**:预测目标在遮挡期间的运动(受道路/场景先验约束),给出拦截点;
- **延长预测 horizon** 或以最后 waypoint 播种一个预测器;拦截不可靠时由 search_mode + 语言重捕获兜底。

> 这是本阶段最重要的**架构假设**:从 "看到→反应" 的 VLA,走向 "**建模世界→预测→拦截→(语言)重捕获**" 的 WAM。这也把 EAR 从"锦上添花"提升为"世界模型的动作接口",与研究点升级自洽。

---

## 五、研究点收敛(去发散):一条主线 + 分层扩展(2026-08-11)

> 08-05~11 派生了三条新线(SEP/路线B、world-model 方法学综述、语言 which-way),有发散感。本节把它们**收回到一条主线的分层**,并明确论文范围与优先级,避免"四个方向并列"。

### 5.1 唯一核心命题(其余全部服务于它)
> **在目标长时间不可见(>6s 丢失)时,系统靠"目标状态记忆 + 任务相关世界模型"预测目标未来 3D 位置、直飞拦截重捕获;而重现时刻的 look-alike 消歧,只有语言能定。**
> = 一个 **world-model-augmented VLA** 在"空中闭环预测-拦截重捕获"上兑现,语言在该时刻不可替代。**这里的"世界模型"不是完整生成式世界模型,而是任务相关因子的预测器**(EAR 目标状态 + SEP 路网先验)。

### 5.2 分层(按确定性 × 优先级 × 论文范围)

| 层 | 内容 | 确定性 | 依赖 | 论文范围 |
|---|---|---|---|---|
| **L0 已验证脊柱** | nolang 证伪 + 位置捷径量化(~3-5%)→ 长丢失升级 | ✅ 已实证 | — | paper1 动机 |
| **L1 核心方法(进行中)** | EAR + **目标状态记忆(WAM 最小实现)** + 长丢失数据 + mask 反转 | 设计就绪、待数据 | data-fix-spec b/c/d | **paper1 主贡献** |
| **L2 路线B / SEP** | 从斜视航拍在线推**时不变路网**,作 identity-agnostic where 先验;时不变/时变 2×2 | 方法学有背书(PERSIST/WorldMem 显式3D派)+ H0-leak guard(§2.0c)、待消融 | L1 + 长丢失数据 + 路网 GT | paper1 增强 **或** paper2 核心 |
| **L3 语言 which-way** | 语言从 which(身份)扩到 which-way(意图种子) | 最低:先例成熟(原理不新)+ 冗余陷阱 + 与 SEP 张力 | 富意图语言 + 分叉歧义场景 | **paper2 / future work** |

### 5.3 三条新线各自归位到主线(不是三个平行方向)
- **world-model 方法学综述 = 工具箱,不是新研究点**:给 L1/L2 提供"怎么抗 EAR drift(因果 AR 的 KV-cache 自 rollout / error-recycling / BAgger,均训练侧、不需像素生成)""SEP 属显式 3D 状态记忆派""drift 上限不可根治(R11)"。→ 服务 L1/L2,**不单列为方向**。(详 `world-model-survey.md`)
- **SEP(L2)= §4.2"需要补的世界建模能力"的具体落地**:不是新方向,是把"WAM 需要的世界模型"**收窄成任务相关因子(路网)**。→ **SEP 就是 WAM 的世界模型实例**。(详 related-work §4D)
- **语言 which-way(L3)= L0"语言=消歧"的推广**:语言从"外观分不清→定 which"扩到"几何+运动定不了→定 which-way",同一原理两个实例。→ 是**语言必要性主张的延伸,不是新任务**。(详 related-work §4E)

> **一句话去发散**:三条新线分别是主线的 **工具箱(综述)/ 世界模型落地(SEP)/ 语言必要性延伸(which-way)**,不是三个平行新课题。

### 5.4 必须调和的内部张力:SEP("语言只定 which") vs which-way("语言也塑造 where")
表面矛盾:SEP 的 2×2 说语言只定身份(which)、SEP 定 where;which-way 又说语言塑造预测(意图→路网/轨迹)。**调和如下(写正文前必须统一,否则 §4D 与 §4E 自相矛盾)**:
- **SEP 定义 where 的"可行集"(环境约束:路能通到哪),语言(which-way)在可行集内做目标特定的"选支路"**。语言**仍不生成环境结构**,只在结构给定的分叉处选择。
- 落到 2×2:**SEP 仍独占"时不变·身份无关"格;语言那一列从"which(身份)"多出一个"which-way(目标意图,在可行集内选支路)"的行为**——2×2 不破。
- 且 which-way 严格受 **§2.0d 意图冗余 guard** 约束:只在"SEP 路网+当前运动都定不了"的分叉才算数(否则退化为位置捷径的同构变体)。

### 5.5 论文范围决策(收敛建议)
- **paper1 = L0(动机)+ L1(核心 WAM 最小实现)+ L2 最小版**(SEP 仅作长丢失先验,只验 H7/g2 不放大 H0)。这已是"新任务 + 新 benchmark + WAM 最小实现 + SEP"的完整贡献。
- **L3(which-way)明确 defer 到 paper2 / future work**:若并入 paper1,H0 归因会从"2 条捷径(位置+连续性)"膨胀到"4 条轴",评审归因风险剧增,得不偿失。
- **判据**:只有当 L1 的长丢失数据 + 闭环 harness 就绪、且 L2 的 SEP 消融站住,才考虑把 L3 提前。

---

## 六、当前状态与下一步

**已完成**:训练管线打通(Stage-1/2)、EAR 定位修复、A/C 消融(C 稳健、A 修正为~20%非2×)、**语言必要性证伪 + 位置捷径量化**(核心发现)、mmap+workers 基建(8 卡可快速重训)、设计文档/实验设计/数据规格三方对齐更新;**两轮深研落盘 + 挂锚**:`world-model-survey.md`(L1/L2 工具箱)、`language-seeded-prediction-survey.md`(L3 背书),related-work §4D/§4E、experiment-design §2.0c/§2.0d、H7/H8、M11/M12、R9–R12。

**下一步(按分层 × 依赖,对齐 §5.2/§5.5)**:
1. **L1 数据侧(关键路径,paper1 命脉)**:按 `data-fix-spec.md` 做 (b) 位置去相关 + (c) 长丢失断裂事件(专家演示拦截、目标重现、重捕获时刻 look-alike)+ (d) 聚焦共视;`benchmark_leak_probe` 作验收门。
2. **L1 训练/架构侧(paper1 主贡献)**:mask 反转已就位(`supervise_intercept` 门控,待长丢失数据即生效);VLA→WAM 的**目标状态记忆最小实现**(最后可见位置+速度传播)。
3. **L1 闭环侧**:建学生策略 rollout harness(兑现"具身闭环"+ 评"重捕获成功率 vs 丢失时长")。
4. **L2 增强(SEP,paper1 增强 或 paper2)**:SEP 头 + CARLA 路网 GT;只验 H7/g2(§2.0c guard,不放大 H0);对照纯隐式长上下文(R10)。
5. **L3(which-way,defer)**:MVP 不做;仅在 §2.0d 立三臂消融 + 分叉歧义场景 + 冗余 guard 的设计,paper2 再兑现。

> **优先级铁律**:L1 未站稳前不投 L2/L3 的实现资源(§5.5)。L2 是"锦上添花可提前、也可留 paper2";L3 一律 defer。

---

## 七、实证证明语言必要性:指标净化 → 多版本去相关 → 分辨率墙 → 干净成立(2026-08-11 → 08-17)

> §二.2 只证伪了"老数据上语言不必要"。这一章是**把任务改造到语言真正必要、并干净证明它**的完整实证历程。一句话:**逐层拆掉每一个让模型"不用语言也能蒙对"的捷径,最后发现最深的墙是图像分辨率。**

### 7.1 先净化指标(否则一切结论都假)
- **发现指标漏洞**:`mis_follow` 被 `argmax` 平局 × "target 恒在候选 idx0"做假——cross-attn 头分数一塌平,平局默认选 idx0=目标 → 假的"突破"(曾误读为语言起作用)。**头本身是排列等变的**(实测),漏洞在指标。
- **修复**:`dataset_stage2` 打乱候选顺序 → target_idx 均匀。头等变 ⇒ 训练不变、指标变诚实。详见 memory `acot-uav-tid-metric-artifact`。

### 7.2 多版本数据去相关(逐个关闭几何捷径)
`benchmark_leak_probe` 作验收门(几何 floor 越接近 chance 越好):

| 版本 | 关掉的捷径 | 几何 floor | E1(有语言 vs 无语言) | 真相 |
|---|---|---|---|---|
| mvp_full | 位置(横向) | 5%→41.6% | — | 位置捷径主导 |
| **v3** | + 深度/跟踪距离 | nearest 31%→**80%** | 有 56% / 无 71%,gap 15pp | **假象**:56%≈几何 floor 54%,gap 是几何被语言条件化带出来的 |
| **v4** | + 居中(相机瞄准偏移) | most-central→chance | 有 **72%≈chance** / 无 76%,gap ~3pp | 语言**想用但用不上**——非数据问题 |
| **v5** | (几何全清)+ **分辨率 336→512** | chance 76% | 有 **58%** / 无 72%,**gap ~13pp** ✅ | **语言真起作用** |

### 7.3 v4 卡住 → 定位"不是数据、不是头,是特征分辨率"
v4 几何全清、E1 却回到 chance。系统排查:
1. **任务可解**:目标完整描述在每帧 **100% 唯一**,语言 oracle 天花板 = **0%**(不是任务无解);
2. **身份不在特征**:`cand_feats→属性` 线性探针 by-episode 只有 make 12%/color 23%;
3. **re-crop 也救不了**、full-frame pool 也救不了 → 一度误判"再大也没用";
4. **决定性一测——同分布 vs by-episode**:make **by-episode 6% 但同分布 55%**、color 62%。**同分布高 ⇒ 27px 特征其实富含身份**(看不清就无法分类);by-episode 低只是"线性映射跨集不迁移"。而**端到端 VLM-grounded tid 头能迁移**(靠 Qwen 预训练车辆知识),线性探针不能。
5. **根因锁定 = 图像分辨率**:336px+90°FOV+远 standoff → 车仅 **12px** → 身份被 tokenize 抹平。**这是"空中小目标该高分辨率、别抄机器人 VLA 的 224/336"的教训**(调研:VisDrone 用 2K-4K;OpenVLA 224 因目标又近又大)。

### 7.4 v5:512px 一改,E1 从失败翻成成功
- 数据侧:512×512 + FOV/standoff 调整 → 车 **12→27px**;
- **唯一变量是分辨率** → E1 有语言从 72%(chance)→ **58%(破 chance 13pp)**,稳定收敛;
- **坐实**:瓶颈就是特征分辨率,不是数据场景、不是 head、不是训练。

### 7.5 训练侧配套(让薄语言信号能稳定学到)
- **稳定化四件套**(v3 起):tid 头 dropout + 独立 weight_decay + 余弦 LR + **best.pt 按 mis_follow 选**(原按 act_mse → 存错 epoch)。否则 val 早峰晚衰、gap 塌回(naive 收敛期 gap 3pp → 稳定后 ~13-15pp)。
- **省磁盘 E1 配方**(共享盘仅 267GB 时):mis_follow 只依赖 tid 头 = 只需**最后一层** → 缓存 `--layers 24` + **fp16**(~43GB/缓存 vs 5层fp32 426GB)+ 跳 EAR/Stage-1 → mis_follow 与完整模型**完全一致**。

### 7.6 一个 gotcha(踩过)
v5 本意 FOV 70,但**渲染实际等效 fov90**(投影中心+尺寸零误差、bbox GT=27px 双证)。**config 的 fov 必须匹配渲染(90)**,否则候选框偏 ~60px、全错。不影响结论——27px 已够。

### 7.7 现在的定论
**在"几何全除净 + 指标诚实 + 稳定训练 + 分辨率够(27px)"四条件下,语言首次干净 load-bearing:有语言 ~58% vs 无语言 ~72%(≈chance),稳定 gap ~13pp。** H0 的核心生死线成立。有语言仍非 0%(共视 look-alike + VLM 细粒度上限),但**方向已证**。下一步瓶颈从"数据/分辨率"转向"把 grounding 做得更强 + 闭环兑现"。详见 memory `acot-uav-language-necessity`。

---

## 八、建 WM 前先测原模型的预测上限:公平 no-WM 基线 → WM 有据(2026-08-18)

> **战略前提**:在给架构加 WM 这样的复杂模块前,先回答"原模型是否已到上限?若未到,先优化原模型别过早加复杂度"(用户的方法论)。E1 的 EAR 是**欠训的**(E1 为省磁盘跳过 Stage-1 预热、只用 1 层缓存、focus 在 mis_follow),不能作为"原模型上限"。故先做一次**公平重训**再测量。

### 8.1 公平重训(`train/train_v5_fair.sh` → `runs/stage2_v5_fair`)
把 E1 省掉的都补回,给原模型最好的机会:**Stage-1a EAR 预热**(err_6s 从 129m→~33m 收敛)+ **完整 5 层 IAR 缓存**(层 4/8/12/16/24,fp16)+ **`supervise_intercept` 开**(intercept-BC)。产物 `stage2_best.pt`(ep25,mis_follow 0.582)。

### 8.2 判据工具:`train/intercept_eval.py`(离线代理,须容器内跑)
闭环真值要 harness(P3);离线先给两个代理:
- **EAR 预测 FDE**(米):采样 EAR 自预测航点 vs GT,按**可见/丢失 × 视界(1/2/4/6s)**拆;加**速度无关 persist 参照**(预测未来=1s 位置);
- **拦截朝向 cos**:用 EAR 自预测航点驱动 DiT 得**模型动作**,对比**专家动作**净位移方向的 cos(专家自身≈1.0)。

### 8.3 结果:公平训练也救不回出画预测

| 指标 | E1(欠训/1层) | **公平(预热/5层)** | 判读 |
|---|---|---|---|
| 可见 EAR FDE 1s/6s | 15.2 / 23.3m | **14.2 / 21.2m** | 仅好 1-2m,**仍不如 persist**(6s: 13.7m) |
| 丢失 EAR FDE 1s/6s | 58.5 / 77.2m | **58.3 / 77.7m** | **几乎不变**——训练救不回 |
| persist 参照(丢失 6s) | 17.6m | 17.6m | EAR 比"假设目标不动"还差 4× |
| 拦截 cos 丢失(中位/>0.5) | 0.92 / 71% | 0.87 / **66%** | 略降;34% 飞错 |
| mis_follow(best) | 0.579 | **0.582** | ≈,不受 EAR/IAR 层数影响 |

### 8.4 四条定论
1. **公平版 EAR ≈ 欠训版 EAR**(丢失帧 FDE 58→58m 几乎不变,可见仅好 1-2m)→ **EAR 预测的烂是架构性的(单帧限制),不是欠训**。给了最好的训练也无济于事。
2. **EAR 连可见帧都不如速度无关的 persist**(21m vs 13.7m@6s)→ 单帧预测目标未来位置 genuinely poor,1s 都偏 14m。
3. **丢失帧灾难(58-78m)且训练不变**→ **反应式单帧模型从根本上无法预测出画目标位置**(比"目标不动"还差 4 倍)。
4. **拦截动作**:cos 丢失 66% 朝目标(靠 `supervise_intercept` 的 BC 惯性,**非准确预测**),34% 飞错 → **拦截行为在但不鲁棒**;WM 给准确预测后应大升。

### 8.5 对战略问题的回答 → WM 有据
- **原模型在"预测"轴已到上限**(公平训练救不回 → 是单帧的根本限制,非调参空间)→ **继续 tune 原模型无用,WM 才是下一步**。
- **语言(mis_follow 0.582)在 grounding 天花板**,不随 EAR/IAR 层数变 → 语言轴的优化走 grounding(§七),不走加层。
- **`stage2_v5_fair` = 干净的 no-WM 离线基线**(EAR-only/M0),即 WM(P2)与 SEP(H7)必须超越的对照;闭环 no-WM 基线待 harness(P3)。
- 详见 memory `acot-uav-wm-roadgraph`(内含 gotcha:`docker run --rm` 会删日志→输出重定向到文件;5 层缓存训练 CPU-bound fp16→fp32 astype,GPU~0% util,~3-5min/epoch)。

### 8.6 便宜基线阶梯 + 深度/横向分解(2026-08-19,`baseline_deadreckon.py` + `intercept_eval` 分解)

判 WM 前再补一层严格性:反应式是否连**最便宜的记忆/匀速**都没做到?两件工具:
- `intercept_eval` 加 **深度(光轴,dim0)/横向(像平面,dim1:2)分解**;
- `baseline_deadreckon.py`:**逐 episode 因果**跑 **②最后可见记忆·零速** + **③匀速外推**(世界系,用 GT 速度 = CV 上界)+ persist,**按 丢失时长 × 转向 分层**,可选 EAR 对照。全在**世界系推算→当前相机系**,自运动完全补偿(③ 用的是目标**绝对世界速度**)。

**发现1 —— 证伪"14m 是深度"**:可见帧 EAR 误差**横向主导**(横向 11.7→17.8m vs 深度 7.8→10.7m,1s→6s)。**不是**单目深度歧义,而是**空间定位/grounding-conditioning 弱**(横向原则可学)→ 可见轴**仍有杠杆**(把 grounding/空间信息喂进 EAR),且与 WM 正交。测了才知道,先前"深度"猜测作废。

**发现2 —— EAR 被便宜基线全面碾压**(FDE@6s,丢失帧):

| 分层 | n | EAR | ②零速 | ③CV | persist参照 |
|---|---|---|---|---|---|
| short_straight | 266 | 68.1 | 26.1 | **10.9** | 16.7 |
| long_straight | 182 | 90.0 | 47.5 | **23.2** | 17.1 |
| short_turn | 20 | 53.1 | 34.3 | 35.8 | 27.1 |
| **long_turn** | 27 | 58.1 | 49.9 | **47.5** | 28.2 |

EAR 每一格都被 CV 碾压 **2-6×** → **反应式网络连"匀速外推"都没学到**,退化成烂先验(它没有把"最后看见的运动"前带的机制)。

**发现3 —— WM 价值精确锁定在转向/路口**:CV 在 **short_straight 10.9m、long_straight 23.2m**(直路即使长丢失也够重捕获!)→ 但 **short_turn 35.8、long_turn 47.5m 崩塌**(CV 直冲出弯,优势归零≈零速)。**杀手是转向/路口,不是时长本身**。→ **WM 的差异化价值 = 转向/路口**(路网知道目标顺车道走、路口分叉),直路那块便宜 CV 就解决;并给 WM 立了**必须超越的基线**(转向分层 CV ~36-48m)。

**发现4 —— 数据前置坐实**:转向丢失帧**严重欠采样**(short_turn n=20、long_turn n=27,合计 47 帧)→ **P1(长丢失+路口重现)数据从 nice-to-have 升为 WM 硬前置**(否则转向格连评测都不稳,更训不动 WM)。

### 8.7 修正后的"是否到顶"结论 + 阶梯路线(取代"EAR 到顶→直接上 WM")

- **反应式并没到实用天花板**:CV 碾压 EAR ⇒ 有**免费优化(记忆+CV)**没做;可见横向定位还有 **grounding-conditioning 杠杆**没试。"EAR 到顶"**不成立**——它连 CV 都没学到。
- **WM 只对一个残差有据**:**转向/路口**(CV 也解不了的 36-48m),这是路网结构不可替代处,且有基线可比。
- **阶梯路线**:① **记忆+CV 作策略先验接入闭环**(训练无关,直路出画预测 90→23m)→ ② **修 EAR 可见横向**(喂 grounding/空间 conditioning)→ ③ **WM 只打转向/路口** + **P1 补转向丢失数据**。这套比"EAR 到顶→上 WM"诚实且有力:证明了反应式没到顶、把 WM 精确限定在转向、给了它基线、点出了数据前置。

### 8.8 闭环三臂:记忆+CV 先验接入 → WHERE 被补上、WHICH 成墙(2026-08-19,`rollout/eval_ex_source.sh`)

把 ②③ 从离线代理**接进闭环**:同一 ckpt(`stage2_v5/best.pt`)、同一 held-out(Town05 seed 91000+),**唯一变量是丢失帧谁填 DiT 的 `z_ex` 槽**(`rollout/target_state.py` 估计器 + `policy.py` 门控,**训练无关**)。三臂 = `ear`(原基线)/ `cv_gated`(记忆+匀速)/ `zerovel_gated`(记忆冻结)。架构对照图 `acot_note/three_arm_zex_injection.html`。

| 指标 | ear(基线) | cv_gated | zerovel | 判读 |
|---|---|---|---|---|
| SR | 0.000 | 0.000 | 0.000 | 需 WHERE+WHICH 同时过 |
| track_seconds | 22.4 | **63.4** | 54.5 | 待在目标附近久 2.5-2.8× |
| reacquire_attempts | 42 | **294** | 247 | **7×**:从"丢了就走"→"反复重接" |
| reacquire_success | 0.381 | **0.483** | 0.405 | 真目标重捕获 +10pp;cv>zerovel |
| reacquire_convergence_s | 4.91 | 3.88 | 2.31 | 重捕更快 |
| reacquire_lock_wrong | 0.476 | **0.793** | 0.781 | **变差**:锁 look-alike 更多 |
| mis_follow | 0.723 | 0.805 | 0.732 | 略升(tid 轴) |

**定论(闭环首次把两失败轴分离)**:
1. **WHERE(预测-拦截)被便宜先验治好**:drone 不再丢了就飞走 → track_s 22→63s、重捕获尝试 7×、真目标重捕获 0.38→0.48、收敛更快。**cv>zerovel(速度项有用)**,与离线一致。
2. **但这把 WHICH(语言认领)顶成瓶颈**:待在目标附近=反复扎进 look-alike 群,grounding(mis_follow 0.72→0.80、lock_wrong 0.48→**0.79**)认不准→锁错暴增。基线 lock_wrong"低"只因它**大多超时**(没接近到能锁),非认得准。
3. **SR 仍 0 根因**:先验打通"丢失-重接"循环(WHERE),但每次**认错车**(WHICH)→ 跟不到底。
4. **路线修正**:① **WM 边际价值被压缩**——CV 先验已大面积补 WHERE,WM 只剩**转向/路口残差**,单靠 WM 抬不动 SR;② **WHICH(grounding)升为并列关键**——要动 SR 必须同时强化 grounding(更强 tid / 重捕获时刻语言仲裁);③ **记忆+CV 先验值得保留进闭环**(velocity 有用)。详见 memory `acot-uav-wm-roadgraph`。

---

## 九、瓶颈重构 → re-ID 主线 → 控制修复证伪 → 时序 reID + 客观度量(2026-08-31 → 09-05)

> 一句话:闭环证据把 SR=0 的墙从 **WHERE(预测-拦截)** 彻底移到 **WHICH(look-alike 中的身份重捕获)**;沿这条线做出了一条**单调改进的客观指标阶梯**(crop-DINOv2 → K-gallery → track → conf-gate → 时序EMA),两次"修控制取景"(BC / RL)被证伪,最后落到"**报客观连续指标而非饱和的 latch-SR**"。

### 9.1 实验记录(逐步改进的证据链)

**① REFRAME:WHERE 已解、WHICH 是墙(2026-08-31)。** 4 臂 WHERE 阶梯(ear→cv_gated→road→**road_oracle**=近完美当前真值锚)把 track_seconds 从 24→**65s**,但 **SR 全程 0**、`reacquire_lock_wrong≈0.74 在所有臂平坦**——即"即便 WM/完美目标位,你仍重捕获**错车**"。⇒ WHERE 便宜地饱和(CV 够),瓶颈是**单帧语言 grounding 认不出 look-alike 的同一实例**。

**② re-ID 主线的客观阶梯(旧数据 19 局同种子配对,`continuous_metrics.py`)** —— 逐步单调改进:

| 客观指标 | cv+xattn(语言) | +vlmpool-EMA | +DINO-K2 | track(可部署) | +conf-gate |
|---|---|---|---|---|---|
| 正确跟踪帧占比(1−mis_inst) | 0.309 | 0.452 | 0.622 | 0.650 | 0.626 |
| 最长错跟时长(均值,s) | 11.98 | 7.06 | 4.40 | 4.27 | **3.53** |
| 无>5s错跟局占比 latch@5s | 0.105 | 0.263 | 0.684 | 0.737 | **0.789** |
| ID 切换 | 16.8 | 16.3 | 18.4 | 11.5 | 11.9 |
| 每事件重捕获率 q_reacq | 0.55 | 0.63 | 0.65 | 0.58 | 0.53 |
| SR(latch,保守操作点) | 0.00 | 0.00 | 0.105 | 0.053 | 0.105 |

正面表述:**正确跟踪时间占比翻倍(0.31→0.65)**、**最长错跟时长缩短 3.4×(12s→3.5s)**、**无长错跟局占比升 7.5×(0.11→0.79)**——全部客观、无手挑阈值、领域标准。

**③ SR 天花板诊断(oracle 分解)。** D1(专家取景+reid)=**0.579**、D2(oracle 身份+track)=**0.579** → **真上界 0.58**,其中 **~42% 是场景难度地板**(超长遮挡,oracle 也败);可部署 0.105 的 gap = **identity×control 恶性级联**(框偏→reid 看偏心 crop→误认→飞错车→丢目标→框更偏)。

**④ 修"控制取景"的两次证伪。** 级联诊断出"框偏"是主导边,遂两次尝试让 DiT 把目标框正:
- **BC-居中**(flow-matching 微调 DiT 末层朝几何居中动作;blend 0.4/0.7/1.0):centering 机械上升,但 **track_seconds 崩(33.8→2.2)**、mis/lock_wrong 升,**无甜点**。
- **Stage-3 RL E0**(SVG 可微多步 rollout,R_center):训练里 R_center **0.78→0.98**,**闭环复现完全相同失效**。
- **定论**:居中在身份**下游**——激进居中把 z_ex(reid 认定车的 CV,可能错)的误差**放大**成大幅甩头。**RL 也修不了**;控制不是 SR 杠杆,身份才是。

**⑤ 时序 reID(query 侧,本轮新增)。** gallery 模板已时序,但 query 一直是**单帧 argmax**。改为按稳定 actor 累积匹配分 EMA、时序积分提交。18 局配对确认(`reid_tavg0` vs `tavg0.4`):**正确跟踪占比 0.627→0.715(+0.088,8-ep 与 18-ep 双复现)**、id_switches 15.7→10.2(显著)、latch@2s/@3s 升;**latch-SR 未动(场景地板)**、q_reacq/latch@4-5s 轻微退化(EMA 跨长丢失持陈旧证据)。甜点 **tavg=0.4**(短记忆:够平滑单帧噪声,又够快纠错)。

**⑥ 客观度量方法学(关键转向)。** latch-SR(90s 内任意 ≥2s 错窗→永久失败,SR≈(1−q)^N)由**最坏错窗+场景地板**主导 → **饱和、无分辨力**(temporal 身份更好却 SR→0)。改用**连续/分段客观指标**(正确跟踪占比 / 最长错跟时长 / 重捕获率&延迟 / 丢失时长 / ID切换)——既更正面又更严谨;SR 保留为一个保守操作点。

### 9.2 模型设计完善(本阶段落到 `rollout/policy.py`)

- **`tid_head=reid`**:判定从"哪个候选最符合**语言描述**"(look-alike 都部分符合)→"哪个是我一直跟踪的**同一实例**"(外观记忆关联),绕开单帧语言 grounding 的天花板。
- **crop-DINOv2 特征**(替代 Qwen 粗网格池化):实例判别力强,look-alike 余弦 0.877→**0.614**;身份信息在 26px 像素里,粗网格池化把它浪费了。
- **K-view 多视角 gallery(K=2)**:存目标多个过去视角为时序模板 + top-m 均值匹配 → 视角鲁棒;离线首破 0.5 墙。
- **`ex-source=track`(Plan A)**:reid 认定车的 CV 驱动 z_ex/控制,去掉 GT-WHERE 拐杖 = 全可部署。
- **置信门控 conf-gate**(`--conf-tau`):top1−top2 margin 低 → 不 commit-fly,保持承诺+航位推算 → 打破"认错→飞错车→丢目标"的控制耦合(SR 0.053→0.105)。
- **★时序 EMA(`--reid-tavg`)**:候选 `cand.idx` 是稳定 actor → 按 actor 累积 `score=α·prev+(1−α)·logit`,用时序积分 argmax(≠单帧)提交(把 MOT 的**轨迹级证据累积**原则补进 query 侧)。
- **搁置的设计尝试**(证伪但有价值):BC-居中(`train/bc_center_train.py`)、Stage-3 RL E0 SVG(`train/rl_e0_center.py`,可微运动学+投影+R_center)、奖励 `train/rl_reward.py`(anti-hack 单测过)——结论:控制侧修取景不可行,留作 future work / 负结果证据。

### 9.3 踩过的坑(避免重复浪费算力)

**方法学 / 严谨性**
- **cherry-picked 子集 vs 全集 aggregate**:warm-start 首次用硬全败子集(91000-05)对比旧 19-ep 全集 → 假警报"SR 退化"。**Guardrail:永远同种子配对**。
- **短 smoke 骗人**:frame-gain 3ep@400 显示"有效"(SR 0.5),6ep@900 反转(放大身份错)。**小 n / 短步 smoke 不可下结论**。
- **frame-gain 符号错**:用了 −30(∝−off),几何+专家 corr 应为 +off;但**两符号都失败**——固定增益太弱且放大 reid 错。
- **latch-SR 无分辨力**:n=8 下 SR=0/1 次成功是纯噪声,误导选参;须用连续指标。

**基础设施**
- **数据被 root 清理删除**(mvp_full_v5,还带走 151GB ctx 缓存)→ 重生成(注意新数据 ~333帧/ep 短于旧 ~1500,不可与旧 run 同框配对)。
- **容器禁占 172.x 网段**(host 默认池保留)→ `docker-compose.yml` pin `cyh-carla-net` 到 `192.168.240.0/24`。
- **CARLA 未自启**:cyh-carla 只跑 `sleep infinity`,需手动 `CarlaUE4.sh -RenderOffScreen -graphicsadapter=1 -carla-rpc-port=2012`;**port-open≠ready** → `set_timeout(30→60)`;`-graphicsadapter` 选 Vulkan 渲染 GPU(忽略 CUDA_VISIBLE_DEVICES)。
- **reid policy-server 缺 `timm`**(镜像没装,DINO 权重在 HF 缓存)→ 启动前 `pip install timm==1.0.29`。
- **docker `-v` 变量未展开** → 首次 NO DATA;用显式绝对路径。

**RL / 可微实现**
- **`action_sample` 带 `@torch.no_grad`** → 切断 SVG 梯度(loss 不 require grad)→ 写本地带梯度采样器 `_flow_sample_grad`。
- **缓存/索引 H 对齐**:缓存用 `d["H"]=16` 做 `range(0,n-H,stride)`,不是 max_offset;从缓存 `dims["H"]` 读,check 显示 6282==6282。
- **可微 rollout 太慢**:sample_steps8×roll_h6=48 forwards/batch × 524 batch → 45min 0 epoch;降到 steps2×roll_h4 + 子采样 1500 帧 → 0.76s/batch(~30×)。
- **torch 投影须与 numpy 对齐**:先 `check_proj`(误差<0.001px)再训,杜绝污染。
- **continuous_metrics 未计分局缺字段** → 跳过(`mis_follow_inst` 不在则丢该种子)。

---

## 十、外部 baseline 对照 → 诊断驱动的借鉴升级(2026-09-05 → 09-07)

> 一句话:把外部方法接进**同一闭环只换 WHICH**,发现我们**非 SOTA**但"外观/时序关联是杠杆"被独立证实;据此做**可借鉴设计排序**,借 DeepSORT 的运动共识(#1)使 **SR 翻倍**、系统升为最佳均衡;语言兜底重锚(甲)彻底证伪。

### 10.1 实验记录(逐步)
**① 定稿内部证据(20ep 同批次)**:主阶梯 l1_xattn→l5_temporal(正确跟踪 0.251→0.738、最长错跟中位 9.7→3.0s)+ 负结果 BC/RL 均失败 + H0(0.579 vs 0.715)+ 客观指标方法学(latch-SR 饱和)。见 §九 + `paper_data_inventory.md`。

**② 外部 baseline(接同一闭环,只换 `tid_head`)—— 诚实:我们非 SOTA。** `assoc`(DeepSORT/OC-SORT)+ `DAM4SAM`(CVPR'25 SOTA,独立 env+socket 桥)。19ep 同种子:track_correct_frac l5 0.738 / assoc-deepsort 0.750 / **DAM4SAM 0.941**;SR l5 0.105 / **assoc-deepsort 0.211** / DAM4SAM 0.105。**assoc-deepsort SR 更高、DAM4SAM 正确率更高(但 16s 粘滞)** → 我们非最优;**但所有外观/关联法(0.67-0.94)≫ 单帧 xattn(0.25)→ WHICH-not-WHERE + "外观/时序关联是杠杆"被外部方法独立证实**(=真贡献)。iKUN/JointNLT 零样本域差,转 future work。

**③ 语言兜底重锚(甲)彻底证伪。** 全可部署 GT-free gallery(committed)身份大崩(0.715→0.363);语言重锚**不帮反差**(→0.225);时序 EMA 抢救**部分降噪但仍跨不过基线**(→0.291<0.363);**oracle 重锚 0.741** 证明**机制有 headroom,但语言(即便时序积分)是太弱的仲裁**——再次坐实单帧语言天花板。

**④ 借鉴设计排序 → borrow#1 运动共识 → SR 翻倍。** 排序原则=**攻真正瓶颈(级联/何时信任身份)非识别精度**(全迭代反复证明单纯提识别不移 SR)。#1=借 DeepSORT(唯一 SR 更高的外部法)把运动做成第二共识信号融进 reid。实现+单测→ **SR 0.105→0.211(2×)**、正确跟踪 0.738→0.835、q_reacq 0.471→0.635、最长错跟中位 3.0→1.8s——多指标一致改善(真信号)。**RMOT 扫描确认稳健**(0.3/0.5/0.7 全优于 l5,correct_frac 单调↑)+ 定甜点 **RMOT≈0.3(SR 0.263)**;RMOT=0.7 过约束(连续最优却 SR 掉,长丢失 CV 陈旧→伤重捕获,又一 latch 钝化实例)。**完整系统(reid+temporal+motion)现为最佳均衡**:追平 assoc-deepsort SR、正确率/错跟更优、远胜 DAM4SAM 的 SR/错跟。

**⑤ borrow#2 共识置信度调制控制(进行中)**:commit-fly 门在 margin 上增"被选候选须运动一致"→ 拦截"外观骗过但空间离谱"的 commit-fly。实现+单测过,20ep 评测中。

### 10.2 模型设计升级(落到 `policy.py`)
- **★borrow#1 运动共识**(`--reid-motion` RMOT):reid 分数 = 外观余弦 + RMOT·运动一致性(CV 图像位预测软门),EMA 前融合 → 空间不一致的 look-alike 赢不了外观。补上 ours 纯外观的弱点(DeepSORT 有、我们没有)。
- **★borrow#2 共识门控**(`--conf-consensus`):`_track` commit-fly 门 = margin≥conf_tau **且** 被选候选运动一致 → 外观/运动分歧时保持+搜索(强化"身份→控制"耦合,ours 独有)。
- 最新完整 WHICH:crop-DINOv2 + K-view gallery + ★运动共识融合 + 时序 EMA + K帧滞回 + ★共识门控 → committed 身份 → z_ex → DiT。架构图 `acot_note/architecture_latest.html`。

### 10.3 踩过的坑
- **外部集成**:DAM4SAM 官方 cu121 撞驱动470→改 cu118+torch2.1+SAM2;numpy 必须<2(opencv-python 5.0 会拉回2.x→只留 headless4.10);vot 精确版 0.7.1/trax4.0.2;PIL 非 ndarray。独立容器+socket 桥(复用 bridge.py)。
- **CARLA 崩**:up 2 天+重负载后模拟器进程挂(端口关)→ 交接/评测加"崩则自动重启"守卫;单 CARLA 严格串行,交接前查 `pgrep run_rollout`=FREE。
- **诚实纪律**:外部 baseline 胜我们的指标(assoc SR、DAM4SAM 正确率)如实报、不写 SOTA;RL/DAM4SAM 的 latch@5s"好"是近零跟踪假象须点破;单点强结果(RMOT0.5)必须邻域扫描确认非幸运数;bash `local a=$1 b=${a}` 在 set -u 下报未绑定→拆开。

---

## 附:关键证据与产物索引

| 项 | 位置 |
|---|---|
| 语言必要性 3-seed 消融(6.16 vs 6.23%) | `runs/seed_exp/{ac,nolang}_s{0,1,2}` |
| **★ v5 E1 干净证明(有58% vs 无72%,gap~13pp)** | `runs/train_v5{,_nolang}.log` · `train/config_v5{,_nolang}.yaml` · `train/train_v5_e1.sh` |
| **身份可读性探针(by-episode 6% vs 同分布 55%)** | `train/fullframe_probe.py`(`--random`/`--recrop`) |
| **省磁盘配方(layer24-only + fp16 缓存)** | `train/backbone_kv.py`(fp16)+ `train/train_v5_e1.sh` |
| 迭代总记忆(v3/v4/v5 arc + 分辨率墙) | memory `acot-uav-language-necessity` |
| 位置捷径探针(~3-5%) | `train/benchmark_leak_probe.py` |
| **★ 公平 no-WM 基线(EAR 出画预测失败→WM 有据)** | `train/train_v5_fair.sh` · `runs/stage2_v5_fair/stage2_best.pt`(mis_follow 0.582) |
| **★ 出画预测判据工具(EAR FDE + 拦截 cos + 深度/横向分解)** | `train/intercept_eval.py`(可见/丢失×视界 + persist + depth/lateral) |
| **★ 便宜基线阶梯(②零速/③CV,丢失时长×转向分层)** | `train/baseline_deadreckon.py` · 结论 `runs/baseline_eval.log`(CV 碾压 EAR、WM 锁定转向) |
| WM 路网图预测器设计(方案A/文献落点/接口) | `related-survey/wm-roadgraph-design.md` · `acot_note/wm_scheme_compare.html` · memory `acot-uav-wm-roadgraph` |
| **★ 闭环记忆+CV 先验(z_ex 门控注入,训练无关)** | `rollout/target_state.py` · `rollout/policy.py`(--ex-source)· `rollout/eval_ex_source.sh` |
| **★ 闭环三臂结果(WHERE补上/WHICH成墙)** | `rollout/eval_out/ex_{ear,cv_gated,zerovel_gated}.json` · 架构图 `acot_note/three_arm_zex_injection.html` |
| EAR 定位地板修复对比 | `acot_note/stage1_camera_frame_fix.html` |
| 数据流水线 / VLM 融合图 | `acot_note/data_pipeline.html` / `vlm_fusion.html` |
| 完整实验设计(H0-H6、四条件、指标族) | `acot_note/experiment-design.md` |
| 数据修复规格(b/c/d) | `acot_note/data-fix-spec.md` |
| 设计文档(§2.3 场景 / §6.1 招牌 / §6.3 核心贡献) | `acot_note/acot-uav-design.md` |
| **L1/L2 工具箱**:world-model 方法学综述(因果AR抗drift/记忆三派/SEP归派/drift上限) | `acot_note/world-model-survey.md` |
| **L3 背书**:语言 which-way 综述(语言条件轨迹预测/SWM/统一 framing) | `acot_note/language-seeded-prediction-survey.md` |
| **L2 定位/切割**:SEP 时不变路网先验(2×2、逐件切割) | `related-work-and-positioning.md` §4D |
| **L3 定位/切割**:语言双功能 which+which-way | `related-work-and-positioning.md` §4E |
| **L2/L3 实验**:SEP guard(§2.0c)、which-way 三臂+冗余 guard(§2.0d)、H7/H8、M11/M12、R9–R12 | `acot_note/experiment-design.md` |
| **★ REFRAME 诊断全链**(WHERE 解/WHICH 墙、D1/D2 上界 0.58、级联、时序 reID、客观度量) | memory `acot-uav-reid-reframe` |
| **★ re-ID 客观阶梯 run**(xattn→vlmpool→DINO-K2→track→conf,旧数据 20ep) | `rollout/eval_out/road_{cv_gated,reid,reiddino,trackreiddino,trackreiddino_conf05}.json` |
| **★ 时序 EMA re-ID + 甜点扫描**(policy `reid_tavg` / `--reid-tavg`;tavg 0/0.4/0.5/0.7) | `rollout/policy.py` · `rollout/eval_reid_temporal.sh` · `eval_out/reid_tavg{0,0.4,0.5,0.7}.json` |
| **★ 客观连续/分段指标工具**(正确跟踪占比/最长错跟时长/重捕获/latch@Ts) | `rollout/continuous_metrics.py` |
| **★ 控制修取景的两次证伪**(BC-居中、Stage-3 RL E0 SVG,均搁置) | `train/bc_center_train.py` · `train/rl_e0_center.py` · `train/rl_reward.py` · `rollout/eval_bc_center.sh` |
| Stage-3 RL 设计(obs/reward/action、GRPO、E0 门、为何 RL≠BC) | `acot_note/stage3-rl-design.md` · memory `acot-uav-stage3-rl` |
| 当前测试架构图(track+reid-DINOv2-K2+conf+时序EMA) | `acot_note/reid_temporal_architecture.html` |
| reid-DINOv2+gallery 流水线图 | `acot_note/reid_dinov2_gallery_pipeline.html` |
