# ACoT-UAV-Track 相关工作与定位（Related Work & Positioning）

> 整合 5 轮 deep-research（2026-07-29/30，每轮 96–106 agents、3 票对抗验证）为单一事实源，供写 intro / related work / 竞品切割 / H0 论证。
> **引用纪律**:每条标注 **venue + 发表状态**(peer-reviewed 已发表 vs arXiv 预印本);预印本数值写作时复核;标 ⚠️ 者为待核/易错点。
> 配套:`acot-uav-design.md`(设计)、`experiment-design.md`(实验/H0 gate/指标)、`positioning-analysis.md`。

---

## 0. 三个必记的纠错点(避免写作翻车)

1. **UAV-Track VLA ≠ 本工作。** UAV-Track VLA(arXiv 2604.02241,BIT/CASIA/BUPT 等)是**别的团队**的空中跟踪 VLA,是本工作要精确切割的**头号竞品**;ACoT-UAV-Track 是把 ACoT-VLA 机制迁移到 UAV。调研 agent 曾因"都是 CARLA 空中 VLA"误判为同一篇——**务必写清谁是谁**。
2. **CosFly-VLA 是虚构的**,勿引(真实工作是 CosFly-Track 数据集 2605.17776 与 CosFly CARLA 管线 2605.19120,均非 VLA)。
3. **TrackVLA 已发表**(CoRL 2025,peer-reviewed)——早期"状态存疑"已解除;TrackVLA++ 仍为预印本。

---

## 1. 定位:三层漏斗 + 四轴交集

```
空间智能(Fei-Fei Li 立为 AI 下一前沿) 
  └→ 具身空间智能(4 篇 survey 一等分支:VLN + VLA)
       └→ 空中俯视几何下的空间推理(AirGroundBench 点名 open problem:跨视角对齐/几何一致/闭环)
            └→ ACoT-UAV-Track:空中主动飞行 + 语言指定车辆 + 视觉相似车消歧 + 具身闭环
                 = 该 open problem 的一个未被占据的四轴交集
```

**可辩护的新颖性主张(收窄,to-our-knowledge)**:截至 2026-07,"**空中主动飞行 + 语言指定目标车辆 + 视觉相似干扰车消歧 + 具身闭环**"这一四轴交集无先例发表。逐轴已占据情况见 §6。

**★ 升级:杀手场景 = 长丢失预测-拦截重捕获(见 §4B + design §6.3)**。把静态消歧推成动态重捕获:语言作为**决定性/唯一**消歧信号出现在"长丢失后重现"这一刻(位置捷径 ∧ 外观捷径**同时失效**),耦合无路网约束的 UAV 预测拦截。deep-research 确认该三者叠加**是真空白**(每个零件已被占据但互相脱节、且重捕获时刻均无语言)。

**核心论点(intro 用)**:当前 SOTA MLLM 在**空中视角几何一致性**上系统性失败(AirGroundBench),而 ACoT-UAV-Track 用 **EAR(动作空间 3D 坐标)+ IAR(单目 3D 先验)+ 相机位姿注入(参照系对齐)** 把空间推理落到闭环可执行的 3D 量,并要求语言在**去相关 + 连续性断裂(重捕获时刻)**成为消歧的唯一信号。

---

## 2. 相关工作 A｜具身/主动视觉追踪(谱系:RL → VLA)

> 定义:智能体主动控制自身运动持续跟随目标(闭环控制),区别于只输出 bbox 的被动 SOT/MOT。

| 工作 | venue/年 | 状态 | 域 | 核心贡献 | 关键指标 | 开源 |
|---|---|---|---|---|---|---|
| End-to-end Active Tracking (Luo et al.) | **ICML 2018** | ✅已发表 | 地面/虚拟 | 首个端到端 RL 主动追踪 | — | — |
| **AD-VAT / AD-VAT+** (Zhong et al., PKU) | **ICLR'19 / TPAMI'21** | ✅已发表 | 地面·UnrealCV | 非对称对抗 tracker-vs-target 对偶;gym-unrealcv 环境族 | Accumulated Reward / Episode Length | ✅ |
| Distraction-Robust Active Tracking | ~ICML'21 (2106.10110) | 预印/会议 | 地面 | 抗干扰主动追踪 | — | — |
| **TrackVLA** (PKU-EPIC/Galbot) | **CoRL 2025** | ✅已发表 | 地面/第一视角 | 共享 LLM 骨干:LM 识别头 + anchor-diffusion 轨迹;建 EVT-Bench + 1.7M 样本 | Gym-UnrealCV 零样本 SOTA;10 FPS | ✅ wsakobe/TrackVLA |
| **TrackVLA++** (PKU 等) | arXiv 2510.07134 (2025.10) | ⚠️预印本 | 地面/四足 GO2 | **Polar-CoT**(相对位置→极坐标 token)+ 门控 **TIM** 记忆 | EVT-Bench DT +5.1/+12;~74% | 项目页 |
| **CoMaTrack** (Amap/Alibaba) | arXiv 2603.22846 (2026.03) | ⚠️预印本 | 室内·Habitat | 竞争博弈多智能体 RL(SFT+GRPO) | EVT-Bench 92.1/74.2/57.5 (STT/DT/AT) | ✅ CoMaTrack-Bench |
| **UAV-Track VLA** (BIT 等) | arXiv 2604.02241 (2026.04) | ⚠️预印本 | **空中·CARLA** | π0.5 + 双分支(3D grounding 辅助头 + flow-matching)+ 时序压缩;25-step 位移;>890K 帧 | SR 61.76%/269.65 帧(**行人"Far"域,非车辆**) | — |
| **DeTrack**(Hu 等)✅号已核 | arXiv 2605.17451 (2026.05) | ⚠️预印本 | **空中·UE4.27/AirSim** | *Drone-embodied Tracking*;主动闭环 + 避障;AaDWorlds(altitude-aware **dual world model**);**纯视觉,无语言**;动态遮挡物(非相似车) | 11,368 轨迹 benchmark | — |

**读数**:范式演进 = 对抗 RL(2018–21) → 语言条件具身 VLA(2025–26);后者主流栈 = **VLM 骨干 + flow-matching/diffusion 动作 + 空间推理 CoT + 时序身份记忆**——本工作 EAR/IAR/AGP 正落此主线。**空中闭环有 UAV-Track VLA(语言) 与 DeTrack(无语言,world-model) 两家**;消歧(DT/AT)只在地面/室内。⚠️ **DeTrack 撞"具身跟踪/world-model"两词但无语言、无相似车消歧,须切清;勿混 NeurIPS'24 同名被动 SOT(2501.02467)**。

---

## 3. 相关工作 B｜语言/referring 跟踪与消歧

| 工作 | venue/年 | 状态 | 任务 | 说明 |
|---|---|---|---|---|
| **RMOT / Refer-KITTI** (Wu et al.) | **CVPR 2023** | ✅已发表 | 被动·驾驶域 referring MOT | 语言指代驱动 MOT,输出任意数目标;Refer-KITTI 18 视频/818 表达/10.7 目标·表达。**被动、无闭环** | ✅ wudongming97/RMOT |
| **AerialMind**(Chen 等)✅号已核 | **AAAI-26 已发表**(2511.21053→Proc.AAAI 40(4):2805-2813) | ✅已发表 | **被动·UAV 视角 referring MOT** | 首个 UAV **RMOT** benchmark(扩 VisDrone/UAVDT);语言指定 referring 消歧(752 no-target+458 reasoning 表达);方法 HawkEyeTrack;**不控制无人机** | 数据集/COALA | 
| **EVT-Bench** (随 TrackVLA) | CoRL'25 起 | ✅ | 室内具身 | **STT/DT(相似干扰消歧)/AT(歧义语言)** 三 split;闭环指标 SR/TR/CR/EL | — |
| **CoMaTrack-Bench** | 2603.22846 | ⚠️预印本 | 室内具身 | "首个开源 Habitat 语言条件竞争式 EVT 协议" | ✅ |

> ⚠️ **AerialMind ≠ UAVNLT**(两篇不同 UAV 语言跟踪工作);AerialMind 现为 **peer-reviewed AAAI-26 强竞品**,做 UAV 语言 referring 消歧但**被动**,与本工作差在**主动飞行闭环**。

**读数**:语言消歧维度只存在于(a)室内具身(EVT-Bench DT/AT)或(b)被动驾驶(Refer-KITTI)。**空中闭环车辆消歧 = 需自建 benchmark;EVT-Bench 的 STT/DT/AT 是可移植协议模板**(=课程 2a/2b/2c,见 experiment-design §3.4)。

---

## 4. 相关工作 C｜被动 UAV / NL 视觉跟踪(可复现基线,兜 R7)

| 工作 | venue/年 | 状态 | 说明 | 开源 |
|---|---|---|---|---|
| **ORTrack** | **CVPR 2025** | ✅已发表 | UAV 航拍 SOT·抗遮挡;六 UAV 集实时 SOTA;ORTrack-D 蒸馏 | ✅ wuyou3474/ORTrack |
| **TCMLTrack** | **Sci. Reports 2025** | ✅已发表 | **UAV 语言引导**跟踪;六集 acc .819/succ .654/61FPS;**自承认消歧失败** | — |
| **UAVNLT** | **Electronics(MDPI) 2024** | ✅已发表 | **UAV 语言引导车辆**跟踪;2000 序列/四城市 | ✅ 代码;⚠️数据"coming soon" |
| **JointNLT** | **CVPR 2023** | ✅已发表 | grounding+track 统一 | ✅ lizhou-cs/JointNLT |
| **CiteTracker** | **ICCV 2023** | ✅已发表 | 图文关联视觉跟踪 | ✅ NorahGreen/CiteTracker |
| **WebUAV-3M** | **IEEE TPAMI 2023** | ✅已发表 | 百万级 UAV SOT 基准(bbox+NL+audio);4500 视频/3.3M 帧/223 类 | ✅ |
| **TNL2K** | **CVPR 2021** | ✅已发表 | 语言消歧跟踪基准;2000 序列/1.24M 帧 | ✅ |
| 补充 | — | — | UAV123 榜首 **LoRAT-g-378**、UAV SOT **CGTrack** | — |

---

## 4B. 相关工作 F｜长丢失预测-拦截重捕获(★升级创新点专项,deep-research 2026-07-30)

> 升级后的杀手场景(design §6.3):目标出画/遮挡致**视觉连续性彻底断裂** → EAR 预测未来 3D 位置、**无路网约束抄近路拦截** → 重现时 ≥2 辆 look-alike 共视,**位置捷径+视觉连续性+外观三者同时失效** → 语言成**唯一**消歧信号。
> **判决:该三者叠加的交集是真空白;但每个"零件"都已被占据 → 新颖性须押在耦合,不可押单件。** 下表逐件切割。

| 零件 | 代表(venue/状态) | 用语言? | 定位:切割 or 支持 |
|---|---|---|---|
| **预测未来位置+飞向拦截("抄近路")** | Fast-Tracker(**ICRA 2021**,开源) / PN·ATPNG 制导(**Intelligent Service Robotics, Springer 2021–22**) | ❌ 纯几何,身份已知 | **must-cut**:拦截机制本身不新 → 新颖性别押在"预测拦截";且它们是"保持在视野/跟随",非"完全丢失后 rendezvous" |
| **FOV 离开-重入时刻干扰消歧** | **DAM4SAM / SAM2.1++**(**CVPR 2025 / IJCV 2026**,开源) | ❌ 外观 anchor 记忆(RAM+DRM) | **最强"时刻"先例,须切**:它**明确点名**"目标离开再进入视野时外部干扰尤其难"——正是你攻的时刻,但用外观解、无文本编码器 → 语言消歧此刻**空位** |
| **完全丢失后重检测** | GlobalTrack(**AAAI'20**)、Siam R-CNN(**CVPR'20**)、LTMU(**CVPR'20**,VOT-LT 冠军)、SiamMDTP(**Info Sciences 2025**) | ❌ 全外观/运动(模板匹配+EKF 轨迹+Meta-Updater) | 支持:重识别只有外观/运动先例 → 语言重捕获空位 |
| **中途持续用语言**(⚠️最锐利 must-cut) | **Feng et al.**(**WACV 2020**,NL-conditioned proposal + LSTM,box-free) | ✅ 但用于**逐帧定位/初始 grounding** | **必须显式切割**:"持续定位"(已占)≠"丢失重现时唯一消歧信号"(空位);它未测"预测出画拦截后 + 多 look-alike 的重捕获" |
| **具身 VLA 抗干扰+遮挡重捕获** | **TrackVLA++**(2510.07134 ⚠️预印本) | ⚠️ 上游 VLA 条件,但**重捕获靠视觉 TIM 记忆(仅视觉 embedding)+ Polar-CoT 空间推理,非语言重调用** | 最近邻竞品:留了"语言重捕获"+"UAV 预测拦截"两个空位;地面反应式跟随,无抄近路 |

**未占据核心(论文主张精确到此句)**:
> 语言作为**决定性/唯一**消歧信号,出现在**长丢失后重现**这一刻(位置捷径 ∧ 外观捷径**同时失效**),并与**无路网约束的 UAV 预测拦截**耦合。DAM4SAM 有"离开-重入"但外观解、Fast-Tracker 有"预测拦截"但身份已知、Feng 有"中途语言"但逐帧定位——**无一篇让三者在同一时刻叠加**。

**⚠️ 写作前必做的显式切割**:
1. **cut Feng et al. (WACV 2020)**:明写"我们非首个中途用语言,而是首个把语言作为**丢失重现时的唯一消歧信号**"。
2. **cut DAM4SAM (CVPR 2025)**:它点名"离开-重入"时刻但用外观 anchor;我们在该时刻用语言。
3. **cut Fast-Tracker/PN(ICRA'21/Springer)**:预测拦截机制本身已有,我们的新在"拦截 + 语言重捕获消歧"耦合。

### 4B.1 pre-VLA NL-tracking「语言作用时机」审计(✅子维度4已锁死,deep-research 2026-07-30)

> 结论:**"语言作为长丢失后重现、多辆 look-alike 共视时的唯一消歧信号"在 pre-VLA NL-tracking 里是真空白(high 置信)**。唯一重叠="语言参与全局重检测"(TNL2K/Li 2017),但那是**外观驱动重定位**(重匹配同一模板),非语言细粒度消歧。

| 工作 | venue | 语言作用时机 | look-alike 用语言消歧? | 定位 |
|---|---|---|---|---|
| **Li et al.** Tracking by NL Specification | **CVPR 2017** | 每帧全局定位 + **丢失后 language re-start** | ❌ 重匹配同一外观模板;自承语言单独混人 | must-cut(全局重检测重叠) |
| **TNL2K (AdaSwitcher)** | **CVPR 2021** | 初始 + **失败切回语言 grounding 全局重检测** | ❌ 通用 global search | must-cut(最强重叠) |
| **JointNLT** | **CVPR 2023** | 语言全局定位**仅首帧** | ❌ 无丢失后重 grounding | 切割:歧义只能首帧改 |
| **SNLT** | **CVPR 2021** | 每帧语言重排 proposal | ❌ 无丢失重捕获 | — |
| **QueryNLT** | **CVPR 2024** | **持续跟踪中**语言分"白鸟 vs 小黄鸟" | ⚠️ 有消歧但**仅连续跟踪,非丢失重现**,无重检测 | must-cut(消歧轴最近) |
| **DecoupleTNL** | **ICCV 2023** | 首帧+每帧局部 | 🟢 **正面力证**:论证语言分支**最大化同类相似度**→干扰分离交给**视觉** | **引作正面弹药** |
| DTVLT | 2024 ⚠️预印本 | 每~100帧周期性 dense text(按 cadence 非丢失触发) | ❌ | — |

**★ 论文 cut 措辞(可直接改写用)**:
> "先前 NL-tracker 用语言做首帧指定与逐帧/持续定位;两篇(TNL2K、Li et al. 2017)在跟踪失败后另调用语言做**全局重检测**——但每次都是靠外观相似度的全局搜索找回**同一视觉模板**;DecoupleTNL(ICCV 2023)更表明语言分支倾向最大化同类相似度、把干扰分离**交给视觉**。**没有任何 pre-VLA NL-tracker 把细粒度描述用作在长丢失后重现、多辆共视 look-alike 中挑出正确目标的唯一消歧信号**——正是本工作的设定。(与 QueryNLT 区分:它有 look-alike 消歧,但只在连续跟踪、从不在丢失重现时刻。)"

**残余 caveat(诚实)**:本轮未独立核 GTI / Feng WACV'19 / CTRNL / TNLTrack / CiteTracker / UVLTrack / All-in-One(由 TNL2K "NL-initialized trackers" 表征推断);"真空白"是对已核子集的 absence-of-evidence,论文写 "to our knowledge"。

---

## 4C. 范式定位｜VLA ↔ World-Action Model(WAM)(deep-research 2026-07-31)

> 用于回答"我们更偏 VLA 还是 WAM"。**诚实落点:光谱轻量端的 world-model-augmented VLA**(见 design §1.2 范式定位块)。

**三方边界(NVIDIA WAM glossary + 3 篇 2026 survey 一致)**:
| 范式 | 定义性判据 | 代表 | 我们 |
|---|---|---|---|
| 纯 VLA | 反应式 obs→action,不建模世界动力学 | π0、OpenVLA、**ACoT-VLA**、TrackVLA | 骨干符合 |
| World Model | 预测未来观测/状态,**本身非策略** | Dreamer/DreamerV3、JEPA/V-JEPA、Genie、GAIA、Vista | ❌ 不符 |
| **WAM / world-model-augmented VLA** | **联合**预测未来世界状态 + 动作 | 见下 | 🟡**部分**(EAR 预测目标状态,非观测) |

**WAM 是真实但"新兴"术语**:NVIDIA(glossary + GR00T/DreamZero,**厂商术语**)命名;2026 survey/tutorial 形式化(2605.12090、2607.00836):"联合未来状态+动作分布,参考架构 Joint Video-Action Diffusion Transformer"。第三方评为 emerging、非唯一学术标准 → **引用时标注"新兴术语"**。

**融合先例(证明"world-model-augmented VLA"是已发表范畴,融合本身不新)**:
| 工作 | venue/状态 | 机制 | 备注 |
|---|---|---|---|
| **UWM** | **RSS 2025** ✅,开源 | action-diffusion + video-diffusion 统一,可当 policy/forward/inverse/video | 最强 peer-reviewed 中间地带 |
| **DriveWorld-VLA** | **ICML 2026 poster** ✅ | LLM 隐状态共享隐空间做 imagination + action | 驾驶域 |
| **DUST** | 2510.27607 ⚠️预印本 | **"world-model augmented VLA"**(直接命名);双流扩散**联合**采样动作+观测 | 术语出处 |
| **WorldVLA** | 2506.21539 ⚠️预印本 | "action world model";预测未来**图像**改进动作 | 观测预测(比 EAR 更"world model") |
| Survey: World Model for Robot Learning | 2605.00080 ⚠️预印本(Abbeel/Malik/Wu 等) | §3.5 "Unified VLA" 列为活跃子线;论证反应式 VLA 缺预测结构 | 合法化中间地带 |
| Survey: (Pure-)VLA | 2509.19012 ⚠️预印本 | 把 world-modeling 当 VLA **组件**,称"proto-world model" | 支持"预测组件 VLA 仍是 VLA" |

**定位结论(写进 intro/limitation)**:
- ✅ 用 **"predictive / world-model-augmented VLA"**;EAR = 辅助预测模块。
- ❌ **不自称 WAM/world model**:WAM 教科书判据预测**观测/视频**,EAR 只预测**目标状态**、无 imagined rollout → 只部分满足。
- **novelty 押在**:域(空中具身跟踪)+ **状态-动作同构洞察**(EAR 拦截点=世界预测∧动作)。**⚠️ 同构洞察未见直接先例(可作任务特定新 framing),但呼应 goal-conditioned RL(state=goal)/pursuit-as-prediction → 非全新原理,别 claim "首个融合"**。
- caveat:WAM 术语新兴 + 多为 2026 预印本(仅 UWM/DriveWorld-VLA 明确 peer-reviewed);投稿前复核 arXiv 号。

---

## 4D. 相关工作 G｜结构化环境先验(时不变路网) vs 时变身份绑定(路线 B 专项,2026-08-06)

> 背景:方案把 EAR 从"运动学目标状态预测"升级为**从输入航拍视频在线推断的结构化环境先验 SEP(Structured Environment Prior)**——只预测**时不变**的路网/车道结构,作 EAR 拦截点预测的先验;**不建完整世界、不预测观测像素、不含身份信息**。**自身侧避障已删**(≥22m 高空俯视设定下楼/树碰撞被设计掉、无任务牵引;若未来降飞行包线穿楼再议)。
> **核心组织原则(回应"where/which 能否按时变性切"):可以,且这是最强 framing——把因子按"时间平稳性 × 身份性"分成 2×2,SEP 只占 identity-agnostic 的静态/准静态格,语言独占动态·身份格。**
> **方法学背书**:SEP 归入长时一致性记忆的**"显式 3D/世界系状态"派**(对应 PERSIST 2603.03482 / WorldMem NeurIPS'25 的持久 3D state,可累积、抗漂移),其时不变轨道约束是 action-conditioned 世界模型漂移的缓解手段——详见 `world-model-survey.md` 第二段 (b)。⚠️ 但"显式 3D 唯一"被隐式方法(StateSpaceDiffuser NeurIPS'25、Geometry-Aware Implicit Memory 2606.02436)反驳 → SEP 措辞守"可辩护"、必做 vs 纯隐式长上下文(TTT/WorldPack)的消融。

### 4D.1 时不变/时变 × 身份 的 2×2(方案的组织原则)

|  | identity-agnostic(身份无关) | identity-bearing(身份相关) |
|---|---|---|
| **时不变(static,可累积、抗漂移)** | 路网拓扑、车道几何、可行驶区 → **SEP 预测的对象("where 的舞台")** | ∅(身份从不静态) |
| **时变(dynamic,须逐步预测)** | 流密/流速场、遮挡状态、全体车辆占据 → 准静态**场**归 SEP,瞬时占据不建 | **目标 vs 干扰的选择("which")→ 语言独占** |

三条直接推论(写进 intro/method):
1. **因子分解从此有原则性判据**,不再是"我们挑了路网+流态"这种任意选择:**判据 = 该因子在 episode horizon(~3min)内是否变化**。路结构不变→可跨帧**累积再确认**(估计越跟越准、不漂移);身份/位置时变→须逐步预测。这把审稿人"为什么是这些因子"的质疑用一个可测量标准挡掉。
2. **SEP 天生抗误差累积**:时不变部分是**累积/平均问题而非 rollout**,不受复合误差影响;并给时变预测提供"轨道约束"(目标被约束在路上)→ **界定** EAR 长时预测的误差上界。这是长视频/世界模型公认 drift 难题的一个任务特定规避。
3. **SEP 强化 H0(非威胁)**:2×2 右下格是 which 的唯一归属;左上(静态·身份无关)分不开同路 look-alike、左下(动态·身份无关,流态)只说"有车往这走"非"你的车"、右上为空 → which 只能由身份相关信号解,重现时外观又失效 → **语言是唯一出口**。SEP 越准、where 越被它解释干净,which 越只剩语言。**⚠️ 但 SEP 身份无关→单主干道里"跟着路走"几乎必对,可能放大位置捷径→掏空 H0**;须严守 experiment-design §2.0c 的非放大 guard(SEP 不喂语言、只吃车道级统计场、加 SEP 后 w/o-lang 增幅不许缩小)。

### 4D.2 逐件先例切割(novelty 押"域 × 用途 × 与语言身份绑定的耦合",非原理)

> 判决:**"只建任务相关因子的部分世界模型"作为原理毫无新颖性**(下述五线均成熟);可辩护点 = **从单目斜视航拍视频在线推路网 + 身份未定(语言绑定)+ 服务具身拦截 + 时不变/时变分工**这一具体兑现。

| 先例线 | 代表(venue/状态) | 它做什么 | 切割 |
|---|---|---|---|
| **地图条件轨迹预测** | VectorNet(**CVPR'20** ⚠️)、LaneGCN(**ECCV'20** ⚠️)、**PGP/Lane-Graph Traversals(CoRL 2021, 2106.15004 ✅)**、QCNet(**CVPR'23** ⚠️) | **给定 HD 地图 + 已知 agent 身份**预测车辆未来 | must-cut:我们**无 HD 地图**(从视频推)、**身份未知**(语言定)、且**闭环控制**非离线预测 |
| **无图轨迹预测(隐式路网)** | 异构图+动态场景约束(**Sci. Reports 2026**, s41598-026-45445-w ✅)、DynaScene-Pred | 从**他车历史轨迹**抽隐式路拓扑、无需地图 | 最近邻:同样"无图",但它从地面他车轨迹抽、我们从**航拍自视频**抽结构且服务拦截+语言消歧 |
| **在线路网/车道拓扑估计** | 航拍车道图 LaneExtraction(He&Balakrishnan, **CVPR 2022** ⚠️核)、Learning Lane Graphs from Aerial Imagery(2024 ⚠️)、TopoBDA(2412.18951 ✅)、StreamMapNet(**WACV 2024** ⚠️)、**Online Road Topology + SD map(2507.01397 ✅)** | **建图/拓扑理解**(多为地面环视或正射航拍) | 切割:它们**离线建图为目的**;我们是**在线、短时、面向拦截预测**的结构先验,输入是 pitch −30° **斜视单目** UAV 视频(≠正射) |
| **单目斜视 UAV 视频→BEV** | Mobile Traffic Camera Calibration(2605.11900 ✅) | 单目斜视 UAV 交通视频→度量 BEV(路面单应) | 支持可行性(证明斜视航拍→BEV 可做);我们**预测未来结构+接 EAR**,非仅当前标定 |
| **因子化/任务相关/对象中心世界模型** | 值等价/bisimulation(DBC, **ICLR'21**, 2006.10742 ⚠️)、Denoised MDP(**ICML'22** ⚠️)、对象中心 WM(2511.02225 ✅)、Task-Sufficient WM(2607.04409 ✅) | 在通用 RL/像素里**学**任务相关抽象 | 切割:它们**学习式**抽象+通用域;我们**手工按时间平稳性分解**(路网/流态)+**直接监督**(CARLA GT)+**语言身份绑定**,域=空中跟踪 |
| **静/动场景分解** | DynaSLAM(⚠️)、NeRF 静+动(D-NeRF/NSFF ⚠️)、背景/前景分离 | 重建静态几何、分离动态物 | 切割:它们**重建**;我们**不重建**,只出面向预测的因子先验,且耦合语言 which |

**未占据核心(主张精确到此句,to-our-knowledge)**:
> 从**单目斜视航拍视频在线推断的时不变路网结构**,作为**身份未定**的目标未来位置先验,服务于**无路网约束的 UAV 预测拦截**,并与**语言身份绑定**在时不变/时变的 2×2 下分工——未见先例。每个零件(地图条件预测、航拍建图、因子化 WM、静动分解)都已被占据但互相脱节,且**无一把"结构先验"与"语言 which"按时间平稳性显式分工**。

**⚠️ 诚实边界(写进 limitation)**:
1. **仍不是 WAM/world model**:SEP 预测**结构(部分观测/环境状态)**,比 EAR 的纯目标状态更靠近观测预测,但仍**非生成式、非完整观测 rollout** → 措辞升级为 **"factored/structured world-model prior"**,**仍不自称 WAM**(接 §4C 落点)。
2. **"手工指定因子"=特征工程质疑**:守法=强调 SEP 预测**未来**结构并据此 rollout 目标未来(预测性)、非当前特征;且时间平稳性判据使因子选择**可证不可任意**。
3. **时不变是"世界系"不是"图像系"**:UAV 移动会揭示新路 → 静态先验须在**度量/世界系累积**(需自运动补偿;CARLA 有位姿,可行但非免费)。
4. **流态是尴尬中间态**:只保留**车道级统计场**(流密/流速/信号相位),**不建瞬时逐车占据**——否则"预测占据≈预测车在哪"会**重新泄漏 H0**(见 experiment-design §2.0c g3)。

### 4D.3 相关文献锚点(数值/号写作前复核)
- 地图条件预测:VectorNet **CVPR 2020**(⚠️)、LaneGCN **ECCV 2020**(⚠️)、**PGP/Lane-Graph Traversals CoRL 2021**(2106.15004 ✅)、QCNet **CVPR 2023**(⚠️)
- 无图/隐式路网:异构图无图预测 **Sci. Reports 2026**(s41598-026-45445-w ✅)
- 在线拓扑/航拍建图:**Online Road Topology + SD map**(2507.01397 ✅)、TopoBDA(2412.18951 ✅)、LaneExtraction **CVPR 2022**(⚠️核 He&Balakrishnan)、StreamMapNet **WACV 2024**(⚠️)
- 斜视 UAV→BEV:Mobile Traffic Camera Calibration(2605.11900 ✅)
- 因子化/任务相关 WM:对象中心 WM(2511.02225 ✅)、Task-Sufficient WM(2607.04409 ✅)、DBC **ICLR 2021**(2006.10742 ⚠️)、Denoised MDP **ICML 2022**(⚠️)

---

## 4E. 相关工作 H｜语言双功能:消歧(which)+ 意图种子(which-way)(2026-08-11,deep-research 背书)

> 背景:新方向把语言从"只做身份消歧(which)"扩展为"**同时作为推理种子塑造结构化预测(which-way:意图/目的地/行为→偏置 EAR 轨迹与 SEP 路网先验)**"。方法学全景、逐件切割、风险详见 `language-seeded-prediction-survey.md`。
> **判决**:"语言超越消歧、条件化结构化预测"是**成熟且 peer-reviewed 的范式**(语言条件轨迹预测)→ 原理无新颖性;可辩护点押 **"空中闭环 + 同一语言双功能(which + which-way)"耦合**(medium 置信空白)+ 统一 framing。

### 4E.1 成熟先例与逐件切割(novelty 押耦合,非原理)

| 先例线 | 代表(venue/状态) | 语言起的作用 | 切割 |
|---|---|---|---|
| **语言条件轨迹预测** | **Trajectory-LLM**(ICLR 2025)、**iMotion-LLM**(WACV 2026)、**LC-LLM**(CommTR 2025)、**Traj-LLM**(IEEE TIV)、**LMTraj**(CVPR 2024+TPAMI) | 指令/意图条件化塑造轨迹;L2T 数据生成;意图+轨迹双任务 | must-cut:**全为地面/离线/非闭环、语言单一功能**(要么意图、要么数据生成、要么生成基底);无 look-alike 消歧、无空中闭环 |
| **文本→结构化布局** | **LayoutGPT**(NeurIPS 2023) | 自由文本→CSS 结构化 2D/3D 布局 | 支持"语言可条件化 SEP 结构化几何"的机制类比;非跟踪/闭环 |
| **语言条件世界模型** | **★ Semantic World Models**(2510.19818 预印本, **PaliGemma-3B**)、GWM(2604.11751 预印本 ⚠️复核)、LGWM(**ACL 2024 工作坊**) | 指令→goal-aware 语义未来;**预测语义状态而非像素** | 🟢 **正面弹药**:SWM 用你候选骨干论证"预测状态而非像素"、"像素视觉逼真却缺决策语义"——现成 peer 背书 EAR/SEP 哲学 |
| **生成即推理** | **Diffusion-of-Thought**(NeurIPS 2024) | CoT 嵌入扩散去噪 | EAR flow-matching 的推理注入原型(goal-based) |

**⚠️ 两个易误引的反例**:**LMTraj** 的"语言"=LLM 主干世界知识(非外部意图指令)→ 证"LLM 推理增值"、非 which-way 先例;**DSC-LLM**(PMC)语言仅作**预测后解释层**→"语言只作后处理"的对立设计。

### 4E.2 未占据核心 + 统一 framing(to-our-knowledge)
> **无任何工作**在**空中闭环**里让**同一条语言同时**做 look-alike 身份消歧(which)+ 预测意图种子(which-way)。统一 framing = **"语言=解决其他通道无法确定的信号"**(重现时刻外观分不清→定 which;预测分叉处几何+运动定不了→定 which-way,同一原理两个实例):与 survey《Trajectory Prediction Meets LLMs》(2506.03408 预印本)"articulate goals + 推演 alternative futures"概念一致、**无同名先例 → 原创综合,且加强 H0**(语言承载几何无法承载的信息)。

**⚠️ 诚实边界**:① "reasoning seed / which / which-way" 均本项目 framing、非原文措辞,署"我们提出";② 空白为否定性论断,强度上限 medium,投稿前定向反证检索(FlightLLM/Holodeck/DiffuScene/UniPi/Hierarchical Diffuser 及其他空中语言条件工作);③ **命名冲突**:两篇 Traj-LLM/Trajectory-LLM 别混、FlightLLM 与 FPGA 同名区分。

### 4E.3 核心风险(接 experiment-design H8 / §2.0d)
1. **打破 SEP 2×2**:语言跨进"where 列"→ "SEP 定 where、语言定 which"分工被破坏,H0 归因复杂化 → 三臂消融拆贡献。
2. **意图冗余陷阱**(与位置捷径同构):语言意图若能被 SEP 路网+运动推出则冗余;仅在携带"路+运动推不出的信息"(视野外目的地/未显现行为/非最短路径)时 load-bearing → 须"分叉歧义"场景验证。

---

## 5. 相关工作 D｜空间推理与空间智能(方法动机)

### 5.1 空间智能作为前沿(motivation,立场文)
- **Fei-Fei Li**《From Words to Worlds: Spatial Intelligence is AI's Next Frontier》(a16z / TIME, 2025.11):空间智能="认知脚手架";LLM="eloquent but ungrounded";需**世界模型(world models)**。**引作 motivation,非实证。**
- **World Labs**:"spatial intelligence company"。
- **Survey(4 篇,均 arXiv 预印本,引作"共识成形")**:LLM-Powered Spatial Intelligence Across Scales(清华/李勇,2504.09848);Spatial Reasoning in MLLMs(2511.15722);Multimodal Spatial Reasoning in the Large Model Era(2510.25760)——**均把 embodied AI(VLN+VLA)列为一等分支**。

### 5.2 VLM 空间推理的系统性局限(简引即可,别过度依赖)
| benchmark | venue | 结果 |
|---|---|---|
| What'sUp | EMNLP 2023 ✅ | 18 VLM 全差;BLIP 56% vs 人 99% |
| VSR | TACL 2023 ✅ | 人 95.4% vs 最好 ~70%;朝向/facing 全随机 |
| BLINK | ECCV 2024 ✅ | GPT-4V 51.3%/Gemini 45.7% vs 人 95.7% |
根因:缺 3D/深度先验,走 2D 共现捷径,参照系混淆。

### 5.3 空间语言的表征基础(参照系)
- **参照系分类**(TiCS 2024,peer-reviewed):egocentric/相对、object-centric、allocentric/绝对;**同场景不同参照系编码互不等价** → 俯视 vs 第一视角歧义的根因。
- → **本工作含义**:语言"front-left"是 egocentric-相对,无人机 pitch −50° 俯视下**本身参照系歧义** → 支撑"用相机位姿注入对齐参照系"与"中性语言/少用方位从句"(与方案A 实测减半自洽)。

### 5.4 三条方法线**镜像本工作三模块**(核心动机论据)
| 文献方法线 | 代表(状态) | 机制 | 对应模块 |
|---|---|---|---|
| **显式参照系实例化** | **APC**(KAIST, **ICCV 2025** ✅,开源) / Allocentric Perceiver(2602.05789 ⚠️预印本) | 心理意象→旋转到参照者坐标系;度量感知→动态参照系。APC left/right 89.67% vs 基线 46–55%;Allocentric +10.8% 免训 | **相机位姿注入** |
| **单目视频注入 3D 结构** | **Spatial-MLLM**(THU, **NeurIPS 2025 Spotlight** ✅,开源) | 双编码器:2D 语义 + 3D 空间编码器(VGGT 几何骨干),connector 融合;**无需深度/3D 输入** | **IAR 隐式先验** |
| **输出坐标而非文字 CoT** | PaliGemma `<loc>` / **RoboPoint**(**CoRL 2024** ✅) / ACoT-VLA 动作空间推理 | 直接回归点/坐标,绕开 VLM 最弱的语言→坐标。RoboPoint 比 GPT-4o +21.8% | **EAR 显式 3D 拦截点** |
| (增强)深度+region+参照系监督 | **SpatialRGPT**(NeurIPS'24 ✅)/ **RoboSpatial**(**CVPR'25 oral** ✅) | 深度插件+区域;ego/object/world 三系+自由空间点。RoboSpatial 微调 RoboPoint 38.9%→70.6% | 数据/监督侧可借鉴 |

> **写作杠杆**:EAR/IAR/相机位姿注入**各有 peer-reviewed 空间推理先例背书**,本工作是三者的**空中·闭环合流**。

### 5.5 空中空间智能 = 被点名的 open problem(最强定位源)
- **AirGroundBench**(BUPT 系, 2606.28049, 2026 ⚠️预印本):UAV–UGV 多视角空间智能诊断;11 环境/1021 空地配对/~62k 双视 VQA/115 闭环 VLN。13 SOTA MLLM:**感知尚可,但跨视角对齐/几何一致/视角变换系统性失败,且传导进闭环导航**;最差项相机位姿;顶级 ~50–60% vs 人 ~82%。差别:UAV–UGV 协作,非单机主动跟踪(类比非等同)。
- 佐证:All-Angles Bench(2504.15280)、Seeing Across Views(2510.19400)。

### 5.6 本工作与空间智能的相关性:四轴归属(deep-research 2026-08-01)

> **判决:强相关,但沿 4 条已被承认的能力轴,不能笼统说"主动跟踪=空间智能"。** 三篇 survey(2511.15722/2510.25760/2504.09848)**均未**把主动跟踪/pursuit 列为 taxonomy 类别 → 不主张它是既定命名子任务。但四个组成能力各自是被承认、可 benchmark 的空间智能维度:

| 我们的能力 | 对应空间智能维度 | 关键锚点(venue/状态) | 我们的延伸 |
|---|---|---|---|
| 主动飞行控制视角 | **主动感知/感知-动作闭环(核心)** | **ESI-Bench**(Stanford, **Fei-Fei Li 等**, 2605.18746 ⚠️预印本,开源);E3VS-Bench(2604.17969) | 它们探索**静态**空间→我们追**动**目标 |
| **EAR** 预测目标未来3D轨迹/拦截 | 动态/时序空间推理 + 运动预测 | **VLM4D(ICCV 2025)** ✅ · GTR-Bench(2510.07791,轨迹预测) · ST-VLM(2503.19355) · **VISTA(CVPR'26 workshop)** ✅ · FSU-QA(2511.18735,世界模型增强 VLM,镜像我们定位) | survey 把"physics-based prediction"列为**未来 gap**→EAR 填公认空缺,非既定类别 |
| 长丢失重捕获/目标状态记忆 | 持久空间记忆/遮挡期状态保持 | survey 2511.15722 列"缺持久空间记忆"为**头号未来方向** · **TAO-Amodal(ICLR 2024)** ✅(遮挡/出画 amodal) · "What Spatial Memory Must Store"(2606.10299,⚠️弱源) | 杀手场景直接落此开放前沿 |
| 语言指定+消歧 | 动目标语言条件空间 grounding | VISTA 的 STVG(时空视频 grounding) | 空中+look-alike 消歧 |
| 俯视/BEV 空间智能 | ⚠️**未充分建立**(除 AirGroundBench) | — | **定位机会,非拥挤类别**(medium 置信) |

**★ 诚实定位一句话(intro 用)**:
> ACoT-UAV-Track 是 world-model-augmented 具身 VLA,在**空间智能 survey 明确标为开放前沿**的三条轴——**主动感知、动态空间推理、持久空间记忆**——给出统一具身实现,并从"静态空间/被动 QA"扩展到"空中·动目标·闭环"这一 under-explored 俯视 regime。**不主张"跟踪=空间智能",而是把三个 open frontier 在一个真实闭环任务里同时兑现**(ESI-Bench/VLM4D/TAO-Amodal 皆被动/静态,我们主动+动目标,正踩其 gap)。

**四条必避过度声称(写进 limitation)**:①别把主动跟踪/pursuit 当既定命名子任务;②别称 EAR 式预测是已确立类别(survey 视为 mental-simulation+未来 gap);③别用 "object permanence",survey 用的是 **"persistent spatial memory"**;④别把静态空间探索等同动目标追击。

### 5.7 领域交叉定位:该去看哪个领域的现有成果(2026-08-01)

> 快速定位:你的**任务**属具身导航,**方法**属操控 VLA,**动机**挂空间智能。以下按交叉度排,决定"去哪对标、别在哪浪费时间"。

| 领域 | 交叉度 | 为什么 | 去哪看 |
|---|---|---|---|
| **★ 具身导航/VLN + 主动跟踪** | 🟢**高(任务+方法,主场)** | 主动跟踪=目标会动的导航;TrackVLA/EVT-Bench 在跟踪∩导航边界;survey 把 VLN+VLA 列一等分支 | §2 谱系 + EVT-Bench/CoMaTrack/AD-VAT/VLN 综述 |
| **操控 Manipulation VLA** | 🟠**方法高/任务低** | 整条方法谱系来自操控(π0/ACoT-VLA/OpenVLA/WorldVLA) | 只看**架构**(flow-matching/动作CoT/world-model-augmented,§4C),不看任务 |
| **空间智能/空间推理** | 🟢**高(动机/框架层)** | 见 §5.1–5.6 四轴 | §5 全节 |
| **SLAM/状态估计** | 🟡低-中 | 不建图不自定位;但**遮挡期目标信念传播**(长丢失缺口)与滤波同源 | 只为该缺口看:EKF/粒子滤波、SiamMDTP(Info Sci'25) |
| **NeRF/3DGS/3D重建** | 🔴**低** | 它们重建静态几何;我们不重建、用冻结 VLM 隐式感知+预测动目标。唯一接口=相机位姿 | **暂不对标**;仅当上"渲染式 world-model/未来视角预测头"(WorldVLA 启发)才看 3DGS 仿真器 |

### 5.8 丢失后重捕获的数据集空缺(支撑"必须自建长丢失数据",中等置信)

> 问题:具身智能里有没有专门评测**目标丢失后重新追踪(re-acquisition)**的数据集?

- **具身侧:无专用"长丢失→重捕获"数据集/指标**(中等置信,负向)。EVT-Bench(遮挡在闭环内、TrackVLA++ TIM 穿遮挡)、Gym-UnrealCV/AD-VAT(有遮挡物)**把恢复当隐含 challenge,非标注的重捕获事件 + 专用指标**。
- **被动侧:"丢失+重检测"成熟但非具身**——VOT-LT、OxUvA(标注 target absence)、LaSOT(full-occlusion/out-of-view)、TNL2K(adversarial+modality-switch)。
- **→ 对本工作**:空缺**正是** `data-fix-spec.md` §3 已识别的"必须自建长丢失(5–15s)+重现+拦截式专家演示"数据的依据;且**反向强化贡献**——"重捕获成功率 vs 丢失时长"指标 + 长丢失语言重捕获评测本身是新 benchmark 贡献。⚠️ 负向为 absence-of-evidence(未穷尽 active object search/embodied re-ID 邻域)→ 投稿前可再定向核。

---

## 6. 相关工作 E｜"语言是否 load-bearing"(支撑 H0,见 experiment-design §2.0)

| 工作 | venue/年 | 状态 | 对 H0 的作用 |
|---|---|---|---|
| **Shortcut Learning in Generalist Robot Policies** | **CoRL 2025** | ✅ | VLA 走捷径两驱动:类内多样性低 + 跨子集割裂 |
| **VLABench** | **ICCV 2025** | ✅ | Track 4 仅换语言 → 微调 VLA 掉 31–45%(单变量语言消融先例) |
| CAST | 2508.13446 | ⚠️预印本 | 有干扰车:标准 VLA 31.7%→反事实增强 58.3%(1.84×);无干扰"指令条件化不必要" → **效应量参照 ~26 点** |
| When Vision Overrides Language / LIBERO-CF | 2602.17659 | ⚠️预印本 | 首个反事实基准 + CAG no-language 分支(no-language 控制先例) |
| Restoring Linguistic Grounding (LGS) | 2603.06001 | ⚠️预印本 | **LGS = 正常 SR − 矛盾 SR**;π0.5 在矛盾指令下 96.2% → LGS 仅 1.2 |
| RoboSemanticBench | 2606.02277 | ⚠️预印本 | 控制抓取后 VLA 选目标≈随机(25%/10%) |

**关键警示**(写进 H0 论证):**观察到"去语言掉点"是必要非充分**——必须先排除**位置捷径**与**时空连续性捷径**(TrackVLA++ 证明空间推理能扛消歧,不必靠语言),否则掉点可能来自别的信号消失。方法学:反事实指令 + LGS + no-language VA 基线 + 连续性断裂再捕获(E5)。

---

## 7. 领域空白(逐轴切割,to-our-knowledge)

| 已占据维度 | 代表 | 缺什么(相对本工作) |
|---|---|---|
| 地面/室内 语言+消歧+闭环 | TrackVLA/++、CoMaTrack、EVT-Bench | 非空中 |
| 空中 语言+4DoF 闭环 | UAV-Track VLA | **无相似车消歧**,行人为主 |
| 被动 车辆语言 referring | Refer-KITTI/RMOT | 无闭环控制 |
| 空中/空地 多视角空间推理 | AirGroundBench | 协作诊断,非单机跟踪;无消歧 |
| 长丢失后重检测/重捕获 | GlobalTrack/SiamR-CNN/LTMU/DAM4SAM/TrackVLA++ TIM | **重识别全靠外观/运动,无语言** |
| UAV 预测拦截("抄近路") | Fast-Tracker/PN·ATPNG | **纯几何,身份已知,无消歧** |
| 中途语言在环 | Feng et al. WACV'20 | 逐帧定位,**非丢失重现时的唯一消歧** |
- **空白①(四轴)= 把车辆语言消歧搬进空中闭环**。归纳性缺失论证(medium 置信)→ 论文写 "to our knowledge"。
- **★ 空白②(升级/杀手场景,见 §4B)= 长丢失预测拦截 + 语言重捕获消歧三者叠加**:每个零件已占据但互相脱节,重捕获时刻均无语言 → high 置信真空白。
- ⚠️ **无任何源在空中 VLA 里评测"视觉相似车消歧"** → 该贡献确 underexplored(定位机会)。
- **★ 空白③(路线 B,见 §4D)= 从单目斜视航拍视频在线推时不变路网结构 + 身份未定的目标位置先验 + 服务 UAV 预测拦截 + 与语言按时不变/时变 2×2 分工**:五条零件线(地图条件预测/无图预测/航拍建图/因子化 WM/静动分解)皆被占据但脱节,**无一显式做"结构先验 × 语言 which"分工** → to-our-knowledge 空白(medium 置信,因未穷尽"航拍建图 × 轨迹预测 × 具身跟踪"三元交叉)。

---

## 8. 推荐基线清单(映射 experiment-design §5.1 M-槽)

- **可直接复现/引用(已发表)**:ORTrack(CVPR'25)、TCMLTrack/UAVNLT(UAV 语言引导)、JointNLT(CVPR'23)/CiteTracker(ICCV'23)、TNL2K/WebUAV-3M(基准) → 兜底 H0/H5 的**感知层与语言消歧对照**(M9/M9b)。
- **需适配/仅相关工作讨论(预印本)**:UAV-Track VLA(头号竞品,查代码可复现性)、TrackVLA++/CoMaTrack(借 **EVT-Bench DT/AT 协议**)、CognitiveDrone-R1(范式讨论) → M8/M8b。
- **π0.5 微调** 通用基线 → M7。

---

## 9. 待核实清单(写作前收口)

| 项 | 状态 |
|---|---|
| TrackVLA 发表状态 | ✅ 已确认 CoRL 2025(解除) |
| UAV-Track VLA 是否=本工作 | ✅ 否,独立竞品(勿自引) |
| CosFly-VLA | ✅ 确认虚构,勿引 |
| **DeTrack(2605.17451)/AerialMind(2511.21053) 编号+数值** | ✅ **已核(2026-08-05):两号均正确**。DeTrack=预印本,主动闭环+动态遮挡物(非相似车)+无语言,撞"具身/world-model"两词须切,⚠️勿混 NeurIPS'24 同名 SOT 2501.02467;**AerialMind 升级=AAAI-26 已发表**(RMOT,被动),与 UAVNLT 是两篇 |
| "把具身追踪列为 open problem 的 survey" | ⚠️ 本轮未surface;空间智能 survey 可替代该功能 |
| **路线 B/SEP 先例号+venue**(§4D:VectorNet/LaneGCN/QCNet/PGP/LaneExtraction/StreamMapNet/DBC/Denoised MDP) | ⚠️ **待核**:仅 2106.15004 / 2507.01397 / 2412.18951 / 2511.02225 / 2607.04409 / 2605.11900 / s41598-026-45445-w 已由检索确认;其余经典号+venue 写作前复核(尤其 LaneExtraction 作者/年、DBC=2006.10742) |
| 预印本数值(AirGroundBench/CAST/CoMaTrack 等) | ⚠️ 写作时复核(部分含晚于知识截止的模型名单) |
| **pre-VLA NL-tracking 在环先例** | ✅ **已锁死(§4B.1)**:语言仅初始化/持续定位/全局重检测,无"丢失重现时 look-alike 语言消歧";残余未核 GTI/CTRNL/CiteTracker 等 → 论文写 to-our-knowledge |
| **必须显式 cut 的先例**(§4B) | ⚠️ Feng WACV'20(中途语言)、DAM4SAM CVPR'25(离开-重入)、Fast-Tracker ICRA'21(预测拦截)、TNL2K/Li'17(全局重检测)、QueryNLT CVPR'24(连续消歧) |

---

## 附:主引用表(arXiv/venue 速查)

**已发表(peer-reviewed,可作硬锚点)**
- Luo et al., End-to-end Active Object Tracking, **ICML 2018**
- Zhong et al., AD-VAT, **ICLR 2019** / AD-VAT+, **IEEE TPAMI 2021**
- TrackVLA, **CoRL 2025** (arXiv 2505.23189) — 开源
- RMOT / Refer-KITTI, **CVPR 2023** (2303.03366) — 开源
- ORTrack, **CVPR 2025** (2504.09228) · JointNLT, **CVPR 2023** (2303.12027) · CiteTracker, **ICCV 2023** (2308.11322)
- WebUAV-3M, **TPAMI 2023** (2201.07425) · TNL2K, **CVPR 2021** (2103.16746) · UAVNLT, **Electronics 2024** · TCMLTrack, **Sci. Reports 2025**
- Shortcut Learning in Generalist Robot Policies, **CoRL 2025** (2508.06426) · VLABench, **ICCV 2025**
- What'sUp, **EMNLP 2023** (2310.19785) · VSR, **TACL 2023** · BLINK, **ECCV 2024** (2404.12390)
- RoboPoint, **CoRL 2024** (2406.10721) · SpatialRGPT, **NeurIPS 2024** (2406.01584) · RoboSpatial, **CVPR 2025 oral** (2411.16537)
- APC (Perspective-Aware Reasoning), **ICCV 2025** (2504.17207) · Spatial-MLLM, **NeurIPS 2025 Spotlight** (2505.23747)
- PaliGemma tech report, DeepMind 2024 (2407.07726) · Spatial reasoning frames, **TiCS 2024**
- **§5.6 空间智能相关性(四轴)**:VLM4D **ICCV 2025** (2508.02095) · VISTA **CVPR 2026 workshop** (2605.01391) · TAO-Amodal **ICLR 2024** (2312.12433) · ESI-Bench 2605.18746(Fei-Fei Li 等,开源) · E3VS-Bench 2604.17969 · GTR-Bench 2510.07791 · ST-VLM 2503.19355 · FSU-QA 2511.18735 · What Spatial Memory Must Store 2606.10299(⚠️弱源) — 除 VLM4D/VISTA/TAO-Amodal 外均 ⚠️预印本
- **§4B 重捕获/拦截线**:GlobalTrack **AAAI 2020** (1912.08531) · Siam R-CNN **CVPR 2020** (1911.12836) · LTMU **CVPR 2020** · DaSiamRPN **ECCV 2018** (1808.06048) · DAM4SAM/SAM2.1++ **CVPR 2025 / IJCV 2026** (2411.17576,开源) · SiamMDTP **Info Sciences 2025** · Fast-Tracker **ICRA 2021** (2011.03968,开源) · PN·ATPNG 制导 **Intelligent Service Robotics (Springer) 2021–22**
- **§4B.1 pre-VLA NL-tracking 线**:Li et al. Tracking by NL Specification **CVPR 2017** · TNL2K/AdaSwitcher **CVPR 2021** (2103.16746) · SNLT **CVPR 2021** (1912.02048) · JointNLT **CVPR 2023** (2303.12027) · QueryNLT **CVPR 2024** (2403.19975) · **DecoupleTNL ICCV 2023**(★正面弹药) · Feng et al. NL-tracking **WACV 2020** (1907.11751) · DTVLT 2410.02492 ⚠️预印本
- **§4C 范式(VLA↔WAM)线**:UWM **RSS 2025** (2504.02792,开源) · DriveWorld-VLA **ICML 2026 poster** (2602.06521) · DUST 2510.27607(★"world-model-augmented VLA"术语) · WorldVLA 2506.21539 · Survey World Model for Robot Learning 2605.00080 · VLA survey 2509.19012 · WAM tutorial/survey 2607.00836 / 2605.12090 · NVIDIA WAM glossary(厂商术语) — §4C 除 UWM/DriveWorld-VLA 外均 ⚠️预印本/厂商

**arXiv 预印本(标注状态,数值复核)**
- TrackVLA++ 2510.07134 · CoMaTrack 2603.22846 · UAV-Track VLA 2604.02241 · CognitiveDrone-R1 2503.01378
- CAST 2508.13446 · LIBERO-CF 2602.17659 · LGS 2603.06001 · RoboSemanticBench 2606.02277
- Allocentric Perceiver 2602.05789 · OmniView-Space 2607.00881 · AirGroundBench 2606.28049
- Surveys: 2504.09848 · 2511.15722 · 2510.25760

- **✅已核(2026-08-05)**:**DeTrack** *A Benchmark and Altitude-Aware Dual World Model for Drone-embodied Tracking* 2605.17451(预印本,主动闭环·无语言·world-model;⚠️勿混 NeurIPS'24 同名被动 SOT 2501.02467)· **AerialMind** *Towards Referring MOT in UAV Scenarios* **AAAI-26 已发表**(2511.21053 → Proc.AAAI 40(4):2805-2813, DOI 10.1609/aaai.v40i4.37270;RMOT 被动)

**⚠️ 待核**:CosFly-Track 2605.17776 / CosFly 2605.19120(非 VLA)
