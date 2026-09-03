# 世界模型长时序预测方法学综述:因果自回归 + 记忆 + action-conditioned

> deep-research 深研管线产出(2026-08-07,6 检索角度 / 29 一手源 / 128 论断抽取 / top-25 三票对抗验证,25/25 confirmed、0 refuted)。
> 用途:为 `acot-uav-design.md`(EAR/IAR)、`related-work-and-positioning.md` §4C/§4D(SEP 定位)、`experiment-design.md` §7.4(SEP × 长丢失消融)提供方法学背书与先例锚点。
> **引文纪律**:每条标 venue + 发表状态;⚠️自报 = 指标为作者自报、未第三方审计;⚠️复核 = 后知识截止(2026)来源,写作前复核 arXiv 号/venue。

---

## 0. 一句话结论

2024–2026,长时序世界模型跨过"实时 + 因果流式 + 数分钟一致"门槛;核心引擎从双向扩散转为**因果自回归扩散 + 记忆机制**。三条抗 drift 主线(自 rollout / attention-sink / error-recycling)与三派记忆(显式记忆库 / 隐式长上下文 / 显式 3D 状态)已成形,但**误差累积仍是缓解非根治、长时一致性无统一评测**。对本项目最有价值的是:**借因果 AR 的训练范式(非像素生成器)抗 EAR 漂移 + 把 SEP 归入"显式 3D/世界系状态"记忆派**。

---

## 第一段 · 方法学综述

### 线 1｜因果自回归视频扩散(Causal AR Video Diffusion)

**根因(peer-reviewed 共识)**:exposure bias —— 训练条件于 clean GT context、推理条件于自生成帧,偏差逐帧复合成 drift(BAgger 摘要逐字;Self Forcing 以 "Bridging the Train-Test Gap" 独立表述;溯至 Bengio 2015 scheduled sampling)。**缓解机制收敛为三类**:(1) 训练时自 rollout + 视频级分布匹配;(2) 全局锚点 + 松弛因果;(3) 自生成误差回收/纠偏轨迹。**三类均在训练侧、不改推理接口**(对下游迁移关键)。

| 代表工作 | venue/状态 | 核心技术 | 关键指标 | 开源 |
|---|---|---|---|---|
| **Self Forcing** | **NeurIPS 2025 Spotlight**(peer-reviewed) | 训练时 KV-cache 自 rollout(条件于自生成帧)+ 视频级 DMD/SiD/GAN 分布匹配 | 单 H100 **17 FPS**、亚秒延迟(frame 0.45s / chunk 0.69s);比 Wan2.1/SkyReels-V2 **≈150×**;RTX 4090 实时(~10FPS 需 FP8/TAEHV)⚠️自报 | ✅ guandeh17/Self-Forcing |
| **Rolling Forcing** | arXiv 2509.25161(预印本,OpenReview 在审) | attention-sink 保初始帧 KV 作全局锚点 + 联合多帧去噪松弛严格因果 | 单卡**多分钟**实时(~16 FPS)、大幅降误差累积 ⚠️自报 | 项目页 |
| **Stable Video Infinity (SVI)** | arXiv 2510.09212(**ICLR 2026 Oral** 据 repo ⚠️复核) | Error-Recycling Fine-Tuning:误差注入 clean 输入模拟累积轨迹 + one-step 双向积分 + 跨时间步 replay-memory error banking | 秒级→"无限时长"、无额外推理成本;兼容 audio/skeleton/text ⚠️自报 | ✅ |
| **BAgger** | arXiv 2512.12080(Stanford,预印本) | Backwards Aggregation:从自身 rollout 构造 corrective trajectories(DAgger 类比);**标准 score/flow-matching 目标**,避 teacher + BPTT | 非蒸馏路线抗 drift ⚠️自报,优势未独立验证 | 项目页 |

> ⚠️ 本轮未独立验证:CausVid、Diffusion Forcing、History-Guided、TempoMaster、One-Forcing(CausVid/Diffusion Forcing 是 Self Forcing 直接前身)——写作时单独核。

### 线 2｜长时一致性记忆机制(三派)

| 代表工作 | venue/状态 | 派别 | 核心技术 | 关键指标 | 开源 |
|---|---|---|---|---|---|
| **WorldMem** | **NeurIPS 2025**(peer-reviewed) | 显式记忆库 | memory bank(frame+pose+timestamp)+ state-aware memory attention | 大视角/时间间隔后精确重建先前场景;含动态演化 | ✅ xizaoqu/WorldMem |
| **TTT One-Minute** | arXiv 2504.05298(预印本) | 隐式长上下文 | 隐状态为 NN 的 TTT 层(表达力 > 线性 RNN);CogVideoX-5B 基座 | 连贯 1 分钟多场景;+34 Elo 人评(vs Mamba2/Gated DeltaNet/滑窗)⚠️自报 | 项目页 |
| **WorldPack** | arXiv 2512.02473(预印本) | 隐式长上下文 | trajectory packing(层次帧压缩)+ geometric selection(pose/FoV overlap) | 有效上下文 **4→22 帧**;+16% 推理 ⚠️自报 | — |
| **PERSIST**(Beyond Pixel Histories) | arXiv 2603.03482(ICML 2026 据 repo ⚠️复核) | 显式 3D 状态 | voxel 潜扩散持久 3D state(env+cam+renderer);主动从 3D state 生成 guidance frames | 数千步 rollout;**仍承认残余 autoregressive drift** ⚠️自报 | ✅ francelico/PERSIST |

**争议(写进 limitation)**:PERSIST "显式 3D state 是长时一致性唯一路线" 被并发**隐式**方法反驳 —— StateSpaceDiffuser(NeurIPS 2025)、Geometry-Aware Implicit Memory(2606.02436)显示隐式也能千帧级一致 → **"显式 3D 唯一"不成立**。

### 线 3｜action-conditioned / 可交互世界模型

| 代表工作 | venue/状态 | 核心技术 | 关键指标 | 开源 |
|---|---|---|---|---|
| **Matrix-Game 2.0** | arXiv 2508.13009(Skywork,预印本) | 键鼠条件 AR 扩散 image-to-world;action injection module 注入 Multimodal DiT;few-step 因果蒸馏 | 单 H100 **25 FPS**、分钟级流式 ⚠️自报 | ✅ |
| **UWM** | arXiv 2504.02792(**RSS 2025** 已接收) | 多模态 diffusion transformer,各模态独立 diffusion timestep 统一 policy/forward/inverse/video | 最强 peer-reviewed 中间地带 | ✅ |
| **WorldVLA / DUST / DriveWorld-VLA** | 2506.21539 / 2510.27607 / 2602.06521(预印本 ⚠️复核) | world-model-augmented VLA(DUST 为术语出处);联合预测观测 + 动作 | 给方向不给数(未进验证集) | 部分 |

> ⚠️ 本轮未独立验证数值:Genie 3、Matrix-Game 3.0、Oasis/GameNGen、Cosmos Predict 2.5/Transfer/Reason(关键数见对话上一轮检索,写作时以其为准并复核)。

### 未解难题与研究空白(三线共性)

1. **长时一致性无统一评测**:VBench 度量短时质量;回访一致 / 3D 一致无跨论文可比协议(WorldMem/PERSIST/WorldPack 各用自定义指标)——领域最大方法学空洞。
2. **误差累积缓解非根治**:attention-sink / error-recycling / corrective-trajectory 全 "substantially reduce",PERSIST 显式 3D 也承认残余 drift;训练目标层面根治 exposure bias 的方法尚不存在。
3. **显式 3D vs 隐式长上下文边界未定**:隐式方法已达千帧级 → 削弱"显式唯一";真实增益边界需具体场景消融。
4. **可控性–漂移权衡无量化**:action 注入越强、可控性越高但可能放大 drift,无直接实验刻画。
5. **自报数值普遍**:几乎所有 FPS/延迟/时长/一致性指标为作者自报、无第三方审计。

---

## 第二段 · 映射到本项目(空中语言指定车辆跟踪 + world-model-augmented VLA)

> 定位:基于已验证机制的**合成推断**(confidence: medium),非单一来源直接断言;EAR/SEP/长丢失三接口来自任务描述、无外部来源。

### (a) EAR(状态-动作同构 3D 拦截点头)
- **可迁移(强)**:EAR 是自回归预测头,与 exposure-bias 结构同构 → 三类抗 drift 机制(**KV-cache 自 rollout / error-recycling / BAgger 纠偏**)可搬进 EAR 训练,**均训练侧、不改推理**。其中"自 rollout"是现有 **staleness augmentation(design 原则5)** 的强化版;BAgger 用标准 flow-matching 目标,契合 EAR/DiT 已是 flow-matching、不引入 teacher/BPTT。
- **冲突/不适用**:**生成式像素 rollout 非必要** —— EAR 只需预测目标未来 3D 状态,不 render 未来帧 → 省掉整条视频生成栈,只借训练范式。positioning 写清"借 causal-AR 训练范式、非像素生成器"(呼应 §4C 不自称 WAM)。
- **实时性背书**:线 1 证单卡亚秒延迟 17–25 FPS 因果流式已成熟 → S1 20Hz 闭环算力站得住(EAR 比像素生成轻数量级)。

### (b) SEP(时不变路网结构先验)
- **归派**:属线 2 **"显式 3D/世界系状态"派**(对应 PERSIST/WorldMem 持久 3D state);路网可累积、抗漂移。你的"时不变(路网,累积抗漂移) vs 时变(目标位置,语言定身份)"2×2 复现 PERSIST 核心论点(显式持久状态 > pixel-history)。
- **增益机制**:时不变轨道约束 = action-conditioned 漂移的缓解手段(目标约束在路上→长时预测误差被路网界定上界);与 Rolling Forcing attention-sink"持久锚点稳长时预测"同类思路。
- **冲突/风险**:"显式 3D 唯一/最优"被隐式方法反驳 → SEP 定位为**"可辩护"非"已证唯一正确"**(呼应 §4D 措辞纪律)。**必做消融**:SEP(显式) vs 纯隐式长上下文(TTT/WorldPack 式扩容)在斜视航拍下的真实增益差(WorldPack 4→22 帧 geometric selection = 隐式路线强基线)。**输入底座**:Mobile Traffic Camera Calibration(2605.11900 ⚠️复核)证斜视航拍→地面度量单应可做。

### (c) 长丢失(>6s)预测-拦截重捕获
- **可迁移(对症,两派叠加)**:显式记忆库(WorldMem 式 state-aware 检索)保持**目标状态记忆** → 补 design §6.1 承认的"长丢失无目标状态记忆"缺口;隐式长上下文(TTT/WorldPack)把 EAR/IAR 有效上下文扩到覆盖丢失窗口。记忆负责 where(带到重现区域)、**语言负责 which(look-alike 消歧)** —— 与 2×2 严丝合缝。
- **可辩护空白(押这)**:①"重捕获成功率 vs 丢失时长"曲线本身是新度量,填补 revisit-consistency 无标准协议的空洞;②"时不变结构记忆 + 时变身份语言绑定"耦合于长丢失重捕获 = to-our-knowledge 空白(WorldMem/PERSIST 均无"身份未定+语言消歧");③ SEP 轨道约束压低 EAR 漂移多少 = 你能做出直接实验证据(通用视频世界模型无路网 GT、做不了)。
- **风险登记(建议并入 experiment-design R 系列)**:
  - **R-drift**:抗 drift 全是缓解非根治 → EAR 长丢失预测存在**不可消除漂移上限**,>10–15s 拦截点不可靠 → search_mode + 语言重捕获兜底;limitation 写明漂移上限存在。
  - **R-隐式反超**:若消融显示纯隐式长上下文 ≈ SEP,则显式结构卖点被削弱 → 预案:SEP 主张收窄到"长丢失 + 抗漂移可解释性",非通用增益。

---

## 附:一手来源

- 因果 AR:[Self Forcing (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/f4823f831af67a3ef15e41a85434422a-Paper-Conference.pdf) · [repo](https://github.com/guandeh17/Self-Forcing) · [Rolling Forcing](https://arxiv.org/abs/2509.25161) · [SVI](https://arxiv.org/abs/2510.09212) · [BAgger](https://arxiv.org/pdf/2512.12080)
- 记忆:[WorldMem](https://arxiv.org/abs/2504.12369) · [PERSIST](https://arxiv.org/pdf/2603.03482) · [TTT One-Minute](https://arxiv.org/abs/2504.05298) · [WorldPack](https://arxiv.org/abs/2512.02473)
- action-conditioned / VLA:[Matrix-Game 2.0](https://arxiv.org/abs/2508.13009) · [UWM (RSS 2025)](https://arxiv.org/abs/2504.02792) · [DUST](https://arxiv.org/pdf/2510.27607)
- SEP 输入底座:[Mobile Traffic Camera Calibration (2605.11900)](https://arxiv.org/abs/2605.11900)

**发表状态速查**:peer-reviewed = Self Forcing(NeurIPS'25 Spotlight)、WorldMem(NeurIPS'25)、UWM(RSS'25);据 repo 待复核 = SVI(ICLR'26 Oral)、PERSIST(ICML'26);其余为 arXiv 预印本。
