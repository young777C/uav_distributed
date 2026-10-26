# 1024 系统级外部基线调研(2025–2026,场景相似 + 任务相似 + 代码开源)

2026-10-24。分三个方向检索:
1. 空中 / 无人机主动跟踪;
2. 具身 VLA 与语言条件跟踪;
3. 抗干扰 / 抗遮挡主动跟踪、跟车系统、带基线的 benchmark。

**核验方式**:
- 每个候选都实际打开 GitHub 或项目页，并对仓库做浅克隆，读取文件树和最后一次提交;权重查 HuggingFace。
- 仓库副本在 scratchpad 的 `repos/`(临时)。
- 网页摘要和克隆结果不一致时以克隆为准。例如 DAT 的权重：网页摘要称"没有",克隆的文件树里实际有 `GCVAT_ckpts/*.pth`,对应 2026-10-20 的提交。

**判定标准**:
- 系统级：从相机观测(加目标指定)直接输出控制;
- 2025–2026 年的工作;
- 代码可运行，最好有权重;
- 场景或任务与我们相近：空中视角、车辆目标、相似干扰、长时丢失与重新捕获。

## 0. 结论速览

1. **严格同时满足"空中 + 实例级身份 + 闭环控制 + 2025–26 + 代码和权重齐全"的，仍然只有 OA-VAT。**
   - 和我们最像的两个 CARLA 无人机 VLA 都没有可运行代码:UAV-Track VLA 是空仓库,CosFly-VLA 没有代码。
   - 正文要写明这两个"已核验，无可运行代码",只做文字对比。
2. **OA-VAT 有一个必须马上处理的变化**:
   - 论文标题的核心组件"遮挡感知规划器"(扩散策略)的权重已在 **2026-10-20** 放出(HF `Harry2222/follow-dp-unrealcv`)。
   - 我们现在的接入里这个组件是关闭的，当时的理由是权重未公开、原系统默认也关闭。
   - 闭环系统级比较应改为同时报"开启规划器(官方完整版)"和"关闭规划器"两个版本，否则会被认为拿掉了对方的核心组件。
3. **另有两个值得纳入的系统级候选，各代表一条不同路线**:
   - **ReferTrack**(VLA,显式在候选中选目标，有"目标不存在"输出;代码、训练和 4B 权重都已放出);
   - **GC-VAT**(NeurIPS'25,空对地强化学习主动跟踪，目标含汽车;权重 10-20 刚放出)。
   - 两者都需要在 CARLA 上重训或微调，成本不低。
4. **只引用、不跑的必引工作**:TrackVLA++、USS(框提示 + 在场判别，设定和我们最像，但无代码)、DeTrack、Track A*(CARLA,无代码)、NavFoM、CST-WM、VP-EVT。

## 1. 候选总表

| 名称 | 出处与日期 | 场景与任务 | 输入 → 输出 | 代码 / 权重 / 许可(已核验) | 接入难度 | 结论 |
|---|---|---|---|---|---|---|
| **OA-VAT** | CVPR'26,arXiv 2604.21453 | 空中跟人和车;用参考图做实例级指定;处理相似干扰;扩散规划器负责遮挡恢复 | RGB + 参考图 → 4-DoF 连续速度 | [SHWplus/OA-VAT](https://github.com/SHWplus/OA-VAT):训练和推理代码齐全，最后提交 2026-10-20("add DP checkpoint download link")。规划器权重在 HF,约 4.4 GB。README 声明 MIT。车辆目标走 Case A,不需要只适用于行人的 Pose2ID / IPG | 已接入(规划器关闭);开启规划器为**中**:规划器在 UnrealCV 的 Planning-20k 上训练，迁到 CARLA 有域差 | **必做**:规划器开 / 关两个版本 |
| **ReferTrack** | arXiv 2607.20061(预印本),2026-07 | 四足 / 人形跟人;YOLO11x + ByteTrack 给候选编号，再由 Refer-CoT 选编号;有 `NO_EXIST` 输出和历史选框队列;目标用语言指定 | 前视 RGB + 语言 → 3D 路点 | [MedlarTea/referTrack](https://github.com/MedlarTea/referTrack):评测、训练、数据引擎全部开源(9-30 完整发布)。权重 [ReferTrack-Qwen3-4B](https://huggingface.co/hjyeee/ReferTrack-Qwen3-4B)。Apache-2.0。最后提交 2026-10-08 | **中高**:要用 CARLA 车辆数据微调(1009 有设计稿);候选可以换成我们的协议候选框，与 OA-VAT 同感知;首帧框可作为选框队列的初值(这是我们的适配，需披露) | **强候选**:代表 VLA 路线，结构上和我们的 WHICH 正面竞争 |
| **GC-VAT(DAT 基准)** | NeurIPS'25,arXiv 2412.00744 | 空对地主动跟踪,24 个城市级场景，目标含汽车;**没有实例条件输入**(只认类别) | 84×84 RGB → 7 个离散动作 | [SHWplus/DAT_Benchmark](https://github.com/SHWplus/DAT_Benchmark):含强化学习训练代码。MIT。权重 `Alg_Base/DAT_Benchmark/models/R_VAT/GCVAT_ckpts/{citystreet,desert,downtown,farmland,lake,village}.pth`,2026-10-20 提交 | **中高**:从 Webots 零样本迁到 CARLA 域差大，直接比等于立稻草人;公平做法是用它的奖励和课程在 CARLA 重训 | **候选**:代表"类别级强化学习"路线，用来说明只认类别的策略分不清克隆车 |
| MiniCPM-RobotTrack | OpenBMB 2026-07,**没有论文** | 四足跟人(EVT-Bench),身份隐式 | 31 帧历史 + 语言 → 8 个路点 | [OpenBMB/MiniCPM-Robot](https://github.com/OpenBMB/MiniCPM-Robot):训练脚本 + 权重(0.9B),Apache-2.0,最后提交 2026-08-25 | 中 | 可选：没有论文，学术分量弱 |
| OmTrackVLA / VLX-Go | 没有独立论文 / arXiv 2610.19361 | 跟人，语言指定 | 视频 + 语言 → 路点 | Om 可训练(0.6B),但许可前后不一致(HF 写 MIT、README 写 Apache、仓库无 LICENSE);VLX-Go 没有训练脚本 | 中 / 高 | 可选 / 只能零样本 |
| Uni-NaVid | RSS'25 | 统一导航，含跟人;没有重捕获机制 | 视频 + 语言 → 离散动作 | [jzhzhang/Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid):MIT,7B 权重 | 中高 | 弱基线(审稿人熟悉，但 EVT-Bench DT 只有 31.9) |
| SPF(See, Point, Fly) | CoRL'25 | 空中，免训练 VLM;语言指令;没有身份记忆 | 图像 + 语言 → 2D 航点 → 控制量 | [Hu-chih-yao/see-point-fly](https://github.com/Hu-chih-yao/see-point-fly):**专有许可,All rights reserved** | 中 | 需作者书面授权，否则只能按论文复现并标"SPF-style" |
| LOVON | arXiv 2507.06747 | 开放词表(含车),只到类别级 | YOLO 检测 + 指令 → 速度 | MIT,有权重 | 低 | 仅参照 |
| Eva-Tracker / UAST | ICRA'26 / CVPR'26 | 空中跟踪 / 搜索 + 跟踪，带重捕获;没有外观身份 | 检测 + LiDAR / RGB-D → 轨迹 | AGPL-3.0 / MIT,ROS + C++ | 高 | 仅作飞行侧参照 |

## 2. 只引用、不跑(无可运行代码)

| 名称 | 为什么必须引 | 代码状态 |
|---|---|---|
| UAV-Track VLA(arXiv 2604.02241) | **CARLA**,无人机，车辆，语言;最近的竞品 | 仓库为空，没有任何 ref |
| CosFly-VLA(arXiv 2607.15004) | **CARLA**,无人机，行人，长时遮挡重捕获 | 没有代码，只有预发布数据集 |
| DeTrack / AaDWorlds(arXiv 2605.17451) | 空中闭环，有移动干扰物 | 没有代码 |
| Track A*(arXiv 2605.05338) | 在 **CARLA** 的 8 张地图上做可见性感知规划 | 没有代码;假设目标轨迹已知 |
| TrackVLA(CoRL'25)/ TrackVLA++ | 具身跟踪 VLA;TrackVLA++ 的 TIM 门控记忆和我们的记忆设计最接近 | TrackVLA 只放了 benchmark;TrackVLA++ 没有代码 |
| USS(arXiv 2606.25880) | **支持框提示 + 在场判别头**,设定和我们最像;相似人场景下框提示 90%、语言 45% | 没有代码链接 |
| NavFoM(ICLR'26)、CST-WM、VP-EVT、CoMaTrack | 跨具身导航跟踪 / 世界模型跟踪 / 可见性保持 / 跟踪 benchmark | 没有代码，或只有 README |
| FAn(RA-L'24)及"FAn + SAM2" | OA-VAT 论文把它当模块化基线 | FAn 开源但属 2024 年;"+SAM2"变体没有公开代码。我们的 DAM4SAM + 共享飞行栈可以表述为同一范式 |

## 3. 建议

**系统级(A 层)基线**,按优先级:

1. **OA-VAT 完整版，规划器开 / 关两个版本(必做)**。
   - 下载官方规划器权重，在闭环方案(1023)里加 `--dp_enable` 这一臂。
   - 预注册：调参网格两版相同;主表报官方完整版(规划器开启),关闭版放附录。
   - 开环 r5 不受影响：回放时由专家飞行，规划器不起作用。
   - 闭环方案的 codex 审查(R4)要把这项改动一并送审。
2. **ReferTrack**(若资源允许)。
   - 用我们的协议候选框代替 YOLO11x + ByteTrack,与 OA-VAT 同感知;用 CARLA 车辆数据按 1009 的方案微调;训练量与 Ours 相当。
   - 主实验 T1 不用语言：需要用模板语言(由首帧框生成)或首帧框初始化选框队列，两种设定都要在正文说明。
   - 成本：4B 模型微调加数据构造，按 1009 估计需要数天，CVPR 截稿前偏紧。
3. **GC-VAT**(二选一，或者放附录)。
   - 零样本迁移只能当参照;想作为正式基线，需要在 CARLA 用它的奖励和课程重训。
   - 它的价值在于说明"类别级主动跟踪在克隆车面前必然失败",这一点也可以由"永不切换 / MOT 退化"的现有结果部分支撑。

**模块级(B 层)**:调研没有发现新的 2025–26 开源身份模块;DAM4SAM 仍是建议恢复的那一个(见 1024 设计稿 §7 第 1 项)。

**正文写法**:
- 写清"最接近的两个 CARLA 无人机 VLA 截至 2026-10-24 没有可运行代码",并给出核验日期。
- TrackVLA++、USS 等在相关工作中按机制对比。

## 4. 风险与注意

- **感知不对等**:ReferTrack、GC-VAT 原本都从像素自己检测。前者可以改用我们的候选框;后者是端到端，无法共享感知，只能标注"自身感知"。
- **目标类型和视角**:所有开源 VLA 跟踪器都只在 Habitat 里训练过跟人，没有一个做过车辆或空中视角，必须同等预算微调后再比。
- **许可**:SPF 是专有许可;OmTrackVLA 许可不一致;OA-VAT 根目录没有 LICENSE 文件，只有 README 声明 MIT;TrackVLA 是 CC BY-NC-SA。论文附录的"使用的资产"要逐一列出。
- **检索不到、不要引用的名字**:Track-On-Robot、TrackGPT、"AVT ICCV 2025"。

## 5. 主要来源

- OA-VAT:https://arxiv.org/abs/2604.21453 ,https://github.com/SHWplus/OA-VAT ,https://huggingface.co/Harry2222/follow-dp-unrealcv
- ReferTrack:https://arxiv.org/abs/2607.20061 ,https://github.com/MedlarTea/referTrack ,https://huggingface.co/hjyeee/ReferTrack-Qwen3-4B
- GC-VAT / DAT:https://arxiv.org/abs/2412.00744 ,https://github.com/SHWplus/DAT_Benchmark
- MiniCPM-RobotTrack:https://github.com/OpenBMB/MiniCPM-Robot
- UAV-Track VLA:https://arxiv.org/abs/2604.02241 ,https://github.com/Hub-Tian/UAV-Track_VLA
- CosFly-VLA:https://arxiv.org/abs/2607.15004
- DeTrack:https://arxiv.org/abs/2605.17451 ;Track A*:https://arxiv.org/abs/2605.05338 ;USS:https://arxiv.org/abs/2606.25880 ;TrackVLA++:https://arxiv.org/abs/2510.07134 ;VP-EVT:https://arxiv.org/abs/2610.21999
- SPF:https://arxiv.org/abs/2509.22653 ,https://github.com/Hu-chih-yao/see-point-fly
- Eva-Tracker:https://arxiv.org/abs/2602.12549 ;UAST:CVPR 2026 open access;Uni-NaVid:https://github.com/jzhzhang/Uni-NaVid ;LOVON:https://github.com/DaojiePENG/LOVON
