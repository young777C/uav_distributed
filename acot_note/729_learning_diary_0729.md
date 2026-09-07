# 学习笔记 · 2026-07-29

**主题:ACoT-UAV 训练数据流水线 & VLM 内部机制**
配套可视化:`data_pipeline.html`(数据流)、`vlm_fusion.html`(融合+训练/推理+connector)、`stage1_camera_frame_fix.html`(Stage-1 定位地板修复)。

---

## 一、数据流水线全景:原始视频 → 样本 → 预计算 → 模型

```
① 原始 episode 视频  →  ② 切成样本  →  ③ 冻结 VLM 预计算  →  ④ 单样本信息量  →  ⑤ 进入模型训练
```

- **① 视频**:60 段 CARLA 随机生成的跟车视频(`.h5`),= 44 train + 6 val + ~10 Town05(测试留出)。每段 `rgb (T,336,336,3)` @10fps + 逐帧标注(target/distractors/search_mode/遮挡/cam位姿/速度)。
- **② 切样本**:砍尾 60 帧(`horizon = max_horizon 6s × 10fps`)→ `valid = n-60`;窗口起点 `i = 0,3,6,…`(stride 3),每段封顶 200 → **44 段 = 8447 train 样本**。
- **③ 预计算**:每(帧+指令)过**冻结 Qwen3-VL-4B** → 存 `(5,100,2560)` 缓存(~27GB,6卡并行)。见第五节。
- **④ 信息量**:见第二节。
- **⑤ 训练**:每 epoch 用 `WeightedRandomSampler` **带放回抽 8447 次**,tier 加权课程 a→c。

**关键概念区分:训练 epoch ≠ 数据 episode**

| | 训练 epoch(ep0-49) | 数据 episode(视频) |
|---|---|---|
| 是什么 | 扫全训练集一遍 | 一段视频 |
| 数量 | 50 | ~60 |
| 轴 | 训练时间 | 数据内容 |

> 一句话:60 段视频切成 8447 个"单帧锚点"样本;50 个训练 epoch 就是把这 8447 个反复扫 50 遍。

---

## 二、一个样本的信息量:重感知、轻动作

| 类别 | 内容 | 数值量 |
|---|---|---|
| **输入** | `vlm_ctx_layers (5,100,2560)` | 1,280,000(99.4%) |
| | `cand_feats (3,2560)` | 7,680 |
| | `proprio (5)` | 5 |
| | **小计** | **≈129 万 float ≈ 5.15 MB** |
| **监督** | `action (16,5)` 80 + `waypoint (4,3)` 12 + 4 标量 | **≈96 个数** |

- **不对称 ≈13000:1**:巨大感知输入 → 极小动作/标签输出(VLA 典型形态)。
- **`(5,100,2560)` 拆解**:5=抽取层数(第4/8/12/16/24层);**100=图像 token 数(10×10 空间网格)**;**2560=每个 token 的特征维**(Qwen hidden size,换骨干会变,PaliGemma=2048)。
- **数值量 ≠ 真实信息量**:129 万 float 是**同一张单帧+语言**的冗余再表示,内在信息 ≈「一帧场景 + 目标此刻姿态」——**缺"目标速率"**(见第三节)。

---

## 三、单帧无历史,如何学"趋势"?

**前提修正**:输入不是纯静态——`proprio` 含 UAV 三轴速度(但**不是目标车速度**)。

趋势的三个来源(都不靠帧历史):
1. **单帧姿态即编码方向**:车头朝向、车道、道路几何 → VLM 读出"往哪开"(从几何推趋势)。
2. **UAV 自身速度**(proprio)→ 相机系 waypoint 扣除自我运动(Stage-1 把定位地板 14m→7m 的修复)。
3. **flow-matching 学条件分布**:学 `P(未来轨迹 | 单帧, 语言)` 的期望,不是物理外推。

**本质局限**:单帧定不准目标**速率** → horizon 越远误差越大(1s 8.7m → 6s 18.3m)。"定位已解,残差=预测不确定性",残差里一大块就是"无历史→速度未知"。

**为什么仍可行**:部署是 **dual-system 闭环**——S2(1Hz)+ S1(20Hz)滚动重规划,每帧只需局部正确,趋势靠高频纠正累积。

> 洞察:**速度是趋势的充分统计量**——与其堆帧历史,不如直接注入速度。省缓存的改进方向:把"目标速度"也喂进 proprio。

---

## 四、VLM 基础四概念

| 概念 | 含义 |
|---|---|
| **Token** | 把输入切块再各变成向量。文字→子词;图像→patch(100 块=10×10) |
| **Embedding** | 每个 token → 一个 2560 维向量(语义空间坐标) |
| **Self-Attention** | 每个 token 环顾其它 token,按权重把它们信息汇入自己;逐层吸收上下文 |
| **Causal Mask** | 每个 token 只能看它**左边**的 token → 顺序重要 |

---

## 五、图像 & 语言的 tokenize 与融合

**图像 tokenize**:输入是**完整 336×336×3 RGB**(非灰度);但 **token 数由空间 336×336 决定**(÷32≈10×10=100),3 个通道被 patch-embed **折进每个 token 的 2560 维特征**,不单独成 token/维。

**语言 prompt 来源**:每段 episode 的 `language` 属性(数据生成时写好),如
`"Pursue the beige Jeep Wrangler in the front-left, which is about to stop, maintaining ~60 m."`
= 身份(beige Jeep Wrangler)+ 方位 + 意图 + 距离。**整段 1800 帧共用同一句**。

**融合机制(语言如何进视觉)**:
1. 拼成**一条序列** `[文本 token] + [图像 token×100]`,**文本在前**。
2. 逐层**自注意力**:因果遮罩下每个图像 token 能 attend 到前面所有文本词;与描述匹配的图块(Query·Key 高)把"yellow/Jeep"的 Value **汇入自己**。
3. 堆叠多层 → **匹配指令的图块被"点亮"**,背景图块保持暗。
4. 抽出 100 个图像位置的隐状态 = `(100,2560)`。

> **关键**:`(100,2560)` 虽在图像 token 位置,但内容**已吸收语言**——不是纯图像特征。证据:换指令("track the **red** car")会点亮**不同**图块。

---

## 六、cand_feats 与目标消歧

- **候选**:目标车 + 所有在画面里的干扰车,每辆一个(`_candidates` 用 3D 投影得到框;目标出画则整样本跳过)。
- **cand_feats**:对每辆车的框,在 VLM 末层 10×10 网格上 `pool_box`(mean-pool)→ 每辆车一个 2560 维向量。形状 `(候选数, 2560)`。
- **用途**:`target_id 头(cand_feats + 语言条件 ctx)→ 选哪辆 → cross-entropy 监督(target_idx)→ 评估指标 mis_follow`。
- 之所以能消歧:图像 token 已被语言染色,pool 出的车特征天然含"是否匹配指令"的信号。

---

## 七、Connector ≠ 融合(重要区分)

```
图像 → ViT → [Connector] → [LLM decoder 自注意力] → (5,100,2560)
              对齐/压缩         ← 融合发生在这里
           进 LLM 前:只搬运    进 LLM 后:才融合
```

你列的 5 种其实分两层:

| 类型 | 层级 | 说明 |
|---|---|---|
| MLP 投影 ★ | connector **架构** | 投到 LLM 维、拼进序列,融合甩给 LLM 自注意力(本项目/LLaVA/Qwen) |
| 交叉注意力 | connector **架构** | 专门 cross-attn 层做文本→图像(Flamingo)——connector 里就融合 |
| Q-Former | connector **架构** | 可学习 query 蒸馏固定少量 token(BLIP-2)——进 LLM 前的部分融合 |
| 空间重排 ★ | token **管理** | 2×2 patch 折叠减 token(本项目 spatial_merge=2) |
| 动态分辨率 ★ | 分辨率**策略** | token 数随图尺寸变 |

★=本项目路径:**动态分辨率 + 空间merge2×2 + MLP投影 → LLM自注意力融合**。

> connector 决定"融合在哪发生":MLP投影(本项目)= connector 只对齐、融合留给 LLM,所以两者是清楚的两站;交叉注意力/Q-Former 才把两步合并。

---

## 八、训练流程 vs 推理流程

| | 训练(离线,有标签) | 推理(在线,无标签,闭环) |
|---|---|---|
| VLM | **预计算成缓存**(每帧跑一次) | **实时**跑(S2 慢系统 1Hz) |
| flow-matching | 学"噪声→动作"的**向量场**(算 loss) | 从噪声沿向量场**积分 ODE 采样**动作 |
| 反传 | 有,**只更新小模型**(VLM 冻结) | 无 |
| 闭环 | 无(逐样本) | 有(S1 20Hz 滚动重规划) |

多任务损失:`L = L_action + 0.3·L_ear + 0.1·L_vis + 0.1·L_man + 0.2·L_tid`。

> flow-matching 一体两面:训练学场、推理沿场走——同一网络,一个算损失、一个做生成。

---

## 九、实验与发现

### 9.1 方位陈旧性审计(`train/audit_bearing.py`)
问题:episode 级固定 prompt 里的"front-left / about to stop"会随采样帧变陈旧吗?
- 22.1% 声明≠实际方位,**但 0% 左右相反**(全是相邻,近中央);
- 结构性缓解:`cap200×stride3` → 只采样每段**前 ~600 帧**(离 prompt 最新鲜);
- 结论:方位噪声**轻且良性**,但意图从句("about to stop")未测,是更真的陈旧。

### 9.2 方案A 消融:剥离时变从句(**关键结果**)
把 prompt 归约成身份-only(`"Pursue the beige Jeep Wrangler."`),重建缓存重训:

| | baseline(原始语言) | 方案A(中性语言) |
|---|---|---|
| **mis_follow** | 3.5% | **1.75%(减半!)** |
| act_mse | 0.049 | 0.034 |

→ **证实时变从句是有害噪声**;身份-only 是更干净、永远正确的语言条件。这是"改预处理白赚一半"的免费提升。

### 9.3 Stage-2 后段课程不稳定
两次训练都在 ~ep40 出现 `act_mse` 从 0.05 飙到 1.0(`ear` 尖峰)。根因:课程 a→c 退火太陡 + a 档地板太低,后段几乎只喂高难度样本冲散动作头。best.pt 按 act_mse 选,已锁好点未受污染。

### 9.4 A+C 修复 + 3-seed 复现(进行中)
- **A**:采纳中性语言默认;**C**:`a_floor 0.15→0.25` + `ramp 1.2→0.9` + `grad_clip 1.0`(全 config 可控,默认复现原行为)。
- 起 6 run(baseline×3 + A+C×3)对比,拿均值±std **坐实 2×**。

---

## 十、工程教训

- **detached 容器**(`DETACH=1` → `docker run -d`)让训练脱离 Claude 会话独立存活(否则会话拆除时被杀,曾丢 2 次 run)。
- **日志重定向到 repo 文件**(`> runs/.../train.log`)——`--rm` 删容器后日志才留得下 traceback。
- **checkpoint 必须真存**:`stage2.py` 原本只跟踪 `best` 从不 `torch.save`,已补 best/last 保存。
- **CPU 线程超订**:80 核机器上 3 并发 run 各用满核 → load 116、GPU 饿死。修法:`OMP_NUM_THREADS=20`(3×20<80)。
- **预计算缓存是冻结的**:换 backbone/分辨率/语言处理都必须**重建缓存**;模型学不到 token 里没编码进去的信息(如目标速率)。

---

## 附:代码 & 可视化索引

| 文件 | 内容 |
|---|---|
| `train/dataset_stage2.py` | 样本切分、相机系 waypoint、proprio、cand_feats |
| `train/backbone_kv.py` | 冻结 VLM 预计算;`neutralize_language`(方案A) |
| `acot_probe/backbones.py` | Qwen encode(文本在前)、`pool_box` |
| `train/curriculum.py` | tier a/b/c、退火权重(a_floor/ramp 可控) |
| `train/stage2.py` | 联合训练、flow-matching、seed/grad_clip、ckpt 保存 |
| `train/audit_bearing.py` | 方位陈旧性审计 |
| `train/seed_exp.sh` | 6-run seed 对比(线程上限) |
| `acot_note/data_pipeline.html` | 数据流水线图 |
| `acot_note/vlm_fusion.html` | 融合 + 训练/推理 + connector 分层图 |
| `acot_note/stage1_camera_frame_fix.html` | Stage-1 定位地板修复对比 |
