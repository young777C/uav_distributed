# ACoT-UAV-Track 训练计划

> 2026-07-27 | 校准到 MVP 数据规模 | 配套：`acot-uav-design.md` §3.4/§4（同目录）、`../acot_probe/`、`../positioning-analysis.md`
> 训练数据：`/nvidia/hque/data/carla_data/mvp`（59 集 / 106,200 帧 / 22 GB）
> 训练机：8×A40（48 GB/卡）

---

## ⚑ 当前状态（2026-07-29）

| Phase | 状态 | 备注 |
|---|---|---|
| A1 修标签 / A2 划分 / A3 dataloader+预计算 | ✅ 完成 | `runs/ctx_cache`(5GB) + `runs/ctx_cache_ml`(mmap) |
| B1 骨干选型 | ⚠️ 部分 | PaliGemma 被 license 挡 → 直接选 Qwen3-VL-4B/第24层/100token；未做 head-to-head |
| C Stage-1 EAR | ✅ 完成 | 6s=18m 未达 <10m 门槛，但已诊断为不可约预测不确定性（相机系+proprio 把地板 14→7m） |
| D Stage-2 端到端 | ✅ 训练完成 | mis_follow ~2%；A(中性语言)+C(稳课程) 已做 3-seed 消融：C 稳健、A 只 ~20-27%（非2×） |
| **D 语言必要性门槛** | 🔴 **未通过（blocker）** | 见下 |
| E Stage-3 RL | ❌ 未启动 | 计划本就可选/后置 |

### 🔴 关键 blocker：benchmark 位置捷径（语言不 load-bearing）

正式 w/o-language 消融（3 seed）：困难子集(≥2候选) mis_follow **有语言 6.16%±1.81 ≈ 无语言 6.23%±1.12**，无差异。根因：**最优纯位置分类器(无外观/无语言)就能到 ~3-5%**（随机 70%、最居中 53%、最近 19%）→ 目标几何位置被系统性泄漏，任何模型不需语言即可选对。

- **验证工具**：`python -m train.benchmark_leak_probe --config <cfg> --split val`（数据修好后位置 floor 应升向 ~70%）。
- **修复（数据侧）**：目标位置/深度/排名与"它是目标"**去相关**——干扰车与目标共享同一位置分布，随机化目标占据的位置槽。见 memory `acot-uav-benchmark-leak`。
- **注意**：这是**数据生成 blocker，非模型 bug**；语言价值在堵住捷径前无法显现，是投稿致命风险。

### 训练基建（本轮新增）
- Stage-2 缓存改 **mmap 共享**（`.ctx.npy` + 瘦 npz）+ **workers=4** → 每 run 私有内存 48GB→2.5GB，可 8 卡齐跑。转换器 `train.mmap_cache`。
- seed/课程/grad_clip 全 config 可控（`train.stage2 --seed/--ckpt-dir/--device`）。

---

## 0. 数据体检结论（决定计划怎么排）

| 项 | 状态 | 影响 |
|---|---|---|
| 规模 | 59 集 / 106K 帧 / 每集 1800 帧(3min@10fps) / **2 地图**(Town03×29, Town04×30) | MVP 级：够 EAR 收敛 + 端到端可演示；**不够**强统计消融/跨地图泛化 |
| cam 位姿 / waypoint GT / 干扰车 3D | ✅ 59/59 齐、可用 | Stage 1 可直接用 |
| 相似干扰物 | ✅ 49/59：same_shape 16 + same_color 10 + same_class 23；distinct 10 | 语言 load-bearing 难例充足；可支撑课程 2a→2c |
| 目标类别 | car 36 / motorcycle 10 / bicycle 9 / scooter 4 | 多类，泛化面好 |
| 🔴 `annotation/{bbox,occlusion,search_mode}` | **被坏投影污染**（occlusion 假高均值 0.546；bbox 仅目标、无干扰车、写死 224²/忽略 pitch=-30°） | **Stage-2 的 L_visibility / L_target_id 依赖它 → 训练前必须重算** |
| waypoint GT (`annotation/waypoints`) | ✅ 用 3D 直接算，不经投影，未受污染 | Stage 1 可信 |
| `acot_probe/projection.py` | ✅ 真图验证正确（绿框稳落目标车，8/8 帧目标在画面内） | 作为重算标签的正确投影源 |

**结论**：数据主体可用；唯一地基问题是 **annotation 投影类标签被污染**，必须先修（Phase A1）。

---

## 1. 分阶段计划（按依赖/优先级排序）

### Phase A — 数据就绪（训练前必做；A1/A3 可在 CPU 完成）

- **A1【最高优先｜修标签】**
  用 `acot_probe/projection.py` 的正确投影替换 `carla_uav_tracking/recording/postprocess.py` 的 `_compute_bboxes` / `_compute_occlusions`：
  - 重算 **目标 + 全部干扰车** 的 2D bbox（当前仅目标）→ 供 Stage-2 `L_target_id`
  - 用正确投影（含 cam_pitch/cam_yaw、336²）重算 occlusion、search_mode
  - 在 59 集上重跑；验证 mean occlusion 回落到合理值、可视化抽查
  - waypoint GT 无需重算
- **A2 数据划分**
  按 episode 分 train/val/test（~47/6/6），按 strategy×class 分层。2 地图只能做**粗**跨地图检查（held-out 一张 town）；强泛化留给 scale-up。
- **A3 dataloader（当前不存在）**
  读 rgb + proprio(8d) + language + waypoint GT + 新 bbox + occlusion/search_mode；实现**冻结 VLM KV 预计算并缓存**（设计原则1），训练循环不再前向 VLM。

### Phase B — 骨干选型（§3.4 gate；现在即可跑，Stage 1 前必须完成）

- **B1**：`acot_probe` 在 ~15 集相似干扰子集上跑 **PaliGemma-2-3B vs Qwen3-VL-4B** 逐层探针（layers 8/12/16/20/24/last）→ 输出 backbone×layer 的 Target-Selection Accuracy + layer-16 PASS/WEAK 判定。
  - 前置：`pip install -r acot_probe/requirements.txt`；先 `python -m acot_probe.visualize` 抽查投影（本机已验证过 ep0）
  - 产出决定 EAR/IAR 读**哪个骨干、截断到哪层**；预期 PaliGemma 在 16 层过、Qwen 需 ~20-24
  - 运行：PaliGemma→cuda:0，Qwen→cuda:1（见 `acot_probe/config.yaml`）

### Phase C — Stage 1：EAR 预热（建模型 + 训练）

- 建 EAR（4 层 Transformer，d=256，flow-matching）+ 选定骨干的 per-layer KV 读取 + Stage-1 训练循环
- 冻结 VLM/IAR/DiT，仅训 EAR；监督 = waypoint GT（干净）
- **门槛**：waypoint MSE < 10m @6s；且 EAR 预测的是**目标车**未来轨迹而非干扰车（early 语言-grounding 检查）

### Phase D — Stage 2：端到端 + 难度课程

- 建 IAR + AGP/DiT(32层) + proprio-MLP + 全损失：
  `L_action + 0.3·L_ear + 0.1·(L_visibility+L_maneuver) + 0.2·L_target_id`
- **课程**（按 `strategy` 字段）：2a distinct → 2b same_class → 2c same_shape/same_color；每档过门再进
- **staleness augmentation**：随机复用过期 EAR/IAR 输出，匹配 S1(20Hz)/S2(1Hz) 异步推理
- 专家 = 特权信息蒸馏（PID 用目标真值，学生只有 RGB+语言）
- **门槛**：held-out 上 Mis-follow Rate 低，且 w/o-Language 消融时显著上升（=语言有净贡献；否则回 A/数据加难）
- **MVP 预期**：收敛 + 可演示 + **初步**消融；统计强消融/跨地图需 scale 到 ~300K + 更多地图

### Phase E — Stage 3：闭环 RL 精调（可选，后置）

- 冻结 VLM/EAR/IAR，训 DiT 末 4 层（或 LoRA）；GRPO，CARLA on-policy
- Reward 含 **R_correct_id**（跟对/跟错被指目标），防 RL 崩成"跟任意车"
- 门槛：SR 与 Mis-follow 均优于 Stage 2 BC；无 reward hacking

---

## 2. MVP 能交付 / 不能交付

- ✅ 骨干选型结论；EAR 收敛；端到端跟踪可演示；语言必要性的**初步**证据（w/o-language 消融、反事实语言）
- ❌ 统计显著的完整消融（数据量偏小）；强跨地图泛化（仅 2 地图）；SOTA 规模对比
- → 打通 MVP 后按 `acot-uav-design.md` §4.1 scale 到 ~300K 帧 + Town05 等更多地图

---

## 3. 立即可动（不需 GPU）与建议顺序

| 步骤 | 需要 | 谁/在哪 |
|---|---|---|
| **A1 修标签 + 加干扰车 bbox**（CPU，可在本机重跑 59 集验证） | h5py/numpy | 现在做 |
| **A3 dataloader + Stage-1 EAR 骨架** | CPU 建码，训练需 GPU | 接 A1 |
| **B1 骨干探针** | 4×A40 + 模型下载 | 你在 A40 上跑 |

**建议：先做 A1（修标签，地基）→ 你并行在 A40 跑 B1（定骨干）→ 再 A3+C（dataloader+EAR）→ D → (E)。**

---

## 4. 关联文件

> 本计划位于 `acot_note/`；设计文档同目录，其余在上一级 `uav-acot-track/`。

- 设计文档：`acot-uav-design.md`（同目录；§3.4 骨干选型/探针、§4 训练策略）
- 骨干探针包：`../acot_probe/`（`run_probe.py` / `projection.py` / `README.md`）
- 待修 bug：`../carla_uav_tracking/recording/postprocess.py`（`_compute_bboxes` 坏投影、仅目标）
- 定位/竞品：`../positioning-analysis.md`
- 训练数据：`/nvidia/hque/data/carla_data/mvp`（绝对路径）
