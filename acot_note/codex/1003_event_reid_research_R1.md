总评：**可行性：中**。多标注车辆能把“跨间隔可见”原始事件放大约一个数量级，但非目标事件分布明显偏。  
**创新性：中**。真正新意不在“所有物体都当样本”，而在“语言指代身份 + 断开 tracklet 后身份延续 + UAV 闭环预测拦截”的组合。  
当前 v5/v6 smoke 还不足以支撑主 claim：v5 有 FOV/划分/重捕获稀缺问题，v6.1 规格方向对，但冒烟样本还没稳定产出足够目标重入事件。  
我建议把数据集设计成“完整 episode + 事件级切片”双轨，评测必须基于会断开的 tracklet，而不是持久 actor ID。

**A. 可行性**

1. **“标注所有车辆”确实放大样本，但不能按 15 倍线性乐观估计。**  
   证据：v5 文档统计为 64 集、85,271 帧，目标车“≥3s 后重新入画且有相似车在场”只有 36 次；TAHRelM 离线缓存实际没有 ≥3s 缺席段，主要因 70°/90° FOV 错位（[1003_data-check.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-check.md:14), [1003_data-check.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-check.md:247)）。  
   我用 h5py 只读统计 v5 的 `annotation/off_screen` + `distractors/bbox`：平均每集 6.84 个场景 actor。按同一“中心在画面内”口径，≥3s 缺席后重入事件为：目标 20 次、非目标 394 次、全部 actor 414 次；非返回缺席为：目标 12 次、非目标 212 次、全部 224 次。也就是说，原始正样本可从目标车约 20 次放大到约 414 次，约 **20.7×**；若以文档的目标重捕获 36 次作分母，则是约 **11.5×**。  
   但这不是同分布放大：非目标 ≥3s 缺席开始时，目标仍可见的比例是 **557/606 = 91.9%**；v6 clean smoke 中也是 **111/120 = 92.5%**。这说明非目标多是“相机跟着目标转而被动离开”，不是“被跟踪目标丢失后需要主动重捕获”。代码也支持这个判断：候选只包含画面内车，目标出画时 `true_idx=-1`（[candidates.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/candidates.py:11)）；专家在目标可见时围绕目标/车群取景，目标丢失后切到真目标恢复（[scene_manager.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scene/scene_manager.py:1059)）；相似车还有回收与强制共视机制（[scene_manager.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scene/scene_manager.py:792), [scene_manager.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/scene/scene_manager.py:1191)）。  
   结论：非目标样本适合做 re-ID/关联的辅助训练和负样本，但直接混入目标重捕获训练，可能教会模型“被动离场/重现”的错误先验。

2. **“只需要丢失前后一小段”只对事件评测成立，对模型训练不完全成立。**  
   身份模块本身依赖可见历史：TAHRelM 记忆包含外观 EMA、位置和速度，并用候选相对预测位置/外观相似度打分（[tah.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah.py:107), [tah.py](/nvidia/hque/code/cyhe/uav-acot-track/train/tah.py:127)）。PGIF 的时间常数约 30s，λ=0.98、tick=0.6s（[1001_innovation-design.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1001_innovation-design.md:59)）。所以“丢失前 1-2 帧”不够，至少要有 5-15s 可见历史；若训练 PGIF 类长期证据，30s 上下的历史更合理。  
   另外，用户已给的 tick 事实很关键：TAHRelM 漂移起点中，可见期间 42 次、重入后 6 次、开局即错 9 次。文档也显示错锁时真目标仍可见 85–92%，tracklet 历史才是有信息的信号（[1001_innovation-design.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1001_innovation-design.md:41)）。因此数据不能只切“丢失前后”，还必须覆盖“可见期间相似车交会/并行/遮挡边缘导致的 ID switch”。  
   负样本也必须系统化：目标不在候选中、目标永不返回、只有错误相似车返回。否则模型会学成“总要在候选里选一个”。现有候选协议已经支持目标缺席帧 `true_idx=-1`（[candidates.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/rollout/candidates.py:12)），应该把它变成显式训练标签。

3. **对控制 DiT / 世界模型，非目标样本有用但有限。**  
   非目标轨迹可用于学习交通流、路网、遮挡先验、候选运动约束；v6 规格已计划逐 actor 位置、可见性、断开 tracklet、traffic blobs 和实例掩码（[1003_data-regen-v6-spec.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-regen-v6-spec.md:37)）。  
   但控制动作是围绕“当前指定目标”生成的。v6.1 里 loss window 内还会标记 `action_valid=0`，说明这些段不能无脑拿来做动作模仿（[recorder.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/recorder.py:251)）。若要把每辆车都变成控制样本，需要离线 retarget 后重算专家动作或生成反事实轨迹；仅靠已有 action 标签不成立。

**B. 创新性**

新的部分：  
语言指定车辆 + 空中主动闭环 + 断开 tracklet 后身份延续 + 丢失事件中心评测，这个组合有空间。仓库相关工作也把新颖性收窄到“长丢失后重现、多 look-alike 共视时，语言作为决定性消歧信号，并与 UAV 预测拦截耦合”（[811_related-work-and-positioning.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/811_related-work-and-positioning.md:84), [811_related-work-and-positioning.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/811_related-work-and-positioning.md:97)）。

已被占据的部分：  
MOT/re-ID 早就把一个视频里的多个 identity 都当训练样本；长时跟踪也早有离开-重检测；主动跟踪也已有遮挡规划。OA-VAT 已做 instance-level active tracking、prototype re-acquisition 和 occlusion-aware planning([openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Instance-level_Visual_Active_Tracking_with_Occlusion-Aware_Planning_CVPR_2026_paper.html?utm_source=openai))；CosFly-VLA 已强调 CARLA UAV 长遮挡课程训练([arxiv.org](https://arxiv.org/abs/2607.15004?utm_source=openai))；UAV-Track VLA 已有 CARLA 空中 VLA 与 89 万帧级数据([arxiv.org](https://arxiv.org/abs/2604.02241?utm_source=openai))；ReferTrack 已是 referring-then-tracking embodied paradigm([arxiv.org](https://arxiv.org/abs/2607.20061?utm_source=openai))；TrackVLA++ 已有 Polar-CoT 和 Target Identification Memory([arxiv.org](https://arxiv.org/abs/2510.07134?utm_source=openai))。  
因此，“用所有物体的间隔做训练样本”本身**不新**；可投稿的新意应放在“事件挖掘如何服务断开 tracklet 的语言身份延续”和“闭环 UAV 重捕获协议”。

**C. 数据集设计**

建议采用双层数据：

**完整 episode 层**：用于闭环控制、DiT、世界模型、最终 SR/重捕获延迟评测。保留完整 RGB、动作、相机位姿、actor 状态、实例分割、traffic blobs。v6.1 规格方向正确：FOV 90、Town03 留作测试、逐车标签、断开 tracklet、目标重入事件（[1003_data-regen-v6-spec.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-regen-v6-spec.md:11)）。

**事件样本层**：以 actor `a` 的可见 tracklet 断裂为中心。定义：  
`pre`：重入/缺席前 5-30s 可见历史；  
`gap`：不可见 ≥3s，分 3-5/5-10/10+；  
`post`：返回后 5-15s；  
`negative`：缺席后在搜索窗内不返回，或候选中只有相似干扰车。  
标签包括：query language、pre tracklet、post tracklet 是否同一 actor、每帧候选集、目标 absent 标志、候选 role、gap cause、重捕获延迟、是否误认。v6 已能按可见性间隔断开 `tracklet_id`，不可见 ≥1s 后再出现换新 ID（[labels_v6.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/labels_v6.py:154), [postprocess.py](/nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking/recording/postprocess.py:173)）。

采样与加权：  
目标车重入事件必须高权重；非目标事件只能作为辅助，建议 cap 到目标事件的 1-3×，不要按自然频率 10-20× 灌入。按角色、gap 时长、原因、共视相似车数、语言属性、城镇、目标框尺寸分层采样。可见期间 ID-switch 事件单独建 bucket，不能被“重入事件”淹没。

负样本：  
必须有 target-absent、never-return、wrong-lookalike-return、open-set traffic、目标被遮挡但未出画、目标出画但相似车仍在画面内。v5 只读统计里 ≥3s 缺席的 never-return 有 224 次，≥10s 有 210 次，这类样本足够但要正确标注为“不乱认”。

划分：  
按 episode/town/seed 划分，不能把同一集里的事件分到 train 和 test。当前 v5 的 Town05/Town10HD 泄漏到训练集（[1003_data-check.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-check.md:137)），必须改。v6 计划 Town03 test、val 由 episode_id 规则决定，这是正确方向（[1003_data-regen-v6-spec.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-regen-v6-spec.md:59)）。

评测协议：  
模型输入只给断开的 tracklet ID，不给持久 actor ID；actor ID 只用于 scorer。指标至少包括 re-acquisition recall、latency、false positive while absent、wrong latch duration、ID switch before loss、closed-loop reacq success。现有主指标只在目标在画面内计分，看不到“找回画面”的能力（[1003_data-check.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-check.md:259)）。

规模估计：  
旧 v5 的硬目标事件约 36/64 = 0.56 次/集。若要数百个目标级重捕获事件，至少需要 400-600 集自然/半构造 episode；v6.1 full plan 的 500 集规模是合理量级（[1003_data-regen-v6-spec.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-regen-v6-spec.md:25)）。但 smoke 还没达标：我在 clean v6 smoke 10 个正式 h5 中只看到 `events/target_reentry` 共 4 行，≥3s 目标重入按逐车 visible 统计仅 2 次；必须等 v6.1 生成稳定后再宣称数据充分。

**D. 最大风险与必须先做的验证**

最大风险是 **Sticky-ID 零假设**：用户已确认 sticky actor ID 与 PGIF 持平，说明旧结果可能被持久 ID 白送。必须先用断开 tracklet + 真实/扰动 MOT 复测。PGIF 对 ID 噪声也敏感，p=0.1/tick 时增益基本消失（[1001_innovation-design.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1001_innovation-design.md:159)）。

第二风险是 **非目标事件带来负迁移**。必须做三组训练：target-only、all-actors uncapped、all-actors balanced，在只含目标车断开重捕获的 test 上比较。

第三风险是 **冷启动语言决定上限**。PGIF 初始锁对时很强，初始锁错时反而坚持错误身份（[1001_innovation-design.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1001_innovation-design.md:125)）。必须单独报告首帧 grounding 准确率、共享开局机制和锁错恢复率。

第四风险是 **数据生成尚未闭环**。v5 有 FOV 错位、目标静止、遮挡/重捕获稀缺；v6.1 规格修得对，但 smoke 通过率与目标重入产量还要审计（[1003_data-check.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-check.md:200), [1003_data-regen-v6-spec.md](/nvidia/hque/code/cyhe/uav-acot-track/acot_note/1003_data-regen-v6-spec.md:107)）。

必须先做的验证实验：  
1. v6.1 全量前，跑事件审计：target/non-target、reentry/nonreturn、gap cause、lookalike 共视、每集事件数。  
2. 断开 tracklet 协议下复现 sticky、PGIF、TAHRelM、DAM4SAM。  
3. target-only vs all-actor 数据消融。  
4. target-absent 负样本校准实验：看 false reacq rate。  
5. 控制分解：oracle identity + learned control、learned identity + oracle intercept，分清 WHICH 与 WHERE/HOW。  
6. 未见城镇 Town03 test，且 test 前不调参。