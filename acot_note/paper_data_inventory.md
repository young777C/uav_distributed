# 论文数据清单 · Paper Data Inventory

> 2026-09-06 · ACoT-UAV-Track 写作用:已完成实验 → 可入论文的真实数据(按章节)。**所有数字均为实测,可由 `eval_out/*.json` / `runs/*.log` + `carla_uav_tracking/rollout/continuous_metrics.py` 复现,无一编造。** 配套 memory `acot-uav-paper-writing-kickoff`(叙事定调 + FINAL PINNED NUMBERS)、`acot-uav-reid-reframe`(诊断全链)。

**闭环数据可比性(已验证)**:所有 closed-loop rollout 在 CARLA 按 seed 实时生成、不读训练数据、ckpt 固定 `stage2_v5` → 旧批次(`road_*`)与新批次(`paper_*`)**同配置结果一致**(track_correct_frac 0.626≈0.627,SR 0.118=0.118)→ 可同框引用。评测协议:20ep、seeds 91000-19、Town05、STEPS 900、s2=6。

---

## 1. 任务 & Benchmark(引言 / 方法 / 数据集卡)
- 任务:CARLA 空中 · 语言指定目标车 · 多 look-alike 消歧 · 闭环 · 长丢失预测-拦截重捕获(四轴交集 + 杀手场景)。
- 文档:`acot_note/experiment-design.md`(H0-H8 / 四条件 / 指标族)、`acot_note/acot-uav-design.md`(§6.3 核心贡献 / §6.1 招牌场景 / 架构)、`acot_note/dataset-card.md`、`acot_note/related-work-and-positioning.md`(竞品逐件切割 + novelty 押耦合)。

## 2. H0 语言 load-bearing(核心主张)
| 结论 | 真实数字 | 源 | 状态 |
|---|---|---|---|
| 语言对身份 load-bearing | with-lang mis_follow **0.579** vs no-lang **0.715**(**Δ≈13.6pp**;no-lang≈chance 0.76、with-lang≪chance) | `runs/train_v5.log` / `runs/train_v5_nolang.log` | ✅ 可引(离线 Stage-2 val,原数据;日志完好) |
| 去相关+分辨率因果链(为何 512px) | grid 8×8→16×16 mis 单调降;vlmpool look-alike cos 0.877 | `train/reid_resolution_probe.py` | ✅ 离线 |
> 叙事:语言对**任务/身份**必要,但**单帧语言 grounding** 闭环不足(见 §4 l1)→ 引出时序 reID。两者不矛盾。

## 3. WHICH-not-WHERE 诊断(头号发现 · 论文脊梁)
| 结论 | 真实数字 | 源(20ep) |
|---|---|---|
| WHERE 阶梯:WHERE→近完美 SR 仍不动 | track_s 24→65s,**SR 全 0**,reacquire_lock_wrong ~0.74 平坦 | `eval_out/road_{ear,road,road_oracle,cv_gated}.json`(另 `ex_*`,`reacq_nowm`) |
| **D1 = 专家取景 + reid-DINOv2** | **SR 0.579**,mis_follow 0.125,lock_wrong 0.182 | `eval_out/road_expert_reiddino.json` |
| **D2 = oracle 身份 + track 控制** | **SR 0.579**,mis_follow 0.0,track_s 51.7 | `eval_out/road_oracleid_track.json` |
| → 真上界 **0.58**、**~42% 场景难度地板**、瓶颈=identity×control 级联 | 两 oracle 独立均 0.579 | D1/D2 |

## 4. 主方法阶梯(头号结果表 · 20ep 同批次,已消 run 方差)
可部署 track 模式,只换 WHICH 机制(ckpt=stage2_v5);19 计分同种子配对:

| 客观指标 | l1_xattn | l2_vlmpool | l3_dino-K2 | l4_+conf | **l5_+temporal** |
|---|---|---|---|---|---|
| 正确跟踪帧占比 (1−mis_inst) | 0.251 | 0.519 | 0.653 | 0.671 | **0.738** |
| 最长错跟时长 max_wrong_run_s 中位(s) | 9.70 | 4.70 | 3.60 | 3.60 | **3.00** |
| 最长错跟时长 均值(s) | 9.44 | 5.23 | 4.20 | 3.78 | 3.18 |
| 无>5s错跟局占比 latch@5s | 0.105 | 0.632 | 0.684 | 0.842 | **0.895** |
| latch@3s | 0.053 | 0.158 | 0.316 | 0.368 | 0.474 |
| id_switches | 13.7 | 12.8 | 15.4 | 14.1 | 9.32 |
| q_reacq(每事件重捕获) | 0.448 | 0.466 | 0.651 | 0.607 | 0.471 |
| SR (latch) | 0.000 | 0.053 | 0.105 | 0.105 | 0.105 |

**正面表述(真实)**:正确跟踪时间占比 **~2.9×**(0.251→0.738)、最长错跟时长 **9.7→3.0s**(中位)、无长错跟局占比 **8.5×**(0.105→0.895)。
源:`eval_out/paper_l{1_xattn,2_vlmpool,3_dino,4_dino_conf,5_dino_conf_tavg}.json`;表由 `continuous_metrics.py` 生成。

## 5. 时序 reID + 甜点扫描(方法消融)
- `reid_tavg{0,0.4,0.5,0.7}`,seeds 91006-13(8ep)+ 91000-17(18ep 确认):**tavg=0.4 甜点**(最低 max_wrong_run_s + 最优 latch@2-3s + track_correct_frac +0.088 双复现)。源 `eval_out/reid_tavg*.json`。
- 机制:候选 actor idx 稳定 → 按 actor 累积匹配分 EMA、时序积分提交(MOT 轨迹级证据累积)。`policy.py reid_tavg`。

## 6. 负结果(诚实贡献 · 现 n=20)
`paper_l4(部署基线) vs neg_bc04(BC-居中) vs neg_rle0(RL-E0)`,20ep:

| 指标 | l4 | BC-b0.4 | RL-E0 |
|---|---|---|---|
| SR (latch) | 0.105 | **0.000** | **0.000** |
| track_fraction(全程跟踪占比) | 0.217 | 0.073 | 0.040 |
| q_reacq | 0.607 | 0.159 | 0.236 |
| track_correct_frac | 0.671 | 0.501 | 0.633 |

- 补充:BC blend 扫描(0.4/0.7/1.0 均失败)、RL-E0(SVG 可微 rollout)训练 R_center **0.78→0.98** 却闭环复现同失效 → **控制侧修取景不可行,居中在身份下游**。
- 源:`eval_out/paper_neg_{bc04,rle0}.json`、`bc_{base,center,b04,b07}.json`、`bc_e0.json`;脚本 `train/bc_center_train.py`、`train/rl_e0_center.py`、`train/rl_reward.py`(anti-hack 单测过)。

## 7. 度量方法学(可复用贡献)
- `carla_uav_tracking/rollout/continuous_metrics.py`:latch-SR(90s 内任意≥2s 错窗→永久失败,SR≈(1−q)^N)饱和无分辨力的实证(§4 l3→l5 SR 0.105 不动而连续指标在动;§5 时序身份更好却 SR 0)→ 主张报客观连续量(正确跟踪占比↔IDF1、id_switches↔IDSW、track 时长、重捕获率&延迟、最长错跟时长、latch@Ts),SR 仅作保守操作点。

## 8. 外部 baseline(⏳ 评测中,~4h 后可用)
- 接入同一闭环、只换 WHICH:`assoc`(DeepSORT/OC-SORT 时序关联,motion/deepsort 两模式)+ `dam4sam`(DAM4SAM CVPR'25 SOTA distractor-aware SAM2 记忆,独立容器+socket 桥)。
- 源(生成中):`eval_out/paper_ext_{assoc_motion,assoc_deepsort,dam4sam}.json`;进度 `acot_note/baseline_compare_906.md`。
- iKUN/JointNLT → future work(零样本域差、边际低;JointNLT 已 de-risk)。

## 9. 架构 / 图(dataviz)
- `acot_note/reid_temporal_architecture.html`(当前系统)、`reid_dinov2_gallery_pipeline.html`、`three_arm_zex_injection.html`、`data_pipeline.html`、`vlm_fusion.html`。

## 10. 补充/附录可用(离线探针)
- 特征探针:crop-DINOv2 look-alike cos **0.877→0.614**(2.7× margin);K-view gallery K2 mis 0.58→**0.46**(首破 0.5,无重渲);分辨率封顶 512(grid 16×16)。`train/reid_resolution_probe.py`。
- reid 度量微调(SupCon):marginal,非杠杆(`train/reid_metric_train.py`)。

---

## ⚠️ 写作必守的诚实 caveat(均为真实数据特征,严禁美化)
1. **不写"所有指标单调"** —— id_switches 非单调(l3 bump 13.7→12.8→15.4→14.1→9.32)。
2. **时序 reID 写"提升连续身份、不抬 latch-SR、q_reacq 长尾退化(0.607→0.471)"**,非"无 trade-off"。
3. **SR 场景地板 0.105** —— 身份大幅改善不移 latch-SR(∵ D1/D2=0.58、~42% 长遮挡不可恢复)。
4. **RL-E0 的 latch@5s=1.0 / max_wrong_run_s=2.53 是近零跟踪(track_fraction 0.040)的假象**,须用 track_fraction/SR/q_reacq 报其失败,勿引其 latch。
5. H0 在**原数据**(已删)离线 val,cite from logs;闭环旧/新批次已验证可比。

## 直接可入论文的表
- **主结果表**(§4,l1-l5,20ep)· **oracle 分解**(§3 D1/D2)· **负结果表**(§6)· **H0**(§2)· 时序甜点(§5)· 方法学(§7)。
- **待补**:外部 baseline 列(§8,~4h 评测完)。
