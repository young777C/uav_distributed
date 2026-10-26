# runs/ 产物交接文档(2026-10-19)

> 用途:把 `runs/` 交接给下一个 agent。这份文档逐项说明 `runs/` 里每个产物是什么、怎么产生的、被谁消费、现在处于什么状态、能不能删。
> 核查方式:2026-10-19 只读清点了 159 个顶层条目,共 21 GB。`.pt` 只看了 key 和元信息,日志都读了全文,对照了 `acot/configs`、代码里的引用,以及 `acot_note/` 的笔记。推断出来的内容标"(推断)"。
> 状态标签含义:
> - 【部署中】:现行方法或协议直接加载。
> - 【论文引用】:数字进了笔记或论文表。
> - 【消融对照】:对照臂。
> - 【已被取代】:有更新的版本替代。
> - 【负结果存档】:实验结论为否定,留档。
> - 【调试/冒烟】:冒烟或调试产物。
> - 【环境/依赖】:运行环境依赖。
> - 【可删候选】:可以考虑删除。

---

## 0. 先读这 10 条

1. **`runs/` 整个被 `.gitignore` 忽略**。这里的权重、判定 JSON、codex 审查记录和**分析脚本**(`headroom/headroom.py`、`id_iter/recapture.py`、`p02_retrain/select_and_pin.py`、`mot_baselines/all_devB3_vs_sticky.py` 等)都只存在本机这一份。交接前建议备份,或者把关键分析脚本迁进 git。
2. **部署的策略权重是 `runs/stage2_v5/stage2_best.pt`,部署的身份头是 `runs/tah_relm_dag.pt`**。其余权重都是对照、消融或已被取代的版本。完整的权重位置见 §1。
3. **数据集不在 `runs/` 里**,在 `/nvidia/hque/data/carla_data/`。10-11 定稿的 64 集 90° 数据在 `mvp_v5_fov90_fast/`,清单是同目录的 `manifest.json`。`runs/datagen_moving/` 只放了生成它的编排脚本、日志和 QC 记录。同级的 `mvp_v5_fov90_fast_qcfail/` 和 `mvp_v5_fov90_fast_run1_contaminated/` **不得用于训练**。
4. **`stage2_v5` 无法复现**:它的训练数据是原始 90° 的 `mvp_full_v5`,09-04 被一次磁盘清理删掉了。现在盘上的 `mvp_full_v5` 是另一份 70° 重生成版。所有 Qwen 上下文缓存(`runs/ctx_cache_*`)也都不在盘上。另外它用的是 naive split,评测城镇 Town05 在训练集里(见 `acot/legacy/artefacts.yaml:79-89`)。
5. **闭环每集的明细不在 `runs/`**,在 `carla_uav_tracking/rollout/eval_out/`:
   - 新框架的结果在 `exp/<实验名>/`。其中只有实验级的 `summary.json`、`manifest.json`、`experiment.resolved.json`、`schedule.jsonl` 进了 git。
   - 旧入口的结果在 `paper_*.json`、`ex_*.json` 等文件里。
   - §8 有一张 runs 目录 ↔ 实验名 ↔ 笔记的对照表。
6. **大量文件属主是 root**(在 docker 里写入的),hque 删不掉,只能用 sudo 或在容器里删。主要是 `.pt`、`mp4_preview/`,以及 `opening_dump/`、`asd/`、`tah_v90/` 的大部分文件。
7. **不能动的东西**:被 `acot/legacy/artefacts.yaml` 按 sha256 钉住的 legacy artefact 实体文件,以及 L0 黄金测试要调用的 `.sh`(清单见 §2.2)。挪动它们不会立刻报错,要等下次校验才暴露。
8. **FOV 错位**:09-04 重生成的数据图像是 70°,`train/config_v5.yaml` 却按 90° 投影。受影响的有 `tah_cache.pt`(因而 `tah_relm_dag.pt` 的离线部分也受影响)、`reid_metric_cache*`、BC/RL、Qwen LoRA。不受影响的有 `stage2_v5`(它用的是被删掉的原始 90° 数据)、tid 系列、`tahdag/` 和 10 月以后的产物。
9. **当前进度**:10-19 第 0 步上限检查得到 D = +0.089 < 0.10,按预注册规则 STOP,停止在现有协议上迭代身份模块。下一步待办在 `acot_note/1019_metric_protocol_discussion.md` §4–§5,**全部等用户拍板,还没动代码**。另有一个已知 bug:`acot/eval/scheduler.py` 里 `restart_carla` 抛出的异常没有被捕获,会导致调度器死锁,跑下一批之前要先修(见 §8.3)。
10. **9 月的 re-ID 和 TAH 数字大多是在 GT 种子图库或 GT 记忆下得到的**,包括 l2–l5、reidmotion、assoc、tah_*_gt。可部署版本要低得多。1007 之后,旧协议的数字一律只作历史参考。

---

## 1. 权重总表:我们的模型、外部基线、消融各在哪里

### 1.1 我们的模型(Ours = `acot/configs/methods/ours_final.yaml` + 协议 `sim_proxy_v2@1`)

| 组件 | 权重位置 | 大小 | 谁加载 | 备注 |
|---|---|---|---|---|
| Stage-2 策略:EAR、IAR(1 层)、DiT、自带的 xattn tid 头 | `runs/stage2_v5/stage2_best.pt` | 554 MB | `acot/configs/protocols/deploy_v1@1.yaml:22`、`sim_proxy_v2@1.yaml:24`(`vlm.stage2_ckpt`) | 08-17 训练,epoch 20,mis_follow 0.579。artefact 为 `stage2_ckpt/stage2_v5@b387`。不可复现,原因见 §0.4 |
| VLM 骨干 Qwen3-VL-4B-Instruct(冻结,只取 L24) | `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct` | 8.3 GB | `train/config_v5.yaml` 的 `model_id`。容器把宿主的 `~/.cache/huggingface` 挂成 `/root/.cache/huggingface`(`acot/configs/infra/default.yaml:44` 的 `paths.hf_cache`) | 在 HF 缓存里,不在仓库内 |
| 身份头 TAHRelM(4,740 参数) | **`runs/tah_relm_dag.pt`** | 22 KB | `ours_final.yaml` 的 `identity.tah.ckpt` | 09-08 由 `a1_finish.sh` 训练,命令是 `tah_train.py --cache runs/tah_cache_dag.pt --motion --aug --epochs 60 --wd 3e-4`,没有指定 `--seed`。训练数据是 64 集离线数据(70° 图像按 90° 投影,步长 0.2 s)加 23 集 DAgger(0.6 s) |
| crop 外观特征 DINOv2 ViT-S/14 | `~/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m` | 85 MB | `acot/runtime/support.py:5`、`frontend.py:24` | timm 预训练权重,Ours、re-ID 和 MOT 基线共用 |
| 开局属性分类器(协议的共享前端,所有方法都用) | `carla_uav_tracking/rollout/eval_out/opening_attr_cls.pt`(线性头)、`opening_attr_ft.pt`(微调) | 126 KB / 83 MB | `sim_proxy_v2@1.yaml:36-37` | **在 eval_out/ 里,不在 runs/**。`eval_out/*.pt` 被 gitignore,属主 root。用 `runs/opening_dump/npz_tr` 训练 |

### 1.2 外部基线(方法 yaml 都在 `acot/configs/methods/`)

| 基线 | 方法 yaml | 权重位置 | 备注 |
|---|---|---|---|
| DAM4SAM(Hiera-L,默认) | `ext_dam4sam.yaml`(`_base.yaml` 里 `dam4sam.model: sam21pp-L`) | `external/DAM4SAM/checkpoints/sam2.1_hiera_large.pt`(898 MB) | 通过 socket 服务 `carla_uav_tracking/rollout/dam4sam_service.py --model sam21pp-L` 运行,映射关系在 `external/DAM4SAM/utils/utils.py:29`。10-16 用户已决定把它从论文主比较中移除 |
| DAM4SAM-S(Hiera-S) | `dam4sam_s_devB3` 实验覆盖成 `sam21pp-S` | `external/DAM4SAM/sam2.1_hiera_small.pt` → 软链接到 `checkpoints/sam2.1_hiera_small.pt`(184 MB) | 软链接是 10-15 建的,因为 `utils.py` 对 S 版写的是 `../sam2.1_hiera_small.pt`。同目录还有 tiny 和 base_plus |
| OA-VAT(CVPR26) | `ext_oavat.yaml` | `external/OA-VAT/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`(343 MB) | 由 `acot/runtime/head_oavat.py` 加载。**没有我们自己训练的权重**,`learned_ckpt: null`。依赖库在 `runs/pydeps_oavat/` |
| OA-VAT 的 DINOv2 变体 | `ext_oavat_dinov2.yaml` | timm DINOv2 ViT-S/14(HF 缓存) | 与 Ours 用同一个特征骨干 |
| re-ID(K 视图外观图库) | `ext_reid.yaml`、`ext_reid_pgif.yaml` | **没有训练权重**,只用 timm DINOv2 ViT-S/14 | `reid: {feature: dinov2, bank: 2, topm: 2}`。9 月训练的 SupCon 投影头 `runs/reid_metric_cache_proj.pt` 没有被任何运行时代码加载 |
| DeepSORT(原版复现) | `ext_deepsort.yaml`、`ext_deepsort_s2.yaml` | **没有训练权重**。原版的 MARS 128 维 CNN 换成了共享的 DINOv2 crop 特征(384 维) | 代码在 `external/deep_sort` @ f08cf1d,通过 `acot/runtime/head_mot.py` 接入。依赖 `runs/pydeps_oavat/`(scipy) |
| OC-SORT(原版复现) | `ext_ocsort.yaml`、`ext_ocsort_s2.yaml` | **不用权重**,只做运动关联 | 代码在 `external/OC_SORT` @ 8462e7e。依赖 `runs/pydeps_mot/`(filterpy) |
| 旧版 assoc(DeepSORT 式 / 纯运动) | `ext_assoc_deepsort*.yaml`、`ext_assoc_motion*.yaml` | 不用权重 | 9 月旧协议用过 |
| 语言 xattn 头 | `lang_xattn.yaml`(`identity.head: xattn`) | 在 **`runs/stage2_v5/stage2_best.pt` 的 `tid` 子模块**里 | 不是 `runs/tid_xattn.pt`。tid 系列权重都没有闭环入口 |
| 永不切换零假设 | `null_sticky.yaml` | 不用权重 | — |
| 真值身份头(上界诊断) | `diag_oracle_head.yaml` | 不用权重 | 第 0 步 headroom 的 U1 |
| `external/` 下其他仓库 | JointNLT、OmTrackVLA、referTrack、UAST、dinov3 | **没有任何方法 yaml 或 runtime 引用**,权重情况未核实 | referTrack 的微调计划见 `1009_refertrack_finetune_plan.md` |

### 1.3 消融模型

**A. 身份模块的机制消融:不换权重,全部复用 `runs/tah_relm_dag.pt`。** 这些 yaml 都 `extends: ours_final`,只改开关:
`ours_no_pgif`、`ours_no_temper`、`ours_no_absent`、`ours_absent_cv`、`ours_no_gate_commit`、`ours_no_gate_memory`、`ours_no_force_conf`、`ours_pgif_bayes`、`ours_pgif_resid`、`diag_gt_memory`、`diag_oracle_absent`。

**B. 身份头换权重的变体:**

| 方法 yaml | 权重 | 含义 / 结论 |
|---|---|---|
| `ours_dinov3.yaml` | 仍然是 `runs/tah_relm_dag.pt`,只把特征零样本换成 DINOv3(`external/OA-VAT/weights/dinov3/…pth`) | DINOv3 零样本换特征 |
| `ours_rt_s0.yaml` | `runs/tah_1006/tah_relm_s0.pt` | G13:用现配方重训 DINOv2 版,检验与原权重是否等价。结果 −0.154,**不等价** |
| `ours_v2off.yaml` | `runs/tah_1006/tah_relm_v2off_s0.pt` | 只用 64 集离线数据,DINOv2 |
| `ours_v3off.yaml` | `runs/tah_1006/tah_relm_v3off_s0.pt` | 只用离线数据,DINOv3。第一阶段 v3off − v2off = +0.079,通过 |
| `ours_v3full.yaml` | `runs/tah_1006/tah_relm_v3full_s0.pt` | DINOv3,离线 + DAgger。第二阶段 −0.089,不通过 |
| `ours_fast_a.yaml` | `runs/artefacts/tah_ckpt/tah_relm_fast_a_s1@1ae00e7a2668/ckpt.pt` | P0-2 stage A,新 90° 数据。+0.060,FAIL |
| `ours_fast_b.yaml` | `runs/artefacts/tah_ckpt/tah_relm_fast_b_s1@6494b5d59771/ckpt.pt` | P0-2 stage B,加了 DAgger v3。+0.016,FAIL |
| (没有方法 yaml,只有配方) | `runs/tah_v90/tah_relm_v90_s{0,1,2}[_last].pt`;复现版在 `runs/artefacts/tah_ckpt/tah_relm_v90_s0_repro@…` | 10-03 修正 FOV 后重训。dev 集 0.573 对旧权重 0.646,没有提升 |
| (没有闭环) | `runs/artefacts/tah_ckpt/tah_relm_t6_ctl_s{0,1,2}` | P0-2 的旧数据对照组,只有离线结果 |
| (离线) | `runs/abl_allactor/runs3/w/*.pt`(106 MB) | TAH-X、rn、rnw 等阶梯,以及全车 vs 只用目标的离线消融(1003 §7–§10) |

**C. 策略侧(Stage-2)消融:**

| 权重 | 含义 | 能否引用 |
|---|---|---|
| `runs/stage2_v5_nolang/stage2_best.pt` | 无语言(H0)。mis 0.715,有语言版是 0.579 | ✓,论文 H0 的数字 |
| `runs/stage2_v5_fair/stage2_best.pt` | 先预热 EAR,IAR 用 5 层的"公平版" | 离线对照,闭环没用过 |
| `runs/stage2_v5_noear/`、`stage2_v5_noiar/` | 去掉 EAR、去掉 IAR | ✗。**只有 epoch 0 的权重**,H1/H2 的闭环从来没产出过 |
| `runs/stage2_bc_center{,_b0.4,_b0.7}.pt`、`runs/stage2_rl_e0.pt` | 控制侧修取景:BC 居中、RL E0 | 负结果。用的是 FOV 错位的数据。**ckpt 里的 epoch/metrics 字段是从基座复制过来的,会误导** |
| `runs/qwen_lora/r16/best.pt`、`r0/best.pt` | Qwen LoRA grounding(10-02) | 负结果。没有达到有效判据 |
| `runs/tid_*.pt`、`tid2_*.pt`、`tid_attrbind*.pt`、`tid_d512.pt`、`tid_combined.pt`、`tid_xattn.pt` | 8 月离线 grounding 头系列 | 负结果。除 attrbind 外没有闭环入口 |

---

## 2. 现行依赖:哪些 runs/ 文件正在被引用

### 2.1 `acot/configs` 里直接引用的 runs/ 路径

| runs 路径 | 引用者 |
|---|---|
| `runs/stage2_v5/stage2_best.pt` | `protocols/deploy_v1@1.yaml`、`protocols/sim_proxy_v2@1.yaml`。@2/@3 继承它 |
| `runs/tah_relm_dag.pt` | `methods/ours_final.yaml`(以及所有不覆盖 ckpt 的子类)、`methods/ours_rt_s0.yaml`、`seeds.yaml` |
| `runs/tah_cache.pt` | `methods/ours_v2off.yaml` |
| `runs/tah_cache_dag.pt` | `methods/ours_rt_s0.yaml`、`experiments/t0_dinov3_dagger{,2}.yaml` |
| `runs/tah_1006/tah_relm_{s0,v2off_s0,v3off_s0,v3full_s0}.pt` | `methods/ours_{rt_s0,v2off,v3off,v3full}.yaml` |
| `runs/tah_1006/tah_cache_dinov3{,_dag}.pt` | `methods/ours_v3off.yaml`、`ours_v3full.yaml`、`experiments/t0_dinov3_dagger{,2}.yaml` |
| `runs/artefacts/tah_ckpt/tah_relm_fast_{a,b}_s1@…/ckpt.pt` | `methods/ours_fast_{a,b}.yaml` |
| `runs/p02_retrain/select_and_pin.py` | `methods/ours_fast_{a,b}.yaml`、`experiments/p02_fast_b_devB3.yaml`。这个脚本会改写上面两个 yaml |
| `runs/tah_v90/b_manifest.json`、`collect_dagger.sh` | `seeds.yaml`、`experiments/legacy/dagger_v2.yaml` |
| `runs/tah_v90/off90.pt` | `prep/tah_seq_v5_fov90.yaml` |
| `runs/tah_v90/tah_relm_v90_s0.pt` | `recipes/tah_relm_v90_s0_repro.yaml` |
| `runs/opening_dump/{chains_attr,f_oracle_absent,f_v4,run_openval}.sh` | `experiments/legacy/*.yaml`、`protocols/sim_proxy_v2_openval@1.yaml`、`methods/diag_oracle_absent.yaml` |
| `runs/acot_logs/aa_legacy.sh` | `experiments/aa_noise.yaml` |

代码里的硬编码引用(不在 configs 里):
- `runs/artefacts/`:`acot/artefacts.py`。
- `runs/infra/`:`acot/infra/lease.py`、`acot/provenance.py`。
- `runs/pydeps_mot`、`runs/pydeps_oavat`:`acot/runtime/head_mot.py:37`、`head_oavat.py:33`。
- `runs/pydeps_dash`、`runs/datagen_moving/dashboard.json`:`carla_uav_tracking/scripts/datagen_dashboard.py`。
- `runs/id_iter/replay_mot.py`:`acot/runtime/head_mot.py`。
- `runs/asd/iter/…`:`rollout/iter_mech.py`、`trk_tune.py`、`trk_quality.py`、`dam_diag.py`。
- `runs/opening_dump/npz*`:`rollout/opening_score.py`、`opening_cls.py`、`acot/train/opening_attr.py`。
- `runs/datagen`:`acot/data/gen.py:18` 的 `GEN_ROOT`。

### 2.2 绝对不能删或挪的文件

- **legacy artefact 的实体文件**:`acot/legacy/artefacts.yaml` 登记并按 sha256 钉住了它们,`runs/artefacts/` 里只放 manifest。包括:
  - `runs/tah_relm_dag.pt`、`runs/tah_cache.pt`、`runs/tah_cache_dag.pt`
  - `runs/tah_v90/{off90,dag_old,dag_v2}.pt`、`runs/tah_v90/tah_relm_v90_*.pt`
  - `runs/tah_1006/tah_relm_*.pt`
  - `runs/stage2_v5/stage2_best.pt`
  - `runs/opening_dump/npz_tr/`、`npz_hi/`
- **L0 黄金测试会调用的脚本**(见 `acot/tests/test_golden_l0.py`、`acot/tests/golden/harness.py`):
  - `runs/asd/iter/launch_wd.sh`
  - `runs/opening_dump/{chains_attr,run_openval,f_v4,f_oracle_absent}.sh`
  - `runs/tah_v90/collect_dagger.sh`
- **环境依赖目录**:`runs/pydeps{,_dash,_mot,_oavat}/`(见 §7.8),以及 `runs/infra/`(有 eval 在跑时绝对不能删)。

### 2.3 悬空引用:代码或配置里写了,但磁盘上不存在

下面这些路径都已不在盘上。新 agent 看到相关配置时**不要以为能直接跑**:
- **`runs/ctx_cache*` 全部**:`ctx_cache_v5`、`ctx_cache_ml_v5`、`_nolang`、`v3`/`v4`/`full`/`r3`/`neutral` 等 18 个。它们是 Qwen3-VL 的上下文缓存,`train/config_*.yaml` 和 `rollout/parity_online_vlm.py` 依赖它们。要重训 Stage-2,或做依赖 `ctx_cache_ml_v5` 的探针,必须先用 `train/backbone_kv.py` 重建。按 `train_v5_e1.sh` 注释估算,只存 L24 fp16 约 43 GB,5 层 fp32 约 426 GB,都要跑数小时(1003_dynamic-retarget-design.md 记录过这个阻塞)。
- **v1–v4 的 Stage-2 权重目录**:`runs/stage2`、`stage2_full{,_nolang}`、`stage2_full_v2{,_nolang}`、`stage2_neutral`、`stage2_r3`、`stage2_v3{,_nolang,_stab,_stab_nolang}`、`stage2_v4{,_nolang}`。
- **BC/RL 的缓存**:`runs/bc_center_cache.pt`、`runs/bc_center_cache2.pt`。
- **其他**:`runs/seed_exp/`、`runs/sweeps/`、`runs/debug/…`、`runs/rollout_m0.json`、`runs/asd/iter/damdiag_g`、`runs/tah_v90/{gen_fov90,pipeline}`(是前缀匹配误报,实际文件是 `gen_fov90.sh`、`pipeline.sh`)。

**训练配置里的一个静默陷阱**:`train/config_v5.yaml` 的 `ear_ckpt` 指向 `runs/stage1_ear_v5`。`train/stage2.py:102` 只在文件存在时才加载它。08-17 训练 stage2_v5 时这个文件还不存在,所以**部署权重没有经过 EAR 预热**。现在这个文件已经存在,用同一个 config 重训会**自动**预热,配方就和部署权重不一样了。

---

## 3. 目录地图(时间顺序索引)

| 时期 | 主题 | runs/ 条目 | 详见 |
|---|---|---|---|
| 07-27 → 08-18 | Stage-1/2 策略训练,v1→v5 逐步去掉捷径 | `stage1_*`、`stage2_v5*`、`train_*.log`、`wolang_full.log`、`sanity_r3.log`、`r3_train_verify.log`、`ablA.log`、`bench_mmap.log`、`mmap_convert.log`、`audit_bearing.log`、`proj_check_ep0.png`、`probe_v*_result.txt`、`recrop_probe.log`、`gate_v5*.log`、`intercept_eval*.log`、`baseline_eval.log`、`ex_source_eval.log`、`expert_control.log` | §4 |
| 08-19 → 08-31 | 语言 grounding 头(tid、attrbind、DAgger),WHICH-not-WHERE | `tid*.pt`、`grounding_*.log`、`gtrain_*.log`、`guardrail_test.log`、`binding_sweep.log`、`capacity_d512.log`、`combined.log`、`attrbind_*.log`、`track_*.log`、`unk_probe.log`、`town03_test.log`、`seen_seed_test.log`、`ground_strat.log`、`ladder_road_*.log`、`dagger_*`、`dagger_v5/` | §5 |
| 09-04 → 09-05 | 控制侧修取景(BC/RL) | `stage2_bc_center*.pt`、`stage2_rl_e0.pt`、`bc_*.log`、`rl_e0_*.log` | §4.C |
| 09-04 → 09-13 | re-ID、重锚、TAH → TAHRelM | `reid_*`、`reidmotion_*`、`reanchor_*`、`langtavg_*`、`paper_batch.log`、`assoc_eval.log`、`dam4sam_eval.log`、`ext_rerun.log`、`conf_consensus_eval.log`、`tah*.pt`、`tah_*.log`、`tahdag/`、`tahdag_collect.log`、`a1_*.log`、`b1_finish.log`、`eval_*.log`、`deploy_eval.log` | §5 |
| 10-01 → 10-04 | PGIF/ASD 闭环迭代(旧入口)、数据审计、v6、FOV 修正后的重训、开局修复 | `asd/`、`data_audit/`、`data_v6/`、`tah_v90/`、`opening_dump/`、`abl_allactor/`、`mp4_preview/`、`datagen/`、`pydeps/` | §6 |
| 10-02 | Qwen LoRA grounding | `qwen_lora/` | §4.F |
| 10-04 → 10-08 | acot 框架,T0 阶段,1007 闭环主表 | `acot_logs/`、`tah_1006/`、`artefacts/`、`pydeps_oavat/` | §6 |
| 10-09 → 10-12 | 64 集 90° 最小重生成,数据看板 | `datagen_moving/`、`pydeps_dash/`、`arch_check/` | §6 |
| 10-12 → 10-19 | P0-2 重训、身份诊断、MOT 原版复现、DAM4SAM-S、headroom | `p02_retrain/`、`id_iter/`、`mot_baselines/`、`dam4sam_s/`、`reacq/`、`headroom/`、`infra/`、`pydeps_mot/` | §7 |

---

## 4. Stage-1 / Stage-2 策略权重谱系与控制侧实验(07-27 → 10-02)

**谱系概述**

1. **7 月底,在 mvp 上打通管线。** 数据是 59 集、336px。先做 Stage-1 EAR 预热(`stage1_ear/`),再做 Stage-2 联合训练(EAR + IAR + DiT + grounding 头 tid)。
2. **08-11,扩到 mvp_full 全量(61 集)。**
   - `train_full.log`(有语言)和 `wolang_full.log`(无语言)是一对对照。
   - v2(`config_full_v2`)只把 target_id 权重从 0.2 提到 0.5。
   - v2 跑出的 0.217 / 0.000 是假突破:目标总在候选 idx0,argmax 遇到平票时又偏向 idx0。`train_v2_ORDERBUG.log` 是 `train_v2.log` 的逐字节副本,只是用来标记这个指标漏洞。
   - 修复方法:`dataset_stage2` 每帧打乱候选顺序。
3. **08-12 → 08-17,逐版去掉几何捷径。**
   - v3 去掉深度和跟距捷径,v3_stab 让训练更稳。
   - v4 去掉居中捷径。
   - 到 v4,语言依然不起作用。`gate_v5*`、`recrop`、`probe_v*` 这几组探针把根因定位到分辨率。
   - v5 改成 512px 后,E1 第一次成立:有语言 mis_follow 0.579,无语言 0.715。
4. **部署的是 `runs/stage2_v5/stage2_best.pt`。**
   - 08-17 训练,epoch 20,按 mis_follow 选出。
   - 只用 Qwen3-VL 第 24 层(n_layers_in=1)。
   - **EAR 没有经过 Stage-1 预热**,在 Stage-2 里从零开始训练。
   - `stage2_v5_fair` 是 08-18 补的公平版(EAR 预热 + IAR 5 层),闭环没有用它。
5. **9 月初的控制侧尝试,全部被证伪。** 在 stage2_v5 基础上做 BC 居中和 RL E0,都只改 DiT 的最后 4 个 block 和输出层。10-02 的 Qwen LoRA grounding 也没达到预先登记的判据。
6. **已经不在盘上的东西**:v1–v4 的全部 Stage-2 权重、所有 ctx_cache、原始 90° 的 mvp_full_v5、mvp/mvp_review_r3/mvp_full/v3/v4 数据。

本节文件合计约 7.9 GB,权重几乎全部属 root。

### 4.A Stage-2 v5 系列(部署与消融)

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `stage2_v5/stage2_best.pt` | 554 MB / 08-17 01:24 / root | `train/train_v5_e1.sh` → `python -m train.stage2 --config train/config_v5.yaml --device cuda:2` | keys:ear(5,071,107)、iar(972,866,n_layers_in=1)、dit(136,778,757)、tid(2,296,577)、epoch=20、metrics{mis_follow 0.5789, act_mse 0.0405}、dims{cond 2560, K4, H16, A5}。**没有 hparams 键**,因为 stage2.py 是 09-03 之后才开始写这个键的。训练数据是原始 90° v5(已删),split 是 naive,Town05 在训练集里 | `protocols/deploy_v1@1.yaml:22`、`sim_proxy_v2@1.yaml:24`;`acot/legacy/artefacts.yaml:81`;旧脚本 `eval_*.sh`;它也是 BC/RL 的基座 | 【部署中】【论文引用】 |
| `stage2_v5/stage2_last.pt` | 554 MB / 08-17 02:08 / root | 同上 | epoch 49,mis 0.599,act_mse 0.038 | 没有引用 | 【已被取代】【可删候选】 |
| `stage2_v5_nolang/stage2_{best,last}.pt` | 554 MB×2 / 08-17 / root | `train_v5_e1.sh` 的并行臂,用 `config_v5_nolang.yaml`(`no_language: true`) | best:ep20,mis 0.7150。last:ep49,0.7158,约等于随机水平 0.76 | `rollout/eval_h0.sh` M1_nolang → `eval_out/m1.json`(08-18,8 局:mis_sus 0.923 vs m0 0.606,SR 都是 0);906 §2 | best【消融对照】【论文引用】;last【可删候选】 |
| `stage2_v5_fair/stage2_{best,last}.pt` | 561 MB×2 / 08-18 / root | `train/train_v5_fair.sh`:Stage-1a → `stage1_ear_v5` → Stage-2(IAR 5 层 + supervise_intercept) | iar 2,749,762,n_layers_in=5。best:ep25,mis 0.582,act 0.0412。last:ep49,mis 0.641 | `intercept_eval_fair.log`、`baseline_eval.log`;闭环没用过 | 【消融对照】 |
| `stage2_v5_noear/stage2_{best,last}.pt` | 554 MB×2 / 08-18 07:02 / root | `config_v5_noear.yaml`(`ablate.ear: true`) | **best 和 last 都是 epoch 0**,mis 0.715,act_mse 0.293(完整模型约 0.04)。训练在 ep5 评估之前就中断了(推断) | `run_h1h2_pipeline.sh` 只对它做过 `ls`;eval_out 里**没有** noear.json | 【调试/冒烟】【可删候选】 |
| `stage2_v5_noiar/stage2_{best,last}.pt` | 554 MB×2 / 08-18 07:02 / root | `config_v5_noiar.yaml`(`ablate.iar: true`) | 同样都是 epoch 0,mis 0.656,act_mse 0.312 | 同上,没有 noiar.json | 【调试/冒烟】【可删候选】 |

### 4.B Stage-1 EAR 系列

v1–v5 的 EAR 结构完全相同:扁平 state_dict,5,071,107 参数,d_model 256,4 层,cond_proj 2560→256。

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `stage1_smoke/smoke_best.pt` | 3.4 MB / 07-28 / hque | `train/smoke_stage1.py`,合成数据,CPU | 853,635 参数 | 只有 smoke_stage1.py | 【调试/冒烟】【可删候选】 |
| `stage1_ear/stage1_ear_best.pt` | 19 MB / 07-28 / root | `train.stage1_ear --config train/config.yaml`(推断),mvp 336px | 没读到训练日志 | `config.yaml`、`config_ac`、`config_ablA`、`config_nolang`、`_leakcheck`;`stage2.py:262` 和 `eval_stage1.py` 的默认路径 | 【已被取代】(数据已不在盘上) |
| `stage1_ear_r3/stage1_ear_best.pt` | 19 MB / 08-07 / root | `train/sanity_r3.sh`,mvp_review_r3,10 集,3 epoch | err_6s 69.6 m,只用来验证管线能跑 | `config_r3_sanity.yaml` | 【调试/冒烟】【可删候选】 |
| `stage1_ear_full/stage1_ear_best.pt` | 19 MB / 08-11 / root | `train/train_full.sh`,mvp_full 61 集 | err_6s 最低 24.08 m(ep30) | `config_full*.yaml`、`config_full_v2*.yaml` | 【已被取代】 |
| `stage1_ear_v3/stage1_ear_best.pt` | 19 MB / 08-12 / root | `train/train_v3_e1.sh`,mvp_full_v3 | 没读到日志 | `config_v3*.yaml` | 【已被取代】 |
| `stage1_ear_v4/stage1_ear_best.pt` | 19 MB / 08-14 / root | `train/train_v4_e1.sh`,mvp_full_v4 | 没读到日志 | `config_v4*.yaml` | 【已被取代】 |
| `stage1_ear_v5/stage1_ear_best.pt` | 19 MB / 08-18 / root | `train_v5_fair.sh` 的第 1 步,数据是原始 90° v5 | 和 stage2_v5 里的 EAR 最大差 1.16,两者无关 | `config_v5*.yaml` 的 ear_ckpt。**闭环不加载它**:`policy.py:74` 只从 stage2 ckpt 里取模块。实际只有 `_fair` 用它预热 | 【消融对照】(重训陷阱见 §2.3) |

### 4.C 控制侧修取景:BC 居中 / RL E0(负结果)

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `stage2_bc_center.pt` | 554 MB / 09-04 22:42 / root | `train/bc_center_train.py`(blend=1.0),输入 `bc_center_cache.pt`(已不在盘上) | 与 stage2_v5 相比只有 DiT blocks 28–31 和输出层不同(74 个键,16,818,693 个可训练参数)。**epoch 和 metrics 是从基座复制来的**(20 / 0.579) | `eval_bc_center.sh` | 【负结果存档】 |
| `stage2_bc_center_b0.4.pt`、`_b0.7.pt` | 554 MB×2 / 09-04 / root | 同一脚本,dyaw = b·center + (1−b)·recorded | 结构同上 | `eval_paper_batch.sh:36`(neg_bc04 → `eval_out/paper_neg_bc04.json`,906 §6,20 局,SR 0) | b0.4【论文引用】【负结果存档】;b0.7【负结果存档】 |
| `stage2_rl_e0.pt` | 554 MB / 09-05 02:48 / root | `train/rl_e0_center.py`,SVG 可微多步 rollout,奖励 R_center,输入 `bc_center_cache2.pt`(已不在盘上) | 结构同上,元数据同样是复制来的 | `eval_paper_batch.sh:37`(neg_rle0,906 §6) | 【论文引用】【负结果存档】 |
| `bc_center.log` | 4.3 KB / 09-04 / root | bc_center_train.py | 缓存 6282 帧,train ep 9–65,val ep 0–8。20 epoch 后 action_flow 从 0.766 降到 0.335 | — | 【负结果存档】 |
| `bc_recache.log` | 3.3 KB / 09-04 / root | 重建缓存 → `bc_center_cache2.pt` | 6282 帧 | RL E0 | 【负结果存档】 |
| `bc_blend_train.log` | 2.4 KB / 09-04 / root | blend 0.4 和 0.7 两次训练 | 最终 flow:0.296 / 0.303 | — | 【负结果存档】 |
| `bc_center_eval_smoke.log` | 3.8 KB / 09-04 / hque | `eval_bc_center.sh`,4 局 × 900 步 | base 对 bc:track_s 33.8→2.2\*,centering 0.010→0.168\*,id_sw 7.75→1.0\*,lock_wrong +0.475\*,SR 都是 0(\* 表示显著) | 907 diary §9.1④ | 【负结果存档】 |
| `bc_blend_eval.log` | 3.8 KB / 09-05 / hque | 同上 | b04:mis_sus 0.677,track_s 10.95;b07:0.706 / 5.85;SR 都是 0 | — | 【负结果存档】 |
| `rl_e0_train.log` | 1.0 KB / 09-05 / root | rl_e0_center.py | R_center 在 10 epoch 内从 0.782 升到 0.982 | — | 【负结果存档】 |
| `rl_e0_eval.log` | 1.5 KB / 09-05 / hque | eval_bc_center.sh 的 e0 臂,4 局 | SR 0,mis_sus 0.603,track_s 5.15,centering 0.196 | `eval_out/bc_e0.json` | 【负结果存档】 |

### 4.D 训练日志(H0 证据链)

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `train_full.log` | 18 KB / 08-11 / root | `train/train_full.sh`:Qwen 缓存 → Stage-1 → Stage-2(`config_full`,target_id 权重 0.2) | val mis 从 0.410 变到 0.424,act_mse 0.036。输出目录 `stage2_full` 已不在盘上 | — | 【已被取代】 |
| `wolang_full.log` | 9.3 KB / 08-11 / root | `wolang_full.sh`,`config_full_nolang` | mis 从 0.366 变到 0.466。这次运行可能早于候选打乱的修复(推断) | — | 【已被取代】 |
| `train_v2.log` / `train_v2_nolang.log` | 1.9 KB×2 / 08-11 / root | `config_full_v2(_nolang)`,tid 权重 0.5 | ep49 mis 0.217 / **0.000**,是指标漏洞造成的假象(826 §4.2;修复后的真实值约 44.9%) | 826、907 §7.1 | 【负结果存档】 |
| `train_v2_ORDERBUG.log` | 1.9 KB / 08-11 / hque | 人工复制 | 与 `train_v2.log` 逐字节相同 | — | 【负结果存档】 |
| `train_v3.log` / `_nolang` | 2.0 KB×2 / 08-12 / root | `train_v3_e1.sh`,mvp_full_v3 | 最终 mis 0.693 / 0.724 | 907 §7.2 | 【负结果存档】 |
| `train_v3_stab.log` / `_nolang` | 2.1 KB×2 / 08-12 / root | `train_v3_stab.sh` | best_mis 0.511 / 0.653。看起来差 14pp,但接近几何下限 54%,是假象 | 907 §7.2 | 【负结果存档】 |
| `train_v4.log` / `_nolang` | 2.0 KB×2 / 08-14 / root | `train_v4_e1.sh`,mvp_full_v4 | best_mis 0.656 / 0.711,最终 0.718 / 0.756(约等于随机)。两臂帧数不同,原因没查到 | 907 §7.2 | 【负结果存档】 |
| `train_v5.log` / `_nolang` | 1.5 KB×2 / 08-17 / root | `train_v5_e1.sh`,只用 L24 fp16 缓存,**日志里没有 EAR warm-start 的行** | best_mis **0.579**(ep20) / **0.715**,差约 13.6pp | 906 §2(H0 的核心数字) | 【论文引用】 |

### 4.E 离线探针与闭环诊断日志

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `probe_v2_result.txt` | 491 B / 08-12 / hque | `train/benchmark_leak_probe.py`,mvp_full_v2 | 随机 68.95,最居中 83.80,最近 **31.30**,MLP 下限 47.00 | 817 §8 | 【负结果存档】 |
| `probe_v4_result.txt` | 491 B / 08-14 / hque | 同一脚本,v4(推断) | 随机 76.24,最居中 75.72,最近 82.79,MLP 下限 70.02,几何捷径已全部清除 | 817 §9、907 §7.2 | 【负结果存档】 |
| `recrop_probe.log` | 949 B / 08-14 / hque | `train/recrop_probe.py`,res 224 | color 23.4%,make 12.1%;0–20px 的框只有 20.8% / 9.1% | 907 §7.3 | 【负结果存档】 |
| `gate_v5.log` / `_recrop` / `_coarse` | 约 1.1 KB / 08-16 / hque | `train/fullframe_probe.py`,v5 512px,按集切分 | make 6.0 / 8.5 / 5.9%,color 14.6%,粗粒度 22.5%。门槛是 make ≥ 40%,全部 FAIL | 907 §7.3 | 【负结果存档】 |
| `gate_v5_indist.log` | 1.2 KB / 08-16 / hque | 同上,随机切分 | color 61.8%,make **55.0%**(25–40px 区间 64.1%),说明 27px 的特征里确实有身份信息 | 907 §7.3,语言必要性记忆 | 【负结果存档】 |
| `intercept_eval.log` | 941 B / 08-18 / hque | `train/intercept_eval.py`,stage2_v5 best | EAR FDE 在可见时 15.2/16.2/19.1/23.0 m(1/2/4/6 s),丢失时 58.5–77.2 m;persist 基线 0/2.7/8.2/13.7 | 推动了 fair 重训 | 【负结果存档】 |
| `intercept_eval_fair.log` | 946 B / 08-18 / hque | 同上,fair best | 可见 14.2–21.2 m,丢失 58.3–77.7 m,几乎没改善 | — | 【负结果存档】 |
| `baseline_eval.log` | 5.1 KB / 08-19 / hque | intercept_eval + `train/baseline_deadreckon.py`,fair ckpt | 丢失时长中位 2.5 s,p90 6.0 s。可见直行段 EAR 13.5–20.0 m 而 CV 0.4–5.9 m;长直丢失段 EAR 67.6–90.0 m 而 CV 13.7–23.2 m | 907:405("CV 碾压 EAR") | 【负结果存档】 |
| `ex_source_eval.log` | 14 KB / 08-19 / hque | `rollout/eval_ex_source.sh`,stage2_v5,20 局,Town05 91000–91019 | ear / cv_gated / zerovel 三种:SR 都是 0;mis_sus 0.723 / 0.805 / 0.732;lock_wrong 0.244 / 0.808 / 0.718 | 907 §8.8、906 §3 | 【论文引用】 |
| `expert_control.log` | 2.4 KB / 08-26 / hque | `run_rollout --expert-control`,GT+PID 取景,策略只负责 tid 打分 | 10 局(9 局计分):SR 0.111,mis_sus 0.592,track_s 88.8,说明取景完美时认车仍有约 59% 是错的 | `eval_out/expert_control.json` | 【负结果存档】 |

### 4.F Qwen LoRA grounding(10-02)

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `qwen_lora/r16/{best.pt,final.json,log.txt}`、`r16.stdout` | 10.5 MB 等 / 10-02 / root | `train/qwen_lora_grounding.py --rank 16`,在 Qwen L16–23 的 q/k/v/o 上加 LoRA,再接 grounding 头 | val 0.650,test 0.601;语言替换后 0.327 | `1002_stage0_qwen_lora.md` | 【负结果存档】 |
| `qwen_lora/r0/…`、`r0.stdout` | 5.3 MB 等 / 10-02 / root | `--rank 0`,只训练头 | val/test 0.558;语言替换后 0.395 | 同上 | 【负结果存档】 |
| `qwen_lora/smoke/` | 10.5 MB / 10-02 / root | 冒烟,12 帧 | — | — | 【调试/冒烟】【可删候选】 |
| `qwen_lora/paired_eval.{json,log}` | 731 KB / 10-02 / root | `train/qwen_lora_paired_eval.py`,按集做 bootstrap | test Δ +0.043 [−0.061, +0.141],不显著。判定为未达到有效判据 | 1002 笔记 | 【负结果存档】 |

### 4.G 早期杂项(07-27 → 08-07)

| 路径 | 大小/日期/属主 | 产生方式 | 内容 | 状态 |
|---|---|---|---|---|
| `proj_check_ep0.png` | 1.4 MB / 07-27 / hque | `acot_probe/visualize.py` | ep0 的 8 帧投影检查拼图 | 【调试/冒烟】 |
| `audit_bearing.log` | 968 B / 07-28 / hque | `train/audit_bearing.py` | 22.1% 的方位声明与实际不符,因此改用中性语言(729 §9.1) | 【已被取代】 |
| `ablA.log` | 7.0 KB / 07-29 / root | 方案 A,`config_ablA.yaml` | 修复前的指标,课程不稳定;`stage2_neutral` 已不在盘上 | 【已被取代】 |
| `bench_mmap.log` | 939 B / 07-29 / root | stage2 + mmap 缓存,测速 | 只跑到 ep5 | 【调试/冒烟】【可删候选】 |
| `mmap_convert.log` | 0 B / 07-29 / root | `train.mmap_cache`(推断) | 空文件 | 【可删候选】 |
| `sanity_r3.log` | 5.7 KB / 08-07 / root | `train/sanity_r3.sh` | 只验证 schema 能跑通 | 【调试/冒烟】 |
| `r3_train_verify.log` | 945 B / 08-07 / root | 同一配置,重跑 2 epoch | mis 0.630 | 【调试/冒烟】【可删候选】 |

### 4.H 本节的坑

1. **BC、RL 和 Qwen LoRA 用的都是 FOV 错位的数据**(09-04 生成的 70° 图像按 90° 投影)。缓存已经删了,引用"控制侧修取景证伪"时要加注。
2. **noear / noiar 只有 epoch 0 的权重**,不能当消融结果引用。
3. **v2 及更早的 mis_follow 不可引用**,包括 train_full、wolang、ablA、sanity_r3。这些都在候选顺序修复之前。
4. **日志里的时间是容器 UTC,文件 mtime 是宿主机时间(UTC−4)**,两者相差 4 小时。
5. **`eval_out/ex_cv_gated.json` 被覆盖过**,mtime 是 08-26。要引用 08-19 的数字,以日志为准。
6. **可删候选约 3.9 GB**,主要是 4 个 `stage2_*_last`,加上 noear 和 noiar。这些文件属主都是 root。

---

## 5. 身份线(8 月 → 9 月):tid → attrbind → DAgger → re-ID → 重锚 → TAH → TAHRelM

### 演进概述

1. **tid 单帧语言 grounding 头(08-19 ~ 08-21)。**
   - 起点:Stage-2 自带的 xattn tid 头,离线 val mis_follow 0.579(tier-c 0.701)。
   - 试过三类改法:换损失(ce / margin / hardneg)、加容量(d512 / h8)、换头(CLIP 式 attrbind、combined)。离线最好降到 0.44。
   - 但闭环里 attrbind 和 xattn 一样,都停在 mis 约 0.74(20 集配对,Δ −0.004,不显著)。
2. **attrbind 上的 DAgger(08-27)。** DAgger-val 从 0.695 降到 0.39–0.43,闭环反而变差:mis 从 0.581 升到 0.623 / 0.687。结论:离线 grounding 的改进不能迁移到闭环。
3. **cv_gated / track / road 对照(08-24 ~ 08-31)。** WHERE 做得再好,SR 也始终是 0。瓶颈在 WHICH(认哪辆车),不在 WHERE(目标在哪)。
4. **时序外观 re-ID(09-05 ~ 09-07)。**
   - 做法:crop-DINOv2,K=2 图库,置信门,时序 EMA 0.4。
   - l1 到 l5 的阶梯:正确帧占比从 0.251 升到 0.738。
   - 加上运动一致性(borrow#1,RMOT=0.3)后 SR 0.263。
   - **这些都是 GT 种子图库(gallery_seed=gt)下的数字。**
5. **去掉 GT 拐杖后(09-06)。**
   - 可部署的 committed 版正确率降到 0.363。
   - 语言重锚,无论单帧还是时序,都更差:0.225 / 0.291。
   - oracle 重锚 0.741,说明有提升空间,但语言救不了。
6. **路线 B:蒸馏成可学的 TAH(09-07 ~ 09-08)。**
   - TAH v1(72 万参数):过拟合,val 0.521。
   - TAHRel(4,610 参数,相对特征):val 0.909,但闭环只有 0.681。
   - TAHRel 加 23 集 DAgger:SR 0.158。
   - **TAHRelM 加 DAgger(4,740 参数)= `tah_relm_dag.pt`**:gt 种子下 SR 0.316、correct 0.746;committed 下 SR 0、correct 0.146(confident drift)。用 GT 检测漂移的上界是 0.721。
7. `tah_relm_dag.pt` 后来成为 10 月 PGIF 主模型(`ours_final`)的身份头权重,一直沿用到现在。

### 5.A 离线 grounding 头权重与训练日志(tid 系列)

**共同信息:**
- 都由 `train/grounding_train.py` 训练,数据是原始 v5(90°),train/val 为 19,115 / 4,004 样本,val 计分 n=3,695。
- ckpt 的键是 `{tid, mis_follow, by_tier, head?}`。
- 按 FOV 审计结果,不受 FOV 错位影响。
- **闭环入口只有一个**:`policy_server --tid-ckpt` 只接受 `tid_head=attrbind`。

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `grounding_probe.log` | 970 B / 08-19 / hque | `train/grounding_probe.py`,ckpt=stage2_v5 | 基线 mis_follow **0.579**(tier-a 0.338,tier-c 0.701) | 901、831、906 | 【论文引用】(H0/基线) |
| `gtrain_dbg.log` | 3.5 KB / 08-19 / hque | grounding_train | DataLoader shm 不足,崩溃 | — | 【调试/冒烟】【可删候选】 |
| `gtrain_smoke.log` | 487 B / 08-19 / hque | 同上,3 epoch | best 0.560,没有输出 ckpt | — | 【调试/冒烟】 |
| `guardrail_test.log` | 373 B / 08-20 / hque | 同上,1 epoch | 正对照:init 0.579 与已有记录一致,PASS | — | 【调试/冒烟】 |
| `grounding_sweep.log` | 5.3 KB / 08-20 / hque | 从零训练,lr=1e-3,30 epoch | ce 0.712 / margin 0.713 / hardneg 0.710,都差于基线(lr 太大) | 901 | 【负结果存档】 |
| `tid_ce.pt` / `tid_margin.pt` / `tid_hardneg.pt` | 各约 9.2 MB / 08-19~20 / root | 上面这轮 sweep 的 best | xattn 结构,d=256 | 无 | 【负结果存档】【可删候选】 |
| `grounding_sweep2.log` | 6.7 KB / 08-20 / hque | lr=2e-4,40 epoch | ce 0.505 / margin 0.558 / hardneg 0.498 | 826 汇报 | 【负结果存档】 |
| `tid2_ce.pt` / `tid2_margin.pt` / `tid2_hardneg.pt` | 各约 9.2 MB / 08-20 / root | 上面这轮的 best | xattn 结构,d=256 | 无 | 【负结果存档】【可删候选】 |
| `capacity_d512.log` → `tid_d512.pt` | 2.2 KB;21 MB / 08-21 / hque;root | `--tid-d 512 --tid-heads 8` | 0.4977,与 d256 相当,说明容量不是杠杆 | 826、901 | 【负结果存档】【可删候选】 |
| `binding_sweep.log` | 4.7 KB / 08-21 / hque | `--head attrbind` 对 `--head xattn` | attrbind **0.441**,xattn 0.510 | 901 | 【负结果存档】 |
| `attrbind_smoke.log` | 352 B / 08-21 / hque | 2 epoch 冒烟 | ep0 0.500 | — | 【调试/冒烟】 |
| `tid_xattn.pt` | 9.2 MB / 08-21 / root | binding_sweep 的 xattn 臂 | 0.5104 | 无(没有闭环入口) | 【负结果存档】【可删候选】 |
| `tid_attrbind.pt` | 3.7 MB / 08-21 / root | binding_sweep 的 attrbind 臂 | 0.4414(c 0.559,a 0.209) | `eval_attrbind.sh`、`dagger_train.py --warm`、`grounding_head_compare.py` 的默认值 | 【负结果存档】(闭环上不迁移) |
| `combined.log` → `tid_combined.pt` | 2.3 KB;12.9 MB / 08-21 | `--head combined` | 0.440,与 attrbind 持平 | 无 | 【负结果存档】【可删候选】 |

### 5.B grounding / WHERE 闭环对照(8 月,都用 stage2_v5,Town05,种子 91000+)

| 路径 | 大小/日期 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `track_smoke.log` | 1.9 KB / 08-24 | `eval_ex_source.sh`(推断),2 集 | track 臂冒烟,SR 0 | — | 【调试/冒烟】 |
| `track_compare.log` | 9.7 KB / 08-24 | 同上,20 集 | cv_gated / track 和 ear 配对比较:三者 mis 都在 0.72–0.74,SR 全是 0。说明闭环的差距不是轨迹协变量造成的 | 901 §2.2、903 §1 | 【论文引用】(WHICH-not-WHERE) |
| `attrbind_wire_smoke.log` | 2.8 KB / 08-24 | `eval_attrbind.sh` | 接线冒烟 | — | 【调试/冒烟】 |
| `attrbind_compare.log` | 8.8 KB / 08-24 | `eval_attrbind.sh`,20 集 | track_xattn(用 stage2 自带的 tid 头)对 track_attr(`tid_attrbind.pt`):mis_sus 0.745 / 0.742,SR 0 / 0;id_sw 从 11.2 降到 7.0(显著) | 901、831 | 【负结果存档】 |
| `unk_probe.log` | 909 B / 08-25 | 不明 | mis 0.96,用途没有记录 | — | 【调试/冒烟】【可删候选】 |
| `town03_test.log` | 1.1 KB / 08-26 | cv_gated,Town03 | **只跑了 1 集就崩溃** | — | 【调试/冒烟】 |
| `seen_seed_test.log` | 2.5 KB / 08-26 | cv_gated,训练种子 6030–6039 | mis 0.742,SR 0:训练时见过的场景也是 0.74 | 831、901 §2.2 | 【负结果存档】 |
| `ground_strat.log` | 3.3 KB / 08-26 | cv_gated 和 ear 配对,10 集 | mis 0.716,lock_wrong 0.66 对 0.20 | — | 【负结果存档】 |
| `ladder_road_0831_0818.log` | 6.2 KB / 08-31 | `train/baseline_deadreckon.py`,stage2_v5_fair,val 9 集 | WM M2 门:在路口帧上 road 比 cv_frame 的 FDE 低(6 s 时 Δ −5.2 m) | 901、831_wm-p2a-plan | 【负结果存档】(WM 线已关闭) |

### 5.C attrbind 的 DAgger(08-27)

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 状态 |
|---|---|---|---|---|
| `dagger_collect.log` | 3.1 KB / 08-27 / hque | `policy.py` 设置 `DAGGER_DIR` | 跑到 ep14 崩溃。**采集种子 91000–91013 和评测种子重叠** | 【负结果存档】 |
| `dagger_v5/dagger_ep0000–0011.npz` | 12 个,20 MB / 08-27 / root | 上面的采集 | `feats(ΣN,2560) fp16`、`n`、`true_idx`、`color`、`make`,共 1,100 帧 | 【负结果存档】【可删候选】 |
| `dagger_train.log` → `tid_attrbind_dagger.pt` | 1.1 KB;3.7 MB / 08-27 | `dagger_train.py`,从 `tid_attrbind.pt` warm start | DAgger-val 0.695 → **0.432**,offline-val 0.441 → 0.525(遗忘) | 【负结果存档】 |
| `dagger_train2.log` → `tid_attrbind_dagger2.pt` | 1.2 KB;3.7 MB / 08-27 | 同上,加 6,000 条离线回放 | DAgger-val **0.391**,offline-val 0.483 | 【负结果存档】 |
| `dagger_eval.log` | 4.8 KB / 08-27 | 闭环 10 集,种子 92000–92009 | offline 0.581 对 dagger 0.623,SR 都是 0 | 【负结果存档】 |
| `dagger2_eval.log` | 2.4 KB / 08-27 | 同上 | dagger2 mis **0.687**,更差 | 【负结果存档】 |

### 5.D 时序 re-ID、运动一致性和外部基线(9 月旧协议,默认 `--gallery-seed gt`)

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `reid_metric_cache.npz` | 102 MB / 09-04 / root | `train/reid_metric_train.py` 第 1 阶段 | DINOv2 crop 特征 `feats(74,647×384)`,55 集,423 个实例。**来自 09-04 的 70° 数据,按 90° 投影** | `reid_metric_train.py --cache` 的默认值 | 【负结果存档】【可删候选】 |
| `reid_metric_cache_proj.pt` | 1.2 MB / 09-04 / root | 同一脚本第 2 阶段,SupCon 投影头 | MLP 384→384→384,效果"边际" | 906 §10;没有运行时代码加载它 | 【负结果存档】【可删候选】 |
| `reid_temporal_eval.log` | 4.3 KB / 09-05 | `eval_reid_temporal.sh`,5 集 | tavg 从 0 到 0.7:mis 0.272 → 0.225,id_sw 13.4 → 6.2 | 906 §5 | 【消融对照】 |
| `reid_temporal_sr.log` | 5.0 KB / 09-05 | 同上 | SR 0.125 → 0,mis 0.386 → 0.252 | 906 §5 | 【消融对照】 |
| `reid_tavg_sweep.log` | 5.8 KB / 09-05 | 同上,tavg 0.4 / 0.5 | mis 0.313 / 0.319 | 906 §5 | 【消融对照】 |
| `reid_tavg04_confirm.log` | 8.2 KB / 09-05 | 同上,18 集 | base 对 0.4:mis 0.377 → 0.302,id_sw 15.7 → 10.2(显著)。tavg 0.4 是甜点 | 906 §5 | 【论文引用】 |
| `paper_batch.log` | 29 KB / 09-06 | `eval_paper_batch.sh`,7 臂 × 20 集 | 主阶梯 l1 到 l5 的 mis_sus:0.752 / 0.487 / 0.354 / 0.334 / 0.269;SR:0 / 0.053 / 0.105 / 0.105 / 0.105。neg_bc04 和 neg_rle0 的 SR 都是 0 | 906 §4/§6、907_baseline_compare | 【论文引用】(旧协议,现降为历史参考) |
| `assoc_eval.log` | 8.1 KB / 09-06 | `eval_ext_assoc.sh` | assoc_motion SR 0.053;deepsort 臂崩溃 | 906 §8 | 【消融对照】 |
| `dam4sam_eval.log` | 544 B / 09-06 | `eval_ext_dam4sam.sh` | CARLA 超时,失败 | — | 【已被取代】【可删候选】 |
| `ext_rerun.log` | 7.9 KB / 09-06 | `eval_ext_rerun.sh` | deepsort SR **0.211**;DAM4SAM SR 0.105,id_sw **0.79** | 906 §8 | 【论文引用】(旧协议) |
| `reidmotion_eval.log` | 5.2 KB / 09-06 | `eval_reidmotion.sh`,RMOT=0.5 | SR 0.211,correct 0.835 | 907_baseline_compare | 【消融对照】 |
| `reidmotion_sweep.log` | 9.7 KB / 09-07 | `eval_reidmotion_sweep.sh` | RMOT 0.3:SR **0.263**,correct 0.810(borrow#1);RMOT 0.7:SR 0.105 | 后续 TAH 对比的基线 | 【消融对照】 |
| `conf_consensus_eval.log` | 5.5 KB / 09-07 | `eval_conf_consensus.sh` | borrow#1 加一致性门,SR 0.053,过阻尼 | a1_post | 【负结果存档】 |

### 5.E 可部署化与重锚(09-06)

| 路径 | 大小/日期 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `reanchor_handoff.log` | 464 B / 09-06 | `handoff_reanchor.sh` 编排 | 记录时间线 | — | 【调试/冒烟】 |
| `reanchor_smoke.log`、`reanchor_smoke_srvlog.txt` | 2.5 KB、871 B / 09-06 | `eval_reanchor.sh`,2 集 | 冒烟 | — | 【调试/冒烟】【可删候选】 |
| `reanchor_full.log` | 14 KB / 09-06 | `eval_reanchor.sh`,3 臂 × 18 集 | GT 图库对 committed:correct **0.715 → 0.363**。committed 下 off / lang / oracle 分别为 **0.363 / 0.225 / 0.741** | 906 §8b | 【论文引用】(语言重锚证伪) |
| `langtavg_rescue.log`、`langtavg_smoke*` | 312 B、4.0 KB、859 B / 09-06 | `run_langtavg_rescue.sh` | 冒烟,n=2,不能引用 | — | 【调试/冒烟】 |
| `langtavg_full.log` | 6.6 KB / 09-06 | 同上,18 集 | 时序语言重锚 correct 0.291,仍低于 off 的 0.363 | 1001 附录 | 【负结果存档】 |

### 5.F TAH 线(路线 B,09-07 ~ 09-13)

**共同信息:**
- 训练脚本 `train/tah_train.py`,在 docker 里运行,所以属主是 root。
- 闭环用 `eval_tah.sh`,默认 `TAH_CKPT=runs/tah_rel.pt`,`GSEED=gt`。
- 运行时只接受 `TAHRel` 和 `TAHRelM` 两种结构。

| 路径 | 大小/日期/属主 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `tah_cache.log` → `tah_cache.pt` | 4.1 KB;217 MB / 09-07 / root | `train/tah_cache.py`,stride 2 | 64 集 `mvp_full_v5`(**70° 图像按 90° 投影**),42,637 帧,`{feat(N,384) fp16, pos(N,3), aidx, tidx}` | `tah_train.py` 等的默认值;`ours_v2off`;legacy artefact `tah_cache_v5` | 【消融对照】(重训原料,**不能删**) |
| `tah_train.log` → `tah.pt` | 3.6 KB;2.9 MB / 09-07 / root | 默认参数,40 epoch | TAH v1(GRU,724,737 参数):train 0.759,val **0.521**,过拟合 | 只在 907 日记里提到;运行时加载不了这个结构 | 【已被取代】【可删候选】 |
| `tah_rel.log` → `tah_rel.pt` | 5.3 KB;21 KB / 09-07 / root | `--rel --aug`,60 epoch | TAHRel,4,610 参数,val **0.909** | `policy.py`、`policy_server.py` 的 `--tah-ckpt` **默认值**;`collect_tahdag.sh` 用它采集 DAgger | 【已被取代】(旧脚本的默认值仍指向它) |
| `tah_eval.log` | 5.4 KB / 09-07 | `eval_tah.sh`,`tah_rel.pt`,gt 种子 | SR 0.053,correct 0.681 | a1_post | 【消融对照】 |
| `tahdag_collect.log` | 4.7 KB / 09-07 | `collect_tahdag.sh`,种子 92000–92023 | 24 集,SR 0.042,生成 23 个文件 | `b1_finish.sh` | 【消融对照】 |
| `tahdag/tahdag_ep0000–0022.pt` | 23 个,12 MB / 09-07 / root | 上面的采集 | 2,156 帧,闭环 90°,无错位 | 合并进 `tah_cache_dag.pt`;重新编码成 `tah_v90/dag_old.pt` | 【消融对照】(原始来源,建议保留) |
| `b1_finish.log` → `tah_cache_dag.pt`、`tah_rel_dag.pt` | 1.3 KB;228 MB;21 KB / 09-07 / root | `b1_finish.sh`:合并,然后 `--rel --aug --epochs 60 --wd 3e-4` | 合并后 87 集(64 + 23)、44,793 帧;TAHRel+DAgger val 0.931,闭环 SR **0.158** | `tah_cache_dag.pt` 是 `tah_relm_dag`、`ours_rt_s0`、P0-2 和 v90 的输入 | `tah_cache_dag.pt` **不能删**;`tah_rel_dag.pt`【已被取代】 |
| `a1_finish.log` → **`tah_relm_dag.pt`** | 2.3 KB;22 KB / 09-08 / root | `a1_finish.sh`:`tah_train.py --cache runs/tah_cache_dag.pt --out runs/tah_relm_dag.pt --motion --aug --epochs 60 --wd 3e-4`(没有 `--seed`) | TAHRelM,4,740 参数,val best **0.990**;闭环 gt 种子 SR **0.316**、correct 0.746 | **`ours_final.yaml` 以及 12 个继承它且不覆盖 ckpt 的方法**;`smoke_iter.py`、`deploy_eval_chain.sh`、`artefacts.yaml` | **【部署中】【论文引用】** |
| `a1_post.log` | 4.0 KB / 09-08 | `a1_post.sh`:稳定性 / 场景切分 / 重新出图 | steady 0.746,recover 0.590 | 907 §11 | 【论文引用】(旧协议,gt 种子) |
| `eval_tahrelm_committed.log` | 5.4 KB / 09-08 | `eval_tah.sh`,`GSEED=committed` | **部署化后崩溃**:SR 0,correct **0.146** | 907 §11、1001 | 【论文引用】(confident drift 的证据) |
| `eval_tahrelm_deploy.log` | 319 B / 09-11 | `deploy_eval_chain.sh` 第一次尝试 | docker 镜像被 prune 掉,失败 | — | 【调试/冒烟】【可删候选】 |
| `deploy_eval.log` | 3.9 KB / 09-11 | `deploy_eval_chain.sh`:committed + 语言冷启动 + `reanchor=lang` | correct 0.263,SR 0。**只有尾部,没有逐集记录** | 1001 主表 | 【论文引用】 |
| `eval_oracledrift.log` | 5.5 KB / 09-13 | `eval_tah.sh`,committed + `reanchor=oracle`(推断) | GT 漂移检测上界:correct **0.721** | 1001 C2 | 【论文引用】(上界) |
| `eval_anchor.log` | 1.2 KB / 09-13 | `eval_tah.sh`,`reanchor=anchor` | CARLA 超时,失败;后续结果在 `runs/asd` | — | 【已被取代】【可删候选】 |

### 5.G 本节的坑

1. **9 月的好数字几乎都是 GT 种子下的**,引用时必须标 "gt-seeded"(907 §11 有铁律)。可部署版本:re-ID 0.363,TAH 0.146 / 0.263。
2. **`seeds.yaml` 里 `train_dagger_v1` 的说明写错了。** 它说是 "DAgger episodes of `tah_relm_dag.pt`",实际采集用的是 `tah_rel.pt`(见 `collect_tahdag.sh:15`)。
3. **`artefacts.yaml` 把 `tah_relm_dag` 的配方记成 "unknown",其实可以还原**:`a1_finish.sh` 和 `a1_finish.log` 里有完整命令。命令里没有 `--rel`,是靠 `--motion` 选出 relm 结构。因为没指定 `--seed`,无法精确复现。
4. **旧入口的 `--tah-ckpt` 默认值仍然是 `runs/tah_rel.pt`**,不是部署用的 `tah_relm_dag.pt`(`policy.py:110`、`policy_server.py:161`)。
5. **eval_out 的 JSON 会被同名覆盖**:`ex_track.json`、`ex_cv_gated.json`、`reid_tavg*.json` 只保留最后一次运行。
6. **DAgger 采集种子和评测种子重叠**:attrbind 线采集用 91000–91013,与评测重叠;TAH 线采集用 92000–92023,正好覆盖了 attrbind 那次评测用的种子。
7. **"闭环全部塌到 0.74"的覆盖面比字面窄。** 真正有闭环日志的只有 xattn 和 attrbind 两个头,其余 tid 变体只有离线日志。
8. **`tid_*` 和 `tid2_*` 只能靠 mtime 加 mis 值区分**:`tid_` 前缀是 lr=1e-3 那轮,`tid2_` 是 lr=2e-4 那轮。

---

## 6. 10 月数据与闭环类目录

| 目录 | 大小 | 文件数 | 日期 | 属主 | 状态 |
|---|---|---|---|---|---|
| `abl_allactor/` | 303M | 394 | 10-03~04 | hque | 【负结果存档】【消融对照】 |
| `acot_logs/` | 2.8M | 95 | 10-04~08 | hque | 【论文引用】(1007 §4b 的计算来源) |
| `arch_check/` | 4.5M | 4 | 10-12 | hque | 【论文引用】(核实架构图) |
| `artefacts/` | 970M | 139 | 10-04~14 | 混合 | 【部署中/现行数据】(acot 产物库) |
| `asd/` | 491M | 342 | 10-01~04 | hque/root 各半 | 【已被取代】;其中 4 个 .sh 是 L0 黄金参照 |
| `data_audit/` | 18M | 42 | 10-03 | hque | 【论文引用】(FOV 错位证据) |
| `data_v6/` | 15M | 282 | 10-03~09 | hque | 【已被取代】 |
| `datagen/` | 12K | 1 | 10-04 | hque | 【调试/冒烟】(`GEN_ROOT`,保留) |
| `datagen_moving/` | 1.2G | 247 | 10-09~12 | hque | 现行数据的生成记录 |
| `mp4_preview/` | 2.2G | 193 | 10-04 | root | 【调试/冒烟】【可删候选】 |
| `opening_dump/` | 5.5G | 1708 | 10-03 | 大部分 root | npz_tr/npz_hi 是现行数据,其余【已被取代】 |
| `tah_1006/` | 848M | 23 | 10-06~07 | 混合 | 【消融对照】【负结果存档】 |
| `tah_v90/` | 513M | 161 | 10-03~04 | 大部分 root | 【负结果存档】,部分文件被现行配置引用 |

### 6.1 `abl_allactor/`:"只用目标 vs 全部车辆"离线消融(1003 §7–§10)

```
build_cache_v5.py → cache_s{0,1,2}.pt(181M,64 集 v5,按 70° 正确投影,全车候选 + 世界坐标)
ablate.py   → runs/  (48 json)  + summary.txt          §7  target/all/balanced/all_cut × 4 折 × 3 种子
ablate2.py  → runs2/ (84 json)  + summary2_P0*.txt      §8  TAH v1(abs)/ TAH-X(tahx.py)
ablate3.py  → runs3/ (80 json) + runs3/w/*.pt(106M 权重) + summary3.txt   §9  事件记录 / relm→rn→rnw→x 阶梯
eval_post.py→ runs_post/(48 json) + summary_post*.txt/json, noise_decomp_L2.json, campose_v5.pt   §10 e/f
smoke3/  tahx.py  tahrelmn.py  summarize*.py  mon*.sh
```
- **数据来源**:`mvp_full_v5`(09-04 生成的 70° 数据),缓存按 70° 正确投影。
- **用法**:见各脚本的 docstring。例如 `python3 runs/abl_allactor/ablate3.py --plan all --workers 64`。
- **下游**:`1003_event-centric-dataset-review.md` §7–§10;画图脚本 `acot_note/figs/fig_tahx_structure.py`、`fig_tahrelm_vs_tahx.py`、`fig_post_ef.py`(会直接读 `runs_post/`)。
- **§7 结论:全车训练没有净收益。** su:acc 在 target / all / all_cut 下是 0.627 / 0.622 / 0.612。
- **§9 结论:阶梯上的增益来自 null + 世界系运动 Δt。** su:acc 依次为 relm 0.639 → rn 0.565 → rnw 0.899 → x/target 0.901。4.9k 参数的 rnw 和 56 万参数的 TAH-X 之间无显著差。x/all − x/target 的 reentry@1s 为 −0.321,说明全车训练有害。主要失败模式是过度弃权(null 率 0.75)。
- **§10 结论**:
  - 在验证集上预先选弃权规则,没有一条胜过 argmax。β=0.5 只是事后观察到的。
  - 世界坐标在 L2 噪声下,rnw su_acc 从 0.899 跌到 0.148,主要由深度噪声造成。

### 6.2 `acot_logs/`:acot 框架运行日志,以及 T0 / 1007 的判定 JSON

```
aa_acot.log / aa_legacy.{sh,log,progress}     1004 A/A 噪声(aa_legacy.sh 就是改名的 chains_attr.sh)
p2_parity_*.log, g2_parity_*.log               回放逐位一致性
import_report.json                             `acot import` 导入旧臂的报告
l3_legacy{.jsonl,.trk.jsonl}, smoke_*.log, ab_where_assign.log, qwen_bench.py
t0/  chain_{A,A2,A3,A4,B,C,C2,O1,O2a,O2b}.sh/.out/.done   T0 与 1008 调参的链式编排
     t0_*.log, opt{1,2a,2b}_screen_devB3.log
     freeze_r.json  g13_equivalence.json  dinov3_stage{1,2}.json  g8_/g9_/g9b_*.json  det_replay*.{py,json}
     provisional_baselines_p3.json  summary_1007.json  sec4b_update_1008.json  tuning_results_1008.json  stack_bench_a40.json
```
**关键数值**:
- `sec4b_update_1008.json`(@3,n=39)pc_correct:Ours(τ 0.05)0.525、DAM4SAM 0.720、re-ID 0.636、xattn 0.634、OA-VAT 0.622、DeepSORT 式 0.569、OC-SORT 式 0.392、记忆写承诺 0.580。
- G13:−0.154 [−0.274, −0.036],不等价。
- DINOv3 第一阶段 +0.079(通过),第二阶段 −0.089(不通过)。
- A40 实测:DiT p50 282 ms,VLM p50 100 ms。

**下游**:`1007_closed_loop_summary.md` §4b–4d、`1006_t0_log.md`、`1011`、`1012`。

### 6.3 `arch_check/`:架构图 v3 → v4 的 codex 核实

- `codex_arch_r1*.md`:**FIGURE DOES NOT MATCH CODE**。
- `codex_arch_v4*.md`:**FIGURE MATCHES CODE**。

两个文件都是完整转录(1.8M / 2.8M),结论只在末尾的 VERDICT 行。下游:`acot_note/figs/README_architecture.md`、`fig_architecture_v4.py`。

### 6.4 `artefacts/`:acot 内容寻址产物库(`<kind>/<name>@<id>/manifest.json`)

产物分两类:
- **legacy_import**:目录里只有 manifest,真实文件在 runs/ 的其他位置,用 sha256 钉住。由 `python -m acot artefacts import` 生成,源头是 `acot/legacy/artefacts.yaml`。
- **acot 原生**:文件就在目录里。由 `python -m acot prep|train|data split` 生成。

```
dataset/      mvp_full_v5@00c5(legacy,70°)  mvp_full_v5_fov90@47c5(legacy)  mvp_v5_fov90_fast@fc57(legacy,10-12 登记)
split/        split_v5_naive / split_v5_fov90_naive / split_v5_fov90_town10hd / split_v5_fov90_fast_naive@561e
tah_seq/      tah_cache_v5@bc5c(legacy → runs/tah_cache.pt)  tah_cache_v5_fov90@3f0b(legacy → runs/tah_v90/off90.pt)
              tah_seq_v5_fov90_fast@cfa1(469M,原生,0.6 s tick,frame_mask train)
              tah_seq_v5_fov90_t6@417a(456M,原生,旧 90° 数据 + 新配方,对照)
              tah_seq_v5_fov90@ea7b(7.2M,没有 manifest,中断的 prep → 可删)
dagger/       tah_dagger_v1@d419(legacy → runs/tah_cache_dag.pt)  _v1_reencoded@d3c2(→ tah_v90/dag_old.pt)  _v2@3c9a(→ tah_v90/dag_v2.pt)
              tah_dagger_v3@69ab(26M,原生,117 集)  tah_dagger_v3_val@00cd(11M,原生,40 集)
tah_ckpt/     tah_relm_dag@0899(legacy → runs/tah_relm_dag.pt = 论文 Ours)
              tah_relm_v90_s{0,1,2}[_last](legacy)  tah_relm_1006_*(legacy → runs/tah_1006/)
              tah_relm_v90_s0_repro[_last](原生,复现结果和原 ckpt 一致,val 0.9907)
              tah_relm_fast_a_s{0,1,2}[_last](原生,10-12,val 0.857–0.858)
              tah_relm_fast_b_s{0,1,2}[_last](原生,10-14,val 0.792–0.802)
              tah_relm_t6_ctl_s{0,1,2}[_last](原生,对照,没跑闭环)
stage2_ckpt/  stage2_v5@b387(legacy → runs/stage2_v5/stage2_best.pt)
opening_cls/  opening_attr_lin@f9a3   opening_ft/ opening_attr_ft@b2fb(legacy → eval_out/opening_attr_{cls,ft}.pt)
opening_snapshots/  _train@8038(→ opening_dump/npz_tr)  _hi_eval@2aa9(→ npz_hi)
```
注意:论文的 Ours(`ours_final`)直接用路径 `runs/tah_relm_dag.pt` 引用权重,不经过 artefact 库。

### 6.5 `asd/`:10-01~04 旧入口时期的 PGIF / ASD 闭环迭代

```
根目录       chains.sh round3.sh launch_*.sh  asd-*.log  {tah,reid}_{observe,tempered,bayes,resid}_91000.jsonl   10-01 PGIF 变体
old_logs/ partial1–4/ gt_aborted/                                                                10-01 早期或中断的运行
deploy_all/  chains.sh rerun_dam4sam.sh build_dam4sam.sh  *.log  tahrelm_pgif.jsonl               10-02 可部署 v1
deploy_v2/   chains_v2{,t,h}.sh resume_v2.sh  smoke_*                                            10-03 可部署 v2(跟踪器 ID + 遮挡候选)
sticky_check/ run.sh compare.py  sticky_{a,b}.log pgif_{a,b}.{log,jsonl}                         10-03 Sticky-ID 零假设
iter/ (486M) launch_wd.sh(L0 黄金参照)  <arm>_p<k>.jsonl(tick 日志)  <arm>_p<k>.trk.jsonl(479M)  failed_*/ hung_1120/ lost_reboot/
```
- **数据去向**:每集结果在 `eval_out/paper_it_*.json`、`dev_sc_*.json`。tick 日志由 `acot import`(`acot/legacy/arm_map.yaml`)、`iter_mech.py`、`pgif_track_metrics.py`、`tick_align.py` 读取。`.trk.jsonl` 由 `trk_tune.py`、`trk_quality.py` 读取。
- **结论**:
  - Sticky 0.579 对 PGIF 0.576,两者没有差别。
  - v1 的 `zoa_pgif_trk` 得 0.646,但 A/A 三次复跑平均约 0.578(见 `1004_aa_noise.md`)。

### 6.6 `data_audit/`:数据构成审计与 FOV 错位证据(10-03)

- `h5_stats.py` + `summarize.py` → `h5_rows.json`
- `event_census.py` → `event_census_v5.json`
- `fov_check_5ep.py` → `fov_check_5ep/{…, summary.json}`
- `fov_check_closedloop.py`、`fov_check{,3}.png`、`codex_regen_prompt.txt`

**关键结论**:磁盘上 `mvp_full_v5` 的图像和标注都是 70°。按 70° 投影误差为 0 px;按 90° 投影,目标偏 27–40 px,干扰车偏 41–70 px。`acot/legacy/artefacts.yaml` 的 camera_override 以此为证据。

### 6.7 `data_v6/`:v6.1 数据(controlled_day_caronly)的编排与冒烟记录

内容包括:
- `orchestrate.py`
- `*.orig` 备份
- `probe_*`
- 冒烟截图
- `smoke_snapshots/uavtrack_v6_smoke*/`,每次冒烟一份,含 plan、provenance、code_snapshot.patch、orchestrate.log

**h5 文件已经不存在。** v6 在 10-03 暂停,10-09 被 64 集方案取代。注意 `1003_data-regen-v6-spec.md:59` 写的 `runs/data_v6/qc_v6.py` 路径不对,实际位置是 `carla_uav_tracking/scripts/qc_v6.py`。

### 6.8 `datagen/`

只有 `dryrun_2seeds/plan.json`。`acot/data/gen.py:18` 的 `GEN_ROOT` 指向这个目录,**需要保留**。

### 6.9 `datagen_moving/`:64 集 90° 运动目标数据(mvp_v5_fov90_fast)的编排与记录(10-09~11)

- **数据本身**:`/nvidia/hque/data/carla_data/mvp_v5_fov90_fast/`(44G,含 `reused/` 和 `shard_a..d/`)。
- **清单**:同目录下的 `manifest.json`,10-11 22:52 生成。QC 版本 v3,条件为 frames≥1499、无 ≥40 s 的静止、moving≥0.35、相机在楼内 ≤5%。逐集记录了种子、城镇、场景、moving、in_building、train_keep。
```
full_fast.py             编排器(run2 修复了跨集遗留的背景车;QC v3);full_{a..d}.log / full_orchestrator.log / full_state.json
full_*_run1*.log/json    run1(被污染:23/64 集里遗留背景车成了死障碍)
qc.py                    QC v1/v2/v3 判定;building_check.py(+_rows.json)检查相机穿楼
smoke{,2,3,4}*.sh/log    冒烟 1–4;qc_archive/
kin_patch/               运动学和防卡死补丁:*.before_kin.py 备份、kin_r{2,3}.diff、full_fast_*.diff、train_frame_mask.diff、test_unjam.py、
                         codex_{kin,full,bldg}_r*.md(都是 NO MAJOR)、README.md
dashboard.json/.log      看板配置(datagen_dashboard.py @ 127.0.0.1:8767)
dash_videos/{full_fast(670M), full_qcfail(120M), run1_contaminated(417M)}   看板自动渲染的 mp4
```
- **训练时必须打开 `data.frame_mask: train`**。它调用 `carla_uav_tracking/recording/building_mask.py:train_keep_mask`,屏蔽挪车和楼内的帧(各前后 ±1 s,约占 1.35%)。
- **acot 侧对应的配置和产物**:`acot/configs/prep/tah_seq_v5_fov90_fast.yaml`(`frame_mask: train`、`tick_stride: 6`)、artefact `dataset/mvp_v5_fov90_fast@fc57e85e2157`、`split/split_v5_fov90_fast_naive@561e3cc7e0c2`。
- **城镇分布**:Town01 19 集、Town02 15 集、Town04/05/10HD 各 10 集。

### 6.10 `mp4_preview/`:训练数据的人工检查视频

子目录:`mvp_full_v5/`(64 个)、`mvp_full_v5_annotated/`(64 个)、`mvp_full_v5_boxes/`(1 个)、`mvp_full_v5_fov90_annotated/`(64 个)。由 `carla_uav_tracking/scripts/h5_to_mp4.py` 生成,属主全是 root。目前没有任何东西引用它们,可以重新生成(见 §9)。

### 6.11 `opening_dump/`:开局身份模块(1004 开局修复)的快照与启动脚本

```
npz_tr/ (3.7G, 810)  训练快照:train_opening 种子 = Town05 96000–96279,tick 0/6/12,rgb 512² + rgb_hi 1536² + 候选/深度/is_target/gt_idx/lang_logits
                     标签在 eval_out/labels_tr/
npz_hi/ (1.6G, 351)  高清评测快照:dev2 93000–93099 + dev 91000–91019
npz/    (166M, 349)  第一版低清快照(已被 npz_hi 取代,但 opening_score.py 默认读这里)
val/    (130M, 78)   留出的开局验证 tick 日志
logs/  failed_h/  hostmap*.txt(容器 hash → dump-k 的映射)  STOP
脚本   chains_attr.sh run_openval.sh f_v4.sh f_oracle_absent.sh(L0 黄金参照)
       launch_*.sh chains_*.sh start_base.sh open_ticker*.sh viz_crops.py ocl_rate.py oracle_cmp.py
```
`opening_cls.py` 和 `opening_ft.py` 用 npz_tr 训练出 `eval_out/opening_attr_{cls,ft}.pt`,也就是现行协议的共享前端。留出集上的开局准确率是 91.5% [86, 95]。

### 6.12 `tah_1006/`:T0 阶段的 G13 等价性检查与 DINOv3 主干试验

```
tah_cache_dinov3.pt (440M)       64 集离线 DINOv3 ViT-B/16 特征(d=768),由 acot/prep/tah_cache_backbone.py 生成
tah_cache_dinov3_dag.pt (449M)   上一个文件 + 24 集 DINOv3 DAgger(t0_dinov3_dagger2),由 acot/prep/tah_dagger_merge.py 生成
tah_relm_s{0,1,2}[_last].pt      G13:**DINOv2** 按现配方重训(cache = runs/tah_cache_dag.pt)→ ours_rt_s0
tah_relm_v2off_s0 / v3off_s0     只用 64 集离线数据的 DINOv2 / DINOv3 对比
tah_relm_v3full_s0               DINOv3 完整版(离线 + DAgger)
train_*.log  cache_dinov3.log  chain_backbone.sh
```
**容易误解的两点**:
- `tah_relm_s*.pt` 其实是 DINOv2 版。
- `ours_dinov3.yaml` 不用这里的权重。

**其他**:`train_s1.log` 只记到 ep8/40,但 ckpt 文件存在,日志可能被截断了(推断)。

### 6.13 `tah_v90/`:10-03~04 修正 FOV 后重训 TAHRelM

```
gen_fov90{,b}.sh  b_fill.py  b_manifest.json(64 个种子,49→108 替补)  seed_town.json  worker_lists.json   生成 90° 复刻数据 mvp_full_v5_fov90
off90_shard{0..3}.pt(229M) → off90.pt(240M)   离线 90° 缓存
collect_dagger.sh(L0 黄金参照) → dag_v2/(101 个文件) → dag_v2.pt
dag_old.pt                       旧 DAgger 重新编码
pipeline{,_v2,_v3}.sh  tah_relm_v90_s{0,1,2}[_last].pt  offline_eval.py offline_selfrun.py  logs/ ticker*.sh STOP
```
- **结论**:dev 集(n=19)上旧权重 0.646,新 best 0.573,新 last 0.543。修正 FOV 后重训**没有提升**,所以继续用 `runs/tah_relm_dag.pt`。
- **仍被现行配置引用的文件**:`dag_old.pt`、`dag_v2.pt`、`b_manifest.json`、`collect_dagger.sh`。P0-2 训练用了 `dag_old.pt`,fast_a 用 `dag_v2.pt` 做验证集。
- **数字不一致**:`1004_tahrelm_retrain_diag.md:70` 说有 105 集,实际只有 101 集。

---

## 7. 10 月中下旬:身份模块迭代与环境依赖(10-12 → 10-19)

**时间线**:
- 10-12、10-14:P0-2 重训,FAIL。
- 10-15:DAM4SAM-S。
- 10-16:身份诊断;DeepSORT、OC-SORT 原版复现。
- 10-18:再捕获统计。
- 10-19:第 0 步上限检查,STOP。

**现在卡在 `1019_metric_protocol_discussion.md`,等用户拍板。**

另外,P0-2 训出来的权重**不在 `p02_retrain/` 里**,在 `runs/artefacts/tah_ckpt/tah_relm_fast_{a,b}_s{0,1,2}[_last]@…` 和 `tah_relm_t6_ctl_s*`;DAgger v3 在 `runs/artefacts/dagger/`。

### 7.1 `p02_retrain/`:P0-2 用新数据重训 TAHRelM

`p02_retrain/` 共 11 MB,对应 `1012_p02_identity_retrain.md`。`probe/*.pt` 和 `offline_eval_b.*` 的属主是 root。状态:【负结果存档】【论文引用】。

| 路径 | 日期 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `decision_a.json` / `decision_a_cal.json` | 10-14 03:01–03:05 | `python -m acot.analysis.p02_decision --stage a` / `a_cal` | stage A(conf_tau 0.05):0.585 对 0.525,**Δ +0.060 [−0.026, +0.149],p=0.20,FAIL**。a_cal(0.63):+0.021,FAIL | 1012 §5,1007 §4b | 【论文引用】 |
| `decision_b.json`/`.log`、`decision_b_vs_a.json`/`.log` | 10-14 07:26 | `p02_decision --stage b` / `b_vs_a` | stage B:**Δ +0.016 [−0.047, +0.080],p=0.63,FAIL**。B − A = −0.044,不显著 | 1012 §6.8 | 【论文引用】 |
| `selection_tah_relm_fast_{a,b}.json`、`select_fast_b.log` | 10-12 / 10-14 | `select_and_pin.py` | 规则:验证集准确率最高者,平局取最小种子。fast_a 选中 s1(ep17,0.8578),fast_b 选中 s1(ep11,0.8019)。脚本会**改写** `ours_fast_{a,b}.yaml` | `ours_fast_{a,b}.yaml` | 【论文引用】 |
| `train_{fast_a,fast_b,t6_ctl}_s{0,1,2}.log` | 10-12 / 10-14 | `acot train`(配方 `recipes/tah_relm_*.yaml`) | 每个文件两行。t6_ctl 的 s0/s1/s2 = 0.8535 / 0.8545 / 0.8640,**没有跑闭环** | 1012 §5 | 存档 |
| `prep_cache.log` / `prep_cache_ctl.log` | 10-12 | `acot prep` | 登记 `tah_seq_v5_fov90_fast@cfa1`、`tah_seq_v5_fov90_t6@417a` | 配方输入 | 存档 |
| `prep_dagger_v3.log` / `_val.log` | 10-14 | `chain_b_collect.sh` | **0 B 空文件**,但对应的 artefact 存在 | — | 【可删候选】 |
| `eval_p02_fast_a_devB3.log` | 10-12 | `acot eval p02_fast_a_devB3.yaml` | 4 组 × 40 个种子全部完成 | → eval_out | 存档 |
| `eval_p02_fast_b_devB3.log`、`chain_b_eval.{sh,out}` | 10-14 | 选种 → 闭环 → 两轮判定 | 2 组 × 40 个种子 | → eval_out | 存档 |
| `eval_p02b_dagger_{smoke,train,val}.log`、`chain_b_collect.{sh,out}` | 10-14 | `acot eval p02b_dagger_*`,再 `acot prep tah_dagger_v3*` | train 118 局完成、2 局跳过;val 40/40 | → artefacts/dagger | 存档 |
| `offline_eval.py`、`offline_eval_b.{json,log,log.rc}` | 10-14 / root | 在 docker 镜像 `acot-uav-train:cu118-acot` 里跑 | teacher-forced 准确率。dag_v3_val 上:论文权重 0.744 / A 0.787 / **B 0.802**;dag_v2 上:0.813 / 0.858 / 0.865 | 1012 §6.7 | 存档 |
| `preassess.py` / `preassess.json` | 10-12 | `PYTHONPATH=. python runs/p02_retrain/preassess.py` | 闭环前用 13 份权重做离线预估。结论:**离线指标预测不了闭环结果** | 1012 §3.2 | 存档 |
| `post_hoc.py` / `post_hoc_b.py` | — | 只打印,不写文件 | 事后分析 | 1012 §5、§6.8 | 存档 |
| `sec4b_rows.py` / `sec4b_p02.json` | 10-14 | 同上 | 1007 §4b 表中 4 行及其配对差。**每行的 `"n": 39` 是循环覆盖留下的值,承诺正确指标实际的配对 n 是 40** | 1007 §4b | 【论文引用】 |
| `p02_fast_b_devB3.yaml.template` | — | 手写 | chain 脚本的模板,正式 yaml 已经存在 | — | 存档 |
| `probe/`(config、split、probe*.pt) | 10-12 / root | 改代码后的小规模试跑(推断) | `probe_ck.pt`:relm,26/101 集,val 0.809 | 无人引用 | 【调试/冒烟】【可删候选】 |
| `patch/*.before`、`*.before_b` | — | 手动备份 | 改代码之前的原文件:stage A 改 tah_cache/train/seq/relm,stage B 改 merge/p02_decision/select_and_pin/seeds/offline_eval/sec4b_rows,另有看板相关文件 | 1012 §2、§6.5 | 存档(用于回滚) |
| `patch/p02_*.diff` | — | 送 codex 审查的 diff | stage A、R2、stage B | — | 存档 |
| `patch/codex_p02{,b}_r{1,2}{,_prompt}.md` | — | codex 只读审查 | 两组都是 R1 MAJOR,R2 **NO MAJOR** | 1012 §3.1、§6.6 | 存档(文件较大,可压缩) |

### 7.2 `id_iter/`:身份迭代诊断(10-16 → 10-19)

全部是对已有闭环日志(`eval_out/exp/*/ticks.jsonl`)或 dagger v3 做的**离线分析,没有新跑闭环**。目录权限 700。

| 文件 | 日期 | 产生方式 | 内容与关键数值 | 被谁引用 | 状态 |
|---|---|---|---|---|---|
| `diag_events.py` → `diag_events.json` | 10-16 | `PYTHONPATH=. python runs/id_iter/diag_events.py` | tick 级分解为 correct / distract / stale / none,外加劫持和换 key 事件,共 8 组 | 1016 §1 | 【论文引用】 |
| `diag_switch.py` → `diag_switch.json` | 10-16 | 同上 | re-ID 可见时 0.739,**首次丢失后 0.557**(Ours 是 0.21) | 1016 §1.2 | 【论文引用】 |
| `loss_kind.py` | 10-16 | 只打印 | 65–80% 的丢失是**跟踪器 ID 互换**造成的 | 1016 §1.2b | 存档 |
| `replay_screen.py` → `replay_screen.json` | 10-16 | `PYTHONPATH=.:carla_uav_tracking python …` | 在 dagger v3 上离线回放:TAHRelM 0.491,re-ID K2m2 0.455,**TAH + 3·gallery 0.549** | 1016 §1.5 | 存档 |
| `replay_mot.py` → `replay_mot.json`/`.log` | 10-16 | 同上 | 共享跟踪器 0.665,DeepSORT 在 0.6 s 步长下 0.025 | 1016 §4.3;`acot/runtime/head_mot.py` 引用这个脚本 | 存档 |
| `replay_mot_r0_oldreadback.*` | 10-16 | 修复前跑的 | json 和修复后的版本**逐字节相同** | — | 【可删候选】 |
| `recapture.py` → `recapture.json`/`.log` | 10-18~19 | `PYTHONPATH=. python runs/id_iter/recapture.py` | 112 组的 ≤6 s 再捕获率:Ours 0.60/0.62,**re-ID 0.82**,MOT 12 组合并 0.65,w/o PGIF 0.79 | 1016 §4.8;`headroom.py` 会 import 它 | 【论文引用】、当前最新 |
| `recapture_paired.py` → `.json`/`.log` | 10-18~19 | 同上 | Ours − re-ID 的 ≤6 s 再捕获:**−0.263 [−0.484, −0.048],p=0.033**,n=21 | 1016 §2.3、§4.8 | 【论文引用】 |
| `1016_sec2_3_v0.md` | 10-18 | 手写 | 1016 §2–3 的原稿,已被修订版取代 | — | 【已被取代】 |
| `__pycache__/` | — | — | 孤立的 pyc 文件 | — | 可删 |

**注意**:`recapture.py` 的 mtime(10-19 00:05)晚于 `recapture.json`(00:00)。00:05 的改动推断是修 codex R2 提的 MINOR 问题,只影响 headroom 这条路径。`recapture.json` 没有用修复后的版本重跑。

### 7.3 `headroom/`:第 0 步身份上限检查(10-19,**最新一步**)

**预注册的判据**:D = U1(真值身份头)− 永不切换。D ≥ 0.15 继续;0.10 ≤ D < 0.15 扩大 test 后继续;D < 0.10 停止。实验配置 `acot/configs/experiments/headroom_devB3.yaml`,四个臂:`u1_r0/r1`(`diag_oracle_head`)、`sticky_r0/r1`(`null_sticky`)。

| 文件 | 日期 | 产生方式 | 内容与关键数值 | 状态 |
|---|---|---|---|---|
| `headroom.py` → `headroom.json`/`.log` | 10-18 23:59 → 10-19 03:11 | `PYTHONPATH=. python runs/headroom/headroom.py`(会 import `id_iter/recapture.py`) | **D = +0.089 [+0.009, +0.177],p=0.049**(0.754 对 0.665),判定 `"decision": "stop"`。D(TR90)= +0.050。永不切换 − MOT 合并 = +0.004。**永不切换 − Ours = +0.140,p=0.005**。场景占比:可见 0.38、遮挡 0.21、出画 0.42 | 最新 |
| `chain.sh` / `chain_resume.sh` / `chain.out` | 10-19 | 第一次运行 01:32 遇到调度器死锁,02:48 手动停掉;`chain_resume.sh` 去掉 2080 后续跑 | 最终 4 × 40 全部完成 | 存档 |
| `eval_headroom_devB3{,_run1}.log` | 10-19 | 第一次运行 | 两份**逐字节相同**,里面有 `scheduler.py:324 restart_carla` 的 Traceback | 删掉一份即可 |
| `eval_headroom_devB3_resume.log` | 10-19 03:11 | 续跑 | — | 存档 |
| `eval_smoke.log` | 10-19 | 冒烟,2 局 | 通过 | 存档 |
| `manifest_run1.json` | 10-19 | 备份 | git e18e427,有 557 个 dirty 文件;镜像 `acot-uav-train:cu118-acot` | 存档 |
| `codex_r{1,2}{,_prompt}.md` | 10-18~19 | codex 审查 | R1 MAJOR,R2 **NO MAJOR** | 存档 |

**下一步**(`1019_metric_protocol_discussion.md` §4–§5,都等用户拍板):
1. 确认 §0 纪律:选口径只看 U1 和永不切换。
2. 确定 M1(严格跟随率)、M2(误跟率)、M3(再捕获)。先在第 0 步的日志上算配对的 D_M1。
3. 修复调度器死锁,先过 codex,然后跑 P1 飞行上限诊断。
4. 设计 P2 身份压力场景。

### 7.4 `mot_baselines/`:DeepSORT / OC-SORT 原版复现(10-16 → 10-17)

| 文件 | 日期 | 产生方式 | 内容与关键数值 | 状态 |
|---|---|---|---|---|
| `chain_grid.sh` / `chain.out` | 10-16 | `acot eval mot_grid_devB3.yaml`,6 个 worker,然后 `grid_summary.py` | 第二次启动 480/480 完成 | 存档 |
| `eval_mot_grid_devB3.log` | 10-16 | 同上 | CARLA 重启或失败 21 次,都自动补齐 | 存档 |
| `eval_mot_grid_devB3.aborted_0750.log` | 10-16 | 第一次启动 | 里面是 12 条 `--tid-head invalid choice`,**和 chain.out 写的"2 小时上限"对不上** | 【可删候选】 |
| `eval_mot_smoke.log` | 10-16 | 4 臂 × 2 | 全部完成 | 存档 |
| `grid_summary.py` → `grid_summary.json`/`.log` | 10-16 | `PYTHONPATH=. python runs/mot_baselines/grid_summary.py` | DeepSORT 10 Hz 下 0.599–0.711;OC-SORT 0.613–0.765;12 组均值 0.660,SD 0.047。每帧开销 DeepSORT 67 ms,OC-SORT 12 ms | 【论文引用】 |
| `commit_dynamics.py` → `.json` | 10-16 | 同上 | **40/40 局里绑定的轨迹都会被删除**;92.5–100% 的局承诺不变,等于"粘住开局 ID" | 【论文引用】 |
| `all_devB3_vs_sticky.py` → `.json`(68 KB) | 10-17 | 同上 | 零假设基线 0.660 / TR90 0.185,所有 dev-B @3 臂逐一做配对比较 | 【论文引用】 |
| `codex_mot_r{1,2}{,_prompt}.md` | 10-16 | codex | R1 MAJOR(OC-SORT 读回顺序错),R2 **NO MAJOR** | 存档 |
| `codex_mot_r1_failed_model.md` | 10-16 | codex 调用失败 | 没有内容 | 可删 |

**结论**:原版复现退化成"粘住开局 ID",pc ≈ 0.66,不低于 re-ID 的 0.636,高于 Ours 的 0.525。按选择规则得到的 OC-SORT 0.765 只是**选择噪声**,论文里不能写成"MOT 身份机制更好"。

### 7.5 `reacq/`:再捕获统计的早期版本(10-18)

`reacq_devB3.py` → `reacq_devB3.json`(122 KB)。**没有任何地方引用它**。同一时间出现的 `id_iter/recapture.py` 取代了它(推断)。【可删候选】

### 7.6 `dam4sam_s/`:DAM4SAM 骨干从 Hiera-L 换成 Hiera-S(10-15)

- 跑 `acot eval dam4sam_s_devB3.yaml`,方法 `ds_r0/r1`。服务实例单独启动:`dam4sam_service.py --model sam21pp-S`,端口 5611–5616。
- `compare_s_vs_l.py` → `compare_s_vs_l.json`:**S 0.671 对 L 0.720**,Δ −0.049 [−0.120, +0.021],p=0.19。身份耗时 L 约 285–348 ms,S 约 149–161 ms。S − Ours = +0.146(p=0.0009)。
- 状态:【论文引用】,但只作内部记录,因为 10-16 已决定把 DAM4SAM 从主比较中移除。

### 7.7 `infra/`:acot 运行时的基础设施状态

| 路径 | 内容 | 产生方 / 消费方 | 状态 |
|---|---|---|---|
| `filehash_cache.json`(54 KB) | 265 条 "路径\|大小\|mtime" → sha256 | `acot/provenance.py:14` | 可以删,删了会重算 |
| `leases/{carla_20x0,gpu_N,tm_block_N}.lock`(22 个) | JSON 格式,记录最后一次持有者 | `acot/infra/lease.py`,用 fcntl.flock 非阻塞加锁 | **有 eval 在跑时不能删**。10-19 用 lslocks 查过,当时没有进程持锁 |

### 7.8 环境依赖:`runs/pydeps*`(pip `--target --no-deps`,用来补 docker 镜像里缺的包)

| 目录 | 大小 | 包和版本 | 谁用,怎么挂载 |
|---|---|---|---|
| `pydeps/` | 16 MB | `timm` 1.0.29 | `runs/tah_v90/pipeline{,_v2}.sh` 用 `-e PYTHONPATH=$R/runs/pydeps` 挂进去,离线 `HF_HUB_OFFLINE=1`。正常 eval 不依赖它:`acot/jobs.py:29`、`scheduler.py:229` 会在容器里 `pip install timm==1.0.29` |
| `pydeps_dash/` | 127 MB | `av` 19.0.1(PyAV) | 数据看板:`PYTHONPATH=runs/pydeps_dash python carla_uav_tracking/scripts/datagen_dashboard.py --config runs/datagen_moving/dashboard.json --port 8767` |
| `pydeps_mot/` | 1.2 MB | `filterpy` 1.4.5 | `acot/runtime/head_mot.py:37`,append 到 sys.path(OC-SORT 用) |
| `pydeps_oavat/` | 290 MB | `scipy` 1.14.1、`opencv_python_headless` 4.10.0.84 | `acot/runtime/head_oavat.py:33`、`head_mot.py:37`,都是 append 到 sys.path |
| `__pycache__/` | 151 B | 孤立 pyc | 可删 |

---

## 8. 交叉索引与已知问题

### 8.1 runs 目录 ↔ `eval_out/exp/` 实验名 ↔ 笔记

`eval_out/exp/` 下一共有 40 个实验。下表只列和 runs/ 有关的。

| runs 目录 | `eval_out/exp/` 实验名(臂) | 实验 yaml | 笔记 |
|---|---|---|---|
| `p02_retrain/` | `p02_fast_a_devB3`(fa_r0/r1、fa_cal_r0/r1) | `p02_fast_a_devB3.yaml` | 1012 §3.3、§5;1007 §4b |
| `p02_retrain/` | `p02_fast_b_devB3`(fb_r0/r1) | `p02_fast_b_devB3.yaml` | 1012 §6.4、§6.8 |
| `p02_retrain/` | `p02b_dagger_{smoke,train,val}`(dag_fa) | `p02b_dagger_*.yaml` | 1012 §6.2、§6.7 |
| 作为参照的 Ours | `opt1_screen_devB3/tau005`(r0)、`opt2a_screen_devB3/tau005_r1`(r1) | `opt1/opt2a_screen_devB3.yaml` | 1007 §4b |
| 作为参照的基线 | `t0_grid_devB`(63 臂:`reid_a0p0_t0p05`、`xattn_t0p0`、`oavat_r0p80_t0p5`、`deepsort_g160_l2_t0p2`、`dam4sam_g160_t0p05` 等) | `t0_grid_devB.yaml` | 1006_t0_log、1007 |
| `id_iter/recapture.py` | 另外还读 `t0_pilot_deploy_devB3`、`t0_pilot_mech_devB3`、`t0_refopen_devB3`、`t0_dinov3_stage2_devB3`、`opt2b_screen_devB3` | 同名 yaml | 1016 §4.7–4.8 |
| `dam4sam_s/` | `dam4sam_s_devB3`(ds_r0/r1) | `dam4sam_s_devB3.yaml` | 1007 §4b、§5.8 |
| `mot_baselines/` | `mot_grid_devB3`(12 臂)、`mot_grid_devB3.aborted_0750`(空跑)、`mot_smoke_devB3` | `mot_grid/smoke_devB3.yaml` | 1016 §4.4–4.7 |
| `headroom/` | `headroom_devB3`(u1_r0/r1、sticky_r0/r1,219 MB,含 trk)、`headroom_smoke_devB3` | `headroom_devB3.yaml` 等 | 1016 §3、1019 §1 |
| `acot_logs/t0/` | `t0_*`、`opt*_screen_devB3` | 同名 yaml | 1006、1007 |
| `asd/`、`opening_dump/val` | 旧入口,没有 exp 目录;结果在 `eval_out/paper_it_*.json`、`dev_sc_*.json` 等 | `experiments/legacy/*.yaml` | 1001–1004 |
| 8–9 月的日志 | 旧入口:`eval_out/paper_*.json`、`ex_*.json`、`reid_*.json`、`m1.json`、`bc_e0.json`、`expert_control.json` 等(**同名文件会被覆盖**) | — | 901、906、907 |

### 8.2 与记忆中结论的核对(都一致)

| 结论 | 来源文件 |
|---|---|
| Stage-2 H0:mis 0.579 / 0.715 | `train_v5.log` / `train_v5_nolang.log` |
| 语言重锚被证伪:0.363 / 0.225 / 0.741 | `reanchor_full.log` |
| TAHRelM 部署化后崩溃:0.146 | `eval_tahrelm_committed.log` |
| 1007 主表:Ours 0.525,DAM4SAM 0.720,re-ID 0.636 | `acot_logs/t0/sec4b_update_1008.json` |
| P0-2:A +0.060,B +0.016,都 FAIL | `p02_retrain/decision_{a,b}.json` |
| re-ID 赢在丢失后的恢复:0.557 对 0.21 | `id_iter/diag_switch.json`(1016 §1.2b 更正:根因主要是跟踪器 ID 互换) |
| MOT 原版退化成粘住开局 ID,pc ≈ 0.66 | `mot_baselines/commit_dynamics.json`、`all_devB3_vs_sticky.json` |
| DAM4SAM-S 0.671 | `dam4sam_s/compare_s_vs_l.json` |
| 第 0 步 D = +0.089,STOP | `headroom/headroom.json` |

### 8.3 未解决的问题和坑(按影响排序)

1. **调度器死锁没修**:`acot/eval/scheduler.py:324 restart_carla` 抛出的异常没有在 `Dispatcher.work` 里被捕获。结果是 worker 线程退出,但活跃计数不减,其他 worker 一直空等。证据见 `headroom/eval_headroom_devB3.log`。跑下一批之前要先修,并过 codex。
2. **所有闭环数字的 WHERE 都用了真值**:`policy.py:_where` 425–444 用的是真值位置,候选框也是真值投影出来的伪框。做完整系统对比之前,要换成不依赖真值的定位(1009 §5)。
3. **部署权重无法复现**:`stage2_v5` 的原因见 §0.4。`tah_relm_dag.pt` 没有固定 seed,训练数据里还混着 FOV 错位的离线部分。
4. **配置里有静默陷阱**:`config_v5.yaml` 的 ear_ckpt 会被静默加载(见 §2.3);旧入口的 `--tah-ckpt` 默认值是 `tah_rel.pt`(见 §5.G)。
5. **元数据会误导**:BC/RL 的 ckpt 里 epoch 和 metrics 是复制来的;`sec4b_p02.json` 里的 n=39 是循环覆盖留下的;`tah_1006/tah_relm_s*` 其实是 DINOv2;`acot_logs/t0/chain_C*.done` 标着失败,但下游 JSON 存在(推断是手动重跑过)。用 `.done` 判断成功与否不可靠。
6. **笔记和实际不一致**:
   - `seeds.yaml` 里 `train_dagger_v1` 的说明写错了。
   - `1004_tahrelm_retrain_diag.md:70` 写 105 集,实际 101 集。
   - `1003_data-regen-v6-spec.md:59` 写的 qc_v6.py 路径不对。
   - `mot_baselines/chain.out` 记录的中止原因和日志对不上。
7. **权限**:`id_iter`、`headroom`、`mot_baselines`、`reacq`、`dam4sam_s`、`pydeps_mot`、`__pycache__` 是 700 权限,别的账号读不了。
8. **时间戳**:日志里的时间是容器 UTC,文件 mtime 是宿主时间(UTC−4)。

---

## 9. 清理建议(只是建议,未执行任何删除;删除前需要用户确认,root 文件要用 sudo 或在容器内删)

| 对象 | 大小 | 能否再生 | 再生方式 | 建议 |
|---|---|---|---|---|
| `mp4_preview/` 全部 | 2.2G | **能** | `python -m carla_uav_tracking.scripts.h5_to_mp4 --src /nvidia/hque/data/carla_data/mvp_full_v5 --out runs/mp4_preview/mvp_full_v5 --fps 10 --workers 8`;带框版加 `--boxes`;fov90 版把 `--src` 换掉。需要 `imageio av pillow` | **可删**(root) |
| `datagen_moving/dash_videos/run1_contaminated` | 417M | 能(看板会自动渲染) | `datagen_dashboard.py` 的 render_loop | **可删** |
| `datagen_moving/dash_videos/{full_fast,full_qcfail}` | 790M | 能 | 同上 | 可删,但看板会重新渲染,占用 CPU |
| 4 个 `stage2_*_last.pt` + `stage2_v5_noear/` + `stage2_v5_noiar/` | 约 3.8G | — | — | 可删候选(root) |
| `stage2_bc_center_b0.7.pt` | 554M | 不能(缓存已删) | — | 可删候选;b0.4 和 rl_e0 被 906 引用,建议保留 |
| `tid_*.pt` / `tid2_*.pt`(除 `tid_attrbind*`) | 约 100M | 不能 | — | 可删候选 |
| `opening_dump/npz_tr` / `npz_hi` | 3.7G / 1.6G | **不能逐位复现** | `run_dumpt.sh` / `run_dumph.sh`(需要 CARLA) | **保留**(现行前端的训练和评测数据,被 artefact 钉住) |
| `opening_dump/npz` | 166M | 不能逐位复现 | `run_dump.sh` | 低优先级可删(注意 `opening_score.py` 默认读它) |
| `asd/iter/*.trk.jsonl` | 479M | 不能 | — | 用 gzip 压缩,不要删 |
| `artefacts/tah_seq/tah_seq_v5_fov90_t6@417a` | 456M | 能 | `python -m acot prep acot/configs/prep/tah_seq_v5_fov90_t6.yaml` | 可删候选(删后 t6_ctl 的输入会悬空) |
| `artefacts/tah_seq/tah_seq_v5_fov90@ea7b` | 7.2M | — | — | **可删**(中断的 prep) |
| `tah_v90/off90_shard{0..3}.pt` | 229M | 能 | 与 off90.pt 内容重复 | 可删候选 |
| `abl_allactor/cache_s*.pt` | 181M | 能 | `python runs/abl_allactor/build_cache_v5.py --shard k --nshards 3 --out runs/abl_allactor/cache_s{k}.pt` | 可删候选 |
| `tah_1006/tah_cache_dinov3.pt` | 440M | 能 | `acot/prep/tah_cache_backbone.py` | 可删候选 |
| `tah_1006/tah_cache_dinov3_dag.pt`、`abl_allactor/runs3/w` | 449M / 106M | 不能 | — | 保留 |
| 重复或空文件:`headroom/eval_headroom_devB3_run1.log`、`id_iter/replay_mot_r0_oldreadback.*`、`p02_retrain/prep_dagger_v3*.log`、`mot_baselines/codex_mot_r1_failed_model.md`、`train_v2_ORDERBUG.log`、`mmap_convert.log` | 很小 | — | — | 可删 |
| 大体积的 codex 转录(`patch/codex_p02_r1.md` 2.5MB、`arch_check/*.md` 等) | 若干 MB | — | — | 可以压缩 |

**不能删的东西**见 §2.2。

---

## 10. runs/ 以外的相关位置

| 内容 | 位置 |
|---|---|
| 训练和评测数据 | `/nvidia/hque/data/carla_data/`:`mvp_v5_fov90_fast/`(现行 64 集,有 manifest)、`mvp_full_v5_fov90/`(90° 复刻)、`mvp_full_v5/`(09-04 的 70° 重生成版)。`*_qcfail`、`*_run1_contaminated` 不能用。`berm*` 属于论文 2 |
| 闭环逐集结果 | `carla_uav_tracking/rollout/eval_out/`(`exp/<实验>/` 和旧的 `*.json`) |
| 开局分类器权重 | `carla_uav_tracking/rollout/eval_out/opening_attr_{cls,ft}.pt` |
| 外部基线代码和权重 | `external/`(DAM4SAM、OA-VAT、deep_sort、OC_SORT、dinov3 等,各自是独立的 git 仓库,被 gitignore) |
| HF 模型缓存 | `~/.cache/huggingface/hub/`(Qwen3-VL-4B-Instruct 8.3G,timm DINOv2 ViT-S/14 85M) |
| 产物登记 | `acot/legacy/artefacts.yaml`(legacy)、`runs/artefacts/index.json`(全部) |
| 进度与决策 | `acot_note/1011_status_and_plan.md` → `1012_p02_identity_retrain.md` → `1016_identity_iteration.md` → `1019_metric_protocol_discussion.md` |
| 闭环代码路径 | `acot_note/1012_closed_loop_impl_map.md` |
