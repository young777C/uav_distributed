# 闭环 Rollout Harness 实现指南

> 2026-08-14 起草 · 2026-08-18 建成并验证 · 配套：`experiment-design.md` §7（闭环评测协议 + 双重角色）、`training-plan.md` Phase E.5（Stage-3 依赖）
> 代码落点：`carla_uav_tracking/rollout/`
> **状态：M1–M5 全部建成并逐个验证（见 §10）。已用 harness 跑出首个真实评测：H0 闭环语言消融判据成立（§10.2）。**

---

## ⚑ 构建状态速览（2026-08-18）

| # | 里程碑 | 产出文件 | 验证 | 结果 |
|---|---|---|---|---|
| **M1** | 在线 VLM | `online_vlm.py` `parity_online_vlm.py` | 与离线缓存对同帧比对 | ✅ **逐位一致**（cos=1.0, diff=0.0） |
| **M2** | 策略加载器 | `policy.py`(`load_policy_modules`+S1/S2) `sanity_policy_load.py` | 加载后跑离线 `_evaluate` | ✅ **复现 mis_follow=0.57889（Δ=0）** |
| **M3** | 候选框构造 | `candidates.py` | 随 M5a 联跑 | ✅ GT 投影+打乱+无身份锚 |
| **M4** | 打分器 | `metrics.py`(`RolloutScorer`) | 专家 smoke 验数值 | ✅ 迟滞 mis-follow/SR/重捕获/捷径 |
| **M5a** | CARLA 反转循环 smoke | `env.py` `smoke_expert_loop.py` | 专家跑通 CARLA | ✅ 200/200 帧, dist30.9m≈语言"30m" |
| **M5b** | socket 桥接真策略 | `bridge.py` `policy_server.py` `remote_policy.py` `run_rollout.py` | 真 VLA 驱动 CARLA | ✅ 端到端出真实闭环指标 |

**选序逻辑**：风险从高到低、依赖从底到顶。M1 风险最高（在线 VLM 若与训练缓存不一致，整个模型行为都错）且可脱离 CARLA 独立验证 → 第一个做；M5 依赖前四个全对 → 最后做。每步有独立 PASS 判据，一步错不带到下一步。评测工具（`eval_h0.sh`/`compare_runs.py`）是 harness 的**应用**，不属 M1–M5 本体。

---

## 0. harness 是什么

一句话：**把现有的离线专家数据生成循环，专家（特权 PID）换成训练好的 VLA 策略，策略只看 RGB + 语言。**

换入点极窄。专家只在 `recording/recorder.py:167` 一处进入：

```python
action, pert = self._scene.expert_action(tpos, drone_pos, drone_yaw, target_visible, elapsed)  # tpos = 目标世界坐标真值
self._scene.step_drone(action)   # recorder.py:188 → KinematicDrone.step(dx,dy,dz,dyaw)
```

`action = (dx,dy,dz,dyaw)` 正好 = DiT 动作头输出（`action[:4] * a_scale`，a_scale=5.0）。无人机运动学（`drone.py:73` 纯积分 spectator）、`world.tick()` 节奏（10 Hz 同步）、场景/目标/干扰车 spawn 全部原样保留。

**双重角色（为什么是硬前置而非仅评测）**：同一 harness 也是 Stage-3 RL 的训练底座——RL 的目标是对「策略自身 rollout 分布」的期望，没有环境在训练回路里就没有轨迹/奖励/可优化对象。BC(Stage 1-2) 读固定离线数据不需要它；RL(Stage 3) 没它无法存在。**先有 harness，才谈得上 Stage-3。**

---

## 1. 相对专家循环的两个结构性改动

### (a) 循环反转 —— 帧必须先于动作

专家循环今天的真实顺序（`recorder.py:167→188→197→198`）：

```
① action = expert_action(tpos, ...)   # 用【目标真值】决策，根本不看画面
② step_drone(action)                  # 移动无人机 + 相机
③ world.tick()                        # 仿真前进 + 相机曝光
④ rgb = get_frame()                   # 最后才拿到画面（只当训练数据录下来）
```

学生的输入**就是画面**，要在第 t 步决策必须先看到第 t 步画面，所以顺序反转：

```
① world.tick()                        # 先让仿真走一步，相机在 P(t) 曝光
② rgb = get_frame()                   # 拿到 frame(t)
③ action = policy(frame, language)    # 用画面决策
④ step_drone(action)                  # 再移动到 P(t+1)
```

| | 决策输入 | 时序 |
|---|---|---|
| 专家 | 目标真值 `tpos` | 决策 → 移动 → tick → 拿帧 |
| 学生 | 画面 `frame` | tick → 拿帧 → 决策 → 移动 |

"拿帧"从决策**之后**挪到决策**之前**。这不是加个 if 能切换的——整段 tick/capture 骨架不同，**所以 fork 而非复用 `recorder.run()`**。

### (b) 特权 GT：控制绕过，仅用于打分

`get_target_state` / `get_distractor_states (D,6)` / `_point_in_frame` / `occ_detector` / `search_mode`——学生的**控制器**绝不读；**打分器**每 tick 读来算 SR/mis-follow/遮挡恢复。专家专属的 centroid 取景、dead-reckoning、强制盲飞 pan 全部删。loss window 只用来给打分器**标记重捕获帧**，不再强制目标躲避、不再把学生飞盲。

### 为什么 fork 而非改 recorder
1. 时序是反的，非小改。
2. `recorder.run()` 是数据生成生产代码，正被 data-gen agent 使用，改它有风险、互相干扰。
3. 学生循环要删大量专家逻辑、保留的 GT 只喂指标——与原循环差异大。

**落地方式**：新建 `rollout/env.py`，把循环搬过去重写；只借用 `EpisodeRecorder` 的两个纯函数（`_create_attached_sensor` / `_point_in_frame`，通过 `__new__` 拿到、不跑其循环）；原 recorder 一行不动。✅ 已完成。

---

## 2. 模块布局

放在 `carla_uav_tracking/rollout/`（需同时 import CARLA 侧 `scene/`+`carla_uav/` 与 `train/`，故落在 CARLA 侧、`train/` 上 sys.path）：

```
rollout/
  online_vlm.py     # Phase E.1：build_backbone 一次，每帧 encode → 缓存张量
  policy.py         # 加载四模块 ckpt，S1/S2 异步调度，.act(obs)->action
  candidates.py     # 每帧构造 cand_feats + 候选框（H0 决策点）
  env.py            # forked recorder 循环：学生驱动，GT 仅打分  ✅已建骨架
  metrics.py        # §7.2 指标族 + 迟滞（sustained）mis-follow
  run_rollout.py    # 入口：N rollout × seed × config，写结果 JSON
  config_rollout.yaml
```

---

## 3. online_vlm.py —— Phase E.1（「VLM 必须在线跑」）

在线函数不存在——`backbone_kv.py` 只写磁盘缓存。但原语可直接调用，建一次、每帧调：

```python
from acot_probe.backbones import build_backbone, pool_box
from train.backbone_kv import _pool_grid   # (Gh,Gw,C) -> (g*g, C)

class OnlineVLM:
    def __init__(self, cfg, device):
        b = cfg["backbone"]
        self.bk = build_backbone(b["kind"], b["model_id"], device,
                                 min_pixels=b.get("min_pixels"), max_pixels=b.get("max_pixels")).load()
        self.layers = cfg["iar_layers"]      # 例 [4,8,12,16]（+ layer 24 作 vlm_ctx）
        self.g = int(b.get("ctx_grid", 10))  # M = 100 tokens, C = 2560

    @torch.no_grad()
    def encode(self, pil_img, lang):
        feats = self.bk.encode(pil_img, lang, self.layers)          # {L:(grid[Gh,Gw,C], text_vec)}
        vlm_ctx_layers = [torch.tensor(_pool_grid(feats[L][0], self.g)) for L in self.layers]  # list[L] (100,2560)
        vlm_ctx = vlm_ctx_layers[-1]                                # (100,2560) → EAR/DiT
        grid_last = feats[self.layers[-1]][0]                       # 供 cand_feats
        return vlm_ctx_layers, vlm_ctx, grid_last
```

- 模型已 frozen/bf16/eval（`backbones.py:134`）。
- `lang` = 整集固定的语言指令（目标身份）。w/o-language 消融喂 `"Track the target vehicle."`（复用 `backbone_kv.py:78` 的 no_language 串）。
- `encode()` 现在每次 `.cpu().numpy()`，1 Hz S2 循环够用；若成瓶颈改留 GPU。

---

## 4. policy.py —— 四模块加载器 + S1/S2 异步调度

**加载器**（无现成入口，从 ckpt `dims` 块 + 训练 `cfg["model"]` 组装）：

```python
ckpt = torch.load("runs/stage2_v3/stage2_best.pt", map_location=device)   # best 按 mis_follow 选
d, m = ckpt["dims"], cfg["model"]     # dims: cond_dim=2560,K=4,H=16,A=5,n_layers_in,cand_dim=2560,pdim=5
ear = EAR(d["cond_dim"], k=d["K"], **m["ear"]).to(device);  ear.load_state_dict(ckpt["ear"])
iar = IAR(d["cond_dim"], d["n_layers_in"], **m["iar"]).to(device);  iar.load_state_dict(ckpt["iar"])
dit = DiT(d["A"], d["H"], d["cond_dim"], d_ex=3, d_im=m["iar"]["d"], proprio_dim=d["pdim"], **m["dit"]).to(device)
dit.load_state_dict(ckpt["dit"])
tid = TargetIDHead(d["cand_dim"], d["cond_dim"], d=m.get("tid_d",256)).to(device);  tid.load_state_dict(ckpt["tid"])
for mod in (ear,iar,dit,tid): mod.eval()
```

**S1/S2 调度（新逻辑）**：仿真 10 Hz，故 **S1 = 每 tick（10 Hz），S2 = 每 `s2_period` tick（例 10 → ~1 Hz）**。S2 重算慢流（VLM encode → EAR `z_ex`、IAR `z_im`、tid logits）；S1 复用缓存的 `z_ex/z_im/vlm_ctx`，仅用新 proprio 重采样 DiT 动作。

```python
class Policy:
    def act(self, obs, gt_candidates):
        tick = obs.step
        if tick % self.s2_period == 0:                      # ---- S2 慢流 ~1 Hz ----
            layers, self.vlm_ctx, grid_last = self.vlm.encode(pil(obs.rgb), obs.language)
            self.ctx_layers = [x[None].to(self.dev) for x in layers]
            self.vlm_ctx_b  = self.vlm_ctx[None].to(self.dev)
            self.ctx_mask   = torch.ones(1, self.vlm_ctx.shape[0], dtype=torch.bool, device=self.dev)
            prop = torch.tensor(obs.proprio)[None].to(self.dev)
            self.z_ex = flow_matching.sample(self.ear, self.vlm_ctx_b, self.ctx_mask,
                                              k=self.K, steps=self.steps, proprio=prop)   # (1,K,3)
            self.z_im, _ = self.iar(self.ctx_layers, self.ctx_mask)                        # (1,n_im,d)
            cand_feats, self.cand_boxes, cand_mask = self.candidates.build(grid_last, gt_candidates)
            logits = self.tid(cand_feats, self.vlm_ctx_b, self.ctx_mask, cand_mask)        # (1,N)
            self.tid_choice = self._commit(logits)          # 迟滞承诺，§6
        # ---- S1 快流：每 tick DiT 用新 proprio 重采样 ----
        prop = torch.tensor(obs.proprio)[None].to(self.dev)
        act = action_sample(self.dit, self.z_ex, self.z_im, self.vlm_ctx_b,
                             prop, self.ctx_mask, self.H, self.A, self.steps)   # (1,H,5)
        dx,dy,dz,dyaw = (act[0,0,:4] * self.a_scale).tolist()   # a_scale=5.0
        search_mode   = float(act[0,0,4] > 0.5)                 # 原始第5通道，自己阈值
        return (dx,dy,dz,dyaw), search_mode, self.tid_choice
```

两个训练已铺好的坑：
- **EAR 必须在线跑**：`_evaluate` 用 GT `b["waypoint"]` 作弊，闭环没 GT，必须 `flow_matching.sample(ear,...)`。DiT 训练时加了 staleness 噪声（`_stale` p=0.3）正是为容忍这个 S1/S2 gap——§3.3 要求扫 `staleness_p∈{0,0.3,0.6}` 对闭环 SR 验证。
- **proprio 是 5 维不是 8**（`dataset_stage2.py:253`）：`[cam_z/30, cam_pitch/90, vx/15, vy/15, vz/15]`。waypoint 是相机系、scale 50，但 EAR 输出只在内部给 DiT，不用解码。

---

## 5. candidates.py —— 唯一真正的设计决策（H0 关键）

`cand_feats (N,2560)` = 候选框池化的 VLM 特征（`pool_box(grid_last, box)`），tid 头在其上 argmax → 「锁定哪辆车」。**rollout 时候选框从哪来？** 直接触及 H0 诚实性：

| 方案 | 做法 | 成本 | H0 风险 |
|---|---|---|---|
| **A. GT 投影框（建议起步）** | 把 GT 目标+干扰车世界 XYZ 经实时相机位姿投影（复用 `acot_probe/projection.py` + `Candidate.frac_box`，同 `dataset_stage2.py:273`） | 极低、确定性 | 给策略一个每帧候选**集**（位置）——只要**不交出身份**即可接受：框无标签且打乱，tid 头仍须靠语言选。与训练完全一致。 |
| **B. 检测器提框** | RGB 上跑车辆检测器 | 真实感知、额外模型 | 完全诚实/可部署，但把跟踪误差与检测误差混在一起，是独立工程。 |

建议：**先 A**（闭环数字与离线可比，训练本就用投影 GT 框），但**每帧打乱候选顺序**（`dataset_stage2.py:283`）保持指标诚实，**绝不给持续首帧 bbox 身份锚**（R3/H0 红线，memory [[acot-uav-benchmark-leak]]、[[acot-uav-tid-metric-artifact]]）。记 `candidate_source` 字段供论文说明。要部署性声明时再上 B。

---

## 6. metrics.py —— §7.2 指标族 + 迟滞规则

每 tick 从 GT（打分器，非控制器）读：目标可见？GT 说哪个候选是目标？UAV-目标距离、在帧内、遮挡、search_mode。**文档反复强调（§3.4 族B、§7）**：闭环 mis-follow **不逐帧 argmax**——那会惩罚 flow-matching 随机性的单帧闪跳。实现**承诺 + 迟滞**：

- 策略维护**承诺目标**；新候选须连赢 **K=5 帧**（~0.5s@10fps）才切换（`_commit()` 在 policy.py）。
- **ID-switch（主指标）** = 承诺目标切到错车并保持 ≥K 帧。episode **失败** = 承诺错车 ≥T=2s。
- **并列报**逐帧（严，诊断）与持续（K=5）两个数。
- **重捕获分开报**（旗舰场景 [[acot-uav-predict-intercept]]）：*重捕获收敛时间*（允许延迟不罚）与 *重捕获锁错率*（硬失败——「语言唯一可分」的核心考验）分开。

产出指标：SR、Avg Tracking Frames、居中率、遮挡恢复率、碰撞/安全、Cross-town SR drop，加族E语言指标（mis-follow 全集+难例、w/o-lang 增幅）。难度档对齐 **EVT-Bench STT/DT/AT**（= 课程 2a/2b/2c），分档报 SR。

---

## 7. env.py（已落地）+ run_rollout.py

`env.py`（✅ 已建骨架）逐字复用 `SceneManager.setup_episode()`（同 seeding `base_seed+episode_id`，可复现且换 seed 偏移即未见），设同步模式（`fixed_delta_seconds=0.1`），跑 90-tick warmup（warmup 用专家飞即可，只需目标动 >1.2 m/s），再跑 §1 的反转学生循环。保留 `occ_detector`/`_point_in_frame`/`get_target_state`/`get_distractor_states` **只**喂 `metrics.py`。loss window 只标记重捕获帧、不强制专家盲飞。

`run_rollout.py`：循环 config × ≥30 rollout × 3 seed，≤180s/集，`test_town=Town05`，写每-rollout + 汇总 JSON 带 per-episode bootstrap CI（§8：配对 M0-vs-Mk，Holm-Bonferroni）。必测：**M0 / M1(w/o-Lang) / M3(w/o-EAR) / M4(w/o-IAR)** + 一个外部基线——离线消融方向必须闭环复现，否则以闭环为准（§3.2）。含**捷径检查**：「目标恒居中」「总跟最近车」占比，高 ⇒ 捷径未除，回 §2.3 加难数据。

运行在现有 `docker-compose.yml` 的 `cyh-carla` 服务内（`CarlaUE4.sh -RenderOffScreen -graphicsadapter=$GPU`——Vulkan-GPU 坑 [[acot-uav-review-loop]]），`train/` 上 sys.path，VLM 放空闲 GPU。守 data-gen 协调边界（topup CARLA 在跑则跳 GPU1，[[acot-uav-data-run-ops]]）。

---

## 8. 构建顺序（每步可独立验证）

1. **`online_vlm.py`** —— 断言其 `vlm_ctx/vlm_ctx_layers/cand_feats` 与已算 `.npz` 缓存对某已知帧数值一致（去风险全局关键：在线 VLM == 离线缓存）。
2. **`policy.py` 加载器 + 单帧 `.act()`** —— 加载 `best.pt`，喂一缓存帧，确认 tid argmax 与离线 `_evaluate` 一致。
3. **`env.py` 先用专家驱动**跑通反转循环 —— 验证循环重排没破坏 CARLA（tick/曝光时序）。
4. **换入学生**，跑 1 集看跟踪，再接指标，再放大到 N×seed。
5. 闭环评测稳定后 → **Stage-3 RL** 挂上：同 `env.py` 作 rollout 源，GRPO 采 K 条动作（flow-matching 天然随机），奖励用 GT 打分器（`R_correct_id` = §6 承诺目标逻辑，防 hacking），冻结 VLM/EAR/IAR，训 DiT 末4层。

---

## 五个会咬人的坑（速查）

1. **循环反转** —— 帧先于动作（§1a）；fork 不改 `recorder.run()`。
2. **EAR 必须在线跑** —— `_evaluate` 的 GT-waypoint 捷径闭环不存在（§4）。
3. **proprio 是 5 维**、相机系、特定归一化（§4）。
4. **10 Hz 仿真** → S1=每 tick、S2=每 ~10 tick；「20 Hz S1」是名义值（§4）。
5. **候选框来源** = H0 决策 —— 起步用 GT 投影 + 打乱 + 不给身份锚（§5）。

---

## 待验证假设（第一集实测确认）
- `step_target(dt)` 与 autopilot 的关系：目标车是 autopilot，真正推进在 `world.tick()`，`step_target` 只更新跟踪状态（is_turning）——按专家用法放 tick 前，跑通第一集确认目标在动。
- 同步模式下 sensor 在 tick 时对**当前** actor 位姿曝光——里程碑 3「先用专家跑通反转循环」正是验这个。

---

## 9. 容器拓扑 & socket 桥（2026-08-18 实建验证，关键约束）

harness 的**最大工程约束**：CARLA 与 VLA policy **不能同进程**。

| | CARLA 容器 `cyh-carla` | 训练容器 `acot-uav-train:cu118` |
|---|---|---|
| python | **3.6**（解释器是 `python`，非 `python3`；numpy1.24+carla） | **3.11**（torch+transformers+VLM） |
| carla client | ✅ 0.9.15 | ❌（0.9.15 无 py3.11 wheel，只 0.9.16/0.9.5） |
| torch/VLM | ❌（py3.6 太老） | ✅ |
| 挂载 | 只 `carla_uav_tracking→/workspace`（**无** acot_probe/train） | repo 根同路径挂载 |
| CARLA server | **端口 2012 / GPU1**（非 2000！graphicsadapter=1） | — |
| 网络 | `cyh-carla-net` | 需 `--network cyh-carla-net` 加入 |

→ **两进程 socket 桥**（都已建、已连通验证：server 日志见 `client connected: 172.18.0.2`）：
- `policy_server.py`（训练容器）：持 `OnlineVLM`+`Policy`，`--network cyh-carla-net --name acot-policy-server`，`0.0.0.0:5555` 收帧发动作。
- `run_rollout.py`（cyh-carla）：跑 `env.py` 反转循环，`RemotePolicy` 每 tick 发 `{rgb,language,proprio,step,gt_candidates}`、收 `{action,search_mode,cand_lite,pred_slot}`。
- `bridge.py`：长度前缀 pickle **proto=2**（py3.6↔3.11 互通）；numpy 数组跨版本可 pickle；**cand_feats 不过网**（只传身份+depth+pred_slot 给 env 侧 scorer）。
- **包约定统一**：`rollout.*` 顶层包；入口脚本把 repo根(train/acot_probe)+carla_uav_tracking(rollout/scene/recording) 都加 sys.path。env.py/metrics.py **torch-free、不依赖 acot_probe**（`_target_central` 内联投影）→ 能在 py3.6 CARLA 容器跑。

**启动顺序**：
```
# 1) policy server（训练容器，挂 cyh-carla-net）
docker run -d --gpus all --network cyh-carla-net --name acot-policy-server \
  -v $REPO:$REPO -v $DATA:$DATA:ro -v $HF:/root/.cache/huggingface -w $REPO \
  acot-uav-train:cu118 python carla_uav_tracking/rollout/policy_server.py \
    --config train/config_v5.yaml --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral
# 2) rollout（cyh-carla，连 server by name）
docker exec cyh-carla python /workspace/rollout/run_rollout.py \
  --policy-host acot-policy-server --episodes 30 --steps 1500 --seed-base 90000 --out ...
```
**性能**：s2_period=1（VLM 每 tick）+786KB 帧过 socket ≈ ~0.5-1s/tick，socket-bound（GPU 仅 ~10-70%）。要加速：s2_period>1（staleness 训练支持）或 JPEG 压帧。同步 sim 不怕慢（只影响 wall-clock）。

**语言消融 A/B（H0 闭环）**：起两个 server，`--language-mode neutral`（M0）vs `none`（M1 w/o-lang），各跑同 seed 集，比 mis_follow_sustained 增幅。

## 10. 首个真实评测（2026-08-18，harness 的应用）

### 10.1 用 harness 验证模型的机制
`RolloutScorer` 每集出 §7.2 指标。验证分两层：**绝对**（SR/track_seconds/mis_follow/centering/reacquire_lock_wrong）+ **对比**（换 `--config/--ckpt/--language-mode` 三参跑不同臂，同 seed 配对，`compare_runs.py` 做 per-episode bootstrap CI）。换臂只改这三参，`policy_server.py` 全参数化。

- **H0 语言消融**：M0(config_v5,best.pt,neutral) vs M1(config_v5_nolang,nolang-best,none)，`eval_h0.sh` 一键。
- **H1/H2 模块消融**：换 `--ckpt` 到 w/o-EAR / w/o-IAR 模型。
- **资源**：VLM server 1 卡 ~11-12GB；CARLA 复用已有 server 不额外占卡；磁盘几乎零（`env.py` 不落帧，只出 KB 级 JSON）。批次大小只影响 wall-clock。
- **速度**：s2_period=1 ~1s/tick（socket+render bound）；`S2_PERIOD=6` 提速（staleness 训练支持）。长批次 detached 跑。

### 10.2 H0 闭环语言消融结果（Town05 held-out, 7 对配对 seed, s2_period=6, best.pt）
```
mis_follow_sustained:  M0 有语言=0.606±0.30  M1 无语言=0.923±0.08
Δ(M1−M0)=+0.317  95% CI [+0.14, +0.51]  ← 排除 0
H0 verdict: LANGUAGE LOAD-BEARING ✅
```
- **闭环 +32pp gap > 离线 +14pp** —— 误差累积**放大**语言依赖；离线证据（[[acot-uav-language-necessity]] 0.579 vs 0.715）首次在 **on-policy 闭环 + 未见地图**兑现，正是 experiment-design §7 要求、之前完全缺失的东西。
- **诚实边界**：两臂 **SR=0**（跟~20s 必丢=BC 分布漂移，与语言消歧正交，属 Stage-3 territory）；**reacquire_lock_wrong=1.0**（重捕获必锁错，predict-intercept 软肋）；N=7/单地图/MVP=**趋势性非定论**（§10 边界，强消融需 scale）。
- 输出：`carla_uav_tracking/rollout/eval_out/{m0,m1}.json`。

### 10.3 下一步评测
1. **Scale**：`EPISODES=30 SEED_BASE=... TOWN=... bash eval_h0.sh`（多地图×3seed）出统计定论。
2. **H1/H2**：训 w/o-EAR / w/o-IAR ckpt → 换 `--ckpt`（进行中）。
3. **治 SR=0**：Stage-3 RL（同 env 作 rollout 源）。

## 代码/接口索引

| 组件 | 关键代码位置 |
|---|---|
| 换入点（专家动作） | `recording/recorder.py:167`（`expert_action`）、`:188`（`step_drone`） |
| 无人机运动学 | `carla_uav/drone.py:73`（`KinematicDrone.step`，纯积分 spectator） |
| 场景/seeding | `scene/scene_manager.py:72`（`setup_episode`）、`:77`（seed）、`episode_language` |
| GT 状态 | `scene_manager.py:666`（`get_target_state`）、`:808`（`get_distractor_states` (D,6)） |
| 相机/在帧内 | `recorder.py:292`（`_create_attached_sensor`）、`:324`（`_point_in_frame`）、`drone.py:119`（`get_camera_transform`） |
| 在线 VLM 原语 | `acot_probe/backbones.py`（`build_backbone`/`encode`/`pool_box`）、`train/backbone_kv.py:43`（`_pool_grid`） |
| 推理参照 | `train/stage2.py:193`（`_evaluate`）、`:49`（`action_sample`）、`flow_matching.py:30`（`sample`） |
| 模块签名 | `train/ear.py:73`、`iar.py:47`、`dit.py:71`（DiT）、`dit.py:85`（TargetIDHead） |
| ckpt 格式 | `train/stage2.py:134`（`_save`，四 state_dict + `dims` 块，按 mis_follow 选 best） |
| 采样输入契约 | `train/dataset_stage2.py:8-20`（样本 dict）、`:287`（构造）、`:253`（proprio 5-D） |
