# 1003 · 动态重指定(Dynamic Re-Targeting)技术方案(草案 v0)

> 2026-10-03 · 状态:**设计草案,未实现,未纳入 1004 冻结协议**。动机见对话记录(Sticky-ID 零假设危机 + "固定目标=语言只在第一帧起作用,不是真 VLA")。
> 前置发现:`acot-uav-sticky-id-null`(0 参数 Sticky-ID 闭环与 PGIF 统计打平,最长连续正确锁定上显著更好 p=0.012)、`1001_innovation-design.md` §附录A(`reanchor=lang` 已证伪:单帧语言在 look-alike 间仲裁精度约 0.25,时序 EMA 后 0.29,仍低于不纠偏基线 0.36;对照 GT-oracle 触发+重定位 0.72)。
> 代码引用以 2026-10-03 工作区为准,行号为近似(±5 行量级),实现前需重新核对。

---

## 0. 设计原则(分层解耦)

**联合(每 tick 重算)**:grounding 计算层——语言与当前帧 RGB 联合产生"当前指代是谁"的判断。这一层**现状已经是联合的**,不是要新建的能力:`TargetIDHead.forward(cand_feats, vlm_ctx, ...)`(`train/dit.py:85-114`)是无状态单帧打分器,`policy.py:663` 附近每个 S2 tick 都重新调用,`vlm_ctx` 也每 tick 重新由 Qwen3-VL 编码(`policy.py:602-603` 附近)。**瓶颈不是架构,是语言内容从未变化**(`obs.language` 是 episode 级常量)和**下游身份记忆从冷启动之后就不读 grounding 结果**。

**解耦(身份记忆主干不吃语言)**:`TAHRelM` 的 7 维特征(`cos, relp_x, relp_y, depth, rank, dmot, msoft`,`train/tah.py:121` 附近)保持不变,不注入语言特征——这是 B0→B0.1 过拟合教训的直接应用(绝对/语言相关信息进身份特征 → 车池窄(64+23 ep)过拟合)。语言只能以**离散事件**的方式,通过一个新的"覆盖(override)"接口作用于 PGIF 的证据状态和 TAHRelM 的记忆内容,不作为逐帧输入特征。

一句话:**语言决定"现在该信任谁",证据滤波器决定"信任之后怎么抗漂移"——这两件事要分开建模,只在事件边界处握手。**

---

## 1. 核心抽象:三类身份信号,不是两类

当前 PGIF(1001 §3)只区分"记忆自己选的"(`s^M_t`)和"证据滤波后的"(`ŝ_t`),背后隐含假设是"身份切换 = 确认偏差误差,应被抑制"。动态重指定引入第三类事件:

| 类型 | 触发方式 | 应该发生什么 | 现有机制 |
|---|---|---|---|
| **隐式漂移**(自更新误差自相关) | 记忆自己选错,无外部信号 | 被 PGIF 回火证据抑制,门控拒绝写入 | PGIF 主逻辑(命题1/2) |
| **检测到漂移后的纠偏** | 置信度持续低 / 外部检测器判定当前≠目标 | 重置记忆,用可信信号(GT/外观锚/语言)重新指认**原目标** | `reanchor={oracle,anchor,lang}`(已有,`lang` 已证伪) |
| **★合法重指定(新)** | 外部显式新指令到达(新的语言字符串) | **立即**、**无条件**(不经证据累积)把身份状态切到新指代对象,旧目标降级 | **不存在,待设计** |

第三类与前两类的本质区别:**它不是"误差",不需要被证据说服,只需要被正确理解**。设计上必须和前两类严格隔离,否则会把"合法切换"误当成"该抑制的漂移"(PGIF 的全部机理就是抑制切换)。

---

## 2. 接口设计

### 2.1 指令事件通道(新)

```
InstructionEvent = {tick: int, language: str, event_id: int}
```

- **数据来源**:
  - 闭环 eval / 数据生成:`run_rollout.py` 按种子预生成一个事件时间表(`retarget_schedule.json`:`{seed: [{tick, language, target_actor_id(仅供离线判分,不进决策)}]}`),复现性与 1004 的种子划分一致。
  - 真实部署(未来):用户侧新指令到达,走同一接口,`target_actor_id` 未知(这正是待评测的能力)。
- **env/policy 侧**:`carla_uav_tracking/rollout/env.py` 的 `Observation.language` 字段(现为 episode 级固定字符串,约 `env.py:37`)改为运行时可变,每 tick 查一次"是否有待处理事件",有则替换 `obs.language` 并置位 `obs.instruction_changed = True`;否则沿用上一条指令(**不是**原始冷启动指令,是"当前有效指令",允许连续多次重指定)。
- **向后兼容**:无事件时行为与现状完全一致(退化为单一 episode 级指令)。

### 2.2 SceneManager 重指定 API(新)

```python
# scene/scene_manager.py
def retarget(self, new_target_actor_id: int) -> bool:
    """把 self._target 切换为场景中已存在的另一 actor;原 target 降级为 distractor。
    返回是否成功(actor 不存在/已销毁则 False)。"""
```

- 下游不需要改:`candidates.py::build_candidates`(`:44`)、`is_target` 标注(`:79`)、scorer 全部是**车辆无关**的、按"当前 `self._target`"动态算的(`get_target_state()` 返回当前指针)。这是 Explore 阶段已确认的好消息。
- 新增一条日志字段:`target_switch_log: [{tick, old_actor_id, new_actor_id, reason: "scripted_event"}]`,供 scorer 和离线分析区分"目标客观上换了"的时刻——这是新协议能不能打分的前提,没有这个字段无法定义"事件后多久重新锁定算成功"。
- **用在数据生成**(`scripts/generate_batch.py` / `recording/recorder.py`):录制时按计划在若干 tick 调用 `retarget()`,语言生成器(`language_generator.py::build_instruction`)在同一 tick 针对新目标生成新指代表达式,两者必须同步触发,否则语言标注和场景真值错位(复刻过去 FOV 错位的教训,务必写进生成器的单元测试)。

### 2.3 PGIF 覆盖接口(新,`rollout/asd.py`)

现有 `ASD._filter_step`(`:190` 附近)只有一条路径(回火证据累积)。新增:

```python
def override(self, new_slot: int, confidence_margin: float):
    """硬重置:不经证据累积,直接把状态切到 new_slot。
    confidence_margin 必须 > override_tau(新超参,独立于 PGIF 原有 conf_tau),
    否则拒绝,本 tick 退回正常 step()(防止歧义指令被当真)。"""
    if confidence_margin < self.override_tau:
        return {"accepted": False}
    self.L = {new_slot: self.override_prior}   # 清空所有 tracklet 的回火证据,只保留新目标
    self.conf_hist = []                         # 秩置信窗口清空(新目标没有历史)
    return {"accepted": True, "fire": True, "slot": new_slot, "kind": "override"}
```

- 触发条件:`obs.instruction_changed == True` **且** 当 tick 的 grounding margin(`lang_logits` top1−top2,复用 `_cold_vote` 同款计算,`policy.py:381` 附近)超过 `override_tau`。margin 不够则**推迟不丢弃**——下个 tick 继续尝试 override(语言内容不变,candidate pool 可能变化使 margin 回升),最多等待 `override_patience` 个 tick 后放弃并记录 `override_failed`。
- `policy.py` 侧:在现有门控分支(`:793-843` 附近,`_asd_gate_memory` 调用处)之前插入 override 检查,override 命中时**跳过** PGIF 正常 `step()`,同时触发 TAHRelM 记忆重播种(复用冷启动同款逻辑:`crop of new_slot` 重写 `app/pos/vel`,等价于把新目标当成一次新的冷启动,而非增量更新)。
- 这条路径与 `reanchor=*` 系列**互斥且语义不同**:`reanchor` 是"检测到对原目标跟丢,用信号找回原目标";`override` 是"目标本身变了,原目标不再是目标"。两者共用"用语言重新 grounding"这个计算,但触发条件和后续动作完全不同,不要合并实现,否则无法做消融区分哪个机制起作用。

### 2.4 命题改写(供最终论文用,现在先占位)

- **命题 1/2 不变**,仍然是"隐式漂移应被回火证据抑制"的主张,适用范围收窄为"无显式事件时"。
- **新增命题 3(待验证)**:合法重指定应该**绕过**证据累积机制(即时切换),且该切换不应被 PGIF 的抗漂移机制误吸收(override 后的新目标应立即获得与冷启动等同的证据地位,而不是要重新攒够 τ≈50 tick 才能反超旧证据——如果走正常 `step()` 而非硬 `override()`,大概率切不过去,这是必须做 override 而非"再触发一次 reanchor=lang"的根本原因)。
- 这条命题如果成立,直接证伪"Sticky-ID 在任何协议下都和 PGIF 打平"——这是新协议下 Sticky-ID 必然崩溃的理论依据(见 §4)。

---

## 3. 数据需求

### 3.1 现状(阻碍)

- `recording/recorder.py` 把语言写成 **episode 级 HDF5 attrs**(单值,约 `:87`),`scene_manager.py` 的 `self.episode_language`(`:61`)同理。所有训练脚本(`qwen_lora_grounding.py`、`reid_resolution_probe.py`、`backbone_kv.py`、`audit_bearing.py`、`bc_center_train.py`)都按 `f.attrs["language"]` 单值读取。**schema 里没有"第二条指令"的位置**,这不是分布外,是格式不支持。

### 3.2 最小改动 schema

把 `attrs["language"]` 换成一个变长数组字段(HDF5 支持变长字符串数组),语义为分段:

```
language_segments = [
  {"start_tick": 0,   "end_tick": 83,  "text": "...", "target_actor_id": 7},
  {"start_tick": 84,  "end_tick": 201, "text": "...", "target_actor_id": 12},
  ...
]
```

- 单段退化为现状格式,**向后兼容**:所有现有训练脚本改为"取 segment[0]"即可不碰其他逻辑跑通,升级是渐进式的。
- 读取端新增一个 `episode_language(tick)` 查询函数,替代直接 `attrs["language"]` 字面量读取,训练脚本逐个切换(工作量是分散的小改动,不是一次性重写)。

### 3.3 新数据生成

- 复用 `retarget()` + 同步语言生成(§2.2),在 `scripts/generate_batch.py` 加一个"重指定概率"参数:每个 episode 以概率 p_retarget(建议先验 0.3~0.5,不是全量)在随机 tick(需有冷却期,不紧跟冷启动、不紧跟上一次切换)插入一次事件,目标车辆从**当前可见 distractor 中按外观判别度抽样**(太容易/太难都要覆盖,参考 1003/1004 现有的 N=5 密度、off-center 分层做法)。
- **先不要大批量生成**:按 §5 的分阶段建议,先做一个几十集的小样本探测,确认 grounding 在非冷启动帧上的可行性,再决定是否值得投入整批数据生成(对照 `acot-uav-v6-datagen-status`:上一轮 v6 数据生成还处于"未过两轮验收冒烟"状态,新协议不应该叠加在一个还没验收通过的数据管线上)。

---

## 4. 评测协议(新增,不替换 1004)

### 4.1 新指标

| 指标 | 定义 | 目的 |
|---|---|---|
| 切换后重新锁定延迟 | 事件 tick 到 scorer 判定 `scorer_committed_key` 命中新目标的 tick 差 | 直接量化 override 机制的有效性 |
| 伪切换率 | 没有事件发生时,承诺身份切换到非原目标的次数/集 | 这是 PGIF 抗漂移性质的直接检验;override 存在不能让这个指标变差 |
| 旧目标回跳率 | 切换后 K 秒内,承诺身份短暂跳回旧目标的比例 | 检验 override 的硬重置是否彻底(是否残留旧证据) |
| 新/旧协议正确锁定比例对比 | 同一批 episode,事件概率=0 时应与 1004 现有协议结果一致 | 向后兼容性检验,防止新协议本身引入偏差 |

### 4.2 Sticky-ID 必须崩溃检验(协议有效性的充分条件)

这是**判断新协议是否值得做**的关键检验,先于任何正式实验:在新协议下重跑 `--tid-head sticky`(`acot-uav-sticky-id-null`里的 0 参数基线),预期:事件发生后正确锁定比例应断崖下跌(因为 sticky 定义就是"永不切换")。**如果 Sticky-ID 在新协议下依然不输给任何方法,说明协议设计仍有漏洞(比如事件太稀疏、太容易、或者 scorer 口径又把问题藏起来了),必须先堵住再往下做**,这是復用 1004 "按 G 门槛验收再推进"的同一套纪律。

### 4.3 与 1004 的关系

- **不修改 1004 已冻结的设计**。1004 的 T0/T1 按原计划推进,这是独立的探索分支,数据和协议都新建,不复用 1004 的 test 种子段(95000+),避免任何交叉污染未来的"从未跑过的 test 种子"承诺。
- 如果 §5 的探测证明这条路可行且有论文价值,**何时并入正文**是后续决策,不在本设计文档里预设。

---

## 5. 分阶段建议(由易到难,每阶段都有明确的"继续/停止"判据)

| 阶段 | 内容 | 成本 | 继续判据 |
|---|---|---|---|
| **S0 探测**(建议先做) | 离线:用现有 ckpt(`TargetIDHead`/Qwen,不重训),在已有 episode 数据的**非首帧**(比如 episode 中段)喂入该 episode 本身语言描述 + 当前帧候选,测 grounding 精度(vs 首帧);再测"把语言换成描述另一个可见 distractor"时的精度(模拟中途换指令,但不需要新数据,只需在现有帧上换文本提示) | 1人日,0 新 rollout,0 新训练 | 若非首帧精度显著低于首帧(模型只认识冷启动场景分布)→ 定位为分布问题,进 S2 前必须先有训练数据;若接近 → 瓶颈是"从未被用过"而非"做不到",可以先接口化(S1)再评估是否要训练 |
| **S1 接口化**(dev-only) | 实现 §2 的全部接口(`retarget`、事件通道、`ASD.override`),但先用 **GT target_actor_id 直接驱动 override**(不经语言,相当于 `reanchor=oracle` 的"主动版"),闭环验证协议本身(§4.2 Sticky-ID 崩溃检验)、PGIF 覆盖后是否污染正常抗漂移能力(伪切换率不能变差) | 数日工程,复用现有 ckpt,n≈20 dev 种子 | 若 GT 驱动下协议跑不通(切换延迟异常大、PGIF 主逻辑被破坏)→ 说明是 override 机制设计问题,先修,不碰语言 |
| **S2 接入语言** | 用真实 `TargetIDHead` margin 驱动 override(非 GT),复用 S0 的精度结论设定 `override_tau`、`override_patience` | 数日 | 若语言驱动版本相对 GT 驱动版本的差距过大(证明语言仍是瓶颈,复现 `reanchor=lang` 的失败模式)→ 停止,把发现写成负结果,正文维持 1004 现有定位 |
| **S3 数据与训练** | 仅当 S0/S2 显示"分布缺失"是主因时,才投入新数据生成(§3.3)和训练 | 1004 量级(数据+重训+闭环) | 由 S0/S2 的量化结果决定是否值得,不预先承诺 |

**不建议跳过 S0 直接做数据生成或重训**——这是本方案和"直接扩大 reanchor=lang 实验规模"的本质区别:先用近乎零成本的离线探测把"语言仲裁精度不够"这个已知瓶颈在新场景下的严重程度量出来,再决定要不要承担数据+训练的代价。

---

## 6. 开放风险

- **R1** `override_tau` 的设定本身是一个新的人工超参,如果设得不合理,会重新引入"PGIF 本来 0 调参超参"这个卖点的瑕疵——需要和 λ/W 一样,说明其先验设定方式、不在 test 种子上调。
- **R2** 歧义指令(语言本身无法唯一确定新目标,比如场景里两辆同色车)在 override 协议下如何处理,目前设计是"拒绝,等待",但如果长期拒绝,算不算失败需要在指标里明确(不能悄悄计为"未发生事件")。
- **R3** 与 1004 §12 的 `gate_commit`/`gate_memory` 开关存在概念重叠(都是"身份状态更新 vs 承诺切换"的解耦),实现时应复用同一套底层记账,不要两套并行的状态机。
- **R4** 当前 `acot-uav-v6-datagen-status` 显示数据生成管线本身还没过冒烟验收——S3 阶段如果真的要做,排期上要晚于 v6 数据生成管线修复。

---

## S0 探测结果(2026-10-03)

**方法**:两条证据链,均复用已有产物,零/低新增算力。

1. **尝试路径(受阻)**:直接用部署头(`train/dit.py::TargetIDHead` + `stage2_best.pt`)在非首帧上测,复用 `train/grounding_probe.py` 的加载方式,新写 `train/grounding_probe_midepisode.py`(按 episode 内经过时间分桶:0-30/30-150/150-400/400+ 帧,对应约 0-3s/3-15s/15-40s/40s+)。**阻塞**:该路径依赖的 Stage-2 多层 VLM 特征缓存 `runs/ctx_cache_ml_v5` 当前**不存在于磁盘**(`find runs -iname "*ctx_cache*"` 零结果),需要对约 9-10 个 val episode(~4500-5000 帧)重新跑 Qwen3-VL-4B 前向重建缓存才能用这条路径,成本从"离线零成本探测"升级为"数小时级重算",**本轮未做**,脚本已就绪,留作需要时的下一步。
2. **替代路径(已出结果)**:复用 `runs/qwen_lora/paired_eval.json`(Stage-0 LoRA grounding 研究的既有产物,`train/qwen_lora_grounding.py`,同样基于 Qwen3-VL-4B layer-24,但接的是另一个更大容量的 bilinear 头,训练/评测帧本身就按 stride 抽样覆盖整个 episode,非只首帧)。按帧在 episode 内的绝对位置分桶,比较"正确语言"(test split)与"替换成另一集目标描述的语言"(test_langswap split,同一组帧)的逐帧准确率差距:

| 桶(帧位置,~10fps) | n | 正确语言 acc | 替换语言 acc | **语言贡献的 gap** | 仅 look-alike 子集 gap |
|---|---|---|---|---|---|
| 0-60(冷启动,≤6s) | 67 | 0.358 | 0.149 | **+0.209** | +0.217 |
| 60-300(6-30s) | 266 | 0.541 | 0.222 | **+0.320** | +0.371 |
| 300-700(30-70s) | 427 | 0.482 | 0.260 | **+0.222** | +0.248 |
| 700+(70s+) | 740 | 0.714 | 0.419 | **+0.295** | +0.304 |

(r16 = 训练过 LoRA 的头;r0 frozen-probe 头模式相同但幅度更小,见原始 json)

**读法**:language 相对于"语言内容不对"的边际贡献(gap)在**全程所有分桶上都显著为正,且非冷启动分桶(+0.22~+0.32)不低于冷启动分桶(+0.21)**——look-alike 子集(真正相关的难例)同样如此。这说明:**语言的可区分信号不会因为"不是第一帧"而系统性衰减**;原始准确率随位置波动(前一版 bucket-only 分析显示非单调)更可能来自取景/遮挡/距离等几何因素(与 1001 的语言必要性墙、off-center 分析一致),而不是"语言机制只在冷启动生效"。

**限制(务必带着看)**:
- 测的是"训练过程中用过的 Stage-0 实验头"(全量 Qwen4B+LoRA+bilinear),不是部署头(轻量 xattn `TargetIDHead`,容量小得多,见 §1 对接口无状态性的分析)——两者容量差异可能导致部署头在同样设定下表现更弱,**这条证据不能替代直接测部署头**。
- "替换语言"是换成**另一集的目标描述**(该目标本不在当前帧内),测的是"语言是否携带真实可用信息",不是"能否在当前帧内精确指向一个新的、确实可见的 distractor"——后者才是中途重指定真正需要的能力,**本轮未直接测**,因为数据里没有 distractor 级语言标注(§3.1 已确认)。

**结论与下一步判据**:按 §5 的判据,"非首帧精度显著低于首帧"这一否决条件**未成立**(反而持平或更好)——语言仲裁瓶颈(`reanchor=lang` 证伪的根因)更可能是"区分 look-alike 本身的绝对精度不够"(look-alike 子集 acc 普遍 0.17-0.67,远未到可靠水平),而不是"位置相关的分布缺失"。**建议跳过"是否值得做 S1"的犹豫,直接进入 S1(GT 驱动 override,不依赖语言精度)**——语言精度问题本就该放到 S2 单独面对,S1 的协议有效性验证(Sticky-ID 必须崩溃检验)不依赖这条结论。若后续需要精确量化部署头本身的非首帧精度,需先重建 `ctx_cache_ml_v5`(数小时级,`train/backbone_kv.py`)。

---

## S1 实现状态(2026-10-03)

**已落地**(GT 驱动,未接语言,未跑闭环):

| 接口 | 文件:位置 | 状态 |
|---|---|---|
| `SceneManager.retarget(k, tick)` | `carla_uav_tracking/scene/scene_manager.py` | 新增,单元验证(纯 Python mock,无需 CARLA):slot k 与 `self._target` 互换,其余 slot 不受影响 |
| `ASD.override(new_actor_id, prior, t)` | `carla_uav_tracking/rollout/asd.py` | 新增,硬重置 `belief`/`conf_hist`;**已用独立脚本验证**(见下)行为符合设计:override 后旧目标的"习惯性"选择被立刻拒绝(gate 回新目标),新目标被连续选中时不再触发 |
| 事件通道 `Observation.retarget_event` | `env.py`(torch-free 定义)+ `policy.py`(torch 侧独立定义,两容器不共享 import)+ `remote_policy.py`/`policy_server.py`(socket 透传) | 新增字段,默认 `False`,向后兼容 |
| 事件触发(S1:单次、GT 驱动) | `env.py::StudentRolloutEnv._fire_retarget`(`--retarget-tick`,`run_rollout.py` 新增 CLI) | 在指定 tick,挑一个当前可见的 distractor(优先 `similar`)调用 `retarget`;找不到可见 distractor 则本集不触发(不重试,S1 从简) |
| Policy 侧 override 消费 | `policy.py::act` TAHRelM 分支(`--event-override`,`policy_server.py` 新增 CLI) | 新增分支,位于 `gallery_seed=="committed"` 内、**优先于**冷启动/`reanchor` 分支:重置 `_tah_mem`、`_pred_slot=cset.true_idx`、调用 `asd.override()` |

**验证方式**:语法检查(`ast.parse` 全部通过)+ 两段独立逻辑单测(`ASD.override` 用真实 `asd.py` 跑通,`retarget` 用纯 Python mock 跑通,均不依赖 CARLA/GPU)。**未跑 CARLA 闭环**——`SceneManager.retarget` 依赖真实 actor/world,`Policy.act` 的 override 分支依赖 TAHRelM ckpt 前向,这两处目前只验证了独立于彼此的局部正确性,没有端到端跑过一集。

**S1 还未做(下一步,闭环验证本身)**:
1. dev 种子(建议先 n≈5 冒烟)上跑通 `run_rollout.py --retarget-tick 500` + `policy_server.py --tid-head tah --gallery-seed committed --asd filter --event-override`,确认不报错、`summary["retarget_log"]` 非空、`[event-override]` 日志行打印次数与 `retarget_log` 条数一致。
2. **Sticky-ID 必须崩溃检验**(§4.2):同样种子跑 `--tid-head sticky`(已存在,未改动——sticky 分支不读 `obs.retarget_event`,设计上就不该反应),确认事件之后正确锁定比例断崖下跌。
3. **伪切换率检验**(§4.1):无事件(`--retarget-tick` 不传)时,Ours(event_override 开)相对现有 PGIF 基线的伪切换率、正确锁定比例应该**不变差**——即 override 分支完全不应该在没有事件的 tick 上被触发(代码上 `getattr(obs, "retarget_event", False)` 默认 `False`,逻辑上不可能误触发,但要用真实闭环数据确认没有引入意外交互,比如 `self._cold_active = False` 这一行在从未触发过 override 的路径上完全不会执行,因为被 `if` 条件短路)。
4. 跑完后用 ASD 的 `_log()`/`override` 的 jsonl 输出(`ASD_LOG` 环境变量)+ `summary["retarget_log"]` 做"切换后重新锁定延迟"的离线分析(找 `true_slot==pred_slot` 在事件 tick 之后首次连续命中的 tick 差)。

---

## 变更记录
- v0.2(2026-10-03):S1 核心接口实现完成(`retarget`/`override`/事件通道/policy 消费分支),逻辑单测通过,闭环验证待跑。
- v0.1(2026-10-03):补 S0 探测结果(替代路径已出,部署头直测受阻于缺失的 ctx cache)。
- v0(2026-10-03):初稿,未实现,未审计。
