# 1009 · ReferTrack 微调与接入方案(设计稿,2026-10-09;未实施，待 codex 复审)

> 背景：候选基线组合为 OA-VAT(唯一的无人机主动跟踪方法，比较可复现部分)+ ReferTrack + OmTrackVLA(两者代码与权重完全开源)。本稿只写 ReferTrack。
> 代码依据：`external/referTrack`(MedlarTea/referTrack @80f8529,Apache-2.0,只读克隆)。下文 `M / D / T / A / B / C` 分别指 `model/referTrack.py`、`dataset/refer_datasets.py`、`train/train_refer.py`、`baseline/trained_agent_refer.py`、`tool/build_refer_jsonl.py`、`eval/cache_gridpool.py`。

## 1. ReferTrack 是什么(按代码)

| 部分 | 原实现 |
|---|---|
| 主干 | Qwen3-4B(36 层，隐层 2560),stage 2 **全量微调**(LoRA 代码存在但未暴露;M:177-213) |
| 视觉 | 冻结、离线预计算:DINOv3 ViT-S/16(384 维)+ SigLIP so400m-patch14-384(1152 维),补黑边成方形后缩放到 384²,逐 patch 拼成 1536 维;历史帧池化为 2×2 = 4 个粗 token,当前帧 8×8 = 64 个细 token(C:163-301) |
| 候选框 | **不画在图上**:每个候选 = `<obj_k>` 词元 + 归一化 xyxy(/384)框 token,最多 20 个(超过取面积最大的 20 个),**每帧随机打乱编号**,不给跟踪器 ID(M:481-576;D:144-151;A:611-621,677) |
| 指代输出 | 单个受约束 token:在 21 个槽位(`<obj_1..20>`、`<NO_EXIST>`)上做分类;目标不在 = `<NO_EXIST>`(M:677-682;D:188) |
| 历史 | 31 帧(10 Hz、3.1 s)+ 当前帧;每帧一个 TVBI token = 时间 + 相机视角(单视角时恒为 0)+ 上一帧所选框的 xyxy(M:82-129,647-675) |
| 航点输出 | 确定性 MLP 回归(`PlannerHead3L`,tanh × α):8 × [x, y, yaw],机体系累计位移，取自 9 步专家速度指令的积分(去掉第 5 步),时域 0.8 s(D:38-49);训练 α = (0.535, 0.535, 1.572),仿真推理写死 α_xy = 0.8(A:826-829) |
| 控制 | 每步(10 Hz)只执行 wp[1],换算成速度指令(A:879-883) |
| 损失 | L = 10 · L_nav(航点 MSE)+ L_cot^nav + L_cot^qa(21 槽位交叉熵);每步一批导航 + 一批 Refer-QA(T:256-268) |
| 优化 | AdamW;LLM lr 2e-5、头 2e-4;余弦 + 2% 预热;DeepSpeed ZeRO-2、bf16;每卡 16 + 16,16 卡，5 轮(发布的是第 3 轮第 20,000 步) |
| 闭环推理 | YOLO11x + ByteTrack(只保留 person 类)→ 候选目录 → 两次 LLM 前向(KV 复用),每步都查询 |
| 依赖 | Python 3.9、torch 2.4.0+cu118、transformers 4.57.3、peft、deepspeed;habitat-sim 0.3.1 只用于其原评测 |

## 2. 用法定位

| 用法 | 用到 ReferTrack 的哪部分 | 对应表格 |
|---|---|---|
| **A. 组件对照(只比 WHICH)** | 只取指代输出(选哪个候选 / 不在),WHERE 与 HOW 用共享规则 | 表 1(共享属性开局，统一交接) |
| **B. 完整系统(WHICH + WHERE + HOW)** | 指代 + 航点都用，航点直接驱动无人机 | 新增的完整系统对比表(与 OA-VAT 完整系统、Ours 完整系统并列) |

- ReferTrack 是唯一**原生用语言做开局**的外部基线(每帧都按语言从候选里选)。在用法 B 中它应使用自己的语言指代，不共享我们的属性开局;用法 A 中按表 1 规则共享属性开局(用开局选择初始化它的所选框历史)。
- 两种用法共用同一个微调后的模型。

## 3. 需要改动的地方(最小改动原则，每项都在论文附录列出)

| # | 原实现 | 我们的域 | 处理 | 性质 |
|---|---|---|---|---|
| 1 | 候选来自 YOLO11x + ByteTrack,只保留 person | 共享感知(真值投影的任务车辆 + 射线遮挡过滤 + 匈牙利跟踪器) | 用共享感知的候选替换检测器;仍按原实现每帧打乱编号、不给 ID | 必须(与所有基线同一感知) |
| 2 | 动作 3 维 [x, y, yaw],平面运动 | 无人机 4 维 (dx, dy, dz, dyaw) | 航点改为 8 × [x, y, z, yaw];planner 最后一层重新初始化(加载时丢掉 `planner.mlp.5`,否则形状不匹配会报错);α 按标签统计重定 | 必须 |
| 3 | 框归一化写死 /384 | 512² 画面 | 画面缩放到 384 后再送入(保持其预训练视觉统计),框坐标同步缩放 | 必须 |
| 4 | 训练样本按「人与机器人高度差 ≤ 0.3 m」过滤(D:105-106) | 无人机在 30 m 高空 | 不写 `human_pos / robot_pos` 字段，跳过该过滤 | 必须 |
| 5 | Refer-QA 用 SYNTH-PEDES(行人，仅限非商用) | 车辆 | 用我们的数据构造车辆 Refer-QA(§4 WP3) | 必须 |
| 6 | 指令是行人描述 | 车辆描述 + 距离要求 | 用每集原始指令(`attrs['language']`)原样输入 | 数据替换 |
| 7 | 专家 = A* + PID 跟随行人 | CARLA 专家飞行 | 用我们数据里的专家动作作标签 | 数据替换 |
| 8 | TVBI 的视角项恒为 0 | 相机俯视 −50°、高度变化 | **保持原样**(不额外注入俯仰 / 高度),只看像素 | 保真优先 |
| 9 | 训练与推理的历史对齐差一帧(子代理读代码发现的疑似问题，见 A:700-701 与 B:141-149) | — | 保持原实现，附录注明 | 保真优先 |
| 10 | 最多 20 个候选、按面积截断 | 我们最多约 6–9 个候选 | 无影响 | — |

已知的结构性劣势(不改，附录说明):细 token 每格约 48 px(384² 下),我们的车辆框在 384² 下约 8–105 px,小车可能不到一个 token;ReferTrack 主要靠框 token 与池化特征区分颜色和车型。

## 4. 工作包

### WP1 环境与权重(约 0.5 天)
- 单独建容器镜像:torch 2.4.0+cu118、transformers 4.57.3、peft ≥ 0.12、deepspeed ≥ 0.14、flash-attn(可选)。我们的策略镜像是 torch 2.6 / transformers 5.17,不兼容，必须分开。
- 下载:Qwen3-4B 权重(仓库里只有分词器和配置);ReferTrack 发布的 stage-2 `refertrack_qwen3_4b.pt` + `model_config.json` + stage-1 `refertrack_qwen3_4b_stage1.pt`(HuggingFace `hjyeee/ReferTrack-Qwen3-4B`);DINOv3 ViT-S/16(我们只有 ViT-B/16,需按 DINOv3 许可另取);SigLIP so400m-patch14-384。
- 冒烟：用其视频推理脚本 `eval_refer_video.sh` 在一段 CARLA 录像上零样本跑通，记录逐步耗时。

### WP2 训练数据转换(约 1.5 天)
- 数据(2026-10-09 用户决定):闭环保持 90°;`mvp_full_v5_fov90` 目标静止帧占比达 62%(26/64 集有 ≥ 60 s 的静止段;70° 的 `mvp_full_v5` 为 39%),改用**最小重生成的 90° 数据** `mvp_v5_fov90_moving`:v5 格式、种子 42–107 与城镇分布同原 v5,场景配置 `config/scenarios/mvp_moving.yaml` = `mvp.yaml` + 三个只作用于场景的开关(目标与干扰车忽略红绿灯、背景车在 30–200 m 环内生成、允许变道;背景车仍 80 辆);v6.1 的强制变道需 v6 标签模式，不使用。验收(运行前固定):每集运动帧 ≥ 50% 且无 ≥ 60 s 静止段;先 5 集冒烟再全量。训练 / 验证按集划分，种子不与 dev-B / test 重叠。
  - **10-09 更新(冒烟 1–4 之后,用户决定)**:改为快速运动学场景 `config/scenarios/mvp_moving_fast.yaml`(目标与干扰车 10–12 m/s、遵守红绿灯、红灯周期 6/2/2 s;专家无人机水平速度严格限 15 m/s,kp_xy 0.9–1.2;防卡死 `scene.kinematics.unjam`:静止 12 s 强制变道 → 20 s 挪走同车道前车,背景车静止 25 s 挪走(画面内 40 s),目标 40 s 内移动 < 5 m 则截断)。生成代码经 codex 三轮审查(R3 NO MAJOR ISSUES,`runs/datagen_moving/kin_patch/`)。Town01/02 背景车减为 40 辆(`mvp_moving_fast_small.yaml`,冒烟 smoke4 验证中)。
  - **验收改为 QC v2**(用户决定 A):未被截断、无 ≥ 40 s 静止段、运动帧 ≥ 35%(红灯等待是真实城市行为，只判异常卡死);`runs/datagen_moving/qc.py`。smoke3 按 v2 为 8/10(v1 为 5/10)。
  - **全量数据(10-09 启动)**:`/nvidia/hque/data/carla_data/mvp_v5_fov90_fast`(64 集,城镇分布同原 v5:Town01 19 / Town02 15 / Town04、05、10HD 各 10;不合格集在 `mvp_v5_fov90_fast_qcfail`,不得用于训练),编排器 `runs/datagen_moving/full_fast.py`。下文 `mvp_v5_fov90_moving` 均指此数据集。
  - **10-11 定稿**:64 集全部通过 QC v3(= v2 + 相机在建筑包围盒内的帧 ≤ 5 %;无人机为无碰撞运动学模型且未开 geometry_avoidance,会穿楼),清单 `mvp_v5_fov90_fast/manifest.json`(每集种子/城镇/运动比例/楼内比例/屏蔽比例)。均值运动帧 0.58,楼内帧 0.34 %。**训练加载必须打开 `data.frame_mask: train`**(train/dataset.py、train/dataset_stage2.py 已支持,缺省关闭;`carla_uav_tracking/recording/building_mask.py: train_keep_mask` = 挪车屏蔽 ∧ 楼内屏蔽,各 ±1 s,共屏蔽 1.35 % 帧;先截 cap 再屏蔽,保证锚帧都在上下文缓存里)。穿楼最严重的 3 集(355、84、78)已用同城镇新种子替换(358、359、401)。生成器修复与 codex 记录:`runs/datagen_moving/kin_patch/`。
  - **训练时必须屏蔽画面内挪车(用户决定 C)**:防卡死挪走画面内的车会造成一辆车在相邻两帧间消失。数据加载时对每集调用 `carla_uav_tracking/recording/unjam_mask.py: unjam_keep_mask(h5)`(画面内挪车前后各 1 s 置 False,smoke3 中约 2.2% 的帧);事件帧号 = h5 attr `kin_unjam` 中的 tick − 90(已在 smoke3 上用干扰车位置跳变逐一核对)。
- 与评测的分布差异：评测场景仍用默认配置(目标会等红灯),训练数据中的目标很少停;附录说明。
- 候选：在录制数据上**回放共享感知模块**(`acot/runtime/perception.py`),得到与闭环完全一致的候选集合与跟踪器 ID;目标标签 = 目标所在候选的 tid,目标不在候选中 → `<NO_EXIST>`。
- 航点标签：对专家动作 (dx, dy, dz, dyaw) 在机体系逐步积分，取 9 步、保留与原实现相同的 8 个下标 → 8 × [x, y, z, yaw]。
- 所选框历史：训练时用真值匹配的目标框(与原实现的教师强制一致)。
- 输出原格式的目录:`frames/ tracks/ jsonl/ vision_cache/`;jsonl 每条含 `images.forward`(≤ 31 帧)、`current.forward`、`instruction`、`actions`(9 × 4)、`track_bboxes.forward`(tid + xyxy@384)、`current_track_target_bbox`、`track_target_bbox_history`。
- 视觉特征预计算(原脚本 `precache_features.py`):每帧 4 × 1536 + 64 × 1536,fp16,约 0.2 MB / 帧;64 集 × 1,500 帧约 96k 帧，约 20 GB。
- 可选加量:DAgger 数据(在闭环中让微调后的模型跑、由专家标注，补 on-policy 状态，与我们 TAHRelM 的 DAgger 做法对称)。

### WP3 车辆 Refer-QA 数据(约 0.5 天)
- 从同一批录制中取「画面内有 ≥ 2 辆候选车」的帧，问题沿用原模板 `Please find <caption> in the video. Answer with object indexes.`,caption 用 `target_desc` 或干扰车的 `desc`(颜色 + 品牌 + 车型),标签为对应槽位;按原比例约 6% 生成「不在画面中」的询问(用本集未出现的车辆描述)→ `<NO_EXIST>`。
- 输出原格式 `images/ info.json val_indices.json vision_cache/`。

### WP4 模型改动(约 0.5 天)
- `action_dims` 3 → 4;α 改为 4 维(按训练标签各维最大累计位移定);损失按 xy / z / yaw 分项(保持 xy 与 yaw 的原权重,z 与 xy 同权);控制换算改为输出我们的 (dx, dy, dz, dyaw)。
- 加载 stage-2 权重时丢掉 `planner.mlp.5` 两个张量(重新初始化)。
- 框归一化与图像缩放(512 → 384)放在数据和推理两端，模型本身不改。
- 单元测试：标签积分正确性、槽位标签与候选一致、NO_EXIST 构造、加载后除最后一层外权重与发布版逐位一致。

### WP5 微调(约 0.5–1 天算力)
- 初始化：发布的 stage-2 权重(保留其在 EVT-Bench 上学到的指代与跟踪能力)。
- 配方：沿用原 stage 2(全量微调 LLM;LLM lr 2e-5、头 2e-4;余弦 + 2% 预热;ZeRO-2、bf16;开梯度检查点)。批大小按 8 × A40 调整(例如每卡 4–8 + 4–8)。
- 轮数与选轮规则**运行前固定**:按验证集上「指代槽位准确率 + 航点 L_nav」选轮(吸取 G13 教训：不要只看会饱和的单一准确率)。
- 算力：约 96k 导航样本，8 卡、每卡批 8 时约 1,500 步 / 轮;按每步 1–2 s 估计每轮约 0.5–1 小时，3–5 轮。训练期间要独占 8 张卡，需与 CARLA 实验错开。
- 备选(若显存或时间不够):LoRA r = 8(代码已有);作为偏离原配方记录在附录。

### WP6 接入闭环(约 1–1.5 天)
- 以独立服务运行(与 DAM4SAM 相同的服务模式;依赖不兼容，不能放进策略镜像)。
- 每步(10 Hz)发送：当前帧(缩放到 384)、共享感知的候选框(xyxy)、指令;服务维护 31 帧历史、所选框队列与 KV 缓存，返回槽位(或 NO_EXIST)与 8 × 4 航点。
- **用法 A(只比 WHICH)**:只在 S2 tick 读取槽位，映射回候选的跟踪器键，作为身份模块输出，经共享的承诺规则(门限 + 5 tick 滞回);WHERE / HOW 共享。conf = 21 槽位 softmax 的 top1 − top2。开局：按表 1 规则，用属性开局选择初始化它的所选框历史。
- **用法 B(完整系统)**:每步执行 wp[1] → (dx, dy, dz, dyaw),替换 EAR / DiT;不使用共享 WHERE;开局用它自己的语言指代。
- 记录每步耗时(视觉编码、两次 LLM 前向),进入 §5 的成本表。

### WP7 验证与评测(约 1 天 + 算力)
- 离线：留出 CARLA 验证集上的槽位准确率、NO_EXIST 精确率 / 召回率、航点误差;并报告**零样本**(未微调)结果作对照，说明域差距。
- 闭环冒烟(2 个种子)→ dev-B pilot(40 个种子)。
- 调参预算与其他基线相同：在 dev-B 上按运行前冻结的网格调参，例如推理 α_xy ∈ {训练值, 1.0×, 1.5×} × 用法 A 的 conf_tau ∈ {0.05, 0.2, 0.5};选择规则同 G12。
- 正式：test 种子、冻结的重复次数;用法 A 进表 1,用法 B 进完整系统对比表。

## 5. 公平性约束
- 训练数据与我们的模型同规格(`mvp_full_v5_fov90`,同划分);dev-B / test 种子不参与训练与选轮。
- 感知相同(共享候选与跟踪器);用法 B 中 ReferTrack 不使用真值世界坐标(它本来也不需要);同时我们的完整系统也要换成不依赖真值的 WHERE 定位(地面反投影，见 1008 讨论),否则完整系统对比对 ReferTrack 不公平。
- 所有对原实现的改动按 §3 列表写入附录;改动只限于域适配，不改其方法本身。

## 6. 风险
1. **数据量小**:约 96k 帧，而原实现训练约百万级样本。靠 stage-2 初始化缓解;若微调后仍明显欠拟合，可加 DAgger 或恢复 v6 数据生成(目前暂停)。
2. **小目标分辨率**:细 token 约 48 px,远处车辆不到一个 token。属结构性劣势，如实报告。
3. **算力冲突**:4B 全量微调需独占多卡，与 CARLA 闭环实验争用 GPU。
4. **延迟**:每步两次 4B 前向，按 10 Hz 运行在 A40 上可能超过 100 ms;同步仿真下不影响结果，但计入 §5 成本表与端侧评估。
5. **身份信号的来源不同**:它每帧重新按语言选择、自身没有显式身份记忆(只有所选框历史);在 look-alike 场景下可能频繁跳变，这正是对比要检验的点。

## 7. 工时与算力合计
- 人力约 5–6 个工作日(WP1 0.5、WP2 1.5、WP3 0.5、WP4 0.5、WP6 1–1.5、WP7 1)。
- 算力：特征预计算约 2–3 小时(单卡);微调约 0.5–1 天(8 卡);闭环评测与调参约 0.5–1 天(7 台 CARLA)。

## 8. 待决定
1. 微调方式：全量(忠于原配方，推荐)还是 LoRA(省算力)。
2. 是否加 DAgger 数据。
3. 完整系统对比是否与「我们的 WHERE 改为地面反投影」同时推进(建议同时，否则对比不公平)。
4. OmTrackVLA 的适配是否按同一模板并行(0.6B,工作量相近、算力更省)。
