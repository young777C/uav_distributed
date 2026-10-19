verdict: 需要补充要求后再生成

**MUST**
- **先冻结“90° 统一口径”和配置来源**。问题: 现在 recorder 读 `output.fov`，但 `postprocess.py` 仍有 90° 默认，单独调用会绕过配置。证据: `default.yaml:177-181`、`recorder.py:40-42,297-303`、`postprocess.py:23,97-102`、`generate_batch.py:52-54`。改法: h5 attrs 写入 `fov/rgb_resolution/camera_pitch/vehicle_size_m/box_min/max/config_snapshot`，postprocess 必须优先读 h5 attrs，缺失则报错而不是静默默认。验收: 任取 20 集，用 attrs 投影目标/干扰框，与 `annotation/*`、`distractors/bbox` 逐帧一致，误差 <1 px。

- **不要只改 FOV 就直接复用旧 split**。问题: `config_v5.yaml` 指向不存在的 split，训练会回退到文件名切分，Town05/Town10HD 泄入训练。证据: `config_v5.yaml:1-8`、`dataset.py:25-39`、`1003_data-check.md:137-141`。改法: 生成后立即跑/修 `train/make_split.py`，产出实际存在的 `train/splits/mvp_full_v5_90_split.json`，并把 `config_v5*` 指向它。若论文要 Town03 未见城镇，训练集不应含 Town03；Town05 若仍做 test，则必须留出或只称“seen-town test”。验收: `_select_episodes` 在 train/val/test 返回非空且 town 集合符合预注册；缺 manifest 时直接 fail。

- **补足重捕获/遮挡配额，否则数据不支撑论文杀手场景**。问题: 当前全数据“≥3s 不可见后重新入画且有相似车”只有 36 次，`occlusion_biased` 全 False。证据: `1003_data-check.md:17,107-118,122-127,247-253`；`default.yaml:158-168` 默认 occlusion off；`mvp.yaml:6-13` 每集仅 2 次 loss。改法: 90° 版至少分层生成: base 复刻原 v5 + occlusion/reacq-heavy 子集；开启 `occlusion.enabled/bias_fraction` 或新增 scenario，并把 `loss_events` 提升到可验收的事件数。验收: 训练前 QC 显示 ≥3s 重入画+相似车事件不少于数百级，结构遮挡 episode 不为 0，每城镇/策略有分布表。

- **新增真实关联协议所需标签**。问题: 当前候选 `idx` 是持久 actor 编号，PGIF/Sticky 可直接跨出画继承身份，不等价真实 tracker。证据: `candidates.py:29-34,63-80`、`policy.py:668-678`、`metrics.py:29-31`、`1001_innovation-design.md:48,230`。改法: h5 同时写 `actor_id` 和按规则断开的 `tracklet_id`，断开规则至少包括 off-screen 连续阈值、结构遮挡阈值、重新入画事件编号；每个 target/distractor 都要有 `visible/off_screen/occ_structural/tracklet_id`。验收: 离线重放时 sticky actor-id 与断开 tracklet-id 得到不同结果；重入画后目标 tracklet id 改变。

- **闭环候选必须接遮挡或在论文中降格为理想检测设定**。问题: 闭环候选只看在画面内，不做遮挡，被挡住的车仍进入候选。证据: `1003_data-check.md:192-195`；`scene/occlusion.py:11-19` 已有 detector 但未接入；`env.py:169-180` 只算 `target_present`。改法: 评测协议若要“遮挡后重捕获”，闭环 candidate/scorer 必须记录并可选择过滤 `occ_structural>=阈值` 的候选。验收: 遮挡帧 candidate 数、目标 visible/occluded 状态、reacquire attempts 在日志中可复现。

- **TAHRelM 缓存和 DAgger 要重建，不只重建 h5**。问题: T0 明确要重训 TAHRelM，旧缓存离线部分受 FOV 错位，DAgger 是旧 TAH 策略收集。证据: `1003_data-check.md:145-154,277-279`、`tah_cache.py:34-38,50-61`、`tah_train.py:71-75,90-92`、`policy.py:145-151,764-768`。改法: 90° h5 生成后重建 `tah_cache.pt`；若最终部署策略或候选可见性协议改了，必须重收 TAH_DAGGER 数据。验收: cache 中 ≥3s 目标缺席段不再为 0，训练 ckpt 写入数据哈希/命令/seed/split。

**SHOULD**
- **分布以 Stage-2 原始 v5/闭环相机为锚，而不是 70° 版**。Stage-2 不重训，所以图像几何应保持 512²、90°、pitch -50、相似车/距离档尽量接近原始 v5；TAHRelM 则用 DAgger 补闭环 off-center 分布。证据: `1003_data-check.md:218-245`、`env.py:62-96`、`scene_manager.py:880-946`。

- **清理所有写死常量入口**。`qwen_lora_grounding.py` 写死 `W=H=512; FOV=90`，可接受但应读 config/attrs；`labels.py` 还写旧 336 默认，容易误用。证据: `qwen_lora_grounding.py:44-65`、`labels.py:24-35`。验收: 全仓 `rg "FOV|336|512|fov=90"` 后，生产路径无未解释默认。

- **结果和数据都写可追溯元数据**。当前 recorder attrs 只有语言/镇/seed 等，没有 git/config/FOV；1004 已要求结果 json 写 git/dirty diff/命令。证据: `recorder.py:67-82,414-416`、`1004_ablation_design.md:38-46`。改法: attrs 加 `git_commit, dirty_diff_hash, command, docker image, carla_version, config_snapshot, scenario_file, generator_seed_base, episode_id`。验收: 任一 h5 可独立复现生成配置。

- **QC 从“目视+形状”升级为自动门槛**。旧 spec 只要求数量、shape、目视；`qc_report.py` 投影默认 90 且不读 attrs。证据: `904_mvp-regen-spec.md:53-56`、`qc_report.py:28-42,49-89`。验收项: 投影对齐、bbox 像素尺寸分布、可见/出画/遮挡占比、重入画计数、similar 共视、黑帧/NaN、每城镇集数、split 泄漏、cache 覆盖率。

- **数据保护必须作为交付条件**。证据: 旧数据被清理，spec 已写保护要求 `904_mvp-regen-spec.md:45-48`。改法: 生成到版本化目录如 `mvp_full_v5_90_202610xx`，只读归档/校验和/清理白名单，不要覆盖唯一副本。验收: manifest + sha256 + 只读备份存在。

**NICE**
- 更新 `727_dataset-card.md`，旧卡仍写 336、MVP v1、摩托/自行车目标，已过时。证据: `727_dataset-card.md:18-24,30-37`、`1003_data-check.md:269-272`。
- 给 `run_rollout.py` 暴露 fov/resolution 参数，而不是硬写 `(512,512)` 和 env 默认 90。证据: `run_rollout.py:121-124`、`env.py:62-65`。
- 新增小规模 smoke: 每城镇 1 集先生成，跑 QC、TAH cache、一次闭环候选可视化，再批量生成。

**其他风险**
- 当前 worktree 很脏，且大量结果文件未跟踪；若现在生成或评测，第三方无法还原代码状态。
- `run_resilient.sh` 用 `.attempted` 只记录尝试编号，跳过/崩溃 episode 可能造成编号洞和城镇配额偏移；最终 manifest 应按 attrs 统计，而不是按 episode id 假设。
- `target_visible=True` 的专家录制与闭环 deployable 控制分布不同；离线 h5 不能替代 on-policy DAgger。