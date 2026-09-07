# 数据验收标准 (训练侧 ← 数据侧 的合同)

> 目的:让**训练侧评审 agent** 无需人工、按客观判据决定一批样本"能不能拿去训模型"。
> 通过后才交人工看,人工 OK 才全量 + 训练。
>
> **可执行合同 = `iter/review_sample.py`**(阈值以脚本里的 `CRITERIA` 为唯一真源;本文档只解释)。
> 评审 agent 跑:
> ```
> python -m iter.review_sample --dir <样本目录> --config train/config.yaml \
>        --out iter/reviews/r<轮>.json --round <轮>
> ```
> 退出码:`0`=APPROVED,`2`=CHANGES_REQUESTED,`1`=样本不足/损坏无法评审。

## 硬门禁 (任一不过 → 直接打回,不看软指标)

| 检查 | 要求 | 为什么 |
|---|---|---|
| `integrity` | 每集含完整 schema:`rgb(T,336,336,3) uint8`、`state/{cam_*,occ_raycast,...}`、`action/{dx,dy,dz,dyaw}`、`target/{tx..tyaw}`、`distractors/positions(T,D,6)` | 缺字段/形状错 → dataset 加载即崩 |
| `integrity` | `state/action/target` 无 NaN/Inf | 一个 NaN 就污染 loss |
| `frames` | 每集 ≥ `max_horizon_s*fps + 10`(=6*10+10=70 帧) | 短于波点视界无法构造监督目标 |
| 样本量 | ≥ `MIN_EPISODES`(8 集) | 少于此 leak probe 拟合不稳、无法评审 |

## 软门禁 (对照阈值给 pass / waived / fail)

| 指标 | pass | waived(可接受) | fail | 方向 | 含义 |
|---|---|---|---|---|---|
| `leak_floor` | ≥0.60 | 0.40–0.60 | <0.40 | 越高越好 | position-only MLP 的 mis_follow;低=几何直接定位目标→语言不吃劲(基准泄漏) |
| `covis` | ≥0.40 | 0.25–0.40 | <0.25 | 越高越好 | 目标在画面内且≥1同类干扰共视的帧占比;逼模型消歧 |
| `offscreen` | 0.05–0.45 | — | 区间外 | 带状 | 目标出画帧占比;要有(练 recovery)但不能过多(否则不可跟) |
| `recovery` | ≥0.20 | 0.08–0.20 | <0.08 | 越高越好 | 含"≥3s 丢失后重新捕获"片段的集占比(predict-intercept 技能) |
| `jitter` | ≤1.0 | 1.0–2.5 | >2.5 | 越低越好 | 遮挡/出画期间相机 yaw **jerk**(\|Δ²yaw\|,度/帧²);用 jerk 而非速率,平滑快扫不会被误判,只抓反复变向的抖动 |

**判定**:硬门禁全过 **且** 无软 `fail`(允许 `waived`)→ `APPROVED`;否则 `CHANGES_REQUESTED`,
`changes_requested[]` 会列出每个不过项 + 当前值 + 目标,数据侧据此改 `config/代码` 后重生成。

## 闭环协议 (双 agent,文件驱动,不限轮次直到通过或收敛)

```
数据侧: run_resilient.sh 生成小样本 (12 集 → /data/mvp_review_r<N>, 不碰 /data/mvp)
        → log_iter --type data_version 记账本
        ▼
训练侧评审 agent: python -m iter.review_sample ... → iter/reviews/r<N>.json
        → log_iter --type train_result 记账本 (verdict + 各 check)
        ▼
   APPROVED ──▶ 渲染检查视频 → 通知人工 → 人工 OK → 全量 60 集(新目录)→ 训练侧 Stage-2
   CHANGES  ──▶ 数据侧按 changes_requested 改 → 重生成 → 回到评审
   收敛判定: 连续 2 轮 leak_floor/covis 无改善(Δ<0.02)→ 暂停并汇报人工
```

### changes_requested → 数据侧可调旋钮 (config/default.yaml 等)
| 不过项 | 优先调 |
|---|---|
| `leak_floor` 低 | `scene.decorrelate_position.cluster_band_m`↑、slot 随机化、质心取景(注意与 recovery 的张力) |
| `covis` 低 | `ambient.cars_only`、`target_class_weights` 全 car、`covisibility.recycle_radius`↓、`min_clearance`↓ |
| `offscreen` 过高 | `environment.max_lost_seconds`↓、recovery hysteresis、降机动强度 |
| `offscreen` 过低 | 主动制造长丢失事件(转弯/遮挡/横穿) |
| `recovery` 低 | 制造 ≥3s 丢失后重捕获片段;`max_lost_seconds` 放宽到 15;recovery 追真值目标 |
| `jitter` 高 | yaw 死区(近天底 atan2 病态)、渲染按 10fps |

**约束**:`/data/mvp` 是训练侧稳定数据,**评审样本一律写到 `/data/mvp_review_r<N>`**,不覆盖;GPU1-7(GPU0=训练);不擅自启动 `train.stage2`。
