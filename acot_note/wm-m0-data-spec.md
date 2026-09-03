# WM M0 数据补标注 spec(给数据 agent)

> 2026-08-31 · WM P2a 的关键路径起点 · 配套 `wm-p2a-plan.md`
> **性质:纯补标注,不重渲染、不重跑仿真**——只读已存的 `target/tx,ty,tz,tspeed,tyaw` + 查 CARLA 地图 → 追加 3 个字段进每个 h5。几秒~几分钟/集的 CPU 活。

---

## 0. 为什么要补 & 用在哪
- WM(路网预测器)在**弯道/路口**上要打赢 CV(直线外推)。离线评测 `baseline_deadreckon` 在 **host 上跑、没有 CARLA** → **road 预测必须由数据 agent 用 CARLA 预计算并存进 h5**;
- 补的字段同时服务:① road-FDE 评测的预测源;② 转向/路口分层;③ 日后 WM 训练的车道 GT。

## 1. 要补的三个字段(每个 episode h5,顶层新增)

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| **`annotation/road_pred`** | `(T, 4, 3)` | float32 | 每帧目标**沿车道遍历**的预测世界位,视界 **{1,2,4,6}s**(K=4),路口选**最直行**分支 |
| **`annotation/target_is_junction`** | `(T,)` | uint8 | `map.get_waypoint(目标位).is_junction` |
| **`annotation/target_lane_id`** | `(T,)` | int32 | `road_id*100 + lane_id`(车道识别;可选但推荐) |

> T = `rgb.shape[0]`(帧数)。视界顺序**必须** `[1,2,4,6]` 秒,与 `config_v5.yaml waypoint.offsets_s` 一致。

## 2. `road_pred` 精确算法(P2a-0 几何版,零学习)

对每帧 f:
```python
p     = (tx[f], ty[f], tz[f])            # 目标世界位(已存)
speed = tspeed[f]                        # 目标速率 m/s(已存;或 |tvx,tvy,tvz|)
ref_yaw = tyaw[f]                        # 目标朝向(已存,度)
wp = carla_map.get_waypoint(carla.Location(*p), project_to_road=True,
                            lane_type=carla.LaneType.Driving)
if wp is None:                           # 目标不在可行驶车道(停车场/越野)
    road_pred[f] = repeat(p, 4)          # 退化:原地(下游会当"车道未知")
    continue
STEP = 2.0                               # 遍历步长(m)
for k, off in enumerate([1, 2, 4, 6]):
    arc = speed * off                    # 沿车道要走的弧长
    cur = wp; remaining = arc
    while remaining > 1e-3:
        d = min(STEP, remaining)
        nxts = cur.next(d)               # 前方后继(路口=多个分支)
        if not nxts: break               # 断头路 → 停在当前
        cur = pick_straightest(nxts, ref_yaw)   # 选朝向最贴 ref_yaw 的(最直行)
        ref_yaw = cur.transform.rotation.yaw     # 更新参考朝向(沿路弯)
        remaining -= d
    loc = cur.transform.location
    road_pred[f, k] = (loc.x, loc.y, loc.z)
```
`pick_straightest(nxts, ref_yaw)`:选 `abs(wrap180(wp.transform.rotation.yaw - ref_yaw))` 最小的后继。

**要点**:
- 用**目标当前帧的位/速/朝向**(不是相机、不是 last-seen)——这是"从该帧起沿车道走"的**每帧量**;
- 路口(`is_junction`)分支一律选**最直行**(P2a-0 启发式;学习版是 P2a-2);
- 弧长用 `speed*off`(速率×视界),沿车道量;`cur.next(d)` 已跟着道路弯。

## 3. 存储与不破坏

- 写进**已有 episode 的同一个 h5**(顶层 `annotation/` 组下),**不新建文件、不动 rgb/其它**;
- 幂等:若字段已存在,覆盖即可;
- 处理 `mvp_full_v5/episode_*.h5` 全部 59 集(注意 `episode_000066` 曾缺 `search_mode/off_screen`,若该集损坏可跳过并记录)。

## 4. 地图加载(CARLA 侧)
- 每集用其 `town` 属性(h5 `f.attrs['town']`,如 `Town05`)`client.load_world(town)` 后 `world.get_map()`;
- **按 town 分组处理**(同 town 的集共用一次地图加载,省时);
- 无需渲染 → 可 `-RenderOffScreen`,或用已起的 cyh-carla 容器 exec。

## 5. QC(交付前自查)
1. **形状/顺序**:`road_pred` 是 `(T,4,3)`,视界 `[1,2,4,6]`;
2. **合理性**:随机抽 20 帧,`road_pred[f,0]`(1s)与目标 1s 后真实位 `(tx[f+10],..)` 的距离**中位 < 5m**(1s、直路应很近);弯道帧应比 CV(直线)更贴真实(可选自查);
3. **is_junction 命中**:`target_is_junction` 均值合理(路口帧占比,通常 5-25%);
4. **退化计数**:报告有多少帧 `wp is None`(目标不在车道)。

## 6. 并行(B 类,新增集 —— 与 M0 不同,是重新生成)
> M0 之外,后续 WM 评测/训练需要 **P1 转向长丢失数据**(与 M0 独立、可并行排):
- 5–15s 主动长丢失,**丢失窗口跨越弯道/路口**(目标在丢失期间转弯);
- 目标重现 + 重现处 ≥2 look-alike 共视;
- 专家直飞拦截演示。
- 现有转向丢失帧仅 n~27,**评测不稳**;需 ~数十集转向丢失事件。此项**重新 rollout 生成**,追加新集,不替换现有。

---

## 交付物
- 59 集 h5 各含 `annotation/{road_pred, target_is_junction, target_lane_id}`;
- QC 报告(§5 四项);
- (并行)P1 转向长丢失数据 spec 的排期。

模型侧对接:`train/baseline_deadreckon.py` 的 `road` 分支已就绪(读 `annotation/road_pred`),M0 一交付即可跑 **M2 离线几何门**(转向 road-FDE < CV-FDE)。
