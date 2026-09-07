# mvp_full_v5 数据集重生成 spec(给数据 agent)

> 2026-09-04 · 起因:`/nvidia/hque/data/carla_data/mvp_full_v5` 被 root 级磁盘清理**整个删除**(carla_data/ 现为空、属主 root)。所有离线工作(reid 度量微调、`reid_resolution_probe`、`baseline_deadreckon`、高分辨率探针基座)都依赖这个数据集 → 需**按原配置复现重生成**。

---

## 0. 目标
用**原有 mvp 配置**重新生成 `mvp_full_v5`(~66 集,512×512),字段与之前一致,供 reid/离线探针使用。**闭环实验不受影响(CARLA 实时渲染),此重生成纯为离线。**

## 1. 生成命令(在 cyh-carla 容器内)
```bash
# 输出到 /data/mvp_full_v5(= 主机 /nvidia/hque/data/carla_data/mvp_full_v5)
bash scripts/run_resilient.sh \
     config/scenarios/mvp.yaml \
     /data/mvp_full_v5 \
     66 \            # N 集(原为 66;≥60 即可)
     42 \            # seed(任意;split 按文件名顺序,不需与原 seed 一致)
     100 \           # max_gb 磁盘上限
     2012 \          # CARLA rpc port(harness 惯例)
     1               # graphicsadapter GPU(Vulkan;-graphicsadapter=1)
```
> 场景由 `config/scenarios/mvp.yaml`(继承 `config/default.yaml`)定义,**无需改配置**。

## 2. 必须一致的关键参数(核对,勿改)
| 项 | 值 | 来源 |
|---|---|---|
| RGB 分辨率 | **512×512** | default.yaml `output.rgb_resolution` |
| FOV | 70 | default.yaml `output.fov` |
| Towns | Town01/02/04/05/10HD | mvp.yaml |
| 主动丢失 | 2 次/集,4.5–6.5s | mvp.yaml `loss_events` |
| 干扰车 | 4–8,≥2 look-alike | mvp.yaml `num_distractors`/`min_similar_distractors` |
| 相似策略 | same_color/same_shape/distinct = 40/40/20 | mvp.yaml |
| 目标类 | 全 car(轿车,look-alike 池丰富) | mvp.yaml |

## 3. h5 必含字段(reid/离线探针要读的 —— 标准 recorder 默认就写,交付前核对)
- `rgb` `(T,512,512,3)` uint8
- `target/{tx,ty,tz, tvx,tvy,tvz, tyaw, tspeed}`
- `state/{cam_x,cam_y,cam_z,cam_pitch,cam_yaw, uav_vx,uav_vy,uav_vz}`
- `distractors/{positions (T,D,6), bbox (T,D,4), similar (D,) uint8}`  ← **similar 标志对 reid 关键**
- `annotation/off_screen`(丢失/重捕获判定)
- 顶层 attrs:`language`、`town`、`target_color`、`target_blueprint`、`seed`

> **WM 的 `annotation/road_pred` 等补标注本次可跳过**(reid 不需要;WM 若恢复再按 `wm-m0-data-spec.md` 补)。

## 4. ⚠️ 保护数据不被清理(重要)
本盘的清理已**反复删除我们的数据+缓存**(151GB ctx 缓存、整个 mvp_full_v5)。重生成后请:
- 与运维/清理脚本**协调**,把 `mvp_full_v5` 标为**不可清理**(如 `chattr +i` 目录、或加入清理白名单、或放受保护路径);
- 至少**通知一声**,避免再被误删(离线训练需要它稳定存在数天)。

## 5. 网络(已修)
cyh-carla-net 已钉死到 `192.168.240.0/24`(非-172,见 docker-compose.yml)——数据 agent 用现有 compose 起容器即可,不会碰 172.x。

## 6. QC(交付前自查)
1. `ls /data/mvp_full_v5/episode_*.h5 | wc -l` ≈ 66;
2. 抽 1 集:`rgb.shape==(T,512,512,3)`;`distractors/similar` 存在且和为 ≥2;`annotation/off_screen` 存在;
3. 目视 1 帧:目标车 + 多个 look-alike 干扰车共视,分辨率清晰。

## 7. 交付后我方对接(立即可跑)
数据一到:① `reid_metric_train.py`(crop→DINOv2→训投影头,同实例跨偏心拉近)+ 评估 central/off mis;② `reid_resolution_probe` 复核;③(可选)高分辨率重渲以此为基座。

---

## 交付物
- `/data/mvp_full_v5/` ~66 集 h5(512×512,§3 字段齐全);
- QC 报告(§6);
- 已加清理保护(§4)。
