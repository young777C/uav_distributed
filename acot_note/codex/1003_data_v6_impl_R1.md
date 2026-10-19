verdict:修复后再生成

投影主公式本身我认为是对的：`labels_v6.py:29-36` 使用 CARLA camera frame `x=forward,y=right,z=up`，`u=f*y/x+W/2, v=-f*z/x+H/2`，与 CARLA 常用投影一致；冒烟 G3/G4 也支持这一点。不能放行的风险主要在“半成品混入、关联不可回修、reacq/occ 有效样本不足”。

**MUST**

- 问题：生成不是原子提交，会留下看起来像成品的半成品 H5。  
  证据：`generate_batch.py:87` 先写 progress，`generate_batch.py:118-120` 先写 summary，`generate_batch.py:143-146` 才 postprocess；`run_job_v6.sh:37-42` top-up 只看文件存在。冒烟 `episode_202000.h5` 缺 `actors/visible`、`events/target_reentry`，QC 报 `G5 postprocess missing`。  
  改法：录制到 `episode_x.tmp.h5`，postprocess + G1-G5 单集 QC 通过后 atomic rename；progress 分 `attempted/completed`，top-up 只数 completed/pass。  
  验收：故意 kill 在录制中和 postprocess 中，resume 后目录内没有任何半成品 `episode_*.h5`。

- 问题：同一输出目录可被多个 orchestrator 同时写，plan/provenance 会被覆盖，日志已发生交织。  
  证据：`orchestrate.py:96-102` 直接覆盖 `provenance.json/plan.json`；冒烟日志 `orchestrate.log:8-15` 显示第二轮 `smoke_reacq` 与第一轮仍在跑的 job 交织，当前 `plan.json:2` 已只剩 `smoke_reacq`，但目录里仍有 base/occ 文件。  
  改法：输出目录加锁文件，非空目录必须 `--resume` 且 plan、git hash、dirty hash 完全一致；新补测写新目录或 run 子目录。  
  验收：并发启动第二个相同 `--out` 直接失败；resume 不会重写 provenance。

- 问题：QC 门槛不足，会让短截断、无重捕获、无遮挡的 episode 进入划分。  
  证据：`qc_v6.py:40` 只要求 `T>=300`；`qc_v6.py:125-141` 只打印数据集统计不设硬门。冒烟里 `reacq Town10HD` 仅 337 帧仍可通过，日志 `job_reacq_Town10HD_210000.log:4-10` 是 structural occlusion 截断；`occ Town04` 通过但 `re_occ=0`。  
  改法：按 subset 加硬门：base 最短时长、reacq 每集/每镇长 gap 重入画数、occ 结构遮挡帧数/事件数、截断率上限。  
  验收：当前 smoke 必须失败；修完后每个 town 的 reacq/occ smoke 都满足预注册事件数。

- 问题：原始 instance id 图被丢弃，关联算法若后验发现错配，无法离线重算，只能重录。  
  证据：`labels_v6.py:58-63` 解码了 raw `iid/tag`，但 `labels_v6.py:81,109,116` 折叠成 uint8 `instance_mask`；`recorder.py:381-382` 只保存 `traffic/blobs` 和折叠 mask。  
  改法：至少保存车辆像素的 raw `iid` 图/压缩 blob mask，或保存每个 iid 的 RLE mask；允许离线重跑关联。  
  验收：删除 `actors/inst_id/modal_box/vis_px` 后，可仅用 H5 内数据重建这些字段并复现当前结果。

- 问题：贪心关联在紧挨/遮挡/ambient occluder 场景仍可能错配，现有 G3/G4 很难抓到。  
  证据：`labels_v6.py:96-105` 只要求 blob ≥50% 像素落入某投影框；若 ambient 车完全落在被遮挡目标 amodal 框内，也会被当目标。冒烟 `episode_205000` 目标 visible inst id 出现 2 次切换。  
  改法：加入 temporal-dominant iid、modal/amodal IoU、中心距离、遮挡状态约束；高遮挡时不允许用全新 iid 直接替换目标。  
  验收：QC 增加“每 actor dominant iid 占比”和“可见连续段内 iid switch=0，除非中间不可见 ≥gap”。

- 问题：reacq 盲飞机制尚未证明能产出可用动作标签和足够重入画。  
  证据：`v6_reacq.yaml:8-12` loss window 5-9s；`default.yaml:135` dead-reckon timeout 5s；`pid_controller.py:120-122` 超时后返回零动作。冒烟 reacq 多数无 `gap>=30+look`。  
  改法：把 dead-reckon timeout 覆盖到 ≥最大 loss window+恢复裕量，或显式写 `action_valid/loss_window`，训练时区分“可监督拦截”和“mask”。  
  验收：reacq 每镇 smoke 至少有稳定的 ≥3s reentry+lookalike，且 loss window 内无长段零动作除非标为 invalid。

- 问题：occ 子集当前不够“occ-heavy”。  
  证据：`v6_occ.yaml:14-18` 虽启用 occlusion bias，但 `default.yaml:80-84` oblique fraction 仍为 0；冒烟 `episode_304000` 只有少量 `occ>=0.4`，无 `occ>=0.6` 和 reentry。  
  改法：occ smoke 按 Town04/05/10HD/Town03 分别验收；必要时 `bias_fraction=1.0`、启用 oblique、提高 Town05/10HD 权重或剔除低效 anchor。  
  验收：occ 集每镇达到最小结构遮挡帧/episode 和结构遮挡事件/episode。

**SHOULD**

- 问题：3D 可追溯标签还不完整。  
  证据：`recorder.py:364-366` 只存 actor id/role；`recorder.py:584-620` distractor 只有位置/速度，没有 yaw、bbox extent、bbox local offset。  
  改法：写 `actors/extents`、`actors/bbox_loc`、`actors/yaw` 或 per-frame yaw；attrs 写显式 `K`。  
  验收：不用 CARLA actor，也能离线重投影任意 actor 的 3D box。

- 问题：amodal box 对穿过近裁剪面的 actor 没做边裁剪。  
  证据：`labels_v6.py:48-55` 直接丢弃 behind vertices，只用 front vertices min/max。  
  改法：对 bbox edges 与 near plane 求交后再投影。  
  验收：合成相机近处/边缘车辆，box 不突变、不变成异常小框。

- 问题：共享 `episode_summaries.jsonl` 不是严格并发安全，也不应作为最终 manifest。  
  证据：`generate_batch.py:118-120` 四进程直接 append 同一文件。  
  改法：每 job 写独立 summary，orchestrator 最后合并；最终 manifest 从 QC pass H5 attrs 生成。  
  验收：manifest episode 数、subset/town 配额、dirty hash 全一致。

**NICE**

- 清理误导注释：`recorder.py:41-42` 仍写“90→70”，`default.yaml:73-76` 仍按 FOV70 解释像素大小，容易让下一轮又改错 FOV。
- QC 可增加失败样本自动 viz：对 inst-id switch、低 reentry、低 occ 的 episode 输出关键帧，便于快速判断是场景问题还是标签问题。