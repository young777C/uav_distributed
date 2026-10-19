# acot — 统一入口

设计文档：`acot_note/1004_code_framework.md`(实施记录见 §18–§21)。当前实现到 P3b,P2 运行时已可用：

- **闭环评测**:有两种 policy 运行时，由 `infra.runtime` 选择。
  - `acot`(默认):`acot/runtime/server.py`,直接读取已解析配置，方法代码在结构上拿不到真值。它实现了 1004 重跑所需的全部臂，与 legacy 的决策逐位一致(`acot replay-parity`,设计文档 §20.3);
  - `legacy`:旧的 `policy_server.py`,经包装入口运行，做配置回显和真值审计。acot 没有实现的臂需要用它;复现旧运行的实验(`experiments/legacy/`、`aa_noise`、`smoke_ours_final`、`smoke_replay`)已固定为 legacy。
- **离线**:产物库(`runs/artefacts/`)、数据生成与 QC、TAH 序列缓存、训练，都走同一入口，产物按内容寻址并带兼容检查。
- **旧脚本**:与新命令的对照见 `acot/legacy/README.md`。

## 常用命令

```bash
python -m acot lint                                   # 校验全部配置与旧配置面覆盖，并列出旧训练配置中的已知冲突
python -m acot show  acot/configs/experiments/<x>.yaml [--arm A] [--diff | --legacy]
python -m acot eval  acot/configs/experiments/<x>.yaml --dry-run                        # 只打印将要执行的命令
python -m acot eval  acot/configs/experiments/<x>.yaml --workers 2 --carla-ports 2050,2060 --gpus 1,2
python -m acot analyze acot/configs/experiments/<x>.yaml                                  # 结果不完整就报错
python -m acot results import-legacy zoa_pgif_trk v2h_pgif_trk                          # 导入旧结果
python -m acot eval  acot/configs/experiments/<x>.yaml infra.runtime=legacy             # 改用旧运行时
python -m acot replay-parity acot/configs/experiments/p2_parity.yaml --record <act_record.pkl> --gpu 2
python -m acot surface                                 # 重新生成 acot_note/1004_legacy_surface.md
python -m acot artefacts ls | verify                   # 产物库(INCOMPLETE = 没有 manifest,下游不可用)
python -m acot data split <dataset> --name N --rule naive|town_heldout
python -m acot data gen  acot/configs/datagen/<x>.yaml --dry-run                          # 冻结计划 + worker 命令
python -m acot prep  acot/configs/prep/<x>.yaml --gpus 1,2                               # tah_seq / vlm_ctx / dagger
python -m acot train acot/configs/recipes/<x>.yaml --gpu 1 [seed=N] [--register-only]
bash acot/docker/build_acot_image.sh                   # acot 运行时的镜像(timm 固定在镜像中)
cd acot && python3 -m pytest -q                        # 单元测试、L0 黄金测试、L2 与旧脚本的一致性、P2 结构隔离
```

## 写一个新实验

1. **选协议**:`acot/configs/protocols/`。最终协议是 `sim_proxy_v2@1`;协议冻结，要改就新建一个版本。
2. **选方法或新建方法**:`acot/configs/methods/`。新方法必须 `extends: _base`(或继承别的方法),只写与父方法不同的字段。
3. **写实验文件**:`acot/configs/experiments/`。一个最小例子：
   ```yaml
   name: my_exp
   protocol: sim_proxy_v2@1
   seeds: {set: dev}                      # 种子集登记在 acot/configs/seeds.yaml
   arms:
     ours:     {method: ours_final}
     no_pgif:  {method: ours_final, override: {filter.mode: "off"}}
     open_head: {method: ours_final, ablation: opening,
                 override: {frontend.opening.module: head, frontend.opening.window: 1}}
   collect: [ticks, trk, gt_access, scenario]
   comparisons:
     - {family: main, a: ours, b: [no_pgif, open_head], metrics: [correct, longC, longW]}
   ```
4. **检查后再运行**:先 `acot show --diff` 看清每个臂相对 `_base` 改了什么，再 `acot eval --dry-run`,最后正式 `acot eval`。

## 规则(由代码强制执行)

- **覆盖权限**:
  - 臂只能覆盖方法字段;
  - 覆盖协议的 `frontend.*` 或 `perception.*` 时，必须声明 `ablation:`;
  - `camera`、`sim`、`scorer` 不能覆盖。
- **真值通道**:方法用到的真值通道，必须在 `diagnostics:` 中如实列出，该臂会被标为 diagnostic,不能进入主表。
- **test 种子**:test 种子集是锁住的，使用时必须加 `--allow-test`,并在实验中写 `prereg_ref`。
- **结果目录**:`carla_uav_tracking/rollout/eval_out/exp/<实验>/<臂>/seeds/<种子>/`,每个种子带 `status.json`,状态为 complete、skipped、failed、partial 之一。
- **真值审计**:运行期审计每个种子读取真值的位置(`gt_access.json`),出现 `acot/config/gt_sites.yaml` 之外的读取点，该种子判为失败。改了 policy 代码之后，需要审核新的读取点并登记。
- **Docker**:只在已有的 `cyh-carla-net` 上启动 policy 容器;CARLA 只用 `docker exec`;从不创建网络。
- **acot 运行时**:不读任何旧的行为环境变量(设置了就拒绝启动);只在 `PolicyInput` 上做决策，标签只进 `ScoringInfo`;诊断臂另得一个只允许已声明通道的 `OracleHandle`。不支持的设置(例如 reanchor、cold window、reid 头)在解析时就被拒绝，这类臂用 `infra.runtime=legacy` 运行。
