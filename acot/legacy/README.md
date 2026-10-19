# 旧脚本与 acot 命令对照(P4)

旧脚本**原地保留、不改动**:笔记里引用了它们，另一会话也在使用。每个脚本的处置记在 `acot/config/legacy_map.yaml`
的 `launchers` 和 `files` 两节，`python -m acot lint` 会核对这份映射是否覆盖了全部旧入口。
新实验一律走 `python -m acot …`。

| 旧入口 | 处置 | 现在的做法 |
|---|---|---|
| `rollout/eval_asd.sh`、`runs/opening_dump/{chains_attr,run_openval,f_v4,f_oracle_absent}.sh`、`rollout/eval_ext_*.sh` | 黄金参照(L0) | `python -m acot eval <实验>`。复现这些脚本的实验在 `acot/configs/experiments/legacy/` |
| `runs/asd/iter/launch_wd.sh`(看门狗、续跑) | 黄金参照 | `acot eval` 的调度器自带看门狗、CARLA 重启、逐种子状态和续跑 |
| `rollout/policy_server.py` 直接启动 | 被 `acot eval` 包装 | `infra.runtime=legacy`:旧服务端加回显和审计;`infra.runtime=acot`:新运行时 |
| `runs/tah_v90/collect_dagger.sh` | 黄金参照 | `acot eval`(`collect: [tah_dagger]`,legacy 运行时)+ `acot prep`(kind dagger,合并成产物) |
| `runs/tah_v90/gen_fov90*.sh`、`scripts/run_resilient.sh`、`scripts/gen_seed_town_list.sh`、`iter/*.sh` | 被取代 / 被包装 | `python -m acot data gen <datagen.yaml>`:冻结计划、QC、重试与替补 |
| `runs/tah_v90/pipeline*.sh`、`train/tah_cache.py`、`train/tah_merge.py`、`train/tah_train.py` | 被包装 | `python -m acot prep <prep.yaml>`、`python -m acot train <recipe.yaml>`(产物登记在 `runs/artefacts/`) |
| `train/backbone_kv.py`(ctx 缓存) | 被包装 | `python -m acot prep`(kind vlm_ctx),写进内容寻址的产物目录，不再共用 `runs/ctx_cache_*` |
| `rollout/opening_cls.py`、`rollout/opening_ft.py` | 被包装 | `python -m acot train recipes/opening_attr_*.yaml` |
| `train/docker/*.sh` | 保留 | 镜像构建和运行;acot 运行时的镜像用 `bash acot/docker/build_acot_image.sh` |
| `scripts/run_job_v6.sh` | 未迁移 | 归 v6 数据会话所有 |

## 旧代码中与实际行为不符的注释(留给代码所有者修正，acot 不改共享文件)

- `rollout/policy.py` 的 `_committed_world`(deploy v2 分支)注释说，未匹配或被遮挡时"the estimator coasts on CV"。实际上返回 None 时 `act()` 会跳过整个 WHERE 更新，z_ex 保持 EAR 本 tick 的采样。PGIF absent 分支的注释"WHERE coasts on its CV estimate"也不对，原因相同。acot 运行时按实际行为实现，并在 docstring 中写明。
- `rollout/tracklets.py` 写的是"Hungarian assignment (scipy; greedy fallback)"。但策略镜像里没有 scipy,所以所有 deploy v2 运行用的都是贪心匹配。acot 的感知模块显式固定为贪心匹配。
- `rollout/opening_zoom.py` 中 `AttrScorer` 的 ft ckpt 路径写死为 `eval_out/opening_attr_ft.pt`,不跟随 `ckpt` 参数。acot 运行时按协议给出的两个路径分别加载。
