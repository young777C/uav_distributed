# CARLA 数据生成容器（cyh-carla）

为 ACoT-UAV-Track 数据生成单独准备的**隔离容器**，与仓库根目录的 `cyh-project1`
Web 栈以及本机其它环境完全独立。一个镜像里同时包含 **CARLA 仿真器服务端**
（`/home/carla/CarlaUE4.sh`）和 **Python 客户端**（`carla==0.9.15` + 仓库依赖），
因此单个容器就能跑通「多 GPU 并行起服务 + 客户端采集」的完整流程。

## 隔离方式

- 镜像：`cyh-carla:0.9.15`　容器：`cyh-carla`　网络：`cyh-carla-net`
- 通过 nvidia runtime 访问 GPU；容器内用户 uid/gid 已重映射为宿主机 `1003:1015`，
  生成的数据归属你本人而非 root
- 代码目录 `.` 绑定挂载到 `/workspace`；数据写到宿主机
  `/nvidia/hque/data/carla_data`（挂载为 `/data`，2 TB 空间）
- 不对外暴露端口；CARLA RPC 端口只在容器内部使用

## 构建 & 启动

```sh
cd /nvidia/hque/code/cyhe/uav-acot-track/carla_uav_tracking

docker compose build          # 首次拉取 carlasim/carla:0.9.15（约 16 GB）
docker compose up -d          # 后台常驻容器
docker compose exec carla bash
```

可选环境变量：

```sh
# 只用部分 GPU（默认全部 8 张 A40）
CYH_CARLA_GPUS=0,1,2,3 docker compose up -d
# 换数据输出目录
CYH_CARLA_DATA=/nvidia/hque/data/other docker compose up -d
```

## 容器内验证与使用

```sh
# 1) 起一个 CARLA 服务端（离屏渲染，无需显示器）
bash $CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 &
sleep 15

# 2) 连通性冒烟测试
python carla_quick_test.py            # 期望输出 "🎉 CARLA is ready for data generation!"

# 3) 单条 / 批量采集
python scripts/generate_single.py --config config/default.yaml
python scripts/generate_batch.py  --config config/default.yaml --output /data/raw

# 4) 多 GPU 并行（每 GPU 一个服务端，端口 2000/2002/...）
python scripts/launch_parallel.py --num-servers 4 --gpus 0,1,2,3 --output /data/raw
```

## 常用命令

```sh
docker compose ps
docker compose logs -f
docker compose exec carla bash
docker compose down            # 停止并移除容器（镜像与 /data 数据保留）
```

## 备注

- 宿主机驱动为 470.256.02（CUDA 11.4）。CARLA 靠 Vulkan 离屏渲染，470 驱动可用；
  若个别地图渲染异常，可把镜像 tag 回退到 `carlasim/carla:0.9.14` 并同步
  `docker/requirements.txt` 里的 `carla==` 版本。
- `config/default.yaml` 的 `host` 已改为 `127.0.0.1`、`data_dir` 改为 `/data/`
  （原先是 Windows/WSL2 遗留的 `172.17.208.1` 和 `/mnt/d/...`）。
