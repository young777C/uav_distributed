#!/usr/bin/env bash
# ====================================================================
# Phase 1: FDLC × PX4+Gazebo Docker 启动脚本
#
# 在 Docker 中运行 PX4 SITL + Gazebo Classic，暴露 MAVSDK 端口。
# 宿主机（WSL2 / Windows / macOS）上的 bridge 脚本可直接连接。
#
# 依赖: Docker Desktop（Windows/Mac/Linux）
#
# 用法:
#   # 终端 1: 启动 PX4 SITL 仿真
#   bash scripts/docker_run_px4_sitl.sh
#
#   # 终端 2: 等 PX4 就绪后运行 FDLC 桥接
#   PYTHONPATH=src python3 scripts/phase1_fdlc_px4_bridge.py \
#     --config configs/experiments/paper1/cases/phase1_px4.yaml \
#     --system configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml \
#     --duration_s 120
#
# 环境变量:
#   PX4_IMAGE=px4io/px4-dev-simulation-focal:2024-12-09  (默认)
#   PX4_MODEL=gazebo-classic_iris                          (默认)
#   HEADLESS=1                                              (默认)
# ====================================================================

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────
PX4_IMAGE="${PX4_IMAGE:-px4io/px4-dev-simulation-focal:2024-12-09}"
PX4_MODEL="${PX4_MODEL:-gazebo-classic_iris}"
HEADLESS="${HEADLESS:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-px4_sitl_fdlc}"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# ── 预检 ──────────────────────────────────────────────────────
command -v docker &>/dev/null || {
    echo -e "${RED}[ERROR] 请安装 Docker Desktop: https://docs.docker.com/desktop/${NC}"
    exit 1
}
docker info &>/dev/null || {
    echo -e "${RED}[ERROR] Docker 未运行。请启动 Docker Desktop 并确保 WSL2 集成已启用。${NC}"
    exit 1
}

# ── 清理旧容器 ────────────────────────────────────────────────
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ── 打印配置 ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    PX4 SITL + Gazebo Classic @ Docker                       ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
printf "${CYAN}║  %-20s %-30s ║${NC}\n" "镜像:" "${PX4_IMAGE}"
printf "${CYAN}║  %-20s %-30s ║${NC}\n" "模型:" "${PX4_MODEL}"
printf "${CYAN}║  %-20s %-30s ║${NC}\n" "HEADLESS:" "${HEADLESS}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
printf "${CYAN}║  %-20s %-30s ║${NC}\n" "MAVSDK 端口:" "14540/udp → 宿主机"
printf "${CYAN}║  %-20s %-30s ║${NC}\n}" "桥接参数:" "--px4_address udp://127.0.0.1:14540"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  ${YELLOW}首次启动需要 5-15 分钟（编译 PX4）${NC}                    ${CYAN}║${NC}"
echo -e "${CYAN}║  ${YELLOW}后续启动仅需 ~15 秒${NC}                                    ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── 将 PX4 源码挂载到容器（如果有本地副本） ─────────────────
PX4_HOST_PATH=""
if [ -d "${HOME}/PX4-Autopilot" ]; then
    PX4_HOST_PATH="${HOME}/PX4-Autopilot"
    echo -e "${GREEN}[INFO] 复用本地 PX4 源码: ${PX4_HOST_PATH}${NC}"
    echo -e "${YELLOW}[INFO] 如首次编译，请确保 ${PX4_HOST_PATH} 已执行过 git submodule update${NC}"
fi

# ── 构造 docker run 命令 ──────────────────────────────────────
DOCKER_VOLUMES=()
if [ -n "${PX4_HOST_PATH}" ]; then
    DOCKER_VOLUMES+=(-v "${PX4_HOST_PATH}:/home/user/PX4-Autopilot:z")
fi
DOCKER_VOLUMES+=(-v /tmp/.X11-unix:/tmp/.X11-unix:ro)

DOCKER_ENVS=(
    -e "HEADLESS=${HEADLESS}"
    -e "DISPLAY=${DISPLAY:-:0}"
    -e "PX4_MODEL=${PX4_MODEL}"
)

PORTS=(
    -p 14540:14540/udp
    -p 14580:14580/udp
)

# ── 启动容器 ──────────────────────────────────────────────────
echo -e "${GREEN}[DOCKER] 启动仿真容器...${NC}"
echo ""

# 使用 -i (非 -it) 以兼容脚本和非 TTY 环境
docker run -i --rm \
    --name "${CONTAINER_NAME}" \
    "${PORTS[@]}" \
    "${DOCKER_ENVS[@]}" \
    "${DOCKER_VOLUMES[@]}" \
    --entrypoint="" \
    "${PX4_IMAGE}" \
    /bin/bash -e /dev/stdin \
<<'SCRIPT'
# ================================================================
# 以下代码在 DOCKER 容器内执行
# ================================================================

PX4_MODEL="${PX4_MODEL:-gazebo-classic_iris}"
HEADLESS="${HEADLESS:-1}"
HOME_DIR="${HOME}/PX4-Autopilot"

echo "[PX4] HOME=${HOME}"
echo "[PX4] PX4_MODEL=${PX4_MODEL}"
echo "[PX4] HEADLESS=${HEADLESS}"

# ── 获取 / 确认 PX4 源码 ──────────────────────────────────
if [ ! -d "${HOME_DIR}" ]; then
    echo "[PX4] 克隆 PX4-Autopilot v1.15.0..."
    cd "${HOME}"
    git clone --quiet --depth=1 --branch v1.15.0 \
        https://github.com/PX4/PX4-Autopilot.git
fi

cd "${HOME_DIR}"

# 确保 sitl_gazebo-classic 子模块存在
if [ ! -d "Tools/simulation/gazebo-classic/sitl_gazebo-classic" ]; then
    echo "[PX4] 初始化子模块..."
    git submodule update --init --depth=1 \
        Tools/simulation/gazebo-classic/sitl_gazebo-classic 2>/dev/null || true
fi

# ── 编译（仅首次） ─────────────────────────────────────────
if [ ! -f "build/px4_sitl_default/bin/px4" ]; then
    echo "[PX4] ————————————————————————————————————————————"
    echo "[PX4] 首次编译 PX4 SITL（需要 5-15 分钟）"
    echo "[PX4] ————————————————————————————————————————————"
    DONT_RUN=1 make px4_sitl "${PX4_MODEL}"
    echo "[PX4] 编译完成."
fi

# ── 启动 SITL ──────────────────────────────────────────────
echo "[PX4] ————————————————————————————————————————————"
echo "[PX4] 启动 SITL"
echo "[PX4]   模型:   ${PX4_MODEL}"
echo "[PX4]   HEADLESS: ${HEADLESS}"
echo "[PX4] ————————————————————————————————————————————"
echo "[PX4] 等待 MAVSDK 端口 14540 就绪..."
echo "[PX4] 就绪后在宿主机的另一终端运行 bridge 脚本。"
echo ""

HEADLESS="${HEADLESS}" make px4_sitl "${PX4_MODEL}"
SCRIPT

# ── 容器退出后 ──────────────────────────────────────────────
EXIT_CODE=$?
echo ""
if [ ${EXIT_CODE} -eq 0 ] || [ ${EXIT_CODE} -eq 130 ] || [ ${EXIT_CODE} -eq 143 ]; then
    echo -e "${GREEN}[DOCKER] PX4 仿真已停止。${NC}"
else
    echo -e "${RED}[DOCKER] PX4 异常退出 (code=${EXIT_CODE})${NC}"
    echo -e "${YELLOW}  容器日志用: docker logs ${CONTAINER_NAME}${NC}"
fi
echo "  重新启动: bash scripts/docker_run_px4_sitl.sh"
