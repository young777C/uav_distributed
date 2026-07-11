#!/usr/bin/env bash
# ====================================================================
# 一站式 FDLC × PX4+Gazebo 仿真（Docker + Bridge）
#
# 自动完成:
#   1. 启动 PX4 SITL @ Docker（若未运行）
#   2. 等待 MAVSDK 端口就绪
#   3. 运行 FDLC 快环桥接
#   4. 停止 Docker 容器
#
# 用法:
#   bash scripts/phase1_docker_bridge.sh --duration 120
#
# 可选参数:
#   --duration <秒>        仿真时长 (默认 120)
#   --config <case.yaml>   Scene+Comm 配置 (默认 phase1_px4)
#   --system <system.yaml> 架构配置 (默认 FDLC)
#   --no-cleanup          结束后不停止 Docker
# ====================================================================

set -euo pipefail

# ── 解析参数 ──────────────────────────────────────────────────
DURATION=120
CONFIG="configs/experiments/paper1/cases/phase1_px4.yaml"
SYSTEM="configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml"
CLEANUP=true
BRIDGE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --system) SYSTEM="$2"; shift 2 ;;
        --no-cleanup) CLEANUP=false; shift ;;
        *) BRIDGE_ARGS+=("$1"); shift ;;
    esac
done

# ── 颜色 ──────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# ── Docker 预检 ────────────────────────────────────────────────
command -v docker &>/dev/null || {
    echo "[ERROR] 请安装 Docker Desktop"; exit 1
}
docker info &>/dev/null || {
    echo "[ERROR] Docker 未运行"; exit 1
}

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  FDLC × PX4+Gazebo 一站式仿真 (Docker)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo "  配置: ${CONFIG}"
echo "  系统: ${SYSTEM}"
echo "  时长: ${DURATION}s"
echo ""

# ── Step 1: 启动 PX4 Docker Container（如果未运行） ────────────
CONTAINER_NAME="px4_sitl_fdlc"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}[1/3] PX4 容器已在运行: ${CONTAINER_NAME}${NC}"
else
    echo -e "${YELLOW}[1/3] 启动 PX4 Docker 容器...${NC}"
    DOCKER_CMD="bash scripts/docker_run_px4_sitl.sh"
    echo "  执行: ${DOCKER_CMD}"
    echo -e "  ${YELLOW}请在新终端中运行上面的命令，等 PX4 就绪后回到这里。${NC}"
    echo ""

    # 在后台启动 Docker
    bash scripts/docker_run_px4_sitl.sh &
    DOCKER_PID=$!

    # 等 mavlink 端口就绪
    echo -e "  ${YELLOW}等待 MAVSDK 端口 14540 就绪...${NC}"
    for i in $(seq 1 60); do
        if ss -tulpn 2>/dev/null | grep -q ":14540"; then
            echo -e "  ${GREEN}✓ MAVSDK 端口就绪 (${i}s)${NC}"
            break
        fi
        if ! kill -0 "${DOCKER_PID}" 2>/dev/null; then
            echo -e "  ${RED}[ERROR] Docker 进程已退出${NC}"
            exit 1
        fi
        sleep 2
    done
fi

# ── Step 2: 运行 FDLC Bridge ──────────────────────────────────
echo -e "${GREEN}[2/3] 启动 FDLC 快环桥接...${NC}"
echo ""

PYTHONPATH=src python3 scripts/phase1_fdlc_px4_bridge.py \
    --config "${CONFIG}" \
    --system "${SYSTEM}" \
    --duration_s "${DURATION}" \
    --px4_address "udp://127.0.0.1:14540" \
    "${BRIDGE_ARGS[@]}"

BRIDGE_EXIT=$?

# ── Step 3: 清理 ──────────────────────────────────────────────
echo ""
if [ "${CLEANUP}" = true ]; then
    echo -e "${YELLOW}[3/3] 停止 Docker 容器...${NC}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    echo -e "${GREEN}✓ 已停止${NC}"
else
    echo -e "${YELLOW}[3/3] 跳过清理 (--no-cleanup)${NC}"
    echo "  Docker 容器仍在运行: ${CONTAINER_NAME}"
    echo "  手动停止: docker stop ${CONTAINER_NAME}"
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
if [ ${BRIDGE_EXIT} -eq 0 ]; then
    echo -e "${GREEN}✅ Phase 1 仿真完成${NC}"
else
    echo -e "${RED}❌ Phase 1 异常退出 (code=${BRIDGE_EXIT})${NC}"
fi
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
exit ${BRIDGE_EXIT}
