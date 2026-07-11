#!/usr/bin/env python3
"""
PX4 SITL + 论文快环集成示例：将快环的速度参考通过 MAVSDK 发送给 PX4。

功能:
  - 连接 PX4 SITL
  - 切换至 Offboard 模式 + arm
  - 按论文快环逻辑发送速度参考 (NED velocity)
  - 记录 PX4 实际飞行轨迹
  - 5 秒后自动降落

在运行此脚本前，先在终端 1 启动 SITL:
  cd ~/PX4-Autopilot && make px4_sitl gazebo-classic_iris
"""

import asyncio
import json
import time
from pathlib import Path
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw, PositionNedYaw


async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for PX4 ...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break

    # 确保有 GPS fix
    async for gps in drone.telemetry.gps_info():
        if gps.fix_type >= 2:
            break

    print("Arming + offboard ...")
    await drone.action.arm()
    await drone.offboard.set_position_ned(PositionNedYaw(0, 0, -10, 0))

    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard error: {e._result.result}")
        return

    # ===== 论文快环控制循环 (5Hz) =====
    t_start = time.time()
    step = 0
    traj_log = []

    print("Sending velocity setpoints for 20s ...")
    while time.time() - t_start < 20:
        # 获取 PX4 实际位置
        async for pos in drone.telemetry.position_nes():
            pos_ned = (pos.north_m, pos.east_m, pos.down_m)
            break

        # ====== 此处替换为论文快环逻辑 ======
        # 当前简化：向目标航点推动
        target_n, target_e = 50.0, 50.0  # 示例目标
        vx = max(-5, min(5, (target_n - pos_ned[0]) * 0.3))
        vy = max(-5, min(5, (target_e - pos_ned[1]) * 0.3))
        # ===================================

        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vx, vy, 0.0, 0.0)
        )

        traj_log.append({
            "t": round(time.time() - t_start, 2),
            "pos_ned": list(pos_ned),
            "v_cmd": [vx, vy, 0.0],
        })
        step += 1
        await asyncio.sleep(0.2)

    # 降落
    print("Landing ...")
    await drone.offboard.stop()
    await drone.action.land()

    # 保存轨迹
    log_path = Path("sitl_traj_log.json")
    log_path.write_text(json.dumps(traj_log, indent=2))
    print(f"Trajectory saved to {log_path}")
    print("✅ 快环 PX4 集成示例完成")


if __name__ == "__main__":
    asyncio.run(run())
