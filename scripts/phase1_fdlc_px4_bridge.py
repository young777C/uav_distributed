#!/usr/bin/env python3
"""
Phase 1: FDLC fast-loop link verification on PX4+Gazebo.

使用 pymavlink 与 PX4 通信（已验证可靠），发送 Offboard 速度指令。

用法:
  终端 1: cd ~/PX4-Autopilot && HEADLESS=1 make px4_sitl gazebo-classic_iris
  终端 2: cd workdir && PYTHONPATH=src python3 scripts/phase1_fdlc_px4_bridge.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pymavlink: 连接 PX4，读写 MAVLink 消息
from pymavlink import mavutil

# FDLC 核心组件
from uavlab.common.config import load_resolved_config, _deep_merge
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastObservation, SlowPlan
from uavlab.paper1.loops.fast.fast_loop import FastLoop
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.paper1.types import FastCommState, CommState
from uavlab.scene.loader import load_scene_config

Point2D = Tuple[float, float]


def _mode_str(mode: FastCommState) -> str:
    return str(mode.value if hasattr(mode, "value") else mode)


# ═══════════════════════════════════════════════════════════════
# 1. 通信链路模型（基于 UAV–GCS 真实距离）
# ═══════════════════════════════════════════════════════════════

def compute_link_from_distance(uav_ne: Point2D, gcs_ne: Point2D,
                               enable_distance_decay=True,
                               d0=0.0, d1=150.0, lmin=0.05, lmax=0.50,
                               base_loss=0.05, delay_s=0.10) -> CommState:
    x, y = uav_ne; gx, gy = gcs_ne
    d = math.hypot(x - gx, y - gy)
    if enable_distance_decay:
        loss = lmin + (lmax - lmin) * min(1.0, max(0.0, (d - d0) / max(1e-9, d1 - d0)))
    else:
        loss = base_loss
    loss = min(max(loss, 0.0), 0.99)
    bw = 1_000_000.0 * (1.0 - loss)
    return CommState(loss_p=loss, delay_s=delay_s, bandwidth_bps=bw)


# ═══════════════════════════════════════════════════════════════
# 2. PX4 桥接器（纯 pymavlink）
# ═══════════════════════════════════════════════════════════════

@dataclass
class Px4Bridge:
    cfg: "Paper1SimConfig"
    contract: Paper1ContractConfig
    fast_params: FastLoopParams
    env: "Paper1Env"
    fast_loop: FastLoop

    # pymavlink 连接
    master: Any = None

    waypoints: List[Point2D] = field(default_factory=list)
    run_dir: Optional[Path] = None
    duration_s: float = 60.0
    dry_run: bool = False          # True=离线模式，跳过 PX4 连接

    _step: int = 0
    _exec_plan: Optional[SlowPlan] = None
    _start_time: float = 0.0
    _traj_log: List[Dict[str, Any]] = field(default_factory=list)
    _mode_step_counts: Dict[str, int] = field(default_factory=dict)
    _comm_state: CommState = field(default_factory=lambda: CommState(0.0, 0.0, 0.0))

    # ── MAVLink 辅助 ──────────────────────────────────────────

    def _wait_msg(self, msg_type: str, timeout: float = 3):
        """Wait for a MAVLink message, return None on timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self.master.recv_match(type=msg_type, blocking=False)
            if m is not None:
                return m
            time.sleep(0.01)
        return None

    def _get_mode(self) -> int:
        """从 HEARTBEAT 读取当前 custom_mode（PX4: 6=OFFBOARD, 4=AUTO, 0=MANUAL）"""
        msg = self._wait_msg("HEARTBEAT", timeout=1)
        if msg:
            return int(msg.custom_mode)
        return -1

    def _get_armed(self) -> bool:
        """检查 UAV 是否已武装"""
        msg = self._wait_msg("HEARTBEAT", timeout=1)
        if msg:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        return False

    def _get_position(self) -> Point2D:
        """从 LOCAL_POSITION_NED 读取位置 (N, E) 米"""
        msg = self._wait_msg("LOCAL_POSITION_NED", timeout=2)
        if msg:
            return (float(msg.x), float(msg.y))
        return (0.0, 0.0)

    def _get_altitude(self) -> float:
        """从 GLOBAL_POSITION_INT 读取相对高度（米）"""
        msg = self._wait_msg("GLOBAL_POSITION_INT", timeout=1)
        if msg:
            return float(msg.relative_alt) / 1000.0  # mm → m
        return 0.0

    def _get_yaw(self) -> float:
        """从 ATTITUDE 读取偏航（弧度）"""
        msg = self._wait_msg("ATTITUDE", timeout=1)
        if msg:
            return float(msg.yaw)
        return 0.0

    def _get_battery(self) -> float:
        """从 SYS_STATUS 读取电量"""
        msg = self._wait_msg("SYS_STATUS", timeout=1)
        if msg:
            return float(msg.battery_remaining) / 100.0
        return 1.0

    def _send_and_stream_setpoints(self, count: int, period_s: float,
                                   vn: float = 0.0, ve: float = 0.0, vd: float = 0.0):
        """连续发送 setpoint 消息（不掉流），用于 Offboard 进入前/中保持"""
        for _ in range(count):
            self.send_velocity(vn, ve, vd)
            time.sleep(period_s)

    # ── 控制 ──────────────────────────────────────────────────

    def send_velocity(self, vn: float, ve: float, vd: float = 0.0, yaw: float = 0.0):
        """通过 SET_POSITION_TARGET_LOCAL_NED 发送速度指令（Offboard 模式）"""
        if self.dry_run:
            return  # 离线模式不发送实际指令
        # MAV_FRAME_LOCAL_NED = 1, type_mask: 仅速度 (0b0000111111000111)
        type_mask = 0b0000111111000111  # ignore pos/accel/yaw, use velocity
        self.master.mav.set_position_target_local_ned_send(
            0,                          # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0.0, 0.0, 0.0,              # x, y, z (position - ignored)
            vn, ve, vd,                  # vx, vy, vz (velocity)
            0.0, 0.0, 0.0,              # afx, afy, afz (accel - ignored)
            yaw, 0.0)                    # yaw, yaw_rate

    def arm_and_offboard(self) -> bool:
        """进入 Offboard 模式并 Arm（带完整验证，不掉流）"""

        # ── 1) 降低传感器频率 ──────────────────────────────────
        print("[PX4] Setting sensor params...")
        self.master.mav.param_set_send(
            self.master.target_system, self.master.target_component,
            b'IMU_INTEG_RATE', 100,
            mavutil.mavlink.MAV_PARAM_TYPE_INT8)
        time.sleep(0.1)

        # ── 2) 等待 GPS 3D fix (PX4 arm 前提条件) ───────────────
        print("[PX4] Waiting for GPS 3D fix...")
        for i in range(30):
            msg = self._wait_msg("GPS_RAW_INT", timeout=0.5)
            if msg and int(msg.fix_type) >= 3:
                print(f"  ✓ GPS 3D fix, sats={msg.satellites_visible}")
                break
            if i % 5 == 0:
                fix_str = f"fix_type={msg.fix_type}" if msg else "no GPS msg"
                print(f"  Waiting GPS... ({fix_str})")
        else:
            print("  ⚠️ No GPS 3D fix after 15s, continuing anyway")

        # ── 3) 建立连续 setpoint 流（≥1s 后才能切 Offboard） ─────
        print("[PX4] Sending initial setpoint stream (1.5s @ 50Hz)...")
        self._send_and_stream_setpoints(75, 0.02)  # 75 × 0.02s = 1.5s

        # ── 4) 保持 setpoint 流的同时切换到 Offboard 模式 ───────
        print("[PX4] Switching to Offboard mode...")
        for attempt in range(3):
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                6)  # PX4: 6 = OFFBOARD

            # setpoint 流不能断 — 在等模式切换时连续发送
            self._send_and_stream_setpoints(10, 0.05)  # 10 × 0.05s = 0.5s

            mode = self._get_mode()
            if mode == 6:
                print("  ✓ Offboard mode confirmed")
                break
            print(f"  ⚠️ Mode={mode}, retrying... (attempt {attempt + 1})")
        else:
            print("  ⚠️ Could not verify Offboard mode, continuing anyway")

        # ── 5) 保持 setpoint 流的同时 Arm ──────────────────────
        print("[PX4] Arming...")
        for attempt in range(3):
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

            # 在等 arm ack 期间持续发 setpoint
            ack = None
            t0 = time.time()
            while time.time() - t0 < 3.0:
                self.send_velocity(0.0, 0.0, 0.0)
                raw = self.master.recv_match(type="COMMAND_ACK", blocking=False)
                if raw is not None and int(raw.command) == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    ack = raw
                    break
                time.sleep(0.05)

            if ack is not None:
                result = int(ack.result)
                if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print("  ✓ Armed (ACCEPTED)")
                    break
                else:
                    result_names = {0: "ACCEPTED", 1: "DENIED", 2: "UNSUPPORTED",
                                    3: "FAILED", 4: "IN_PROGRESS"}
                    rn = result_names.get(result, f"UNKNOWN({result})")
                    print(f"  ⚠️ Arm result={rn}, retrying...")
            else:
                print("  ⚠️ No arm ACK after 3s, retrying...")
        else:
            print("  ⚠️ Arm may have failed, continuing anyway")
            print(f"  Current armed={self._get_armed()}, mode={self._get_mode()}")

        time.sleep(0.2)
        return True

    def land(self):
        """降落"""
        print("[PX4] Landing...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(1)
        print("[PX4] Land command sent")

    # ── FDLC 快环循环 ────────────────────────────────────────

    def takeoff(self, altitude_m: float = 15.0):
        """起飞: 持续发送向上速度指令，不掉流"""
        print(f"[PX4] Takeoff to {altitude_m}m...")
        t0 = time.time()
        timeout_s = 25.0  # 给 PX4 更多容错时间

        climbing_logged = False
        while time.time() - t0 < timeout_s:
            current_alt = self._get_altitude()
            if current_alt >= altitude_m:
                if not climbing_logged:
                    print(f"  ✓ Reached {altitude_m}m (current alt={current_alt:.1f}m)")
                    climbing_logged = True
                # 高度达标后悬停
                self.send_velocity(0.0, 0.0, 0.0)
                time.sleep(0.1)
                continue

            # 爬升阶段: 高度越低 -> 速度越大
            remaining = altitude_m - current_alt
            if current_alt < 1.0:
                vd = -2.0  # 初始快速离地
            elif remaining > altitude_m * 0.3:
                vd = -1.0
            else:
                vd = -0.4  # 接近目标慢速爬升

            self.send_velocity(0.0, 0.0, vd)

            elapsed = time.time() - t0
            if elapsed % 1.0 < 0.15 and not climbing_logged:
                armed = self._get_armed()
                mode = self._get_mode()
                print(f"  alt={current_alt:.1f}m, vd={vd:.1f}, "
                      f"armed={armed}, mode={mode}")

            time.sleep(0.1)  # 10Hz setpoint 流

        if not climbing_logged:
            print(f"  ⚠️ Takeoff timeout — alt after {timeout_s}s: {self._get_altitude():.1f}m")

        # 悬停保持（持续发 setpoint）
        print(f"[PX4] Hover hold...")
        for _ in range(20):
            self.send_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)

        final_alt = self._get_altitude()
        print(f"  Altitude after hold: {final_alt:.1f}m")

    def run_episode(self) -> List[Dict]:
        dt = 1.0 / max(1, self.env.cfg.step_hz)
        self._init_slow_plan()
        self.fast_loop.reset()
        self.env.reset()
        self._start_time = time.time()
        self._step = 0
        self._traj_log.clear()
        self._mode_step_counts.clear()
        self._comm_state = CommState(0.0, 0.0, 0.0)

        print(f"[FDLC] Starting fast loop at {self.env.cfg.step_hz} Hz...\n")

        while time.time() - self._start_time < self.duration_s:
            loop_start = time.time()

            # a) 读取位置（PX4 真实位置 或 离线仿真位置）
            if self.dry_run:
                # 离线模式：用 FDLC env 内部的运动学仿真
                from uavlab.paper1.contracts.contract_types import SlowPlan as SP
                plan = self._exec_plan or SP(goal_id=None, goal_ne=(0,0))
                self.env.step_fast(target_ne=plan.goal_ne, vel_ne_cmd=None, dt=1.0/self.env.cfg.step_hz)
                uav_pos = self.env.pos_ne
                uav_yaw = self.env.yaw_rad
            else:
                uav_pos = self._get_position()
                uav_yaw = self._get_yaw()

            # b) 覆盖 FDLC env 状态
            self.env.pos_ne = uav_pos
            self.env.yaw_rad = uav_yaw
            self.env.t = self._step

            # c) 基于真实距离计算通信链路
            comm = compute_link_from_distance(
                uav_pos, self.env.cfg.gcs_ne,
                enable_distance_decay=self.env.cfg.enable_distance_decay,
                d0=self.env.cfg.distance_d0_m,
                d1=self.env.cfg.distance_d1_m,
                lmin=self.env.cfg.distance_loss_min,
                lmax=self.env.cfg.distance_loss_max,
                base_loss=self.env.cfg.base_loss,
                delay_s=self.env.cfg.delay_mean_s,
            )
            self._comm_state = comm

            # d) FastObservation
            obs = FastObservation(
                step=self._step, pos_ne=uav_pos,
                comm_mode=self.env.comm_mode,
                backlog_bits=float(self.env.backlog_bits),
                link_loss_p=float(comm.loss_p),
            )

            # e) FDLC FastLoop 决策
            cmd = self.fast_loop.step(obs=obs, plan=self._exec_plan, dt=dt)
            self.env.comm_mode = cmd.next_comm_mode
            mode_key = _mode_str(cmd.next_comm_mode).strip().lower()
            self._mode_step_counts[mode_key] = self._mode_step_counts.get(mode_key, 0) + 1

            # f) 航点推进
            self.maybe_advance_waypoint()

            # g) POI 覆盖 + backlog 注入
            for poi in self.env.pois:
                if poi.poi_id == self._exec_plan.goal_id and poi.poi_id not in self.env.covered:
                    d = math.hypot(uav_pos[0] - poi.pos_ne[0], uav_pos[1] - poi.pos_ne[1])
                    if d <= self.env.cfg.visit_radius_m:
                        if poi.poi_id not in self.env.covered:
                            self.env.covered.add(poi.poi_id)
                            self.env.backlog_bits += poi.key_bits
                            print(f"  → POI {poi.poi_id} covered, backlog={self.env.backlog_bits:.0f} bits")

            # h) 模拟回传
            if _mode_str(cmd.next_comm_mode).strip().lower() in ("stx", "srec") and self.env.backlog_bits > 0:
                b_eff = 1_000_000.0 * (1.0 - comm.loss_p)
                self.env.backlog_bits = max(0.0, self.env.backlog_bits - b_eff * dt)

            # i) 计算速度指令
            if cmd.vel_ne_cmd is not None:
                vn, ve = float(cmd.vel_ne_cmd[0]), float(cmd.vel_ne_cmd[1])
            else:
                tx, ty = cmd.target_ne
                vn = (tx - uav_pos[0]) * 1.0
                ve = (ty - uav_pos[1]) * 1.0

            # 限幅
            spd = math.hypot(vn, ve)
            vmax = self.env.cfg.v_xy_max
            if spd > vmax > 1e-9:
                s = vmax / spd
                vn *= s; ve *= s

            # 到达减速
            if self._exec_plan.goal_id is not None:
                dg = math.hypot(uav_pos[0] - self._exec_plan.goal_ne[0],
                                uav_pos[1] - self._exec_plan.goal_ne[1])
                slow_r = self.env.cfg.visit_radius_m * 3.0
                if 0.1 < dg < slow_r:
                    ratio = max(0.3, dg / slow_r)
                    vn *= ratio; ve *= ratio

            # j) 发送速度指令到 PX4
            self.send_velocity(vn, ve, 0.0)

            # j2) 定期检查 PX4 实际模式（非 dry-run）
            if not self.dry_run and self._step % max(1, int(self.env.cfg.step_hz * 5)) == 0:
                armed = self._get_armed()
                mode = self._get_mode()
                pos_px4 = self._get_position()
                print(f"  [PX4 MON] step={self._step} armed={armed} "
                      f"mode={mode} pos=({pos_px4[0]:.1f},{pos_px4[1]:.1f}) "
                      f"cmd=({vn:.1f},{ve:.1f})")

            # k) 日志
            self._traj_log.append({
                "t_wall_s": round(time.time() - self._start_time, 3),
                "step": self._step,
                "pos_ne": [round(uav_pos[0], 2), round(uav_pos[1], 2)],
                "yaw_rad": round(uav_yaw, 3),
                "vel_cmd": [round(vn, 3), round(ve, 3)],
                "comm_loss_p": round(comm.loss_p, 4),
                "comm_delay_s": round(comm.delay_s, 4),
                "comm_bw_bps": round(comm.bandwidth_bps, 1),
                "fsm_mode": _mode_str(cmd.next_comm_mode),
                "goal_id": self._exec_plan.goal_id,
                "goal_ne": [round(self._exec_plan.goal_ne[0], 2),
                            round(self._exec_plan.goal_ne[1], 2)],
                "backlog_bits": round(self.env.backlog_bits, 1),
            })

            # 进度
            if self._step % max(1, int(self.env.cfg.step_hz * 2)) == 0:
                print(f"  t={time.time()-self._start_time:.1f}s "
                      f"step={self._step} "
                      f"pos=({uav_pos[0]:.0f},{uav_pos[1]:.0f}) "
                      f"loss={comm.loss_p:.2f} "
                      f"mode={_mode_str(cmd.next_comm_mode)} "
                      f"goal={self._exec_plan.goal_id}")

            self._step += 1

            # l) 保持频率
            elapsed = time.time() - loop_start
            sleep_s = max(0.0, dt - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

        print(f"[FDLC] Episode complete: {self._step} steps, "
              f"modes={dict(self._mode_step_counts)}")
        return self._traj_log

    def _init_slow_plan(self):
        if self.waypoints:
            wp = self.waypoints[0]
            self._exec_plan = SlowPlan(goal_id=0, goal_ne=(float(wp[0]), float(wp[1])))
        else:
            gcs = self.env.cfg.gcs_ne
            self._exec_plan = SlowPlan(goal_id=None, goal_ne=(float(gcs[0]), float(gcs[1])))

    def maybe_advance_waypoint(self) -> bool:
        if not self.waypoints or self._exec_plan is None:
            return False
        gid = self._exec_plan.goal_id
        if gid is None or gid >= len(self.waypoints) - 1:
            return False
        d = math.hypot(self.env.pos_ne[0] - self._exec_plan.goal_ne[0],
                       self.env.pos_ne[1] - self._exec_plan.goal_ne[1])
        if d <= self.env.cfg.visit_radius_m:
            new_id = int(gid) + 1
            new_wp = self.waypoints[new_id]
            self._exec_plan = SlowPlan(goal_id=new_id, goal_ne=(float(new_wp[0]), float(new_wp[1])))
            print(f"  → Advance to waypoint {new_id}: {new_wp}")
            return True
        return False


# ═══════════════════════════════════════════════════════════════
# 3. 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Phase 1: FDLC × PX4+Gazebo (pymavlink)")
    p.add_argument("--config", type=str, default="configs/experiments/paper1/cases/phase1_px4.yaml")
    p.add_argument("--system", type=str, default="configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml")
    p.add_argument("--duration_s", type=float, default=120.0)
    p.add_argument("--run_dir", type=str, default="")
    p.add_argument("--waypoints", type=str, default="")
    p.add_argument("--dry-run", action="store_true",
                   help="离线模式：不连接 PX4，直接跑 FDLC 快环仿真验证 FSM")
    args = p.parse_args()

    # ── 加载 FDLC 配置 ────────────────────────────────────────
    cfg = load_resolved_config(args.config)
    if args.system:
        cfg = _deep_merge(cfg, load_resolved_config(args.system))
    cfg = apply_experiment_presets(cfg)

    scene_path = cfg.get("scene_file", "configs/scenes/phase1_px4.yaml")
    sim_cfg = from_resolved_config(cfg, load_scene_yaml(scene_path))
    env = build_env(sim_cfg, load_scene_config(scene_path))
    env.reset()

    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=sim_cfg.waypoint_delta_max_m)
    fast_params = FastLoopParams.from_contract(contract)
    fast_loop = FastLoop(env=env, params=fast_params, contract=contract)

    # ── 航点 ──────────────────────────────────────────────────
    waypoints: List[Point2D] = []
    if args.waypoints.strip():
        for seg in args.waypoints.split(";"):
            pts = seg.strip().split(",")
            if len(pts) >= 2:
                waypoints.append((float(pts[0]), float(pts[1])))
    else:
        waypoints = list(sim_cfg.poi_list)

    # ── 输出目录 ──────────────────────────────────────────────
    run_dir: Optional[Path] = None
    if args.run_dir.strip():
        run_dir = Path(args.run_dir).resolve()
    else:
        from datetime import datetime
        run_dir = Path(f"runs/debug/phase1_fdlc_px4_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    # ── 打印配置摘要 ──────────────────────────────────────────
    print("=" * 60)
    print("FDLC × PX4+Gazebo Phase 1 Bridge (pymavlink)")
    print("=" * 60)
    print(f"  Config:     {args.config}")
    print(f"  System:     {args.system}")
    print(f"  Structure:  {contract.structure}")
    print(f"  Step Hz:    {sim_cfg.step_hz}")
    print(f"  Duration:   {args.duration_s}s")
    print(f"  V cruise:   {sim_cfg.v_xy_cruise} m/s")
    print(f"  V max:      {sim_cfg.v_xy_max} m/s")
    print(f"  GCS pos:    {sim_cfg.gcs_ne}")
    print(f"  POIs:       {len(waypoints)} points -> {waypoints}")
    print(f"  FSM:        fsm={fast_params.enable_fsm} back={fast_params.enable_back_mode} "
          f"safe={fast_params.enable_safety_mode} rec={fast_params.enable_recovery_mode}")
    print(f"  Comm model: distance_decay={sim_cfg.enable_distance_decay}, "
          f"d0={sim_cfg.distance_d0_m}, d1={sim_cfg.distance_d1_m}")
    print(f"  Dual link:  ctrl_max_loss={contract.dual_link.control_max_loss_p}, "
          f"data_max_loss={contract.dual_link.data_max_loss_p}")
    print(f"  Mode:       {'DRY-RUN (offline)' if args.dry_run else 'PX4+Gazebo'}")
    print(f"  Output dir: {run_dir}")
    print("=" * 60)

    # ── 连接 PX4（dry-run 时跳过） ────────────────────────────
    master = None
    if not args.dry_run:
        print(f"\n[PX4] Connecting via pymavlink (udpin:0.0.0.0:14540)...")
        master = mavutil.mavlink_connection('udpin:0.0.0.0:14540')
        master.wait_heartbeat(timeout=15)
        print(f"  ✓ Heartbeat from system={master.target_system}, comp={master.target_component}")

        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
        time.sleep(1)
        print("  ✓ Data stream requested")
    else:
        print(f"\n[DRY-RUN] 离线模式：跳过 PX4 连接，使用内部运动学仿真")

    # ── 构建桥接器 ────────────────────────────────────────────
    bridge = Px4Bridge(
        cfg=sim_cfg, contract=contract, fast_params=fast_params,
        env=env, fast_loop=fast_loop, master=master,
        waypoints=waypoints, run_dir=run_dir, duration_s=args.duration_s,
        dry_run=args.dry_run,
    )

    # ── 运行 ──────────────────────────────────────────────────
    try:
        if not args.dry_run:
            bridge.arm_and_offboard()
            bridge.takeoff(altitude_m=15.0)
        traj = bridge.run_episode()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if not args.dry_run:
            bridge.land()
        sys.exit(1)

    if not args.dry_run:
        bridge.land()

    # ── 保存日志 ──────────────────────────────────────────────
    traj_path = run_dir / "traj.jsonl"
    with traj_path.open("w") as f:
        for row in traj:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[LOG] Trajectory saved: {traj_path} ({len(traj)} steps)")

    total = sum(bridge._mode_step_counts.values()) or 1
    print("\n[STATS] FSM mode time ratios:")
    for mode, count in sorted(bridge._mode_step_counts.items()):
        print(f"  {mode:>10s}: {count:4d} steps ({count/total*100:.1f}%)")

    losses = [r["comm_loss_p"] for r in traj]
    print(f"\n[STATS] Link loss: min={min(losses):.3f} "
          f"max={max(losses):.3f} mean={sum(losses)/max(1,len(losses)):.3f}")

    print("\n✅ Phase 1 complete.")


if __name__ == "__main__":
    main()
