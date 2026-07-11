#!/usr/bin/env python3
"""
Paper: paper1_v2
Purpose: Level 2 deployment validation — PX4 SITL + Gazebo closed-loop timing.
         Runs FDLC fast loop against PX4 Offboard mode, records wall-clock
         timing at each stage, and compares SITL trajectory against pure sim.
Inputs: PX4 SITL running (HEADLESS=1 make px4_sitl gazebo-classic_iris)
        FDLC configs (phase1_px4 scene or C3 scenes)
Outputs: results_v2/level2_sitl/<timestamp>/
         - traj_sitl.jsonl   (SITL trajectory + timing per step)
         - traj_sim.jsonl    (pure sim trajectory for comparison)
         - timing_stats.json (aggregate timing statistics)
         - summary.md        (human-readable summary)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymavlink import mavutil

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


# ═══════════════════════════════════════════════════════════════════
# 1. Communication model (distance-based, same as Level 1)
# ═══════════════════════════════════════════════════════════════════

def compute_link_from_distance(
    uav_ne: Point2D, gcs_ne: Point2D,
    enable_distance_decay: bool = True,
    d0: float = 0.0, d1: float = 150.0,
    lmin: float = 0.05, lmax: float = 0.50,
    base_loss: float = 0.05, delay_s: float = 0.10,
) -> CommState:
    x, y = uav_ne; gx, gy = gcs_ne
    d = math.hypot(x - gx, y - gy)
    if enable_distance_decay:
        loss = lmin + (lmax - lmin) * min(1.0, max(0.0, (d - d0) / max(1e-9, d1 - d0)))
    else:
        loss = base_loss
    loss = min(max(loss, 0.0), 0.99)
    bw = 1_000_000.0 * (1.0 - loss)
    return CommState(loss_p=loss, delay_s=delay_s, bandwidth_bps=bw)


# ═══════════════════════════════════════════════════════════════════
# 2. PX4 SITL Manager (start/stop PX4 in HEADLESS mode)
# ═══════════════════════════════════════════════════════════════════

class Px4SitlManager:
    """Manage PX4 SITL process lifecycle."""

    def __init__(self, px4_home: str = "", headless: bool = True):
        self.px4_home = Path(px4_home) if px4_home else Path.home() / "PX4-Autopilot"
        self.headless = headless
        self._proc: Optional[subprocess.Popen] = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout_s: float = 60.0) -> bool:
        """Start PX4 SITL in HEADLESS mode. Returns True when MAVSDK port is ready."""
        if not self.px4_home.is_dir():
            print(f"[ERROR] PX4-Autopilot not found at {self.px4_home}")
            return False

        env = os.environ.copy()
        env["HEADLESS"] = "1" if self.headless else "0"
        env["PX4_SIM_SPEED_FACTOR"] = "1.0"

        cmd = ["make", "px4_sitl", "gazebo-classic_iris"]
        print(f"[PX4 SITL] Starting: HEADLESS={env['HEADLESS']} make px4_sitl gazebo-classic_iris")
        print(f"[PX4 SITL] Working dir: {self.px4_home}")

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.px4_home),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # Wait for MAVSDK port 14540
        print(f"[PX4 SITL] Waiting for MAVSDK port 14540 (timeout={timeout_s}s)...")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self._proc.poll() is not None:
                print(f"[ERROR] PX4 process exited with code {self._proc.returncode}")
                return False
            try:
                result = subprocess.run(
                    ["ss", "-tulpn"], capture_output=True, text=True, timeout=5
                )
                if ":14540" in result.stdout:
                    elapsed = time.time() - t0
                    print(f"[PX4 SITL] ✓ MAVSDK port 14540 ready ({elapsed:.1f}s)")
                    # Extra wait for full initialization
                    time.sleep(3.0)
                    return True
            except Exception:
                pass
            time.sleep(2.0)

        print(f"[ERROR] Timeout waiting for PX4 SITL port 14540")
        return False

    def stop(self):
        """Kill PX4 SITL and Gazebo processes."""
        if self._proc is not None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._proc = None

        # Clean up any remaining Gazebo/PX4 processes
        for name in ["gzserver", "gzclient", "px4"]:
            subprocess.run(["pkill", "-f", name], capture_output=True)
        time.sleep(1.0)
        print("[PX4 SITL] Stopped.")


# ═══════════════════════════════════════════════════════════════════
# 3. Level 2 SITL Bridge (robust arm/offboard + timing instrumentation)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TimingRecord:
    """Per-step wall-clock timing breakdown."""
    step: int = 0
    t_wall_s: float = 0.0           # wall-clock since episode start
    t_read_telemetry_ms: float = 0.0  # MAVLink recv time
    t_fdlc_decision_ms: float = 0.0   # FastLoop.step() time
    t_send_command_ms: float = 0.0    # MAVLink send time
    t_sleep_ms: float = 0.0           # sleep to maintain rate
    t_loop_total_ms: float = 0.0      # total per-step wall time
    deadline_miss: bool = False       # loop_total > dt_budget


@dataclass
class Level2Bridge:
    """Level 2 SITL bridge with robust arm/offboard and timing instrumentation."""

    # FDLC components
    cfg: Any = None
    contract: Any = None
    fast_params: Any = None
    env: Any = None
    fast_loop: Any = None

    # MAVLink
    master: Any = None

    # Experiment config
    waypoints: List[Point2D] = field(default_factory=list)
    run_dir: Optional[Path] = None
    duration_s: float = 120.0
    step_hz: int = 5

    # State
    _step: int = 0
    _exec_plan: Optional[SlowPlan] = None
    _start_time: float = 0.0
    _comm_state: CommState = field(default_factory=lambda: CommState(0.0, 0.0, 0.0))
    _mode_step_counts: Dict[str, int] = field(default_factory=dict)

    # Logs
    traj_log: List[Dict] = field(default_factory=list)
    timing_log: List[TimingRecord] = field(default_factory=list)

    # Background setpoint thread
    _setpoint_thread: Optional[threading.Thread] = None
    _setpoint_running: bool = False
    _setpoint_lock: Any = field(default_factory=threading.Lock)
    _last_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── Background setpoint stream (keeps Offboard alive) ──────

    def _start_setpoint_stream(self, hz: int = 20):
        """Start a background thread that continuously sends velocity setpoints."""
        if self._setpoint_running:
            return
        self._setpoint_running = True
        period_s = 1.0 / max(1, hz)

        def _stream():
            while self._setpoint_running:
                with self._setpoint_lock:
                    vn, ve, vd = self._last_cmd
                self._send_velocity_raw(vn, ve, vd)
                time.sleep(period_s)

        self._setpoint_thread = threading.Thread(target=_stream, daemon=True)
        self._setpoint_thread.start()
        print(f"[Setpoint] Background stream started @ {hz}Hz")

    def _stop_setpoint_stream(self):
        self._setpoint_running = False
        if self._setpoint_thread is not None:
            self._setpoint_thread.join(timeout=2.0)
        print("[Setpoint] Background stream stopped")

    def _send_velocity_raw(self, vn: float, ve: float, vd: float = 0.0):
        """Send velocity setpoint via MAVLink (raw, no timing)."""
        if self.master is None:
            return
        type_mask = 0b0000111111000111  # ignore pos/accel/yaw_rate, use velocity
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0.0, 0.0, 0.0,          # x, y, z (position - ignored)
            vn, ve, vd,              # vx, vy, vz (velocity)
            0.0, 0.0, 0.0,          # afx, afy, afz (accel - ignored)
            0.0, 0.0,               # yaw, yaw_rate
        )

    def send_velocity(self, vn: float, ve: float, vd: float = 0.0):
        """Send velocity command (updates background stream setpoint)."""
        with self._setpoint_lock:
            self._last_cmd = (float(vn), float(ve), float(vd))
        # Also send immediately
        self._send_velocity_raw(float(vn), float(ve), float(vd))

    # ── Telemetry helpers ──────────────────────────────────────

    def _wait_msg(self, msg_type: str, timeout: float = 3.0) -> Any:
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self.master.recv_match(type=msg_type, blocking=False)
            if m is not None:
                return m
            time.sleep(0.005)
        return None

    # PX4 custom_mode encoding on little-endian:
    #   union { uint16_t reserved; uint8_t main_mode; uint8_t sub_mode; };
    #   custom_mode = (sub_mode << 24) | (main_mode << 16) | reserved
    PX4_OFFBOARD_CUSTOM_MODE: int = 6 << 16  # PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
    PX4_AUTO_LOITER_CUSTOM_MODE: int = 4 << 16  # PX4_CUSTOM_MAIN_MODE_AUTO = 4

    def _get_mode(self) -> int:
        """Return PX4 main_mode (byte 2 of custom_mode)."""
        msg = self._wait_msg("HEARTBEAT", timeout=1.0)
        if msg:
            return (int(msg.custom_mode) >> 16) & 0xFF
        return -1

    def _is_offboard(self) -> bool:
        """Check if PX4 is in OFFBOARD mode."""
        return self._get_mode() == 6  # PX4_CUSTOM_MAIN_MODE_OFFBOARD

    def _get_armed(self) -> bool:
        msg = self._wait_msg("HEARTBEAT", timeout=1.0)
        if msg:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        return False

    def _get_position(self) -> Point2D:
        msg = self._wait_msg("LOCAL_POSITION_NED", timeout=0.5)
        if msg:
            return (float(msg.x), float(msg.y))
        return (0.0, 0.0)

    def _get_altitude(self) -> float:
        msg = self._wait_msg("GLOBAL_POSITION_INT", timeout=0.5)
        if msg:
            return float(msg.relative_alt) / 1000.0
        return 0.0

    def _get_yaw(self) -> float:
        msg = self._wait_msg("ATTITUDE", timeout=0.5)
        if msg:
            return float(msg.yaw)
        return 0.0

    # ── Robust Arm + Offboard sequence ─────────────────────────

    def arm_and_offboard(self) -> bool:
        """Robust arm+offboard sequence.

        Key fix over v1: the setpoint stream MUST be continuous from BEFORE
        the mode switch through arm confirmation. We start a background thread
        that sends setpoints at 20Hz, then switch mode and arm while the
        stream never stops.
        """
        print("[PX4] === Robust Arm + Offboard Sequence ===")

        # ── 0) Set SITL/HITL parameters ────────────────────────
        print("[PX4] Configuring SITL parameters...")
        sitl_params = [
            ("COM_RCL_EXCEPT", 4),     # Disable RC loss for Offboard
            ("NAV_RCL_ACT", 0),        # Disable RC loss action
        ]
        for name, val in sitl_params:
            self.master.mav.param_set_send(
                self.master.target_system, self.master.target_component,
                name.encode(), float(val),
                mavutil.mavlink.MAV_PARAM_TYPE_INT32,
            )
            time.sleep(0.15)

        # ── 1) Wait for GPS 3D fix ────────────────────────────
        print("[PX4] Waiting for GPS 3D fix...")
        for i in range(40):
            msg = self._wait_msg("GPS_RAW_INT", timeout=0.5)
            fix_type = int(msg.fix_type) if msg else 0
            if fix_type >= 3:
                print(f"  ✓ GPS 3D fix, sats={msg.satellites_visible}")
                break
            if i % 10 == 0:
                print(f"  Waiting... fix_type={fix_type}")
        else:
            print("  ⚠️ No GPS 3D fix after 20s, continuing")

        # ── 2) START background setpoint stream ────────────────
        # CRITICAL: stream must run BEFORE switching to Offboard
        print("[PX4] Starting background setpoint stream (20Hz)...")
        self._start_setpoint_stream(hz=20)
        time.sleep(1.5)  # Let stream stabilize (PX4 needs ≥1s of stream)

        # ── 3) Switch to Offboard mode (stream keeps running) ──
        print("[PX4] Switching to Offboard mode...")
        for attempt in range(5):
            # Send mode change command (PX4 OFFBOARD main_mode=6)
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                self.PX4_OFFBOARD_CUSTOM_MODE,  # = 6 << 16
            )
            time.sleep(0.8)

            mode = self._get_mode()
            if mode == 6:
                print(f"  ✓ Offboard mode confirmed (attempt {attempt + 1})")
                break
            print(f"  Mode={mode}, retry {attempt + 1}/5")
        else:
            print("  ❌ Failed to enter Offboard mode")
            return False

        # ── 4) Arm (stream keeps running) ──────────────────────
        print("[PX4] Arming...")
        for attempt in range(5):
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0,
            )

            # Wait for COMMAND_ACK
            t0 = time.time()
            while time.time() - t0 < 3.0:
                raw = self.master.recv_match(type="COMMAND_ACK", blocking=False)
                if raw is not None and int(raw.command) == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    if int(raw.result) == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        print(f"  ✓ Armed (ACCEPTED, attempt {attempt + 1})")
                        break
                    else:
                        result_names = {0: "ACCEPTED", 1: "DENIED", 2: "UNSUPPORTED",
                                        3: "FAILED", 4: "IN_PROGRESS", 5: "CANCELLED"}
                        rn = result_names.get(int(raw.result), f"UNKNOWN({raw.result})")
                        print(f"  Arm result={rn}, retrying...")
                        break
                time.sleep(0.05)
            else:
                print(f"  No arm ACK after 3s, retrying...")
                continue

            if raw is not None and int(raw.result) == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                break
        else:
            print("  ⚠️ Arm may have failed — checking state...")
            print(f"  armed={self._get_armed()}, mode={self._get_mode()}")

        # ── 5) Verify final state ──────────────────────────────
        time.sleep(0.5)
        msg = self._wait_msg("HEARTBEAT", timeout=3.0)
        if msg:
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            mode = (int(msg.custom_mode) >> 16) & 0xFF
            is_offboard = (mode == 6)
            print(f"[PX4] Final state: armed={armed}, mode={mode}, offboard={is_offboard}")
            return armed and is_offboard
        else:
            print(f"[PX4] ⚠️ No heartbeat for final state check")
            return False

    def takeoff(self, altitude_m: float = 10.0):
        """Takeoff to target altitude using velocity control (stream already running)."""
        print(f"[PX4] Takeoff to {altitude_m}m (velocity-based, stream running)...")
        t0 = time.time()
        timeout_s = 30.0

        while time.time() - t0 < timeout_s:
            current_alt = self._get_altitude()

            if current_alt >= altitude_m:
                print(f"  ✓ Reached {altitude_m}m (alt={current_alt:.1f}m)")
                break

            # Climb rate proportional to remaining distance
            remaining = altitude_m - current_alt
            if current_alt < 1.0:
                vd = -1.5       # Initial liftoff
            elif remaining > altitude_m * 0.3:
                vd = -0.8
            else:
                vd = -0.3       # Slow approach

            self.send_velocity(0.0, 0.0, vd)

            elapsed = time.time() - t0
            if elapsed % 2.0 < 0.2:
                print(f"  alt={current_alt:.1f}m, vd={vd:.1f}, armed={self._get_armed()}, mode={self._get_mode()}")

            time.sleep(0.1)

        # Hover stabilization
        print("[PX4] Hover hold...")
        for _ in range(30):
            self.send_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)

        final_alt = self._get_altitude()
        print(f"  Final altitude: {final_alt:.1f}m")

    def land(self):
        """Land via MAV_CMD_NAV_LAND."""
        print("[PX4] Landing...")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        time.sleep(3.0)

        # Wait for disarm
        for _ in range(50):
            if not self._get_armed():
                print("[PX4] Disarmed.")
                break
            time.sleep(0.5)
        else:
            print("[PX4] Force disarm...")
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0,
            )

    # ── Timing-instrumented velocity send ─────────────────────

    def _measure_send_velocity(self, vn: float, ve: float, vd: float = 0.0) -> float:
        """Send velocity and measure wall-clock time. Returns elapsed ms."""
        t0 = time.perf_counter()
        self.send_velocity(vn, ve, vd)
        return (time.perf_counter() - t0) * 1000.0

    def _measure_read_telemetry(self) -> Tuple[Point2D, float, float]:
        """Read position + yaw and measure timing. Returns (pos, yaw, elapsed_ms)."""
        t0 = time.perf_counter()
        pos = self._get_position()
        yaw = self._get_yaw()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return pos, yaw, elapsed_ms

    # ── FDLC fast loop with timing instrumentation ─────────────

    def run_episode_sitl(self) -> Tuple[List[Dict], List[TimingRecord]]:
        """Run FDLC fast loop against PX4 SITL with full timing instrumentation."""
        dt = 1.0 / max(1, self.step_hz)
        dt_budget_ms = dt * 1000.0  # e.g., 200ms for 5Hz
        self._init_slow_plan()
        self.fast_loop.reset()
        self.env.reset()
        self._start_time = time.time()
        self._step = 0
        self.traj_log.clear()
        self.timing_log.clear()
        self._mode_step_counts.clear()
        self._comm_state = CommState(0.0, 0.0, 0.0)

        # Resume background setpoint stream if stopped
        if not self._setpoint_running:
            self._start_setpoint_stream(hz=20)

        print(f"\n[FDLC×SITL] Starting fast loop @ {self.step_hz}Hz, duration={self.duration_s}s")
        print(f"[FDLC×SITL] Budget: {dt_budget_ms:.0f}ms/step\n")

        while time.time() - self._start_time < self.duration_s:
            loop_start = time.perf_counter()

            # ── a) Read PX4 telemetry (timed) ──────────────────
            uav_pos, uav_yaw, t_read_ms = self._measure_read_telemetry()

            # ── b) Overlay FDLC env state ──────────────────────
            self.env.pos_ne = uav_pos
            self.env.yaw_rad = uav_yaw
            self.env.t = self._step

            # ── c) Compute communication link (distance-based) ──
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

            # ── d) Build FastObservation ───────────────────────
            obs = FastObservation(
                step=self._step, pos_ne=uav_pos,
                comm_mode=self.env.comm_mode,
                backlog_bits=float(self.env.backlog_bits),
                link_loss_p=float(comm.loss_p),
            )

            # ── e) FDLC FastLoop decision (timed) ──────────────
            t0_fdlc = time.perf_counter()
            cmd = self.fast_loop.step(obs=obs, plan=self._exec_plan, dt=dt)
            t_fdlc_ms = (time.perf_counter() - t0_fdlc) * 1000.0

            self.env.comm_mode = cmd.next_comm_mode
            mode_key = str(cmd.next_comm_mode.value if hasattr(cmd.next_comm_mode, "value") else cmd.next_comm_mode)
            mode_key = mode_key.strip().lower()
            self._mode_step_counts[mode_key] = self._mode_step_counts.get(mode_key, 0) + 1

            # ── f) Waypoint advance ────────────────────────────
            self._maybe_advance_waypoint()

            # ── g) POI coverage + backlog ──────────────────────
            for poi in self.env.pois:
                if poi.poi_id == self._exec_plan.goal_id and poi.poi_id not in self.env.covered:
                    d = math.hypot(uav_pos[0] - poi.pos_ne[0], uav_pos[1] - poi.pos_ne[1])
                    if d <= self.env.cfg.visit_radius_m:
                        self.env.covered.add(poi.poi_id)
                        self.env.backlog_bits += poi.key_bits
                        print(f"  → POI {poi.poi_id} covered, backlog={self.env.backlog_bits:.0f} bits")

            # ── h) Backlog return simulation ───────────────────
            if mode_key in ("stx", "srec") and self.env.backlog_bits > 0:
                b_eff = 1_000_000.0 * (1.0 - comm.loss_p)
                self.env.backlog_bits = max(0.0, self.env.backlog_bits - b_eff * dt)

            # ── i) Compute velocity command ────────────────────
            if cmd.vel_ne_cmd is not None:
                vn, ve = float(cmd.vel_ne_cmd[0]), float(cmd.vel_ne_cmd[1])
            else:
                tx, ty = cmd.target_ne
                vn = (tx - uav_pos[0]) * 1.0
                ve = (ty - uav_pos[1]) * 1.0

            # Speed limit
            spd = math.hypot(vn, ve)
            vmax = self.env.cfg.v_xy_max
            if spd > vmax > 1e-9:
                s = vmax / spd
                vn *= s; ve *= s

            # Arrival deceleration
            if self._exec_plan.goal_id is not None:
                dg = math.hypot(uav_pos[0] - self._exec_plan.goal_ne[0],
                                uav_pos[1] - self._exec_plan.goal_ne[1])
                slow_r = self.env.cfg.visit_radius_m * 3.0
                if 0.1 < dg < slow_r:
                    ratio = max(0.3, dg / slow_r)
                    vn *= ratio; ve *= ratio

            # ── j) Send velocity to PX4 (timed) ────────────────
            t_send_ms = self._measure_send_velocity(vn, ve, 0.0)

            # ── k) Log ─────────────────────────────────────────
            self.traj_log.append({
                "t_wall_s": round(time.time() - self._start_time, 3),
                "step": self._step,
                "pos_ne": [round(uav_pos[0], 2), round(uav_pos[1], 2)],
                "yaw_rad": round(uav_yaw, 3),
                "vel_cmd": [round(vn, 3), round(ve, 3)],
                "comm_loss_p": round(comm.loss_p, 4),
                "comm_delay_s": round(comm.delay_s, 4),
                "fsm_mode": mode_key,
                "goal_id": self._exec_plan.goal_id,
                "backlog_bits": round(self.env.backlog_bits, 1),
            })

            # ── l) Record timing ───────────────────────────────
            loop_end = time.perf_counter()
            t_loop_total_ms = (loop_end - loop_start) * 1000.0

            timing = TimingRecord(
                step=self._step,
                t_wall_s=round(time.time() - self._start_time, 4),
                t_read_telemetry_ms=round(t_read_ms, 4),
                t_fdlc_decision_ms=round(t_fdlc_ms, 4),
                t_send_command_ms=round(t_send_ms, 4),
                t_sleep_ms=0.0,  # filled below
                t_loop_total_ms=round(t_loop_total_ms, 4),
                deadline_miss=(t_loop_total_ms > dt_budget_ms),
            )
            self.timing_log.append(timing)

            # Progress
            if self._step % max(1, int(self.step_hz * 2)) == 0:
                print(f"  t={time.time()-self._start_time:.1f}s "
                      f"step={self._step} "
                      f"pos=({uav_pos[0]:.0f},{uav_pos[1]:.0f}) "
                      f"loss={comm.loss_p:.2f} "
                      f"mode={mode_key} "
                      f"loop={t_loop_total_ms:.1f}ms "
                      f"{'⚠️DEADLINE' if timing.deadline_miss else '✓'}")

            self._step += 1

            # ── m) Rate control ────────────────────────────────
            remaining_ms = dt_budget_ms - t_loop_total_ms
            if remaining_ms > 0:
                t0_sleep = time.perf_counter()
                time.sleep(remaining_ms / 1000.0)
                t_sleep_ms = (time.perf_counter() - t0_sleep) * 1000.0
                self.timing_log[-1].t_sleep_ms = round(t_sleep_ms, 4)
            else:
                self.timing_log[-1].t_sleep_ms = 0.0

        print(f"\n[FDLC×SITL] Episode complete: {self._step} steps")
        print(f"[FDLC×SITL] Modes: {dict(self._mode_step_counts)}")
        return self.traj_log, self.timing_log

    # ── Pure simulation episode (for comparison) ───────────────

    def run_episode_sim(self, takeoff_altitude_m: float = 8.0,
                        takeoff_climb_rate: float = 1.5) -> Tuple[List[Dict], List[TimingRecord]]:
        """Run FDLC fast loop in pure simulation mode for comparison.

        Includes a simulated takeoff phase to match SITL: UAV holds at (0,0)
        while altitude simulates climbing, then begins horizontal flight.
        """
        dt = 1.0 / max(1, self.step_hz)
        dt_budget_ms = dt * 1000.0
        max_steps = int(self.duration_s * self.step_hz)

        # Compute takeoff duration to match SITL climb rate
        takeoff_duration_s = takeoff_altitude_m / max(0.1, takeoff_climb_rate)
        takeoff_steps = int(takeoff_duration_s * self.step_hz)
        _sim_altitude = 0.0  # simulated altitude during takeoff

        self._init_slow_plan()
        self.fast_loop.reset()
        self.env.reset()
        self._start_time = time.time()
        self._step = 0
        self.traj_log.clear()
        self.timing_log.clear()
        self._mode_step_counts.clear()
        self._comm_state = CommState(0.0, 0.0, 0.0)

        print(f"\n[FDLC×SIM] Starting fast loop @ {self.step_hz}Hz, "
              f"duration={self.duration_s}s, max_steps={max_steps}")
        print(f"[FDLC×SIM] Takeoff simulation: {takeoff_steps} steps "
              f"({takeoff_duration_s:.1f}s climb to {takeoff_altitude_m}m)\n")

        from uavlab.paper1.contracts.contract_types import SlowPlan as SP

        while self._step < max_steps:
            loop_start = time.perf_counter()

            # ── a) Simulated takeoff phase ────────────────────────
            if self._step < takeoff_steps:
                # Hold horizontal position at origin; simulate climb
                _sim_altitude = min(takeoff_altitude_m,
                                    _sim_altitude + takeoff_climb_rate * dt)
                # UAV stays at (0,0) during climb — no horizontal motion
                uav_pos = (0.0, 0.0)
                uav_yaw = 0.0
                t_step_ms = 0.0
                # Use takeoff velocity command (zero horizontal)
                vn, ve = 0.0, 0.0
            else:
                # ── b) Normal simulation ──────────────────────────
                plan = self._exec_plan or SP(goal_id=None, goal_ne=(0, 0))
                t0_step = time.perf_counter()
                self.env.step_fast(target_ne=plan.goal_ne, vel_ne_cmd=None, dt=dt)
                uav_pos = self.env.pos_ne
                uav_yaw = self.env.yaw_rad
                t_step_ms = (time.perf_counter() - t0_step) * 1000.0

            # b) Communication link
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

            # c) FastObservation
            obs = FastObservation(
                step=self._step, pos_ne=uav_pos,
                comm_mode=self.env.comm_mode,
                backlog_bits=float(self.env.backlog_bits),
                link_loss_p=float(comm.loss_p),
            )

            # d) FDLC decision
            t0_fdlc = time.perf_counter()
            cmd = self.fast_loop.step(obs=obs, plan=self._exec_plan, dt=dt)
            t_fdlc_ms = (time.perf_counter() - t0_fdlc) * 1000.0

            self.env.comm_mode = cmd.next_comm_mode
            mode_key = str(cmd.next_comm_mode.value if hasattr(cmd.next_comm_mode, "value") else cmd.next_comm_mode)
            mode_key = mode_key.strip().lower()
            self._mode_step_counts[mode_key] = self._mode_step_counts.get(mode_key, 0) + 1

            # e) Waypoint + POI
            self._maybe_advance_waypoint()
            for poi in self.env.pois:
                if poi.poi_id == self._exec_plan.goal_id and poi.poi_id not in self.env.covered:
                    d = math.hypot(uav_pos[0] - poi.pos_ne[0], uav_pos[1] - poi.pos_ne[1])
                    if d <= self.env.cfg.visit_radius_m:
                        self.env.covered.add(poi.poi_id)
                        self.env.backlog_bits += poi.key_bits

            if mode_key in ("stx", "srec") and self.env.backlog_bits > 0:
                b_eff = 1_000_000.0 * (1.0 - comm.loss_p)
                self.env.backlog_bits = max(0.0, self.env.backlog_bits - b_eff * dt)

            # f) Compute velocity for logging
            if cmd.vel_ne_cmd is not None:
                vn, ve = float(cmd.vel_ne_cmd[0]), float(cmd.vel_ne_cmd[1])
            else:
                tx, ty = cmd.target_ne
                vn = (tx - uav_pos[0]) * 1.0
                ve = (ty - uav_pos[1]) * 1.0
            spd = math.hypot(vn, ve)
            vmax = self.env.cfg.v_xy_max
            if spd > vmax > 1e-9:
                s = vmax / spd; vn *= s; ve *= s

            # ── Override: no horizontal motion during takeoff ────
            if self._step < takeoff_steps:
                vn, ve = 0.0, 0.0

            # g) Log
            self.traj_log.append({
                "t_wall_s": round(time.time() - self._start_time, 3),
                "step": self._step,
                "pos_ne": [round(uav_pos[0], 2), round(uav_pos[1], 2)],
                "yaw_rad": round(uav_yaw, 3),
                "vel_cmd": [round(vn, 3), round(ve, 3)],
                "comm_loss_p": round(comm.loss_p, 4),
                "comm_delay_s": round(comm.delay_s, 4),
                "fsm_mode": mode_key,
                "goal_id": self._exec_plan.goal_id,
                "backlog_bits": round(self.env.backlog_bits, 1),
            })

            self.timing_log.append(TimingRecord(
                step=self._step,
                t_wall_s=round(time.time() - self._start_time, 4),
                t_read_telemetry_ms=round(t_step_ms, 4),
                t_fdlc_decision_ms=round(t_fdlc_ms, 4),
                t_send_command_ms=0.0,
                t_loop_total_ms=round((time.perf_counter() - loop_start) * 1000.0, 4),
            ))

            self._step += 1

            if self._step % max(1, int(self.step_hz * 2)) == 0:
                print(f"  t={time.time()-self._start_time:.1f}s step={self._step} "
                      f"pos=({uav_pos[0]:.0f},{uav_pos[1]:.0f}) "
                      f"loss={comm.loss_p:.2f} mode={mode_key}")

        print(f"\n[FDLC×SIM] Episode complete: {self._step} steps")
        print(f"[FDLC×SIM] Modes: {dict(self._mode_step_counts)}")
        return self.traj_log, self.timing_log

    # ── Helpers ────────────────────────────────────────────────

    def _init_slow_plan(self):
        if self.waypoints:
            wp = self.waypoints[0]
            self._exec_plan = SlowPlan(goal_id=0, goal_ne=(float(wp[0]), float(wp[1])))
        else:
            gcs = self.env.cfg.gcs_ne
            self._exec_plan = SlowPlan(goal_id=None, goal_ne=(float(gcs[0]), float(gcs[1])))

    def _maybe_advance_waypoint(self) -> bool:
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


# ═══════════════════════════════════════════════════════════════════
# 4. Timing statistics computation
# ═══════════════════════════════════════════════════════════════════

def compute_timing_stats(timing_log: List[TimingRecord], label: str) -> Dict:
    """Compute aggregate timing statistics from a timing log."""
    if not timing_log:
        return {"label": label, "error": "empty log"}

    arr_read = np.array([t.t_read_telemetry_ms for t in timing_log])
    arr_fdlc = np.array([t.t_fdlc_decision_ms for t in timing_log])
    arr_send = np.array([t.t_send_command_ms for t in timing_log])
    arr_total = np.array([t.t_loop_total_ms for t in timing_log])
    n_deadline = sum(1 for t in timing_log if t.deadline_miss)

    def _stats(a: np.ndarray) -> Dict:
        return {
            "mean": round(float(np.mean(a)), 4),
            "median": round(float(np.median(a)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "p99": round(float(np.percentile(a, 99)), 4),
            "max": round(float(np.max(a)), 4),
            "min": round(float(np.min(a)), 4),
            "std": round(float(np.std(a)), 4),
        }

    return {
        "label": label,
        "n_steps": len(timing_log),
        "read_telemetry_ms": _stats(arr_read),
        "fdlc_decision_ms": _stats(arr_fdlc),
        "send_command_ms": _stats(arr_send),
        "loop_total_ms": _stats(arr_total),
        "deadline_miss_count": n_deadline,
        "deadline_miss_rate": round(n_deadline / max(1, len(timing_log)), 6),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Trajectory comparison
# ═══════════════════════════════════════════════════════════════════

def compare_trajectories(
    traj_sitl: List[Dict], traj_sim: List[Dict],
) -> Dict:
    """Compare SITL vs simulation trajectories."""
    # Align by step count
    n_common = min(len(traj_sitl), len(traj_sim))

    pos_errors = []
    for i in range(n_common):
        p_sitl = traj_sitl[i]["pos_ne"]
        p_sim = traj_sim[i]["pos_ne"]
        err = math.hypot(p_sitl[0] - p_sim[0], p_sitl[1] - p_sim[1])
        pos_errors.append(round(err, 3))

    if pos_errors:
        arr = np.array(pos_errors)
        return {
            "n_common_steps": n_common,
            "pos_error_mean_m": round(float(np.mean(arr)), 3),
            "pos_error_max_m": round(float(np.max(arr)), 3),
            "pos_error_p95_m": round(float(np.percentile(arr, 95)), 3),
            "pos_error_final_m": round(pos_errors[-1], 3),
        }
    return {"error": "no common steps"}


# ═══════════════════════════════════════════════════════════════════
# 6. Main entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Level 2: FDLC × PX4 SITL validation")
    p.add_argument("--config", type=str, default="configs/experiments/paper1/cases/phase1_px4.yaml")
    p.add_argument("--system", type=str, default="configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml")
    p.add_argument("--duration_s", type=float, default=120.0)
    p.add_argument("--step_hz", type=int, default=5)
    p.add_argument("--altitude_m", type=float, default=10.0)
    p.add_argument("--waypoints", type=str, default="")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--px4_home", type=str, default=str(Path.home() / "PX4-Autopilot"))
    p.add_argument("--sim-only", action="store_true",
                   help="Run only pure simulation (no PX4 SITL)")
    p.add_argument("--no-sim-compare", action="store_true",
                   help="Skip pure simulation comparison")
    p.add_argument("--sitl-already-running", action="store_true",
                   help="PX4 SITL is already running (skip auto-start)")
    args = p.parse_args()

    # ── Output directory ────────────────────────────────────────
    if args.output_dir:
        run_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"results_v2/level2_sitl/{ts}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {run_dir}")

    # ── Load FDLC config ────────────────────────────────────────
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

    # ── Waypoints ───────────────────────────────────────────────
    waypoints: List[Point2D] = []
    if args.waypoints.strip():
        for seg in args.waypoints.split(";"):
            pts = seg.strip().split(",")
            if len(pts) >= 2:
                waypoints.append((float(pts[0]), float(pts[1])))
    else:
        waypoints = list(sim_cfg.poi_list)

    # ── Print config summary ────────────────────────────────────
    print("=" * 64)
    print("Level 2: FDLC × PX4 SITL Validation")
    print("=" * 64)
    print(f"  Config:     {args.config}")
    print(f"  System:     {args.system}")
    print(f"  Step Hz:    {args.step_hz}")
    print(f"  Duration:   {args.duration_s}s")
    print(f"  Altitude:   {args.altitude_m}m")
    print(f"  V cruise:   {sim_cfg.v_xy_cruise} m/s")
    print(f"  V max:      {sim_cfg.v_xy_max} m/s")
    print(f"  GCS pos:    {sim_cfg.gcs_ne}")
    print(f"  POIs:       {len(waypoints)} points")
    print(f"  Structure:  {contract.structure}")
    print(f"  Mode:       {'SIM-ONLY' if args.sim_only else 'PX4 SITL + SIM compare'}")
    print("=" * 64)

    # ── Save config ─────────────────────────────────────────────
    (run_dir / "config.json").write_text(json.dumps({
        "config": args.config, "system": args.system,
        "duration_s": args.duration_s, "step_hz": args.step_hz,
        "altitude_m": args.altitude_m, "waypoints": waypoints,
    }, indent=2))

    sitl_manager: Optional[Px4SitlManager] = None
    master = None
    all_results = {}

    try:
        # ════════════════════════════════════════════════════════
        # A) SITL Run
        # ════════════════════════════════════════════════════════
        if not args.sim_only:
            # Start PX4 SITL if not already running
            if not args.sitl_already_running:
                sitl_manager = Px4SitlManager(px4_home=args.px4_home, headless=True)
                if not sitl_manager.start(timeout_s=90.0):
                    print("[ERROR] Failed to start PX4 SITL")
                    sys.exit(1)

            # Connect via pymavlink
            print("[MAVLink] Connecting to udpin:0.0.0.0:14540...")
            master = mavutil.mavlink_connection("udpin:0.0.0.0:14540")
            master.wait_heartbeat(timeout=15)
            print(f"  ✓ Heartbeat from sys={master.target_system}, comp={master.target_component}")

            # Request data streams
            master.mav.request_data_stream_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
            )
            time.sleep(1.0)
            print("  ✓ Data stream requested")

            # Build bridge
            bridge = Level2Bridge(
                cfg=sim_cfg, contract=contract, fast_params=fast_params,
                env=env, fast_loop=fast_loop, master=master,
                waypoints=waypoints, run_dir=run_dir,
                duration_s=args.duration_s, step_hz=args.step_hz,
            )

            # Arm + Offboard + Takeoff
            if not bridge.arm_and_offboard():
                print("[ERROR] Arm/Offboard failed")
                bridge.land()
                sys.exit(1)

            bridge.takeoff(altitude_m=args.altitude_m)

            # Run FDLC episode
            traj_sitl, timing_sitl = bridge.run_episode_sitl()

            # Land
            bridge.land()
            bridge._stop_setpoint_stream()

            # Save SITL logs
            _save_logs(run_dir, "sitl", traj_sitl, timing_sitl, bridge._mode_step_counts)
            stats_sitl = compute_timing_stats(timing_sitl, "SITL")
            all_results["sitl"] = stats_sitl
            _print_timing_stats(stats_sitl)

        # ════════════════════════════════════════════════════════
        # B) Pure Simulation Run (for comparison)
        # ════════════════════════════════════════════════════════
        if not args.no_sim_compare:
            # Re-create env and fast_loop (fresh state)
            env2 = build_env(sim_cfg, load_scene_config(scene_path))
            env2.reset()
            fast_loop2 = FastLoop(env=env2, params=fast_params, contract=contract)

            bridge_sim = Level2Bridge(
                cfg=sim_cfg, contract=contract, fast_params=fast_params,
                env=env2, fast_loop=fast_loop2, master=None,
                waypoints=waypoints, run_dir=run_dir,
                duration_s=args.duration_s, step_hz=args.step_hz,
            )

            traj_sim, timing_sim = bridge_sim.run_episode_sim()

            # Save sim logs
            _save_logs(run_dir, "sim", traj_sim, timing_sim, bridge_sim._mode_step_counts)
            stats_sim = compute_timing_stats(timing_sim, "SIM")
            all_results["sim"] = stats_sim
            _print_timing_stats(stats_sim)

            # ── Cross-comparison ─────────────────────────────────
            if not args.sim_only:
                comparison = compare_trajectories(traj_sitl, traj_sim)
                all_results["comparison"] = comparison
                print("\n[SITL vs SIM] Trajectory comparison:")
                print(f"  Common steps:     {comparison.get('n_common_steps', 'N/A')}")
                print(f"  Pos error mean:   {comparison.get('pos_error_mean_m', 'N/A')} m")
                print(f"  Pos error p95:    {comparison.get('pos_error_p95_m', 'N/A')} m")
                print(f"  Pos error max:    {comparison.get('pos_error_max_m', 'N/A')} m")

        # ── Save aggregate results ───────────────────────────────
        (run_dir / "timing_stats.json").write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False)
        )
        _write_summary_md(run_dir, all_results, args, sim_cfg)
        print(f"\n[✓] Level 2 validation complete. Results: {run_dir}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Cleaning up...")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if master is not None and not args.sim_only:
            try:
                bridge2 = Level2Bridge(master=master)
                bridge2.land()
            except Exception:
                pass
        if sitl_manager is not None:
            sitl_manager.stop()


def _save_logs(run_dir: Path, tag: str, traj: List[Dict],
               timing: List[TimingRecord], mode_counts: Dict):
    """Save trajectory and timing logs."""
    # Trajectory
    traj_path = run_dir / f"traj_{tag}.jsonl"
    with traj_path.open("w") as f:
        for row in traj:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[SAVE] {traj_path} ({len(traj)} rows)")

    # Timing (as JSONL)
    timing_path = run_dir / f"timing_{tag}.jsonl"
    with timing_path.open("w") as f:
        for t in timing:
            f.write(json.dumps({
                "step": t.step, "t_wall_s": t.t_wall_s,
                "t_read_telemetry_ms": t.t_read_telemetry_ms,
                "t_fdlc_decision_ms": t.t_fdlc_decision_ms,
                "t_send_command_ms": t.t_send_command_ms,
                "t_sleep_ms": t.t_sleep_ms,
                "t_loop_total_ms": t.t_loop_total_ms,
                "deadline_miss": t.deadline_miss,
            }, ensure_ascii=False) + "\n")
    print(f"[SAVE] {timing_path} ({len(timing)} rows)")

    # Mode counts
    mode_path = run_dir / f"mode_counts_{tag}.json"
    mode_path.write_text(json.dumps(mode_counts, indent=2, ensure_ascii=False))


def _print_timing_stats(stats: Dict):
    """Pretty-print timing statistics."""
    print(f"\n[STATS] {stats['label']} Timing (ms):")
    for key in ["read_telemetry_ms", "fdlc_decision_ms", "send_command_ms", "loop_total_ms"]:
        if key in stats and isinstance(stats[key], dict):
            s = stats[key]
            print(f"  {key}:")
            print(f"    mean={s['mean']:.3f}  median={s['median']:.3f}  "
                  f"p95={s['p95']:.3f}  p99={s['p99']:.3f}  max={s['max']:.3f}")
    print(f"  deadline_miss: {stats.get('deadline_miss_count', 'N/A')} / {stats.get('n_steps', 'N/A')} "
          f"({stats.get('deadline_miss_rate', 0)*100:.2f}%)")


def _write_summary_md(run_dir: Path, results: Dict, args, sim_cfg):
    """Write human-readable summary markdown."""
    lines = [
        "# Level 2: FDLC × PX4 SITL Validation Results",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Config**: {args.config}",
        f"- **System**: {args.system}",
        f"- **Duration**: {args.duration_s}s @ {args.step_hz}Hz",
        f"- **Cruise speed**: {sim_cfg.v_xy_cruise} m/s",
        "",
        "## Timing Breakdown",
        "",
    ]

    for tag in ["sitl", "sim"]:
        if tag not in results:
            continue
        s = results[tag]
        lines.append(f"### {tag.upper()}")
        lines.append("")
        lines.append("| Component | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Max (ms) |")
        lines.append("|-----------|-----------|-------------|----------|----------|----------|")
        for key, label in [
            ("read_telemetry_ms", "Telemetry Read"),
            ("fdlc_decision_ms", "FDLC Decision"),
            ("send_command_ms", "Command Send"),
            ("loop_total_ms", "Loop Total"),
        ]:
            if key in s and isinstance(s[key], dict):
                d = s[key]
                lines.append(f"| {label} | {d['mean']:.3f} | {d['median']:.3f} | "
                           f"{d['p95']:.3f} | {d['p99']:.3f} | {d['max']:.3f} |")
        lines.append("")
        dl = s.get("deadline_miss_count", 0)
        dl_rate = s.get("deadline_miss_rate", 0) * 100
        lines.append(f"- **Deadline misses**: {dl} / {s.get('n_steps', 'N/A')} ({dl_rate:.2f}%)")
        lines.append("")

    if "comparison" in results:
        c = results["comparison"]
        lines.append("## SITL vs Simulation Comparison")
        lines.append("")
        lines.append(f"- Common steps: {c.get('n_common_steps', 'N/A')}")
        lines.append(f"- Position error mean: {c.get('pos_error_mean_m', 'N/A')} m")
        lines.append(f"- Position error p95: {c.get('pos_error_p95_m', 'N/A')} m")
        lines.append(f"- Position error max: {c.get('pos_error_max_m', 'N/A')} m")
        lines.append("")

    (run_dir / "summary.md").write_text("\n".join(lines))
    print(f"[SAVE] {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
