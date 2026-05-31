from __future__ import annotations

from dataclasses import dataclass

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig

# 这个脚本定义了快环（Fast Loop）的策略参数（FastLoopParams）类，用于配置和管理快环中的有限状态机（FSM）阈值和功能开关，
# 参数值从契约配置（Paper1ContractConfig）对象中提取。主要包括链路丢包率的安全阈值、恢复阈值、安全悬停步数以及各类状态
# 模式的开关（如 FSM、本地返航、安全保护、恢复模式等），为快环决策逻辑的行为提供参数支持。

@dataclass
class FastLoopParams:
    """FSM thresholds and feature toggles (filled from ``Paper1ContractConfig``)."""

    link_loss_safe: float = 0.05  # 控制/遥测链路 (§9.1)
    link_loss_recover: float = 0.20  # 任务数据链路 (§9.2)
    safe_hover_steps: int = 10   # 安全悬停步数
    enable_fsm: bool = True   
    enable_back_mode: bool = True   
    enable_safety_mode: bool = True
    enable_recovery_mode: bool = True

    @staticmethod
    def from_contract(contract: Paper1ContractConfig) -> "FastLoopParams":
        fl = dict(contract.fast_loop or {})
        thr = dict(fl.get("fsm_thresholds") or {})
        return FastLoopParams(
            link_loss_safe=float(thr.get("link_loss_safe", 0.05)),
            link_loss_recover=float(thr.get("link_loss_recover", 0.20)),
            safe_hover_steps=int(thr.get("safe_hover_steps", 10)),
            enable_fsm=bool(fl.get("enable_fsm", True)),
            enable_back_mode=bool(fl.get("enable_back_mode", True)),
            enable_safety_mode=bool(fl.get("enable_safety_mode", True)),
            enable_recovery_mode=bool(fl.get("enable_recovery_mode", True)),
        )
