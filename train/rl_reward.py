"""Stage-3 RL per-frame reward (design: acot_note/stage3-rl-design.md §4).

The reward that fixes the SR bottleneck = CONTROL FRAMING: reward keeping the TRUE target
CENTERED (R_center), tracking the RIGHT target (R_correct_id, anti-hack), re-acquiring after
loss (R_reacquire), minus off-screen / wrong-commit penalties (aligned to the SR criterion).

ANTI-HACK / decouples the decorrelate-vs-center tension: R_center is on the TRUE target's
projected position (GT, TRAINING-ONLY — GT enters the reward, NEVER the observation). So the
policy is rewarded for centering the *identity-conditioned true target*, not "whatever car is
central" → it does NOT re-introduce the removed 'most-central' shortcut, and centering a
distractor earns R_center≈0 (the true target is off-center) + R_correct_id penalty.

Pure, torch-free, unit-testable; the env calls frame_reward() each tick during RL rollout.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class RewardCfg:
    w_center: float = 1.0
    w_track: float = 0.3
    w_id: float = 1.0
    w_reacq: float = 5.0
    sigma_frac: float = 0.25       # R_center Gaussian width (fraction of frame)
    id_wrong_pen: float = 2.0      # committed a distractor → −this (anti-hack)
    off_pen: float = 1.0           # per-tick penalty once off ≥ max_lost_s
    wrong_pen: float = 1.0         # per-tick penalty once committed-wrong ≥ t_fail_s
    max_lost_s: float = 5.0
    t_fail_s: float = 2.0


def frame_reward(*, tgt_uv_frac, committed_is_true, off_run_s, wrong_run_s,
                 reacquired: bool = False, cfg: RewardCfg = RewardCfg()) -> dict:
    """One tick's reward + components.

    tgt_uv_frac : (u/W, v/H) of the TRUE target, or None if off-screen (GT, reward-only).
    committed_is_true : True if the reid-committed id == true target; False if a distractor;
                        None if nothing committed / target off-screen.
    off_run_s   : consecutive seconds the target has been off-screen.
    wrong_run_s : consecutive seconds committed to the WRONG id.
    reacquired  : True on the tick a loss ends with the TRUE target re-committed.
    """
    # R_center — on the TRUE target's offset from image center (anti-hack: not "any car").
    if tgt_uv_frac is not None:
        du, dv = tgt_uv_frac[0] - 0.5, tgt_uv_frac[1] - 0.5
        r_center = math.exp(-(du * du + dv * dv) / (cfg.sigma_frac * cfg.sigma_frac))
        r_track = 1.0
    else:
        r_center = 0.0
        r_track = 0.0

    if committed_is_true is True:
        r_id = 1.0
    elif committed_is_true is False:
        r_id = -cfg.id_wrong_pen
    else:
        r_id = 0.0

    r_reacq = 1.0 if reacquired else 0.0

    pen = 0.0
    if off_run_s >= cfg.max_lost_s:
        pen += cfg.off_pen
    if wrong_run_s >= cfg.t_fail_s:
        pen += cfg.wrong_pen

    total = (cfg.w_center * r_center + cfg.w_track * r_track +
             cfg.w_id * r_id + cfg.w_reacq * r_reacq - pen)
    return {"total": total, "center": r_center, "track": r_track,
            "id": r_id, "reacq": r_reacq, "penalty": pen}


if __name__ == "__main__":  # unit tests (torch-free)
    C = RewardCfg()
    # 1) centered + correct id → high reward
    a = frame_reward(tgt_uv_frac=(0.5, 0.5), committed_is_true=True, off_run_s=0, wrong_run_s=0)
    # 2) off-center + correct id → lower R_center
    b = frame_reward(tgt_uv_frac=(0.9, 0.1), committed_is_true=True, off_run_s=0, wrong_run_s=0)
    # 3) ANTI-HACK: policy centers a DISTRACTOR → true target is off-center, committed wrong
    c = frame_reward(tgt_uv_frac=(0.95, 0.5), committed_is_true=False, off_run_s=0, wrong_run_s=2.5)
    # 4) target lost long → penalty
    d = frame_reward(tgt_uv_frac=None, committed_is_true=None, off_run_s=6.0, wrong_run_s=0)
    # 5) reacquire bonus
    e = frame_reward(tgt_uv_frac=(0.5, 0.5), committed_is_true=True, off_run_s=0, wrong_run_s=0, reacquired=True)
    print(f"1 centered+correct : total={a['total']:.3f}  (center={a['center']:.3f})")
    print(f"2 off-center+correct: total={b['total']:.3f}  (center={b['center']:.3f})")
    print(f"3 center-a-distractor(hack): total={c['total']:.3f}  (center={c['center']:.3f}, id={c['id']:.0f})")
    print(f"4 lost long        : total={d['total']:.3f}")
    print(f"5 reacquire        : total={e['total']:.3f}")
    assert a["total"] > b["total"] > 0, "centered should beat off-center"
    assert c["total"] < 0, "ANTI-HACK: centering a distractor must be net-negative"
    assert c["center"] < 0.1, "ANTI-HACK: R_center small when the TRUE target is off-center"
    assert d["total"] < 0, "losing the target must be penalized"
    assert e["total"] > a["total"], "reacquire bonus should raise reward"
    print("ALL REWARD UNIT TESTS PASS")
