verdict: **NO MAJOR ISSUES**

**R3 MAJOR 复核**

1. **持久 tracklet 接口 / oracle association: resolved for PoC, not for final paper deployment claim.**  
   v4 已在一句话、方法接口、贡献范围、主表脚注中明确限定“在持久 tracklet 接口下”，closure §3.1 也补了 ID 噪声敏感度。对“CCF-A 方法向 PoC 阶段”的闭环标准，这已不再是 MAJOR；它是清楚的适用前提与局限。

2. **泛化不足: resolved under narrowed claim.**  
   Town10HD 与同会话 reid 已补齐。单设定 correct 不显著，但 latch@5 在 reid/Town10HD 均显著，合并 n=58 correct/latch 显著；文档也把泛化 claim 收窄到“持续错锁减少更稳健”。这符合 PoC 闭环，不再构成 MAJOR。

我复跑了指定命令：Town10HD correct +0.100 CI 含 0、latch@5 0.15→0.60；reid correct +0.110 CI 含 0、latch@5 0.21→0.53。与文档一致。

**剩余 MAJOR**

无。

**MINOR**

- `eval_out/...` 从 repo root 不存在，实际在 `carla_uav_tracking/rollout/eval_out/...`；建议文档统一路径或声明相对 `rollout/`。
- `innovation §5` 仍写“reid 方向为正(未显著,进行中)”，应同步为“correct 未显著、latch@5 显著”。
- `ID 噪声 ≥5%/tick 增益消失` 稍强；表中 5% 仍有小增益，建议写“显著衰减，10% 基本消失”。
- max_wrong 在多个设定 CI 含 0，当前已如实写明；最终稿中不要把“错窗长度改善”作为主 claim。
- “任意 MOT 关联层”措辞可再收紧为“中等质量持久关联层”。