"""Closed-loop rollout harness for the ACoT-UAV-Track VLA policy.

See acot_note/harness-implementation-guide.md. Modules:
  online_vlm  — Phase E.1: run the frozen VLM online per frame (no disk cache).
  policy      — load the 4-module Stage-2 ckpt, S1/S2 async scheduler, .act().
  candidates  — build cand_feats + candidate boxes per frame.
  metrics     — §7.2 metric family + hysteresis (sustained) mis-follow.
  env         — forked recorder loop: student-driven episode, GT score-only.
"""
