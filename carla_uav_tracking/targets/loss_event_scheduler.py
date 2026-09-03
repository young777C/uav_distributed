"""Deliberate SUSTAINED target-loss events for predict-intercept recovery data.

Round-2 lesson: stochastic TM maneuvers only fragment the loss into sub-second
blips (the expert re-frames the privileged true target too fast) AND re-correlate
the target's geometry (hurting the benchmark-leak floor). Instead, schedule a few
BOUNDED windows per episode where:

  1. the target is forced to deviate (accel + lane change) at window onset, and
  2. the expert is flown BLIND (target_visible=False → PID dead-reckons on the
     last-seen velocity) for the whole window, so it does NOT instantly re-center.

The target genuinely leaves frame and STAYS off for ~the window length (>=3s),
then at window end the expert regains the true target and flies to intercept →
a clean, controllable "loss → predict → re-acquire" arc. Because loss is confined
to these bounded windows (and co-visible time keeps centroid framing), the leak
floor is not re-correlated the way continuous chasing did in round 2.
"""

from __future__ import annotations

import random


class LossEventScheduler:
    """Places K non-overlapping blind-predict windows per episode (by wall time)."""

    def __init__(self, cfg_env: dict, fps: int = 10):
        le = (cfg_env or {}).get("loss_events", {}) or {}
        self.enabled = bool(le.get("enabled", False))
        self.per_episode = int(le.get("per_episode", 2))
        self.dur_range = tuple(le.get("dur_range", [3.5, 6.0]))
        self.min_start_s = float(le.get("min_start_s", 12.0))   # let lock-on settle first
        self.end_margin_s = float(le.get("end_margin_s", 12.0)) # leave room to re-acquire
        self.duration_s = float(cfg_env.get("max_duration_seconds", 150))
        self._windows: list[tuple[float, float]] = []
        self._fired: set[int] = set()
        self._fired_off: set[int] = set()

    def reset(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self._windows = []
        self._fired = set()
        self._fired_off = set()
        if not self.enabled or self.per_episode <= 0:
            return
        lo = self.min_start_s
        hi = self.duration_s - self.end_margin_s
        if hi - lo < self.dur_range[1]:
            return                                   # episode too short for any event
        # One window per equal-width slot → guarantees non-overlap + spacing.
        slot = (hi - lo) / self.per_episode
        for k in range(self.per_episode):
            d = rng.uniform(*self.dur_range)
            s_lo = lo + k * slot
            s_hi = lo + (k + 1) * slot - d
            start = rng.uniform(s_lo, s_hi) if s_hi > s_lo else s_lo
            self._windows.append((start, start + d))

    def in_predict(self, elapsed: float) -> bool:
        """True while inside a loss window → the expert should fly blind (predict)."""
        return any(s <= elapsed < e for s, e in self._windows)

    def onset(self, elapsed: float) -> float | None:
        """Return the window duration exactly ONCE, at the first step inside it
        (so the caller triggers the target's forced deviation); else None."""
        for i, (s, e) in enumerate(self._windows):
            if s <= elapsed < e and i not in self._fired:
                self._fired.add(i)
                return e - s
        return None

    def offset(self, elapsed: float) -> bool:
        """True exactly ONCE, on the first step AFTER a (started) window ends — the target
        is re-appearing → caller forces >=2 look-alikes co-visible for language re-acquisition."""
        for i, (s, e) in enumerate(self._windows):
            if i in self._fired and i not in self._fired_off and elapsed >= e:
                self._fired_off.add(i)
                return True
        return False

    @property
    def windows(self) -> list[tuple[float, float]]:
        return self._windows
