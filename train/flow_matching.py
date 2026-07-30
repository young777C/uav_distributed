"""Conditional (rectified) flow matching for EAR waypoint prediction.

x1 = ground-truth waypoints (B,K,3); z ~ N(0,I); t ~ U(0,1).
  x_t = (1-t)*z + t*x1        (straight-line interpolant)
  target velocity  v = x1 - z
The model predicts v_hat = model(x_t, t, cond); loss = MSE(v_hat, v).
Sampling integrates dx/dt = v_hat from z (t=0) to x1_hat (t=1) with Euler steps.

Same flow-matching recipe as pi0 / ACoT-VLA, applied to the coarse 3D waypoints.
"""

from __future__ import annotations

import torch


def cfm_loss(model, x1: torch.Tensor, cond, cond_mask=None, **mkw) -> torch.Tensor:
    """Conditional flow-matching MSE loss. x1: (B,K,3). **mkw forwarded to model
    (e.g. proprio=...)."""
    b = x1.shape[0]
    z = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device)
    tt = t.view(b, 1, 1)
    x_t = (1.0 - tt) * z + tt * x1
    v = x1 - z
    v_hat = model(x_t, t, cond, cond_mask, **mkw)
    return torch.nn.functional.mse_loss(v_hat, v)


@torch.no_grad()
def sample(model, cond, cond_mask, k: int, steps: int = 8, **mkw) -> torch.Tensor:
    """Integrate the learned velocity field from noise -> waypoints. Returns (B,K,3)."""
    b = cond.shape[0]
    x = torch.randn(b, k, 3, device=cond.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=cond.device)
        x = x + dt * model(x, t, cond, cond_mask, **mkw)
    return x
