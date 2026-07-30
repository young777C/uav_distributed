"""Non-destructive label adapters — fix data-check P2/P3 at the label layer.

The raw HDF5 already contains everything needed; rather than rewriting 60 large
files, we derive the corrected/enriched labels on the fly from raw fields:

  P3  candidate boxes (target + distractors) for L_target_id
      <- distractors/positions + target/* + state/cam_*  (via acot_probe.projection)

  P2  occlusion supervision that USES the rich raycast signal
      - `combined_occlusion` = max(out-of-frame projection occ, structural occ_raycast)
      - `search_mode_target` recomputed from combined occlusion (was projection-only)
      - `iar_occlusion_target` = state/occ_raycast (structural, for IAR's occ head)

Used by the Stage-2 dataloader. (P1 waypoint-tail degeneracy is already handled in
dataset._build_index; P4 occlusion sparsity is a data-coverage caveat, not code.)
"""

from __future__ import annotations

import numpy as np

from acot_probe.projection import extract_candidates

# Matches the recorder (336×336, 90° FOV). Vehicle pseudo-box size in meters.
IMAGE_DEFAULT = {
    "width": 336, "height": 336, "fov_deg": 90.0,
    "vehicle_size_m": 4.0, "box_min_px": 10, "box_max_px": 140,
}


# --- P3: candidate boxes for L_target_id -----------------------------------
def candidate_boxes(ep, i, image=None):
    """Target + in-frame distractor boxes for frame i. Returns list[Candidate];
    the target has is_target=True. Empty if the target is off-screen that frame."""
    return extract_candidates(ep, i, {"image": image or IMAGE_DEFAULT})


# --- P2: occlusion / visibility supervision using the raycast signal --------
def out_of_frame_occ(ep) -> np.ndarray:
    return np.asarray(ep["annotation/occlusion"][:], np.float32)


def structural_occ(ep) -> np.ndarray:
    """state/occ_raycast (in-frame-but-blocked). Zeros if the field is absent."""
    if "state/occ_raycast" in ep:
        return np.asarray(ep["state/occ_raycast"][:], np.float32)
    n = ep["rgb"].shape[0]
    return np.zeros(n, np.float32)


def combined_occlusion(ep) -> np.ndarray:
    """Fixes P2: fold structural raycast occlusion into the visibility signal."""
    return np.maximum(out_of_frame_occ(ep), structural_occ(ep))


def search_mode_target(ep, dt: float = 0.1) -> np.ndarray:
    """Recompute search_mode from COMBINED occlusion (design §5.3 formula), so the
    drone also widens search when the target is hidden behind a building/tunnel."""
    occ = combined_occlusion(ep)
    sm = np.zeros_like(occ)
    time_lost = 0.0
    for i, o in enumerate(occ):
        time_lost = time_lost + dt if o > 0.8 else max(0.0, time_lost - 2 * dt)
        sm[i] = float(np.clip(o * (1.0 + time_lost / 5.0), 0.0, 1.0))
    return sm


def iar_occlusion_target(ep) -> np.ndarray:
    """Structural occlusion target for IAR's "about-to-be-occluded" head (§6.2 #1)."""
    return structural_occ(ep)


def visible_mask(ep, thresh: float = 0.8) -> np.ndarray:
    """Per-frame 'target genuinely observable' flag (combined occ below thresh).
    Stage-2 masks L_action/L_ear on ¬visible frames (avoid supervising blind follow)."""
    return (combined_occlusion(ep) < thresh).astype(np.float32)
