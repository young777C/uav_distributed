"""Runtime data types of the acot policy process (framework §7.2, P2).

PolicyInput / Detection are EVERYTHING a method may see: camera frames, language, proprio, the own camera pose and the
boxes the protocol's perception produced (with the privileged quantities the protocol discloses: GT-projected box,
depth, matched-box world position). ScoringInfo carries the labels (which box is the target, simulator actor ids,
target attributes, visibility) — it goes to the scorer reply, the logs and the collectors, never to a method.
A diagnostic arm additionally receives an OracleHandle restricted to the GT channels it declared; every call is
counted per purpose (→ gt_access.json). Deployable arms get None."""
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np

# fields that must never appear in what a method receives (test_policy_input_fields; AST scan test_method_no_gt_attr)
FORBIDDEN = frozenset({"is_target", "true_idx", "gt_idx", "target_color", "target_bp", "tpos", "dstates",
                       "road_pred_world", "vis", "retarget_event", "gt_candidates", "seed"})


@dataclass(frozen=True)
class Detection:
    """one candidate box. `key` is the perception's identity key (tracker id), never the simulator actor index."""
    key: int
    u: float
    v: float
    w: float
    h: float
    depth_m: float                                   # privileged (protocol: cand_depth_gt)
    world_xyz: Optional[Tuple[float, float, float]]  # privileged (protocol: where_world_gt_matched_box)
    vlm_feat: np.ndarray                             # (C,) VLM grid pooled inside the box (float32)
    crop_feat: Optional[np.ndarray]                  # (384,) L2-normalised crop-DINOv2 (float64) or None

    @property
    def idx(self):                                   # the attribute name legacy helpers (e.g. AttrScorer) read
        return self.key


@dataclass(frozen=True)
class PolicyInput:
    step: int
    s2: bool                                         # slow-stream tick (VLM / perception / identity run)
    rgb: np.ndarray
    rgb_hi: Optional[np.ndarray]                     # k× frame of the same camera (opening ticks only)
    language: str
    proprio: np.ndarray                              # (5,) float32
    campose: Tuple[float, ...]                       # own camera pose (deployable)
    vlm: Optional[tuple]                             # S2: (ctx_layers [(M,C)], ctx (M,C)) of this frame
    detections: Optional[Tuple[Detection, ...]]      # S2: boxes of this frame (None: nothing in frame / S1 tick)
    shared_open_slot: Optional[int] = None           # 1006 §3.4: slot of the shared-opening reference actor at an
                                                     # opening tick (−1 = not among the detections); None otherwise


@dataclass(frozen=True)
class ScoringInfo:
    step: int
    true_idx: int                                    # slot of the target among the detections, -1 if absent
    is_target: Tuple[bool, ...]
    gt_idx: Tuple[int, ...]                          # simulator actor index per slot (-1 = the target)
    target_color: str
    target_bp: str
    vis: Optional[tuple]
    # 1024 HOW ladder: the target's TRUE world position at this tick, visible or not (filled by the perception only
    # for arms with control.ex_source = oracle; reaches a method only through OracleHandle.true_world)
    target_world: Optional[Tuple[float, float, float]] = None


class OracleError(PermissionError):
    pass


class OracleHandle:
    """GT access for DIAGNOSTIC arms only: `true_slot(purpose)` with purpose ∈ the arm's declared diagnostics."""

    def __init__(self, allowed):
        self.allowed = frozenset(allowed)
        self.__info = None
        self.calls = {}

    def bind(self, info):                            # called by the server with this tick's ScoringInfo
        self.__info = info

    def true_slot(self, purpose):
        if purpose not in self.allowed:
            raise OracleError(f"GT channel {purpose!r} was not declared in the method's diagnostics "
                              f"({sorted(self.allowed)})")
        self.calls[purpose] = self.calls.get(purpose, 0) + 1
        return -1 if self.__info is None else int(self.__info.true_idx)

    def true_world(self, purpose):
        """1024 HOW ladder: the target's true world position (np.ndarray (3,)) of the bound tick, None if unknown."""
        if purpose not in self.allowed:
            raise OracleError(f"GT channel {purpose!r} was not declared in the method's diagnostics "
                              f"({sorted(self.allowed)})")
        self.calls[purpose] = self.calls.get(purpose, 0) + 1
        tw = None if self.__info is None else self.__info.target_world
        return None if tw is None else np.asarray(tw, np.float64)

    def reset_calls(self):
        out, self.calls = self.calls, {}
        return out


@dataclass
class Decision:
    """what the policy returns each tick (the server turns it into the wire reply)."""
    action: Tuple[float, float, float, float]
    search: float
    pred_slot: int
    trace: dict                                      # S2 decision trace for ticks.jsonl (no labels in it)


def field_names(cls):
    return {f.name for f in fields(cls)}
