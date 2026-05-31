# src/uavlab/scene/loader.py
'''本模块负责：
1）读取yaml
2）规范化任务场景配置
3）自动化将boundary变成nofly rect
4）返回统一结构，供geometry和env使用'''
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy
import yaml

from uavlab.common.config import load_resolved_config


Point2D = Tuple[float, float]
Circle = Tuple[float, float, float]
Rect = Tuple[float, float, float, float]


@dataclass
class SceneConfig:
    # map bounds
    n_min: float
    n_max: float
    e_min: float
    e_max: float

    # common scene objects
    collision_margin_m: float = 0.0
    obstacles_circles: List[Circle] = field(default_factory=list)
    nofly_circles: List[Circle] = field(default_factory=list)
    nofly_rects: List[Rect] = field(default_factory=list)

    # boundary -> nofly fence
    boundary_as_nofly: bool = True
    boundary_buffer_m: float = 3.0

    # TaskA specific
    start_ne: Optional[Point2D] = None
    goal_ne: Optional[Point2D] = None
    goal_radius_m: Optional[float] = None
    # 地面控制站 GCS（默认与起降点 start_ne 一致，见 docs/uav_gcs_collaboration.md）
    gcs_ne: Optional[Point2D] = None

    # TaskB / 巡检任务点
    poi_list: List[Point2D] = field(default_factory=list)
    visit_radius_m: Optional[float] = None
    # 高价值数据采集半径（默认回退到 visit_radius_m / goal_radius_m）
    key_data_radius_m: Optional[float] = None
    # 通信退化区域（圆），与 CommChannel.blackhole 一致，预设于场景
    communication_blackholes: List[Circle] = field(default_factory=list)

    # raw payload for extensibility
    extras: Dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _as_point2(x: Any, name: str) -> Point2D:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise ValueError(f"{name} must be [n, e], got {x}")
    return float(x[0]), float(x[1])


def _as_circle_list(xs: Any) -> List[Circle]:
    if xs is None:
        return []
    out: List[Circle] = []
    for x in xs:
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(f"Circle must be [n, e, r], got {x}")
        out.append((float(x[0]), float(x[1]), float(x[2])))
    return out


def _as_rect_list(xs: Any) -> List[Rect]:
    if xs is None:
        return []
    out: List[Rect] = []
    for x in xs:
        if not isinstance(x, (list, tuple)) or len(x) != 4:
            raise ValueError(f"Rect must be [n_min, n_max, e_min, e_max], got {x}")
        out.append((float(x[0]), float(x[1]), float(x[2]), float(x[3])))
    return out


def _as_point_list(xs: Any) -> List[Point2D]:
    if xs is None:
        return []
    return [_as_point2(x, "poi") for x in xs]


def _build_boundary_nofly_rects(
    n_min: float,
    n_max: float,
    e_min: float,
    e_max: float,
    buffer_m: float,
) -> List[Rect]:
    """
    把边界转成 4 条 nofly 矩形围栏带。
    注意：这些围栏带贴着边界向内“吃” buffer_m。
    """
    b = float(buffer_m)
    if b <= 0.0:
        return []

    return [
        (n_min, n_min + b, e_min, e_max),   # west strip
        (n_max - b, n_max, e_min, e_max),   # east strip
        (n_min, n_max, e_min, e_min + b),   # south strip
        (n_min, n_max, e_max - b, e_max),   # north strip
    ]


def load_scene_config(path: str | Path) -> SceneConfig:
    # Support `extends` in scene YAMLs (same semantics as load_resolved_config).
    raw = load_resolved_config(path)

    n_min = float(raw["n_min"])
    n_max = float(raw["n_max"])
    e_min = float(raw["e_min"])
    e_max = float(raw["e_max"])

    cfg = SceneConfig(
        n_min=n_min,
        n_max=n_max,
        e_min=e_min,
        e_max=e_max,
        collision_margin_m=float(raw.get("collision_margin_m", 0.0)),
        obstacles_circles=_as_circle_list(raw.get("obstacles_circles", [])),
        nofly_circles=_as_circle_list(raw.get("nofly_circles", [])),
        nofly_rects=_as_rect_list(raw.get("nofly_rects", [])),
        boundary_as_nofly=bool(raw.get("boundary_as_nofly", True)),
        boundary_buffer_m=float(raw.get("boundary_buffer_m", 3.0)),
        extras={},
    )

    if "start_ne" in raw:
        cfg.start_ne = _as_point2(raw["start_ne"], "start_ne")

    if "gcs_ne" in raw:
        cfg.gcs_ne = _as_point2(raw["gcs_ne"], "gcs_ne")

    if "goal_ne" in raw:
        cfg.goal_ne = _as_point2(raw["goal_ne"], "goal_ne")

    if "goal_radius_m" in raw:
        cfg.goal_radius_m = float(raw["goal_radius_m"])

    if "poi_list" in raw:
        cfg.poi_list = _as_point_list(raw["poi_list"])

    if "visit_radius_m" in raw:
        cfg.visit_radius_m = float(raw["visit_radius_m"])

    if "key_data_radius_m" in raw:
        cfg.key_data_radius_m = float(raw["key_data_radius_m"])

    if "communication_blackholes" in raw:
        cfg.communication_blackholes = _as_circle_list(raw["communication_blackholes"])

    # 存下额外字段，方便未来扩展
    known_keys = {
        "n_min", "n_max", "e_min", "e_max",
        "collision_margin_m",
        "obstacles_circles",
        "nofly_circles",
        "nofly_rects",
        "boundary_as_nofly",
        "boundary_buffer_m",
        "start_ne",
        "gcs_ne",
        "goal_ne",
        "goal_radius_m",
        "poi_list",
        "visit_radius_m",
        "key_data_radius_m",
        "communication_blackholes",
    }
    cfg.extras = {k: copy.deepcopy(v) for k, v in raw.items() if k not in known_keys}

    if cfg.boundary_as_nofly:
        cfg.nofly_rects.extend(
            _build_boundary_nofly_rects(
                n_min=cfg.n_min,
                n_max=cfg.n_max,
                e_min=cfg.e_min,
                e_max=cfg.e_max,
                buffer_m=cfg.boundary_buffer_m,
            )
        )

    return cfg


def load_taskA_scene(path: str | Path) -> SceneConfig:
    cfg = load_scene_config(path)

    if cfg.start_ne is None:
        raise ValueError("TaskA scene must define start_ne")
    if cfg.goal_ne is None:
        raise ValueError("TaskA scene must define goal_ne")
    if cfg.goal_radius_m is None:
        raise ValueError("TaskA scene must define goal_radius_m")

    # GCS 未显式配置时与起降点同址（指挥中心在基地）
    if cfg.gcs_ne is None:
        cfg.gcs_ne = cfg.start_ne

    return cfg


def resolve_gcs_ne(cfg: SceneConfig) -> Point2D:
    """地面控制站平面坐标；未配置时等于 start_ne。"""
    if cfg.gcs_ne is not None:
        return cfg.gcs_ne
    if cfg.start_ne is not None:
        return cfg.start_ne
    raise ValueError("Scene has no gcs_ne or start_ne")


def load_taskB_scene(path: str | Path) -> SceneConfig:
    cfg = load_scene_config(path)

    if cfg.start_ne is None:
        raise ValueError("TaskB scene must define start_ne")
    if not cfg.poi_list:
        raise ValueError("TaskB scene must define poi_list")
    if cfg.visit_radius_m is None:
        raise ValueError("TaskB scene must define visit_radius_m")

    return cfg