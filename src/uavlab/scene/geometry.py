# src/uavlab/scene/geometry.py

'''本模块定位：负责计算点与场景的几何关系'''
from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple


Point2D = Tuple[float, float]
Circle = Tuple[float, float, float]          # (cn, ce, r)
Rect = Tuple[float, float, float, float]     # (n_min, n_max, e_min, e_max)


def _as_float_tuple(x: Sequence[Any], expected_len: int) -> Tuple[float, ...]:
    if len(x) != expected_len:
        raise ValueError(f"Expected sequence of len={expected_len}, got {len(x)}")
    return tuple(float(v) for v in x)


def _iter_circles(container: Any) -> List[Circle]:
    """
    兼容多种 circle 写法：
    - [n, e, r]
    - {"n":..., "e":..., "r":...}
    - object.n / object.e / object.r
    """
    if container is None:
        return []

    vals = list(container.values()) if isinstance(container, dict) else container
    out: List[Circle] = []

    for it in vals:
        if it is None:
            continue

        if isinstance(it, (list, tuple)) and len(it) >= 3:
            cn, ce, r = float(it[0]), float(it[1]), float(it[2])
            out.append((cn, ce, r))
            continue

        if isinstance(it, dict):
            cn = it.get("n", it.get("N", it.get("north", it.get("x"))))
            ce = it.get("e", it.get("E", it.get("east", it.get("y"))))
            r = it.get("r", it.get("R", it.get("radius", it.get("rad"))))
            if cn is None or ce is None or r is None:
                continue
            out.append((float(cn), float(ce), float(r)))
            continue

        cn = getattr(it, "n", getattr(it, "N", getattr(it, "north", None)))
        ce = getattr(it, "e", getattr(it, "E", getattr(it, "east", None)))
        r = getattr(it, "r", getattr(it, "R", getattr(it, "radius", None)))
        if cn is None or ce is None or r is None:
            continue
        out.append((float(cn), float(ce), float(r)))

    return out


def _iter_rects(container: Any) -> List[Rect]:
    """
    兼容多种 rect 写法：
    - [n_min, n_max, e_min, e_max]
    - {"n_min":..., "n_max":..., "e_min":..., "e_max":...}
    """
    if container is None:
        return []

    vals = list(container.values()) if isinstance(container, dict) else container
    out: List[Rect] = []

    for it in vals:
        if it is None:
            continue

        if isinstance(it, (list, tuple)) and len(it) >= 4:
            n_min, n_max, e_min, e_max = (
                float(it[0]),
                float(it[1]),
                float(it[2]),
                float(it[3]),
            )
            out.append((n_min, n_max, e_min, e_max))
            continue

        if isinstance(it, dict):
            n_min = it.get("n_min", it.get("north_min"))
            n_max = it.get("n_max", it.get("north_max"))
            e_min = it.get("e_min", it.get("east_min"))
            e_max = it.get("e_max", it.get("east_max"))
            if None in (n_min, n_max, e_min, e_max):
                continue
            out.append((float(n_min), float(n_max), float(e_min), float(e_max)))
            continue

    return out


def get_obstacle_circles(cfg: Any) -> List[Circle]:
    for name in ("obstacles_circles", "obstacles", "obstacle_list", "obstacle_disks", "obstacle_circles"):
        if hasattr(cfg, name):
            return _iter_circles(getattr(cfg, name))
        if isinstance(cfg, dict) and name in cfg:
            return _iter_circles(cfg[name])
    return []


def get_nofly_circles(cfg: Any) -> List[Circle]:
    for name in ("nofly_circles", "nofly_zones", "nofly", "nofly_disks", "nofly_circles"):
        if hasattr(cfg, name):
            return _iter_circles(getattr(cfg, name))
        if isinstance(cfg, dict) and name in cfg:
            return _iter_circles(cfg[name])
    return []


def get_nofly_rects(cfg: Any) -> List[Rect]:
    for name in ("nofly_rects", "nofly_rectangles", "nofly_boxes"):
        if hasattr(cfg, name):
            return _iter_rects(getattr(cfg, name))
        if isinstance(cfg, dict) and name in cfg:
            return _iter_rects(cfg[name])
    return []


def dist_to_point(n: float, e: float, point_ne: Point2D) -> float:
    pn, pe = point_ne
    return float(math.hypot(float(n) - float(pn), float(e) - float(pe)))


def dist_to_goal(n: float, e: float, goal_ne: Point2D) -> float:
    return dist_to_point(n, e, goal_ne)


def point_reached(n: float, e: float, point_ne: Point2D, radius_m: float) -> bool:
    return dist_to_point(n, e, point_ne) <= float(radius_m)


def goal_reached(n: float, e: float, goal_ne: Point2D, goal_radius_m: float) -> bool:
    return point_reached(n, e, goal_ne, goal_radius_m)


def in_key_data_collection_zone(n: float, e: float, cfg: Any) -> bool:
    """
    判断 (n, e) 是否处于预设高价值数据采集区（与 data_key_mode='poi' 配合）。

    - 若配置了非空 poi_list：仅在**任务点/POI（星标）**半径内为 True；
      起降点即使与 goal 重合也不算高价值采集区（与 scene.jpg 语义一致）。
    - 若 poi_list 为空：回退为终点 goal 周围（兼容旧场景）。

    半径优先级：key_data_radius_m → visit_radius_m → goal_radius_m。
    """
    r = getattr(cfg, "key_data_radius_m", None)
    if r is None:
        r = getattr(cfg, "visit_radius_m", None)
    if r is None:
        r = getattr(cfg, "goal_radius_m", None)
    if r is None or float(r) <= 0.0:
        return False
    rr = float(r)

    poi_list = getattr(cfg, "poi_list", None) or []
    if poi_list:
        # Optional: restrict key zones to a subset of POIs via scene extras.
        # This enables "high-utility POIs" scenario families without changing the POI geometry itself.
        key_ids = None
        extras = getattr(cfg, "extras", None)
        if isinstance(extras, dict):
            key_ids = extras.get("key_poi_ids")
        if isinstance(key_ids, list) and key_ids:
            idxs = []
            for x in key_ids:
                try:
                    i = int(x)
                    if 0 <= i < len(poi_list):
                        idxs.append(i)
                except Exception:
                    continue
            selected = [poi_list[i] for i in idxs] if idxs else poi_list
        else:
            selected = poi_list

        for pn, pe in selected:
            if math.hypot(n - float(pn), e - float(pe)) <= rr:
                return True
        return False

    goal_ne = getattr(cfg, "goal_ne", None)
    if goal_ne is not None:
        gn, ge = float(goal_ne[0]), float(goal_ne[1])
        if math.hypot(n - gn, e - ge) <= rr:
            return True
    return False


def out_of_bounds(n: float, e: float, cfg: Any) -> bool:
    n_min = float(getattr(cfg, "n_min", 0.0))
    n_max = float(getattr(cfg, "n_max", 0.0))
    e_min = float(getattr(cfg, "e_min", 0.0))
    e_max = float(getattr(cfg, "e_max", 0.0))
    return bool((n < n_min) or (n > n_max) or (e < e_min) or (e > e_max))


def in_circle(n: float, e: float, circle: Circle, margin: float = 0.0) -> bool:
    cn, ce, r = circle
    rr = float(r) + float(margin)
    return ((n - cn) ** 2 + (e - ce) ** 2) <= rr ** 2


def in_rect(n: float, e: float, rect: Rect, margin: float = 0.0) -> bool:
    n_min, n_max, e_min, e_max = rect
    return (
        (n >= n_min - margin)
        and (n <= n_max + margin)
        and (e >= e_min - margin)
        and (e <= e_max + margin)
    )


def collide(n: float, e: float, cfg: Any) -> bool:
    margin = float(getattr(cfg, "collision_margin_m", 0.0))
    obstacles = get_obstacle_circles(cfg)
    return any(in_circle(n, e, c, margin=margin) for c in obstacles)


def in_nofly(n: float, e: float, cfg: Any) -> bool:
    circles = get_nofly_circles(cfg)
    rects = get_nofly_rects(cfg)

    for c in circles:
        if in_circle(n, e, c, margin=0.0):
            return True

    for r in rects:
        if in_rect(n, e, r, margin=0.0):
            return True

    return False


def _circle_signed_dist_and_closest_point(
    n: float, e: float, circle: Circle
) -> Tuple[float, float, float]:
    """
    返回:
      signed_dist: 圆外为正，圆内为负
      cp_n, cp_e: 圆边界上最近点
    """
    cn, ce, r = circle
    dx = float(n - cn)
    dy = float(e - ce)
    d = math.hypot(dx, dy)

    if d < 1e-9:
        # 正好在圆心，任意给一个边界点
        return -float(r), float(cn + r), float(ce)

    signed = d - float(r)
    scale = float(r) / d
    cp_n = float(cn + dx * scale)
    cp_e = float(ce + dy * scale)
    return float(signed), cp_n, cp_e


def _rect_signed_dist_and_closest_point(
    n: float, e: float, rect: Rect
) -> Tuple[float, float, float]:
    """
    对 axis-aligned rectangle 返回:
      signed_dist: 外部为正，内部为负
      cp_n, cp_e: 矩形边界上最近点
    """
    n_min, n_max, e_min, e_max = rect

    clamped_n = min(max(n, n_min), n_max)
    clamped_e = min(max(e, e_min), e_max)

    inside = (n_min <= n <= n_max) and (e_min <= e <= e_max)

    if not inside:
        d = math.hypot(n - clamped_n, e - clamped_e)
        return float(d), float(clamped_n), float(clamped_e)

    # inside: 到四条边最短距离，记为负
    d_left = n - n_min
    d_right = n_max - n
    d_bottom = e - e_min
    d_top = e_max - e

    dmin = min(d_left, d_right, d_bottom, d_top)

    if dmin == d_left:
        return -float(dmin), float(n_min), float(e)
    if dmin == d_right:
        return -float(dmin), float(n_max), float(e)
    if dmin == d_bottom:
        return -float(dmin), float(n), float(e_min)
    return -float(dmin), float(n), float(e_max)


def nearest_nofly_dist_dir(n: float, e: float, cfg: Any) -> Tuple[float, float, float]:
    """
    返回:
      signed_dist: 最近禁飞区边界 signed distance（外正内负）
      dir_n, dir_e: 从最近边界点指向当前位置的单位方向
                    对 inside 情况，可视为“逃离禁飞区”的方向
    若没有禁飞区，返回 (1e3, 0.0, 0.0)
    """
    best_abs = 1e18
    best_signed = 1e3
    best_cp = None

    for c in get_nofly_circles(cfg):
        signed, cp_n, cp_e = _circle_signed_dist_and_closest_point(n, e, c)
        if abs(signed) < best_abs:
            best_abs = abs(signed)
            best_signed = signed
            best_cp = (cp_n, cp_e)

    for r in get_nofly_rects(cfg):
        signed, cp_n, cp_e = _rect_signed_dist_and_closest_point(n, e, r)
        if abs(signed) < best_abs:
            best_abs = abs(signed)
            best_signed = signed
            best_cp = (cp_n, cp_e)

    if best_cp is None:
        return 1e3, 0.0, 0.0

    cp_n, cp_e = best_cp
    dx = float(n - cp_n)
    dy = float(e - cp_e)
    norm = math.hypot(dx, dy)

    if norm < 1e-9:
        return float(best_signed), 0.0, 0.0

    return float(best_signed), float(dx / norm), float(dy / norm)