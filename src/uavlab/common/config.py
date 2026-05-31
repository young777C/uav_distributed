# src/uavlab/common/config.py

'''负责管理整个实验的的配置文件，提供统一的接口读取并解析YAML配置文件
配置参数包括：超参数、环境配置、通信退化参数等'''
# src/uavlab/common/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import copy
import yaml


ConfigDict = Dict[str, Any]


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_yaml(path: str | Path) -> ConfigDict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def load_config(
    base_path: str | Path,
    override_path: Optional[str | Path] = None,
) -> ConfigDict:
    base_cfg = load_yaml(base_path)
    if override_path is None:
        return base_cfg
    override_cfg = load_yaml(override_path)
    return _deep_merge(base_cfg, override_cfg)


def load_resolved_config(path: str | Path) -> ConfigDict:
    """
    加载单文件配置；若根节点含 `extends`，则先加载基座再深度合并。

    支持两种形式：
    - `extends: "../base.yaml"`：单基座
    - `extends: ["../base.yaml", "../comm_profiles/degraded_blackholes.yaml", ...]`：多基座（按顺序依次合并）

    用于将实验配置拆分为“正交因子”（场景/通信/目标/基线等）后进行可组合管理。
    """
    path = Path(path).resolve()
    def _load_recursive(p: Path, *, _seen: set[Path]) -> ConfigDict:
        p = p.resolve()
        if p in _seen:
            raise ValueError(f"Cyclic extends detected at: {p}")
        _seen.add(p)
        data = load_yaml(p)
        extends = data.get("extends")
        if extends is None:
            return data

        def _resolve_one(base: str | Path) -> Path:
            bp = Path(base)
            if not bp.is_absolute():
                bp = (p.parent / bp).resolve()
            return bp

        if isinstance(extends, list):
            base_cfg: ConfigDict = {}
            for item in extends:
                if not isinstance(item, (str, Path)):
                    raise ValueError(f"extends list items must be paths (str): {p}")
                base_cfg = _deep_merge(base_cfg, _load_recursive(_resolve_one(item), _seen=_seen))
        elif isinstance(extends, (str, Path)):
            base_cfg = _load_recursive(_resolve_one(extends), _seen=_seen)
        else:
            raise ValueError(f"extends must be a path or list of paths: {p}")

        override = {k: v for k, v in data.items() if k != "extends"}
        return _deep_merge(base_cfg, override)

    path = Path(path).resolve()
    data = _load_recursive(path, _seen=set())
    return data


def save_yaml(data: ConfigDict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def get_by_path(cfg: ConfigDict, dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur