"""Feature caching for the grounding probe.

For each frame we build the candidate set (target + in-frame distractors), run the
frozen backbone ONCE (all requested layers via output_hidden_states), pool the
per-layer image grid inside each candidate box, and store rows:

    feat  = concat([pooled_box_feature, instruction_text_feature])   (2C,)
    group = frame id (candidates within a frame compete)
    is_target = bool

Cached to <cache_dir>/<backbone>.npz so probe training over many layers is cheap.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .backbones import pool_box
from .projection import extract_candidates


def _iter_frames(ep, cfg):
    """Yield frame indices for one open episode, subsampled per config."""
    n = ep["rgb"].shape[0]
    stride = int(cfg["data"].get("frame_stride", 5))
    cap = int(cfg["data"].get("max_frames_per_episode", 40))
    idxs = list(range(0, n, stride))[:cap]
    return idxs


def _load_image(ep, i):
    from PIL import Image
    return Image.fromarray(np.asarray(ep["rgb"][i]))


def build_cache(backbone, cfg, episode_paths, layers) -> dict:
    """Run `backbone` over all frames; return {layer: dict(feat, group, is_target)}.

    Rows across episodes get globally-unique group ids so frames never mix.
    """
    import h5py

    W, H = cfg["image"]["width"], cfg["image"]["height"]
    min_c = int(cfg["data"].get("min_candidates", 2))
    # Row buckets are created lazily, keyed by the RESOLVED layer index that the
    # backbone actually returns (e.g. -1 -> last hidden_states index), so requests
    # like [16, -1] never mismatch the pre-init keys.
    rows: dict = {}
    gid = 0
    for path in episode_paths:
        with h5py.File(path, "r") as ep:
            instruction = _instruction(ep)
            for i in _iter_frames(ep, cfg):
                cands = extract_candidates(ep, i, cfg)
                if len(cands) < min_c or not any(c.is_target for c in cands):
                    continue
                image = _load_image(ep, i)
                feats = backbone.encode(image, instruction, layers)   # {layer:(grid,text)}
                for L, (grid, text_vec) in feats.items():
                    r = rows.setdefault(L, {"feat": [], "group": [], "is_target": []})
                    for c in cands:
                        box = pool_box(grid, c.frac_box(W, H))
                        r["feat"].append(np.concatenate([box, text_vec]))
                        r["group"].append(gid)
                        r["is_target"].append(c.is_target)
                gid += 1
    out = {}
    for L, r in rows.items():
        if not r["feat"]:
            continue
        out[L] = {
            "feat": np.asarray(r["feat"], dtype=np.float32),
            "group": np.asarray(r["group"], dtype=np.int64),
            "is_target": np.asarray(r["is_target"], dtype=bool),
        }
    return out


def save_cache(cache: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for L, r in cache.items():
        flat[f"{L}/feat"] = r["feat"]
        flat[f"{L}/group"] = r["group"]
        flat[f"{L}/is_target"] = r["is_target"]
    flat["__layers__"] = np.asarray(sorted(cache.keys()), dtype=np.int64)
    np.savez_compressed(path, **flat)


def load_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {}
    for L in z["__layers__"].tolist():
        out[int(L)] = {
            "feat": z[f"{L}/feat"], "group": z[f"{L}/group"],
            "is_target": z[f"{L}/is_target"],
        }
    return out


def resolve_episode_paths(cfg) -> list[str]:
    return sorted(glob.glob(cfg["data"]["episodes_glob"]))


# ---------------------------------------------------------------------------
def _instruction(ep) -> str:
    a = ep.attrs
    lang = a.get("language", "")
    if isinstance(lang, bytes):
        lang = lang.decode()
    if lang:
        return str(lang)
    desc = a.get("target_desc", "")
    if isinstance(desc, bytes):
        desc = desc.decode()
    return f"Track the {desc}." if desc else "Track the target vehicle."
