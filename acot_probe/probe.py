"""Lightweight probe head + group-wise train/eval.

The probe reads a FROZEN backbone's layer-L features and only learns to SELECT the
referred target among the candidates in each frame. Metric: Target-Selection
Accuracy (TSA) = P(argmax score == target). This directly answers the design-doc
question "does layer-16 already carry target-discriminative grounding?".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeResult:
    layer: int
    tsa: float               # target-selection accuracy (val)
    chance: float            # mean 1/group_size (val)
    n_groups_val: int
    n_groups_train: int


def _make_groups(cache_layer: dict):
    """Return list of (row_indices, target_pos) per group; drop malformed groups."""
    group = cache_layer["group"]
    is_t = cache_layer["is_target"]
    groups = []
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        tpos = np.where(is_t[idx])[0]
        if len(tpos) != 1 or len(idx) < 2:
            continue
        groups.append((idx, int(tpos[0])))
    return groups


def _split(groups, val_frac, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_val = max(1, int(len(groups) * val_frac))
    val = [groups[i] for i in order[:n_val]]
    train = [groups[i] for i in order[n_val:]]
    return train, val


def train_probe(cache_layer: dict, layer: int, cfg: dict) -> ProbeResult:
    import torch
    import torch.nn as nn

    feat = cache_layer["feat"]
    groups = _make_groups(cache_layer)
    if len(groups) < 4:
        return ProbeResult(layer, float("nan"), float("nan"), len(groups), 0)

    train_g, val_g = _split(groups, cfg["probe"].get("val_frac", 0.2), cfg.get("seed", 0))

    # standardize with TRAIN stats
    tr_rows = np.concatenate([idx for idx, _ in train_g])
    mu = feat[tr_rows].mean(0, keepdims=True)
    sd = feat[tr_rows].std(0, keepdims=True) + 1e-6
    X = ((feat - mu) / sd).astype(np.float32)

    dev = cfg["probe"].get("head_device", "cpu")
    Xt = torch.from_numpy(X).to(dev)
    head = nn.Sequential(
        nn.Linear(X.shape[1], cfg["probe"].get("hidden", 256)), nn.ReLU(),
        nn.Linear(cfg["probe"].get("hidden", 256), 1),
    ).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["probe"].get("lr", 1e-3),
                           weight_decay=cfg["probe"].get("weight_decay", 1e-4))

    B = cfg["probe"].get("batch_groups", 64)
    rng = np.random.default_rng(cfg.get("seed", 0))
    for _ in range(cfg["probe"].get("epochs", 30)):
        order = rng.permutation(len(train_g))
        for s in range(0, len(order), B):
            opt.zero_grad()
            batch = order[s:s + B]
            loss = torch.zeros((), device=dev)
            for gi in batch:
                idx, tpos = train_g[gi]
                logits = head(Xt[idx]).squeeze(-1)          # (k,)
                loss = loss + nn.functional.cross_entropy(
                    logits[None], torch.tensor([tpos], device=dev))
            (loss / max(1, len(batch))).backward()
            opt.step()

    # eval
    head.eval()
    with torch.no_grad():
        correct, chance = 0, 0.0
        for idx, tpos in val_g:
            logits = head(Xt[idx]).squeeze(-1)
            correct += int(torch.argmax(logits).item() == tpos)
            chance += 1.0 / len(idx)
    n = len(val_g)
    return ProbeResult(layer, correct / n, chance / n, n, len(train_g))


def run_layer_sweep(cache: dict, cfg: dict) -> list[ProbeResult]:
    return [train_probe(cache[L], L, cfg) for L in sorted(cache.keys())]
